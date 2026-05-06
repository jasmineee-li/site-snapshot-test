"""HF-generation tool-calling runner with optional residual-stream steering.

Mirrors `tool_calling_runner.ToolCallingRunner.run_task()`'s interface
(same args, same return type) so the existing AgentDojo / InjecAgent
runners can swap in this runner via a backend selector and not change
their orchestration code.

Why this exists: vLLM doesn't accept residual-stream forward-pre-hooks
(it runs out-of-process and uses custom CUDA kernels), so steered
benchmark cells can't go through the OpenAI-protocol path. This runner
loads the model with HF transformers, installs the steering hook from
`probes.steering`, and drives a tool-calling loop in pure HF — parsing
Hermes-format <tool_call> blocks out of the model's text output to
build OpenAI-style `tool_calls` lists for the existing tool executor.

Limitations vs ToolCallingRunner:
  - One forward at a time on the GPU (concurrency is gated by an
    asyncio.Lock — the runner stays async-callable but tasks serialize).
  - No streaming. Each turn does one full `model.generate()` call.
  - Sampling defaults to greedy (`temperature == 0`); set
    `temperature > 0` to enable nucleus sampling.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import field
from pathlib import Path
from typing import Any

import torch

from eval_awareness_experiments.injection_modifier import InjectionCondition
from eval_awareness_experiments.system_prompt_frame import (
    SystemPromptFrame,
    wrap_system_prompt,
)
from eval_awareness_experiments.tool_calling_runner import (
    AttackEvaluator,
    ToolCallTaskResult,
    ToolExecutor,
    TurnRecord,
)

logger = logging.getLogger(__name__)


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _parse_hermes_output(text: str) -> tuple[list[dict], str, str]:
    """Parse a Qwen2.5/Hermes-style assistant output.

    Returns:
        (tool_calls, content, thought)
        - tool_calls: OpenAI-format list (may be empty).
        - content: assistant text with <tool_call> blocks stripped, also
          stripped of <think>...</think> blocks (those go in `thought`).
        - thought: concatenated text from <think> blocks (may be empty).
    """
    thought_parts = _THINK_RE.findall(text)
    thought = "\n".join(t.strip() for t in thought_parts).strip()
    text_no_think = _THINK_RE.sub("", text)

    tool_calls: list[dict] = []
    for i, raw in enumerate(_TOOL_CALL_RE.findall(text_no_think)):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse <tool_call> #{i}: {e}\nraw={raw!r}")
            continue
        name = obj.get("name") or obj.get("tool_name")
        args = obj.get("arguments") or obj.get("parameters") or {}
        if not name:
            logger.warning(f"<tool_call> missing name: {obj}")
            continue
        tool_calls.append({
            "id": f"call_{i}",
            "type": "function",
            "function": {
                "name": name,
                # OpenAI-format requires arguments as a JSON string.
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        })

    content = _TOOL_CALL_RE.sub("", text_no_think).strip()
    # Strip the canonical end-of-turn marker if it leaks in.
    content = content.removesuffix("<|im_end|>").strip()
    return tool_calls, content, thought


class HFToolCallingRunner:
    """Drop-in HF-generation analog of ToolCallingRunner.

    Same `run_task(...)` signature; loads the model on GPU, installs an
    optional `probes.steering.steering_hook(probe, layer, alpha)` around
    each `model.generate()` call.
    """

    def __init__(
        self,
        model_name: str,
        condition: InjectionCondition,
        extra_instructions: str | None,
        extra_instructions_preset: str,
        system_prompt_frame: SystemPromptFrame = "none",
        max_turns: int = 15,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        retries: int = 0,  # unused — HF generate doesn't have transient API errors
        device: str | None = None,
        dtype: torch.dtype | None = None,
        probe_dir: str | Path | None = None,
        steering_layer: int | None = None,
        steering_layers: list[int] | None = None,
        steering_alpha: float = 0.0,
    ):
        # Strip the local/ prefix and :thinking suffix; HF cares about the
        # actual HF repo, not the runtime convenience prefix.
        if model_name.startswith("local/"):
            from models.common import resolve_local_model
            self.hf_repo = resolve_local_model(model_name).hf_repo
            self.served_model_name = resolve_local_model(model_name).resolve_served_name()
        else:
            self.hf_repo = model_name.removesuffix(":thinking")
            self.served_model_name = self.hf_repo
        self.condition = condition
        self.extra_instructions = extra_instructions
        self.extra_instructions_preset = extra_instructions_preset
        self.system_prompt_frame = system_prompt_frame
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.steering_alpha = steering_alpha
        if steering_layer is not None and steering_layers is not None:
            raise ValueError(
                "Pass exactly one of `steering_layer` or `steering_layers`."
            )
        self.steering_layer = steering_layer
        self.steering_layers = list(steering_layers) if steering_layers else None
        self.probe_dir = str(probe_dir) if probe_dir else None

        # Lock so concurrent run_task() awaits serialize on the single GPU.
        self._gpu_lock = asyncio.Lock()

        # Load the model + tokenizer once. `probes.model_loader.load_model`
        # returns a `(HookedTransformerShim, tokenizer)` pair; we use the
        # shim's `.model` for generate() and the tokenizer for chat-template
        # rendering + decoding.
        from probes.model_loader import load_model
        if dtype is None:
            dtype = torch.bfloat16
        logger.info(
            f"Loading HF model {self.hf_repo} for HFToolCallingRunner "
            f"(steering_alpha={steering_alpha}, layer={steering_layer})"
        )
        self.shim, self.tokenizer = load_model(
            self.hf_repo, device=device, dtype=dtype,
        )
        self.hf_model = self.shim.model

        # Load the probe if steering is requested.
        self.probe = None
        if self.steering_alpha != 0.0:
            if not self.probe_dir:
                raise ValueError(
                    "steering_alpha != 0 requires probe_dir to be set "
                    "(point at a probes/trained/<model>/ directory)"
                )
            from probes.contrastive_probe import ContrastiveProbe
            self.probe = ContrastiveProbe.load(self.probe_dir)
            if self.steering_layer is None and self.steering_layers is None:
                self.steering_layer = self.probe.best_layer
            if self.steering_layers is not None:
                logger.info(
                    f"Steering enabled: alpha_per_layer={self.steering_alpha} "
                    f"layers={self.steering_layers} probe={self.probe_dir}"
                )
            else:
                logger.info(
                    f"Steering enabled: alpha={self.steering_alpha} "
                    f"layer={self.steering_layer} probe={self.probe_dir}"
                )

    async def run_task(
        self,
        *,
        task_id: str,
        task_instruction: str,
        system_prompt: str,
        tools: list[dict],
        tool_executor: ToolExecutor,
        benchmark: str,
        safety_category: str,
        attack_evaluator: AttackEvaluator | None = None,
        metadata: dict | None = None,
    ) -> ToolCallTaskResult:
        full_system = system_prompt
        if self.extra_instructions:
            full_system = f"{system_prompt}\n\n{self.extra_instructions}".strip()
        full_system = wrap_system_prompt(full_system, self.system_prompt_frame)

        messages: list[dict] = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": task_instruction},
        ]

        turns: list[TurnRecord] = []
        final_answer = ""
        hit_max_turns = False

        for turn_idx in range(self.max_turns):
            output_text = await self._generate_one_turn(messages, tools, task_id, turn_idx)
            tool_calls, content, thought = _parse_hermes_output(output_text)

            if tool_calls:
                # Append the assistant turn with tool_calls to history.
                messages.append({
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                })
                for tc_idx, tc in enumerate(tool_calls):
                    name = tc["function"]["name"]
                    try:
                        args = json.loads(tc["function"]["arguments"] or "{}")
                    except json.JSONDecodeError as e:
                        observation = f"[tool arg parse error: {e}]"
                        args = {}
                    else:
                        observation = await tool_executor(name, args, tc["id"])
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "name": name,
                        "content": observation,
                    })
                    turns.append(TurnRecord(
                        thought=thought if tc_idx == 0 else "",
                        preamble=content if tc_idx == 0 else "",
                        action=f"{name}({json.dumps(args, ensure_ascii=False)})",
                        observation=observation,
                    ))
            else:
                messages.append({"role": "assistant", "content": content})
                turns.append(TurnRecord(
                    thought=thought,
                    action=content,
                    observation="",
                ))
                final_answer = content
                break
        else:
            hit_max_turns = True
            logger.warning(
                f"[{benchmark}:{task_id}] hit max_turns={self.max_turns} without final text"
            )

        attack_success: bool | None = None
        passed: bool | None = None
        if attack_evaluator is not None:
            attack_success, passed = await attack_evaluator(messages)

        merged_meta = {
            "hit_max_turns": hit_max_turns,
            "n_turns": len(turns),
            "backend": "hf",
            "steering_alpha": self.steering_alpha,
            "steering_layer": self.steering_layer,
            "steering_layers": self.steering_layers,
            "probe_dir": self.probe_dir,
            **(metadata or {}),
        }

        agent_label = self.served_model_name
        if self.steering_alpha != 0.0:
            agent_label = (
                f"{agent_label}+steer(layer={self.steering_layer},"
                f"alpha={self.steering_alpha})"
            )

        return ToolCallTaskResult(
            task_id=task_id,
            task_instruction=task_instruction,
            benchmark=benchmark,
            agent=agent_label,
            condition=self.condition,
            extra_instructions_preset=self.extra_instructions_preset,
            system_prompt_frame=self.system_prompt_frame,
            system_prompt=full_system,
            steps=turns,
            final_answer=final_answer,
            passed=passed,
            attack_success=attack_success,
            safety_category=safety_category,
            metadata=merged_meta,
        )

    async def run_task_to_file(
        self,
        out_dir: Path,
        **run_task_kwargs: Any,
    ) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        result = await self.run_task(**run_task_kwargs)
        out_path = out_dir / f"{result.task_id}.json"
        out_path.write_text(json.dumps(result.to_json(), indent=2))
        return out_path

    async def _generate_one_turn(
        self,
        messages: list[dict],
        tools: list[dict],
        task_id: str,
        turn_idx: int,
    ) -> str:
        """Run one HF generate() call under the GPU lock + steering hook."""
        async with self._gpu_lock:
            # Rendering + decode are CPU-bound; generate() is GPU-bound and
            # blocking. Push the whole thing to a thread so the event loop
            # stays responsive (e.g. for the tool executor's I/O).
            return await asyncio.to_thread(
                self._generate_sync, messages, tools, task_id, turn_idx,
            )

    def _generate_sync(
        self,
        messages: list[dict],
        tools: list[dict],
        task_id: str,
        turn_idx: int,
    ) -> str:
        # Render the chat template with tools. Qwen2.5's template handles
        # OpenAI-style tool dicts natively; if the local tokenizer doesn't,
        # this will raise and the caller can patch tools->plain-text.
        try:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tools=tools or None,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            )
        except Exception as e:
            logger.error(
                f"apply_chat_template failed for {self.hf_repo} "
                f"({task_id}:t{turn_idx}): {e}"
            )
            raise

        if isinstance(inputs, dict):
            input_ids = inputs["input_ids"]
        else:
            input_ids = inputs
        input_ids = input_ids.to(self.hf_model.device)
        prompt_len = input_ids.shape[1]

        gen_kwargs: dict = {
            "max_new_tokens": self.max_tokens,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        if self.temperature and self.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.temperature
        else:
            gen_kwargs["do_sample"] = False

        from contextlib import nullcontext
        if self.steering_alpha != 0.0 and self.probe is not None:
            from probes.steering import steering_hook
            if self.steering_layers is not None:
                ctx = steering_hook(
                    self.shim, self.probe,
                    layers=self.steering_layers,
                    alpha_per_layer=self.steering_alpha,
                )
            else:
                ctx = steering_hook(
                    self.shim, self.probe,
                    layer=self.steering_layer, alpha=self.steering_alpha,
                )
        else:
            ctx = nullcontext()

        with ctx:
            with torch.no_grad():
                output_ids = self.hf_model.generate(input_ids, **gen_kwargs)

        gen_ids = output_ids[0, prompt_len:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=False)
