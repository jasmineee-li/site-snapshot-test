"""Shared tool-calling agent runner for AgentDojo and InjecAgent.

Runs a standard OpenAI-style function-calling loop against OpenRouter, records
per-turn reasoning + tool calls + observations, and emits a JSON per task that
`trajectory_converter.parse_toolcalling_trajectory` can read.

Benchmark-specific logic (how to load tasks, how to inject attacks into tool
outputs, how to evaluate attack success) lives in the per-benchmark runners —
this module is deliberately benchmark-agnostic.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from openai import AsyncOpenAI

from eval_awareness_experiments.injection_modifier import InjectionCondition
from eval_awareness_experiments.retry import call_with_retry
from eval_awareness_experiments.system_prompt_frame import (
    SystemPromptFrame,
    wrap_system_prompt,
)

# Local-model registry: when model_name starts with `local/<short-id>`, route
# through the vLLM server registered in models/common/registry.py instead of
# OpenRouter. Imported lazily inside __init__ so this module still imports
# cleanly in environments without the models/ package on path.

logger = logging.getLogger(__name__)

# .env loader — matches llm.py behavior so OPENROUTER_API_KEY picks up from .env.
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            val = val.strip('"').strip("'")
            os.environ.setdefault(key.strip(), val)


# A tool_executor takes (tool_name, arguments_dict, call_id) and returns the
# observation text the agent will see. The executor is responsible for applying
# `wrap_injection` to the observation if this tool-call is the attack vector.
ToolExecutor = Callable[[str, dict, str], Awaitable[str]]

# An attack_evaluator takes the full post-run message list and returns
# (attack_success, passed). Either may be None if the benchmark doesn't
# provide a signal for it.
AttackEvaluator = Callable[
    [list[dict]], Awaitable[tuple[bool | None, bool | None]]
]


def _sanitize_openai_schema(schema: Any) -> Any:
    """Return a provider-compatible copy of an OpenAI tool JSON schema.

    Some benchmark schemas, notably InjecAgent, specify array parameters
    without `items`. OpenAI/Azure-style providers reject those schemas before
    the model can run, so fill in a conservative default while preserving the
    rest of the schema.
    """
    if isinstance(schema, list):
        return [_sanitize_openai_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema

    sanitized = {
        key: _sanitize_openai_schema(value)
        for key, value in schema.items()
    }
    schema_type = sanitized.get("type")
    is_array = (
        schema_type == "array"
        or (isinstance(schema_type, list) and "array" in schema_type)
    )
    if is_array and "items" not in sanitized:
        sanitized["items"] = {"type": "string"}
    return sanitized


def _sanitize_openai_tools(tools: list[dict]) -> list[dict]:
    return [_sanitize_openai_schema(tool) for tool in tools]


@dataclass
class TurnRecord:
    """One model turn: reasoning + either tool_calls or final text.

    `preamble` captures `message.content` emitted before tool_calls — some
    models (e.g. GLM-5 in tool-use mode) route forced scratchpad output here
    instead of into the reasoning channel, so it must be preserved for the
    judge to see.
    """
    thought: str
    action: str
    observation: str
    preamble: str = ""


@dataclass
class ToolCallTaskResult:
    task_id: str
    task_instruction: str
    benchmark: str
    agent: str
    condition: str
    extra_instructions_preset: str
    system_prompt_frame: str
    steps: list[TurnRecord]
    final_answer: str
    passed: bool | None
    attack_success: bool | None
    safety_category: str
    system_prompt: str = ""
    metadata: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_instruction": self.task_instruction,
            "benchmark": self.benchmark,
            "agent": self.agent,
            "condition": self.condition,
            "extra_instructions_preset": self.extra_instructions_preset,
            "system_prompt_frame": self.system_prompt_frame,
            "system_prompt": self.system_prompt,
            "steps": [asdict(s) for s in self.steps],
            "final_answer": self.final_answer,
            "passed": self.passed,
            "attack_success": self.attack_success,
            "safety_category": self.safety_category,
            "metadata": self.metadata,
        }


class ToolCallingRunner:
    """OpenRouter tool-calling agent loop.

    The `:thinking` suffix on `model_name` enables OpenRouter's reasoning mode;
    reasoning tokens are read off `message.reasoning` and recorded per-turn.
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
        temperature: float = 0.1,
        concurrency: int = 16,
        retries: int = 3,
    ):
        self.thinking = model_name.endswith(":thinking")
        self.model_name = model_name.removesuffix(":thinking") if self.thinking else model_name
        self.condition = condition
        self.extra_instructions = extra_instructions
        self.extra_instructions_preset = extra_instructions_preset
        self.system_prompt_frame = system_prompt_frame
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.retries = retries
        self._semaphore = asyncio.Semaphore(concurrency)
        # Route `local/<short-id>` to the vLLM server registered in
        # models/common/registry.py; everything else to OpenRouter.
        if self.model_name.startswith("local/"):
            from models.common import resolve_local_model
            from models.common.vllm_client import make_async_openai_client
            spec = resolve_local_model(self.model_name)
            self.model_name = spec.resolve_served_name()
            # Local servers don't speak OpenRouter's reasoning-effort knob;
            # `:thinking` was stripped above but disable extra_kwargs too.
            self.thinking = False
            self._client = make_async_openai_client(spec)
        else:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise RuntimeError("OPENROUTER_API_KEY not set (check .env or environment)")
            self._client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
                # SDK retries off — call_with_retry owns the policy and needs to
                # release the semaphore between attempts.
                max_retries=0,
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
        """Run one task through the agent loop.

        Args:
            task_id: Unique identifier for the task (filename stem for outputs).
            task_instruction: The user's instruction to the agent.
            system_prompt: Benchmark-specific system prompt. `extra_instructions`
                (e.g. scratchpad/motive) is appended verbatim to the end, and
                the result is then optionally wrapped by `system_prompt_frame`
                (e.g. `<llm_safety_test>...</llm_safety_test>`).
            tools: OpenAI-format tool schemas (list of {"type":"function","function":{...}}).
            tool_executor: async callable (name, args, call_id) -> observation text.
                The executor must apply `wrap_injection` to the observation when
                the call triggers the attack vector.
            benchmark: "agentdojo" | "injecagent" (recorded into the trajectory).
            safety_category: per-benchmark category string (recorded into trajectory).
            attack_evaluator: optional async callable that inspects the full
                message history and returns (attack_success, passed).
            metadata: extra per-task metadata to record.

        Returns:
            ToolCallTaskResult — serializable via .to_json() and parseable by
            `parse_toolcalling_trajectory`.
        """
        full_system = system_prompt
        if self.extra_instructions:
            full_system = f"{system_prompt}\n\n{self.extra_instructions}".strip()
        # Wrap *after* extra_instructions are appended so the frame envelopes
        # the entire system prompt, not just the benchmark portion.
        full_system = wrap_system_prompt(full_system, self.system_prompt_frame)

        messages: list[dict] = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": task_instruction},
        ]

        turns: list[TurnRecord] = []
        final_answer = ""
        hit_max_turns = False
        safe_tools = _sanitize_openai_tools(tools) if tools else []

        for turn_idx in range(self.max_turns):
            async def _do_turn():
                async with self._semaphore:
                    return await self._client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        tools=safe_tools if safe_tools else None,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        **self._extra_kwargs(),
                    )

            resp = await call_with_retry(
                _do_turn,
                retries=self.retries,
                label=f"toolcall:{benchmark}:{task_id}:t{turn_idx}",
            )
            msg = resp.choices[0].message
            thought = (getattr(msg, "reasoning", None) or "") or ""
            content = msg.content or ""
            tool_calls = getattr(msg, "tool_calls", None) or []

            if tool_calls:
                # Record assistant turn with tool calls.
                messages.append({
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                })
                # Execute each tool call sequentially. One TurnRecord per call.
                for tc_idx, tc in enumerate(tool_calls):
                    name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError as e:
                        observation = f"[tool arg parse error: {e}]"
                        args = {}
                    else:
                        observation = await tool_executor(name, args, tc.id)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": observation,
                    })
                    turns.append(TurnRecord(
                        # Attribute reasoning + preamble only to the first tool-call of the turn
                        thought=thought if tc_idx == 0 else "",
                        preamble=content if tc_idx == 0 else "",
                        action=f"{name}({json.dumps(args, ensure_ascii=False)})",
                        observation=observation,
                    ))
            else:
                # Final text turn. Record and stop.
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
            **(metadata or {}),
        }

        return ToolCallTaskResult(
            task_id=task_id,
            task_instruction=task_instruction,
            benchmark=benchmark,
            agent=self.model_name + (":thinking" if self.thinking else ""),
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
        """Run one task and persist the JSON next to its siblings. Returns the path."""
        out_dir.mkdir(parents=True, exist_ok=True)
        result = await self.run_task(**run_task_kwargs)
        out_path = out_dir / f"{result.task_id}.json"
        out_path.write_text(json.dumps(result.to_json(), indent=2))
        return out_path

    def _extra_kwargs(self) -> dict:
        if not self.thinking:
            return {}
        return {"extra_body": {"reasoning": {"effort": "high"}}}
