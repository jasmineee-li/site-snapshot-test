"""Build either a `ToolCallingRunner` (OpenAI-protocol: OpenRouter or
local vLLM) or an `HFToolCallingRunner` (HF transformers + steering).

Picks the runner based on whether steering is requested:
  - steering_alpha != 0  → HF runner (only path that can install hooks)
  - steering_alpha == 0  → OpenAI-protocol runner (vLLM for local/, OpenRouter otherwise)
                           — fast, batched, default for the no-steering cells.

You can force the HF path via `backend="hf"` if you want everything on
HF (e.g. for a strictly apples-to-apples baseline-vs-steered comparison
where you don't want the OAI-protocol path's batching to confound).
"""

from __future__ import annotations

from typing import Literal

from eval_awareness_experiments.injection_modifier import InjectionCondition
from eval_awareness_experiments.system_prompt_frame import SystemPromptFrame
from eval_awareness_experiments.tool_calling_runner import ToolCallingRunner

Backend = Literal["auto", "openai", "hf"]


def build_tool_calling_runner(
    *,
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
    backend: Backend = "auto",
    probe_dir: str | None = None,
    steering_layer: int | None = None,
    steering_layers: list[int] | None = None,
    steering_alpha: float = 0.0,
):
    """Return a runner with the unified `run_task(...)` interface.

    Auto rule:
      - steering_alpha != 0           → "hf"
      - else                          → "openai" (works for OpenRouter
        ids and `local/<short-id>` ids alike — the latter routes to vLLM
        via models/common/registry.py inside ToolCallingRunner.__init__)
    """
    if backend == "auto":
        backend = "hf" if steering_alpha != 0.0 else "openai"

    if backend == "openai":
        if steering_alpha != 0.0:
            raise ValueError(
                "steering_alpha != 0 is not supported on the OpenAI-protocol "
                "backend (vLLM/OpenRouter don't accept residual-stream hooks). "
                "Use backend='hf' or leave backend='auto'."
            )
        return ToolCallingRunner(
            model_name=model_name,
            condition=condition,
            extra_instructions=extra_instructions,
            extra_instructions_preset=extra_instructions_preset,
            system_prompt_frame=system_prompt_frame,
            max_turns=max_turns,
            max_tokens=max_tokens,
            temperature=temperature,
            concurrency=concurrency,
            retries=retries,
        )

    if backend == "hf":
        # Lazy import: the `probes` + HF stack is heavy and not always present.
        from eval_awareness_experiments.hf_tool_calling_runner import (
            HFToolCallingRunner,
        )
        return HFToolCallingRunner(
            model_name=model_name,
            condition=condition,
            extra_instructions=extra_instructions,
            extra_instructions_preset=extra_instructions_preset,
            system_prompt_frame=system_prompt_frame,
            max_turns=max_turns,
            max_tokens=max_tokens,
            temperature=temperature,
            probe_dir=probe_dir,
            steering_layer=steering_layer,
            steering_layers=steering_layers,
            steering_alpha=steering_alpha,
        )

    raise ValueError(f"Unknown backend {backend!r}")
