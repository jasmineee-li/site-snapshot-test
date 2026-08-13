"""Phase 4 adaptive strategy variation package.

The package exports host-side API helpers lazily so lightweight modules such as
``pvpo_capture`` can be imported by the isolated AgentLab sidecar without also
requiring host judge dependencies like Instructor/Anthropic.
"""

from __future__ import annotations

from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "ExtractedStep": ("warp_taskgen.phase_4.aer_trajectory_extract", "ExtractedStep"),
    "ExtractedTrajectory": ("warp_taskgen.phase_4.aer_trajectory_extract", "ExtractedTrajectory"),
    "as_aer_view": ("warp_taskgen.phase_4.aer_trajectory_extract", "as_aer_view"),
    "as_judge_view": ("warp_taskgen.phase_4.aer_trajectory_extract", "as_judge_view"),
    "extract_trajectory": ("warp_taskgen.phase_4.aer_trajectory_extract", "extract_trajectory"),
    "get_client": ("warp_taskgen.phase_4.anthropic_client", "get_client"),
    "get_api_semaphore": ("warp_taskgen.phase_4.concurrency", "get_api_semaphore"),
    "run_judge_api": ("warp_taskgen.phase_4.judge_api", "run_judge_api"),
    "run_placement_api": ("warp_taskgen.phase_4.placement_api", "run_placement_api"),
    "ALLOWED_STRATEGIES": ("warp_taskgen.phase_4.strategy_catalog", "ALLOWED_STRATEGIES"),
    "REFUSAL_TRIGGERS": ("warp_taskgen.phase_4.strategy_catalog", "REFUSAL_TRIGGERS"),
    "SURFACE_CHANGE_SENTINEL": ("warp_taskgen.phase_4.strategy_catalog", "SURFACE_CHANGE_SENTINEL"),
    "TRIGGER_TO_STRATEGIES": ("warp_taskgen.phase_4.strategy_catalog", "TRIGGER_TO_STRATEGIES"),
    "strategies_for_trigger": ("warp_taskgen.phase_4.strategy_catalog", "strategies_for_trigger"),
    "generate_variant_api": ("warp_taskgen.phase_4.variant_api", "generate_variant_api"),
    "phase_4_variant_budget_choices": (
        "warp_taskgen.phase_4.options",
        "phase_4_variant_budget_choices",
    ),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    from importlib import import_module

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORTS)
