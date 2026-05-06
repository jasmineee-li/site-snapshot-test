"""Phase 4 adaptive strategy variation package.

The package exports host-side API helpers lazily so lightweight modules such as
``pvpo_capture`` can be imported by the isolated AgentLab sidecar without also
requiring host judge dependencies like Instructor/Anthropic.
"""

from __future__ import annotations

from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "ExtractedStep": ("worldsim.phase_4.aer_trajectory_extract", "ExtractedStep"),
    "ExtractedTrajectory": ("worldsim.phase_4.aer_trajectory_extract", "ExtractedTrajectory"),
    "as_aer_view": ("worldsim.phase_4.aer_trajectory_extract", "as_aer_view"),
    "as_judge_view": ("worldsim.phase_4.aer_trajectory_extract", "as_judge_view"),
    "extract_trajectory": ("worldsim.phase_4.aer_trajectory_extract", "extract_trajectory"),
    "get_client": ("worldsim.phase_4.anthropic_client", "get_client"),
    "get_api_semaphore": ("worldsim.phase_4.concurrency", "get_api_semaphore"),
    "run_judge_api": ("worldsim.phase_4.judge_api", "run_judge_api"),
    "run_placement_api": ("worldsim.phase_4.placement_api", "run_placement_api"),
    "ALLOWED_STRATEGIES": ("worldsim.phase_4.strategy_catalog", "ALLOWED_STRATEGIES"),
    "REFUSAL_TRIGGERS": ("worldsim.phase_4.strategy_catalog", "REFUSAL_TRIGGERS"),
    "SURFACE_CHANGE_SENTINEL": ("worldsim.phase_4.strategy_catalog", "SURFACE_CHANGE_SENTINEL"),
    "TRIGGER_TO_STRATEGIES": ("worldsim.phase_4.strategy_catalog", "TRIGGER_TO_STRATEGIES"),
    "strategies_for_trigger": ("worldsim.phase_4.strategy_catalog", "strategies_for_trigger"),
    "generate_variant_api": ("worldsim.phase_4.variant_api", "generate_variant_api"),
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
