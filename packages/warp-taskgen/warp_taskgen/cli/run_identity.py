"""CLI adapter for resolving immutable Run identity before phase dispatch."""

from __future__ import annotations

import argparse
from typing import Any

from warp_taskgen.phase_1.novel_task_site_plan import DEFAULT_NOVEL_TASKS_PER_SITE
from warp_taskgen.phase_2.text_fill.constants import DEFAULT_TEXT_FILL_MODEL, DEFAULT_TEXTS_PER_PLAN
from warp_taskgen.phase_4.options import (
    DEFAULT_PHASE_4_EVAL_AWARENESS_MAX_ITERATIONS,
    DEFAULT_PHASE_4_VARIANT_BUDGET_PRESET,
    DEFAULT_PHASE_4_VARIANT_SYSTEM,
)
from warp_taskgen.run_definition import define_run
from warp_taskgen.run_definition_contracts import RunTransition
from warp_taskgen.run_transition import resolve_run_request
from warp_taskgen.state import get_state_dir, get_state_file, load_state_for_current_root

_ARG_ALIASES = {
    "benchmark": "benchmark_path",
    "config": "manifest_path",
    "instances": "instances_path",
    "host_inventory_instances": "host_inventory_instances_path",
    "runner": "agent_runner",
    "task_card_plan": "task_card_plan_path",
}
_STATIC_DEFAULTS: dict[str, object] = {
    "novel_tasks_per_site": DEFAULT_NOVEL_TASKS_PER_SITE,
    "phase_2b_texts_per_plan": DEFAULT_TEXTS_PER_PLAN,
    "phase_2_text_model": DEFAULT_TEXT_FILL_MODEL,
    "task_origin": "all",
    "phase_4_variant_budget": DEFAULT_PHASE_4_VARIANT_BUDGET_PRESET,
    "phase_4_variant_system": DEFAULT_PHASE_4_VARIANT_SYSTEM,
    "phase_4_eval_awareness_max_iterations": DEFAULT_PHASE_4_EVAL_AWARENESS_MAX_ITERATIONS,
}
_UNSET = object()


def _definition_inputs(args: argparse.Namespace, *, apply_defaults: bool) -> dict[str, object]:
    inputs = {key: value for key, value in vars(args).items() if value is not None}
    for source, target in _ARG_ALIASES.items():
        if source in inputs:
            inputs[target] = inputs[source]
    if apply_defaults:
        for field, value in _STATIC_DEFAULTS.items():
            inputs.setdefault(field, value)
        inputs.setdefault("manifest_path", get_state_dir() / "phase_0a" / "BENCHMARK_MANIFEST.json")
    return inputs


def _existing_state_for_root() -> dict[str, Any] | None:
    state = load_state_for_current_root()
    if state is None and get_state_file().exists():
        raise ValueError("existing state root is unreadable; refusing to create a new Run identity")
    return state


def resume_state_inputs(state: dict[str, Any]) -> dict[str, Any]:
    """Overlay persisted definition inputs beneath checkpoint-local metadata."""

    definition = define_run(state)
    if definition.legacy:
        return dict(state)
    return {**definition.input_projection(), **state}


def resolve_cli_run_transition(
    args: argparse.Namespace,
    *,
    existing_state: dict[str, Any] | None | object = _UNSET,
) -> RunTransition:
    """Resolve explicit resume overrides or a phase's normalized static defaults."""

    supplied = existing_state is not _UNSET
    state = existing_state if supplied else _existing_state_for_root()
    if state is not None and not isinstance(state, dict):
        raise ValueError("existing_state must contain a pipeline-state object or null")
    return resolve_run_request(
        _definition_inputs(args, apply_defaults=not supplied),
        existing_state=state,
    )


__all__ = ["resolve_cli_run_transition", "resume_state_inputs"]
