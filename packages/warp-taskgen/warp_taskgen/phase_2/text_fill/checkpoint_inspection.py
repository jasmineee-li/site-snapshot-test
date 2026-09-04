"""Read-only Phase 2b text-fill checkpoint status projection."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from warp_taskgen.phase_2.text_fill.checkpoint_reader import inspect_text_fill_checkpoint
from warp_taskgen.phase_2.text_fill.checkpoints import (
    text_fill_checkpoint_path,
    text_fill_task_id,
)
from warp_taskgen.phases.phase_1_tasks import _parse_sites_filter
from warp_taskgen.run_definition import define_run

_MAX_STATUS_UNITS = 100


def inspect_text_fill_checkpoints(run_root: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct Phase 2b work and inspect its feature-owned envelopes.

    This projection deliberately reads only persisted Phase 2 plans, the
    effective text-fill settings, and the run definition.  It never seeds a
    site, calls a provider, writes a checkpoint, or decides lifecycle state.
    """

    plans_path = run_root / "phase_2" / "adversarial_plans.json"
    settings = _text_fill_settings(state)
    if settings is None:
        return _not_inspected_text_fill_checkpoints(
            reason_code="text_fill_settings_missing",
            path=run_root / "pipeline_state.json",
        )
    if settings is False:
        return _not_inspected_text_fill_checkpoints(
            reason_code="text_fill_settings_invalid",
            path=run_root / "pipeline_state.json",
        )

    try:
        raw_plans = json.loads(plans_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _not_inspected_text_fill_checkpoints(
            reason_code="text_fill_plans_missing",
            path=plans_path,
        )
    except (OSError, UnicodeError):
        return _not_inspected_text_fill_checkpoints(
            reason_code="text_fill_plans_unreadable",
            path=plans_path,
        )
    except json.JSONDecodeError:
        return _not_inspected_text_fill_checkpoints(
            reason_code="text_fill_plans_invalid",
            path=plans_path,
        )
    if not isinstance(raw_plans, list) or any(not isinstance(plan, dict) for plan in raw_plans):
        return _not_inspected_text_fill_checkpoints(
            reason_code="text_fill_plans_invalid",
            path=plans_path,
        )

    try:
        definition = define_run(state)
    except (TypeError, ValueError):
        return _not_inspected_text_fill_checkpoints(reason_code="run_definition_unavailable")
    if definition.legacy or not definition.run_id:
        return _not_inspected_text_fill_checkpoints(reason_code="run_definition_unavailable")

    text_model, texts_per_plan, concurrency, sites_filter = settings
    candidate_plans = [
        plan
        for plan in raw_plans
        if sites_filter is None or str(plan.get("site", "")) in sites_filter
    ]
    normalized_candidate_ids: list[str] = []
    seen: set[str] = set()
    try:
        for plan in candidate_plans:
            task_id = text_fill_task_id(plan)
            if task_id in seen:
                raise ValueError(f"duplicate normalized text-fill task id: {task_id!r}")
            seen.add(task_id)
            normalized_candidate_ids.append(task_id)
    except (TypeError, ValueError):
        return _not_inspected_text_fill_checkpoints(
            reason_code="text_fill_plans_invalid",
            path=plans_path,
        )

    selected_plans = [plan for plan in candidate_plans if "seed_template" in plan]
    normalized_ids = [
        task_id
        for plan, task_id in zip(candidate_plans, normalized_candidate_ids, strict=True)
        if "seed_template" in plan
    ]

    checkpoint_dir = run_root / "phase_2" / "text_fill" / "checkpoints"
    checkpoint_settings = {"text_fill_concurrency": concurrency}
    units: list[dict[str, Any]] = []
    counts = {
        "compatible_count": 0,
        "pending_count": 0,
        "stale_count": 0,
        "malformed_count": 0,
    }
    for plan, task_id in zip(selected_plans, normalized_ids, strict=True):
        path = text_fill_checkpoint_path(checkpoint_dir, task_id)
        inspection = inspect_text_fill_checkpoint(
            path,
            plan,
            definition=definition,
            text_model=text_model,
            texts_per_plan=texts_per_plan,
            settings=checkpoint_settings,
        )
        counts[f"{inspection.status}_count"] += 1
        if len(units) < _MAX_STATUS_UNITS:
            units.append(
                {
                    "task_id": task_id,
                    "site": str(plan.get("site", "")),
                    "status": inspection.status,
                    "reason_code": inspection.reason_code,
                    "path": str(path),
                }
            )

    return {
        "status": "inspected",
        "authority": "advisory:phase_2.text_fill.checkpoint_inspection",
        "expected_count": len(selected_plans),
        **counts,
        "not_inspected_count": 0,
        "units": units,
        "units_truncated": max(0, len(selected_plans) - len(units)),
        "reason_code": None,
        "path": None,
        "effects": {"writes": False, "model_calls": False, "network": False},
    }


def _text_fill_settings(
    state: Mapping[str, Any],
) -> tuple[str, int, int, set[str] | None] | bool | None:
    required = (
        "phase_2_text_model",
        "phase_2b_texts_per_plan",
        "phase_2_text_fill_concurrency",
        "sites",
    )
    if any(name not in state for name in required):
        return None
    model = state.get("phase_2_text_model")
    texts_per_plan = state.get("phase_2b_texts_per_plan")
    concurrency = state.get("phase_2_text_fill_concurrency")
    if not isinstance(model, str) or not model.strip():
        return False
    if type(texts_per_plan) is not int or texts_per_plan <= 0:
        return False
    if type(concurrency) is not int or concurrency <= 0:
        return False
    raw_sites = state.get("sites")
    if raw_sites is not None and not isinstance(raw_sites, (str, list, tuple, set, frozenset)):
        return False
    if isinstance(raw_sites, (list, tuple, set, frozenset)) and any(
        not isinstance(site, str) for site in raw_sites
    ):
        return False
    try:
        sites_filter = _parse_sites_filter(raw_sites)
    except (TypeError, ValueError):
        return False
    return model, texts_per_plan, concurrency, sites_filter


def _not_inspected_text_fill_checkpoints(
    *,
    reason_code: str,
    path: Path | None = None,
) -> dict[str, Any]:
    return {
        "status": "not_inspected",
        "authority": "advisory:phase_2.text_fill.checkpoint_inspection",
        "expected_count": None,
        "compatible_count": None,
        "pending_count": None,
        "stale_count": None,
        "malformed_count": None,
        "not_inspected_count": None,
        "units": [],
        "units_truncated": 0,
        "reason_code": reason_code,
        "path": str(path) if path is not None else None,
        "effects": {"writes": False, "model_calls": False, "network": False},
    }


__all__ = ["inspect_text_fill_checkpoints"]
