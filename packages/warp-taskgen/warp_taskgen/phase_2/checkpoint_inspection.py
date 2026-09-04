"""Read-only Phase 2a planning checkpoint status projection."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from warp_taskgen.phase_2 import generation as _generation
from warp_taskgen.phase_2.planning_specs import build_planning_shard_specs
from warp_taskgen.phase_2.runner import (
    _with_phase1_route_surface_overlays,
)
from warp_taskgen.phase_2.shards import inspect_planning_shard_checkpoints
from warp_taskgen.run_definition import define_run


def inspect_planning_checkpoints(run_root: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct expected shards and inspect them through shard ownership."""

    tasks_path = run_root / "phase_1" / "benign_tasks.json"
    try:
        raw_tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
        if not isinstance(raw_tasks, list) or any(not isinstance(task, dict) for task in raw_tasks):
            raise ValueError("Phase 1 benign tasks must be a list of objects")
        if any(
            not isinstance(task.get("id"), str)
            or not task.get("id", "").strip()
            or not isinstance(task.get("site"), str)
            or not task.get("site", "").strip()
            for task in raw_tasks
        ):
            raise ValueError("Phase 1 benign tasks require non-empty id and site")
        expected_shards, benign_by_id, tasks_by_site = _expected_planning_shards(
            raw_tasks,
            state,
        )
    except FileNotFoundError:
        return _not_inspected_planning_checkpoints(
            reason_code="planning_inputs_missing",
            path=tasks_path,
        )
    except (OSError, UnicodeError):
        return _not_inspected_planning_checkpoints(
            reason_code="planning_inputs_unreadable",
            path=tasks_path,
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return _not_inspected_planning_checkpoints(
            reason_code="planning_inputs_invalid",
            path=tasks_path,
        )

    try:
        definition = define_run(state)
    except (TypeError, ValueError):
        return _not_inspected_planning_checkpoints(reason_code="run_definition_unavailable")
    if definition.legacy or not definition.run_id:
        return _not_inspected_planning_checkpoints(reason_code="run_definition_unavailable")

    # The shard owner needs the same profile context as the paused-run reuse
    # path for Option A output validation. Every required profile and route
    # overlay is a precondition for inspection; do not invent a denominator
    # from a partial or malformed profile set.
    try:
        site_profile_paths, profile_errors = _generation._collect_site_profiles(
            tasks_by_site,
            run_root / "phase_0c",
        )
    except (AttributeError, OSError, TypeError, UnicodeError, ValueError):
        first_site = next(iter(tasks_by_site), None)
        return _not_inspected_planning_checkpoints(
            reason_code="planning_profile_unavailable",
            path=(
                run_root / "phase_0c" / f"BENCHMARK_PROFILE_{first_site}.json"
                if first_site is not None
                else run_root / "phase_0c"
            ),
        )
    if profile_errors:
        missing_or_invalid = next(
            (
                run_root / "phase_0c" / f"BENCHMARK_PROFILE_{site}.json"
                for site in tasks_by_site
                if site not in site_profile_paths
            ),
            run_root / "phase_0c",
        )
        return _not_inspected_planning_checkpoints(
            reason_code="planning_profile_unavailable",
            path=missing_or_invalid,
        )

    site_profiles: dict[str, dict[str, Any]] = {}
    for site, profile_path in site_profile_paths.items():
        route_path = run_root / "phase_1" / f"TASK_ROUTE_CONTRACTS_{site}.json"
        try:
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            if not isinstance(profile, dict):
                raise ValueError("site profile must be an object")
            _validate_route_contract_source(route_path)
            profile = _with_phase1_route_surface_overlays(site, profile, state_dir=run_root)
            site_profiles[site] = profile
        except (
            AttributeError,
            FileNotFoundError,
            OSError,
            UnicodeError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            return _not_inspected_planning_checkpoints(
                reason_code="planning_profile_unavailable",
                path=route_path if route_path.exists() else profile_path,
            )

    return inspect_planning_shard_checkpoints(
        run_root / "phase_2" / "shards",
        definition=definition,
        expected_shards=expected_shards,
        benign_by_id=benign_by_id,
        site_profiles=site_profiles,
    )


def _expected_planning_shards(
    benign_tasks: list[dict[str, Any]],
    state: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Reuse the Phase 2 runner's pure filtering and shard boundaries."""

    planning_plan = build_planning_shard_specs(
        benign_tasks,
        task_origin=state.get("task_origin") or "all",
        max_tasks_per_site=state.get("max_tasks_per_site"),
        sites_filter=state.get("sites"),
        phase_label="Phase 2 status",
    )
    filtered = planning_plan.filtered_tasks
    tasks_by_site = planning_plan.tasks_by_site

    all_input_ids = [str(task["id"]).strip() for task in filtered]
    if len(all_input_ids) != len(set(all_input_ids)):
        raise ValueError("Phase 1 benign task IDs must be unique")
    expected = [
        {
            "label": str(spec["label"]),
            "site": str(spec["site"]),
            "input_task_ids": [str(value).strip() for value in spec["input_task_ids"]],
        }
        for spec in planning_plan.specs
    ]
    return expected, {str(task["id"]): task for task in filtered}, tasks_by_site


def _validate_route_contract_source(path: Path) -> None:
    """Reject malformed route-contract roots before the overlay adapter reads them."""

    if not path.exists():
        return
    route_contracts = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(route_contracts, Mapping):
        raise ValueError("route contracts must be an object")
    route_families = route_contracts.get("route_families")
    if route_families is None:
        return
    if not isinstance(route_families, list):
        raise ValueError("route_families must be a list")
    for route in route_families:
        if not isinstance(route, Mapping):
            raise ValueError("route family must be an object")
        overlay = route.get("profile_surface_overlay")
        if overlay is not None and not isinstance(overlay, Mapping):
            raise ValueError("profile_surface_overlay must be an object")


def _not_inspected_planning_checkpoints(
    *,
    reason_code: str,
    path: Path | None = None,
) -> dict[str, Any]:
    return {
        "status": "not_inspected",
        "authority": "advisory:phase_2.planning.checkpoint_inspection",
        "expected_count": None,
        "compatible_count": None,
        "pending_count": None,
        "stale_count": None,
        "malformed_count": None,
        "not_inspected_count": None,
        "shards": [],
        "reason_code": reason_code,
        "path": str(path) if path is not None else None,
        "effects": {"writes": False, "model_calls": False, "network": False},
    }
