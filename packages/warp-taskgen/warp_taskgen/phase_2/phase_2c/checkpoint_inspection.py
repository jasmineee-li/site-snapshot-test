"""Read-only Phase 2c feasibility checkpoint status projection."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from warp_taskgen.benchmark_capabilities import (
    infer_instances_config_benchmark,
    normalize_benchmark_name,
)
from warp_taskgen.phase_2.phase_2c.checkpoints import (
    checkpoint_context,
    checkpoint_is_fresh,
    checkpoint_path,
    load_checkpoint,
    task_identity,
)
from warp_taskgen.phase_2.phase_2c.config import (
    _extract_instances_list,
    _filter_instances_for_phase_2c,
    _filter_records_for_sites,
    _gate_phase_2c_benchmark,
    _sites_filter_from_value,
    _validate_phase_2c_instances_payload,
    _with_benchmark,
)
from warp_taskgen.phase_2.phase_2c.fingerprints import _host_fingerprint, _task_content_hash
from warp_taskgen.phase_2.phase_2c.outcomes import _resolve_seed_site
from warp_taskgen.phase_2.phase_2c.policy import FeasibilityPolicyCatalog
from warp_taskgen.run_definition import define_run
from warp_taskgen.runtime_composition import runtime_composition_for_name

_MAX_STATUS_UNITS = 100
_TERMINAL_PHASE_2_STATUSES = frozenset({"complete", "partial_complete"})


def inspect_feasibility_checkpoints(
    run_root: Path,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Inspect Phase 2c task checkpoints without entering the verifier.

    The verifier and aggregate artifact writer remain the only admission
    authorities.  This function reconstructs their non-secret context from
    persisted inputs, then delegates each envelope decision to the existing
    ``load_checkpoint`` validator.  It never writes, seeds, probes, cleans up,
    or changes lifecycle state.
    """

    skip_feasibility = state.get("skip_feasibility")
    if skip_feasibility is not None and type(skip_feasibility) is not bool:
        return _not_inspected(
            reason_code="feasibility_skip_invalid",
            path=run_root / "pipeline_state.json",
        )
    if skip_feasibility is True:
        return _not_inspected(
            reason_code="feasibility_skipped",
            path=run_root / "pipeline_state.json",
        )

    tasks_path = run_root / "phase_2" / "adversarial_tasks.json"
    instances_path, path_error = _instances_path(run_root, state)
    if path_error is not None:
        return _not_inspected(reason_code=path_error[0], path=path_error[1])
    assert instances_path is not None

    terminal_promotion = terminal_promotion_completed(state)
    try:
        raw_tasks = _read_json(tasks_path)
    except FileNotFoundError:
        return _not_inspected(reason_code="feasibility_tasks_missing", path=tasks_path)
    except (OSError, UnicodeError):
        return _not_inspected(reason_code="feasibility_tasks_unreadable", path=tasks_path)
    except json.JSONDecodeError:
        return _not_inspected(reason_code="feasibility_tasks_invalid", path=tasks_path)
    if not isinstance(raw_tasks, list) or any(not isinstance(task, dict) for task in raw_tasks):
        return _not_inspected(reason_code="feasibility_tasks_invalid", path=tasks_path)

    if terminal_promotion:
        infeasible_path = tasks_path.with_name(f"{tasks_path.stem}.infeasible{tasks_path.suffix}")
        try:
            raw_infeasible = _read_json(infeasible_path)
        except FileNotFoundError:
            return _not_inspected(
                reason_code="feasibility_infeasible_missing",
                path=infeasible_path,
            )
        except (OSError, UnicodeError):
            return _not_inspected(
                reason_code="feasibility_infeasible_unreadable",
                path=infeasible_path,
            )
        except json.JSONDecodeError:
            return _not_inspected(
                reason_code="feasibility_infeasible_invalid",
                path=infeasible_path,
            )
        if not isinstance(raw_infeasible, list) or any(
            not isinstance(task, dict) for task in raw_infeasible
        ):
            return _not_inspected(
                reason_code="feasibility_infeasible_invalid",
                path=infeasible_path,
            )
        raw_tasks = [*raw_tasks, *raw_infeasible]

    sites_value = state.get("sites")
    if sites_value is not None and not isinstance(sites_value, str):
        return _not_inspected(
            reason_code="feasibility_sites_invalid",
            path=run_root / "pipeline_state.json",
        )
    sites_filter = _sites_filter_from_value(sites_value)
    selected_tasks = _filter_records_for_sites(raw_tasks, sites_filter)
    if not _task_ids_are_unique(selected_tasks) or not _task_shapes_are_valid(selected_tasks):
        return _not_inspected(reason_code="feasibility_tasks_invalid", path=tasks_path)

    try:
        definition = define_run(state)
    except (TypeError, ValueError):
        return _not_inspected(
            reason_code="run_definition_unavailable",
            path=run_root / "pipeline_state.json",
        )
    if definition.legacy or not definition.run_id:
        return _not_inspected(
            reason_code="run_definition_unavailable",
            path=run_root / "pipeline_state.json",
        )

    try:
        runtime_name = state.get("runtime_composition")
        if runtime_name is not None and not isinstance(runtime_name, str):
            raise ValueError("runtime composition name must be a string")
        runtime_composition = runtime_composition_for_name(runtime_name)
        raw_instances = _read_json(instances_path)
        _validate_phase_2c_instances_payload(raw_instances)
        instances = _extract_instances_list(raw_instances)
        if not instances:
            raise ValueError("instances must not be empty")
        if selected_tasks:
            benchmark = _gate_phase_2c_benchmark(
                task_records=selected_tasks,
                raw_instances=raw_instances,
                instances=instances,
                runtime_composition=runtime_composition,
            )
        else:
            benchmark = infer_instances_config_benchmark(raw_instances)
            if benchmark is None:
                raise ValueError("instances are missing benchmark metadata")
            benchmark = normalize_benchmark_name(benchmark)
        instances = [_with_benchmark(instance, benchmark) for instance in instances]
        verification_instances = _filter_instances_for_phase_2c(
            instances,
            selected_tasks,
            sites_filter=sites_filter,
        )
        if not verification_instances:
            raise ValueError("no instances match selected task sites")
    except FileNotFoundError:
        return _not_inspected(reason_code="feasibility_context_invalid", path=instances_path)
    except (AttributeError, OSError, UnicodeError):
        return _not_inspected(
            reason_code="feasibility_instances_unreadable",
            path=instances_path,
        )
    except json.JSONDecodeError:
        return _not_inspected(reason_code="feasibility_instances_invalid", path=instances_path)
    except (TypeError, ValueError):
        return _not_inspected(reason_code="feasibility_context_invalid", path=instances_path)

    try:
        topology = _host_fingerprint(instances_path.name, verification_instances)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return _not_inspected(
            reason_code="feasibility_topology_unavailable",
            path=instances_path,
        )
    if not _valid_topology(topology):
        return _not_inspected(
            reason_code="feasibility_topology_unavailable",
            path=instances_path,
        )

    try:
        policy_catalog = runtime_composition.feasibility_policy_catalog
        policy_available = _policy_context_available(
            selected_tasks,
            benchmark,
            policy_catalog=policy_catalog,
        )
    except (AttributeError, TypeError, ValueError):
        policy_available = False
    if selected_tasks and not policy_available:
        return _not_inspected(
            reason_code="feasibility_policy_unavailable",
            path=tasks_path,
        )

    checkpoint_dir = run_root / "phase_2" / "feasibility_checkpoints"
    force_value = state.get("force_reverify")
    if force_value is not None and type(force_value) is not bool:
        return _not_inspected(
            reason_code="feasibility_force_invalid",
            path=run_root / "pipeline_state.json",
        )
    force_reverify = force_value is True
    ttl_hours = _ttl_hours(state)
    if ttl_hours is _INVALID:
        return _not_inspected(
            reason_code="feasibility_ttl_invalid",
            path=run_root / "pipeline_state.json",
        )

    counts = {
        "compatible_count": 0,
        "pending_count": 0,
        "stale_count": 0,
        "malformed_count": 0,
        "verified_count": 0,
        "infeasible_count": 0,
    }
    units: list[dict[str, Any]] = []
    for task in selected_tasks:
        task_id = task_identity(task)
        path = checkpoint_path(checkpoint_dir, task_id)
        status, reason_code, outcome = _inspect_task_checkpoint(
            task,
            path=path,
            checkpoint_dir=checkpoint_dir,
            definition=definition,
            topology=topology,
            ttl_hours=ttl_hours,
            force_reverify=force_reverify,
        )
        counts[f"{status}_count"] += 1
        if status == "compatible":
            if outcome == "verified":
                counts["verified_count"] += 1
            elif outcome == "infeasible":
                counts["infeasible_count"] += 1
        if len(units) < _MAX_STATUS_UNITS:
            units.append(
                {
                    "task_id": task_id,
                    "site": str(task.get("site", "")),
                    "status": status,
                    "reason_code": reason_code,
                    "path": str(path),
                    "outcome": outcome,
                }
            )

    return {
        "status": "inspected",
        "authority": "advisory:phase_2.phase_2c.checkpoint_inspection",
        "expected_count": len(selected_tasks),
        **counts,
        "compatible_verified_count": counts["verified_count"],
        "compatible_infeasible_count": counts["infeasible_count"],
        "not_inspected_count": 0,
        "units": units,
        "units_truncated": max(0, len(selected_tasks) - len(units)),
        "reason_code": None,
        "path": None,
        "effects": {"writes": False, "model_calls": False, "network": False},
    }


def _inspect_task_checkpoint(
    task: Mapping[str, Any],
    *,
    path: Path,
    checkpoint_dir: Path,
    definition: Any,
    topology: Mapping[str, str],
    ttl_hours: float | None,
    force_reverify: bool,
) -> tuple[str, str, str | None]:
    seed = task.get("adversarial_data_seed")
    calls = seed.get("editor_calls") if isinstance(seed, Mapping) else []
    if not isinstance(calls, list):
        calls = []
    content_hash = _task_content_hash(
        calls,
        exposure_contract=task.get("exposure_contract"),
    )
    context = checkpoint_context(
        run_id=definition.run_id,
        definition_digest=definition.definition_digest,
        task=task,
        task_content_hash=content_hash,
        topology_fingerprint=topology,
    )
    if context is None:
        return "stale", "checkpoint_context_unavailable", None
    loaded = load_checkpoint(checkpoint_dir, context=context)
    if loaded.reason == "missing":
        return "pending", "checkpoint_missing", None
    if loaded.reason != "compatible":
        return _classify_load_reason(loaded.reason), _reason_code(loaded.reason), None
    if force_reverify:
        return "stale", "checkpoint_reverify_required", None
    if not checkpoint_is_fresh(loaded, ttl_hours=ttl_hours):
        return "stale", "checkpoint_ttl_expired", None
    result = loaded.result or {}
    feasibility = result.get("feasibility")
    outcome = feasibility.get("status") if isinstance(feasibility, Mapping) else None
    if outcome not in {"verified", "infeasible"}:
        return "malformed", "checkpoint_invalid_outcome", None
    return "compatible", "checkpoint_compatible", str(outcome)


def _classify_load_reason(reason: str) -> str:
    if (
        reason in {"legacy_or_schema_mismatch"}
        or reason.endswith("_drift")
        or reason
        in {
            "unbound",
            "topology_drift",
        }
    ):
        return "stale"
    if reason == "cleanup_incomplete" or reason == "tampered" or reason.startswith("malformed"):
        return "malformed"
    return "malformed"


def _reason_code(reason: str) -> str:
    if reason == "tampered":
        return "checkpoint_digest_invalid"
    if reason == "cleanup_incomplete":
        return "checkpoint_cleanup_incomplete"
    if reason.startswith("malformed: invalid outcome"):
        return "checkpoint_invalid_outcome"
    if reason.startswith("malformed:"):
        detail = reason.removeprefix("malformed:").strip()
        return "checkpoint_malformed_" + "_".join(detail.split())[:80]
    if reason == "unbound":
        return "checkpoint_run_or_definition_mismatch"
    return "checkpoint_" + reason


def _policy_context_available(
    tasks: list[dict[str, Any]],
    benchmark: str,
    *,
    policy_catalog: FeasibilityPolicyCatalog,
) -> bool:
    catalog = policy_catalog
    for task in tasks:
        site = _resolve_seed_site(task)
        if catalog.get(benchmark, site) is None:
            return False
    return True


def _task_ids_are_unique(tasks: list[dict[str, Any]]) -> bool:
    ids = [task_identity(task) for task in tasks]
    return all(
        isinstance(task.get("id") or task.get("task_id"), str) and task_id
        for task, task_id in zip(tasks, ids, strict=True)
    ) and len(ids) == len(set(ids))


def _task_shapes_are_valid(tasks: list[dict[str, Any]]) -> bool:
    for task in tasks:
        seed = task.get("adversarial_data_seed")
        if not isinstance(seed, Mapping):
            return False
        calls = seed.get("editor_calls")
        if calls is not None and not isinstance(calls, list):
            return False
        if not _resolve_seed_site(task):
            return False
    return True


def _instances_path(
    run_root: Path,
    state: Mapping[str, Any],
) -> tuple[Path | None, tuple[str, Path] | None]:
    value = state.get("feasibility_instances")
    if not isinstance(value, str) or not value.strip():
        return None, ("feasibility_instances_missing", run_root / "pipeline_state.json")
    path = Path(value.strip()).expanduser()
    if not path.is_absolute():
        return None, ("feasibility_instances_unresolved", path)
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError):
        return None, ("feasibility_instances_unresolved", path)
    if not resolved.is_file() or path.is_symlink():
        return None, ("feasibility_instances_unresolved", path)
    return resolved, None


def _valid_topology(topology: Mapping[str, Any]) -> bool:
    required = ("host_config", "instances_digest", "editor_commit", "dataset_commit")
    if not all(isinstance(topology.get(key), str) and topology[key].strip() for key in required):
        return False
    return not any(topology[key].strip().lower() == "unknown" for key in required)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def terminal_promotion_completed(state: Mapping[str, Any]) -> bool:
    return (
        state.get("status") in _TERMINAL_PHASE_2_STATUSES
        and state.get("phase_2_stage") in {"feasibility", "complete"}
        and isinstance(state.get("feasibility_completed_at"), str)
        and bool(state["feasibility_completed_at"].strip())
    )


def _ttl_hours(state: Mapping[str, Any]) -> float | None | object:
    value = state.get("feasibility_ttl_hours")
    if value is None:
        return None
    if type(value) is int or type(value) is float:
        return float(value) if value >= 0 and math.isfinite(value) else _INVALID
    return _INVALID


_INVALID = object()


def _not_inspected(*, reason_code: str, path: Path | None = None) -> dict[str, Any]:
    return {
        "status": "not_inspected",
        "authority": "advisory:phase_2.phase_2c.checkpoint_inspection",
        "expected_count": None,
        "compatible_count": None,
        "pending_count": None,
        "stale_count": None,
        "malformed_count": None,
        "verified_count": None,
        "infeasible_count": None,
        "compatible_verified_count": None,
        "compatible_infeasible_count": None,
        "not_inspected_count": None,
        "units": [],
        "units_truncated": 0,
        "reason_code": reason_code,
        "path": str(path) if path is not None else None,
        "effects": {"writes": False, "model_calls": False, "network": False},
    }


__all__ = ["inspect_feasibility_checkpoints", "terminal_promotion_completed"]
