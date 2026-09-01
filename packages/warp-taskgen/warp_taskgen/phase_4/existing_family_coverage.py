"""Read-only coverage projection for existing WARP task families."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from warp_taskgen.phase_4 import result_summary as phase4_result_summary
from warp_taskgen.phase_4.artifact_audit import load_json

_EXISTING_FAMILY_COVERAGE_SCHEMA_VERSION = "existing_family_coverage_v1"


def build_existing_family_coverage(
    path: Path,
    *,
    task_bank_path: Path | None = None,
) -> dict[str, Any]:
    """Compose a read-only Phase 1 → Phase 4 coverage funnel."""

    base = path.parent if path.is_file() else path
    run_dir = base.parent if base.name == "phase_4" else base
    warnings: list[str] = []

    phase1_path = run_dir / "phase_1" / "benign_tasks.json"
    phase1_rows, phase1_error = _coverage_rows(phase1_path, label="Phase 1 candidates")
    if phase1_error:
        warnings.append(f"Phase 1 candidates unavailable: {phase1_error}")

    phase3_path = run_dir / "phase_3" / "contracts.json"
    phase3_rows, phase3_error = _coverage_rows(phase3_path, label="Phase 3 contracts")
    if phase3_error:
        warnings.append(f"Phase 3 validation unavailable: {phase3_error}")

    phase2_path = run_dir / "phase_2" / "adversarial_tasks.json"
    infeasible_path = run_dir / "phase_2" / "adversarial_tasks.infeasible.json"
    dropped_path = run_dir / "phase_2" / "adversarial_tasks.dropped_source_data.json"
    phase2_rows, phase2_error = _coverage_rows(
        phase2_path,
        label="Phase 2c verified tasks",
    )
    infeasible_rows, infeasible_error = _coverage_rows(
        infeasible_path,
        label="Phase 2c infeasible tasks",
    )
    dropped_rows, dropped_error = _coverage_rows(
        dropped_path,
        label="Phase 2c dropped source data",
    )
    for label, error in (
        ("Phase 2c verified tasks", phase2_error),
        ("Phase 2c infeasible tasks", infeasible_error),
        ("Phase 2c dropped source data", dropped_error),
    ):
        if error and error != "artifact not found":
            warnings.append(f"{label} unavailable: {error}")

    report_path = run_dir / "phase_2" / "feasibility_report.json"
    phase2_report, report_error = _coverage_object(report_path, label="Phase 2c report")
    if report_error and report_error != "artifact not found":
        warnings.append(f"Phase 2c report unavailable: {report_error}")

    pipeline_state_path = run_dir / "pipeline_state.json"
    pipeline_state, pipeline_state_error = _coverage_object(
        pipeline_state_path,
        label="pipeline state",
    )
    if pipeline_state_error and pipeline_state_error != "artifact not found":
        warnings.append(f"Phase 2c pipeline state unavailable: {pipeline_state_error}")

    phase2c_statuses, dropped_by_kind, status_errors = _phase2c_coverage_statuses(
        phase2_rows=phase2_rows,
        phase2_error=phase2_error,
        infeasible_rows=infeasible_rows,
        infeasible_error=infeasible_error,
        dropped_rows=dropped_rows,
        dropped_error=dropped_error,
        report=phase2_report,
        warnings=warnings,
    )
    phase2_error = status_errors["verified"] or phase2_error
    infeasible_error = status_errors["infeasible"] or infeasible_error
    phase2c_state = _coverage_phase2c_state(
        pipeline_state=pipeline_state,
        pipeline_state_error=pipeline_state_error,
        report=phase2_report,
        report_error=report_error,
    )

    bank_path = task_bank_path or run_dir / "task_bank" / "events.jsonl"
    bank_events, bank_error = _coverage_task_bank(bank_path)
    bank_summary: dict[str, Any] | None = None
    if bank_error and bank_error != "artifact not found":
        warnings.append(f"Task bank unavailable: {bank_error}")
    if bank_events is not None:
        from warp_taskgen.task_bank import is_active_task_bank_event, summarize_task_bank

        bank_summary = summarize_task_bank(bank_events)
        active_events, retired_events = [], []
        for event in bank_events:
            if event.get("event_type") != "admit_task":
                continue
            (active_events if is_active_task_bank_event(event) else retired_events).append(event)
    else:
        active_events = None
        retired_events = None

    results_path = (
        path
        if path.is_file() and path.name == "results.json"
        else next(
            (
                candidate
                for candidate in (run_dir / "phase_4" / "results.json", run_dir / "results.json")
                if candidate.exists()
            ),
            None,
        )
    )
    results: list[dict[str, Any]] | None = None
    results_error = "artifact not found"
    if results_path is not None:
        results, results_error = _coverage_rows(results_path, label="Phase 4 results")
        if results_error:
            warnings.append(f"Phase 4 results unavailable: {results_error}")

    phase1_task_map = {
        str(row.get("id") or "").strip(): row
        for row in phase1_rows or []
        if str(row.get("id") or "").strip()
    }
    phase3_valid_rows = [
        dict(row.get("task")) if isinstance(row.get("task"), Mapping) else dict(row)
        for row in phase3_rows or []
        if str(row.get("validity_status") or "").strip().lower() == "valid"
    ]
    phase2_admitted_rows = [
        row for row in phase2_rows or [] if _coverage_phase2c_status(row) == "verified"
    ]
    result_task_ids = {str(row.get("task_id") or "") for row in results or []}
    known_task_ids = set(phase1_task_map)
    known_task_ids.update(str(row.get("id") or "") for row in phase2_rows or [])
    known_task_ids.update(str(row.get("id") or "") for row in infeasible_rows or [])
    known_task_ids.update(
        str(value).strip()
        for event in bank_events or []
        for value in (event.get("task_id"), event.get("event_id"))
        if str(value or "").strip()
    )
    for task_id in sorted(result_task_ids - known_task_ids):
        if task_id:
            warnings.append(
                f"result task {task_id} is not present in Phase 1/Phase 2c/task-bank artifacts"
            )
    for row in results or []:
        if _coverage_result_failed(row):
            task_id = str(row.get("task_id") or "<unknown>")
            status = str(row.get("final_status") or "missing").strip().lower()
            outcome = str(row.get("outcome_fine") or "missing").strip()
            warnings.append(
                f"Phase 4 result task {task_id} retained as failed/attrition: "
                f"final_status={status}, outcome_fine={outcome}"
            )

    candidate_metric = _coverage_metric(
        None if phase1_rows is None else len(phase1_rows),
        source="phase_1/benign_tasks.json",
        reason=phase1_error,
    )
    validated_metric = _coverage_metric(
        None if phase3_rows is None else len(phase3_valid_rows),
        source="phase_3/contracts.json",
        reason=phase3_error,
    )
    admitted_metric = _coverage_metric(
        None if phase2_rows is None else phase2c_statuses["verified"],
        source="phase_2/adversarial_tasks.json",
        reason=phase2_error,
    )
    active_metric = _coverage_metric(
        None if bank_summary is None else bank_summary["active_admitted_tasks"],
        source=_coverage_source_label(run_dir, bank_path),
        reason=bank_error,
    )
    retired_metric = _coverage_metric(
        None if bank_summary is None else bank_summary["retired_admitted_tasks"],
        source=_coverage_source_label(run_dir, bank_path),
        reason=bank_error,
    )
    evaluated_metric = _coverage_metric(
        None if results is None else len(results),
        source=_coverage_source_label(run_dir, results_path),
        reason=results_error,
    )
    failed_metric = _coverage_metric(
        None if results is None else sum(1 for row in results if _coverage_result_failed(row)),
        source=_coverage_source_label(run_dir, results_path),
        reason=results_error,
    )
    stage_records: list[tuple[str, Mapping[str, Any]]] = []
    stage_records.extend(("candidate", row) for row in phase1_rows or [])
    stage_records.extend(("validated", row) for row in phase3_valid_rows)
    stage_records.extend(("admitted", row) for row in phase2_admitted_rows)
    stage_records.extend(("active", row) for row in active_events or [])
    stage_records.extend(("retired", row) for row in retired_events or [])
    stage_records.extend(
        ("evaluated", _coverage_result_task(row, phase1_task_map, phase2_rows))
        for row in results or []
    )
    stage_records.extend(
        ("failed", _coverage_result_task(row, phase1_task_map, phase2_rows))
        for row in results or []
        if _coverage_result_failed(row)
    )

    unavailable: dict[str, str] = {}
    for name, metric in (
        ("candidate", candidate_metric),
        ("validated", validated_metric),
        ("admitted", admitted_metric),
        ("active", active_metric),
        ("retired", retired_metric),
        ("evaluated", evaluated_metric),
        ("failed", failed_metric),
    ):
        if metric["status"] == "unavailable" and metric.get("reason"):
            unavailable[name] = f"{metric['reason']}: {metric['source']}"
    for name, count, error, source in (
        ("verified", phase2c_statuses["verified"], phase2_error, phase2_path),
        ("unverified", phase2c_statuses["unverified"], phase2_error, phase2_path),
        ("skipped", phase2c_statuses["skipped"], phase2_error, phase2_path),
        ("infeasible", phase2c_statuses["infeasible"], infeasible_error, infeasible_path),
        ("dropped_source", phase2c_statuses["dropped_source"], dropped_error, dropped_path),
    ):
        if count is None:
            unavailable[f"phase2c.{name}"] = (
                f"{error or 'artifact not found'}: {source.relative_to(run_dir)}"
            )

    return {
        "schema_version": _EXISTING_FAMILY_COVERAGE_SCHEMA_VERSION,
        "run_dir": str(run_dir),
        "phase2c_state": phase2c_state,
        "funnel": {
            "candidate": candidate_metric,
            "validated": validated_metric,
            "admitted": admitted_metric,
            "active": active_metric,
            "retired": retired_metric,
            "evaluated": evaluated_metric,
            "failed": failed_metric,
        },
        "phase2c_statuses": phase2c_statuses,
        "dropped_source_by_kind": dropped_by_kind,
        "breakdowns": _coverage_breakdowns(stage_records),
        "sources": {
            "phase1_candidates": _coverage_source(phase1_path, phase1_rows, phase1_error),
            "phase3_contracts": _coverage_source(phase3_path, phase3_rows, phase3_error),
            "phase2c_verified": _coverage_source(phase2_path, phase2_rows, phase2_error),
            "phase2c_infeasible": _coverage_source(
                infeasible_path,
                infeasible_rows,
                infeasible_error,
            ),
            "phase2c_dropped_source": _coverage_source(dropped_path, dropped_rows, dropped_error),
            "phase2c_report": _coverage_source(report_path, phase2_report, report_error),
            "task_bank": _coverage_source(bank_path, bank_events, bank_error),
            "phase4_results": _coverage_source(results_path, results, results_error),
        },
        "unavailable": unavailable,
        "warnings": warnings,
    }


def _coverage_source_label(run_dir: Path, path: Path | None) -> str:
    if path is None:
        return "phase_4/results.json"
    return str(path.relative_to(run_dir)) if path.is_relative_to(run_dir) else str(path)


def _coverage_metric(
    count: int | None,
    *,
    source: str,
    reason: str | None,
) -> dict[str, Any]:
    return {
        "count": count,
        "status": "available" if count is not None else "unavailable",
        "source": source,
    } | ({"reason": reason or "owning artifact unavailable"} if count is None else {})


def _coverage_source(
    path: Path | None,
    rows: list[dict[str, Any]] | dict[str, Any] | None,
    error: str | None,
) -> dict[str, Any]:
    return {
        "path": str(path) if path is not None else None,
        "status": "available" if rows is not None and error is None else "unavailable",
    } | ({"reason": error} if error else {})


def _coverage_phase2c_state(
    *,
    pipeline_state: dict[str, Any] | None,
    pipeline_state_error: str | None,
    report: dict[str, Any] | None,
    report_error: str | None,
) -> dict[str, Any]:
    pipeline_owned = pipeline_state is not None and (
        pipeline_state.get("step") == "phase_2"
        or pipeline_state.get("phase_2_stage") in {"feasibility", "complete"}
    )
    payload = pipeline_state if pipeline_owned else report
    source = "pipeline_state.json" if pipeline_owned else "phase_2/feasibility_report.json"
    raw_status = payload.get("status") or payload.get("phase_2_status") if payload else None
    if not isinstance(raw_status, str) or not raw_status.strip():
        return {
            "status": None,
            "reason": pipeline_state_error or report_error or "owning artifact unavailable",
            "source": source,
        }
    state: dict[str, Any] = {"status": raw_status.strip().lower(), "source": source}
    for key in ("reason", "failure_reason", "feasibility_error", "benchmark_error", "error"):
        value = payload.get(key) if payload else None
        if isinstance(value, str) and value.strip():
            state["reason"] = value.strip()
            break
    return state


def _coverage_rows(
    path: Path,
    *,
    label: str,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    payload, error = _coverage_payload(path)
    if error is not None:
        return None, error
    if isinstance(payload, dict):
        payload = payload.get("tasks", payload.get("results"))
    if not isinstance(payload, list):
        return None, f"{label} artifact must be a JSON array"
    if not all(isinstance(item, dict) for item in payload):
        return None, f"{label} artifact contains non-object rows"
    return payload, None


def _coverage_object(
    path: Path,
    *,
    label: str,
) -> tuple[dict[str, Any] | None, str | None]:
    payload, error = _coverage_payload(path)
    if error is not None:
        return None, error
    if not isinstance(payload, dict):
        return None, f"{label} artifact must be a JSON object"
    return payload, None


def _coverage_payload(path: Path) -> tuple[Any, str | None]:
    if not path.exists():
        return None, "artifact not found"
    try:
        return load_json(path), None
    except (OSError, ValueError) as exc:
        return None, f"invalid JSON: {exc}"


def _coverage_task_bank(
    path: Path,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    if not path.exists():
        return None, "artifact not found"
    try:
        from warp_taskgen.task_bank import TaskBankError, load_task_bank

        return load_task_bank(path), None
    except (OSError, TaskBankError, ValueError) as exc:
        return None, f"invalid task-bank artifact: {exc}"


def _coverage_phase2c_status(row: Mapping[str, Any]) -> str | None:
    feasibility = row.get("feasibility")
    status = feasibility.get("status") if isinstance(feasibility, Mapping) else None
    return str(status).strip().lower() or None if status is not None else None


def _coverage_status_error(
    rows: list[dict[str, Any]] | None,
    *,
    expected: tuple[str, ...],
    label: str,
) -> str | None:
    if rows is None:
        return None
    invalid = [
        f"{row.get('id') or '<unknown>'}={status or '<missing>'!r}"
        for row in rows
        if (status := _coverage_phase2c_status(row)) not in expected
    ]
    return (
        f"{label} artifact contains invalid feasibility.status ({', '.join(invalid)})"
        if invalid
        else None
    )


def _phase2c_coverage_statuses(
    *,
    phase2_rows: list[dict[str, Any]] | None,
    phase2_error: str | None,
    infeasible_rows: list[dict[str, Any]] | None,
    infeasible_error: str | None,
    dropped_rows: list[dict[str, Any]] | None,
    dropped_error: str | None,
    report: dict[str, Any] | None,
    warnings: list[str],
) -> tuple[dict[str, int | None], dict[str, int], dict[str, str | None]]:
    verified_error = phase2_error or _coverage_status_error(
        phase2_rows,
        expected=("verified", "unverified"),
        label="Phase 2c verified tasks",
    )
    infeasible_status_error = infeasible_error or _coverage_status_error(
        infeasible_rows,
        expected=("infeasible",),
        label="Phase 2c infeasible tasks",
    )
    if verified_error and phase2_error is None:
        warnings.append(f"Phase 2c verified tasks unavailable: {verified_error}")
    if infeasible_status_error and infeasible_error is None:
        warnings.append(f"Phase 2c infeasible tasks unavailable: {infeasible_status_error}")
    phase2_available = phase2_rows is not None and verified_error is None
    infeasible_available = infeasible_rows is not None and infeasible_status_error is None
    status_counts = Counter(_coverage_phase2c_status(row) for row in phase2_rows or [])
    verified = status_counts["verified"] if phase2_available else None
    unverified = status_counts["unverified"] if phase2_available else None
    infeasible = len(infeasible_rows) if infeasible_available else None
    dropped = len(dropped_rows) if dropped_rows is not None else None
    skipped = (
        sum(
            isinstance(row.get("feasibility"), Mapping)
            and "last_reverify_skipped_at" in row["feasibility"]
            for row in phase2_rows or []
        )
        if phase2_available
        else None
    )

    statuses = {
        "verified": verified,
        "infeasible": infeasible,
        "skipped": skipped,
        "unverified": unverified,
        "dropped_source": dropped,
    }
    if report is not None:
        for key, actual in (
            ("verified_count", verified),
            ("infeasible_count", infeasible),
            ("skipped_already_verified_count", skipped),
            ("unverified_count", unverified),
            ("source_data_dropped_count", dropped),
        ):
            reported = report.get(key)
            if not isinstance(reported, int) or isinstance(reported, bool) or reported < 0:
                reported = None
            if reported is not None and actual is not None and reported != actual:
                warnings.append(
                    "Phase 2c feasibility_report.json "
                    f"{key}={reported} disagrees with owning artifact count={actual}"
                )

    dropped_by_kind: dict[str, int] = {}
    for row in dropped_rows or []:
        issue = row.get("source_data_issue")
        kind = str(issue.get("kind") or "unknown") if isinstance(issue, Mapping) else "unknown"
        dropped_by_kind[kind] = dropped_by_kind.get(kind, 0) + 1
    return (
        statuses,
        dict(sorted(dropped_by_kind.items())),
        {
            "verified": verified_error,
            "unverified": verified_error,
            "skipped": verified_error,
            "infeasible": infeasible_status_error,
            "dropped_source": dropped_error,
        },
    )


def _coverage_result_failed(result: Mapping[str, Any]) -> bool:
    status = str(result.get("final_status") or "missing").strip().lower()
    return status not in (phase4_result_summary.COMPLIED_FINAL_STATUSES | {"resistant"})


def _coverage_result_task(
    result: Mapping[str, Any],
    phase1_task_map: Mapping[str, Mapping[str, Any]],
    phase2_rows: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    task_id = str(result.get("task_id") or "")
    if task_id in phase1_task_map:
        return dict(phase1_task_map[task_id])
    return next(
        (dict(row) for row in phase2_rows or [] if str(row.get("id") or "") == task_id),
        {"id": task_id},
    )


def _coverage_dimensions(record: Mapping[str, Any]) -> dict[str, str]:
    task_value = record.get("task")
    task = dict(task_value) if isinstance(task_value, Mapping) else record
    contract = task.get("exposure_contract")
    contract = contract if isinstance(contract, Mapping) else {}
    provenance = task.get("task_provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    carrier = record.get("carrier_contract")
    surface = task.get("target_surface_id") or contract.get("target_surface_id")
    if isinstance(carrier, Mapping):
        surface = surface or carrier.get("target_surface_id")
    archetype = (
        provenance.get("archetype_id") or task.get("archetype_id") or record.get("archetype_id")
    )
    return {
        "site": str(task.get("site") or record.get("site") or "unknown"),
        "origin": str(task.get("origin") or record.get("origin") or "unknown"),
        "surface": str(surface or "unknown"),
        "task_archetype": str(archetype or "unknown"),
    }


def _coverage_breakdowns(
    stage_records: Iterable[tuple[str, Mapping[str, Any]]],
) -> dict[str, dict[str, dict[str, int]]]:
    dimensions = {name: {} for name in ("site", "origin", "surface", "task_archetype")}
    for stage, record in stage_records:
        values = _coverage_dimensions(record)
        for dimension, value in values.items():
            bucket = dimensions[dimension].setdefault(value, {})
            bucket[stage] = bucket.get(stage, 0) + 1
    return {dimension: dict(sorted(values.items())) for dimension, values in dimensions.items()}
