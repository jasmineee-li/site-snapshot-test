"""Phase 2 phase_2c stage behavior."""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

from warp_taskgen.atomic_io import write_json_atomic
from warp_taskgen.phase_2 import target_stage as _target_stage
from warp_taskgen.phase_2.output import _effective_task_site, _merge_preserving_unfiltered_sites
from warp_taskgen.phase_2.phase_2c import artifacts as _phase_2c_artifacts
from warp_taskgen.phase_2.phase_2c.artifacts import (
    _merged_dropped_source_data,
    _phase_2c_report_summary_with_artifacts,
    _validate_phase_2c_artifact_payloads,
)
from warp_taskgen.phase_2.phase_2c.config import (
    _extract_instances_list,
    _filter_instances_for_phase_2c,
    _filter_records_for_sites,
    _gate_phase_2_skip_benchmark,
    _gate_phase_2c_benchmark,
    _sites_filter_from_value,
    _terminal_phase_2_status,
    _validate_phase_2c_instances_payload,
    _with_benchmark,
)
from warp_taskgen.phase_2.phase_2c.outcomes import skipped_task_stanza
from warp_taskgen.phase_2.phase_2c.pause_control import promotion_boundary
from warp_taskgen.phase_2.phase_2c.runner import verify_feasibility
from warp_taskgen.phase_2.phase_2c.types import FeasibilityReport
from warp_taskgen.run_definition import define_run
from warp_taskgen.state import load_state_for_current_root, save_state

logger = logging.getLogger(__name__)


def _phase_2c_artifact_facade_bindings() -> dict[str, Any]:
    return {
        "write_json_atomic": write_json_atomic,
        "_merge_preserving_unfiltered_sites": _merge_preserving_unfiltered_sites,
        "_merged_dropped_source_data": _merged_dropped_source_data,
        "_phase_2c_report_summary_with_artifacts": _phase_2c_report_summary_with_artifacts,
        "_validate_phase_2c_artifact_payloads": _validate_phase_2c_artifact_payloads,
    }


def _call_phase_2c_artifact_helper(helper_name: str, *args: Any, **kwargs: Any):
    bindings = _phase_2c_artifact_facade_bindings()
    previous = {name: getattr(_phase_2c_artifacts, name) for name in bindings}
    for name, value in bindings.items():
        setattr(_phase_2c_artifacts, name, value)
    try:
        return getattr(_phase_2c_artifacts, helper_name)(*args, **kwargs)
    finally:
        for name, value in previous.items():
            setattr(_phase_2c_artifacts, name, value)


def _write_phase_2c_artifacts(*args: Any, **kwargs: Any):
    return _call_phase_2c_artifact_helper("_write_phase_2c_artifacts", *args, **kwargs)


def _write_dropped_source_data_sidecar(*args: Any, **kwargs: Any):
    return _call_phase_2c_artifact_helper("_write_dropped_source_data_sidecar", *args, **kwargs)


async def _run_feasibility_stage(
    *,
    args: argparse.Namespace,
    output_path: Path,
    output_dir: Path,
    state_metadata: dict[str, Any],
    prior_phase_2_status: str | None,
) -> int:
    """Phase 2c wrapper — runs verification, writes the three artifacts,
    and records ``phase_2_stage="feasibility"`` in pipeline state.

    Honors ``--skip-feasibility`` (tags every task as ``unverified``) and
    ``--feasibility-only`` (re-verifies whatever is currently on disk).
    """
    infeasible_path = output_path.with_name(output_path.stem + ".infeasible.json")
    dropped_source_path = output_path.with_name(output_path.stem + ".dropped_source_data.json")
    report_path = output_dir / "feasibility_report.json"
    instances_arg = getattr(args, "feasibility_instances", None) or "instances.smoke.json"
    concurrency_raw = getattr(args, "feasibility_concurrency", None)
    concurrency = 10 if concurrency_raw is None else max(1, int(concurrency_raw))
    retry_raw = getattr(args, "feasibility_retry_count", None)
    retry_count = 1 if retry_raw is None else max(0, int(retry_raw))
    ttl_hours = getattr(args, "feasibility_ttl_hours", None)
    force_reverify = bool(getattr(args, "force_reverify", False))
    sites_filter = _sites_filter_from_value(
        getattr(args, "sites", None) or state_metadata.get("sites")
    )
    checkpoint_dir = output_dir / "feasibility_checkpoints"
    state_dir = output_dir.parent

    save_state(
        "phase_2",
        status="running",
        phase_2_stage="feasibility",
        adversarial_tasks_path=str(output_path),
        feasibility_report_path=str(report_path),
        feasibility_infeasible_path=str(infeasible_path),
        skip_feasibility=bool(getattr(args, "skip_feasibility", False)),
        feasibility_instances=str(instances_arg),
        feasibility_concurrency=concurrency,
        feasibility_retry_count=retry_count,
        feasibility_ttl_hours=ttl_hours,
        force_reverify=force_reverify,
        **state_metadata,
    )

    # The outer CLI binds an identified Run Definition to this state root.
    # Legacy/direct phase calls intentionally remain identity-less and cannot
    # manufacture reusable checkpoints.
    checkpoint_run_id: str | None = None
    checkpoint_definition_digest: str | None = None
    try:
        bound_state = load_state_for_current_root() or {}
        definition = define_run(bound_state)
        if not definition.legacy:
            checkpoint_run_id = definition.run_id
            checkpoint_definition_digest = definition.definition_digest
    except (TypeError, ValueError):
        logger.warning(
            "Phase 2c: unable to resolve identified Run Definition; "
            "task checkpoints remain disabled for this legacy invocation",
            exc_info=True,
        )

    try:
        current = json.loads(output_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Phase 2c: failed to read %s: %s", output_path, exc)
        return 1
    if not isinstance(current, list):
        logger.error("Phase 2c: %s must contain a JSON array", output_path)
        return 1

    if getattr(args, "skip_feasibility", False):
        selected_current = _filter_records_for_sites(current, sites_filter)
        try:
            benchmark_name = _gate_phase_2_skip_benchmark(selected_current)
        except ValueError as exc:
            logger.error("Phase 2c benchmark gate failed: %s", exc)
            save_state(
                "phase_2",
                status="failed",
                phase_2_stage="feasibility",
                reason="unsupported_benchmark",
                benchmark_error=str(exc),
                adversarial_tasks_path=str(output_path),
                **state_metadata,
            )
            return 1
        state_metadata["benchmark_name"] = benchmark_name
        logger.warning("Phase 2c: --skip-feasibility active; stamping tasks as unverified")
        stamped = [skipped_task_stanza(task) for task in selected_current]
        report_summary = {
            "generated_at": _target_stage._utcnow_iso(),
            "instances": str(instances_arg),
            "host_fingerprint": {},
            "elapsed_seconds": 0.0,
            "phase_2_status": _terminal_phase_2_status(prior_phase_2_status),
            "verified_count": 0,
            "infeasible_count": 0,
            "skipped_already_verified_count": 0,
            "unverified_count": len(stamped),
            "cleanup_warnings": [],
            "per_site": {},
            "checkpoint_reused_count": 0,
            "source_data_dropped_count": 0,
            "source_data_dropped_by_kind": {},
        }
        with promotion_boundary(state_dir):
            artifact_result = _write_phase_2c_artifacts(
                output_path=output_path,
                infeasible_path=infeasible_path,
                dropped_source_path=dropped_source_path,
                report_path=report_path,
                verified=stamped,
                infeasible=[],
                dropped_source_data=[],
                report_summary=report_summary,
                sites_filter=sites_filter,
                allow_unverified=True,
            )
            summary = artifact_result.summary
            completed_at = _target_stage._utcnow_iso()
            save_state(
                "phase_2",
                status=_terminal_phase_2_status(prior_phase_2_status),
                phase_2_stage="feasibility",
                adversarial_tasks_path=str(output_path),
                feasibility_report_path=str(report_path),
                feasibility_infeasible_path=str(infeasible_path),
                feasibility_dropped_source_data_path=str(dropped_source_path),
                feasibility_completed_at=completed_at,
                feasibility_verified_count=summary["verified_count"],
                feasibility_infeasible_count=summary["infeasible_count"],
                feasibility_skipped_count=int(summary.get("skipped_already_verified_count") or 0),
                feasibility_unverified_count=summary["unverified_count"],
                feasibility_dropped_source_data_count=len(artifact_result.dropped_source_data),
                feasibility_checkpoint_reused_count=0,
                feasibility_skipped_via_flag=True,
                _pause_lock_held=True,
                **state_metadata,
            )
        state_metadata.update(
            {
                "feasibility_report_path": str(report_path),
                "feasibility_infeasible_path": str(infeasible_path),
                "feasibility_dropped_source_data_path": str(dropped_source_path),
                "feasibility_completed_at": completed_at,
                "feasibility_verified_count": summary["verified_count"],
                "feasibility_infeasible_count": summary["infeasible_count"],
                "feasibility_skipped_count": int(
                    summary.get("skipped_already_verified_count") or 0
                ),
                "feasibility_unverified_count": summary["unverified_count"],
                "feasibility_dropped_source_data_count": len(artifact_result.dropped_source_data),
                "feasibility_checkpoint_reused_count": 0,
                "feasibility_skipped_via_flag": True,
            }
        )
        return 0

    instances_path = Path(instances_arg)
    if not instances_path.exists():
        logger.error(
            "Phase 2c requires --feasibility-instances path; %s does not exist",
            instances_path,
        )
        return 1

    try:
        raw_instances = json.loads(instances_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Phase 2c: failed to read instances %s: %s", instances_path, exc)
        return 1
    try:
        _validate_phase_2c_instances_payload(raw_instances)
    except ValueError as exc:
        logger.error("Phase 2c: invalid instances %s: %s", instances_path, exc)
        return 1
    instances = _extract_instances_list(raw_instances)
    if not instances:
        logger.error(
            "Phase 2c: %s contained no instances; feasibility cannot run",
            instances_path,
        )
        return 1

    selected_current = _filter_records_for_sites(current, sites_filter)
    try:
        benchmark_name = _gate_phase_2c_benchmark(
            task_records=selected_current,
            raw_instances=raw_instances,
            instances=instances,
        )
    except ValueError as exc:
        logger.error("Phase 2c benchmark gate failed: %s", exc)
        save_state(
            "phase_2",
            status="failed",
            phase_2_stage="feasibility",
            reason="unsupported_benchmark",
            benchmark_error=str(exc),
            adversarial_tasks_path=str(output_path),
            **state_metadata,
        )
        return 1
    state_metadata["benchmark_name"] = benchmark_name
    instances = [_with_benchmark(instance, benchmark_name) for instance in instances]
    verification_instances = _filter_instances_for_phase_2c(
        instances,
        selected_current,
        sites_filter=sites_filter,
    )
    if not verification_instances:
        logger.error(
            "Phase 2c: no benchmark instances match selected task sites %s",
            sorted({_effective_task_site(task) for task in selected_current}),
        )
        return 1

    logger.info(
        "Phase 2c: verifying %s against %s (concurrency=%d, retry=%d, ttl_hours=%s, force=%s)",
        output_path,
        instances_path,
        concurrency,
        retry_count,
        ttl_hours,
        force_reverify,
    )

    try:
        benchmark_root = None
        if isinstance(raw_instances, dict):
            raw_benchmark_root = raw_instances.get("benchmark_codebase")
            if isinstance(raw_benchmark_root, str) and raw_benchmark_root.strip():
                benchmark_root = Path(raw_benchmark_root.strip())
        verification_input = output_path
        temporary_input: Path | None = None
        if sites_filter is not None:
            temporary = tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                suffix=".adversarial_tasks.json",
                dir=output_dir,
                delete=False,
            )
            with temporary:
                json.dump(selected_current, temporary, indent=2)
            temporary_input = Path(temporary.name)
            verification_input = temporary_input
        report: FeasibilityReport = await verify_feasibility(
            verification_input,
            instances=verification_instances,
            instances_label=instances_path.name,
            benchmark_root=benchmark_root,
            concurrency=concurrency,
            retry_count=retry_count,
            ttl_hours=ttl_hours,
            force_reverify=force_reverify,
            phase_2_status=prior_phase_2_status,
            checkpoint_dir=checkpoint_dir,
            state_dir=output_dir.parent,
            run_id=checkpoint_run_id,
            definition_digest=checkpoint_definition_digest,
        )
    except Exception as exc:
        logger.error("Phase 2c verification failed: %s", exc)
        save_state(
            "phase_2",
            status="failed",
            phase_2_stage="feasibility",
            reason="feasibility_preflight",
            feasibility_error=str(exc),
            adversarial_tasks_path=str(output_path),
            **state_metadata,
        )
        return 1
    finally:
        if "temporary_input" in locals() and temporary_input is not None:
            try:
                temporary_input.unlink()
            except OSError:
                logger.warning("Phase 2c: failed to remove temporary input %s", temporary_input)

    feasibility_metadata = {
        "feasibility_report_path": str(report_path),
        "feasibility_infeasible_path": str(infeasible_path),
        "feasibility_dropped_source_data_path": str(dropped_source_path),
        "feasibility_completed_at": _target_stage._utcnow_iso(),
        "feasibility_verified_count": 0,
        "feasibility_infeasible_count": 0,
        "feasibility_skipped_count": 0,
        "feasibility_unverified_count": 0,
        "feasibility_cleanup_warning_count": len(report.cleanup_warnings),
        "feasibility_dropped_source_data_count": len(report.dropped_source_data),
        "feasibility_checkpoint_reused_count": int(report.reused_checkpoints),
    }
    with promotion_boundary(state_dir):
        # A second boundary immediately before promotion closes the short
        # helper-return race. The writer and terminal state transition share
        # one lock, so a request that wins first cannot leave a partial
        # canonical aggregate behind.
        artifact_result = _write_phase_2c_artifacts(
            output_path=output_path,
            infeasible_path=infeasible_path,
            dropped_source_path=dropped_source_path,
            report_path=report_path,
            verified=report.verified,
            infeasible=report.infeasible,
            dropped_source_data=report.dropped_source_data,
            report_summary=_target_stage._report_summary_dict(
                report,
                instances_path=instances_path.name,
            ),
            sites_filter=sites_filter,
        )
        summary = artifact_result.summary
        verified_count = summary["verified_count"]
        infeasible_count = summary["infeasible_count"]
        skipped_count = int(summary.get("skipped_already_verified_count") or 0)
        fresh_count = verified_count - skipped_count
        feasibility_metadata.update(
            {
                "feasibility_completed_at": _target_stage._utcnow_iso(),
                "feasibility_verified_count": verified_count,
                "feasibility_infeasible_count": infeasible_count,
                "feasibility_skipped_count": skipped_count,
                "feasibility_dropped_source_data_count": len(artifact_result.dropped_source_data),
            }
        )
        save_state(
            "phase_2",
            status=_terminal_phase_2_status(prior_phase_2_status),
            phase_2_stage="feasibility",
            adversarial_tasks_path=str(output_path),
            **feasibility_metadata,
            **state_metadata,
            _pause_lock_held=True,
        )
    logger.info(
        "Phase 2c complete: %d admitted (%d fresh + %d reused via idempotency) / "
        "%d infeasible (elapsed=%.1fs)",
        verified_count,
        fresh_count,
        skipped_count,
        infeasible_count,
        report.elapsed_seconds,
    )
    if report.cleanup_warnings:
        logger.warning(
            "Phase 2c cleanup warnings (%d): first=%s",
            len(report.cleanup_warnings),
            report.cleanup_warnings[0],
        )
    state_metadata.update(feasibility_metadata)
    return 0
