"""Read-only operator status and inspection helpers for WARP Taskgen runs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from warp_taskgen.cost_tracker import tracker as cost_tracker
from warp_taskgen.phase_4.result_summary import summarize_results
from warp_taskgen.run_control_status import build_run_control_projection
from warp_taskgen.state import get_state_dir


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def resolve_run_root(path: Path | None) -> Path:
    if path is None:
        return get_state_dir()
    if path.is_file():
        if path.name == "results.json" and path.parent.name == "phase_4":
            return path.parent.parent
        if path.name == "pipeline_state.json":
            return path.parent
        return path.parent
    return path


def resolve_phase4_results_path(run_root: Path) -> Path | None:
    candidates = [
        run_root / "phase_4" / "results.json",
        run_root / "results.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_task_lookup(run_root: Path) -> dict[str, dict[str, Any]]:
    path = run_root / "phase_2" / "adversarial_tasks.json"
    if not path.exists():
        return {}
    data = load_json(path)
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and isinstance(data.get("tasks"), list):
        items = data["tasks"]
    elif isinstance(data, dict):
        items = [data]
    else:
        return {}
    lookup: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        task_id = item.get("id")
        if isinstance(task_id, str) and task_id:
            lookup[task_id] = item
    return lookup


def _load_json_array_if_present(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = load_json(path)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        rows: list[dict[str, Any]] = []
        for value in data.values():
            if isinstance(value, list):
                rows.extend(item for item in value if isinstance(item, dict))
            elif isinstance(value, dict):
                rows.append(value)
        return rows
    return []


def _task_group_key(task: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    contract = task.get("exposure_contract")
    if not isinstance(contract, dict):
        contract = {}
    feasibility = task.get("feasibility")
    if not isinstance(feasibility, dict):
        feasibility = {}
    default_status = "source_data_dropped" if task.get("drop_reason") else "verified"
    status = str(feasibility.get("status") or default_status)
    reason = str(
        feasibility.get("reason")
        or feasibility.get("failure_reason")
        or task.get("preflight_reason")
        or task.get("drop_reason")
        or "none"
    )
    return (
        str(task.get("site") or "unknown"),
        str(task.get("origin") or "unknown"),
        str(task.get("route_id") or "unknown"),
        str(task.get("target_surface_id") or contract.get("target_surface_id") or "unknown"),
        str(task.get("editor_method") or contract.get("editor_method") or "unknown"),
        f"{status}:{reason}",
    )


def _count_task_rows(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(_task_group_key(task) for task in tasks)
    rows = []
    for (site, origin, route_id, surface, editor_method, status_reason), count in sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        status, _, reason = status_reason.partition(":")
        rows.append(
            {
                "count": count,
                "site": site,
                "origin": origin,
                "route_id": route_id,
                "target_surface_id": surface,
                "editor_method": editor_method,
                "status": status,
                "reason": reason,
            }
        )
    return rows


def summarize_phase2c(run_root: Path) -> dict[str, Any] | None:
    phase_2 = run_root / "phase_2"
    admitted_path = phase_2 / "adversarial_tasks.json"
    infeasible_path = phase_2 / "adversarial_tasks.infeasible.json"
    dropped_path = phase_2 / "adversarial_tasks.dropped_source_data.json"
    report_path = phase_2 / "feasibility_report.json"
    ineligible_path = phase_2 / "exposure_ineligible.json"
    no_contract_path = phase_2 / "dropped_no_contract.json"

    if not any(
        path.exists() for path in (admitted_path, infeasible_path, dropped_path, report_path)
    ):
        return None

    admitted = _load_json_array_if_present(admitted_path)
    infeasible = _load_json_array_if_present(infeasible_path)
    dropped = _load_json_array_if_present(dropped_path)
    ineligible = _load_json_array_if_present(ineligible_path)
    no_contract = _load_json_array_if_present(no_contract_path)
    report = load_json(report_path) if report_path.exists() else {}
    if not isinstance(report, dict):
        report = {}

    return {
        "admitted_count": len(admitted),
        "infeasible_count": len(infeasible),
        "source_data_dropped_count": len(dropped),
        "exposure_ineligible_count": len(ineligible),
        "dropped_no_contract_count": len(no_contract),
        "report": report,
        "admitted_rows": _count_task_rows(admitted),
        "infeasible_rows": _count_task_rows(infeasible),
        "source_data_dropped_rows": _count_task_rows(dropped),
    }


def build_status_payload(path: Path | None = None) -> dict[str, Any]:
    run_root = resolve_run_root(path)
    state_path = run_root / "pipeline_state.json"
    progress_path = run_root / "phase_4" / "progress.json"
    results_path = resolve_phase4_results_path(run_root)
    cost_path = run_root / "cost_report.json"
    manifest_path = run_root / "artifact_manifest.json"
    task_bank_path = run_root / "task_bank" / "events.jsonl"
    reachability_path = run_root / "phase_0c" / "REACHABILITY_REPORT.json"

    payload: dict[str, Any] = {
        "run_root": str(run_root),
        "pipeline_state_path": str(state_path) if state_path.exists() else None,
        "phase4_progress_path": str(progress_path) if progress_path.exists() else None,
        "phase4_results_path": str(results_path) if results_path is not None else None,
        "cost_report_path": str(cost_path) if cost_path.exists() else None,
        "artifact_manifest_path": str(manifest_path) if manifest_path.exists() else None,
        "task_bank_path": str(task_bank_path) if task_bank_path.exists() else None,
        "phase0_reachability_path": str(reachability_path) if reachability_path.exists() else None,
    }
    if state_path.exists():
        state = load_json(state_path)
        if isinstance(state, dict):
            from warp_taskgen.run_definition import (
                _redact_state_for_status,
                define_run,
                plan_resume,
            )

            payload["pipeline_state"] = _redact_state_for_status(state)

            if (
                state.get("step") == "phase_1"
                and state.get("status") in {"running", "failed"}
                and state.get("generate_novel")
            ):
                from warp_taskgen.phase_1.resume_status import inspect_phase1_generation_resume

                phase1_generation = inspect_phase1_generation_resume(run_root, state)
                if phase1_generation is not None:
                    payload["phase1_generation"] = phase1_generation

            try:
                run_definition = define_run(state)
                resume_plan = plan_resume(run_definition, state, run_root=run_root)
            except ValueError as exc:
                payload["run_definition_error"] = str(exc)
            else:
                payload["run_definition"] = run_definition.to_dict()
                payload["resume_plan"] = resume_plan.to_dict()
            payload.update(build_run_control_projection(run_root, state))
            run_control = {
                key: payload[key]
                for key in (
                    "lifecycle_status",
                    "state_status",
                    "supported_stage",
                    "supported",
                    "supported_stages",
                    "pause_request",
                    "pause_request_error",
                    "checkpoint_counts",
                    "feature_checkpoint_counts",
                    "planning_checkpoint_inspection",
                    "next_action",
                    "transition_history",
                    "pause_request_id",
                    "pause_reason_code",
                    "pause_age_seconds",
                )
                if key in payload
            }
            payload["run_control"] = run_control
            # Preserve the existing status keys only for a valid pausing
            # marker; malformed/stale markers stay explanatory and cannot
            # masquerade as an active lifecycle state.
            if run_control.get("pause_request_error"):
                payload.pop("lifecycle_status", None)
                payload.pop("pause_request", None)
    if progress_path.exists():
        progress = load_json(progress_path)
        if isinstance(progress, dict):
            payload["phase4_progress"] = progress
    if results_path is not None:
        results = load_json(results_path)
        if not isinstance(results, list):
            raise ValueError(f"{results_path} must contain a list of result objects")
        task_lookup = load_task_lookup(run_root)
        phase4_summary = summarize_results(results, task_lookup=task_lookup)
        intermediate_path = results_path.parent / "intermediate_asr_summary.json"
        if intermediate_path.exists():
            intermediate = load_json(intermediate_path)
            if isinstance(intermediate, dict):
                payload["phase4_intermediate_asr_path"] = str(intermediate_path)
                payload["phase4_intermediate_asr"] = intermediate
                phase4_summary["intermediate_asr_summary"] = intermediate
                for key in (
                    "intermediate_asr",
                    "intermediate_asr_numerator",
                    "intermediate_asr_denominator",
                    "intermediate_asr_encounter_conditioned",
                    "intermediate_asr_encounter_conditioned_numerator",
                    "intermediate_asr_encounter_conditioned_denominator",
                ):
                    if key in intermediate:
                        phase4_summary[key] = intermediate[key]
        payload["phase4_summary"] = phase4_summary
    cost_inspection = cost_tracker.inspect_report(cost_path)
    cost_observation = cost_inspection.to_dict()
    payload["cost_observation"] = cost_observation
    if cost_inspection.report is not None:
        # Keep the existing report field as the raw artifact for compatibility;
        # the sibling projection is the only derived status authority.
        payload["cost_report"] = cost_inspection.report
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        if isinstance(manifest, dict):
            payload["artifact_manifest"] = manifest
    if reachability_path.exists():
        reachability = load_json(reachability_path)
        if isinstance(reachability, dict):
            payload["phase0_reachability"] = reachability
    phase2c_summary = summarize_phase2c(run_root)
    if phase2c_summary is not None:
        payload["phase2c_summary"] = phase2c_summary
    if task_bank_path.exists():
        from warp_taskgen.task_bank import load_task_bank, summarize_task_bank

        payload["task_bank_summary"] = summarize_task_bank(load_task_bank(task_bank_path))
    return payload


def _fmt_count_map(values: dict[str, Any]) -> str:
    if not values:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in sorted(values.items()))


def _fmt_rate(rate: Any) -> str:
    return f"{rate:.2f}" if isinstance(rate, int | float) else "n/a"


def _format_variant_audit(summary: dict[str, Any]) -> list[str]:
    audit = summary.get("variant_regeneration_audit")
    if not isinstance(audit, dict) or not audit.get("tasks_entered"):
        return []

    lines = [
        (
            "Phase 4 variation: "
            f"entered={audit.get('tasks_entered', 0)} "
            f"planned={audit.get('planned_attempts', 0)} "
            f"generated={audit.get('generated_attempts', 0)} "
            f"rejected_pre_eval={audit.get('rejected_before_eval', 0)} "
            f"evaluated={audit.get('evaluated_attempts', 0)} "
            f"pvpo_valid={audit.get('gate1_valid_evaluations', 0)} "
            f"complied={audit.get('compliant_evaluations', 0)}"
        ),
        (
            "Variation method: resisted, ecologically valid initial attacks are judged "
            "for refusal trigger; fixed strategies may rewrite only the selected payload; "
            "contract-safe variants re-enter PVPO."
        ),
    ]

    triggers = audit.get("judge_trigger_counts") or {}
    if triggers:
        lines.append(f"Variation triggers: {_fmt_count_map(triggers)}")

    generation = audit.get("generation_status_counts") or {}
    if generation:
        lines.append(f"Variation generation: {_fmt_count_map(generation)}")

    rows = audit.get("trigger_strategy_rows") or []
    if rows:
        lines.append("Variation strategy flow:")
        for row in rows[:5]:
            if not isinstance(row, dict):
                continue
            lines.append(
                "  "
                f"{row.get('refusal_trigger', 'unknown')} -> "
                f"{row.get('strategy', 'unknown')}: "
                f"planned={row.get('planned', 0)} "
                f"generated={row.get('generated', 0)} "
                f"rejected={row.get('rejected', 0)} "
                f"evaluated={row.get('evaluated', 0)} "
                f"pvpo_valid={row.get('gate1_valid', 0)} "
                f"complied={row.get('complied', 0)}"
            )

    errors = summary.get("variant_error_buckets") or []
    if errors:
        first = errors[0]
        if isinstance(first, dict):
            lines.append(
                "Top variant rejection: "
                f"{first.get('count', 0)} {first.get('class', 'unknown')}: "
                f"{first.get('reason', '')}"
            )
    return lines


def _format_phase2c_rows(label: str, rows: list[dict[str, Any]], *, limit: int) -> list[str]:
    if not rows:
        return []
    lines = [f"{label}:"]
    for row in rows[: max(limit, 0)]:
        lines.append(
            "  "
            f"{row.get('count', 0)} "
            f"{row.get('site', 'unknown')} "
            f"{row.get('origin', 'unknown')} "
            f"{row.get('target_surface_id', 'unknown')} "
            f"{row.get('editor_method', 'unknown')} "
            f"route={row.get('route_id', 'unknown')} "
            f"{row.get('status', 'unknown')}:{row.get('reason', 'none')}"
        )
    return lines


def format_status_payload(payload: dict[str, Any], *, inspect_limit: int = 5) -> str:
    lines = [f"WARP Taskgen status: {payload['run_root']}"]
    cost_observation = payload.get("cost_observation")
    if isinstance(cost_observation, dict):
        status = cost_observation.get("status", "unknown")
        known_total = cost_observation.get("known_total_cost_usd")
        if isinstance(known_total, int | float):
            known_total_text = f"{known_total:.4f}"
        else:
            known_total_text = "unknown"
        lines.append(
            "Observed cost: "
            f"status={status} "
            f"known_total_cost_usd={known_total_text} "
            f"known_entry_count={cost_observation.get('known_entry_count', 'unknown')} "
            f"unknown_entry_count={cost_observation.get('unknown_entry_count', 'unknown')} "
            f"completeness={cost_observation.get('completeness', 'unknown')}"
        )
    state = payload.get("pipeline_state")
    if isinstance(state, dict):
        lifecycle_status = payload.get("lifecycle_status", state.get("status", "unknown"))
        lines.append(
            "Pipeline: "
            f"step={state.get('step', 'unknown')} "
            f"status={lifecycle_status} "
            f"timestamp={state.get('timestamp', 'unknown')}"
        )
        task_dir_root = state.get("task_dir_root")
        if isinstance(task_dir_root, str) and task_dir_root:
            lines.append(f"Task dir root: {task_dir_root}")
        supported_stage = payload.get("supported_stage")
        if supported_stage is not None or "supported" in payload:
            lines.append(
                "Run control: "
                f"supported_stage={supported_stage or 'none'} "
                f"supported={str(bool(payload.get('supported'))).lower()}"
            )
        counts = payload.get("checkpoint_counts")
        if isinstance(counts, dict):
            lines.append(
                "Run checkpoints: "
                f"queued={counts.get('queued', 'unknown') if counts.get('queued') is not None else 'unknown'} "
                f"admitted={counts.get('admitted', 'unknown') if counts.get('admitted') is not None else 'unknown'} "
                f"completed={counts.get('completed', 'unknown') if counts.get('completed') is not None else 'unknown'} "
                f"authority={counts.get('authority', 'unknown')}"
            )
        planning = payload.get("planning_checkpoint_inspection")
        if isinstance(planning, dict):
            lines.append(
                "Phase 2a planning checkpoints: "
                f"status={planning.get('status', 'unknown')} "
                f"expected={planning.get('expected_count', 'unknown')} "
                f"compatible={planning.get('compatible_count', 'unknown')} "
                f"pending={planning.get('pending_count', 'unknown')} "
                f"stale={planning.get('stale_count', 'unknown')} "
                f"malformed={planning.get('malformed_count', 'unknown')} "
                f"not_inspected={planning.get('not_inspected_count', 'unknown')}"
            )
            for row in planning.get("shards", [])[: max(inspect_limit, 0)]:
                if not isinstance(row, dict):
                    continue
                path = row.get("path") or "unavailable"
                lines.append(
                    "  "
                    f"{row.get('label', 'unknown')}: "
                    f"{row.get('status', 'unknown')} "
                    f"({row.get('reason_code', 'unknown')}) "
                    f"path={path}"
                )
            if planning.get("status") == "not_inspected":
                reason = planning.get("reason_code", "unknown")
                path = planning.get("path") or "unavailable"
                lines.append(f"  reason={reason} path={path}")
    else:
        lines.append("Pipeline: no pipeline_state.json found")

    run_definition = payload.get("run_definition")
    if isinstance(run_definition, dict):
        lines.append(
            "Run definition: "
            f"schema={run_definition.get('schema_version', 'unknown')} "
            f"digest={run_definition.get('definition_digest', 'unknown')} "
            f"run_id={run_definition.get('run_id') or 'not-persisted'} "
            f"legacy={str(bool(run_definition.get('legacy'))).lower()}"
        )
    resume_plan = payload.get("resume_plan")
    if isinstance(resume_plan, dict):
        lines.append(
            "Resume plan: "
            f"mode={resume_plan.get('mode', 'unknown')} "
            f"action={resume_plan.get('lifecycle_action', 'unknown')} "
            f"target={resume_plan.get('target_step') or 'none'}"
        )
    elif payload.get("run_definition_error"):
        lines.append(f"Run definition: unavailable ({payload['run_definition_error']})")
    pause_request = payload.get("pause_request")
    if isinstance(pause_request, dict):
        lines.append(
            "Pause request: "
            f"id={pause_request.get('request_id', 'unknown')} "
            f"reason={pause_request.get('reason_code', 'unknown')} "
            f"age={pause_request.get('age_seconds', 'unknown')}s "
            f"requested_at={pause_request.get('requested_at', 'unknown')}"
        )
    elif payload.get("pause_request_error"):
        lines.append(f"Pause request: malformed ({payload['pause_request_error']})")
    next_action = payload.get("next_action")
    if isinstance(next_action, dict):
        lines.append(f"Next action: {next_action.get('description', 'unknown')}")
        if next_action.get("command"):
            lines.append(f"  command={next_action['command']}")
    history = payload.get("transition_history")
    if isinstance(history, list) and history:
        lines.append(f"Run-control history: {len(history)} advisory event(s)")

    phase1_generation = payload.get("phase1_generation")
    if isinstance(phase1_generation, dict):
        environment_binding = phase1_generation.get("environment_binding")
        if isinstance(environment_binding, dict):
            lines.append(
                "Phase 1 cache identity (advisory): "
                f"{environment_binding.get('name', 'unknown')}="
                f"{environment_binding.get('current', 'unknown')} "
                f"normalized_value={environment_binding.get('normalized_value', 'unknown')} "
                "persisted_in_run_definition="
                f"{str(bool(environment_binding.get('persisted_in_run_definition'))).lower()}"
            )
        if phase1_generation.get("status") == "inspected":
            lines.append(
                "Phase 1 generation (advisory): "
                f"reusable={phase1_generation.get('reusable_tasks', 0)}/"
                f"{phase1_generation.get('requested_tasks', 0)} "
                f"remaining={phase1_generation.get('remaining_tasks', 0)} "
                f"source={phase1_generation.get('reuse_source', 'unknown')}"
            )
            for site in phase1_generation.get("sites", []):
                if not isinstance(site, dict):
                    continue
                lines.append(
                    "  "
                    f"{site.get('site', 'unknown')}: "
                    f"{site.get('cache_status', 'unknown')} "
                    f"({site.get('reason_code', 'unknown')}) "
                    f"reusable={site.get('reusable_tasks', 0)}/"
                    f"{site.get('requested_tasks', 0)}"
                )
        else:
            lines.append(
                "Phase 1 generation (advisory): unavailable "
                f"({phase1_generation.get('reason_code', 'unknown')})"
            )
        merged_output = phase1_generation.get("merged_output")
        if isinstance(merged_output, dict):
            lines.append(
                "  merged_output="
                f"{merged_output.get('status', 'unknown')} "
                f"({merged_output.get('reason_code', 'unknown')})"
            )
        if phase1_generation.get("resume_blocker"):
            lines.append(f"  resume_blocker={phase1_generation['resume_blocker']}")
        lines.append(f"  resume={phase1_generation.get('resume_command', 'unavailable')}")
        lines.append(f"  caution={phase1_generation.get('resume_caveat', 'unavailable')}")

    progress = payload.get("phase4_progress")
    if isinstance(progress, dict):
        lines.append(
            "Phase 4 progress: "
            f"status={progress.get('status', 'unknown')} "
            f"stage={progress.get('stage', 'unknown')} "
            f"initial={progress.get('completed_initial_tasks', 0)}/"
            f"{progress.get('total_tasks', 0)} "
            f"initial_started={progress.get('initial_started_tasks', 0)}/"
            f"{progress.get('total_tasks', 0)} "
            f"initial_active={progress.get('active_initial_tasks', 0)} "
            f"started={progress.get('postprocess_started_tasks', 0)}/"
            f"{progress.get('total_tasks', 0)} "
            f"active={progress.get('active_postprocess_tasks', 0)} "
            f"postprocessed={progress.get('postprocessed_tasks', 0)}/"
            f"{progress.get('total_tasks', 0)} "
            f"postprocess_attempted={progress.get('postprocess_attempted_tasks', 0)}/"
            f"{progress.get('total_tasks', 0)} "
            f"postprocess_failed={progress.get('postprocess_failed_tasks', 0)}"
        )
        variant_progress = progress.get("variant_progress")
        if isinstance(variant_progress, dict) and variant_progress:
            rewrite_attempted = variant_progress.get(
                "rewrite_attempted", variant_progress.get("generation_attempted", 0)
            )
            variant_evaluated = variant_progress.get(
                "variant_evaluated", variant_progress.get("evaluated", 0)
            )
            rejection_records = variant_progress.get(
                "variant_rejection_records", variant_progress.get("generation_failed", 0)
            )
            lines.append(
                "Phase 4 variant progress: "
                f"budget={variant_progress.get('budget_preset', 'unknown')} "
                f"entered={variant_progress.get('entered_tasks', 0)} "
                f"active={variant_progress.get('active_tasks', 0)} "
                f"rewrite_attempted={rewrite_attempted} "
                f"variant_evaluated={variant_evaluated} "
                f"rejection_records={rejection_records} "
                f"pre_browser_rejections={variant_progress.get('pre_browser_rejections', 0)} "
                f"post_eval_rejections={variant_progress.get('post_eval_rejections', 0)} "
                f"schema_validation_failures={variant_progress.get('schema_validation_failures', 0)} "
                f"legacy_generated={variant_progress.get('generation_generated', 0)}/"
                f"{variant_progress.get('generation_attempted', 0)} "
                f"evaluated={variant_progress.get('evaluated', 0)} "
                f"pvpo_valid={variant_progress.get('pvpo_valid', 0)} "
                f"complied={variant_progress.get('complied', 0)}"
            )
        active_workers = progress.get("process_pool_active_workers")
        if isinstance(active_workers, list) and active_workers:
            lines.append("Phase 4 process-pool active workers:")
            for worker in active_workers[:8]:
                if not isinstance(worker, dict):
                    continue
                lines.append(
                    "  "
                    f"worker={worker.get('worker_id')} "
                    f"slot={worker.get('slot_id')} "
                    f"task={worker.get('task_id')} "
                    f"status={worker.get('sidecar_status') or worker.get('status') or worker.get('child_phase4_status') or 'unknown'} "
                    f"step={worker.get('current_phase') or worker.get('current_step') or 'unknown'} "
                    f"trace={worker.get('task_trace_dir') or 'pending'} "
                    f"state={worker.get('state_dir')} "
                    f"stderr={worker.get('stderr')}"
                )

    manifest = payload.get("artifact_manifest")
    if isinstance(manifest, dict):
        artifacts = manifest.get("artifacts")
        artifact_count = len(artifacts) if isinstance(artifacts, list) else 0
        lines.append(
            "Artifact provenance: "
            f"source={manifest.get('artifacts_source', 'unknown')} "
            f"generated_at={manifest.get('generated_at', 'unknown')} "
            f"artifacts={artifact_count}"
        )

    reachability = payload.get("phase0_reachability")
    if isinstance(reachability, dict):
        sites = reachability.get("sites")
        if isinstance(sites, list):
            parts = []
            for site in sites:
                if not isinstance(site, dict):
                    continue
                counts = site.get("channel_counts")
                if not isinstance(counts, dict):
                    counts = {}
                count_text = _fmt_count_map(counts)
                parts.append(
                    f"{site.get('site', 'unknown')}={site.get('status', 'unknown')}({count_text})"
                )
            if parts:
                lines.append(f"Phase 0c reachability: {'; '.join(parts)}")

    task_bank_summary = payload.get("task_bank_summary")
    if isinstance(task_bank_summary, dict):
        lines.append(
            "Task bank: "
            f"events={task_bank_summary.get('total_events', 0)} "
            f"admitted={task_bank_summary.get('admitted_tasks', 0)} "
            f"phase4_results={task_bank_summary.get('phase4_results', 0)} "
            f"sites={_fmt_count_map(task_bank_summary.get('by_site') or {})} "
            f"archetypes={_fmt_count_map(task_bank_summary.get('by_archetype') or {})}"
        )

    phase2c = payload.get("phase2c_summary")
    if isinstance(phase2c, dict):
        report = phase2c.get("report")
        if not isinstance(report, dict):
            report = {}
        per_site = report.get("per_site")
        if not isinstance(per_site, dict):
            per_site = {}
        lines.append(
            "Phase 2c: "
            f"status={report.get('phase_2_status', 'unknown')} "
            f"admitted={phase2c.get('admitted_count', 0)} "
            f"infeasible={phase2c.get('infeasible_count', 0)} "
            f"source_data_dropped={phase2c.get('source_data_dropped_count', 0)} "
            f"exposure_ineligible={phase2c.get('exposure_ineligible_count', 0)} "
            f"no_contract={phase2c.get('dropped_no_contract_count', 0)}"
        )
        if per_site:
            site_parts = []
            for site, counts in sorted(per_site.items()):
                if not isinstance(counts, dict):
                    continue
                site_parts.append(
                    f"{site}:v={counts.get('verified', 0)} "
                    f"i={counts.get('infeasible', 0)} "
                    f"s={counts.get('skipped', 0)}"
                )
            if site_parts:
                lines.append(f"Phase 2c by site: {'; '.join(site_parts)}")
        lines.extend(
            _format_phase2c_rows(
                "Phase 2c admitted buckets",
                phase2c.get("admitted_rows") or [],
                limit=inspect_limit,
            )
        )
        lines.extend(
            _format_phase2c_rows(
                "Phase 2c infeasible buckets",
                phase2c.get("infeasible_rows") or [],
                limit=inspect_limit,
            )
        )
        lines.extend(
            _format_phase2c_rows(
                "Phase 2c source-data drops",
                phase2c.get("source_data_dropped_rows") or [],
                limit=inspect_limit,
            )
        )

    summary = payload.get("phase4_summary")
    if isinstance(summary, dict):
        lines.append(
            "Phase 4 results: "
            f"total={summary.get('total', 0)} "
            f"final_status={_fmt_count_map(summary.get('final_status_counts') or {})} "
            f"sites={_fmt_count_map(summary.get('site_counts') or {})}"
        )
        lines.append(
            "Phase 4 final ASR: "
            f"{summary.get('final_asr_numerator', summary.get('headline_asr_numerator', summary.get('asr_raw_numerator', 0)))} / "
            f"{summary.get('final_asr_denominator', summary.get('headline_asr_denominator', summary.get('asr_raw_denominator', 0)))} = "
            f"{_fmt_rate(summary.get('final_asr', summary.get('headline_asr', summary.get('asr_raw'))))}"
        )
        lines.append(
            "Phase 4 final ASR, encounter-conditioned: "
            f"{summary.get('final_asr_encounter_conditioned_numerator', summary.get('gate1_asr_numerator', summary.get('asr_valid_numerator', 0)))} / "
            f"{summary.get('final_asr_encounter_conditioned_denominator', summary.get('gate1_asr_denominator', summary.get('asr_valid_denominator', 0)))} = "
            f"{_fmt_rate(summary.get('final_asr_encounter_conditioned', summary.get('gate1_asr', summary.get('asr_valid'))))}"
        )
        if "intermediate_asr" in summary:
            lines.append(
                "Phase 4 intermediate ASR: "
                f"{summary.get('intermediate_asr_numerator', 0)} / "
                f"{summary.get('intermediate_asr_denominator', 0)} = "
                f"{_fmt_rate(summary.get('intermediate_asr'))}"
            )
            lines.append(
                "Phase 4 intermediate ASR, encounter-conditioned: "
                f"{summary.get('intermediate_asr_encounter_conditioned_numerator', 0)} / "
                f"{summary.get('intermediate_asr_encounter_conditioned_denominator', 0)} = "
                f"{_fmt_rate(summary.get('intermediate_asr_encounter_conditioned'))}"
            )
        lines.extend(_format_variant_audit(summary))
        inspection = summary.get("inspection_index")
        if isinstance(inspection, list) and inspection and inspect_limit:
            lines.append("Inspect next:")
            for idx, row in enumerate(inspection[: max(inspect_limit, 0)], start=1):
                if not isinstance(row, dict):
                    continue
                lines.append(
                    "  "
                    f"{idx}. [{row.get('priority_reason', 'inspect')}] "
                    f"{row.get('task_id', 'unknown')} "
                    f"{row.get('site', 'unknown')} "
                    f"{row.get('surface', 'unknown')} "
                    f"{row.get('final_status', 'missing')}: "
                    f"{row.get('why', '')}"
                )
                trace = row.get("primary_inspection_trace")
                if isinstance(trace, str) and trace:
                    lines.append(f"     trace={trace}")
                task_id = row.get("task_id")
                results_path = payload.get("phase4_results_path")
                if isinstance(task_id, str) and isinstance(results_path, str):
                    run_root = str(Path(results_path).parent.parent)
                    lines.append(
                        "     next="
                        f"uv run python -m warp_taskgen.main trace task {run_root} "
                        f"{task_id} --iterator --refs"
                    )
            results_path = payload.get("phase4_results_path")
            if isinstance(results_path, str):
                run_root = str(Path(results_path).parent.parent)
                lines.append(
                    f"Trace summary: uv run python -m warp_taskgen.main trace summary {run_root}"
                )
                lines.append(
                    "Trace resistant_unaware slice: "
                    f"uv run python -m warp_taskgen.main trace slice {run_root} "
                    "--outcome resistant_unaware "
                    "--fields task_id,site,action,max_coverage,tp,vea,iterator_stop "
                    "--limit 20"
                )
    else:
        lines.append("Phase 4 results: not found")

    return "\n".join(lines)


def build_inspection_payload(task_id: str, path: Path | None = None) -> dict[str, Any]:
    status = build_status_payload(path)
    summary = status.get("phase4_summary")
    if not isinstance(summary, dict):
        raise ValueError("Phase 4 results are required for task inspection")
    rows = summary.get("inspection_index")
    if not isinstance(rows, list):
        raise ValueError("Phase 4 inspection index missing from summary")
    for row in rows:
        if isinstance(row, dict) and row.get("task_id") == task_id:
            return {
                "run_root": status["run_root"],
                "phase4_results_path": status.get("phase4_results_path"),
                "task": row,
            }
    raise KeyError(f"task {task_id!r} not found in Phase 4 inspection index")


def format_inspection_payload(payload: dict[str, Any]) -> str:
    task = payload.get("task")
    if not isinstance(task, dict):
        raise ValueError("inspection payload missing task object")
    lines = [
        f"WARP Taskgen task inspection: {task.get('task_id', 'unknown')}",
        (f"Status: {task.get('final_status', 'missing')} ({task.get('outcome_fine', 'missing')})"),
        (
            "Surface: "
            f"{task.get('site', 'unknown')} "
            f"{task.get('surface', 'unknown')} "
            f"{task.get('editor_method', 'unknown')} "
            f"route={task.get('route_variant', 'unknown')}"
        ),
        f"Why: {task.get('why', '')}",
    ]
    lines.append(
        "PVPO: "
        f"max_coverage={task.get('max_coverage', 'unknown')} "
        f"initial_max_coverage={task.get('initial_max_coverage', 'unknown')} "
        f"reference_step={task.get('reference_step', 'unknown')} "
        f"status={task.get('pvpo_status', 'unknown')}"
    )
    lines.append(
        "Judge: "
        f"trigger={task.get('judge_trigger', 'unknown')} "
        f"confidence={task.get('judge_confidence', 'unknown')}"
    )
    successful_strategy = task.get("successful_strategy")
    if isinstance(successful_strategy, str) and successful_strategy:
        lines.append(f"Successful strategy: {successful_strategy}")
    for key in (
        "initial_trace",
        "current_trace",
        "successful_variant_trace",
        "primary_inspection_trace",
    ):
        value = task.get(key)
        if isinstance(value, str) and value:
            lines.append(f"{key}: {value}")
    artifacts = task.get("artifacts")
    if isinstance(artifacts, dict) and artifacts:
        lines.append("Artifacts:")
        for key, value in sorted(artifacts.items()):
            lines.append(f"  {key}: {value}")
    rejected = task.get("rejected_variants")
    if isinstance(rejected, list) and rejected:
        lines.append("Rejected variants:")
        for item in rejected:
            if not isinstance(item, dict):
                continue
            lines.append(
                "  "
                f"{item.get('index', '?')} "
                f"{item.get('strategy', 'unknown')} "
                f"{item.get('status', 'unknown')}: "
                f"{item.get('reason', '')}"
            )
    return "\n".join(lines)
