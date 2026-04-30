"""Phase 4 artifact-level audit helpers.

These helpers are intentionally observational. They summarize already-written
Phase 4 results and variant-generation artifacts without changing admission,
PVPO, rewards, or final-status semantics.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from worldsim.phase_4.result_summary import (
    COMPLIED_FINAL_STATUSES,
    ecologically_valid,
    summarize_results,
    task_editor_method,
    task_origin,
    task_route_variant,
    task_site,
    task_surface,
    variant_adversarial_reward_passed,
)


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_json_or_empty(path: Path) -> dict[str, Any]:
    try:
        data = load_json(path)
    except FileNotFoundError:
        return {}
    return data if isinstance(data, dict) else {}


def resolve_phase4_results_path(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = [
        path / "phase_4" / "results.json",
        path / "results.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"could not find Phase 4 results at {path}/phase_4/results.json or {path}/results.json"
    )


def phase4_dir_for_results(results_path: Path) -> Path:
    return results_path.parent


def default_task_paths(results_path: Path) -> list[Path]:
    paths: list[Path] = []
    parent = results_path.parent
    if parent.name == "phase_4":
        paths.append(parent.parent / "phase_2" / "adversarial_tasks.json")
    paths.append(parent / "phase_2" / "adversarial_tasks.json")
    return [path for path in paths if path.exists()]


def load_task_lookup(paths: list[Path]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for path in paths:
        data = load_json(path)
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and isinstance(data.get("tasks"), list):
            items = data["tasks"]
        elif isinstance(data, dict):
            items = [data]
        else:
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            task_id = item.get("id")
            if isinstance(task_id, str) and task_id:
                lookup[task_id] = item
    return lookup


def _count_map(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _compact_text(value: Any, *, limit: int = 220) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1].rstrip()}..."


def _strategy_name(value: Any) -> str:
    if isinstance(value, dict):
        name = value.get("strategy")
        if isinstance(name, str) and name.strip():
            return name.strip()
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "unknown"


def _judge(result: dict[str, Any]) -> dict[str, Any]:
    variation = result.get("strategy_variation")
    if not isinstance(variation, dict):
        return {}
    judge = variation.get("judge_diagnosis")
    return judge if isinstance(judge, dict) else {}


def _variant_results(result: dict[str, Any]) -> list[dict[str, Any]]:
    variation = result.get("strategy_variation")
    if not isinstance(variation, dict):
        return []
    raw = variation.get("variant_results")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _variant_generation_records(result: dict[str, Any]) -> list[dict[str, Any]]:
    variation = result.get("strategy_variation")
    if not isinstance(variation, dict):
        return []
    raw = variation.get("variant_generation_records")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _generation_record_status(record: dict[str, Any]) -> str:
    status = record.get("status")
    if isinstance(status, str) and status.strip():
        return status.strip()
    if isinstance(record.get("variant"), dict):
        return "generated"
    if isinstance(record.get("error"), str):
        return "error"
    return "unknown"


def discover_variant_generation_artifacts(phase4_dir: Path) -> list[dict[str, Any]]:
    """Read variant-generation attempt artifacts under a Phase 4 directory."""

    attempts: list[dict[str, Any]] = []
    for summary_path in sorted(
        phase4_dir.rglob("variant_generation/*/*/request_summary.json")
    ):
        attempt_dir = summary_path.parent
        request_summary = load_json_or_empty(summary_path)
        host_validation = load_json_or_empty(attempt_dir / "host_validation.json")
        failure_context = load_json_or_empty(attempt_dir / "failure_context.json")
        payload_diff = load_json_or_empty(attempt_dir / "payload_diff.json")
        task_id = request_summary.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            task_id = attempt_dir.parent.parent.parent.name
        attempt = {
            "task_id": task_id,
            "strategy_index": request_summary.get("strategy_index"),
            "strategy": str(request_summary.get("strategy") or "unknown"),
            "attempt": str(request_summary.get("attempt") or attempt_dir.name),
            "generation_status": str(request_summary.get("status") or "missing"),
            "generation_reason": _compact_text(request_summary.get("reason"), limit=260),
            "retry_feedback": _compact_text(
                request_summary.get("retry_feedback"), limit=260
            ),
            "host_status": str(host_validation.get("status") or "missing"),
            "host_reason": _compact_text(host_validation.get("reason"), limit=260),
            "has_failure_context": bool(failure_context),
            "failure_context_schema_version": failure_context.get("schema_version"),
            "failure_context_trace_digest_status": (
                failure_context.get("trace_digest", {}).get("trace_digest_status")
                if isinstance(failure_context.get("trace_digest"), dict)
                else request_summary.get("failure_context_trace_digest_status")
            ),
            "has_payload_diff": bool(payload_diff),
            "payload_changed_seed": payload_diff.get("changed_seed"),
            "payload_meaningful_token_change": payload_diff.get(
                "meaningful_token_change"
            ),
            "payload_revised_chars": payload_diff.get("revised_chars"),
            "payload_attack_witness_offset": payload_diff.get("attack_witness_offset"),
            "payload_max_attack_witness_offset": payload_diff.get(
                "max_attack_witness_offset"
            ),
            "artifact_dir": str(attempt_dir),
        }
        attempts.append(attempt)
    return attempts


def _artifact_flags(
    *,
    result: dict[str, Any],
    attempts: list[dict[str, Any]],
    generated_records: int,
    evaluated: int,
) -> list[str]:
    flags: list[str] = []
    has_variation = isinstance(result.get("strategy_variation"), dict)
    if has_variation and not attempts:
        flags.append("missing_variant_generation_artifacts")
    if attempts and not any(attempt.get("has_failure_context") for attempt in attempts):
        flags.append("missing_failure_context_artifacts")
    generated_attempts = [
        attempt for attempt in attempts if attempt.get("generation_status") == "generated"
    ]
    generated_without_diff = [
        attempt for attempt in generated_attempts if not attempt.get("has_payload_diff")
    ]
    if generated_without_diff:
        flags.append("generated_without_payload_diff")
    if generated_records and attempts and generated_records != len(generated_attempts):
        flags.append("generation_record_artifact_mismatch")
    host_passed = [attempt for attempt in attempts if attempt.get("host_status") == "passed"]
    if host_passed and evaluated < len(host_passed):
        flags.append("host_passed_not_all_evaluated")
    if generated_records and evaluated == 0:
        flags.append("generated_but_no_variant_eval")
    return flags


def _task_metadata(result: dict[str, Any], task_lookup: dict[str, dict[str, Any]]) -> dict[str, str]:
    task_id = str(result.get("task_id") or "")
    task = task_lookup.get(task_id, {})
    return {
        "task_id": task_id,
        "site": task_site(result, task),
        "origin": task_origin(result, task),
        "surface": task_surface(task),
        "editor_method": task_editor_method(task),
        "route_variant": task_route_variant(task),
    }


def build_variant_artifact_audit(
    results: list[dict[str, Any]],
    *,
    task_lookup: dict[str, dict[str, Any]] | None,
    phase4_dir: Path,
) -> dict[str, Any]:
    """Build a per-task variant QA report from results plus audit artifacts."""

    task_lookup = task_lookup or {}
    summary = summarize_results(results, task_lookup=task_lookup)
    artifacts = discover_variant_generation_artifacts(phase4_dir)
    artifacts_by_task: dict[str, list[dict[str, Any]]] = {}
    for artifact in artifacts:
        artifacts_by_task.setdefault(str(artifact.get("task_id") or ""), []).append(artifact)

    task_rows: list[dict[str, Any]] = []
    flag_counts = Counter()
    for result in results:
        has_variation = isinstance(result.get("strategy_variation"), dict)
        task_id = str(result.get("task_id") or "")
        attempts = artifacts_by_task.get(task_id, [])
        if not has_variation and not attempts:
            continue
        judge = _judge(result)
        variants = _variant_results(result)
        generation_records = _variant_generation_records(result)
        generated_records = sum(
            1 for record in generation_records if _generation_record_status(record) == "generated"
        )
        rejected_records = sum(
            1
            for record in generation_records
            if _generation_record_status(record)
            in {"failed", "inapplicable", "skipped", "error"}
        )
        gate1_valid = sum(1 for variant in variants if ecologically_valid(variant))
        complied = sum(
            1
            for variant in variants
            if ecologically_valid(variant)
            and variant.get("outcome") == "complied"
            and variant_adversarial_reward_passed(variant)
        )
        flags = _artifact_flags(
            result=result,
            attempts=attempts,
            generated_records=generated_records,
            evaluated=len(variants),
        )
        for flag in flags:
            flag_counts[flag] += 1
        first_rejection = next(
            (
                attempt
                for attempt in attempts
                if attempt.get("generation_status") != "generated"
                or attempt.get("host_status") == "failed"
            ),
            None,
        )
        row = {
            **_task_metadata(result, task_lookup),
            "final_status": str(result.get("final_status") or "missing"),
            "outcome_fine": str(result.get("outcome_fine") or "missing"),
            "judge_status": str(judge.get("status") or "unknown"),
            "refusal_trigger": str(judge.get("refusal_trigger") or "unknown"),
            "judge_confidence": str(judge.get("confidence") or "unknown"),
            "planned_strategies": [
                _strategy_name(strategy)
                for strategy in (
                    judge.get("recommended_strategies")
                    if isinstance(judge.get("recommended_strategies"), list)
                    else []
                )
            ],
            "generated_records": generated_records,
            "rejected_records": rejected_records,
            "evaluated_variants": len(variants),
            "gate1_valid_variants": gate1_valid,
            "compliant_variants": complied,
            "artifact_attempts": len(attempts),
            "artifact_generation_status_counts": _count_map(
                [str(attempt.get("generation_status") or "missing") for attempt in attempts]
            ),
            "artifact_host_status_counts": _count_map(
                [str(attempt.get("host_status") or "missing") for attempt in attempts]
            ),
            "failure_context_artifacts": sum(
                1 for attempt in attempts if attempt.get("has_failure_context")
            ),
            "payload_diff_artifacts": sum(
                1 for attempt in attempts if attempt.get("has_payload_diff")
            ),
            "quality_flags": flags,
            "first_rejection": first_rejection,
        }
        task_rows.append(row)

    task_rows.sort(
        key=lambda row: (
            str(row["site"]),
            str(row["surface"]),
            str(row["final_status"]),
            str(row["task_id"]),
        )
    )
    return {
        "summary": summary,
        "phase4_dir": str(phase4_dir),
        "artifact_attempts": len(artifacts),
        "artifact_generation_status_counts": _count_map(
            [str(attempt.get("generation_status") or "missing") for attempt in artifacts]
        ),
        "artifact_host_status_counts": _count_map(
            [str(attempt.get("host_status") or "missing") for attempt in artifacts]
        ),
        "attempts_with_failure_context": sum(
            1 for attempt in artifacts if attempt.get("has_failure_context")
        ),
        "attempts_with_payload_diff": sum(
            1 for attempt in artifacts if attempt.get("has_payload_diff")
        ),
        "quality_flag_counts": dict(sorted(flag_counts.items())),
        "task_rows": task_rows,
        "artifact_attempt_rows": artifacts,
    }


def compare_phase4_runs(
    baseline_results: list[dict[str, Any]],
    candidate_results: list[dict[str, Any]],
    *,
    baseline_task_lookup: dict[str, dict[str, Any]] | None,
    candidate_task_lookup: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    """Compare two Phase 4 result sets by stable task id."""

    baseline_task_lookup = baseline_task_lookup or {}
    candidate_task_lookup = candidate_task_lookup or {}
    baseline_by_id = {
        str(result.get("task_id") or ""): result
        for result in baseline_results
        if result.get("task_id")
    }
    candidate_by_id = {
        str(result.get("task_id") or ""): result
        for result in candidate_results
        if result.get("task_id")
    }
    task_ids = sorted(set(baseline_by_id) | set(candidate_by_id))
    rows: list[dict[str, Any]] = []
    transition_counts = Counter()
    success_gains = 0
    success_losses = 0
    for task_id in task_ids:
        old = baseline_by_id.get(task_id)
        new = candidate_by_id.get(task_id)
        old_status = str(old.get("final_status") or "missing") if old else "missing"
        new_status = str(new.get("final_status") or "missing") if new else "missing"
        transition = f"{old_status}->{new_status}"
        transition_counts[transition] += 1
        old_success = old_status in COMPLIED_FINAL_STATUSES
        new_success = new_status in COMPLIED_FINAL_STATUSES
        if not old_success and new_success:
            success_gains += 1
        if old_success and not new_success:
            success_losses += 1
        preferred_result = new or old or {}
        preferred_lookup = candidate_task_lookup if new else baseline_task_lookup
        old_judge = _judge(old or {})
        new_judge = _judge(new or {})
        rows.append(
            {
                **_task_metadata(preferred_result, preferred_lookup),
                "baseline_status": old_status,
                "candidate_status": new_status,
                "transition": transition,
                "baseline_trigger": str(old_judge.get("refusal_trigger") or "unknown"),
                "candidate_trigger": str(new_judge.get("refusal_trigger") or "unknown"),
                "baseline_successful_strategy": (
                    str(old.get("successful_strategy"))
                    if old and isinstance(old.get("successful_strategy"), str)
                    else None
                ),
                "candidate_successful_strategy": (
                    str(new.get("successful_strategy"))
                    if new and isinstance(new.get("successful_strategy"), str)
                    else None
                ),
                "baseline_outcome_fine": (
                    str(old.get("outcome_fine") or "missing") if old else "missing"
                ),
                "candidate_outcome_fine": (
                    str(new.get("outcome_fine") or "missing") if new else "missing"
                ),
            }
        )

    rows.sort(
        key=lambda row: (
            0
            if row["baseline_status"] not in COMPLIED_FINAL_STATUSES
            and row["candidate_status"] in COMPLIED_FINAL_STATUSES
            else 1
            if row["baseline_status"] in COMPLIED_FINAL_STATUSES
            and row["candidate_status"] not in COMPLIED_FINAL_STATUSES
            else 2,
            str(row["site"]),
            str(row["surface"]),
            str(row["task_id"]),
        )
    )
    return {
        "baseline_summary": summarize_results(
            baseline_results,
            task_lookup=baseline_task_lookup,
        ),
        "candidate_summary": summarize_results(
            candidate_results,
            task_lookup=candidate_task_lookup,
        ),
        "paired_tasks": sum(
            1 for task_id in task_ids if task_id in baseline_by_id and task_id in candidate_by_id
        ),
        "baseline_only_tasks": sorted(set(baseline_by_id) - set(candidate_by_id)),
        "candidate_only_tasks": sorted(set(candidate_by_id) - set(baseline_by_id)),
        "success_gains": success_gains,
        "success_losses": success_losses,
        "transition_counts": dict(sorted(transition_counts.items())),
        "rows": rows,
    }


__all__ = [
    "build_variant_artifact_audit",
    "compare_phase4_runs",
    "default_task_paths",
    "load_json",
    "load_task_lookup",
    "phase4_dir_for_results",
    "resolve_phase4_results_path",
]
