"""Human-readable Phase 4 variant trace exports.

This module is report-only. It reconciles already-written Phase 4 runtime
artifacts into a stable row model for JSON, CSV, and HTML reports. It does not
alter Phase 4 final statuses, rewards, PVPO gates, or admission.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from worldsim.phase_4.artifact_audit import (
    default_task_paths,
    load_json,
    load_task_lookup,
    phase4_dir_for_results,
    resolve_phase4_results_path,
)
from worldsim.phase_4.variant_trace_text import answer_label, compact_text


def build_variant_trace_export(
    path: Path,
    *,
    task_paths: Iterable[Path] = (),
    include: str = "all",
    payload_limit: int | None = None,
) -> dict[str, Any]:
    """Build a stable JSON-serializable trace table for one Phase 4 run."""

    results_path = resolve_phase4_results_path(path)
    raw_results = load_json(results_path)
    if not isinstance(raw_results, list):
        raise ValueError(f"{results_path} must contain a list of result objects")
    results = [item for item in raw_results if isinstance(item, dict)]
    phase4_dir = phase4_dir_for_results(results_path)
    task_lookup = load_task_lookup([*task_paths, *default_task_paths(results_path)])
    rows = [
        row
        for index, result in enumerate(results)
        if (row := _build_task_row(index, result, phase4_dir, task_lookup, payload_limit))
        is not None
    ]
    filtered = [_filter_row(row, include=include) for row in rows]
    filtered_rows = [row for row in filtered if row is not None]
    warnings = [
        {"task_id": row["task_id"], "warning": warning}
        for row in filtered_rows
        for warning in row.get("warnings", [])
    ]
    return {
        "schema_version": "phase4_variant_trace_export_v1",
        "results_path": str(results_path),
        "phase4_dir": str(phase4_dir),
        "include": include,
        "total_results": len(results),
        "row_count": len(filtered_rows),
        "warning_count": len(warnings),
        "warnings": warnings,
        "rows": filtered_rows,
    }


def _filter_row(row: dict[str, Any], *, include: str) -> dict[str, Any] | None:
    if include == "all":
        return row
    if include == "variant-entered":
        return row if row.get("variants") else None
    if include == "success-on-variant":
        return row if row.get("final_status") == "success_on_variant" else None
    raise ValueError("--include must be one of: all, variant-entered, success-on-variant")


def _build_task_row(
    result_index: int,
    result: dict[str, Any],
    phase4_dir: Path,
    task_lookup: dict[str, dict[str, Any]],
    payload_limit: int | None,
) -> dict[str, Any] | None:
    task_id = str(result.get("task_id") or "")
    if not task_id:
        return None
    task = task_lookup.get(task_id, {})
    initial_trace = _initial_trace_path(result, phase4_dir=phase4_dir, task_id=task_id)
    current_trace = _trace_path(
        result.get("current_trace") or result.get("trajectory_dir"),
        phase4_dir=phase4_dir,
        task_id=task_id,
    )
    initial_result = _load_result_json(initial_trace)
    checkpoint = _load_checkpoint(initial_trace)
    generation_attempts = _generation_attempts_by_index(initial_trace, payload_limit)
    variant_records = _strategy_records(result, checkpoint)
    variants = [
        _build_variant_row(
            task_id=task_id,
            record_index=index,
            strategy_record=strategy_record,
            attempt=_select_generation_attempt(
                generation_attempts.get(
                    _variant_join_index(strategy_record, fallback=index),
                    [],
                )
            ),
            phase4_dir=phase4_dir,
            result=result,
        )
        for index, strategy_record in enumerate(variant_records)
    ]
    warnings = _task_warnings(result, variants, checkpoint)
    variant_loop = _variant_loop_view(result, variants)
    return {
        "result_index": result_index,
        "task_id": task_id,
        "final_status": result.get("final_status"),
        "outcome": result.get("outcome"),
        "outcome_fine": result.get("outcome_fine"),
        "site": task.get("site") or result.get("site"),
        "surface": task.get("target_surface_id"),
        "editor_method": task.get("editor_method"),
        "instruction": task.get("instruction") or _instruction_from_attempts(generation_attempts),
        "initial": {
            "trace": str(initial_trace) if initial_trace else None,
            "outcome": initial_result.get("outcome") or result.get("initial_outcome"),
            "answer_label": answer_label(
                initial_result.get("final_result") or result.get("initial_final_result")
            ),
            "final_result": initial_result.get("final_result")
            or result.get("initial_final_result"),
        },
        "current": {
            "trace": str(current_trace) if current_trace else None,
            "answer_label": answer_label(
                result.get("primary_final_result") or result.get("current_final_result")
            ),
            "final_result": result.get("primary_final_result")
            or result.get("current_final_result"),
        },
        "generation": {
            "refusal_trigger": _nested_get(
                checkpoint,
                "judge_diagnosis",
                "refusal_trigger",
            ),
            "judge_status": _nested_get(checkpoint, "judge_diagnosis", "status"),
            "planned_strategies": [
                item.get("strategy")
                for item in _nested_get(
                    checkpoint,
                    "judge_diagnosis",
                    "recommended_strategies",
                    default=[],
                )
                if isinstance(item, dict) and isinstance(item.get("strategy"), str)
            ],
        },
        "variant_loop": variant_loop,
        "successful_variant_index": result.get("successful_variant_index"),
        "successful_strategy": result.get("successful_strategy"),
        "variants": variants,
        "warnings": warnings,
    }


def _build_variant_row(
    *,
    task_id: str,
    record_index: int,
    strategy_record: dict[str, Any],
    attempt: dict[str, Any] | None,
    phase4_dir: Path,
    result: dict[str, Any],
) -> dict[str, Any]:
    join_index = _variant_join_index(strategy_record, fallback=record_index)
    round_index = _int_or_none(strategy_record.get("round_index"))
    round_variant_index = _int_or_none(strategy_record.get("round_variant_index"))
    generation_status = str(strategy_record.get("generation_status") or "")
    host_rejected = generation_status in {"failed", "inapplicable", "skipped", "error"}
    variant_trace = None
    variant_result: dict[str, Any] = {}
    if not host_rejected:
        variant_trace = _variant_trace_path(task_id, join_index, phase4_dir, result)
        variant_result = _load_result_json(variant_trace)
    adversarial_passed = variant_result.get("adversarial_passed")
    worked = (
        bool(adversarial_passed)
        if adversarial_passed is not None
        else variant_result.get("outcome") == "complied"
    )
    selected_success = result.get("final_status") == "success_on_variant" and (
        _variant_matches_selected_success(
            result,
            index=join_index,
            variant_trace=variant_trace,
        )
    )
    delta = _payload_delta(attempt)
    return {
        "variant_index": join_index,
        "global_variant_index": join_index,
        "strategy_index": _int_or_none(strategy_record.get("index")),
        "row_index": record_index,
        "round_index": round_index,
        "round_variant_index": round_variant_index,
        "parent_global_variant_index": strategy_record.get("parent_global_variant_index"),
        "root_attempt_id": strategy_record.get("root_attempt_id"),
        "parent_attempt_id": strategy_record.get("parent_attempt_id"),
        "strategy": strategy_record.get("strategy"),
        "strategy_outcome": strategy_record.get("outcome"),
        "strategy_ecologically_valid": strategy_record.get("ecologically_valid"),
        "selected_success": selected_success,
        "generation": _generation_view(attempt, strategy_record),
        "delta": delta,
        "evaluation": {
            "trace": str(variant_trace) if variant_trace else None,
            "status": (
                "not_evaluated_host_rejected"
                if host_rejected and variant_trace is None
                else "evaluated"
                if variant_trace is not None
                else "missing"
            ),
            "outcome": variant_result.get("outcome"),
            "outcome_fine": variant_result.get("outcome_fine"),
            "answer_label": answer_label(variant_result.get("final_result")),
            "final_result": variant_result.get("final_result"),
            "worked": worked,
        },
        "warnings": _variant_warnings(
            join_index,
            attempt,
            variant_trace,
            variant_result,
            host_rejected=host_rejected,
        ),
    }


def _strategy_records(result: dict[str, Any], checkpoint: dict[str, Any]) -> list[dict[str, Any]]:
    raw_generation = checkpoint.get("variant_generation_records")
    if not isinstance(raw_generation, list):
        raw_rounds = checkpoint.get("variant_rounds") or checkpoint.get("adaptive_rounds")
        if isinstance(raw_rounds, list):
            raw_generation = [
                item
                for round_record in raw_rounds
                if isinstance(round_record, dict)
                for item in round_record.get("variant_generation_records", [])
                if isinstance(item, dict)
            ]
    if isinstance(raw_generation, list) and raw_generation:
        return [
            {
                "index": item.get("index"),
                "round_index": item.get("round_index"),
                "round_variant_index": item.get("round_variant_index"),
                "global_variant_index": item.get("global_variant_index"),
                "parent_global_variant_index": item.get("parent_global_variant_index"),
                "root_attempt_id": item.get("root_attempt_id"),
                "parent_attempt_id": item.get("parent_attempt_id"),
                "generation_status": item.get("status"),
                "generation_reason": item.get("reason"),
                "host_finalization_status": item.get("host_finalization_status"),
                "host_finalization_reason": item.get("host_finalization_reason"),
                "strategy": _nested_get(item, "strategy", "strategy")
                or _nested_get(item, "strategy")
                or "unknown",
            }
            for item in raw_generation
            if isinstance(item, dict)
        ]
    variation = result.get("strategy_variation")
    if isinstance(variation, dict):
        raw_generation = variation.get("variant_generation_records")
        if not isinstance(raw_generation, list):
            raw_rounds = variation.get("variant_rounds") or variation.get("adaptive_rounds")
            if isinstance(raw_rounds, list):
                raw_generation = [
                    item
                    for round_record in raw_rounds
                    if isinstance(round_record, dict)
                    for item in round_record.get("variant_generation_records", [])
                    if isinstance(item, dict)
                ]
        if isinstance(raw_generation, list) and raw_generation:
            return [
                {
                    "index": item.get("index"),
                    "round_index": item.get("round_index"),
                    "round_variant_index": item.get("round_variant_index"),
                    "global_variant_index": item.get("global_variant_index"),
                    "parent_global_variant_index": item.get("parent_global_variant_index"),
                    "root_attempt_id": item.get("root_attempt_id"),
                    "parent_attempt_id": item.get("parent_attempt_id"),
                    "generation_status": item.get("status"),
                    "generation_reason": item.get("reason"),
                    "host_finalization_status": item.get("host_finalization_status"),
                    "host_finalization_reason": item.get("host_finalization_reason"),
                    "strategy": _nested_get(item, "strategy", "strategy")
                    or _nested_get(item, "strategy")
                    or "unknown",
                }
                for item in raw_generation
                if isinstance(item, dict)
            ]
    raw = result.get("strategies_attempted")
    if isinstance(raw, list) and raw:
        return [item for item in raw if isinstance(item, dict)]
    return []


def _variant_loop_view(
    result: dict[str, Any],
    variants: list[dict[str, Any]],
) -> dict[str, Any]:
    """Explain whether the post-resistance strategy loop actually ran."""

    entered = bool(variants)
    final_status = str(result.get("final_status") or "unknown")
    initial_outcome = str(result.get("initial_outcome") or result.get("outcome") or "unknown")
    if entered:
        reason = "entered_after_initial_refusal_or_ignore"
    elif final_status == "complied":
        reason = "stopped_after_initial_compliance"
    elif final_status == "injection_not_encountered":
        reason = "stopped_before_strategy_variation_no_pvpo_encounter"
    elif final_status == "task_broke":
        reason = "stopped_before_strategy_variation_task_broke"
    elif final_status == "seed_preflight_mismatch":
        reason = "stopped_before_strategy_variation_seed_preflight_mismatch"
    elif final_status == "error":
        reason = "stopped_before_strategy_variation_error"
    else:
        reason = f"not_entered_final_status_{final_status}"
    return {
        "base_seeded_ipi_present": True,
        "post_resistance_variants_entered": entered,
        "reason": reason,
        "initial_outcome": initial_outcome,
    }


def _generation_attempts_by_index(
    initial_trace: Path | None,
    payload_limit: int | None,
) -> dict[int, list[dict[str, Any]]]:
    attempts: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if initial_trace is None:
        return attempts
    variant_root = initial_trace / "variant_generation"
    if not variant_root.exists():
        return attempts
    for summary_path in sorted(variant_root.glob("*/*/request_summary.json")):
        attempt_dir = summary_path.parent
        request = _load_json_dict(summary_path)
        strategy_index = request.get("strategy_index")
        if not isinstance(strategy_index, int):
            strategy_index = _index_from_generation_dir(attempt_dir.parent.name)
        if strategy_index is None:
            continue
        global_variant_index = request.get("global_variant_index")
        if isinstance(global_variant_index, int):
            strategy_index = global_variant_index
        host_validation = _load_json_dict(attempt_dir / "host_validation.json")
        contract_qa = _load_json_dict(attempt_dir / "contract_qa.json")
        payload_diff = _load_json_dict(attempt_dir / "payload_diff.json")
        if payload_limit is not None:
            payload_diff = {
                **payload_diff,
                "original_rendered_payload": compact_text(
                    payload_diff.get("original_rendered_payload"),
                    limit=payload_limit,
                ),
                "revised_rendered_payload": compact_text(
                    payload_diff.get("revised_rendered_payload"),
                    limit=payload_limit,
                ),
            }
        attempts[strategy_index].append(
            {
                "strategy_index": strategy_index,
                "global_variant_index": global_variant_index,
                "round_index": request.get("round_index"),
                "round_variant_index": request.get("round_variant_index"),
                "strategy": request.get("strategy"),
                "attempt": request.get("attempt") or attempt_dir.name,
                "request_summary": request,
                "host_validation": host_validation,
                "contract_qa": contract_qa,
                "payload_diff": payload_diff,
                "artifact_dir": str(attempt_dir),
            }
        )
    return attempts


def _select_generation_attempt(attempts: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not attempts:
        return None
    passed = [
        attempt
        for attempt in attempts
        if _nested_get(attempt, "request_summary", "status") == "generated"
        and _nested_get(attempt, "host_validation", "status") == "passed"
    ]
    if passed:
        retry_passed = [attempt for attempt in passed if attempt.get("attempt") == "host_retry"]
        return (retry_passed or passed)[-1]
    return attempts[-1]


def _generation_view(
    attempt: dict[str, Any] | None,
    strategy_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    strategy_record = strategy_record if isinstance(strategy_record, dict) else {}
    if attempt is None:
        return {
            "attempt": None,
            "status": strategy_record.get("generation_status") or "missing",
            "reason": strategy_record.get("generation_reason"),
            "host_status": strategy_record.get("host_finalization_status") or "missing",
            "host_reason": strategy_record.get("host_finalization_reason"),
            "contract_qa_status": "missing",
            "artifact_dir": None,
        }
    return {
        "attempt": attempt.get("attempt"),
        "status": _nested_get(attempt, "request_summary", "status"),
        "reason": _nested_get(attempt, "request_summary", "reason"),
        "host_status": _nested_get(attempt, "host_validation", "status"),
        "host_reason": _nested_get(attempt, "host_validation", "reason"),
        "contract_qa_status": _nested_get(attempt, "contract_qa", "status"),
        "contract_qa_failure_classes": _nested_get(
            attempt,
            "contract_qa",
            "failure_classes",
            default=[],
        ),
        "artifact_dir": attempt.get("artifact_dir"),
    }


def _payload_delta(attempt: dict[str, Any] | None) -> dict[str, Any]:
    payload_diff = attempt.get("payload_diff") if isinstance(attempt, dict) else None
    if not isinstance(payload_diff, dict):
        return {}
    original_chars = payload_diff.get("original_chars")
    revised_chars = payload_diff.get("revised_chars")
    char_delta = (
        revised_chars - original_chars
        if isinstance(original_chars, int) and isinstance(revised_chars, int)
        else None
    )
    return {
        "original_payload": payload_diff.get("original_rendered_payload"),
        "revised_payload": payload_diff.get("revised_rendered_payload"),
        "original_chars": original_chars,
        "revised_chars": revised_chars,
        "char_delta": char_delta,
        "changed_seed": payload_diff.get("changed_seed"),
        "meaningful_token_change": payload_diff.get("meaningful_token_change"),
        "attack_witness_offset": payload_diff.get("attack_witness_offset"),
        "max_attack_witness_offset": payload_diff.get("max_attack_witness_offset"),
        "final_attack_witness_offset": payload_diff.get("final_attack_witness_offset"),
        "payload_text_resynchronized": payload_diff.get("payload_text_resynchronized"),
        "diff_excerpt": _diff_excerpt(
            payload_diff.get("original_rendered_payload"),
            payload_diff.get("revised_rendered_payload"),
        ),
    }


def _diff_excerpt(original: Any, revised: Any, *, limit: int = 420) -> str | None:
    if not isinstance(original, str) or not isinstance(revised, str):
        return None
    original_words = original.split()
    revised_words = revised.split()
    prefix = 0
    while (
        prefix < len(original_words)
        and prefix < len(revised_words)
        and original_words[prefix] == revised_words[prefix]
    ):
        prefix += 1
    suffix = 0
    while (
        suffix < len(original_words) - prefix
        and suffix < len(revised_words) - prefix
        and original_words[-(suffix + 1)] == revised_words[-(suffix + 1)]
    ):
        suffix += 1
    old_mid = " ".join(original_words[prefix : len(original_words) - suffix])
    new_mid = " ".join(revised_words[prefix : len(revised_words) - suffix])
    excerpt = f"- {old_mid}\n+ {new_mid}".strip()
    return compact_text(excerpt, limit=limit)


def _variant_trace_path(
    task_id: str,
    index: int,
    phase4_dir: Path,
    result: dict[str, Any],
) -> Path | None:
    successful_trace = _trace_path(
        result.get("successful_variant_trace"),
        phase4_dir=phase4_dir,
        task_id=f"{task_id}_variant_{index}",
    )
    selected_index = _selected_success_index(result)
    if (
        selected_index == index
        and successful_trace
        and _trace_path_matches_variant_index(successful_trace, task_id=task_id, index=index)
    ):
        return successful_trace
    if (
        selected_index is None
        and successful_trace
        and _trace_path_matches_variant_index(successful_trace, task_id=task_id, index=index)
    ):
        return successful_trace
    initial_trace = _initial_trace_path(result, phase4_dir=phase4_dir, task_id=task_id)
    if initial_trace is not None:
        candidate = initial_trace.parent / f"{task_id}_variant_{index}"
        if candidate.exists():
            return candidate
    matches = sorted(phase4_dir.glob(f"*/*{task_id}_variant_{index}"))
    return matches[0] if matches else None


def _selected_success_index(result: dict[str, Any]) -> int | None:
    global_index = _int_or_none(result.get("successful_variant_global_index"))
    legacy_index = _int_or_none(result.get("successful_variant_index"))
    if global_index is not None:
        return global_index
    return legacy_index


def _variant_matches_selected_success(
    result: dict[str, Any],
    *,
    index: int,
    variant_trace: Path | None,
) -> bool:
    selected_global = _int_or_none(result.get("successful_variant_global_index"))
    selected_legacy = _int_or_none(result.get("successful_variant_index"))
    selected_trace = _path_or_none(result.get("successful_variant_trace"))
    if selected_global is not None:
        if selected_global != index:
            return False
        if selected_legacy is not None and selected_legacy != index:
            return False
        if selected_trace is not None:
            return variant_trace is not None and _same_path(selected_trace, variant_trace)
        return True
    if selected_legacy is not None:
        if selected_legacy != index:
            return False
        if selected_trace is not None:
            return variant_trace is not None and _same_path(selected_trace, variant_trace)
        return True
    if selected_trace is None or variant_trace is None:
        return False
    return _same_path(selected_trace, variant_trace)


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return left == right


def _trace_path_matches_variant_index(path: Path, *, task_id: str, index: int) -> bool:
    return path.name == f"{task_id}_variant_{index}"


def _initial_trace_path(
    result: dict[str, Any],
    *,
    phase4_dir: Path,
    task_id: str,
) -> Path | None:
    return _trace_path(
        result.get("initial_trace") or result.get("trajectory_dir"),
        phase4_dir=phase4_dir,
        task_id=task_id,
    )


def _trace_path(raw: Any, *, phase4_dir: Path, task_id: str) -> Path | None:
    path = _path_or_none(raw)
    if path is not None and path.exists():
        return path
    matches = sorted(phase4_dir.glob(f"*/{task_id}"))
    return matches[0] if matches else path


def _task_warnings(
    result: dict[str, Any],
    variants: list[dict[str, Any]],
    checkpoint: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    if result.get("strategies_attempted") and not checkpoint:
        warnings.append("missing_strategy_variation_checkpoint")
    if result.get("final_status") == "success_on_variant" and not any(
        variant.get("selected_success") for variant in variants
    ):
        warnings.append("missing_selected_success_variant")
    if _selected_success_metadata_conflict(result, variants):
        warnings.append("conflicting_selected_success_metadata")
    for variant in variants:
        for warning in variant.get("warnings", []):
            warnings.append(f"variant_{variant['variant_index']}:{warning}")
    return warnings


def _selected_success_metadata_conflict(
    result: dict[str, Any],
    variants: list[dict[str, Any]],
) -> bool:
    selected_global = _int_or_none(result.get("successful_variant_global_index"))
    selected_legacy = _int_or_none(result.get("successful_variant_index"))
    selected_trace = _path_or_none(result.get("successful_variant_trace"))
    if selected_global is not None and selected_legacy is not None and selected_global != selected_legacy:
        return True
    selected_index = selected_global if selected_global is not None else selected_legacy
    if selected_index is None or selected_trace is None:
        return False
    for variant in variants:
        if variant.get("global_variant_index") != selected_index:
            continue
        trace = _path_or_none(_nested_get(variant, "evaluation", "trace"))
        return trace is not None and not _same_path(selected_trace, trace)
    return False


def _variant_warnings(
    index: int,
    attempt: dict[str, Any] | None,
    variant_trace: Path | None,
    variant_result: dict[str, Any],
    *,
    host_rejected: bool = False,
) -> list[str]:
    warnings: list[str] = []
    if attempt is None:
        warnings.append("missing_generation_attempt")
    else:
        if not attempt.get("payload_diff"):
            warnings.append("missing_payload_diff")
        if not attempt.get("contract_qa"):
            warnings.append("missing_contract_qa")
        dir_index = _index_from_generation_dir(
            Path(str(attempt.get("artifact_dir", ""))).parent.name
        )
        if dir_index is not None and dir_index != index:
            warnings.append("strategy_index_mismatch")
    if host_rejected:
        return warnings
    if variant_trace is None:
        warnings.append("missing_variant_trace")
    elif not variant_result:
        warnings.append("missing_variant_result")
    return warnings


def _instruction_from_attempts(attempts: dict[int, list[dict[str, Any]]]) -> str | None:
    for per_index in attempts.values():
        for attempt in per_index:
            prompt_path = Path(str(attempt.get("artifact_dir"))) / "prompt_input_redacted.json"
            prompt = _load_json_dict(prompt_path)
            task_json = prompt.get("task_json")
            if isinstance(task_json, dict) and isinstance(task_json.get("instruction"), str):
                return task_json["instruction"]
    return None


def _load_checkpoint(initial_trace: Path | None) -> dict[str, Any]:
    if initial_trace is None:
        return {}
    return _load_json_dict(initial_trace / "strategy_variation_checkpoint.json")


def _load_result_json(trace: Path | None) -> dict[str, Any]:
    if trace is None:
        return {}
    return _load_json_dict(trace / "result.json")


def _load_json_dict(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _path_or_none(value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return Path(value)


def _nested_get(payload: Any, *keys: str, default: Any = None) -> Any:
    current = payload
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def _index_from_generation_dir(name: str) -> int | None:
    prefix = name.split("_", 1)[0]
    if not prefix.isdigit():
        return None
    return int(prefix)


def _variant_join_index(strategy_record: dict[str, Any], *, fallback: int) -> int:
    for key in ("global_variant_index", "variant_index", "index"):
        value = _int_or_none(strategy_record.get(key))
        if value is not None:
            return value
    return fallback


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return None


__all__ = [
    "answer_label",
    "build_variant_trace_export",
    "compact_text",
]
