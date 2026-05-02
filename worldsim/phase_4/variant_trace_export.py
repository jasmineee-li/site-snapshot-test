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
    variants = [
        _build_variant_row(
            task_id=task_id,
            index=index,
            strategy_record=strategy_record,
            attempt=_select_generation_attempt(generation_attempts.get(index, [])),
            phase4_dir=phase4_dir,
            result=result,
        )
        for index, strategy_record in enumerate(_strategy_records(result, checkpoint))
    ]
    warnings = _task_warnings(result, variants, checkpoint)
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
        "successful_variant_index": result.get("successful_variant_index"),
        "successful_strategy": result.get("successful_strategy"),
        "variants": variants,
        "warnings": warnings,
    }


def _build_variant_row(
    *,
    task_id: str,
    index: int,
    strategy_record: dict[str, Any],
    attempt: dict[str, Any] | None,
    phase4_dir: Path,
    result: dict[str, Any],
) -> dict[str, Any]:
    variant_trace = _variant_trace_path(task_id, index, phase4_dir, result)
    variant_result = _load_result_json(variant_trace)
    adversarial_passed = variant_result.get("adversarial_passed")
    worked = (
        bool(adversarial_passed)
        if adversarial_passed is not None
        else variant_result.get("outcome") == "complied"
    )
    selected_success = (
        result.get("final_status") == "success_on_variant"
        and result.get("successful_variant_index") == index
    )
    delta = _payload_delta(attempt)
    return {
        "variant_index": index,
        "strategy_index": index,
        "strategy": strategy_record.get("strategy"),
        "strategy_outcome": strategy_record.get("outcome"),
        "strategy_ecologically_valid": strategy_record.get("ecologically_valid"),
        "selected_success": selected_success,
        "generation": _generation_view(attempt),
        "delta": delta,
        "evaluation": {
            "trace": str(variant_trace) if variant_trace else None,
            "outcome": variant_result.get("outcome"),
            "outcome_fine": variant_result.get("outcome_fine"),
            "answer_label": answer_label(variant_result.get("final_result")),
            "final_result": variant_result.get("final_result"),
            "worked": worked,
        },
        "warnings": _variant_warnings(index, attempt, variant_trace, variant_result),
    }


def _strategy_records(result: dict[str, Any], checkpoint: dict[str, Any]) -> list[dict[str, Any]]:
    raw = result.get("strategies_attempted")
    if isinstance(raw, list) and raw:
        return [item for item in raw if isinstance(item, dict)]
    raw_generation = checkpoint.get("variant_generation_records")
    if isinstance(raw_generation, list):
        return [
            {
                "strategy": _nested_get(item, "strategy", "strategy")
                or _nested_get(item, "strategy")
                or "unknown",
            }
            for item in raw_generation
            if isinstance(item, dict)
        ]
    return []


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


def _generation_view(attempt: dict[str, Any] | None) -> dict[str, Any]:
    if attempt is None:
        return {
            "attempt": None,
            "status": "missing",
            "host_status": "missing",
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
    if result.get("successful_variant_index") == index and successful_trace:
        return successful_trace
    initial_trace = _initial_trace_path(result, phase4_dir=phase4_dir, task_id=task_id)
    if initial_trace is not None:
        candidate = initial_trace.parent / f"{task_id}_variant_{index}"
        if candidate.exists():
            return candidate
    matches = sorted(phase4_dir.glob(f"*/*{task_id}_variant_{index}"))
    return matches[0] if matches else None


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
    for variant in variants:
        for warning in variant.get("warnings", []):
            warnings.append(f"variant_{variant['variant_index']}:{warning}")
    return warnings


def _variant_warnings(
    index: int,
    attempt: dict[str, Any] | None,
    variant_trace: Path | None,
    variant_result: dict[str, Any],
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


__all__ = [
    "answer_label",
    "build_variant_trace_export",
    "compact_text",
]
