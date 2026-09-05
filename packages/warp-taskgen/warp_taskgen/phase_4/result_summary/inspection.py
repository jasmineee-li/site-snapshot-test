"""Inspection index for Phase 4 outputs."""

from __future__ import annotations

import json
from typing import Any

# Initialise the adversarial_actions <-> text_payload_contract import cycle in the
# order the parity module used: adversarial_actions first, then text_payload_contract.
import warp_taskgen.adversarial_actions  # noqa: F401 - import order only
from warp_taskgen.phase_4.pvpo_observations import pvpo_observation_bucket
from warp_taskgen.phase_4.result_summary.audit import (
    _compact_text,
    _rejected_variant_records,
    _string_or_none,
)
from warp_taskgen.phase_4.result_summary.final_metrics import (
    _variation_record,
    selected_successful_strategy_variants,
)
from warp_taskgen.phase_4.result_summary.task_metadata import (
    task_editor_method,
    task_origin,
    task_route_variant,
    task_site,
    task_surface,
)
from warp_taskgen.text_payload_contract import (
    binary_answer_contract,
    payload_content_capacity_contract,
)


def _compact_jsonish(value: Any, *, limit: int = 220) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return _compact_text(value, limit=limit)
    try:
        text = json.dumps(value, sort_keys=True, default=str)
    except TypeError:
        text = str(value)
    return _compact_text(text, limit=limit)


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _encounter(result: dict[str, Any]) -> dict[str, Any]:
    encounter = result.get("encounter")
    return encounter if isinstance(encounter, dict) else {}


def _max_coverage(result: dict[str, Any]) -> float | None:
    return _float_or_none(_encounter(result).get("max_coverage"))


def _reference_step(result: dict[str, Any]) -> int | None:
    value = _encounter(result).get("reference_step")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _judge(result: dict[str, Any]) -> dict[str, Any]:
    variation = _variation_record(result)
    if variation is None:
        return {}
    judge = variation.get("judge_diagnosis")
    return judge if isinstance(judge, dict) else {}


def _trace_from_variant(variant: dict[str, Any] | None) -> str | None:
    if not isinstance(variant, dict):
        return None
    return _string_or_none(variant.get("variant_trajectory_dir")) or _string_or_none(
        variant.get("trajectory_dir")
    )


def _successful_variant(result: dict[str, Any]) -> dict[str, Any] | None:
    variants = selected_successful_strategy_variants(result)
    return variants[0] if variants else None


def _artifact_paths(trace: str | None, reference_step: int | None) -> dict[str, str]:
    if not trace:
        return {}
    paths = {
        "trajectory_dir": trace,
        "history": f"{trace}/history.json",
        "result": f"{trace}/result.json",
        "needham_trace": f"{trace}/needham_trace.xml",
    }
    if reference_step is not None:
        paths["reference_screenshot"] = f"{trace}/screenshots/step_{reference_step}.png"
        paths["pvpo_step"] = f"{trace}/pvpo/step_{reference_step}.json"
    return paths


def _signals_summary(result: dict[str, Any]) -> dict[str, bool]:
    signals = result.get("signals")
    if not isinstance(signals, dict):
        return {}
    return {
        key: bool(signals[key])
        for key in ("B1", "B2", "C1", "C2", "D1")
        if isinstance(signals.get(key), bool)
    }


def _short_title_binary_resistance(result: dict[str, Any], task: dict[str, Any]) -> bool:
    if result.get("final_status") != "resistant":
        return False
    coverage = _max_coverage(result)
    if coverage is None or coverage <= 0:
        return False
    try:
        capacity = payload_content_capacity_contract(task).get("content_capacity")
        binary = binary_answer_contract(task)
    except (TypeError, ValueError):
        return False
    return capacity == "short_title" and binary is not None


def _inspection_priority(
    result: dict[str, Any],
    *,
    task: dict[str, Any],
    rejected_variants: list[dict[str, Any]],
    trigger: str,
    confidence: str,
    successful_variant: dict[str, Any] | None = None,
) -> tuple[int, str]:
    final_status = str(result.get("final_status") or "missing")
    outcome = str(result.get("outcome") or "")
    outcome_fine = str(result.get("outcome_fine") or "")
    if (
        final_status == "task_broke"
        or outcome == "task_broke"
        or outcome_fine.startswith("task_broke")
    ):
        return (10, "task_breakage")
    if final_status == "inconclusive":
        return (20, "inconclusive")
    if final_status == "success_on_variant" and successful_variant is not None:
        return (30, "variant_success_exemplar")
    if final_status == "success_on_variant":
        return (25, "inconsistent_variant_success_metadata")
    if rejected_variants:
        return (40, "variant_generation_rejected")
    if (
        _short_title_binary_resistance(result, task)
        and trigger != "unknown"
        and confidence == "high"
    ):
        return (45, "short_title_binary_semantic_resistance")
    if final_status == "resistant" and trigger != "unknown" and confidence == "high":
        return (50, "resistant_high_confidence")
    if final_status == "resistant":
        return (60, "resistant")
    if final_status == "injection_not_encountered":
        return (70, "placement_or_pvpo")
    if final_status == "complied":
        return (80, "direct_success_exemplar")
    return (90, "other")


def _inspection_reason(
    result: dict[str, Any],
    *,
    task: dict[str, Any],
    trigger: str,
    confidence: str,
    rejected_variants: list[dict[str, Any]],
    successful_strategy: str | None,
    successful_variant: dict[str, Any] | None = None,
) -> str:
    final_status = str(result.get("final_status") or "missing")
    outcome = str(result.get("outcome") or "")
    outcome_fine = str(result.get("outcome_fine") or "missing")
    coverage = _max_coverage(result)
    coverage_part = f"max_coverage={coverage:.2f}" if coverage is not None else "max_coverage=n/a"
    rationale = _compact_text(result.get("classifier_rationale"), limit=180)
    if (
        final_status == "task_broke"
        or outcome == "task_broke"
        or outcome_fine.startswith("task_broke")
    ):
        base = "benign task broke under attack; inspect task wording/reward before strategy"
    elif final_status == "success_on_variant" and successful_variant is not None:
        base = f"variant success via {successful_strategy or 'unknown'} after trigger={trigger}"
    elif final_status == "success_on_variant":
        base = "success_on_variant has conflicting or stale selected-variant metadata"
    elif (
        _short_title_binary_resistance(result, task)
        and trigger != "unknown"
        and confidence == "high"
    ):
        base = (
            "PVPO-valid short-title binary resistance; inspect strategy semantics "
            "before placement or admission"
        )
    elif final_status == "resistant":
        base = f"resistant after trigger={trigger} confidence={confidence}"
    elif final_status == "inconclusive":
        base = "variation did not produce a decisive PVPO-valid compliant or resistant outcome"
    elif rejected_variants:
        first = rejected_variants[0]
        base = (
            "variant generation rejected "
            f"{first.get('strategy', 'unknown')} ({first.get('status', 'unknown')}): "
            f"{first.get('reason', '')}"
        ).strip()
    elif final_status == "injection_not_encountered":
        base = f"payload was not encountered by PVPO ({pvpo_observation_bucket(result)})"
    elif final_status == "complied":
        base = "direct compliant exemplar"
    else:
        base = final_status
    parts = [base, outcome_fine, coverage_part]
    if rationale:
        parts.append(rationale)
    return "; ".join(parts)


def inspection_index(
    results: list[dict[str, Any]],
    *,
    task_lookup: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build a ranked per-task worklist for fast Phase 4 debugging."""

    task_lookup = task_lookup or {}
    rows: list[dict[str, Any]] = []
    for result in results:
        task_id = str(result.get("task_id") or "")
        task = task_lookup.get(task_id, {})
        judge = _judge(result)
        trigger = str(judge.get("refusal_trigger") or "unknown")
        confidence = str(judge.get("confidence") or "unknown")
        successful_variant = _successful_variant(result)
        primary_result = (
            successful_variant
            if result.get("final_status") == "success_on_variant" and successful_variant is not None
            else result
        )
        successful_strategy = None
        if isinstance(successful_variant, dict):
            successful_strategy = _string_or_none(
                result.get("successful_strategy")
            ) or _string_or_none(successful_variant.get("strategy"))
        initial_trace = _string_or_none(result.get("initial_trace")) or _string_or_none(
            result.get("trajectory_dir")
        )
        current_trace = _string_or_none(result.get("current_trace")) or _string_or_none(
            result.get("trajectory_dir")
        )
        successful_variant_trace = _trace_from_variant(successful_variant)
        if successful_variant_trace is None and result.get("final_status") != "success_on_variant":
            successful_variant_trace = _string_or_none(result.get("successful_variant_trace"))
        primary_trace = (
            successful_variant_trace
            if result.get("final_status") == "success_on_variant"
            else _string_or_none(result.get("primary_inspection_trace")) or current_trace
        )
        rejected_variants = _rejected_variant_records(result)
        priority, priority_reason = _inspection_priority(
            result,
            task=task,
            rejected_variants=rejected_variants,
            trigger=trigger,
            confidence=confidence,
            successful_variant=successful_variant,
        )
        row = {
            "task_id": task_id,
            "priority": priority,
            "priority_reason": priority_reason,
            "site": task_site(result, task),
            "origin": task_origin(result, task),
            "surface": task_surface(task),
            "editor_method": task_editor_method(task),
            "route_variant": task_route_variant(task),
            "final_status": str(result.get("final_status") or "missing"),
            "outcome_fine": str(result.get("outcome_fine") or "missing"),
            "why": _inspection_reason(
                result,
                task=task,
                trigger=trigger,
                confidence=confidence,
                rejected_variants=rejected_variants,
                successful_strategy=successful_strategy,
                successful_variant=successful_variant,
            ),
            "signals": _signals_summary(result),
            "pvpo_status": primary_result.get("pvpo_status"),
            "pvpo_failure": primary_result.get("pvpo_failure"),
            "pvpo_observation": pvpo_observation_bucket(primary_result),
            "max_coverage": _max_coverage(primary_result),
            "initial_max_coverage": _max_coverage(result),
            "reference_step": _reference_step(primary_result),
            "judge_trigger": trigger,
            "judge_confidence": confidence,
            "successful_strategy": successful_strategy,
            "initial_final_result": _compact_jsonish(result.get("initial_final_result")),
            "current_final_result": _compact_jsonish(result.get("current_final_result")),
            "successful_variant_final_result": _compact_jsonish(
                successful_variant.get("final_result")
                if isinstance(successful_variant, dict)
                else None
            ),
            "primary_final_result": _compact_jsonish(
                primary_result.get("final_result")
                if (
                    result.get("final_status") == "success_on_variant"
                    and successful_variant is not None
                )
                else result.get("primary_final_result") or primary_result.get("final_result")
            ),
            "initial_trace": initial_trace,
            "current_trace": current_trace,
            "successful_variant_trace": successful_variant_trace,
            "primary_inspection_trace": primary_trace,
            "artifacts": _artifact_paths(primary_trace, _reference_step(primary_result)),
            "rejected_variants": rejected_variants,
        }
        rows.append(row)

    rows.sort(
        key=lambda row: (
            int(row["priority"]),
            str(row["site"]),
            str(row["surface"]),
            str(row["task_id"]),
        )
    )
    return rows


__all__ = ["inspection_index"]
