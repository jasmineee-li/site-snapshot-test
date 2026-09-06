"""Budget accounting, selection, and stop reasons for the Phase 4 iterator.

These builders own the iterator's adaptive budget report, per-iteration
progress counts, TP transition direction, the selection of the best iteration,
and the checkpoint-to-result projection the process pool salvages.
"""

from __future__ import annotations

from typing import Any

from warp_taskgen.phase_4.eval_awareness_cue_diagnosis import (
    _ecologically_valid,
    _tp_requires_iteration,
)
from warp_taskgen.phase_4.options import (
    normalize_eval_awareness_max_iterations as _normalize_eval_awareness_max_iterations,
)
from warp_taskgen.phase_4.resume import _PHASE_4_EVAL_AWARENESS_ITERATOR_VERSION
from warp_taskgen.phase_4.variant_accounting import semantic_variant_accounting

_ITERATOR_STRATEGY = {
    "strategy": "eval_awareness_iterator",
    "rationale": "sequential TP-aware payload rewrite",
}

_STOP_REWRITE_LIMIT_REACHED = "rewrite_limit_reached"

_STOP_TP_REGRESSION = "tp_regression"


def _iterator_budget_report(
    *,
    max_iterations: int,
    iteration_records: list[dict[str, Any]],
    stop_reason: str,
) -> dict[str, Any]:
    results = [
        item.get("result") for item in iteration_records if isinstance(item.get("result"), dict)
    ]
    consumed = [item for item in iteration_records if _iteration_consumes_budget(item)]
    rejected = [
        item
        for item in iteration_records
        if item.get("status")
        in {
            "rewrite_failed",
            "rejected",
            _STOP_TP_REGRESSION,
            "task_broke",
            "lost_pvpo_encounter",
        }
    ]
    return {
        "algorithm": "eval-awareness-iterator",
        "version": _PHASE_4_EVAL_AWARENESS_ITERATOR_VERSION,
        "max_rewrites": max_iterations,
        "rewrites_attempted": len(consumed),
        "rejected_rewrites": len(rejected),
        "browser_evaluated": len(results),
        "pvpo_valid": sum(1 for result in results if _ecologically_valid(result)),
        "complied": sum(
            1
            for result in results
            if _ecologically_valid(result) and result.get("outcome") == "complied"
        ),
        "tp_real": sum(
            1 for result in results if result.get("transcript_purpose_classification") == "Real"
        ),
        "tp_regressed": sum(
            1 for item in iteration_records if _tp_transition_direction(item) == "regressed"
        ),
        "remaining_budget": max(0, max_iterations - len(consumed)),
        "stop_reason": stop_reason,
    }


def _iteration_progress_counts(iteration_records: list[dict[str, Any]]) -> dict[str, int]:
    consumed = [item for item in iteration_records if _iteration_consumes_budget(item)]
    variant_results = [
        item["result"] for item in iteration_records if isinstance(item.get("result"), dict)
    ]
    generation_errors = [
        item.get("generation_error")
        if isinstance(item.get("generation_error"), dict)
        else {"failure_class": item.get("status") or "unknown"}
        for item in iteration_records
        if item.get("status")
        in {
            "rewrite_failed",
            "rejected",
            _STOP_TP_REGRESSION,
            "task_broke",
            "lost_pvpo_encounter",
        }
    ]
    return {
        "generation_attempted": len(consumed),
        "generation_generated": sum(
            1
            for item in iteration_records
            if isinstance(item.get("rewrite"), dict)
            and item.get("status") not in {"rewrite_failed", "rejected"}
        ),
        "generation_failed": sum(
            1
            for item in iteration_records
            if item.get("status")
            in {
                "rewrite_failed",
                "rejected",
                _STOP_TP_REGRESSION,
                "task_broke",
                "lost_pvpo_encounter",
            }
        ),
        "evaluated": len(
            [item for item in iteration_records if isinstance(item.get("result"), dict)]
        ),
        "pvpo_valid": sum(
            1
            for item in iteration_records
            if isinstance(item.get("result"), dict) and _ecologically_valid(item["result"])
        ),
        "complied": sum(
            1
            for item in iteration_records
            if isinstance(item.get("result"), dict)
            and _ecologically_valid(item["result"])
            and item["result"].get("outcome") == "complied"
        ),
        "tp_regressed": sum(
            1 for item in iteration_records if _tp_transition_direction(item) == "regressed"
        ),
        **semantic_variant_accounting(
            variant_results=variant_results,
            generation_errors=generation_errors,
        ),
    }


def _iteration_consumes_budget(record: dict[str, Any]) -> bool:
    if isinstance(record.get("result"), dict):
        return True
    if record.get("status") in {
        "rewrite_failed",
        "rejected",
        _STOP_TP_REGRESSION,
        "task_broke",
        "lost_pvpo_encounter",
    }:
        return True
    if isinstance(record.get("rewrite"), dict):
        return True
    return False


def _iteration_is_terminal(record: dict[str, Any]) -> bool:
    if isinstance(record.get("result"), dict):
        return True
    return record.get("status") in {"rewrite_failed", "rejected", _STOP_TP_REGRESSION}


def _tp_transition_direction(record: dict[str, Any]) -> str | None:
    transition = record.get("tp_transition")
    if isinstance(transition, dict):
        direction = transition.get("direction")
        return direction if isinstance(direction, str) else None
    result = record.get("result")
    if isinstance(result, dict):
        transition = result.get("transcript_purpose_transition")
        if isinstance(transition, dict):
            direction = transition.get("direction")
            return direction if isinstance(direction, str) else None
    return None


def _best_iterator_result(
    baseline: dict[str, Any],
    iteration_records: list[dict[str, Any]],
) -> tuple[dict[str, Any], int, str]:
    candidates: list[tuple[int, dict[str, Any], str]] = [(0, baseline, "baseline")]
    for record in iteration_records:
        if _tp_transition_direction(record) == "regressed":
            continue
        result = record.get("result")
        if isinstance(result, dict):
            candidates.append((int(record.get("iteration", 0) or 0), result, "iteration"))

    def score(item: tuple[int, dict[str, Any], str]) -> tuple[int, int, int, int]:
        iteration, result, _kind = item
        pvpo_valid = 1 if _ecologically_valid(result) else 0
        complied = 1 if pvpo_valid and result.get("outcome") == "complied" else 0
        tp_real = 1 if result.get("transcript_purpose_classification") == "Real" else 0
        return (complied and tp_real, complied, tp_real and pvpo_valid, pvpo_valid, iteration)

    best_iteration, best_result, best_kind = max(candidates, key=score)
    if best_kind == "baseline":
        reason = "baseline"
    elif (
        best_result.get("outcome") == "complied"
        and best_result.get("transcript_purpose_classification") == "Real"
    ):
        reason = "tp_real_and_complied"
    elif best_result.get("outcome") == "complied":
        reason = "complied"
    elif best_result.get("transcript_purpose_classification") == "Real":
        reason = "tp_real"
    else:
        reason = "latest_valid_attempt"
    return best_result, best_iteration, reason


def build_eval_awareness_iterator_result_from_checkpoint(
    *,
    initial_result: dict[str, Any],
    checkpoint: dict[str, Any],
    max_iterations: int | None = None,
    stop_reason_override: str | None = None,
) -> dict[str, Any] | None:
    """Build the iterator result envelope from a persisted checkpoint.

    This is used by the normal iterator return path and by process-pool salvage
    when an outer worker timeout fires after some variants have already been
    evaluated. It intentionally preserves only completed iteration records as
    variant results; an in-flight variant without a result remains diagnostic
    metadata, not a scored browser evaluation.
    """

    if not isinstance(checkpoint, dict):
        return None
    max_rewrites = _normalize_eval_awareness_max_iterations(
        max_iterations or checkpoint.get("max_iterations")
    )
    iteration_records = [
        item for item in checkpoint.get("iterations", []) if isinstance(item, dict)
    ]
    stop_reason = str(
        stop_reason_override
        or checkpoint.get("stop_reason")
        or (
            _STOP_REWRITE_LIMIT_REACHED
            if _tp_requires_iteration(checkpoint.get("current_result") or initial_result)
            else "tp_real"
        )
    )
    variant_results = [
        item["result"] for item in iteration_records if isinstance(item.get("result"), dict)
    ]
    generation_errors = [
        {
            "iteration": item.get("iteration"),
            **(
                item.get("generation_error")
                if isinstance(item.get("generation_error"), dict)
                else {"reason": item.get("status")}
            ),
        }
        for item in iteration_records
        if item.get("status")
        in {
            "rewrite_failed",
            "rejected",
            _STOP_TP_REGRESSION,
            "task_broke",
            "lost_pvpo_encounter",
        }
    ]
    if stop_reason_override:
        generation_errors.append(
            {
                "failure_class": stop_reason_override,
                "reason": "process-pool worker timed out after completed iterator variants",
            }
        )
    best_result, selected_iteration, selection_reason = _best_iterator_result(
        initial_result,
        iteration_records,
    )
    if selected_iteration == 0 and not variant_results and stop_reason == "tp_real":
        status = "tp_real_baseline"
    elif (
        selected_iteration == 0
        and not variant_results
        and stop_reason == _STOP_REWRITE_LIMIT_REACHED
    ):
        status = "resistant"
    elif stop_reason in {
        "rewrite_failed",
        "rewrite_rejected",
        "rewrite_inapplicable_irreconcilable_contract",
        "rewrite_inapplicable_trajectory_only",
        "rewrite_inapplicable_insufficient_causal_evidence",
        _STOP_TP_REGRESSION,
    }:
        status = "rewrite_failed"
    else:
        status = "iterated"
    budget = _iterator_budget_report(
        max_iterations=max_rewrites,
        iteration_records=iteration_records,
        stop_reason=stop_reason,
    )
    return {
        "status": status,
        "algorithm": "eval-awareness-iterator",
        "version": _PHASE_4_EVAL_AWARENESS_ITERATOR_VERSION,
        "attempts": [initial_result],
        "baseline_attempt": checkpoint.get("baseline_attempt"),
        "iterations": iteration_records,
        "variant_results": variant_results,
        "variant_rounds": [
            {
                "round_index": item.get("iteration"),
                "round_kind": "eval_awareness_iteration",
                "planned_strategies": [_ITERATOR_STRATEGY],
                "variant_generation_records": [
                    {
                        "index": item.get("iteration"),
                        "global_variant_index": item.get("iteration"),
                        "round_index": item.get("iteration"),
                        "round_kind": "eval_awareness_iteration",
                        "round_variant_index": 0,
                        "strategy": _ITERATOR_STRATEGY,
                        "variant": item.get("finalized_task"),
                        "status": item.get("status"),
                        "cue_diagnosis": item.get("cue_diagnosis"),
                        "contract_qa": item.get("contract_qa"),
                        "tp_transition": item.get("tp_transition"),
                    }
                ],
                "variant_generation_errors": [item.get("generation_error")]
                if isinstance(item.get("generation_error"), dict)
                else [],
                "variant_results": [item["result"]] if isinstance(item.get("result"), dict) else [],
                "variant_results_complete": isinstance(item.get("result"), dict),
                "stop_reason": item.get("status"),
            }
            for item in iteration_records
        ],
        "adaptive_budget": {
            "preset": "eval-awareness-iterator",
            "shape": [1] * max_rewrites,
            "max_browser_variants": max_rewrites,
            "generated": len(iteration_records),
            "remaining_budget": budget["remaining_budget"],
            "stop_reason": stop_reason,
            "rounds": [],
        },
        "budget": budget,
        "generation_errors": generation_errors,
        "variant_generation_errors": generation_errors,
        "selected_iteration": selected_iteration,
        "selected_result": best_result,
        "selection_reason": selection_reason,
        "stop_reason": stop_reason,
    }


def _variant_runtime_stop_detail(
    stop_reason: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Return compact diagnostics for evaluated variants that cannot continue.

    These entries are not rewrite/schema failures. They explain why TP/VEA may
    be missing or why a browser-evaluated variant cannot be selected.
    """

    encounter = result.get("encounter") if isinstance(result.get("encounter"), dict) else {}
    detail = {
        "failure_class": stop_reason,
        "reason": (
            "evaluated variant stopped before a scoreable TP/VEA comparison"
            if stop_reason == "task_broke"
            else "evaluated variant lost PVPO encounter evidence"
        ),
        "variant_outcome": result.get("outcome"),
        "variant_final_status": result.get("final_status"),
        "pvpo_failure": result.get("pvpo_failure") or encounter.get("pvpo_failure"),
        "pvpo_observation": encounter.get("pvpo_observation"),
        "max_coverage": encounter.get("max_coverage"),
        "transcript_purpose_failure": result.get("transcript_purpose_failure"),
        "verbalized_eval_awareness_failure": result.get("verbalized_eval_awareness_failure"),
    }
    return {key: value for key, value in detail.items() if value is not None}
