"""Variant regeneration audit summary."""

from __future__ import annotations

from collections import Counter
from typing import Any

from warp_taskgen.phase_4.result_summary.final_metrics import (
    _variation_record,
    ecologically_valid,
    variant_adversarial_reward_passed,
)
from warp_taskgen.phase_4.variant_accounting import (
    is_post_eval_rejection,
    semantic_variant_accounting,
)


def _string_or_none(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _compact_text(value: Any, *, limit: int = 220) -> str | None:
    text = _string_or_none(value)
    if text is None:
        return None
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1].rstrip()}..."


def _variant_results(result: dict[str, Any]) -> list[dict[str, Any]]:
    variation = _variation_record(result)
    if variation is None:
        return []
    variants = variation.get("variant_results")
    if not isinstance(variants, list):
        return []
    return [variant for variant in variants if isinstance(variant, dict)]


def _rejected_variant_records(result: dict[str, Any]) -> list[dict[str, Any]]:
    variation = _variation_record(result)
    if variation is None:
        return []
    records: list[dict[str, Any]] = []
    raw_generation_records = variation.get("variant_generation_records")
    if isinstance(raw_generation_records, list):
        for raw_record in raw_generation_records:
            if not isinstance(raw_record, dict):
                continue
            status = _variant_generation_record_status(raw_record)
            if status == "generated":
                continue
            records.append(
                {
                    "index": raw_record.get("index"),
                    "strategy": _strategy_name(raw_record.get("strategy")),
                    "status": status,
                    "reason": _compact_text(
                        raw_record.get("reason") or raw_record.get("error"),
                        limit=240,
                    )
                    or "",
                }
            )
    if records:
        return records

    raw_generation_errors = variation.get("variant_generation_errors")
    if isinstance(raw_generation_errors, list):
        for raw_error in raw_generation_errors:
            if not isinstance(raw_error, dict):
                continue
            records.append(
                {
                    "index": raw_error.get("index"),
                    "strategy": _strategy_name(raw_error.get("strategy")),
                    "status": str(raw_error.get("status") or "error"),
                    "reason": _compact_text(
                        raw_error.get("reason") or raw_error.get("error"),
                        limit=240,
                    )
                    or "",
                }
            )
    return records


def _variant_rounds(variation: dict[str, Any]) -> list[dict[str, Any]]:
    raw_rounds = variation.get("variant_rounds") or variation.get("adaptive_rounds")
    if not isinstance(raw_rounds, list):
        return []
    return [round_record for round_record in raw_rounds if isinstance(round_record, dict)]


def _round_index(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _generation_records_for_round(
    variation: dict[str, Any],
    round_record: dict[str, Any],
    round_index: int,
) -> list[dict[str, Any]]:
    records = round_record.get("variant_generation_records")
    if isinstance(records, list) and records:
        return [item for item in records if isinstance(item, dict)]
    raw_records = variation.get("variant_generation_records")
    if not isinstance(raw_records, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw_records:
        if not isinstance(item, dict):
            continue
        item_round = _round_index(item.get("round_index"))
        if item_round == round_index or (item_round is None and round_index == 1):
            out.append(item)
    return out


def _variant_results_for_round(
    variation: dict[str, Any],
    round_record: dict[str, Any],
    round_index: int,
) -> list[dict[str, Any]]:
    records = round_record.get("variant_results")
    if isinstance(records, list) and records:
        return [item for item in records if isinstance(item, dict)]
    raw_records = variation.get("variant_results")
    if not isinstance(raw_records, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw_records:
        if not isinstance(item, dict):
            continue
        item_round = _round_index(item.get("round_index"))
        if item_round == round_index or (item_round is None and round_index == 1):
            out.append(item)
    return out


def _planned_strategy_names_for_rounds(
    variant_rounds: list[dict[str, Any]],
) -> list[str]:
    planned: list[str] = []
    for round_record in variant_rounds:
        raw = round_record.get("planned_strategies")
        if not isinstance(raw, list):
            continue
        for strategy in raw:
            name = _strategy_name(strategy)
            if name != "unknown":
                planned.append(name)
    return planned


def _adaptive_budget_shape(variation: dict[str, Any]) -> list[int]:
    raw_budget = variation.get("adaptive_budget")
    if not isinstance(raw_budget, dict):
        return []
    raw_shape = raw_budget.get("shape")
    if not isinstance(raw_shape, list):
        return []
    shape: list[int] = []
    for item in raw_shape:
        if isinstance(item, int):
            shape.append(item)
    return shape


def _normalized_adaptive_budget(
    variation: dict[str, Any],
    *,
    variant_rounds: list[dict[str, Any]],
    budget_shape: list[int],
) -> dict[str, Any]:
    rounds: list[dict[str, Any]] = []
    consumed = 0
    for offset, budget in enumerate(budget_shape, start=1):
        round_record = next(
            (item for item in variant_rounds if _round_index(item.get("round_index")) == offset),
            {},
        )
        generation_records = (
            _generation_records_for_round(variation, round_record, offset)
            if isinstance(round_record, dict)
            else []
        )
        variant_results = (
            _variant_results_for_round(variation, round_record, offset)
            if isinstance(round_record, dict)
            else []
        )
        generated = len(generation_records)
        consumed += generated
        rounds.append(
            {
                "round_index": offset,
                "budget": budget,
                "generated": generated,
                "host_rejected": sum(
                    1 for item in generation_records if item.get("status") != "generated"
                ),
                "evaluated": len(variant_results),
                "pvpo_valid": sum(1 for item in variant_results if ecologically_valid(item)),
                "compliant": sum(
                    1
                    for item in variant_results
                    if ecologically_valid(item) and item.get("outcome") == "complied"
                ),
                "remaining_round_budget": max(0, budget - generated),
                "stop_reason": (
                    str(round_record.get("stop_reason") or "unknown")
                    if isinstance(round_record, dict) and round_record
                    else "not_started"
                ),
                "lineage": [
                    {
                        "global_variant_index": item.get("global_variant_index", item.get("index")),
                        "parent_global_variant_index": item.get("parent_global_variant_index"),
                        "round_variant_index": item.get("round_variant_index"),
                    }
                    for item in generation_records
                ],
            }
        )
    return {
        "shape": budget_shape,
        "max_browser_variants": sum(budget_shape),
        "generated": consumed,
        "remaining_budget": max(0, sum(budget_shape) - consumed),
        "stop_reason": str(variation.get("stop_reason") or "unknown"),
        "rounds": rounds,
    }


def _strategy_name(strategy: Any) -> str:
    if isinstance(strategy, dict):
        value = strategy.get("strategy")
        if isinstance(value, str) and value.strip():
            return value.strip()
    if isinstance(strategy, str) and strategy.strip():
        return strategy.strip()
    return "unknown"


def _recommended_strategy_names(judge: dict[str, Any]) -> list[str]:
    strategies = judge.get("recommended_strategies")
    if not isinstance(strategies, list):
        return []
    names: list[str] = []
    for strategy in strategies[:3]:
        name = _strategy_name(strategy)
        if name != "unknown":
            names.append(name)
    return names


def _variant_generation_record_status(record: dict[str, Any]) -> str:
    status = record.get("status")
    if isinstance(status, str) and status.strip():
        return status.strip()
    if isinstance(record.get("variant"), dict):
        return "generated"
    if isinstance(record.get("error"), str):
        return "error"
    return "unknown"


def variant_regeneration_audit(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize the judge -> strategy -> generation -> PVPO variant flow."""

    tasks_entered = 0
    planned_attempts = 0
    generated_attempts = 0
    evaluated_attempts = 0
    gate1_valid_evaluations = 0
    compliant_evaluations = 0
    rejected_before_eval = 0
    rejected_after_eval = 0
    schema_validation_failures = 0
    tp_regression_rejections = 0
    contract_inapplicable_rejections = 0
    tasks_with_adaptive_rounds = 0
    max_rounds_observed = 0
    max_budget_observed = 0
    judge_status_counts = Counter()
    judge_trigger_counts = Counter()
    judge_confidence_counts = Counter()
    generation_status_counts = Counter()
    round_status_counts = Counter()
    round_kind_counts = Counter()
    trigger_strategy_rows = Counter()
    task_records: list[dict[str, Any]] = []

    for result in results:
        variation = _variation_record(result)
        if not isinstance(variation, dict):
            continue
        if variation.get("status") == "skipped":
            continue
        tasks_entered += 1
        judge = variation.get("judge_diagnosis")
        judge = judge if isinstance(judge, dict) else {}
        judge_status = str(judge.get("status") or "unknown")
        trigger = str(judge.get("refusal_trigger") or "unknown")
        confidence = str(judge.get("confidence") or "unknown")
        judge_status_counts[judge_status] += 1
        judge_trigger_counts[trigger] += 1
        judge_confidence_counts[confidence] += 1

        recommended = _recommended_strategy_names(judge)
        raw_variant_results = variation.get("variant_results")
        variant_results = (
            [variant for variant in raw_variant_results if isinstance(variant, dict)]
            if isinstance(raw_variant_results, list)
            else []
        )
        raw_generation_errors = variation.get("variant_generation_errors")
        generation_errors = raw_generation_errors if isinstance(raw_generation_errors, list) else []
        accounting = semantic_variant_accounting(
            variant_results=variant_results,
            generation_errors=generation_errors,
        )
        raw_generation_records = variation.get("variant_generation_records")
        generation_records = (
            [record for record in raw_generation_records if isinstance(record, dict)]
            if isinstance(raw_generation_records, list)
            else []
        )
        variant_rounds = _variant_rounds(variation)
        budget_shape = _adaptive_budget_shape(variation)
        if variant_rounds:
            tasks_with_adaptive_rounds += 1
            max_rounds_observed = max(max_rounds_observed, len(variant_rounds))
            for round_record in variant_rounds:
                round_index = _round_index(round_record.get("round_index"))
                if round_index is not None:
                    round_status_counts[
                        f"r{round_index}:{round_record.get('stop_reason') or 'unknown'}"
                    ] += 1
                kind = str(round_record.get("round_kind") or "unknown")
                round_kind_counts[kind] += 1
        if budget_shape:
            max_budget_observed = max(max_budget_observed, sum(budget_shape))

        planned = _planned_strategy_names_for_rounds(variant_rounds)
        if not planned:
            planned = list(recommended)
        if not planned:
            for variant in variant_results:
                name = _strategy_name(variant.get("strategy"))
                if name != "unknown":
                    planned.append(name)
            for error in generation_errors:
                if isinstance(error, dict):
                    name = _strategy_name(error.get("strategy"))
                    if name != "unknown":
                        planned.append(name)

        planned_attempts += len(planned)
        for strategy in planned:
            trigger_strategy_rows[(trigger, strategy, "planned")] += 1

        generated_for_task = 0
        rejected_for_task = 0
        post_eval_rejected_for_task = 0
        if generation_records:
            for record in generation_records:
                name = _strategy_name(record.get("strategy"))
                status = _variant_generation_record_status(record)
                generation_status_counts[status] += 1
                if status == "generated":
                    generated_attempts += 1
                    generated_for_task += 1
                    trigger_strategy_rows[(trigger, name, "generated")] += 1
                elif status in {"failed", "inapplicable", "skipped", "error"}:
                    rejected_before_eval += 1
                    rejected_for_task += 1
                    trigger_strategy_rows[(trigger, name, "rejected")] += 1
                else:
                    trigger_strategy_rows[(trigger, name, "unknown_generation_status")] += 1
        else:
            generated_attempts += len(variant_results)
            generated_for_task += len(variant_results)
            generation_status_counts["generated"] += len(variant_results)
            rejected_before_eval += accounting["pre_browser_rejections"]
            rejected_after_eval += accounting["post_eval_rejections"]
            schema_validation_failures += accounting["schema_validation_failures"]
            tp_regression_rejections += accounting["tp_regression_rejections"]
            contract_inapplicable_rejections += accounting["contract_inapplicable_rejections"]
            rejected_for_task += accounting["pre_browser_rejections"]
            post_eval_rejected_for_task += accounting["post_eval_rejections"]
            if accounting["pre_browser_rejections"]:
                generation_status_counts["pre_browser_rejected"] += accounting[
                    "pre_browser_rejections"
                ]
            if accounting["post_eval_rejections"]:
                generation_status_counts["post_eval_rejected"] += accounting["post_eval_rejections"]
            for variant in variant_results:
                trigger_strategy_rows[
                    (trigger, _strategy_name(variant.get("strategy")), "generated")
                ] += 1
            for error in generation_errors:
                if isinstance(error, dict):
                    metric = "post_eval_rejected" if is_post_eval_rejection(error) else "rejected"
                    trigger_strategy_rows[
                        (trigger, _strategy_name(error.get("strategy")), metric)
                    ] += 1

        evaluated_attempts += len(variant_results)
        for variant in variant_results:
            name = _strategy_name(variant.get("strategy"))
            trigger_strategy_rows[(trigger, name, "evaluated")] += 1
            if ecologically_valid(variant):
                gate1_valid_evaluations += 1
                trigger_strategy_rows[(trigger, name, "gate1_valid")] += 1
            else:
                trigger_strategy_rows[(trigger, name, "gate1_invalid")] += 1
            if (
                ecologically_valid(variant)
                and variant.get("outcome") == "complied"
                and variant_adversarial_reward_passed(variant)
            ):
                compliant_evaluations += 1
                trigger_strategy_rows[(trigger, name, "complied")] += 1

        task_records.append(
            {
                "task_id": str(result.get("task_id") or ""),
                "final_status": str(result.get("final_status") or "missing"),
                "judge_status": judge_status,
                "refusal_trigger": trigger,
                "confidence": confidence,
                "planned_strategies": planned,
                "rounds": len(variant_rounds),
                "budget_shape": budget_shape,
                "adaptive_budget": _normalized_adaptive_budget(
                    variation,
                    variant_rounds=variant_rounds,
                    budget_shape=budget_shape,
                ),
                "stop_reason": str(variation.get("stop_reason") or "unknown"),
                "generated": generated_for_task,
                "rejected_before_eval": rejected_for_task,
                "rejected_after_eval": post_eval_rejected_for_task,
                "evaluated": len(variant_results),
                "variant_outcomes": dict(
                    sorted(
                        Counter(
                            str(variant.get("outcome") or "missing") for variant in variant_results
                        ).items()
                    )
                ),
                "rejected_variants": _rejected_variant_records(result),
            }
        )

    row_map: dict[tuple[str, str], dict[str, Any]] = {}
    for (trigger, strategy, metric), count in trigger_strategy_rows.items():
        key = (trigger, strategy)
        row = row_map.setdefault(
            key,
            {
                "refusal_trigger": trigger,
                "strategy": strategy,
                "planned": 0,
                "generated": 0,
                "rejected": 0,
                "evaluated": 0,
                "gate1_valid": 0,
                "gate1_invalid": 0,
                "complied": 0,
            },
        )
        row[metric] = count

    return {
        "tasks_entered": tasks_entered,
        "planned_attempts": planned_attempts,
        "generated_attempts": generated_attempts,
        "rejected_before_eval": rejected_before_eval,
        "rejected_after_eval": rejected_after_eval,
        "schema_validation_failures": schema_validation_failures,
        "tp_regression_rejections": tp_regression_rejections,
        "contract_inapplicable_rejections": contract_inapplicable_rejections,
        "evaluated_attempts": evaluated_attempts,
        "gate1_valid_evaluations": gate1_valid_evaluations,
        "compliant_evaluations": compliant_evaluations,
        "tasks_with_adaptive_rounds": tasks_with_adaptive_rounds,
        "max_rounds_observed": max_rounds_observed,
        "max_budget_observed": max_budget_observed,
        "judge_status_counts": dict(sorted(judge_status_counts.items())),
        "judge_trigger_counts": dict(sorted(judge_trigger_counts.items())),
        "judge_confidence_counts": dict(sorted(judge_confidence_counts.items())),
        "generation_status_counts": dict(sorted(generation_status_counts.items())),
        "round_status_counts": dict(sorted(round_status_counts.items())),
        "round_kind_counts": dict(sorted(round_kind_counts.items())),
        "trigger_strategy_rows": sorted(
            row_map.values(),
            key=lambda row: (str(row["refusal_trigger"]), str(row["strategy"])),
        ),
        "task_records": sorted(task_records, key=lambda row: str(row["task_id"])),
    }


__all__ = ["variant_regeneration_audit"]
