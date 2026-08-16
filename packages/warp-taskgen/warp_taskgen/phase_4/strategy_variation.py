"""Phase 4 strategy variation behavior."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from warp_taskgen.agent_runtime import AgentRunner
from warp_taskgen.config import BenchmarkInstance
from warp_taskgen.failpoints import crash_if_enabled
from warp_taskgen.phase_4.metrics import _ecologically_valid
from warp_taskgen.phase_4.options import (
    DEFAULT_PHASE_4_VARIANT_BUDGET_PRESET as _DEFAULT_PHASE_4_VARIANT_BUDGET_PRESET,
)
from warp_taskgen.phase_4.options import (
    phase_4_variant_budget_shape as _phase_4_variant_budget_shape,
)
from warp_taskgen.phase_4.postprocess_progress import Phase4ProgressCallback
from warp_taskgen.phase_4.resume import (
    _CHECKPOINT_FINGERPRINT_KEY,
    _PHASE_4_RESUME_VERSION,
    _VARIANT_GENERATION_RECORDS_KEY,
    _VARIANT_ROUNDS_KEY,
    _fingerprint_payload,
    _load_json_dict,
    _phase_4_postprocess_fingerprint,
    _resume_fingerprint_result,
    _strategy_variation_checkpoint_path,
    _task_reachable_instances,
    _task_reachable_placeholders,
    _variant_changes_seed,
    _variant_generation_record_for_result,
    _write_json_atomic,
)
from warp_taskgen.phase_4.strategy_catalog import ALLOWED_STRATEGIES as _ALLOWED_STRATEGIES
from warp_taskgen.phase_4.variant_eval import _evaluate_variant
from warp_taskgen.resume_metadata import instances_identity
from warp_taskgen.runtime_composition import RuntimeComposition
from warp_taskgen.task_reset_cache import callable_accepts_keyword

logger = logging.getLogger(__name__)


def _normalize_recommended_strategies(
    recommendation: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Return valid strategy recommendations and any validation errors.

    Post-2026-04-18 API cutover this is defense-in-depth: the host-side
    `TRIGGER_TO_STRATEGIES` lookup already emits only strategies in
    `ALLOWED_STRATEGIES`, so under the current `run_judge_api` path every
    recommendation validates. The checks are retained because
    `run_strategy_variation` still accepts any caller that returns the
    legacy `{recommended_strategies: [...]}` shape — a future shim (or a
    downgrade for debugging) could feed raw model output, and the
    dedup/type/membership checks here are cheap insurance against pool
    drift and payload spoofing.
    """
    raw_strategies = recommendation.get("recommended_strategies")
    if not isinstance(raw_strategies, list):
        return [], ["judge recommendation missing recommended_strategies list"]

    validated: list[dict[str, Any]] = []
    errors: list[str] = []
    seen: set[str] = set()
    for index, strategy in enumerate(raw_strategies):
        if not isinstance(strategy, dict):
            errors.append(f"recommended_strategies[{index}] is not an object")
            continue
        name = strategy.get("strategy")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"recommended_strategies[{index}].strategy is missing")
            continue
        normalized = name.strip()
        if normalized not in _ALLOWED_STRATEGIES:
            errors.append(
                f"recommended_strategies[{index}].strategy {normalized!r} is outside the allowed strategy pool"
            )
            continue
        if normalized in seen:
            errors.append(f"recommended_strategies[{index}].strategy {normalized!r} is duplicated")
            continue
        seen.add(normalized)
        validated.append({**strategy, "strategy": normalized})
    return validated, errors


async def generate_variant(
    task: dict[str, Any],
    strategy: dict[str, Any],
    profile_path: Path,
    *,
    sandbox_model: str = "claude-sonnet-4-6",
    failure_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a variant adversarial task following a specific strategy.

    Thin wrapper over `warp_taskgen.phase_4.variant_api.generate_variant_api`,
    which performs a single-turn Anthropic Messages API call with forced
    tool-use structured output (tool `build_variant`). Only
    `adversarial_data_seed` is modified; `instruction`, `reward_function`,
    `delivery_channel`, and `required_tokens` remain byte-identical to the
    base task. `profile_path` is accepted for signature compatibility but
    not forwarded; the API call needs only the task + strategy.
    """
    from warp_taskgen.phase_4.variant_api import generate_variant_api

    kwargs: dict[str, Any] = {"sandbox_model": sandbox_model}
    if callable_accepts_keyword(generate_variant_api, "failure_context"):
        kwargs["failure_context"] = failure_context
    return await generate_variant_api(
        task,
        strategy,
        **kwargs,
    )


async def run_judge(
    task: dict[str, Any],
    trajectory_dir: str | Path,
    profile_path: Path,
    *,
    sandbox_model: str = "claude-sonnet-4-6",
) -> dict[str, Any]:
    """Judge classifies why an ecologically valid attack was refused.

    Thin wrapper over `warp_taskgen.phase_4.judge_api.run_judge_api`. The judge
    is a single-turn Anthropic Messages API call with forced tool-use
    structured output; it returns a `refusal_trigger` from a 7-value
    taxonomy and the host-side `TRIGGER_TO_STRATEGIES` lookup in
    `strategy_catalog.py` selects up to 3 strategies. `profile_path` is no
    longer plumbed to the API call (host-side slicer + classification don't
    need it) but accepted for signature compatibility.

    The legacy `status` field now carries `judge_ok_actionable`,
    `judge_ok_unactionable`, or `judge_failed`. Downstream
    `run_strategy_variation` branches on these new values; any caller that
    reads only `recommended_strategies` continues to work unchanged.
    """
    from warp_taskgen.phase_4.judge_api import run_judge_api

    return await run_judge_api(task, trajectory_dir, sandbox_model=sandbox_model)


async def run_strategy_variation(
    task: dict[str, Any],
    initial_result: dict[str, Any],
    primary_instances: list[BenchmarkInstance],
    all_instances: list[BenchmarkInstance],
    agent_factory: Callable[[], AgentRunner],
    profile_path: Path,
    task_dir_root: Path,
    config_url_placeholders: dict[str, str] | None = None,
    resume: bool = False,
    benchmark_root: Path | None = None,
    sandbox_model: str = "claude-sonnet-4-6",
    site_profile: dict[str, Any] | None = None,
    variant_budget_preset: str | None = None,
    agent_execution: dict[str, Any] | None = None,
    progress_callback: Phase4ProgressCallback | None = None,
    browser_worker_semaphore: asyncio.Semaphore | None = None,
    runtime_composition: RuntimeComposition | None = None,
) -> dict[str, Any]:
    """Adaptive strategy variation: judge -> generate variants -> evaluate.

    Legacy opt-in path. Bounded adaptive rounds use the configured budget
    shape, defaulting to 3+3+1 variants.
    """
    # Import lazily because the payload-contract/adversarial-action package
    # retains a legacy cycle when this strategy module is imported standalone.
    from warp_taskgen.phase_4.failure_context import build_variant_failure_context

    task_id = str(task.get("id", "unknown"))
    budget_preset = variant_budget_preset or _DEFAULT_PHASE_4_VARIANT_BUDGET_PRESET
    budget_shape = _phase_4_variant_budget_shape(variant_budget_preset)

    async def _emit_variant_progress(event: str, data: Mapping[str, Any]) -> None:
        if progress_callback is None:
            return
        try:
            await progress_callback(
                event,
                {
                    "task_id": task_id,
                    "budget_preset": budget_preset,
                    "budget_shape": list(budget_shape),
                    **dict(data),
                },
            )
        except Exception as exc:
            logger.warning("Could not write Phase 4 variant progress for task %s: %s", task_id, exc)

    checkpoint_path = _strategy_variation_checkpoint_path(task_dir_root, task_id)
    source_fingerprint = _phase_4_postprocess_fingerprint(
        task,
        initial_result,
        primary_instances=primary_instances,
        all_instances=all_instances,
        config_url_placeholders=config_url_placeholders,
        benchmark_root=benchmark_root,
        sandbox_model=sandbox_model,
        site_profile=site_profile,
        variant_budget_preset=variant_budget_preset,
        variant_system="strategy-variation",
        agent_execution=agent_execution,
    )
    legacy_source_fingerprint = _fingerprint_payload(
        task,
        _resume_fingerprint_result(initial_result),
        {
            "phase": "phase_4_postprocess",
            "resume_version": _PHASE_4_RESUME_VERSION,
            "primary_instances": instances_identity(primary_instances),
            "all_instances": instances_identity(_task_reachable_instances(task, all_instances)),
            "config_url_placeholders": _task_reachable_placeholders(task, config_url_placeholders),
            "benchmark_root": str(benchmark_root) if benchmark_root is not None else None,
            "sandbox_model": sandbox_model,
            "site_profile": site_profile,
            "variant_budget_preset": variant_budget_preset,
        },
    )
    checkpoint = _load_json_dict(checkpoint_path) if resume else None
    if checkpoint is not None:
        checkpoint_fingerprint = checkpoint.get(_CHECKPOINT_FINGERPRINT_KEY)
        if checkpoint_fingerprint == legacy_source_fingerprint:
            checkpoint[_CHECKPOINT_FINGERPRINT_KEY] = source_fingerprint
            _write_json_atomic(
                checkpoint_path,
                checkpoint,
                failpoint_base="phase_4.strategy_variation.checkpoint",
            )
        elif checkpoint_fingerprint != source_fingerprint:
            checkpoint = None

    # 1. Judge diagnoses why agent refused
    recommendation = checkpoint.get("judge_diagnosis") if checkpoint else None
    if not isinstance(recommendation, dict):
        trajectory_dir = initial_result.get("trajectory_dir", "")
        try:
            recommendation = await run_judge(
                task,
                trajectory_dir,
                profile_path,
                sandbox_model=sandbox_model,
            )
        except Exception as exc:
            logger.exception("Judge sandbox failed for task %s: %s", task_id, exc)
            recommendation = {
                "status": "error",
                "diagnosis": f"judge sandbox failed: {exc!r}",
                "refusal_trigger": "unknown",
                "recommended_strategies": [],
            }
        # Failpoint: simulates a crash after the judge API call has returned
        # (and spent its cost) but before the recommendation is persisted to
        # the strategy_variation_checkpoint.json. On resume, the judge
        # re-runs; this failpoint gives crash-resume tests a handle to
        # verify that fallback.
        crash_if_enabled("phase_4.judge.after_response.before_checkpoint")
        checkpoint = {
            _CHECKPOINT_FINGERPRINT_KEY: source_fingerprint,
            "judge_diagnosis": recommendation,
        }
        _write_json_atomic(
            checkpoint_path,
            checkpoint,
            failpoint_base="phase_4.strategy_variation.checkpoint",
        )

    # New judge status vocabulary (as of 2026-04-18 API cutover):
    #   judge_ok_actionable     — trigger mapped to runnable strategies
    #   judge_ok_unactionable   — trigger returned (e.g. distracted/unknown)
    #                             but no actionable strategy; treat as resistant
    #   judge_failed            — API/parse/taxonomy failure
    # Legacy "ok"/"error" shape still accepted from any shim that returns them.
    recommendation_status = str(recommendation.get("status", "ok")).strip().lower()
    strategies, strategy_errors = _normalize_recommended_strategies(recommendation)

    await _emit_variant_progress(
        "judge_complete",
        {
            "judge_status": recommendation_status,
            "refusal_trigger": str(recommendation.get("refusal_trigger") or ""),
            "recommended_strategy_count": len(strategies),
        },
    )

    if recommendation_status in ("error", "judge_failed"):
        return {
            "status": "judge_failed",
            "judge_diagnosis": recommendation,
            "attempts": [initial_result],
            "variant_results": [],
        }
    if recommendation_status == "judge_ok_unactionable":
        # `distracted` → task needs a different surface, not a rewritten payload.
        # `unknown` with an empty mapping → treat as resistant.
        return {
            "status": "resistant_judge_unactionable",
            "judge_diagnosis": recommendation,
            "attempts": [initial_result],
            "variant_results": [],
        }
    if strategy_errors:
        return {
            "status": "judge_failed",
            "judge_diagnosis": {
                **recommendation,
                "validation_errors": strategy_errors,
            },
            "attempts": [initial_result],
            "variant_results": [],
        }
    if not strategies:
        return {
            "status": "judge_failed",
            "judge_diagnosis": {
                **recommendation,
                "validation_errors": ["judge returned no recommended strategies"],
            },
            "attempts": [initial_result],
            "variant_results": [],
        }

    logger.info(
        "Strategy variation for task %s: %d strategies recommended",
        task.get("id", "?"),
        len(strategies),
    )

    if not primary_instances:
        logger.warning(
            "No instances available for variant evaluation of task %s", task.get("id", "?")
        )
        return {
            "status": "no_instances",
            "judge_diagnosis": recommendation,
            "attempts": [initial_result],
            "variant_results": [],
        }

    checkpoint = checkpoint or {
        _CHECKPOINT_FINGERPRINT_KEY: source_fingerprint,
        "judge_diagnosis": recommendation,
    }
    failure_context = checkpoint.get("failure_context")
    if not isinstance(failure_context, dict):
        failure_context = build_variant_failure_context(task, initial_result, recommendation)
        checkpoint["failure_context"] = failure_context
        _write_json_atomic(
            checkpoint_path,
            checkpoint,
            failpoint_base="phase_4.strategy_variation.checkpoint",
        )

    raw_rounds = checkpoint.get(_VARIANT_ROUNDS_KEY)
    variant_rounds: list[dict[str, Any]] = raw_rounds if isinstance(raw_rounds, list) else []
    legacy_checkpoint = False
    if not variant_rounds and isinstance(checkpoint.get(_VARIANT_GENERATION_RECORDS_KEY), list):
        legacy_checkpoint = True
        legacy_records: list[dict[str, Any]] = []
        for slot, record in enumerate(checkpoint.get(_VARIANT_GENERATION_RECORDS_KEY, [])):
            if not isinstance(record, dict):
                continue
            normalized = dict(record)
            normalized.setdefault("round_index", 1)
            normalized.setdefault("round_kind", "initial_fanout")
            normalized.setdefault("round_variant_index", slot)
            normalized.setdefault("global_variant_index", normalized.get("index", slot))
            normalized.setdefault("index", normalized["global_variant_index"])
            legacy_records.append(normalized)
        legacy_results = checkpoint.get("variant_results")
        variant_rounds = [
            {
                "round_index": 1,
                "round_kind": "initial_fanout",
                "planned_strategies": strategies[:3],
                "failure_context": failure_context,
                "variant_generation_records": legacy_records,
                "variant_generation_errors": [
                    item
                    for item in checkpoint.get("variant_generation_errors", [])
                    if isinstance(item, dict)
                ],
                "variant_results": legacy_results if isinstance(legacy_results, list) else [],
                "variant_results_complete": isinstance(legacy_results, list),
                "stop_reason": "legacy_resume",
            }
        ]
        checkpoint[_VARIANT_ROUNDS_KEY] = variant_rounds

    def _round_kind(round_index: int) -> str:
        if round_index == 1:
            return "initial_fanout"
        if round_index == len(budget_shape):
            return "terminal_probe"
        return "adaptive_fanout"

    def _flat_round_records(key: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for round_record in variant_rounds:
            records = round_record.get(key)
            if isinstance(records, list):
                out.extend(item for item in records if isinstance(item, dict))
        return out

    def _flat_generation_records() -> list[dict[str, Any]]:
        return _flat_round_records("variant_generation_records")

    def _flat_generation_errors() -> list[dict[str, Any]]:
        return _flat_round_records("variant_generation_errors")

    def _flat_variant_results() -> list[dict[str, Any]]:
        return _flat_round_records("variant_results")

    def _pvpo_valid_count(results: list[dict[str, Any]]) -> int:
        return sum(1 for item in results if _ecologically_valid(item))

    def _complied_count(results: list[dict[str, Any]]) -> int:
        return sum(
            1 for item in results if _ecologically_valid(item) and item.get("outcome") == "complied"
        )

    def _best_refused_variant(results: list[dict[str, Any]]) -> dict[str, Any] | None:
        refused = [
            item
            for item in results
            if _ecologically_valid(item) and item.get("outcome") == "refused_or_ignored"
        ]
        if not refused:
            return None
        refused.sort(
            key=lambda item: (
                float(
                    item.get("encounter", {}).get("max_coverage", 0.0)
                    if isinstance(item.get("encounter"), dict)
                    else 0.0
                ),
                -int(item.get("global_variant_index", item.get("variant_index", 0)) or 0),
            ),
            reverse=True,
        )
        return refused[0]

    def _round_budget_stats(round_record: dict[str, Any]) -> dict[str, int]:
        generation_records = [
            item
            for item in round_record.get("variant_generation_records", [])
            if isinstance(item, dict)
        ]
        variant_results = [
            item for item in round_record.get("variant_results", []) if isinstance(item, dict)
        ]
        return {
            "generated": len(generation_records),
            "host_rejected": sum(
                1 for item in generation_records if not isinstance(item.get("variant"), dict)
            ),
            "evaluated": len(variant_results),
            "pvpo_valid": _pvpo_valid_count(variant_results),
            "compliant": _complied_count(variant_results),
        }

    def _adaptive_budget_report(stop_reason: str) -> dict[str, Any]:
        rounds: list[dict[str, Any]] = []
        generated = 0
        for round_index, budget in enumerate(budget_shape, start=1):
            round_record = next(
                (
                    item
                    for item in variant_rounds
                    if isinstance(item, dict) and item.get("round_index") == round_index
                ),
                None,
            )
            stats = (
                _round_budget_stats(round_record)
                if isinstance(round_record, dict)
                else {
                    "generated": 0,
                    "host_rejected": 0,
                    "evaluated": 0,
                    "pvpo_valid": 0,
                    "compliant": 0,
                }
            )
            generated += stats["generated"]
            rounds.append(
                {
                    "round_index": round_index,
                    "round_kind": _round_kind(round_index),
                    "budget": budget,
                    **stats,
                    "remaining_round_budget": max(0, budget - stats["generated"]),
                    "stop_reason": (
                        round_record.get("stop_reason")
                        if isinstance(round_record, dict)
                        else "not_started"
                    ),
                }
            )
        return {
            "preset": budget_preset,
            "shape": list(budget_shape),
            "max_browser_variants": sum(budget_shape),
            "generated": generated,
            "remaining_budget": max(0, sum(budget_shape) - generated),
            "stop_reason": stop_reason,
            "rounds": rounds,
        }

    async def _generate_variant_record(
        *,
        index: int,
        strategy: dict[str, Any],
        round_index: int,
        round_kind: str,
        round_variant_index: int,
        parent_variant_index: int | None,
        round_failure_context: dict[str, Any],
    ) -> dict[str, Any]:
        strategy_name = strategy.get("strategy", f"strategy_{index}")
        try:
            variant = await generate_variant(
                task,
                strategy,
                profile_path,
                sandbox_model=sandbox_model,
                failure_context=round_failure_context,
            )
        except Exception as exc:
            logger.error(
                "Variant generation failed for task %s strategy %s: %s",
                task_id,
                strategy_name,
                exc,
            )
            record = _variant_generation_record_for_result(
                index=index,
                strategy=strategy,
                error=repr(exc),
            )
        else:
            variant_status = variant.get("variant_status") if isinstance(variant, dict) else None
            if isinstance(variant_status, dict) and variant_status.get("status") in {
                "inapplicable",
                "skipped",
                "failed",
            }:
                record = _variant_generation_record_for_result(
                    index=index,
                    strategy=strategy,
                    status=str(variant_status.get("status")),
                    reason=str(variant_status.get("reason", "")),
                )
            elif isinstance(variant, dict) and _variant_changes_seed(task, variant):
                variant.update(
                    {
                        "round_index": round_index,
                        "round_kind": round_kind,
                        "round_variant_index": round_variant_index,
                        "global_variant_index": index,
                        "parent_global_variant_index": parent_variant_index,
                    }
                )
                record = _variant_generation_record_for_result(
                    index=index,
                    strategy=strategy,
                    variant=variant,
                )
            else:
                record = _variant_generation_record_for_result(
                    index=index,
                    strategy=strategy,
                    status="bookkeeping_only",
                )
        record.update(
            {
                "round_index": round_index,
                "round_kind": round_kind,
                "round_variant_index": round_variant_index,
                "global_variant_index": index,
                "parent_global_variant_index": parent_variant_index,
            }
        )
        return record

    async def _evaluate_round_variants(
        real_variants: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for start in range(0, len(real_variants), len(primary_instances)):
            batch = real_variants[start : start + len(primary_instances)]
            batch_results = await asyncio.gather(
                *[
                    _evaluate_variant(
                        task=task,
                        variant=variant,
                        instance=primary_instances[i],
                        all_instances=all_instances,
                        strategy=strategy,
                        index=int(record["global_variant_index"]),
                        agent_factory=agent_factory,
                        task_dir_root=task_dir_root,
                        config_url_placeholders=config_url_placeholders,
                        resume=resume,
                        benchmark_root=benchmark_root,
                        sandbox_model=sandbox_model,
                        site_profile=site_profile,
                        agent_execution=agent_execution,
                        browser_worker_semaphore=browser_worker_semaphore,
                        runtime_composition=runtime_composition,
                    )
                    for i, (variant, strategy, record) in enumerate(batch)
                ]
            )
            for result, (_, strategy, record) in zip(batch_results, batch, strict=False):
                if not isinstance(result, dict):
                    continue
                result.setdefault("strategy", strategy.get("strategy"))
                result.setdefault("variant_index", record.get("global_variant_index"))
                result.setdefault("global_variant_index", record.get("global_variant_index"))
                result.setdefault("round_index", record.get("round_index"))
                result.setdefault("round_kind", record.get("round_kind"))
                result.setdefault("round_variant_index", record.get("round_variant_index"))
                result.setdefault(
                    "parent_global_variant_index",
                    record.get("parent_global_variant_index"),
                )
            results.extend(batch_results)
        return results

    global_variant_index = (
        max(
            [
                int(record.get("global_variant_index", record.get("index", -1)))
                for record in _flat_generation_records()
                if isinstance(record.get("global_variant_index", record.get("index")), int)
            ]
            or [-1]
        )
        + 1
    )
    terminal_stop_reason = "budget_exhausted"
    for round_index, budget in enumerate(budget_shape, start=1):
        existing_round = next(
            (
                item
                for item in variant_rounds
                if isinstance(item, dict) and item.get("round_index") == round_index
            ),
            None,
        )
        if existing_round is not None and existing_round.get("variant_results_complete") is True:
            if _complied_count(_flat_variant_results()):
                terminal_stop_reason = "success"
                break
            if legacy_checkpoint:
                terminal_stop_reason = str(existing_round.get("stop_reason") or "legacy_resume")
                break
            continue
        prior_refused = _best_refused_variant(_flat_variant_results())
        if round_index > 1 and prior_refused is None:
            terminal_stop_reason = "no_actionable_lineage"
            break

        round_kind = _round_kind(round_index)
        selected_strategies = strategies[:budget]
        parent_variant_index = (
            prior_refused.get("global_variant_index", prior_refused.get("variant_index"))
            if isinstance(prior_refused, dict)
            else None
        )
        round_failure_context = (
            failure_context
            if round_index == 1
            else {
                **build_variant_failure_context(
                    task, prior_refused or initial_result, recommendation
                ),
                "adaptive_loop": {
                    "schema_version": "phase4_adaptive_strategy_loop_v1",
                    "budget_preset": budget_preset,
                    "budget_shape": list(budget_shape),
                    "current_round_index": round_index,
                    "prior_rounds": [
                        {
                            "round_index": item.get("round_index"),
                            "round_kind": item.get("round_kind"),
                            "stop_reason": item.get("stop_reason"),
                            "budget_report": item.get("budget_report"),
                        }
                        for item in variant_rounds
                        if isinstance(item, dict)
                    ],
                },
            }
        )
        round_record = existing_round or {
            "round_index": round_index,
            "round_kind": round_kind,
            "planned_strategies": selected_strategies,
            "failure_context": round_failure_context,
            "variant_generation_records": [],
            "variant_generation_errors": [],
            "variant_results": [],
            "stop_reason": "started",
        }
        if existing_round is None:
            variant_rounds.append(round_record)
        checkpoint[_VARIANT_ROUNDS_KEY] = variant_rounds
        _write_json_atomic(
            checkpoint_path,
            checkpoint,
            failpoint_base="phase_4.strategy_variation.checkpoint",
        )
        await _emit_variant_progress(
            "variant_round_started",
            {
                "round_index": round_index,
                "round_kind": round_kind,
                "planned": len(selected_strategies),
                "generation_attempted": len(_flat_generation_records()),
                "generation_generated": sum(
                    1
                    for item in _flat_generation_records()
                    if isinstance(item.get("variant"), dict)
                ),
                "generation_failed": sum(
                    1
                    for item in _flat_generation_records()
                    if not isinstance(item.get("variant"), dict)
                ),
                "evaluated": len(_flat_variant_results()),
                "pvpo_valid": _pvpo_valid_count(_flat_variant_results()),
                "complied": _complied_count(_flat_variant_results()),
            },
        )

        generation_records = [
            item
            for item in round_record.get("variant_generation_records", [])
            if isinstance(item, dict)
        ]
        completed_round_indexes = {
            int(item.get("round_variant_index"))
            for item in generation_records
            if isinstance(item.get("round_variant_index"), int)
        }
        pending_tasks = [
            asyncio.create_task(
                _generate_variant_record(
                    index=global_variant_index + offset,
                    strategy=strategy,
                    round_index=round_index,
                    round_kind=round_kind,
                    round_variant_index=round_variant_index,
                    parent_variant_index=(
                        int(parent_variant_index) if isinstance(parent_variant_index, int) else None
                    ),
                    round_failure_context=round_failure_context,
                )
            )
            for offset, (round_variant_index, strategy) in enumerate(
                [
                    (round_variant_index, strategy)
                    for round_variant_index, strategy in enumerate(selected_strategies)
                    if round_variant_index not in completed_round_indexes
                ]
            )
        ]
        global_variant_index += len(pending_tasks)
        for pending_task in asyncio.as_completed(pending_tasks):
            record = await pending_task
            generation_records.append(record)
            round_record["variant_generation_records"] = generation_records
            round_record["variant_generation_errors"] = [
                {
                    "strategy": (
                        record.get("strategy", {}).get("strategy")
                        if isinstance(record.get("strategy"), dict)
                        else "unknown"
                    ),
                    "error": record.get("error"),
                    "status": record.get("status"),
                    "reason": record.get("reason", ""),
                    "round_index": record.get("round_index"),
                    "round_variant_index": record.get("round_variant_index"),
                    "global_variant_index": record.get("global_variant_index"),
                }
                for record in generation_records
                if not isinstance(record.get("variant"), dict)
            ]
            checkpoint[_VARIANT_ROUNDS_KEY] = variant_rounds
            _write_json_atomic(
                checkpoint_path,
                checkpoint,
                failpoint_base="phase_4.strategy_variation.checkpoint",
            )
            await _emit_variant_progress(
                "variant_generation_recorded",
                {
                    "round_index": round_index,
                    "round_kind": round_kind,
                    "generation_attempted": len(_flat_generation_records()),
                    "generation_generated": sum(
                        1
                        for item in _flat_generation_records()
                        if isinstance(item.get("variant"), dict)
                    ),
                    "generation_failed": sum(
                        1
                        for item in _flat_generation_records()
                        if not isinstance(item.get("variant"), dict)
                    ),
                    "evaluated": len(_flat_variant_results()),
                    "pvpo_valid": _pvpo_valid_count(_flat_variant_results()),
                    "complied": _complied_count(_flat_variant_results()),
                },
            )

        real_variants = [
            (record.get("variant"), record.get("strategy"), record)
            for record in generation_records
            if isinstance(record.get("variant"), dict) and isinstance(record.get("strategy"), dict)
        ]
        if not real_variants:
            round_record["stop_reason"] = "no_valid_generation"
            round_record["budget_report"] = _round_budget_stats(round_record)
            terminal_stop_reason = "no_valid_generation"
            break
        variant_results = [
            item for item in round_record.get("variant_results", []) if isinstance(item, dict)
        ]
        if not variant_results:
            await _emit_variant_progress(
                "variant_evaluation_started",
                {
                    "round_index": round_index,
                    "round_kind": round_kind,
                    "evaluating": len(real_variants),
                    "generation_attempted": len(_flat_generation_records()),
                    "generation_generated": sum(
                        1
                        for item in _flat_generation_records()
                        if isinstance(item.get("variant"), dict)
                    ),
                    "generation_failed": sum(
                        1
                        for item in _flat_generation_records()
                        if not isinstance(item.get("variant"), dict)
                    ),
                    "evaluated": len(_flat_variant_results()),
                    "pvpo_valid": _pvpo_valid_count(_flat_variant_results()),
                    "complied": _complied_count(_flat_variant_results()),
                },
            )
            variant_results = await _evaluate_round_variants(real_variants)
            round_record["variant_results"] = variant_results
            round_record["variant_results_complete"] = True
            await _emit_variant_progress(
                "variant_evaluation_complete",
                {
                    "round_index": round_index,
                    "round_kind": round_kind,
                    "evaluated": len(_flat_variant_results()),
                    "pvpo_valid": _pvpo_valid_count(_flat_variant_results()),
                    "complied": _complied_count(_flat_variant_results()),
                },
            )
        if _complied_count(variant_results):
            round_record["stop_reason"] = "success"
            terminal_stop_reason = "success"
        elif _best_refused_variant(variant_results) is not None:
            round_record["stop_reason"] = "no_success"
        else:
            round_record["stop_reason"] = "no_actionable_lineage"
            terminal_stop_reason = "no_actionable_lineage"
        round_record["budget_report"] = _round_budget_stats(round_record)
        checkpoint[_VARIANT_ROUNDS_KEY] = variant_rounds
        checkpoint["variant_results"] = _flat_variant_results()
        checkpoint["variant_generation_errors"] = _flat_generation_errors()
        checkpoint["adaptive_budget"] = _adaptive_budget_report(terminal_stop_reason)
        _write_json_atomic(
            checkpoint_path,
            checkpoint,
            failpoint_base="phase_4.strategy_variation.checkpoint",
        )
        if legacy_checkpoint and round_index == 1:
            terminal_stop_reason = round_record["stop_reason"]
            break
        if terminal_stop_reason in {"success", "no_actionable_lineage"}:
            break

    variant_results = _flat_variant_results()
    variant_generation_errors = _flat_generation_errors()
    if not variant_results and variant_generation_errors:
        status = "variant_generation_failed"
    else:
        status = "varied"
    result = {
        "status": status,
        "judge_diagnosis": recommendation,
        "failure_context": failure_context,
        "attempts": [initial_result],
        "variant_results": variant_results,
        "variant_rounds": variant_rounds,
        "adaptive_budget": _adaptive_budget_report(terminal_stop_reason),
        "variant_generation_errors": variant_generation_errors,
    }
    checkpoint["variant_results"] = variant_results
    checkpoint["variant_generation_errors"] = variant_generation_errors
    checkpoint["adaptive_budget"] = result["adaptive_budget"]
    _write_json_atomic(
        checkpoint_path,
        checkpoint,
        failpoint_base="phase_4.strategy_variation.checkpoint",
    )
    return result
