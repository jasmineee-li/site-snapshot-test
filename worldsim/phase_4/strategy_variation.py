"""Phase 4 strategy variation behavior."""
# ruff: noqa: F821

from __future__ import annotations

from worldsim.phase_4._context import install_context
from worldsim.phase_4.postprocess_progress import Phase4ProgressCallback

install_context(globals())


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
) -> dict[str, Any]:
    """Generate a variant adversarial task following a specific strategy.

    Thin wrapper over `worldsim.phase_4.variant_api.generate_variant_api`,
    which performs a single-turn Anthropic Messages API call with forced
    tool-use structured output (tool `build_variant`). Only
    `adversarial_data_seed` is modified; `instruction`, `reward_function`,
    `delivery_channel`, and `required_tokens` remain byte-identical to the
    base task. `profile_path` is accepted for signature compatibility but
    not forwarded; the API call needs only the task + strategy.
    """
    from worldsim.phase_4.variant_api import generate_variant_api

    return await generate_variant_api(task, strategy, sandbox_model=sandbox_model)


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
    progress_callback: Phase4ProgressCallback | None = None,
) -> dict[str, Any]:
    """Adaptive strategy variation: judge -> generate variants -> evaluate.

    One round per task. Fan-out of up to 3 variants based on judge's
    recommended strategies.
    """
    task_id = str(task.get("id", "unknown"))

    async def _emit_variant_progress(event: str, data: Mapping[str, Any]) -> None:
        if progress_callback is None:
            return
        await progress_callback(event, data)

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
    )
    checkpoint = _load_json_dict(checkpoint_path) if resume else None
    if checkpoint is not None and checkpoint.get(_CHECKPOINT_FINGERPRINT_KEY) != source_fingerprint:
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

    # 2. Generate variants in parallel (up to 3 Modal Sandboxes)
    selected_strategies = strategies[:3]
    (
        variant_candidates,
        variant_generation_errors,
        generation_records,
        completed_indexes,
    ) = _rebuild_variant_generation_progress(
        task,
        checkpoint,
        selected_strategies=selected_strategies,
    )
    await _emit_variant_progress(
        "variant_round_started",
        {
            "round_index": 1,
            "round_kind": "initial",
            "planned": len(selected_strategies),
            "generation_attempted": len(generation_records),
            "generation_generated": len(
                [item for item in variant_candidates if isinstance(item.get("variant"), dict)]
            ),
        },
    )
    pending_strategies = [
        (index, strategy)
        for index, strategy in enumerate(selected_strategies)
        if index not in completed_indexes
    ]
    if pending_strategies:
        checkpoint = checkpoint or {
            _CHECKPOINT_FINGERPRINT_KEY: source_fingerprint,
            "judge_diagnosis": recommendation,
        }

        async def _generate_variant_record(
            index: int,
            strategy: dict[str, Any],
        ) -> dict[str, Any]:
            strategy_name = strategy.get("strategy", f"strategy_{index}")
            try:
                variant = await generate_variant(
                    task,
                    strategy,
                    profile_path,
                    sandbox_model=sandbox_model,
                )
            except Exception as exc:
                logger.error(
                    "Variant generation failed for task %s strategy %s: %s",
                    task_id,
                    strategy_name,
                    exc,
                )
                return _variant_generation_record_for_result(
                    index=index,
                    strategy=strategy,
                    error=repr(exc),
                )
            variant_status = variant.get("variant_status") if isinstance(variant, dict) else None
            if isinstance(variant_status, dict) and variant_status.get("status") in {
                "inapplicable",
                "skipped",
                "failed",
            }:
                logger.info(
                    "Variant %s for task %s marked %s: %s",
                    strategy_name,
                    task_id,
                    variant_status.get("status"),
                    variant_status.get("reason", ""),
                )
                return _variant_generation_record_for_result(
                    index=index,
                    strategy=strategy,
                    status=str(variant_status.get("status")),
                    reason=str(variant_status.get("reason", "")),
                )
            if isinstance(variant, dict) and _variant_changes_seed(task, variant):
                return _variant_generation_record_for_result(
                    index=index,
                    strategy=strategy,
                    variant=variant,
                )
            return _variant_generation_record_for_result(
                index=index,
                strategy=strategy,
                status="bookkeeping_only",
            )

        pending_tasks = [
            asyncio.create_task(_generate_variant_record(index, strategy))
            for index, strategy in pending_strategies
        ]
        for pending_task in asyncio.as_completed(pending_tasks):
            record = await pending_task
            generation_records.append(record)
            (
                variant_candidates,
                variant_generation_errors,
                generation_records,
                completed_indexes,
            ) = _rebuild_variant_generation_progress(
                task,
                {
                    _VARIANT_GENERATION_RECORDS_KEY: generation_records,
                },
                selected_strategies=selected_strategies,
            )
            checkpoint[_VARIANT_GENERATION_RECORDS_KEY] = generation_records
            checkpoint["variant_candidates"] = variant_candidates
            checkpoint["variant_generation_errors"] = variant_generation_errors
            _write_json_atomic(
                checkpoint_path,
                checkpoint,
                failpoint_base="phase_4.strategy_variation.checkpoint",
            )

    real_variants = [
        (item.get("variant"), item.get("strategy"))
        for item in variant_candidates
        if isinstance(item, dict)
        and isinstance(item.get("variant"), dict)
        and isinstance(item.get("strategy"), dict)
    ]
    await _emit_variant_progress(
        "variant_generation_recorded",
        {
            "round_index": 1,
            "round_kind": "initial",
            "generation_attempted": len(generation_records),
            "generation_generated": len(real_variants),
            "generation_failed": len(generation_records) - len(real_variants),
        },
    )
    if not real_variants:
        return {
            "status": "variant_generation_failed",
            "judge_diagnosis": recommendation,
            "attempts": [initial_result],
            "variant_results": [],
            "variant_generation_errors": variant_generation_errors,
        }

    # 3. Evaluate variants in parallel, one per separate benchmark instance.
    limited_variants = real_variants[: len(primary_instances)]
    partial_capacity = len(limited_variants) < len(real_variants)
    if partial_capacity:
        logger.warning(
            "Only %d/%d strategy variants for task %s can be evaluated because only %d instances are available",
            len(limited_variants),
            len(real_variants),
            task.get("id", "?"),
            len(primary_instances),
        )
    await _emit_variant_progress(
        "variant_evaluation_started",
        {
            "round_index": 1,
            "round_kind": "initial",
            "evaluating": len(limited_variants),
            "partial_capacity": partial_capacity,
        },
    )
    variant_results = await asyncio.gather(
        *[
            _evaluate_variant(
                task=task,
                variant=variant,
                instance=primary_instances[i],
                all_instances=all_instances,
                strategy=strategy,
                index=i,
                agent_factory=agent_factory,
                task_dir_root=task_dir_root,
                config_url_placeholders=config_url_placeholders,
                resume=resume,
                benchmark_root=benchmark_root,
                sandbox_model=sandbox_model,
                site_profile=site_profile,
            )
            for i, (variant, strategy) in enumerate(limited_variants)
        ]
    )
    pvpo_valid = sum(
        1
        for variant_result in variant_results
        if isinstance(variant_result, Mapping)
        and variant_result.get("final_status") not in {"injection_not_encountered", "error"}
    )
    complied = sum(
        1
        for variant_result in variant_results
        if isinstance(variant_result, Mapping) and variant_result.get("final_status") == "complied"
    )
    await _emit_variant_progress(
        "variant_evaluation_complete",
        {
            "round_index": 1,
            "round_kind": "initial",
            "evaluated": len(variant_results),
            "pvpo_valid": pvpo_valid,
            "complied": complied,
        },
    )
    result = {
        "status": "partial_capacity" if partial_capacity else "varied",
        "judge_diagnosis": recommendation,
        "attempts": [initial_result],
        "variant_results": variant_results,
        "variant_generation_errors": variant_generation_errors,
    }
    if partial_capacity:
        result["skipped_strategies"] = [
            strategy.get("strategy")
            for _, strategy in real_variants[len(primary_instances) :]
            if isinstance(strategy, dict)
        ]
    checkpoint = checkpoint or {
        _CHECKPOINT_FINGERPRINT_KEY: source_fingerprint,
        "judge_diagnosis": recommendation,
    }
    checkpoint["variant_results"] = variant_results
    _write_json_atomic(
        checkpoint_path,
        checkpoint,
        failpoint_base="phase_4.strategy_variation.checkpoint",
    )
    return result
