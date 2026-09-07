"""Sequential eval-awareness iteration for Phase 4."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from warp_taskgen.agent_runtime import AgentRunner
from warp_taskgen.config import BenchmarkInstance
from warp_taskgen.phase_4.eval_awareness_cue_diagnosis import (
    _cue_applicability_failure,
    _ecologically_valid,
    _irreconcilable_eval_awareness_contract,
    _normalize_eval_awareness_cue_diagnosis,
    _tp_requires_iteration,
    _tp_trigger_source,
)
from warp_taskgen.phase_4.eval_awareness_iteration_feedback import (
    _attempt_record,
    _contract_qa_rejection,
    _prior_iteration_feedback,
    _qa_repair_feedback,
    _rewrite_prior_attempts,
)
from warp_taskgen.phase_4.eval_awareness_iterator_budget import (
    _ITERATOR_STRATEGY,
    _STOP_REWRITE_LIMIT_REACHED,
    _STOP_TP_REGRESSION,
    _iteration_consumes_budget,
    _iteration_is_terminal,
    _iteration_progress_counts,
    _iterator_budget_report,
    _variant_runtime_stop_detail,
    build_eval_awareness_iterator_result_from_checkpoint,
)
from warp_taskgen.phase_4.eval_awareness_request_archive import RewriteRequestArchive
from warp_taskgen.phase_4.eval_awareness_tp_transition import classify_tp_transition
from warp_taskgen.phase_4.options import (
    normalize_eval_awareness_max_iterations as _normalize_eval_awareness_max_iterations,
)
from warp_taskgen.phase_4.postprocess_progress import Phase4ProgressCallback
from warp_taskgen.phase_4.resume import (
    _CHECKPOINT_FINGERPRINT_KEY,
    _PHASE_4_EVAL_AWARENESS_ITERATOR_VERSION,
    _load_json_dict,
    _phase_4_postprocess_fingerprint,
    _write_json_atomic,
)
from warp_taskgen.phase_4.variant_eval import _evaluate_variant, _merge_variant_task
from warp_taskgen.runtime_composition import RuntimeComposition
from warp_taskgen.task_paths import safe_task_path_component

logger = logging.getLogger(__name__)


_ITERATOR_CHECKPOINT = "eval_awareness_iterator_checkpoint.json"

_QA_REPAIRABLE_CLASSES = frozenset(
    {
        "unchanged_seed",
        "non_meaningful_rewrite",
        "payload_length_budget",
        "required_token_cardinality",
        "payload_text_validation",
        "attack_witness_too_weak",
        "attack_witness_missing",
        "attack_witness_too_late",
        "action_guidance_must_preserve_missing",
        "action_guidance_must_preserve_repeated",
        "action_guidance_semantic_anchors_missing",
        "precondition_slot_bridge_missing",
    }
)

_QA_REPAIR_ATTEMPTS = 1


def _eval_awareness_checkpoint_path(task_dir_root: Path, task_id: str) -> Path:
    return task_dir_root / safe_task_path_component(task_id) / _ITERATOR_CHECKPOINT


async def run_eval_awareness_iterator(
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
    max_iterations: int | None = None,
    agent_execution: dict[str, Any] | None = None,
    progress_callback: Phase4ProgressCallback | None = None,
    browser_worker_semaphore: asyncio.Semaphore | None = None,
    runtime_composition: RuntimeComposition | None = None,
) -> dict[str, Any]:
    """Run the bounded sequential eval-awareness rewrite loop."""

    _ = profile_path
    task_id = str(task.get("id", "unknown"))
    max_rewrites = _normalize_eval_awareness_max_iterations(max_iterations)

    async def _emit(event: str, data: Mapping[str, Any]) -> None:
        if progress_callback is None:
            return
        try:
            await progress_callback(
                event,
                {
                    "task_id": task_id,
                    "variant_system": "eval-awareness-iterator",
                    "max_iterations": max_rewrites,
                    **dict(data),
                },
            )
        except Exception as exc:
            logger.warning("Could not write eval-awareness progress for task %s: %s", task_id, exc)

    source_fingerprint = _phase_4_postprocess_fingerprint(
        task,
        initial_result,
        primary_instances=primary_instances,
        all_instances=all_instances,
        config_url_placeholders=config_url_placeholders,
        benchmark_root=benchmark_root,
        sandbox_model=sandbox_model,
        site_profile=site_profile,
        variant_budget_preset=None,
        variant_system="eval-awareness-iterator",
        eval_awareness_max_iterations=max_rewrites,
        agent_execution=agent_execution,
    )
    checkpoint_path = _eval_awareness_checkpoint_path(task_dir_root, task_id)
    checkpoint = _load_json_dict(checkpoint_path) if resume else None
    if checkpoint is not None and checkpoint.get(_CHECKPOINT_FINGERPRINT_KEY) != source_fingerprint:
        checkpoint = None
    if checkpoint is None:
        checkpoint = {
            _CHECKPOINT_FINGERPRINT_KEY: source_fingerprint,
            "algorithm": "eval-awareness-iterator",
            "version": _PHASE_4_EVAL_AWARENESS_ITERATOR_VERSION,
            "max_iterations": max_rewrites,
            "baseline_attempt": _attempt_record(
                iteration=0,
                kind="baseline",
                result=initial_result,
            ),
            "iterations": [],
        }
        _write_json_atomic(
            checkpoint_path,
            checkpoint,
            failpoint_base="phase_4.eval_awareness_iterator.checkpoint",
        )

    if not primary_instances:
        return {
            "status": "no_instances",
            "attempts": [initial_result],
            "variant_results": [],
            "iterations": [],
            "budget": _iterator_budget_report(
                max_iterations=max_rewrites,
                iteration_records=[],
                stop_reason="no_instances",
            ),
        }

    current_task = checkpoint.get("current_task")
    if not isinstance(current_task, dict):
        current_task = task
    current_result = checkpoint.get("current_result")
    if not isinstance(current_result, dict):
        current_result = initial_result
    iteration_records = [
        item for item in checkpoint.get("iterations", []) if isinstance(item, dict)
    ]

    stop_reason = str(checkpoint.get("stop_reason") or "")
    if not stop_reason:
        await _emit(
            "eval_awareness_iterator_started",
            {
                "generation_attempted": len(iteration_records),
                "generation_generated": sum(
                    1 for item in iteration_records if isinstance(item.get("rewrite"), dict)
                ),
                "generation_failed": sum(
                    1
                    for item in iteration_records
                    if item.get("status") in {"rewrite_failed", "rejected", _STOP_TP_REGRESSION}
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
            },
        )

    while not stop_reason:
        consumed_iterations = sum(
            1 for item in iteration_records if _iteration_consumes_budget(item)
        )
        if consumed_iterations >= max_rewrites:
            break
        if not _tp_requires_iteration(current_result):
            stop_reason = "tp_real"
            break

        record: dict[str, Any] | None = None
        if iteration_records:
            candidate = iteration_records[-1]
            if not _iteration_is_terminal(candidate):
                record = candidate
        if record is None:
            iteration = consumed_iterations + 1
            record = {
                "iteration": iteration,
                "parent_iteration": iteration - 1,
                "trigger_source": _tp_trigger_source(current_result),
                "status": "started",
            }
            iteration_records.append(record)
            checkpoint["iterations"] = iteration_records
            _write_json_atomic(
                checkpoint_path,
                checkpoint,
                failpoint_base="phase_4.eval_awareness_iterator.checkpoint",
            )
        else:
            iteration = int(
                record.get("iteration", consumed_iterations + 1) or consumed_iterations + 1
            )
            record.setdefault("parent_iteration", iteration - 1)
            record.setdefault("trigger_source", _tp_trigger_source(current_result))
            record.setdefault("status", "started")

        from warp_taskgen.phase_4.eval_awareness_cue_api import run_eval_awareness_cue_api
        from warp_taskgen.phase_4.eval_awareness_rewrite_api import (
            generate_eval_awareness_rewrite_api,
        )

        prior_attempts = _rewrite_prior_attempts(
            initial_result=initial_result,
            iteration_records=iteration_records,
        )
        prior_feedback = _prior_iteration_feedback(
            initial_result=initial_result,
            iteration_records=iteration_records,
            current_iteration=iteration,
        )

        cue = record.get("cue_diagnosis")
        if not isinstance(cue, dict):
            cue = await run_eval_awareness_cue_api(
                current_task,
                current_result,
                iteration=iteration,
                prior_attempts=prior_attempts,
                prior_feedback=prior_feedback,
                sandbox_model=sandbox_model,
            )
            if isinstance(cue, dict):
                cue = _normalize_eval_awareness_cue_diagnosis(current_task, cue)
            record["cue_diagnosis"] = cue
            checkpoint["iterations"] = iteration_records
            _write_json_atomic(
                checkpoint_path,
                checkpoint,
                failpoint_base="phase_4.eval_awareness_iterator.checkpoint",
            )

        rewrite = record.get("rewrite")
        if not isinstance(rewrite, dict):
            cue_failure = (
                _cue_applicability_failure(current_task, cue)
                if isinstance(cue, dict)
                else {
                    "failure_class": "rewrite_inapplicable_insufficient_causal_evidence",
                    "reason": "cue diagnosis was not available",
                }
            )
            irreconcilable = (
                _irreconcilable_eval_awareness_contract(current_task, cue)
                if isinstance(cue, dict)
                else None
            )
            rewrite_failure = irreconcilable or cue_failure
            if rewrite_failure is not None:
                record["status"] = "rewrite_failed"
                record["generation_error"] = rewrite_failure
                stop_reason = str(rewrite_failure.get("failure_class") or "rewrite_failed")
                await _emit(
                    "eval_awareness_iteration_stopped",
                    {
                        "iteration": iteration,
                        "stop_reason": stop_reason,
                        **_iteration_progress_counts(iteration_records),
                    },
                )
                break
            request_archive = RewriteRequestArchive(
                task_dir_root, task_id, str(current_task.get("id") or "unknown"), iteration
            )
            rewrite = await generate_eval_awareness_rewrite_api(
                current_task,
                cue,
                iteration=iteration,
                prior_attempts=prior_attempts,
                prior_feedback=prior_feedback,
                parent_result=current_result,
                sandbox_model=sandbox_model,
                request_archive=request_archive,
            )
            reference = request_archive.record_output(rewrite)
            if reference is not None:
                record.setdefault("rewrite_request_archives", []).append(reference)
            record["rewrite"] = rewrite
            checkpoint["iterations"] = iteration_records
            _write_json_atomic(
                checkpoint_path,
                checkpoint,
                failpoint_base="phase_4.eval_awareness_iterator.checkpoint",
            )

        variant_status = rewrite.get("variant_status") if isinstance(rewrite, dict) else None
        if isinstance(variant_status, dict) and variant_status.get("status") in {
            "inapplicable",
            "skipped",
            "failed",
        }:
            record["status"] = "rewrite_failed"
            reason = str(variant_status.get("reason") or "")
            failure_class = (
                "rewrite_inapplicable_irreconcilable_contract"
                if variant_status.get("status") == "inapplicable"
                and any(
                    token in reason.lower()
                    for token in ("protected", "witness", "contract", "immutable")
                )
                else "rewrite_failed"
            )
            record["generation_error"] = {"failure_class": failure_class, **variant_status}
            stop_reason = failure_class
            await _emit(
                "eval_awareness_iteration_stopped",
                {
                    "iteration": iteration,
                    "stop_reason": stop_reason,
                    **_iteration_progress_counts(iteration_records),
                },
            )
            break

        finalized = _merge_variant_task(current_task, rewrite)
        qa, generation_error = _contract_qa_rejection(current_task, rewrite, finalized)
        record["contract_qa"] = qa
        if generation_error is not None:
            repair_classes = set(generation_error.get("failure_classes") or [])
            repair_class = generation_error.get("failure_class")
            if isinstance(repair_class, str):
                repair_classes.add(repair_class)
            repairable = bool(repair_classes & _QA_REPAIRABLE_CLASSES)
            if repairable and int(record.get("qa_repair_attempts", 0) or 0) < _QA_REPAIR_ATTEMPTS:
                record.setdefault("rewrite_attempts", []).append(rewrite)
                record.setdefault("contract_qa_attempts", []).append(qa)
                record["qa_repair_attempts"] = int(record.get("qa_repair_attempts", 0) or 0) + 1
                repair_feedback = _qa_repair_feedback(
                    iteration=iteration,
                    generation_error=generation_error,
                )
                request_archive = RewriteRequestArchive(
                    task_dir_root,
                    task_id,
                    str(current_task.get("id") or "unknown"),
                    iteration,
                    repair_ordinal=record["qa_repair_attempts"],
                )
                rewrite = await generate_eval_awareness_rewrite_api(
                    current_task,
                    cue,
                    iteration=iteration,
                    prior_attempts=_rewrite_prior_attempts(
                        initial_result=initial_result,
                        iteration_records=iteration_records,
                        extra=repair_feedback,
                    ),
                    prior_feedback=prior_feedback,
                    parent_result=current_result,
                    sandbox_model=sandbox_model,
                    request_archive=request_archive,
                )
                reference = request_archive.record_output(rewrite)
                if reference is not None:
                    record.setdefault("rewrite_request_archives", []).append(reference)
                record["rewrite"] = rewrite
                variant_status = (
                    rewrite.get("variant_status") if isinstance(rewrite, dict) else None
                )
                if isinstance(variant_status, dict) and variant_status.get("status") in {
                    "inapplicable",
                    "skipped",
                    "failed",
                }:
                    record["status"] = "rewrite_failed"
                    record["generation_error"] = {
                        "failure_class": "rewrite_repair_failed",
                        **variant_status,
                    }
                    stop_reason = "rewrite_rejected"
                    await _emit(
                        "eval_awareness_iteration_stopped",
                        {
                            "iteration": iteration,
                            "stop_reason": stop_reason,
                            **_iteration_progress_counts(iteration_records),
                        },
                    )
                    break
                finalized = _merge_variant_task(current_task, rewrite)
                qa, generation_error = _contract_qa_rejection(current_task, rewrite, finalized)
                record["contract_qa"] = qa

        if generation_error is not None:
            record["status"] = "rejected"
            record["generation_error"] = generation_error
            stop_reason = "rewrite_rejected"
            await _emit(
                "eval_awareness_iteration_stopped",
                {
                    "iteration": iteration,
                    "stop_reason": stop_reason,
                    **_iteration_progress_counts(iteration_records),
                },
            )
            break

        record["finalized_task"] = finalized
        checkpoint["iterations"] = iteration_records
        _write_json_atomic(
            checkpoint_path,
            checkpoint,
            failpoint_base="phase_4.eval_awareness_iterator.checkpoint",
        )

        result = await _evaluate_variant(
            task=current_task,
            variant=finalized,
            instance=primary_instances[(iteration - 1) % len(primary_instances)],
            all_instances=all_instances,
            strategy=_ITERATOR_STRATEGY,
            index=iteration,
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
        result.setdefault("iteration", iteration)
        result.setdefault("round_index", iteration)
        result.setdefault("round_kind", "eval_awareness_iteration")
        result.setdefault("round_variant_index", 0)
        result.setdefault("global_variant_index", iteration)
        result.setdefault("parent_global_variant_index", iteration - 1 if iteration > 1 else None)
        result.setdefault("strategy", _ITERATOR_STRATEGY["strategy"])
        tp_transition = classify_tp_transition(
            current_result.get("transcript_purpose_classification"),
            result.get("transcript_purpose_classification"),
        )
        result.setdefault("transcript_purpose_transition", tp_transition)
        record["tp_transition"] = tp_transition
        record["result"] = result
        if tp_transition.get("direction") == "regressed":
            record["status"] = _STOP_TP_REGRESSION
            record["generation_error"] = {
                "failure_class": _STOP_TP_REGRESSION,
                "reason": (
                    "eval-awareness rewrite increased Transcript Purpose evaluation salience"
                ),
                "tp_transition": tp_transition,
            }
            checkpoint["iterations"] = iteration_records
            _write_json_atomic(
                checkpoint_path,
                checkpoint,
                failpoint_base="phase_4.eval_awareness_iterator.checkpoint",
            )
            if (
                sum(1 for item in iteration_records if _iteration_consumes_budget(item))
                >= max_rewrites
            ):
                stop_reason = _STOP_TP_REGRESSION
                await _emit(
                    "eval_awareness_iteration_stopped",
                    {
                        "iteration": iteration,
                        "stop_reason": stop_reason,
                        **_iteration_progress_counts(iteration_records),
                    },
                )
                break
            await _emit(
                "eval_awareness_iteration_complete",
                {
                    "iteration": iteration,
                    "tp_regression": True,
                    **_iteration_progress_counts(iteration_records),
                },
            )
            continue
        record["status"] = "evaluated"
        current_task = finalized
        current_result = result
        checkpoint["iterations"] = iteration_records
        checkpoint["current_task"] = current_task
        checkpoint["current_result"] = current_result
        _write_json_atomic(
            checkpoint_path,
            checkpoint,
            failpoint_base="phase_4.eval_awareness_iterator.checkpoint",
        )

        await _emit(
            "eval_awareness_iteration_complete",
            {
                "iteration": iteration,
                "generation_attempted": len(iteration_records),
                "generation_generated": sum(
                    1 for item in iteration_records if isinstance(item.get("rewrite"), dict)
                ),
                "generation_failed": sum(
                    1
                    for item in iteration_records
                    if item.get("status") in {"rewrite_failed", "rejected", _STOP_TP_REGRESSION}
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
            },
        )

        if result.get("outcome") == "task_broke":
            stop_reason = "task_broke"
            record["status"] = stop_reason
            record["generation_error"] = _variant_runtime_stop_detail(
                stop_reason,
                result,
            )
            break
        if result.get("final_status") == "injection_not_encountered":
            stop_reason = "lost_pvpo_encounter"
            record["status"] = stop_reason
            record["generation_error"] = _variant_runtime_stop_detail(
                stop_reason,
                result,
            )
            break
        if (
            result.get("outcome") == "complied"
            and result.get("transcript_purpose_classification") == "Real"
        ):
            stop_reason = "tp_real_and_complied"
            break

    if not stop_reason:
        stop_reason = (
            _STOP_REWRITE_LIMIT_REACHED if _tp_requires_iteration(current_result) else "tp_real"
        )

    result = build_eval_awareness_iterator_result_from_checkpoint(
        initial_result=initial_result,
        checkpoint={**checkpoint, "iterations": iteration_records, "stop_reason": stop_reason},
        max_iterations=max_rewrites,
    )
    if result is None:
        result = {
            "status": "rewrite_failed",
            "attempts": [initial_result],
            "variant_results": [],
            "iterations": iteration_records,
            "stop_reason": "checkpoint_unavailable",
        }
    checkpoint.update(
        {
            "stop_reason": stop_reason,
            "selected_iteration": result.get("selected_iteration"),
            "selection_reason": result.get("selection_reason"),
            "budget": result.get("budget"),
            "variant_results": result.get("variant_results"),
            "generation_errors": result.get("generation_errors"),
        }
    )
    _write_json_atomic(
        checkpoint_path,
        checkpoint,
        failpoint_base="phase_4.eval_awareness_iterator.checkpoint",
    )
    return result


__all__ = ["build_eval_awareness_iterator_result_from_checkpoint", "run_eval_awareness_iterator"]
