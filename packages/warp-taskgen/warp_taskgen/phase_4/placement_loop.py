"""Phase 4 placement loop behavior."""

from __future__ import annotations

import asyncio
import logging
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from warp_taskgen.agent_config import bind_task_to_instance
from warp_taskgen.agent_runtime import AgentRunner
from warp_taskgen.config import BenchmarkInstance
from warp_taskgen.phase_4.aer_trajectory_extract import (
    extract_task_intent,
    extract_trajectory,
)
from warp_taskgen.phase_4.encounter_detection import EncounterResult, determine_encounter
from warp_taskgen.phase_4.metrics import (
    _gate_miss_payload,
    _null_metric_keys,
    _observational_metrics_unavailable,
    _trajectory_available_for_observational_metrics,
)
from warp_taskgen.phase_4.needham_trace import (
    format_trace_xml,
    load_trace_xml,
    write_trace_artifacts,
)
from warp_taskgen.phase_4.payload_text import _adversarial_seed_equivalent
from warp_taskgen.phase_4.placement_api import run_placement_api
from warp_taskgen.phase_4.resume import (
    _CHECKPOINT_FINGERPRINT_KEY,
    PLACEMENT_FIX_MAX_ITERATIONS,
    _load_json_dict,
    _load_saved_placement_iteration_result,
    _placement_fix_checkpoint_path,
    _placement_iteration_result_fingerprint,
    _write_placement_fix_checkpoint,
)
from warp_taskgen.phase_4.transcript_purpose_api import (
    _failure_result as _transcript_purpose_failure_result,
)
from warp_taskgen.phase_4.transcript_purpose_api import (
    run_transcript_purpose_api,
)
from warp_taskgen.phase_4.variant_eval import _merge_variant_task, _rerun_adversarial_task
from warp_taskgen.phase_4.verbalized_eval_awareness_api import (
    _failure_result as _vea_failure_result,
)
from warp_taskgen.phase_4.verbalized_eval_awareness_api import (
    run_verbalized_eval_awareness_api,
)
from warp_taskgen.site_lock import task_lock
from warp_taskgen.task_paths import safe_task_path_component

logger = logging.getLogger(__name__)


async def _run_pvpo_gate(
    task: dict[str, Any],
    task_dir: Path,
    *,
    sandbox_model: str,
) -> dict[str, Any]:
    """Run Stage 1 encounter detection plus observational awareness metrics.

    Flow:

    1. PVPO ``determine_encounter`` — unchanged.
    2. ``max_coverage == 0`` → stamp ``injection_not_encountered=True``,
       but still run Transcript Purpose and VEA over the trajectory when the
       Needham transcript can be built. Both metrics are observational only
       and never change encounter/ASR status.
    3. ``max_coverage > 0`` → fan out Transcript Purpose and VEA via
       ``asyncio.gather(return_exceptions=True)``. Neither metric gates
       anything; either failure is independent. Resume correctness is
       driven by the ``processed_result.json`` fingerprint check at
       ``_postprocess_one_task`` — no in-flight sentinel is written here.
    """
    try:
        encounter = determine_encounter(task, task_dir)
    except FileNotFoundError as exc:
        logger.warning(
            "pvpo: determine_encounter found inconsistent artifacts for %s; "
            "classifying as PVPO artifact failure: %s",
            task_dir,
            exc,
        )
        encounter = EncounterResult(
            max_coverage=0.0,
            reference_step=None,
            reference_screenshot_path=None,
            pvpo_status="artifact_inconsistent",
            pvpo_failure="missing_paired_screenshot",
            pvpo_observation="artifact_inconsistent",
        )
        payload = _observational_metrics_unavailable(
            encounter_dict=encounter.as_dict(),
            task_dir=task_dir,
            failure_class="pvpo_artifact_inconsistent",
        )
        payload["pvpo_artifact_failure"] = True
        payload["infra_failure_class"] = "infra_pvpo_artifact_inconsistent"
        return payload
    except Exception as exc:
        logger.warning(
            "pvpo: determine_encounter failed for %s; treating as no-encounter: %s",
            task_dir,
            exc,
        )
        encounter = EncounterResult(
            max_coverage=0.0,
            reference_step=None,
            reference_screenshot_path=None,
            pvpo_status="detector_failed",
            pvpo_failure="determine_encounter_exception",
        )
    encounter_dict = encounter.as_dict()
    injection_not_encountered = (
        encounter.max_coverage == 0 or encounter.reference_screenshot_path is None
    )

    # Extract the trajectory once and build one Needham XML transcript shared by
    # Transcript Purpose and VEA. Both metrics run even when PVPO reports
    # non-encounter; encounter/ASR status remains fail-closed.
    try:
        traj = extract_trajectory(task_dir)
    except (FileNotFoundError, ValueError) as exc:
        logger.warning(
            "pvpo: trajectory extract failed for %s; observational metrics unavailable: %s",
            task_dir,
            exc,
        )
        if injection_not_encountered:
            payload = _gate_miss_payload(encounter_dict)
            payload.update(_transcript_purpose_failure_result("trajectory_unavailable", task_dir))
            return payload
        return _observational_metrics_unavailable(
            encounter_dict=encounter_dict,
            task_dir=task_dir,
            failure_class="trajectory_unavailable",
        )

    if not _trajectory_available_for_observational_metrics(traj):
        logger.warning(
            "pvpo: trajectory for %s is partial; observational metrics unavailable",
            task_dir,
        )
        if injection_not_encountered:
            payload = _gate_miss_payload(encounter_dict)
            payload.update(_transcript_purpose_failure_result("trajectory_unavailable", task_dir))
            return payload
        return _observational_metrics_unavailable(
            encounter_dict=encounter_dict,
            task_dir=task_dir,
            failure_class="trajectory_unavailable",
        )

    task_instruction = extract_task_intent(task)
    try:
        needham_xml = load_trace_xml(task_dir)
        if needham_xml is None:
            needham_xml = format_trace_xml(task_instruction=task_instruction, extracted=traj)
            write_trace_artifacts(
                task_dir,
                task_instruction=task_instruction,
                extracted=traj,
            )
    except Exception as exc:
        logger.warning(
            "pvpo: Needham trace build failed for %s; observational metrics unavailable: %s",
            task_dir,
            exc,
        )
        if injection_not_encountered:
            payload = _gate_miss_payload(encounter_dict)
            payload.update(_transcript_purpose_failure_result("trajectory_unavailable", task_dir))
            return payload
        return _observational_metrics_unavailable(
            encounter_dict=encounter_dict,
            task_dir=task_dir,
            failure_class="trajectory_unavailable",
        )

    async def _run_transcript_purpose() -> dict[str, Any]:
        return await run_transcript_purpose_api(
            task,
            task_dir,
            traj,
            sandbox_model=sandbox_model,
            needham_xml=needham_xml,
        )

    async def _run_vea() -> dict[str, Any]:
        return await run_verbalized_eval_awareness_api(
            task,
            task_dir,
            traj,
            sandbox_model=sandbox_model,
            needham_xml=needham_xml,
        )

    screenshot_path = encounter.reference_screenshot_path
    screenshot_bytes = encounter.reference_screenshot_bytes
    # The encounter detector captures the PNG bytes at strict-validation
    # time; if we're missing them, the run is still a PVPO gate miss, but
    # observational metrics can classify the full transcript.
    if injection_not_encountered or screenshot_path is None or not screenshot_bytes:
        if not injection_not_encountered:
            logger.warning(
                "pvpo: reference screenshot unavailable at Stage 2 start for %s; "
                "routing to placement-fix",
                task_dir,
            )
        tp_task = asyncio.create_task(_run_transcript_purpose())
        vea_task = asyncio.create_task(_run_vea())
        try:
            tp_result_raw, vea_result_raw = await asyncio.gather(
                tp_task, vea_task, return_exceptions=True
            )
            if isinstance(tp_result_raw, BaseException):
                logger.warning("pvpo: transcript_purpose raised unexpectedly: %s", tp_result_raw)
                tp_result = _transcript_purpose_failure_result("api_error", task_dir)
            else:
                tp_result = tp_result_raw
            if isinstance(vea_result_raw, BaseException):
                logger.warning("pvpo: VEA raised unexpectedly: %s", vea_result_raw)
                vea_result = _vea_failure_result("api_error")
            else:
                vea_result = vea_result_raw
        finally:
            for metric_task in (tp_task, vea_task):
                if not metric_task.done():
                    metric_task.cancel()
            await asyncio.gather(tp_task, vea_task, return_exceptions=True)
        payload = _gate_miss_payload(encounter_dict)
        payload.update(tp_result)
        payload.update(vea_result)
        return payload

    tp_task = asyncio.create_task(_run_transcript_purpose())
    vea_task = asyncio.create_task(_run_vea())
    try:
        tp_result_raw, vea_result_raw = await asyncio.gather(
            tp_task, vea_task, return_exceptions=True
        )

        # Convert any raised exceptions into classified failure dicts so one
        # metric blowing up never aborts the other.
        if isinstance(tp_result_raw, asyncio.CancelledError):
            if not vea_task.done():
                vea_task.cancel()
                await asyncio.gather(vea_task, return_exceptions=True)
            raise tp_result_raw
        if isinstance(tp_result_raw, BaseException):
            logger.warning("pvpo: transcript_purpose raised unexpectedly: %s", tp_result_raw)
            tp_null = _null_metric_keys()
            tp_null["transcript_purpose_failure"] = "api_error"
            tp_result = {k: v for k, v in tp_null.items() if k.startswith("transcript_purpose_")}
        else:
            tp_result = tp_result_raw

        if isinstance(vea_result_raw, asyncio.CancelledError):
            if not tp_task.done():
                tp_task.cancel()
                await asyncio.gather(tp_task, return_exceptions=True)
            raise vea_result_raw
        if isinstance(vea_result_raw, BaseException):
            logger.warning("pvpo: VEA raised unexpectedly: %s", vea_result_raw)
            vea_result = {
                "verbalized_eval_awareness_classification": None,
                "verbalized_eval_awareness_is_aware": None,
                "verbalized_eval_awareness_evidence": [],
                "verbalized_eval_awareness_failure": "api_error",
                "verbalized_eval_awareness_transcript_format": "needham-xml-v1",
                "verbalized_eval_awareness_raw_response_path": None,
            }
        else:
            vea_result = vea_result_raw
    finally:
        for metric_task in (tp_task, vea_task):
            if not metric_task.done():
                metric_task.cancel()
        await asyncio.gather(tp_task, vea_task, return_exceptions=True)

    return {
        "encounter": encounter_dict,
        "injection_not_encountered": False,
        "pvpo_status": encounter_dict.get("pvpo_status"),
        "pvpo_failure": encounter_dict.get("pvpo_failure"),
        **tp_result,
        **vea_result,
    }


def _placement_fix_succeeded(
    current_result: dict[str, Any],
    *,
    triggered_by_task_broke: bool,
    triggered_by_not_encountered: bool,
) -> bool:
    """Return True iff the revised seed addressed the trigger condition.

    Placement-fix has two trigger conditions and they have different success
    criteria:

    - ``task_broke`` trigger: the revised seed must stop breaking the benign
      task. Only scoreable, non-broken outcomes count (``complied`` or
      ``refused_or_ignored``).
    - ``injection_not_encountered`` trigger: the revised seed must get the
      payload in front of the agent. Once encounter is restored, the normal
      Phase 4 tree should handle whether the attack complied or was refused.
      So any scoreable, non-broken outcome with
      ``final_status != "injection_not_encountered"`` counts.

    When both conditions triggered the loop (same run broke the task *and*
    never encountered the injection), both sub-goals must be satisfied.
    """
    outcome = current_result.get("outcome")
    final_status = current_result.get("final_status")
    if outcome not in {"complied", "refused_or_ignored"}:
        return False
    if triggered_by_task_broke and outcome == "task_broke":
        return False
    if triggered_by_not_encountered and final_status == "injection_not_encountered":
        return False
    return True


def _should_run_placement_fix(initial_result: Mapping[str, Any]) -> bool:
    """Return True only when PVPO proves the injection was not encountered."""

    return initial_result.get("final_status") == "injection_not_encountered"


async def _run_placement_fix_loop(
    task: dict[str, Any],
    initial_result: dict[str, Any],
    instance: BenchmarkInstance,
    all_instances: list[BenchmarkInstance],
    agent_factory: Callable[[], AgentRunner],
    profile_path: Path,
    task_dir_root: Path,
    benchmark_root: Path | None = None,
    sandbox_model: str = "claude-sonnet-4-6",
    site_profile: dict[str, Any] | None = None,
    resume: bool = False,
    source_fingerprint: str | None = None,
    browser_worker_semaphore: asyncio.Semaphore | None = None,
) -> dict[str, Any] | None:
    """Retry non-encountered attacks with placement-only seed fixes.

    Placement-fix is an encounter gate repair. Task-break outcomes may indicate
    reward, seed, or benign-capability issues, so they should not route through
    this loop unless PVPO reported zero paint coverage.
    """
    triggered_by_task_broke = False
    triggered_by_not_encountered = _should_run_placement_fix(initial_result)
    if not triggered_by_not_encountered:
        return None

    checkpoint_path = _placement_fix_checkpoint_path(
        task_dir_root,
        str(task.get("id", "unknown")),
    )
    attempts: list[dict[str, Any]] = [initial_result]
    current_task = task
    current_result = initial_result
    start_iteration = 0
    pending_iteration: int | None = None
    if resume and source_fingerprint is not None:
        checkpoint = _load_json_dict(checkpoint_path)
        if (
            isinstance(checkpoint, dict)
            and checkpoint.get(_CHECKPOINT_FINGERPRINT_KEY) == source_fingerprint
        ):
            completed_result = checkpoint.get("completed_result")
            if isinstance(completed_result, dict):
                return completed_result
            saved_attempts = checkpoint.get("attempts")
            if isinstance(saved_attempts, list) and all(
                isinstance(item, dict) for item in saved_attempts
            ):
                attempts = list(saved_attempts)
            saved_task = checkpoint.get("current_task")
            if isinstance(saved_task, dict):
                current_task = saved_task
            saved_result = checkpoint.get("current_result")
            if isinstance(saved_result, dict):
                current_result = saved_result
            next_iteration = checkpoint.get("next_iteration")
            if (
                isinstance(next_iteration, int)
                and 0 <= next_iteration <= PLACEMENT_FIX_MAX_ITERATIONS
            ):
                start_iteration = next_iteration
            saved_pending = checkpoint.get("pending_iteration")
            if isinstance(saved_pending, int) and 0 <= saved_pending < PLACEMENT_FIX_MAX_ITERATIONS:
                pending_iteration = saved_pending
                start_iteration = saved_pending

    def _persist_progress(
        *,
        next_iteration: int,
        pending_iteration_value: int | None,
        completed_result: dict[str, Any] | None = None,
    ) -> None:
        if source_fingerprint is None:
            return
        payload: dict[str, Any] = {
            "attempts": attempts,
            "current_task": current_task,
            "current_result": current_result,
            "next_iteration": next_iteration,
            "pending_iteration": pending_iteration_value,
        }
        if completed_result is not None:
            payload["completed_result"] = completed_result
        _write_placement_fix_checkpoint(
            checkpoint_path,
            source_fingerprint=source_fingerprint,
            payload=payload,
        )

    for iteration in range(start_iteration, PLACEMENT_FIX_MAX_ITERATIONS):
        iteration_dir = task_dir_root / safe_task_path_component(
            f"{task.get('id', 'unknown')}__placement_{iteration + 1}"
        )
        iteration_fingerprint = (
            _placement_iteration_result_fingerprint(
                current_task,
                base_source_fingerprint=source_fingerprint,
                iteration=iteration,
            )
            if source_fingerprint is not None
            else None
        )
        if pending_iteration == iteration:
            async with task_lock(bind_task_to_instance(current_task, instance, all_instances)):
                current_result = await _rerun_adversarial_task(
                    task=current_task,
                    instance=instance,
                    all_instances=all_instances,
                    agent_factory=agent_factory,
                    task_dir=iteration_dir,
                    resume=resume,
                    resume_fingerprint=iteration_fingerprint,
                    benchmark_root=benchmark_root,
                    sandbox_model=sandbox_model,
                    site_profile=site_profile,
                    browser_worker_semaphore=browser_worker_semaphore,
                )
            attempts.append(current_result)
            pending_iteration = None
            if _placement_fix_succeeded(
                current_result,
                triggered_by_task_broke=triggered_by_task_broke,
                triggered_by_not_encountered=triggered_by_not_encountered,
            ):
                completed = {
                    "status": "fixed",
                    "attempts": attempts,
                    "final_result": current_result,
                    "final_task": current_task,
                }
                _persist_progress(
                    next_iteration=iteration + 1,
                    pending_iteration_value=None,
                    completed_result=completed,
                )
                return completed
            _persist_progress(next_iteration=iteration + 1, pending_iteration_value=None)
            continue

        placement_outcome = await run_placement_api(
            current_task,
            trajectory_dir=Path(current_result.get("trajectory_dir", "")),
            sandbox_model=sandbox_model,
        )
        if placement_outcome["status"] != "ok":
            # API-side failure — couldn't get a revised seed back. Treat as
            # "no_change" so the loop exits cleanly with the failure recorded.
            completed = {
                "status": "no_change",
                "attempts": attempts,
                "final_result": current_result,
                "final_task": current_task,
                "placement_failure_class": placement_outcome.get("failure_class"),
                "placement_diagnosis": placement_outcome.get("diagnosis"),
            }
            _persist_progress(
                next_iteration=iteration,
                pending_iteration_value=None,
                completed_result=completed,
            )
            return completed
        revised_task = _merge_variant_task(current_task, placement_outcome["new_task"])
        if _adversarial_seed_equivalent(current_task, revised_task):
            completed = {
                "status": "no_change",
                "attempts": attempts,
                "final_result": current_result,
                "final_task": current_task,
            }
            _persist_progress(
                next_iteration=iteration,
                pending_iteration_value=None,
                completed_result=completed,
            )
            return completed
        current_task = revised_task
        _persist_progress(next_iteration=iteration, pending_iteration_value=iteration)
        # Wipe any leftover artefacts from a prior crashed run before re-entering.
        # Even with the placement-fix checkpoint, a stale partial rerun with no
        # reusable result.json must start from a clean iteration dir so PVPO
        # step files cannot mix old and new captures.
        if iteration_dir.exists():
            reusable = (
                iteration_fingerprint is not None
                and _load_saved_placement_iteration_result(
                    iteration_dir,
                    source_fingerprint=iteration_fingerprint,
                )
                is not None
            )
            if not reusable:
                try:
                    shutil.rmtree(iteration_dir)
                except OSError as exc:
                    logger.warning(
                        "placement-fix: could not wipe leftover iteration dir %s: %s",
                        iteration_dir,
                        exc,
                    )
        bound_task = bind_task_to_instance(current_task, instance, all_instances)
        async with task_lock(bound_task):
            current_result = await _rerun_adversarial_task(
                task=bound_task,
                instance=instance,
                all_instances=all_instances,
                agent_factory=agent_factory,
                benchmark_root=benchmark_root,
                sandbox_model=sandbox_model,
                site_profile=site_profile,
                task_dir=iteration_dir,
                resume=resume,
                resume_fingerprint=iteration_fingerprint,
                browser_worker_semaphore=browser_worker_semaphore,
            )

        attempts.append(current_result)
        pending_iteration = None
        if _placement_fix_succeeded(
            current_result,
            triggered_by_task_broke=triggered_by_task_broke,
            triggered_by_not_encountered=triggered_by_not_encountered,
        ):
            completed = {
                "status": "fixed",
                "attempts": attempts,
                "final_result": current_result,
                "final_task": current_task,
            }
            _persist_progress(
                next_iteration=iteration + 1,
                pending_iteration_value=None,
                completed_result=completed,
            )
            return completed
        _persist_progress(next_iteration=iteration + 1, pending_iteration_value=None)

    completed = {
        "status": "still_broken",
        "attempts": attempts,
        "final_result": current_result,
        "final_task": current_task,
    }
    _persist_progress(
        next_iteration=PLACEMENT_FIX_MAX_ITERATIONS,
        pending_iteration_value=None,
        completed_result=completed,
    )
    return completed
