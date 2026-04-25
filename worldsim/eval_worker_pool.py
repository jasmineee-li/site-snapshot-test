"""Parallel evaluation worker pool.

Canonical source: ``docs/worldsim-v5-technical-specifcation.md`` "Parallel Evaluation" section.

Each worker is pinned to one pre-running benchmark instance. Tasks are
deterministically partitioned onto per-instance queues before startup so the
same task routes to the same replica across reruns and resume paths. Workers
still start staggered (``STAGGER_DELAY`` seconds apart) to avoid hammering all
instances simultaneously on pool startup.

The worker pool is phase-agnostic: it accepts a ``task_runner`` callable
``(task, agent, instance, task_dir) -> awaitable[dict]``. Phase 3 passes
its ``run_task`` for benign evaluation; Phase 4 passes ``run_adversarial_task``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from worldsim.browser_use_agent import AgentRunner
from worldsim.config import BenchmarkInstance
from worldsim.instance_selection import ordered_instances, stable_index_for_task
from worldsim.placeholders import normalize_site_name
from worldsim.resume_metadata import RESULT_FINGERPRINT_KEY
from worldsim.task_paths import safe_task_path_component

logger = logging.getLogger(__name__)

#: Seconds between successive worker startups. The v5 spec pins this at 5s.
STAGGER_DELAY = 5
_MAX_WORKERS_ENV = "WORLDSIM_MAX_WORKERS"
_GLOBAL_WORKER_SEMAPHORES: dict[tuple[int, int], asyncio.Semaphore] = {}

_OUTCOME_RESULT_EXTRA_KEYS: tuple[str, ...] = (
    "error",
    "final_status",
    "seed_preflight_mismatches",
    "seed_preflight_mismatch_records",
    "pvpo_status",
    "pvpo_failure",
    "transcript_purpose_classification",
    "transcript_purpose_is_aware",
    "transcript_purpose_guess",
    "transcript_purpose_explanation",
    "transcript_purpose_failure",
    "transcript_purpose_format",
    "transcript_purpose_attempts",
    "transcript_purpose_raw_response_path",
    "verbalized_eval_awareness_classification",
    "verbalized_eval_awareness_is_aware",
    "verbalized_eval_awareness_evidence",
    "verbalized_eval_awareness_failure",
    "verbalized_eval_awareness_transcript_format",
    "verbalized_eval_awareness_raw_response_path",
    "outcome_fine",
    "flags",
    "diagnosable_confidence",
    "signals",
    "classifier_version",
)


TaskRunner = Callable[
    [dict[str, Any], AgentRunner, BenchmarkInstance, Path],
    Awaitable[dict[str, Any]],
]


def _normalize_completed_result(data: dict[str, Any], *, trajectory_dir: Path) -> dict[str, Any]:
    """Project a saved result sentinel back to the live task-runner shape."""
    task_id = str(data.get("task_id") or "")
    if any(key in data for key in ("outcome", "encounter")):
        outcome = data.get("outcome")
        normalized: dict[str, Any] = {
            "task_id": task_id,
            "trajectory_dir": str(trajectory_dir),
        }
        for key in ("outcome", "encounter", "elapsed", "steps"):
            if key in data:
                normalized[key] = data.get(key)
        for key in _OUTCOME_RESULT_EXTRA_KEYS:
            if key in data:
                normalized[key] = data.get(key)
        saved_passed = data.get("passed") if "passed" in data else None
        if outcome == "error":
            error = data.get("error") or data.get("message")
            if error is not None:
                normalized["error"] = error
            normalized["benign_passed"] = bool(saved_passed) if saved_passed is not None else False
            normalized["adversarial_passed"] = False
        else:
            if saved_passed is not None:
                normalized["benign_passed"] = bool(saved_passed)
            else:
                normalized["benign_passed"] = outcome != "task_broke"
            normalized["adversarial_passed"] = outcome == "complied"
        return normalized

    normalized = {
        "task_id": task_id,
        "trajectory_dir": str(trajectory_dir),
    }
    for key in ("passed", "message", "elapsed", "steps", "is_done"):
        if key in data:
            normalized[key] = data.get(key)
    for key in (
        "triage_decision",
        "triage_likely_root_cause",
        "triage_confidence",
        "triage_reason",
        "triage_source",
        "triage_escalate",
        "triage_cache_version",
    ):
        if key in data:
            normalized[key] = data.get(key)
    return normalized


def load_completed_results(
    task_dir_root: Path,
    *,
    expected_fingerprints: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Scan task_dir_root for existing result.json sentinel files.

    Returns a dict mapping task_id -> result dict for tasks that completed
    in a prior run. Tasks with history.json but no result.json are treated
    as incomplete (crashed mid-execution) and are not included, so they
    will re-run on resume.

    Only loads results from directories whose name matches the task_id's
    canonical path component. This filters out variant, ecological-fix,
    and placement-fix result.json files (which use suffixed directory
    names like ``<task_id>_variant_0``) to avoid dict-key collision
    with the initial run's result for the same task_id.

    Also reconstructs ``trajectory_dir`` from the subdirectory path so
    downstream code can locate trajectory artifacts on resume.
    """
    completed: dict[str, dict[str, Any]] = {}
    if not task_dir_root.exists():
        return completed
    for subdir in task_dir_root.iterdir():
        if not subdir.is_dir():
            continue
        result_file = subdir / "result.json"
        if not result_file.exists():
            continue
        try:
            data = json.loads(result_file.read_text())
            if not isinstance(data, dict):
                logger.warning("Invalid result file at %s, expected JSON object", result_file)
                continue
            task_id = str(data.get("task_id") or "")
            if not task_id:
                continue
            # Only load initial-run results. Variant/rerun directories
            # have suffixed names (e.g., task_42_variant_0, task_42__ecoval_1)
            # that won't match the canonical path for the bare task_id.
            if subdir.name != safe_task_path_component(task_id):
                continue
            if expected_fingerprints is not None:
                expected_fingerprint = expected_fingerprints.get(task_id)
                actual_fingerprint = data.get(RESULT_FINGERPRINT_KEY)
                if not expected_fingerprint or actual_fingerprint != expected_fingerprint:
                    logger.info(
                        "Ignoring stale result for task %s at %s due to resume fingerprint mismatch",
                        task_id,
                        result_file,
                    )
                    continue
            if not _has_required_resume_artifacts(data, trajectory_dir=subdir):
                logger.info(
                    "Ignoring incomplete result for task %s at %s due to missing or malformed sidecars",
                    task_id,
                    result_file,
                )
                continue
            completed[task_id] = _normalize_completed_result(data, trajectory_dir=subdir)
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt result file at %s, will re-run", result_file)
    return completed


def _has_required_resume_artifacts(data: dict[str, Any], *, trajectory_dir: Path) -> bool:
    """Return True iff resume can safely trust the saved result sentinel.

    Phase 4 only needs ``history.json`` when postprocess must re-enter
    placement-fix or strategy-variation. Deterministic preflight failures and
    terminal outcomes (``error`` / ``complied``) remain reusable from
    ``result.json`` alone.
    """
    outcome = data.get("outcome")
    if outcome is None or outcome in {"seed_preflight_mismatch", "error", "complied"}:
        return True
    for name in ("history.json",):
        path = trajectory_dir / name
        if not path.exists():
            return False
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return False
        if not isinstance(payload, (dict, list)):
            return False
    return True


async def staggered_worker(
    worker_id: int,
    delay: float,
    task_queue: asyncio.Queue,
    results: list[dict[str, Any]],
    results_lock: asyncio.Lock,
    agent_factory: Callable[[], AgentRunner],
    instance: BenchmarkInstance,
    task_runner: TaskRunner,
    task_dir_root: Path,
    task_binder: Callable[[dict[str, Any], BenchmarkInstance], dict[str, Any]] | None,
    stop_event: asyncio.Event,
    worker_semaphore: asyncio.Semaphore | None = None,
) -> None:
    """Worker coroutine pinned to one benchmark instance.

    The worker waits ``delay`` seconds before starting, creates one
    ``AgentRunner`` for its lifetime via ``agent_factory``, and processes
    the deterministic task subset pre-assigned to its instance.
    """
    if delay > 0:
        await asyncio.sleep(delay)
    if stop_event.is_set():
        return

    async def _run_worker_lifetime() -> None:
        try:
            agent = agent_factory()
            await agent.setup(instance.site_url)
        except Exception as e:
            logger.exception("worker %d failed during setup: %s", worker_id, e)
            stop_event.set()
            return

        try:
            while True:
                if stop_event.is_set():
                    return
                try:
                    task = task_queue.get_nowait()
                except asyncio.QueueEmpty:
                    return

                task_id = str(task.get("id", f"task_{id(task):x}"))
                task_dir = task_dir_root / safe_task_path_component(task_id)
                task_dir.mkdir(parents=True, exist_ok=True)

                try:
                    bound_task = task_binder(task, instance) if task_binder is not None else task
                    result = await task_runner(bound_task, agent, instance, task_dir)
                    async with results_lock:
                        results.append(result)
                except Exception as e:
                    logger.exception("worker %d failed task %s: %s", worker_id, task_id, e)
                    async with results_lock:
                        results.append(
                            {
                                "task_id": task_id,
                                "passed": False,
                                "outcome": "error",
                                "error": repr(e),
                                "message": f"worker task failed: {e}",
                                "worker_id": worker_id,
                            }
                        )
                finally:
                    task_queue.task_done()
        finally:
            await agent.teardown()

    if worker_semaphore is not None:
        logger.info("worker %d waiting for global worker slot", worker_id)
        async with worker_semaphore:
            logger.info("worker %d acquired global worker slot", worker_id)
            await _run_worker_lifetime()
        return

    await _run_worker_lifetime()


def _resolve_max_workers(max_workers: int | None = None) -> int | None:
    if max_workers is not None:
        if max_workers <= 0:
            raise ValueError("max_workers must be positive when provided")
        return max_workers
    raw = os.environ.get(_MAX_WORKERS_ENV, "").strip()
    if not raw:
        return None
    try:
        resolved = int(raw)
    except ValueError as exc:
        raise ValueError(f"{_MAX_WORKERS_ENV} must be a positive integer, got {raw!r}") from exc
    if resolved <= 0:
        raise ValueError(f"{_MAX_WORKERS_ENV} must be a positive integer, got {raw!r}")
    return resolved


def _global_worker_semaphore(max_workers: int | None) -> asyncio.Semaphore | None:
    if max_workers is None:
        return None
    loop = asyncio.get_running_loop()
    key = (max_workers, id(loop))
    semaphore = _GLOBAL_WORKER_SEMAPHORES.get(key)
    if semaphore is None:
        semaphore = asyncio.Semaphore(max_workers)
        _GLOBAL_WORKER_SEMAPHORES[key] = semaphore
    return semaphore


async def run_eval(
    tasks: list[dict[str, Any]],
    instances: list[BenchmarkInstance],
    agent_factory: Callable[[], AgentRunner],
    task_runner: TaskRunner,
    task_dir_root: Path,
    task_binder: Callable[[dict[str, Any], BenchmarkInstance], dict[str, Any]] | None = None,
    resume: bool = False,
    expected_result_fingerprints: dict[str, str] | None = None,
    max_workers: int | None = None,
) -> list[dict[str, Any]]:
    """Distribute ``tasks`` across ``instances`` with staggered worker startup.

    Args:
        tasks: List of task dicts. Each must include an ``id`` field.
        instances: List of pre-running benchmark instances.
            ``len(instances)`` caps the worker count.
        agent_factory: Zero-arg callable returning a fresh ``AgentRunner``
            per worker. Called once per worker.
        task_runner: Per-task async callable
            ``(task, agent, instance, task_dir) -> dict``. Phase 3 uses
            ``run_task``; Phase 4 uses ``run_adversarial_task``.
        task_dir_root: Directory under which per-task subdirectories are
            created (e.g. ``logs/phase_3/<timestamp>/``).
        resume: If True, scan task_dir_root for existing result.json files
            and skip tasks that already completed in a prior run.

    Returns:
        List of result dicts in arbitrary order (workers race).
    """
    resolved_max_workers = _resolve_max_workers(max_workers)
    worker_semaphore = _global_worker_semaphore(resolved_max_workers)
    if resolved_max_workers is not None:
        logger.info("Worker pool global concurrency cap: %d", resolved_max_workers)

    # On resume, load prior results and filter out completed tasks.
    prior_results: list[dict[str, Any]] = []
    if resume:
        completed = load_completed_results(
            task_dir_root,
            expected_fingerprints=expected_result_fingerprints,
        )
        if completed:
            original_count = len(tasks)
            tasks = [t for t in tasks if str(t.get("id", "")) not in completed]
            prior_results = list(completed.values())
            logger.info(
                "Resume: %d/%d tasks already completed, %d remaining",
                len(prior_results),
                original_count,
                len(tasks),
            )
    if not tasks:
        logger.info("All tasks already completed in prior run")
        return prior_results

    ordered = ordered_instances(instances)
    if not ordered:
        logger.error("No benchmark instances available for evaluation")
        failed = [
            {
                "task_id": str(task.get("id", f"task_{id(task):x}")),
                "passed": False,
                "outcome": "error",
                "message": "no benchmark instances configured for worker pool",
            }
            for task in tasks
        ]
        return prior_results + failed

    site_names = {
        normalize_site_name(instance.site_name)
        for instance in ordered
        if normalize_site_name(instance.site_name)
    }
    if len(site_names) > 1:
        logger.error("Worker pool requires same-site instances, got %s", sorted(site_names))
        failed = [
            {
                "task_id": str(task.get("id", f"task_{id(task):x}")),
                "passed": False,
                "outcome": "error",
                "message": (
                    f"worker pool requires same-site instances; got {', '.join(sorted(site_names))}"
                ),
            }
            for task in tasks
        ]
        return prior_results + failed

    replica_count = len(ordered)
    task_queues: list[asyncio.Queue] = [asyncio.Queue() for _ in range(replica_count)]
    instance_site = normalize_site_name(ordered[0].site_name)
    for task in tasks:
        worker_index = _worker_index_for_task(
            task,
            replica_count=replica_count,
            site_name=instance_site,
        )
        task_queues[worker_index].put_nowait(task)

    results: list[dict[str, Any]] = []
    results_lock = asyncio.Lock()
    stop_event = asyncio.Event()

    worker_specs = [
        (worker_index, ordered[worker_index], task_queues[worker_index])
        for worker_index in range(replica_count)
        if not task_queues[worker_index].empty()
    ]
    workers = [
        staggered_worker(
            worker_id=worker_index,
            delay=launch_index * STAGGER_DELAY,
            task_queue=task_queue,
            results=results,
            results_lock=results_lock,
            agent_factory=agent_factory,
            instance=instance,
            task_runner=task_runner,
            task_dir_root=task_dir_root,
            task_binder=task_binder,
            stop_event=stop_event,
            worker_semaphore=worker_semaphore,
        )
        for launch_index, (worker_index, instance, task_queue) in enumerate(worker_specs)
    ]
    gather_results = await asyncio.gather(*workers, return_exceptions=True)
    for i, gr in enumerate(gather_results):
        if isinstance(gr, Exception):
            logger.error("worker %d raised an unhandled exception: %s", i, gr)

    if stop_event.is_set():
        await _drain_queues_as_failures(
            task_queues,
            results,
            results_lock,
            message="worker setup failed before queued task could start",
        )

    if len(results) < len(tasks):
        seen_ids = {str(result.get("task_id")) for result in results}
        missing_tasks = [
            task for task in tasks if str(task.get("id", f"task_{id(task):x}")) not in seen_ids
        ]
        if missing_tasks:
            async with results_lock:
                results.extend(
                    {
                        "task_id": str(task.get("id", f"task_{id(task):x}")),
                        "passed": False,
                        "outcome": "error",
                        "message": "task was not processed by worker pool",
                    }
                    for task in missing_tasks
                )
    return prior_results + results


def _worker_index_for_task(
    task: dict[str, Any],
    *,
    replica_count: int,
    site_name: str,
) -> int:
    return stable_index_for_task(task, replica_count, salt=site_name)


async def _drain_queues_as_failures(
    task_queues: list[asyncio.Queue],
    results: list[dict[str, Any]],
    results_lock: asyncio.Lock,
    *,
    message: str,
) -> None:
    drained: list[dict[str, Any]] = []
    for task_queue in task_queues:
        while True:
            try:
                task = task_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            drained.append(
                {
                    "task_id": str(task.get("id", f"task_{id(task):x}")),
                    "passed": False,
                    "outcome": "error",
                    "message": message,
                }
            )
            task_queue.task_done()

    if drained:
        async with results_lock:
            results.extend(drained)
