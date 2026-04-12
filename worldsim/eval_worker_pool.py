"""Parallel evaluation worker pool.

Canonical source: ``docs/worldsim-v5-technical-specifcation.md`` "Parallel Evaluation" section.

Each worker is pinned to one pre-running benchmark instance and pulls tasks
from a shared ``asyncio.Queue``. Workers start staggered (``STAGGER_DELAY``
seconds apart) to avoid hammering all instances simultaneously on pool
startup.

The worker pool is phase-agnostic: it accepts a ``task_runner`` callable
``(task, agent, instance, task_dir) -> awaitable[dict]``. Phase 3 passes
its ``run_task`` for benign evaluation; Phase 4 passes ``run_adversarial_task``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from worldsim.browser_use_agent import AgentRunner
from worldsim.config import BenchmarkInstance
from worldsim.task_paths import safe_task_path_component

logger = logging.getLogger(__name__)

#: Seconds between successive worker startups. The v5 spec pins this at 5s.
STAGGER_DELAY = 5


TaskRunner = Callable[
    [dict[str, Any], AgentRunner, BenchmarkInstance, Path],
    Awaitable[dict[str, Any]],
]


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
) -> None:
    """Worker coroutine pinned to one benchmark instance.

    The worker waits ``delay`` seconds before starting (for staggered pool
    startup), creates one ``AgentRunner`` for its lifetime via
    ``agent_factory``, and repeatedly pulls tasks from ``task_queue`` until
    empty.
    """
    if delay > 0:
        await asyncio.sleep(delay)
    if stop_event.is_set():
        return

    try:
        agent = agent_factory()
        await agent.setup(instance.site_url)
    except Exception as e:  # noqa: BLE001
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
                bound_task = (
                    task_binder(task, instance)
                    if task_binder is not None
                    else task
                )
                result = await task_runner(bound_task, agent, instance, task_dir)
                async with results_lock:
                    results.append(result)
            except Exception as e:  # noqa: BLE001
                logger.exception("worker %d failed task %s: %s", worker_id, task_id, e)
                async with results_lock:
                    results.append(
                        {
                            "task_id": task_id,
                            "passed": False,
                            "outcome": "error",
                            "ecologically_valid": False,
                            "error": repr(e),
                            "message": f"worker task failed: {e}",
                            "worker_id": worker_id,
                        }
                    )
            finally:
                task_queue.task_done()
    finally:
        await agent.teardown()


async def run_eval(
    tasks: list[dict[str, Any]],
    instances: list[BenchmarkInstance],
    agent_factory: Callable[[], AgentRunner],
    task_runner: TaskRunner,
    task_dir_root: Path,
    task_binder: Callable[[dict[str, Any], BenchmarkInstance], dict[str, Any]] | None = None,
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

    Returns:
        List of result dicts in arbitrary order (workers race).
    """
    num_workers = min(len(instances), len(tasks))
    task_queue: asyncio.Queue = asyncio.Queue()
    for t in tasks:
        await task_queue.put(t)

    results: list[dict[str, Any]] = []
    results_lock = asyncio.Lock()
    stop_event = asyncio.Event()

    workers = [
        staggered_worker(
            worker_id=i,
            delay=i * STAGGER_DELAY,
            task_queue=task_queue,
            results=results,
            results_lock=results_lock,
            agent_factory=agent_factory,
            instance=instances[i],
            task_runner=task_runner,
            task_dir_root=task_dir_root,
            task_binder=task_binder,
            stop_event=stop_event,
        )
        for i in range(num_workers)
    ]
    gather_results = await asyncio.gather(*workers, return_exceptions=True)
    for i, gr in enumerate(gather_results):
        if isinstance(gr, Exception):
            logger.error("worker %d raised an unhandled exception: %s", i, gr)

    if stop_event.is_set():
        await _drain_queue_as_failures(
            task_queue,
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
                        "ecologically_valid": False,
                        "message": "task was not processed by worker pool",
                    }
                    for task in missing_tasks
                )
    return results


async def _drain_queue_as_failures(
    task_queue: asyncio.Queue,
    results: list[dict[str, Any]],
    results_lock: asyncio.Lock,
    *,
    message: str,
) -> None:
    drained: list[dict[str, Any]] = []
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
                "ecologically_valid": False,
                "message": message,
            }
        )
        task_queue.task_done()

    if drained:
        async with results_lock:
            results.extend(drained)
