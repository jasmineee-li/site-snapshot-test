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
from pathlib import Path
from typing import Any, Awaitable, Callable

from worldsim.browser_use_agent import AgentRunner

logger = logging.getLogger(__name__)

#: Seconds between successive worker startups. The v5 spec pins this at 5s.
STAGGER_DELAY = 5


TaskRunner = Callable[
    [dict[str, Any], AgentRunner, dict[str, Any], Path],
    Awaitable[dict[str, Any]],
]


async def staggered_worker(
    worker_id: int,
    delay: float,
    task_queue: asyncio.Queue,
    results: list[dict[str, Any]],
    results_lock: asyncio.Lock,
    agent_factory: Callable[[], AgentRunner],
    instance: dict[str, Any],
    task_runner: TaskRunner,
    task_dir_root: Path,
) -> None:
    """Worker coroutine pinned to one benchmark instance.

    The worker waits ``delay`` seconds before starting (for staggered pool
    startup), creates one ``AgentRunner`` for its lifetime via
    ``agent_factory``, and repeatedly pulls tasks from ``task_queue`` until
    empty.
    """
    if delay > 0:
        await asyncio.sleep(delay)

    agent = agent_factory()
    await agent.setup(instance["site_url"])
    try:
        while True:
            try:
                task = task_queue.get_nowait()
            except asyncio.QueueEmpty:
                return

            task_id = task.get("id", f"task_{id(task):x}")
            task_dir = task_dir_root / task_id
            task_dir.mkdir(parents=True, exist_ok=True)

            try:
                result = await task_runner(task, agent, instance, task_dir)
                async with results_lock:
                    results.append(result)
            except Exception as e:  # noqa: BLE001
                logger.exception("worker %d failed task %s: %s", worker_id, task_id, e)
                async with results_lock:
                    results.append(
                        {
                            "task_id": task_id,
                            "passed": False,
                            "error": repr(e),
                            "worker_id": worker_id,
                        }
                    )
            finally:
                task_queue.task_done()
    finally:
        await agent.teardown()


async def run_eval(
    tasks: list[dict[str, Any]],
    instances: list[dict[str, Any]],
    agent_factory: Callable[[], AgentRunner],
    task_runner: TaskRunner,
    task_dir_root: Path,
) -> list[dict[str, Any]]:
    """Distribute ``tasks`` across ``instances`` with staggered worker startup.

    Args:
        tasks: List of task dicts. Each must include an ``id`` field.
        instances: List of pre-running benchmark instance dicts.
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
        )
        for i in range(num_workers)
    ]
    await asyncio.gather(*workers)
    return results
