"""Cooperative admission for Phase 2b text-fill Atomic Work Units."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path

from worldsim.run_control import PauseBoundaryReached, pause_control_lock, pause_requested
from worldsim.state import get_state_dir


async def run_text_fill_units[T, R](
    items: Sequence[T],
    operation: Callable[[T], Awaitable[R]],
    *,
    concurrency: int,
    state_dir: Path | None = None,
) -> list[R | BaseException]:
    """Run admitted text-fill units to checkpoint completion.

    Claiming an item and observing the pause marker are serialized by the
    request lock.  A request that wins that lock prevents every later claim;
    workers that already hold a claim are allowed to finish their operation,
    including the feature-owned checkpoint write, before the boundary is
    reported to the outer lifecycle adapter.
    """

    if concurrency <= 0:
        raise ValueError("Phase 2b text-fill concurrency must be positive")
    root = (state_dir or get_state_dir()).expanduser().resolve(strict=False)
    queue: asyncio.Queue[tuple[int, T]] = asyncio.Queue()
    for index, item in enumerate(items):
        queue.put_nowait((index, item))
    missing = object()
    results: list[object] = [missing] * len(items)
    paused = asyncio.Event()
    claim_lock = asyncio.Lock()

    async def worker() -> None:
        while True:
            async with claim_lock:
                with pause_control_lock(root):
                    if pause_requested(root):
                        paused.set()
                        return
                    try:
                        index, item = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
            try:
                try:
                    results[index] = await operation(item)
                except PauseBoundaryReached:
                    # An operation must not normally raise this marker, but
                    # preserving it makes the boundary fail closed for custom
                    # callers instead of claiming another unit.
                    paused.set()
                    return
                except Exception as exc:
                    results[index] = exc
            finally:
                queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(min(concurrency, len(items)))]
    if workers:
        await asyncio.gather(*workers)
    if paused.is_set():
        # A failed admitted operation has no valid Atomic Work Unit.  Do not
        # let the outer adapter record a false ``paused`` state; continuation
        # will rerun the failed unit from its missing/invalid checkpoint.
        failures = [
            result
            for result in results
            if result is not missing and isinstance(result, BaseException)
        ]
        if failures:
            raise RuntimeError(
                "Phase 2b text-fill unit failed before checkpoint write"
            ) from failures[0]
        raise PauseBoundaryReached()
    if any(result is missing for result in results):
        raise RuntimeError("Phase 2b text-fill scheduler lost a unit")
    return [result for result in results if result is not missing]  # type: ignore[misc]


__all__ = ["run_text_fill_units"]
