"""Cooperative admission for Phase 2c verification work units.

Phase 2c has one pause boundary after source-data preflight and one atomic
work unit per verification task.  This module owns only the feature-specific
claim/drain protocol; :mod:`worldsim.run_control` owns the request marker and
the eventual lifecycle transition.

The claim lock and the state-root pause lock are deliberately acquired in
that order.  A request that wins the state-root lock therefore prevents a
later claim, while a task that already completed the claim is allowed to run
through seed, render/readback/reachability, cleanup, and its checkpoint write.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from contextlib import contextmanager
from pathlib import Path

from worldsim.run_control import PauseBoundaryReached, pause_control_lock, pause_requested
from worldsim.state import get_state_dir


def _state_root(state_dir: Path | None) -> Path:
    return (state_dir or get_state_dir()).expanduser().resolve(strict=False)


def assert_preflight_boundary(state_dir: Path | None = None) -> None:
    """Stop before the first verification claim when setup observed a request.

    Source-data preflight is intentionally one bounded setup operation.  It
    has no per-task checkpoints and is allowed to drain even after an operator
    requests a pause.  The caller invokes this immediately after preflight,
    before constructing any verification workers.
    """

    # Direct/legacy verifier callers do not participate in lifecycle control.
    # The phase stage supplies an explicit state root when pause semantics are
    # active; do not inspect the process-wide default root implicitly.
    if state_dir is None:
        return
    root = _state_root(state_dir)
    with pause_control_lock(root):
        if pause_requested(root):
            raise PauseBoundaryReached()


async def run_verification_units[T, R](
    items: Sequence[T],
    operation: Callable[[T], Awaitable[R]],
    *,
    concurrency: int,
    state_dir: Path | None = None,
) -> list[R]:
    """Drain admitted Phase 2c units and stop claiming after a pause.

    Every claim is serialized with the non-secret pause request marker.  The
    operation must include all site effects, evidence/readback, cleanup, and
    the feature-owned atomic checkpoint write; this scheduler never cancels
    an admitted operation.  A pause request is raised only after all admitted
    operations have returned.  A final lock-protected marker check covers a
    request that arrives after the last claim but before aggregate promotion.
    """

    if concurrency <= 0:
        raise ValueError("Phase 2c verification concurrency must be positive")

    pause_enabled = state_dir is not None
    root = _state_root(state_dir) if pause_enabled else None
    queue: asyncio.Queue[tuple[int, T]] = asyncio.Queue()
    for index, item in enumerate(items):
        queue.put_nowait((index, item))

    missing = object()
    results: list[object] = [missing] * len(items)
    paused = asyncio.Event()
    claim_lock = asyncio.Lock()

    async def worker() -> None:
        while True:
            # The local lock prevents two workers from inspecting the queue
            # concurrently; the state-root lock serializes that inspection
            # with request_pause().
            async with claim_lock:
                # A custom operation can fail closed by raising the boundary
                # marker. Once one worker observes that marker, no sibling may
                # claim another unit; already-admitted operations still drain.
                if paused.is_set():
                    return
                if pause_enabled:
                    assert root is not None
                    with pause_control_lock(root):
                        if pause_requested(root):
                            paused.set()
                            return
                        try:
                            index, item = queue.get_nowait()
                        except asyncio.QueueEmpty:
                            return
                else:
                    try:
                        index, item = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return

            try:
                try:
                    results[index] = await operation(item)
                except PauseBoundaryReached:
                    # A feature operation should not normally raise this
                    # marker, but fail closed if a custom operation does.
                    paused.set()
                    return
                except Exception as exc:
                    results[index] = exc
            finally:
                queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(min(concurrency, len(items)))]
    if workers:
        await asyncio.gather(*workers)

    # A request can arrive after the final item was claimed and while the
    # admitted unit is draining.  Arbitration happens before the caller may
    # construct or promote a canonical aggregate.
    if pause_enabled:
        assert root is not None
        with pause_control_lock(root):
            if pause_requested(root):
                paused.set()

    failures = [
        result for result in results if result is not missing and isinstance(result, BaseException)
    ]
    if failures:
        # A failed admitted unit has no reusable complete checkpoint.  Do not
        # report a deliberate pause over an actual verification failure.
        raise RuntimeError("Phase 2c verification unit failed before checkpoint write") from (
            failures[0]
        )
    if paused.is_set():
        raise PauseBoundaryReached()
    if any(result is missing for result in results):
        raise RuntimeError("Phase 2c verification scheduler lost a unit")
    return [result for result in results if result is not missing]  # type: ignore[misc]


@contextmanager
def promotion_boundary(state_dir: Path | None = None):
    """Serialize canonical aggregate promotion with a pause request.

    The caller must persist the terminal lifecycle state inside this context
    using ``save_state(..., _pause_lock_held=True)``.  If a request already
    won the lock, no aggregate writer is entered.  A request that arrives
    after this boundary is deliberately ordered after the completed Run.
    """

    if state_dir is None:
        yield
        return
    root = _state_root(state_dir)
    with pause_control_lock(root):
        if pause_requested(root):
            raise PauseBoundaryReached()
        yield


__all__ = ["assert_preflight_boundary", "promotion_boundary", "run_verification_units"]
