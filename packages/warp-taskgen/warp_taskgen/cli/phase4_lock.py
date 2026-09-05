"""Phase 4 CLI run lock and bounded async shutdown."""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PHASE4_ASYNC_SHUTDOWN_TIMEOUT_ENV = "WORLDSIM_PHASE4_ASYNC_SHUTDOWN_TIMEOUT_S"
_PHASE4_ASYNC_SHUTDOWN_TIMEOUT_DEFAULT_S = 10.0


class Phase4AlreadyRunning(RuntimeError):
    """Raised when the per-state-dir Phase 4 run lock is held."""


def _phase4_async_shutdown_timeout() -> float:
    raw = os.environ.get(_PHASE4_ASYNC_SHUTDOWN_TIMEOUT_ENV, "").strip()
    if not raw:
        return _PHASE4_ASYNC_SHUTDOWN_TIMEOUT_DEFAULT_S
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "Invalid %s=%r; using %.1fs",
            _PHASE4_ASYNC_SHUTDOWN_TIMEOUT_ENV,
            raw,
            _PHASE4_ASYNC_SHUTDOWN_TIMEOUT_DEFAULT_S,
        )
        return _PHASE4_ASYNC_SHUTDOWN_TIMEOUT_DEFAULT_S
    if value <= 0:
        logger.warning(
            "Invalid %s=%r; using %.1fs",
            _PHASE4_ASYNC_SHUTDOWN_TIMEOUT_ENV,
            raw,
            _PHASE4_ASYNC_SHUTDOWN_TIMEOUT_DEFAULT_S,
        )
        return _PHASE4_ASYNC_SHUTDOWN_TIMEOUT_DEFAULT_S
    return value


def _run_phase4_with_bounded_async_shutdown(coro: Any, *, shutdown_timeout_s: float) -> Any:
    """Run Phase 4 without letting third-party background tasks hang process exit.

    Browser Use can leave storage-state/watchdog tasks alive after WARP Taskgen has
    written complete Phase 4 artifacts. ``asyncio.run`` waits indefinitely for
    cancellation during shutdown, which keeps registered r5 jobs marked
    ``running`` and prevents post-run exporters from executing. Phase 4 owns
    browser-agent lifecycle boundaries, so it gets a bounded shutdown wrapper
    while the rest of the CLI keeps standard ``asyncio.run`` behavior.
    """

    loop = asyncio.new_event_loop()
    try:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("bounded Phase 4 runner cannot be called from a running event loop")
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
        if pending:
            for task in pending:
                task.cancel()
            done, still_pending = loop.run_until_complete(
                asyncio.wait(pending, timeout=shutdown_timeout_s)
            )
            for task in done:
                with contextlib.suppress(asyncio.CancelledError):
                    task.result()
            if still_pending:
                for task in still_pending:
                    with contextlib.suppress(Exception):
                        task._log_destroy_pending = False  # type: ignore[attr-defined]
                logger.warning(
                    "Phase 4 async shutdown timed out after %.1fs with %d pending task(s); "
                    "closing loop after completed artifacts were returned",
                    shutdown_timeout_s,
                    len(still_pending),
                )
        loop.run_until_complete(loop.shutdown_asyncgens())
        try:
            loop.run_until_complete(loop.shutdown_default_executor(timeout=shutdown_timeout_s))
        except TypeError:
            try:
                loop.run_until_complete(
                    asyncio.wait_for(loop.shutdown_default_executor(), timeout=shutdown_timeout_s)
                )
            except TimeoutError:
                logger.warning(
                    "Phase 4 default executor shutdown timed out after %.1fs",
                    shutdown_timeout_s,
                )
        except TimeoutError:
            logger.warning(
                "Phase 4 default executor shutdown timed out after %.1fs",
                shutdown_timeout_s,
            )
        asyncio.set_event_loop(None)
        loop.close()


@contextlib.contextmanager
def _phase4_run_lock(state_dir: Path):
    """Prevent concurrent Phase 4 runs from resetting the same benchmark stack."""
    lock_dir = state_dir / "phase_4"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / ".phase4_run.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            existing = handle.read().strip()
            detail = f"; holder: {existing}" if existing else ""
            raise Phase4AlreadyRunning(
                f"another Phase 4 run already holds {lock_path}{detail}"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} cwd={Path.cwd()} cmd={' '.join(sys.argv)}\n")
        handle.flush()
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


__all__ = [
    "Phase4AlreadyRunning",
    "_phase4_async_shutdown_timeout",
    "_phase4_run_lock",
    "_run_phase4_with_bounded_async_shutdown",
]
