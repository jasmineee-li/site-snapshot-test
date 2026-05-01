"""Shared ``HeadlessExperimental.beginFrame`` coordination for PVPO.

Chrome's begin-frame-control mode allows only one in-flight beginFrame per
target. A Python ``asyncio.Lock`` is not sufficient if a timed-out CDP request
continues running after ``wait_for`` returns, because the lock can be released
while Chrome still has a pending frame. This coordinator serializes normal
requests and treats Chrome's real pending-frame guard as a bounded transient
condition. If the local CDP future times out, it is abandoned rather than
drained forever; the next caller retries on Chrome's explicit
``Another frame is pending`` response.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_PENDING_RETRIES_ENV = "WORLDSIM_PVPO_BEGINFRAME_PENDING_RETRIES"
_PENDING_BACKOFF_ENV = "WORLDSIM_PVPO_BEGINFRAME_PENDING_BACKOFF_MS"
_DEFAULT_PENDING_RETRIES = 2
_DEFAULT_PENDING_BACKOFF_MS = 50.0


class BeginFrameTimeout(TimeoutError):
    """A beginFrame request or pending-frame drain exceeded its budget."""


def is_beginframe_pending_error(exc: BaseException) -> bool:
    return "Another frame is pending" in str(exc)


def _pending_retries() -> int:
    raw = os.environ.get(_PENDING_RETRIES_ENV, "").strip()
    if not raw:
        return _DEFAULT_PENDING_RETRIES
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not an integer; using %d",
            _PENDING_RETRIES_ENV,
            raw,
            _DEFAULT_PENDING_RETRIES,
        )
        return _DEFAULT_PENDING_RETRIES
    return max(0, value)


def _pending_backoff_s() -> float:
    raw = os.environ.get(_PENDING_BACKOFF_ENV, "").strip()
    if not raw:
        return _DEFAULT_PENDING_BACKOFF_MS / 1000.0
    try:
        value_ms = float(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not a number; using %.0fms",
            _PENDING_BACKOFF_ENV,
            raw,
            _DEFAULT_PENDING_BACKOFF_MS,
        )
        return _DEFAULT_PENDING_BACKOFF_MS / 1000.0
    return max(0.0, value_ms) / 1000.0


def _consume_task_result(task: asyncio.Task[Any]) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except Exception:
        pass


class BeginFrameCoordinator:
    """Serialize and drain beginFrame requests for one BrowserSession/CDP target."""

    def __init__(self, *, lock: asyncio.Lock | None = None):
        self.lock = lock or asyncio.Lock()
        self.timeout_count = 0
        self.pending_error_count = 0

    async def send(
        self,
        cdp: Any,
        params: dict[str, Any],
        *,
        timeout_s: float | None,
        label: str,
    ) -> dict[str, Any]:
        async with self.lock:
            retries = _pending_retries()
            backoff_s = _pending_backoff_s()
            for attempt in range(retries + 1):
                try:
                    return await self._send_once(
                        cdp,
                        params,
                        timeout_s=timeout_s,
                        label=label,
                    )
                except Exception as exc:
                    if not is_beginframe_pending_error(exc) or attempt >= retries:
                        raise
                    self.pending_error_count += 1
                    logger.debug(
                        "pvpo beginFrame %s saw pending frame; retrying (%d/%d)",
                        label,
                        attempt + 1,
                        retries,
                    )
                    await asyncio.sleep(backoff_s * (attempt + 1))
            raise AssertionError("unreachable beginFrame retry state")

    async def _send_once(
        self,
        cdp: Any,
        params: dict[str, Any],
        *,
        timeout_s: float | None,
        label: str,
    ) -> dict[str, Any]:
        task = asyncio.create_task(
            cdp.send("HeadlessExperimental.beginFrame", params),
            name=f"pvpo-beginframe-{label}",
        )
        task.add_done_callback(_consume_task_result)
        try:
            if timeout_s is None:
                result = await task
            else:
                result = await asyncio.wait_for(asyncio.shield(task), timeout=timeout_s)
        except TimeoutError as exc:
            self.timeout_count += 1
            raise BeginFrameTimeout(
                f"pvpo beginFrame {label} timed out after {timeout_s:.2f}s"
            ) from exc
        if not isinstance(result, dict):
            return {}
        return result
