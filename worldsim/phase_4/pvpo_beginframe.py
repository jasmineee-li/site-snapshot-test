"""Shared ``HeadlessExperimental.beginFrame`` coordination for PVPO.

Chrome's begin-frame-control mode allows only one in-flight beginFrame per
target. A Python ``asyncio.Lock`` is not sufficient if a timed-out CDP
request continues running after ``wait_for`` returns, because the lock can be
released while Chrome still has a pending frame. This coordinator tracks that
orphaned future and refuses to issue a second beginFrame until the first one
drains or the endpoint is marked dirty and recycled.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from worldsim.pvpo_endpoint import canonical_pvpo_endpoint_identity

logger = logging.getLogger(__name__)

_PENDING_RETRIES_ENV = "WORLDSIM_PVPO_BEGINFRAME_PENDING_RETRIES"
_PENDING_BACKOFF_ENV = "WORLDSIM_PVPO_BEGINFRAME_PENDING_BACKOFF_MS"
_DRAIN_TIMEOUT_ENV = "WORLDSIM_PVPO_BEGINFRAME_DRAIN_TIMEOUT_S"
_DEFAULT_PENDING_RETRIES = 2
_DEFAULT_PENDING_BACKOFF_MS = 50.0
_DEFAULT_DRAIN_TIMEOUT_S = 2.0
_ENDPOINT_COORDINATORS: dict[tuple[int, str], BeginFrameCoordinator] = {}
_ENDPOINT_LEASE_LOCKS: dict[tuple[int, str], asyncio.Lock] = {}


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


def _drain_timeout_s() -> float:
    raw = os.environ.get(_DRAIN_TIMEOUT_ENV, "").strip()
    if not raw:
        return _DEFAULT_DRAIN_TIMEOUT_S
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not a number; using %.1fs",
            _DRAIN_TIMEOUT_ENV,
            raw,
            _DEFAULT_DRAIN_TIMEOUT_S,
        )
        return _DEFAULT_DRAIN_TIMEOUT_S
    if value <= 0:
        logger.warning(
            "%s=%r is not positive; using %.1fs",
            _DRAIN_TIMEOUT_ENV,
            raw,
            _DEFAULT_DRAIN_TIMEOUT_S,
        )
        return _DEFAULT_DRAIN_TIMEOUT_S
    return value


def _consume_task_result(task: asyncio.Task[Any]) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except Exception:
        pass


def _endpoint_key(raw_url: str) -> tuple[int, str]:
    loop = asyncio.get_running_loop()
    return (id(loop), canonical_pvpo_endpoint_identity(raw_url))


def coordinator_for_pvpo_endpoint(raw_url: str) -> BeginFrameCoordinator:
    """Return the process-local coordinator for one PVPO CDP endpoint.

    Phase 4 assigns one external chrome-headless-shell endpoint per worker,
    but Browser Use creates fresh sessions per task. The beginFrame invariant
    belongs to the external browser endpoint, not to a single Python session,
    so the coordinator is keyed by canonical CDP endpoint identity.
    """
    key = _endpoint_key(raw_url)
    coordinator = _ENDPOINT_COORDINATORS.get(key)
    if coordinator is None:
        coordinator = BeginFrameCoordinator(endpoint_identity=key[1])
        _ENDPOINT_COORDINATORS[key] = coordinator
    return coordinator


@asynccontextmanager
async def pvpo_endpoint_lease(raw_url: str) -> AsyncIterator[BeginFrameCoordinator]:
    """Serialize task lifecycles that accidentally share a PVPO endpoint."""
    key = _endpoint_key(raw_url)
    lock = _ENDPOINT_LEASE_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _ENDPOINT_LEASE_LOCKS[key] = lock
    async with lock:
        yield coordinator_for_pvpo_endpoint(raw_url)


def reset_pvpo_beginframe_state_for_tests() -> None:
    """Clear process-local endpoint state. Intended for unit tests only."""
    _ENDPOINT_COORDINATORS.clear()
    _ENDPOINT_LEASE_LOCKS.clear()


class BeginFrameCoordinator:
    """Serialize beginFrame requests for one PVPO browser endpoint."""

    def __init__(
        self,
        *,
        lock: asyncio.Lock | None = None,
        endpoint_identity: str | None = None,
    ):
        self.lock = lock or asyncio.Lock()
        self.endpoint_identity = endpoint_identity
        self.timeout_count = 0
        self.pending_error_count = 0
        self.prior_drain_count = 0
        self.prior_drain_timeout_count = 0
        self.send_count = 0
        self._inflight: asyncio.Task[Any] | None = None
        self._dirty_reason: str | None = None

    @property
    def dirty_reason(self) -> str | None:
        return self._dirty_reason

    def mark_dirty(self, reason: str) -> None:
        self._dirty_reason = reason

    def reset_after_recycle(self) -> None:
        """Clear endpoint-local pending/dirty state after confirmed recycle."""
        inflight = self._inflight
        if inflight is not None and not inflight.done():
            inflight.cancel()
            with contextlib.suppress(Exception):
                inflight.add_done_callback(_consume_task_result)
        self._inflight = None
        self._dirty_reason = None

    def stats(self) -> dict[str, Any]:
        """Return JSON-serializable counters for runtime artifacts."""
        payload: dict[str, Any] = {
            "beginframe_sends": self.send_count,
            "beginframe_timeouts": self.timeout_count,
            "beginframe_pending_errors": self.pending_error_count,
            "beginframe_prior_drains": self.prior_drain_count,
            "beginframe_prior_drain_timeouts": self.prior_drain_timeout_count,
        }
        if self.endpoint_identity:
            payload["beginframe_endpoint_identity"] = self.endpoint_identity
        if self._dirty_reason:
            payload["beginframe_dirty_reason"] = self._dirty_reason
        if self._inflight is not None and not self._inflight.done():
            payload["beginframe_inflight_pending"] = True
        return payload

    async def send(
        self,
        cdp: Any,
        params: dict[str, Any],
        *,
        timeout_s: float | None,
        label: str,
    ) -> dict[str, Any]:
        async with self.lock:
            if self._dirty_reason is not None:
                raise BeginFrameTimeout(
                    f"pvpo beginFrame endpoint is dirty: {self._dirty_reason}"
                )
            await self._drain_prior_inflight(label=label)
            retries = _pending_retries()
            backoff_s = _pending_backoff_s()
            deadline = (
                asyncio.get_running_loop().time() + timeout_s
                if timeout_s is not None
                else None
            )
            attempt = 0
            while True:
                try:
                    return await self._send_once(
                        cdp,
                        params,
                        timeout_s=timeout_s,
                        label=label,
                    )
                except Exception as exc:
                    if not is_beginframe_pending_error(exc):
                        raise
                    self.pending_error_count += 1
                    if deadline is None and attempt >= retries:
                        raise
                    remaining = (
                        max(0.0, deadline - asyncio.get_running_loop().time())
                        if deadline is not None
                        else None
                    )
                    if remaining is not None and remaining <= 0:
                        reason = (
                            f"Chrome kept reporting another pending beginFrame for {label} "
                            f"for {timeout_s:.2f}s"
                        )
                        self.mark_dirty(reason)
                        raise BeginFrameTimeout(reason) from exc
                    logger.debug(
                        "pvpo beginFrame %s saw pending frame; retrying (%d/%d)",
                        label,
                        attempt + 1,
                        retries,
                    )
                    sleep_s = backoff_s * (attempt + 1)
                    if remaining is not None:
                        sleep_s = min(sleep_s, remaining)
                    await asyncio.sleep(sleep_s)
                    attempt += 1

    async def _drain_prior_inflight(self, *, label: str) -> None:
        task = self._inflight
        if task is None:
            return
        if task.done():
            _consume_task_result(task)
            if self._inflight is task:
                self._inflight = None
            return

        self.prior_drain_count += 1
        drain_timeout_s = _drain_timeout_s()
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=drain_timeout_s)
        except TimeoutError as exc:
            self.prior_drain_timeout_count += 1
            reason = (
                f"prior pvpo beginFrame was still pending before {label} "
                f"after {drain_timeout_s:.2f}s"
            )
            self.mark_dirty(reason)
            raise BeginFrameTimeout(reason) from exc
        except Exception as exc:
            logger.debug("pvpo beginFrame prior in-flight task finished with error: %s", exc)
        finally:
            if task.done() and self._inflight is task:
                self._inflight = None

    async def _send_once(
        self,
        cdp: Any,
        params: dict[str, Any],
        *,
        timeout_s: float | None,
        label: str,
    ) -> dict[str, Any]:
        self.send_count += 1
        task = asyncio.create_task(
            cdp.send("HeadlessExperimental.beginFrame", params),
            name=f"pvpo-beginframe-{label}",
        )
        self._inflight = task
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
        finally:
            if task.done() and self._inflight is task:
                self._inflight = None
        if not isinstance(result, dict):
            return {}
        return result
