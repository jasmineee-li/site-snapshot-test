"""Per-session ``HeadlessExperimental.beginFrame`` pump.

Chrome launched with ``--enable-begin-frame-control`` issues frames only over
the DevTools Protocol (see ``headless/public/switches.h``:
``// Whether or not begin frames should be issued over DevToolsProtocol``).
browser-use 0.12.6 never calls ``HeadlessExperimental.beginFrame``, so the
compositor stalls on the first navigation — no frames, no lifecycle events,
no completion. PVPO's per-step capture would drive the compositor, but it
runs *after* the agent step and therefore cannot help step 1.

This module drives a minimal side-effect-only ``beginFrame`` at a short
interval from a background task tied to the agent session. It is gated off
during :func:`atomic_capture_with_visibility` (via an ``asyncio.Event``) so
the atomic capture remains the single source of truth for the visible PNG
bytes.

Kill switch: set ``WORLDSIM_PVPO_FRAME_PUMP_MS=0`` to disable the pump
without touching code. Any value <= 0 disables it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

from worldsim.phase_4.pvpo_cdp import normalize_cdp_session

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL_MS = 50.0
_DEFAULT_BEGINFRAME_TIMEOUT_S = 1.0
_SHUTDOWN_GRACE_S = 2.0


def _resolve_interval_s(interval_s: float | None) -> float:
    if interval_s is not None:
        return interval_s
    raw = os.environ.get("WORLDSIM_PVPO_FRAME_PUMP_MS", "").strip()
    if not raw:
        return _DEFAULT_INTERVAL_MS / 1000.0
    try:
        return float(raw) / 1000.0
    except ValueError:
        logger.warning(
            "WORLDSIM_PVPO_FRAME_PUMP_MS=%r is not a number; using default %sms",
            raw,
            _DEFAULT_INTERVAL_MS,
        )
        return _DEFAULT_INTERVAL_MS / 1000.0


def _beginframe_timeout_s() -> float:
    raw = os.environ.get("WORLDSIM_PVPO_FRAME_PUMP_TIMEOUT_S", "").strip()
    if not raw:
        return _DEFAULT_BEGINFRAME_TIMEOUT_S
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "WORLDSIM_PVPO_FRAME_PUMP_TIMEOUT_S=%r is not a number; using default %.1fs",
            raw,
            _DEFAULT_BEGINFRAME_TIMEOUT_S,
        )
        return _DEFAULT_BEGINFRAME_TIMEOUT_S
    if value <= 0:
        logger.warning(
            "WORLDSIM_PVPO_FRAME_PUMP_TIMEOUT_S=%r is not positive; using default %.1fs",
            raw,
            _DEFAULT_BEGINFRAME_TIMEOUT_S,
        )
        return _DEFAULT_BEGINFRAME_TIMEOUT_S
    return value


async def _pump_once(cdp_session: Any) -> None:
    """Issue one ``HeadlessExperimental.beginFrame`` with defaults.

    Defaults (per CDP docs): no screenshot, ``noDisplayUpdates=false``. That
    drives the compositor to commit a visible frame, which is what Chrome
    needs to fire ``firstPaint`` / ``firstContentfulPaint`` / ``load``
    lifecycle events browser-use's navigation handler is waiting on.
    """
    await normalize_cdp_session(cdp_session).send("HeadlessExperimental.beginFrame", {})


async def _pump_loop(
    session: Any,
    capturing: asyncio.Event,
    beginframe_lock: asyncio.Lock,
    stop: asyncio.Event,
    interval_s: float,
) -> None:
    beginframe_timeout_s = _beginframe_timeout_s()
    while not stop.is_set():
        try:
            if capturing.is_set():
                await asyncio.sleep(interval_s)
                continue
            target_id = getattr(session, "agent_focus_target_id", None)
            if not target_id:
                await asyncio.sleep(interval_s)
                continue
            try:
                cdp_session = await session.get_or_create_cdp_session(
                    target_id=target_id, focus=False
                )
            except Exception as exc:
                logger.debug("pvpo pump: cdp session unavailable: %s", exc)
                await asyncio.sleep(interval_s)
                continue
            # Serialize beginFrame with any concurrent atomic capture so we
            # don't trip Chrome's "Another frame is pending" guard. The
            # capturing Event still short-circuits the loop above as a
            # cheap skip during captures, but the lock is the real mutex.
            try:
                async with beginframe_lock:
                    await asyncio.wait_for(_pump_once(cdp_session), timeout=beginframe_timeout_s)
            except TimeoutError:
                logger.debug(
                    "pvpo pump: beginFrame timed out after %.2fs",
                    beginframe_timeout_s,
                )
            except Exception as exc:
                logger.debug("pvpo pump: beginFrame failed: %s", exc)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - paranoia around session quirks
            logger.debug("pvpo pump: unexpected error: %s", exc)
        await asyncio.sleep(interval_s)


@contextlib.asynccontextmanager
async def frame_pump(
    session: Any,
    *,
    interval_s: float | None = None,
) -> AsyncIterator[asyncio.Event]:
    """Run a per-session ``beginFrame`` pump for the lifetime of the block.

    Yields a ``capturing`` :class:`asyncio.Event` with an attached
    ``beginframe_lock`` attribute (``asyncio.Lock``) that callers
    (specifically :func:`worldsim.phase_4.pvpo_capture.atomic_capture_with_visibility`)
    should use to serialize any ``HeadlessExperimental.beginFrame`` call
    against the pump's own ticks. Setting ``capturing`` still causes the
    pump to cheaply skip its tick during a capture; the lock is what
    actually prevents Chrome's ``Another frame is pending`` guard from
    tripping when the pump has a frame in flight.

    The pump is disabled (context manager yields immediately, no task is
    spawned) when ``interval_s`` is <= 0, or when the env var
    ``WORLDSIM_PVPO_FRAME_PUMP_MS`` is set to 0. Use that as the emergency
    kill switch without having to edit code.
    """
    capturing = asyncio.Event()
    existing_lock = getattr(session, "_worldsim_pvpo_beginframe_lock", None)
    beginframe_lock = existing_lock if existing_lock is not None else asyncio.Lock()
    # Attach the lock to the yielded Event so existing callers that just
    # `async with frame_pump(...) as capturing` keep their one-arg binding,
    # and the capture path can reach the lock via ``capturing.beginframe_lock``.
    capturing.beginframe_lock = beginframe_lock  # type: ignore[attr-defined]
    session._worldsim_pvpo_beginframe_lock = beginframe_lock
    effective_interval = _resolve_interval_s(interval_s)
    if effective_interval <= 0:
        logger.info("pvpo frame pump disabled (interval<=0)")
        yield capturing
        return

    stop = asyncio.Event()
    task = asyncio.create_task(
        _pump_loop(session, capturing, beginframe_lock, stop, effective_interval),
        name="pvpo-frame-pump",
    )
    try:
        yield capturing
    finally:
        stop.set()
        capturing.clear()
        try:
            await asyncio.wait_for(task, timeout=_SHUTDOWN_GRACE_S)
        except TimeoutError:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - task body already swallows
            logger.debug("pvpo pump: teardown raised: %s", exc)
