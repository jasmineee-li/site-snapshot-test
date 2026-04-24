"""Unit tests for the per-session PVPO ``beginFrame`` pump.

The pump exists because Chrome launched with ``--enable-begin-frame-control``
(PVPO rigor) only issues frames over the DevTools Protocol; browser-use
0.12.6 never calls ``HeadlessExperimental.beginFrame``, so the compositor
stalls on step-1 navigation unless something else drives it. The pump is
that "something else." It must:

* fire ``HeadlessExperimental.beginFrame`` on each tick while the session
  has a focused target;
* pause while an atomic PVPO capture is in flight (gated via the
  ``capturing`` :class:`asyncio.Event`);
* swallow CDP errors at debug log level rather than crash the trajectory;
* stop cleanly on context exit regardless of how the enclosed block exited;
* honor the ``WORLDSIM_PVPO_FRAME_PUMP_MS=0`` kill switch.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import AsyncMock

import pytest

from worldsim.phase_4.pvpo_frame_pump import frame_pump


class _FakeSession:
    """Minimal stand-in for browser-use's ``BrowserSession`` surface."""

    def __init__(self, *, target_id: str | None = "target-1"):
        self.agent_focus_target_id = target_id
        self._cdp = AsyncMock()
        self._cdp.send = AsyncMock(return_value={})
        self.get_or_create_cdp_session_calls = 0

    async def get_or_create_cdp_session(self, *, target_id: str, focus: bool = False) -> Any:
        self.get_or_create_cdp_session_calls += 1
        return self._cdp


def _begin_frame_calls(session: _FakeSession) -> list[tuple[str, dict | None]]:
    return [
        (call.args[0], call.args[1] if len(call.args) > 1 else None)
        for call in session._cdp.send.call_args_list
        if call.args and call.args[0] == "HeadlessExperimental.beginFrame"
    ]


@pytest.mark.asyncio
async def test_pump_sends_begin_frame_on_tick():
    session = _FakeSession()
    async with frame_pump(session, interval_s=0.01):
        await asyncio.sleep(0.08)  # ~8 ticks
    calls = _begin_frame_calls(session)
    assert len(calls) >= 2, f"expected repeated beginFrame calls, got {calls!r}"
    # Defaults: no screenshot, default noDisplayUpdates (=false).
    for _, params in calls:
        assert params == {}


@pytest.mark.asyncio
async def test_pump_respects_capturing_event():
    session = _FakeSession()
    async with frame_pump(session, interval_s=0.01) as capturing:
        capturing.set()
        await asyncio.sleep(0.05)
        calls_while_set = len(_begin_frame_calls(session))
        capturing.clear()
        await asyncio.sleep(0.05)
        calls_after_clear = len(_begin_frame_calls(session))
    # Gate held → at most ~zero beginFrame calls; some slop tolerated for the
    # tick that was already in flight when we set the event. Once cleared,
    # beginFrame calls must resume.
    assert calls_while_set <= 1
    assert calls_after_clear > calls_while_set


@pytest.mark.asyncio
async def test_pump_skips_tick_when_focus_target_missing():
    session = _FakeSession(target_id=None)
    async with frame_pump(session, interval_s=0.01):
        await asyncio.sleep(0.05)
    assert _begin_frame_calls(session) == []
    assert session.get_or_create_cdp_session_calls == 0


@pytest.mark.asyncio
async def test_pump_swallows_cdp_errors_at_debug(caplog):
    session = _FakeSession()
    session._cdp.send = AsyncMock(side_effect=RuntimeError("boom"))
    with caplog.at_level(logging.DEBUG, logger="worldsim.phase_4.pvpo_frame_pump"):
        async with frame_pump(session, interval_s=0.01):
            await asyncio.sleep(0.04)
    # Pump survived the loop (no exception propagated) and logged debug info.
    assert any("beginFrame failed" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_pump_stops_on_context_exit_even_if_block_raises():
    session = _FakeSession()
    AsyncMock(return_value={})

    async def _slow_send(method: str, params: dict | None = None):
        await asyncio.sleep(0.005)
        return {}

    session._cdp.send = AsyncMock(side_effect=_slow_send)
    with pytest.raises(ValueError, match="inner"):
        async with frame_pump(session, interval_s=0.01):
            await asyncio.sleep(0.03)
            raise ValueError("inner")
    # Give any residual tick a chance to complete; then confirm no tasks leak.
    await asyncio.sleep(0.05)
    pump_tasks = [t for t in asyncio.all_tasks() if t.get_name() == "pvpo-frame-pump"]
    assert pump_tasks == [], f"pump task leaked: {pump_tasks!r}"


@pytest.mark.asyncio
async def test_pump_disabled_when_interval_zero():
    session = _FakeSession()
    async with frame_pump(session, interval_s=0) as capturing:
        assert isinstance(capturing, asyncio.Event)
        await asyncio.sleep(0.02)
    # No beginFrame calls should have fired and no pump task should ever
    # have been spawned.
    assert _begin_frame_calls(session) == []


@pytest.mark.asyncio
async def test_pump_disabled_via_env(monkeypatch):
    monkeypatch.setenv("WORLDSIM_PVPO_FRAME_PUMP_MS", "0")
    session = _FakeSession()
    async with frame_pump(session):
        await asyncio.sleep(0.02)
    assert _begin_frame_calls(session) == []


@pytest.mark.asyncio
async def test_pump_interval_from_env(monkeypatch):
    monkeypatch.setenv("WORLDSIM_PVPO_FRAME_PUMP_MS", "10")
    session = _FakeSession()
    async with frame_pump(session):
        await asyncio.sleep(0.08)
    # ~8 ticks at 10ms should produce at least 2 beginFrame calls.
    assert len(_begin_frame_calls(session)) >= 2


@pytest.mark.asyncio
async def test_pump_swallows_get_or_create_cdp_session_errors(caplog):
    class _BadSession(_FakeSession):
        async def get_or_create_cdp_session(self, *, target_id: str, focus: bool = False):
            raise RuntimeError("no cdp")

    session = _BadSession()
    with caplog.at_level(logging.DEBUG, logger="worldsim.phase_4.pvpo_frame_pump"):
        async with frame_pump(session, interval_s=0.01):
            await asyncio.sleep(0.04)
    assert any("cdp session unavailable" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_pump_yields_beginframe_lock_on_capturing_event():
    """Regression: the yielded capturing Event must carry a ``beginframe_lock``
    asyncio.Lock so :func:`pvpo_capture.atomic_capture_with_visibility` can
    serialize its own ``HeadlessExperimental.beginFrame`` call against the
    pump's ticks. Without this mutex the capture races the pump and Chrome
    returns ``{'code': -32000, 'message': 'Another frame is pending'}``.
    """
    session = _FakeSession()
    async with frame_pump(session, interval_s=0.01) as capturing:
        lock = getattr(capturing, "beginframe_lock", None)
        assert isinstance(lock, asyncio.Lock), (
            "frame_pump must attach an asyncio.Lock to the yielded Event; "
            "without it atomic_capture_with_visibility cannot serialize beginFrame"
        )
        assert not lock.locked()


@pytest.mark.asyncio
async def test_pump_blocks_when_beginframe_lock_held():
    """When a capture holds the ``beginframe_lock``, the pump must not race it.
    Simulate that by grabbing the lock ourselves and verifying zero
    beginFrame calls complete while held; pump resumes after release.
    """
    session = _FakeSession()
    async with frame_pump(session, interval_s=0.005) as capturing:
        lock = capturing.beginframe_lock  # type: ignore[attr-defined]
        # Make the pump's beginFrame calls block inside the lock so we can
        # prove it really serializes; without blocking, the pump might grab
        # the lock in between our acquire/release and silently complete.
        pump_blocker = asyncio.Event()
        original_send = session._cdp.send

        async def blocking_send(method: str, params: dict | None = None):
            if method == "HeadlessExperimental.beginFrame":
                await pump_blocker.wait()
            return await original_send(method, params)

        session._cdp.send = AsyncMock(side_effect=blocking_send)

        async with lock:
            # Pump ticks during this window; each tick tries to acquire the
            # lock and waits. Zero beginFrame calls should complete.
            await asyncio.sleep(0.04)
            calls_while_held = len(_begin_frame_calls(session))
        # Release: unblock pump's sends, let it finish a couple ticks.
        pump_blocker.set()
        await asyncio.sleep(0.04)
        calls_after_release = len(_begin_frame_calls(session))

    assert calls_while_held == 0, (
        f"pump should block on beginframe_lock, but issued {calls_while_held} "
        "beginFrame calls while the lock was held"
    )
    assert calls_after_release > 0
