from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from worldsim.phase_4.pvpo_beginframe import BeginFrameCoordinator, BeginFrameTimeout


@pytest.mark.asyncio
async def test_beginframe_timeout_abandons_orphaned_future_and_retries_pending_guard():
    coordinator = BeginFrameCoordinator()
    release_first = asyncio.Event()
    calls = 0

    async def _send(method: str, params: dict):
        nonlocal calls
        assert method == "HeadlessExperimental.beginFrame"
        calls += 1
        if calls == 1:
            await release_first.wait()
        if calls == 2:
            raise RuntimeError({"code": -32000, "message": "Another frame is pending"})
        return {"hasDamage": True}

    cdp = AsyncMock()
    cdp.send = AsyncMock(side_effect=_send)

    with pytest.raises(BeginFrameTimeout):
        await coordinator.send(cdp, {}, timeout_s=0.01, label="test")

    result = await coordinator.send(cdp, {}, timeout_s=0.5, label="test")
    release_first.set()
    await asyncio.sleep(0)

    assert result == {"hasDamage": True}
    assert calls == 3
    assert coordinator.timeout_count == 1
    assert coordinator.pending_error_count == 1


@pytest.mark.asyncio
async def test_beginframe_pending_error_retries_within_controller():
    coordinator = BeginFrameCoordinator()
    calls = 0

    async def _send(method: str, params: dict):
        nonlocal calls
        assert method == "HeadlessExperimental.beginFrame"
        calls += 1
        if calls == 1:
            raise RuntimeError({"code": -32000, "message": "Another frame is pending"})
        return {"screenshotData": "png"}

    cdp = AsyncMock()
    cdp.send = AsyncMock(side_effect=_send)

    result = await coordinator.send(
        cdp,
        {"screenshot": {"format": "png"}},
        timeout_s=0.5,
        label="test",
    )

    assert result == {"screenshotData": "png"}
    assert calls == 2
    assert coordinator.pending_error_count == 1
