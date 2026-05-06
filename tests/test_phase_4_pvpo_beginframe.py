from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from worldsim.phase_4.pvpo_beginframe import (
    BeginFrameCoordinator,
    BeginFrameTimeout,
    coordinator_for_pvpo_endpoint,
    pvpo_endpoint_lease,
    reset_pvpo_beginframe_state_for_tests,
)


@pytest.mark.asyncio
async def test_beginframe_timeout_blocks_next_send_until_prior_frame_drains():
    coordinator = BeginFrameCoordinator()
    release_first = asyncio.Event()
    calls = 0

    async def _send(method: str, params: dict):
        nonlocal calls
        assert method == "HeadlessExperimental.beginFrame"
        calls += 1
        if calls == 1:
            await release_first.wait()
        return {"hasDamage": True}

    cdp = AsyncMock()
    cdp.send = AsyncMock(side_effect=_send)

    with pytest.raises(BeginFrameTimeout):
        await coordinator.send(cdp, {}, timeout_s=0.01, label="test")

    release_first.set()
    result = await coordinator.send(cdp, {}, timeout_s=0.5, label="test")

    assert result == {"hasDamage": True}
    assert calls == 2
    assert coordinator.timeout_count == 1
    assert coordinator.prior_drain_count == 1


@pytest.mark.asyncio
async def test_beginframe_timeout_does_not_issue_second_frame_while_prior_is_pending(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("WORLDSIM_PVPO_BEGINFRAME_DRAIN_TIMEOUT_S", "0.01")
    coordinator = BeginFrameCoordinator()
    release_first = asyncio.Event()
    calls = 0

    async def _send(method: str, params: dict):
        nonlocal calls
        assert method == "HeadlessExperimental.beginFrame"
        calls += 1
        await release_first.wait()
        return {"hasDamage": True}

    cdp = AsyncMock()
    cdp.send = AsyncMock(side_effect=_send)

    with pytest.raises(BeginFrameTimeout):
        await coordinator.send(cdp, {}, timeout_s=0.01, label="first")
    with pytest.raises(BeginFrameTimeout, match="prior pvpo beginFrame"):
        await coordinator.send(cdp, {}, timeout_s=0.5, label="second")

    assert calls == 1
    assert coordinator.prior_drain_timeout_count == 1
    assert "prior pvpo beginFrame" in (coordinator.dirty_reason or "")
    release_first.set()


@pytest.mark.asyncio
async def test_beginframe_public_drain_quiesces_prior_inflight():
    coordinator = BeginFrameCoordinator()
    release_first = asyncio.Event()

    async def _send(method: str, params: dict):
        assert method == "HeadlessExperimental.beginFrame"
        await release_first.wait()
        return {"hasDamage": True}

    cdp = AsyncMock()
    cdp.send = AsyncMock(side_effect=_send)

    with pytest.raises(BeginFrameTimeout):
        await coordinator.send(cdp, {}, timeout_s=0.01, label="navigation-tick")

    release_first.set()
    await coordinator.drain_prior(label="post-navigation-tick", timeout_s=0.5)

    assert coordinator.dirty_reason is None
    assert coordinator.prior_drain_count == 1
    await coordinator.send(cdp, {}, timeout_s=0.5, label="atomic-capture")
    assert cdp.send.await_count == 2


@pytest.mark.asyncio
async def test_beginframe_sync_timeout_marks_endpoint_dirty():
    coordinator = BeginFrameCoordinator()

    async def _send(method: str, params: dict):
        assert method == "HeadlessExperimental.beginFrame"
        raise TimeoutError("sync CDP send exceeded timeout")

    cdp = AsyncMock()
    cdp.send = AsyncMock(side_effect=_send)

    with pytest.raises(BeginFrameTimeout, match="pvpo beginFrame browsergym-screenshot"):
        await coordinator.send(cdp, {}, timeout_s=0.5, label="browsergym-screenshot")

    assert "browsergym-screenshot" in (coordinator.dirty_reason or "")
    with pytest.raises(BeginFrameTimeout, match="endpoint is dirty"):
        await coordinator.send(cdp, {}, timeout_s=0.5, label="atomic-capture")
    assert cdp.send.await_count == 1


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


@pytest.mark.asyncio
async def test_beginframe_repeated_pending_guard_becomes_typed_timeout(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("WORLDSIM_PVPO_BEGINFRAME_PENDING_BACKOFF_MS", "1")
    coordinator = BeginFrameCoordinator()

    async def _send(method: str, params: dict):
        assert method == "HeadlessExperimental.beginFrame"
        raise RuntimeError({"code": -32000, "message": "Another frame is pending"})

    cdp = AsyncMock()
    cdp.send = AsyncMock(side_effect=_send)

    with pytest.raises(BeginFrameTimeout, match="Chrome kept reporting"):
        await coordinator.send(cdp, {}, timeout_s=0.01, label="atomic-capture")

    assert coordinator.pending_error_count > 0
    assert "Chrome kept reporting" in (coordinator.dirty_reason or "")


@pytest.mark.asyncio
async def test_endpoint_coordinator_is_shared_by_canonical_cdp_url():
    reset_pvpo_beginframe_state_for_tests()

    first = coordinator_for_pvpo_endpoint("http://localhost:9222/")
    second = coordinator_for_pvpo_endpoint("http://127.0.0.1:9222")
    other = coordinator_for_pvpo_endpoint("http://127.0.0.1:9223")

    assert first is second
    assert first is not other


@pytest.mark.asyncio
async def test_endpoint_lease_serializes_same_endpoint_but_not_distinct_endpoints():
    reset_pvpo_beginframe_state_for_tests()
    order: list[str] = []
    first_inside = asyncio.Event()
    release_first = asyncio.Event()

    async def _same_endpoint_worker(name: str):
        async with pvpo_endpoint_lease("http://127.0.0.1:9222"):
            order.append(f"{name}:enter")
            if name == "first":
                first_inside.set()
                await release_first.wait()
            order.append(f"{name}:exit")

    first = asyncio.create_task(_same_endpoint_worker("first"))
    await first_inside.wait()
    second = asyncio.create_task(_same_endpoint_worker("second"))
    await asyncio.sleep(0)
    assert order == ["first:enter"]
    release_first.set()
    await asyncio.gather(first, second)
    assert order == ["first:enter", "first:exit", "second:enter", "second:exit"]

    active = 0
    peak = 0

    async def _distinct_endpoint_worker(port: int):
        nonlocal active, peak
        async with pvpo_endpoint_lease(f"http://127.0.0.1:{port}"):
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0)
            active -= 1

    await asyncio.gather(_distinct_endpoint_worker(9222), _distinct_endpoint_worker(9223))
    assert peak == 2
