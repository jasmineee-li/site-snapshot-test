from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from worldsim.phase_4 import pvpo_browser_lifecycle as lifecycle


@pytest.mark.asyncio
async def test_recycle_pvpo_browser_closes_browser_and_waits_down_up(monkeypatch):
    close = AsyncMock()
    states = iter([True, True, False, False, True])

    async def endpoint_reachable(_url: str) -> bool:
        return next(states, True)

    monkeypatch.setattr(lifecycle, "_endpoint_reachable", endpoint_reachable)
    monkeypatch.setattr(lifecycle, "_POLL_INTERVAL_S", 0)

    session = SimpleNamespace(
        cdp_client=SimpleNamespace(send=SimpleNamespace(Browser=SimpleNamespace(close=close)))
    )

    artifact = await lifecycle.recycle_pvpo_browser_after_task(
        session,
        "http://127.0.0.1:9222",
        timeout_s=1,
    )

    close.assert_awaited_once_with()
    assert artifact["recycle_status"] == "recycled"
    assert artifact["recycle_down_observed"] is True
    assert artifact["recycle_up_observed"] is True


@pytest.mark.asyncio
async def test_recycle_pvpo_browser_records_unconfirmed_restart(monkeypatch):
    close = AsyncMock()

    async def endpoint_reachable(_url: str) -> bool:
        return True

    monkeypatch.setattr(lifecycle, "_endpoint_reachable", endpoint_reachable)
    monkeypatch.setattr(lifecycle, "_POLL_INTERVAL_S", 0)

    session = SimpleNamespace(
        cdp_client=SimpleNamespace(send=SimpleNamespace(Browser=SimpleNamespace(close=close)))
    )

    artifact = await lifecycle.recycle_pvpo_browser_after_task(
        session,
        "http://127.0.0.1:9222",
        timeout_s=0.01,
    )

    assert artifact["recycle_status"] == "recycled_unconfirmed"
    assert artifact["recycle_down_observed"] is False
    assert artifact["recycle_up_observed"] is True


@pytest.mark.asyncio
async def test_recycle_pvpo_browser_can_be_disabled(monkeypatch):
    close = AsyncMock()
    monkeypatch.setenv("WORLDSIM_PVPO_BROWSER_RECYCLE", "0")
    session = SimpleNamespace(
        cdp_client=SimpleNamespace(send=SimpleNamespace(Browser=SimpleNamespace(close=close)))
    )

    artifact = await lifecycle.recycle_pvpo_browser_after_task(
        session,
        "http://127.0.0.1:9222",
    )

    close.assert_not_called()
    assert artifact["recycle_enabled"] is False
    assert artifact["recycle_status"] == "disabled"


@pytest.mark.asyncio
async def test_recycle_pvpo_browser_falls_back_to_target_session(monkeypatch):
    cdp_session = SimpleNamespace(send=AsyncMock(return_value={}))
    page = SimpleNamespace(_target_id="target-1")
    session = SimpleNamespace(
        get_current_page=AsyncMock(return_value=page),
        get_or_create_cdp_session=AsyncMock(return_value=cdp_session),
    )
    states = iter([False, True])

    async def endpoint_reachable(_url: str) -> bool:
        return next(states, True)

    monkeypatch.setattr(lifecycle, "_endpoint_reachable", endpoint_reachable)
    monkeypatch.setattr(lifecycle, "_POLL_INTERVAL_S", 0)

    artifact = await lifecycle.recycle_pvpo_browser_after_task(
        session,
        "http://127.0.0.1:9222",
        timeout_s=1,
    )

    session.get_or_create_cdp_session.assert_awaited_once_with(target_id="target-1", focus=False)
    cdp_session.send.assert_awaited_once_with("Browser.close", {})
    assert artifact["recycle_status"] == "recycled"
