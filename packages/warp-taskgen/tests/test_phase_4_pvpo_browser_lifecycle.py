from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from worldsim.phase_4 import pvpo_browser_lifecycle as lifecycle


@pytest.mark.asyncio
async def test_recycle_pvpo_browser_closes_browser_and_waits_down_up(monkeypatch):
    close = AsyncMock()
    states = iter([True, True, False, False, True])
    monkeypatch.setenv("WORLDSIM_PVPO_BROWSER_RECYCLE_MODE", "cdp")

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
    monkeypatch.setenv("WORLDSIM_PVPO_BROWSER_RECYCLE_MODE", "cdp")

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
    monkeypatch.setenv("WORLDSIM_PVPO_BROWSER_RECYCLE_MODE", "cdp")

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


@pytest.mark.asyncio
async def test_recycle_pvpo_browser_restarts_managed_loopback_container(monkeypatch):
    calls: list[str] = []

    async def docker_restart(container_name: str, *, timeout_s: float):
        calls.append(container_name)
        assert timeout_s == 1
        return {"returncode": 0, "stdout": container_name, "stderr": ""}

    async def wait_state(cdp_url: str, *, reachable: bool, timeout_s: float):
        assert cdp_url == "http://127.0.0.1:9226"
        return True

    monkeypatch.setattr(lifecycle.shutil, "which", lambda _name: "/usr/bin/docker")
    monkeypatch.setattr(lifecycle, "_run_docker_restart", docker_restart)
    monkeypatch.setattr(lifecycle, "_wait_for_endpoint_state", wait_state)

    artifact = await lifecycle.recycle_pvpo_browser_after_task(
        SimpleNamespace(),
        "http://127.0.0.1:9226",
        timeout_s=1,
    )

    assert calls == ["pvpo-chrome-9226"]
    assert artifact["recycle_method"] == "docker_restart"
    assert artifact["recycle_container"] == "pvpo-chrome-9226"
    assert artifact["recycle_status"] == "recycled"
    assert artifact["docker_restart_returncode"] == 0


@pytest.mark.asyncio
async def test_recycle_pvpo_browser_falls_back_to_cdp_when_docker_unavailable(monkeypatch):
    close = AsyncMock()
    monkeypatch.setattr(lifecycle.shutil, "which", lambda _name: None)
    states = iter([False, True])

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
    assert artifact["recycle_method"] == "cdp_browser_close"
    assert artifact["recycle_status"] == "recycled"


@pytest.mark.asyncio
async def test_docker_mode_fails_when_no_managed_container(monkeypatch):
    monkeypatch.setenv("WORLDSIM_PVPO_BROWSER_RECYCLE_MODE", "docker")

    artifact = await lifecycle.recycle_pvpo_browser_after_task(
        SimpleNamespace(),
        "http://203.0.113.9:9222",
        timeout_s=1,
    )

    assert artifact["recycle_method"] == "docker_restart"
    assert artifact["recycle_status"] == "failed"
    assert "no managed pvpo-chrome" in artifact["recycle_failure"]
