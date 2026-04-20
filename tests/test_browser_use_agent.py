from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, call

import pytest

from worldsim import browser_use_agent
from worldsim.phase_4.pvpo_capture import Rect, StepCapture


class _FakeHistory:
    def __init__(self, screenshot_path: str):
        self.history = [object()]
        self._screenshot_path = screenshot_path

    def save_to_file(self, path):
        path.write_text('{"history":[{"step":1}]}')

    def screenshot_paths(self):
        return [self._screenshot_path]

    def is_done(self):
        return False

    def final_result(self):
        return "partial"

    def errors(self):
        return ["partial failure"]


def test_write_agent_artifacts_persists_partial_history_and_failure_metadata(tmp_path):
    screenshot = tmp_path / "source.png"
    screenshot.write_bytes(b"png")
    history = _FakeHistory(str(screenshot))

    browser_use_agent._write_agent_artifacts(
        task_dir=tmp_path / "task",
        history=history,
        status="timeout",
        extra_errors=["agent timed out after 30s"],
    )

    task_dir = tmp_path / "task"
    assert (task_dir / "history.json").exists()
    assert (task_dir / "screenshots" / "step_0.png").exists()

    final_response = json.loads((task_dir / "final_response.json").read_text())
    assert final_response["status"] == "TIMEOUT"
    assert "agent timed out after 30s" in final_response["errors"]


def test_write_agent_artifacts_preserves_existing_pvpo_screenshot(tmp_path):
    screenshot = tmp_path / "source.png"
    screenshot.write_bytes(b"history")
    history = _FakeHistory(str(screenshot))

    task_dir = tmp_path / "task"
    screenshots_dir = task_dir / "screenshots"
    screenshots_dir.mkdir(parents=True)
    existing = screenshots_dir / "step_0.png"
    existing.write_bytes(b"pvpo")

    browser_use_agent._write_agent_artifacts(
        task_dir=task_dir,
        history=history,
        status="done",
        extra_errors=[],
    )

    assert existing.read_bytes() == b"pvpo"


@pytest.mark.asyncio
async def test_pvpo_callback_warns_once_and_persists_capture_summary(tmp_path, caplog):
    class BrokenSession:
        async def get_current_page(self):
            raise RuntimeError("cdp disconnected")

    callback = browser_use_agent._make_pvpo_step_callback(
        BrokenSession(),
        tmp_path,
        "payload text",
    )

    with caplog.at_level("WARNING", logger="worldsim.browser_use_agent"):
        await callback(SimpleNamespace(), SimpleNamespace(), 1)
        await callback(SimpleNamespace(), SimpleNamespace(), 2)

    warnings = [record.message for record in caplog.records if record.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "current_page_unavailable" in warnings[0]
    assert "degraded mode" in warnings[0]

    summary = json.loads((tmp_path / "pvpo" / "capture_summary.json").read_text())
    assert summary["status"] == "degraded"
    assert summary["steps_seen"] == 2
    assert summary["steps_captured"] == 0
    assert summary["issue_steps"] == 2
    assert summary["first_issue_class"] == "current_page_unavailable"
    assert summary["first_issue_step"] == 1
    assert summary["last_issue_step"] == 2


@pytest.mark.asyncio
async def test_pvpo_callback_writes_artifacts_on_browser_use_success_path(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    cdp_session = object()
    page = object()

    class BrowserUseLikeSession:
        async def get_current_page(self):
            return page

        async def get_or_create_cdp_session(self):
            return cdp_session

    capture = StepCapture(
        screenshot_png=b"png-bytes",
        visibility_vec=[{"i": 0, "char": "A", "layoutVisible": True, "liveRect": {"x": 1, "y": 2, "w": 3, "h": 4}}],
        background_color=(255, 255, 255),
        has_damage=True,
        clip=Rect(x=0, y=0, w=640, h=480),
    )

    inject = AsyncMock()
    viewport = AsyncMock(return_value={"w": 640, "h": 480})
    atomic_capture = AsyncMock(return_value=capture)
    monkeypatch.setattr("worldsim.phase_4.pvpo_browser_config.inject_animation_killer", inject)
    monkeypatch.setattr("worldsim.phase_4.pvpo_cdp.runtime_evaluate_value", viewport)
    monkeypatch.setattr("worldsim.phase_4.pvpo_capture.atomic_capture_with_visibility", atomic_capture)

    callback = browser_use_agent._make_pvpo_step_callback(
        BrowserUseLikeSession(),
        tmp_path,
        "payload text",
    )

    await callback(SimpleNamespace(), SimpleNamespace(), 1)

    inject.assert_awaited_once_with(page, cdp_session)
    viewport.assert_awaited_once()
    atomic_capture.assert_awaited_once_with(
        cdp_session,
        viewport_rect=Rect(x=0, y=0, w=640, h=480),
        payload_text="payload text",
    )

    summary = json.loads((tmp_path / "pvpo" / "capture_summary.json").read_text())
    assert summary["status"] == "ok"
    assert summary["steps_seen"] == 1
    assert summary["steps_captured"] == 1
    assert (tmp_path / "pvpo" / "step_1.json").exists()
    assert (tmp_path / "screenshots" / "step_1.png").read_bytes() == b"png-bytes"


def test_resolve_pvpo_cdp_url_rejects_remote_without_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("WORLDSIM_ALLOW_REMOTE_PVPO_CDP_URL", raising=False)
    with pytest.raises(ValueError, match="loopback"):
        browser_use_agent._resolve_pvpo_cdp_url("http://203.0.113.9:9222")


def test_resolve_pvpo_cdp_url_allows_remote_with_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WORLDSIM_ALLOW_REMOTE_PVPO_CDP_URL", "1")
    assert (
        browser_use_agent._resolve_pvpo_cdp_url("http://203.0.113.9:9222")
        == "http://203.0.113.9:9222"
    )


@pytest.mark.asyncio
async def test_cleanup_external_cdp_state_clears_storage_cookies_and_page(monkeypatch: pytest.MonkeyPatch):
    agent = browser_use_agent.BrowserUseAgent(llm=object())
    agent._pvpo_cdp_url = "http://127.0.0.1:9222"
    page_one = SimpleNamespace(_target_id="target-1")
    page_two = SimpleNamespace(_target_id="target-2")
    cdp_session = object()

    class FakeSession:
        get_pages = AsyncMock(return_value=[page_one, page_two])
        get_current_page = AsyncMock(return_value=page_one)
        get_or_create_cdp_session = AsyncMock(side_effect=[cdp_session, cdp_session])
        clear_cookies = AsyncMock()
        close_page = AsyncMock()

    runtime_eval = AsyncMock(return_value=True)
    monkeypatch.setattr("worldsim.phase_4.pvpo_cdp.runtime_evaluate", runtime_eval)

    session = FakeSession()
    await agent._cleanup_external_cdp_state(session)

    session.get_pages.assert_awaited_once()
    session.get_current_page.assert_not_called()
    assert session.get_or_create_cdp_session.await_args_list == [
        call(target_id="target-1", focus=False),
        call(target_id="target-2", focus=False),
    ]
    assert runtime_eval.await_count == 2
    session.clear_cookies.assert_awaited_once()
    assert session.close_page.await_args_list == [
        call(page_one),
        call(page_two),
    ]
