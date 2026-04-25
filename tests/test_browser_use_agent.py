from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
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


def _write_storage_state(path: Path, *, domain: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "cookies": [
                    {
                        "name": "session",
                        "value": "abc",
                        "domain": domain,
                        "path": "/",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_resolve_auth_falls_back_when_declared_storage_state_is_wrong_host(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    declared = tmp_path / "declared.json"
    fallback = tmp_path / "phase_0d" / "gitlab" / "storage_state.json"
    _write_storage_state(declared, domain="old.example")
    _write_storage_state(fallback, domain="gitlab.example")
    monkeypatch.setattr(browser_use_agent, "_phase_0d_fallback_path", lambda _task: fallback)

    session_kwargs, deferred = browser_use_agent._resolve_auth(
        {"type": "storage_state", "storage_state": {"path": str(declared)}},
        {"site": "gitlab"},
        benchmark_root=tmp_path,
        site_url="http://gitlab.example",
    )

    storage_state = json.loads(Path(session_kwargs["storage_state"]).read_text())
    assert Path(session_kwargs["storage_state"]) == fallback.resolve()
    assert storage_state["cookies"][0]["domain"] == "gitlab.example"
    assert storage_state["cookies"][0]["sameSite"] == "Lax"
    assert deferred == []


def test_resolve_auth_prefers_valid_declared_storage_state_over_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    declared = tmp_path / "declared.json"
    fallback = tmp_path / "phase_0d" / "gitlab" / "storage_state.json"
    _write_storage_state(declared, domain="gitlab.example")
    _write_storage_state(fallback, domain="old.example")
    monkeypatch.setattr(browser_use_agent, "_phase_0d_fallback_path", lambda _task: fallback)

    session_kwargs, deferred = browser_use_agent._resolve_auth(
        {"type": "storage_state", "storage_state": {"path": str(declared)}},
        {"site": "gitlab"},
        benchmark_root=tmp_path,
        site_url="http://gitlab.example",
    )

    storage_state = json.loads(Path(session_kwargs["storage_state"]).read_text())
    assert Path(session_kwargs["storage_state"]) == declared.resolve()
    assert storage_state["cookies"][0]["domain"] == "gitlab.example"
    assert deferred == []


def test_resolve_auth_materializes_per_task_storage_state_copy(
    tmp_path: Path,
):
    declared = tmp_path / "declared.json"
    runtime_dir = tmp_path / "task" / "auth"
    _write_storage_state(declared, domain="gitlab.example")

    session_kwargs, deferred = browser_use_agent._resolve_auth(
        {"type": "storage_state", "storage_state": {"path": str(declared)}},
        {"site": "gitlab"},
        benchmark_root=tmp_path,
        site_url="http://gitlab.example",
        storage_state_runtime_dir=runtime_dir,
    )

    runtime_path = Path(session_kwargs["storage_state"])
    assert runtime_path == (runtime_dir / "storage_state.json").resolve()
    assert runtime_path.exists()
    runtime_payload = json.loads(runtime_path.read_text())
    source_payload = json.loads(declared.read_text())
    assert runtime_payload["cookies"][0]["sameSite"] == "Lax"
    assert "sameSite" not in source_payload["cookies"][0]
    assert deferred == []


def test_resolve_auth_validates_phase_0d_fallback_storage_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    declared = "missing.json"
    fallback = tmp_path / "phase_0d" / "gitlab" / "storage_state.json"
    _write_storage_state(fallback, domain="old.example")
    monkeypatch.setattr(browser_use_agent, "_phase_0d_fallback_path", lambda _task: fallback)

    with pytest.raises(browser_use_agent.AuthArtifactMissingError, match="do not match live host"):
        browser_use_agent._resolve_auth(
            {"type": "storage_state", "storage_state": {"path": declared}},
            {"site": "gitlab"},
            benchmark_root=tmp_path,
            site_url="http://gitlab.example",
        )
    assert "do not match live host" in caplog.text


@pytest.mark.asyncio
async def test_scoped_header_shutdown_suppresses_poll_task_errors():
    injector = browser_use_agent._ScopedHeaderAuthInjector(
        origin="http://gitlab.test",
        headers={"X-Test": "1"},
    )

    async def fail_poll():
        raise RuntimeError("poll failed")

    injector._running = True
    injector._poll_task = asyncio.create_task(fail_poll())
    await asyncio.sleep(0)

    await injector.stop()
    assert injector._poll_task is None


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
    page = SimpleNamespace()

    class BrowserUseLikeSession:
        async def get_current_page(self):
            return page

        async def get_or_create_cdp_session(self, *, target_id=None, focus=False):
            assert target_id == "target-1"
            assert focus is False
            return cdp_session

    page._target_id = "target-1"
    capture = StepCapture(
        screenshot_png=b"png-bytes",
        visibility_vec=[
            {
                "i": 0,
                "char": "A",
                "layoutVisible": True,
                "liveRect": {"x": 1, "y": 2, "w": 3, "h": 4},
            }
        ],
        background_color=(255, 255, 255),
        has_damage=True,
        clip=Rect(x=0, y=0, w=640, h=480),
    )

    inject = AsyncMock()
    viewport = AsyncMock(return_value={"w": 640, "h": 480})
    atomic_capture = AsyncMock(return_value=capture)
    monkeypatch.setattr("worldsim.phase_4.pvpo_browser_config.inject_animation_killer", inject)
    monkeypatch.setattr("worldsim.phase_4.pvpo_cdp.runtime_evaluate_value", viewport)
    monkeypatch.setattr(
        "worldsim.phase_4.pvpo_capture.atomic_capture_with_visibility", atomic_capture
    )

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
        capturing=None,
        cdp_timeout_s=5.0,
    )

    summary = json.loads((tmp_path / "pvpo" / "capture_summary.json").read_text())
    assert summary["status"] == "ok"
    assert summary["steps_seen"] == 1
    assert summary["steps_captured"] == 1
    assert (tmp_path / "pvpo" / "step_1.json").exists()
    assert (tmp_path / "screenshots" / "step_1.png").read_bytes() == b"png-bytes"


@pytest.mark.asyncio
async def test_pvpo_callback_degrades_on_cdp_timeout(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    class SlowSession:
        async def get_current_page(self):
            await asyncio.sleep(0.05)
            return SimpleNamespace()

    monkeypatch.setenv("WORLDSIM_PVPO_CDP_TIMEOUT_S", "0.01")
    callback = browser_use_agent._make_pvpo_step_callback(
        SlowSession(),
        tmp_path,
        "payload text",
    )

    await callback(SimpleNamespace(), SimpleNamespace(), 1)

    summary = json.loads((tmp_path / "pvpo" / "capture_summary.json").read_text())
    assert summary["status"] == "degraded"
    assert summary["steps_seen"] == 1
    assert summary["steps_captured"] == 0
    assert summary["first_issue_class"] == "current_page_unavailable"
    assert "timed out after 0.01s" in summary["first_issue_message"]


@pytest.mark.asyncio
async def test_pvpo_callback_adopts_new_task_target(tmp_path, monkeypatch: pytest.MonkeyPatch):
    page = SimpleNamespace(_target_id="foreign-1")
    cdp_session = object()

    class BrowserUseLikeSession:
        get_or_create_cdp_session = AsyncMock(return_value=cdp_session)

        async def get_current_page(self):
            return page

    capture = StepCapture(
        screenshot_png=b"png-bytes",
        visibility_vec=[],
        background_color=(255, 255, 255),
        has_damage=False,
        clip=Rect(x=0, y=0, w=640, h=480),
    )
    inject = AsyncMock()
    viewport = AsyncMock(return_value={"w": 640, "h": 480})
    atomic_capture = AsyncMock(return_value=capture)
    monkeypatch.setattr("worldsim.phase_4.pvpo_browser_config.inject_animation_killer", inject)
    monkeypatch.setattr("worldsim.phase_4.pvpo_cdp.runtime_evaluate_value", viewport)
    monkeypatch.setattr(
        "worldsim.phase_4.pvpo_capture.atomic_capture_with_visibility", atomic_capture
    )

    owned_target_ids = {"target-1"}
    callback = browser_use_agent._make_pvpo_step_callback(
        BrowserUseLikeSession(),
        tmp_path,
        "payload text",
        owned_target_ids=owned_target_ids,
    )

    await callback(SimpleNamespace(), SimpleNamespace(), 1)

    summary = json.loads((tmp_path / "pvpo" / "capture_summary.json").read_text())
    assert summary["status"] == "ok"
    assert summary["steps_captured"] == 1
    assert owned_target_ids == {"target-1", "foreign-1"}
    BrowserUseLikeSession.get_or_create_cdp_session.assert_awaited_once_with(
        target_id="foreign-1",
        focus=False,
    )


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
async def test_cleanup_external_cdp_state_clears_storage_cookies_and_page(
    monkeypatch: pytest.MonkeyPatch,
):
    agent = browser_use_agent.BrowserUseAgent(llm=object())
    agent._pvpo_cdp_url = "http://127.0.0.1:9222"
    agent._owned_target_ids = {"target-1"}
    agent._task_origins = {"https://closed.example"}
    page_one = SimpleNamespace(_target_id="target-1")
    page_two = SimpleNamespace(_target_id="target-2")
    cdp_session = object()
    clear_data_for_origin = AsyncMock()

    class _StorageSender:
        clearDataForOrigin = clear_data_for_origin

    class _SendRoot:
        Storage = _StorageSender()

    class _CDPClient:
        send = _SendRoot()

    class FakeSession:
        get_pages = AsyncMock(return_value=[page_one, page_two])
        get_or_create_cdp_session = AsyncMock(side_effect=[cdp_session, cdp_session])
        clear_cookies = AsyncMock()
        close_page = AsyncMock()
        cdp_client = _CDPClient()

    runtime_eval = AsyncMock(return_value=True)
    monkeypatch.setattr("worldsim.phase_4.pvpo_cdp.runtime_evaluate", runtime_eval)
    page_one.get_url = AsyncMock(return_value="https://one.example/path")
    page_two.get_url = AsyncMock(return_value="https://two.example/other")

    session = FakeSession()
    await agent._cleanup_external_cdp_state(session)

    session.get_pages.assert_awaited_once()
    assert session.get_or_create_cdp_session.await_args_list == [
        call(target_id="target-1", focus=False),
        call(target_id="target-2", focus=False),
    ]
    assert runtime_eval.await_count == 2
    assert clear_data_for_origin.await_args_list == [
        call(params={"origin": "https://closed.example", "storageTypes": "all"}),
        call(params={"origin": "https://one.example", "storageTypes": "all"}),
        call(params={"origin": "https://two.example", "storageTypes": "all"}),
    ]
    session.clear_cookies.assert_awaited_once()
    assert session.close_page.await_args_list == [
        call(page_one),
        call(page_two),
    ]


@pytest.mark.asyncio
async def test_network_trace_recorder_adopts_new_page_targets():
    adopted_targets = {"target-1"}
    enabled = AsyncMock()
    page_target = SimpleNamespace(target_id="target-2")

    class FakeBrowserSession:
        session_manager = SimpleNamespace(get_all_page_targets=lambda: [page_target])
        cdp_client = SimpleNamespace()

        async def get_or_create_cdp_session(self, target_id, focus=False):
            _ = focus
            return SimpleNamespace(
                session_id=f"session-{target_id}",
                cdp_client=SimpleNamespace(
                    send=SimpleNamespace(
                        Network=SimpleNamespace(enable=enabled),
                        Page=SimpleNamespace(enable=enabled),
                    )
                ),
            )

    recorder = browser_use_agent._NetworkTraceRecorder(
        FakeBrowserSession(),
        Path("/tmp/task"),
        target_filter=adopted_targets,
    )

    await recorder._enable_current_page_sessions()

    assert adopted_targets == {"target-1", "target-2"}


def test_network_trace_redacts_configured_auth_header_names():
    redacted = browser_use_agent._NetworkTraceRecorder._redact_trace_entry(
        {
            "url": "http://reddit.test/f/news",
            "query_params": {},
            "headers": {
                "X-Postmill-Auto-Login": "alice:pw",
                "X-Visible": "ok",
            },
            "response_headers": {},
        },
        sensitive_header_names={"X-Postmill-Auto-Login"},
    )

    assert redacted["headers"]["X-Postmill-Auto-Login"] == "<redacted>"
    assert redacted["headers"]["X-Visible"] == "ok"


@pytest.mark.asyncio
async def test_network_trace_stop_returns_evaluator_trace_with_auth_headers_redacted(
    monkeypatch, tmp_path
):
    recorder = browser_use_agent._NetworkTraceRecorder(
        SimpleNamespace(cdp_client=None),
        tmp_path,
        sensitive_header_names={"X-Postmill-Auto-Login"},
    )
    monkeypatch.setattr(
        recorder,
        "_finalize_trace",
        lambda: [
            {
                "url": "http://reddit.test/f/news?q=needle&token=secret#frag",
                "query_params": {"q": ["needle"]},
                "headers": {
                    "X-Postmill-Auto-Login": "alice:pw",
                    "Referer": "http://reddit.test/ref?token=secret#frag",
                },
                "response_headers": {
                    "Set-Cookie": "sid=secret",
                    "Location": "http://reddit.test/done?token=secret#frag",
                },
                "response_cookies": {"sid": "secret"},
                "post_data": "payload=needle",
                "redirect_chain": [
                    {"url": "http://reddit.test/login?token=secret#frag", "status": 302}
                ],
            }
        ],
    )
    recorder._nav_events = [
        {
            "url": "http://reddit.test/f/news?token=secret#frag",
            "navigation_type": "Navigation",
            "timestamp": 1700000000.0,
            "kind": "document",
            "pageref": "page_1",
        }
    ]

    trace = await recorder.stop()

    assert trace[0]["headers"]["X-Postmill-Auto-Login"] == "<redacted>"
    assert trace[0]["headers"]["Referer"] == "http://reddit.test/ref?token=secret#frag"
    assert trace[0]["response_headers"]["Location"] == ("http://reddit.test/done?token=secret#frag")
    assert trace[0]["query_params"] == {"q": ["needle"]}
    assert trace[0]["post_data"] == "payload=needle"
    assert trace[0]["response_cookies"] == {"sid": "<redacted>"}

    persisted_trace = json.loads((tmp_path / "network_trace.json").read_text())
    assert persisted_trace[0]["url"] == (
        "http://reddit.test/f/news?q=%3Credacted%3E&token=%3Credacted%3E"
    )
    assert persisted_trace[0]["headers"]["Referer"] == (
        "http://reddit.test/ref?token=%3Credacted%3E"
    )
    assert persisted_trace[0]["response_headers"]["Location"] == (
        "http://reddit.test/done?token=%3Credacted%3E"
    )
    assert persisted_trace[0]["query_params"] == {"q": ["<redacted>"]}
    assert persisted_trace[0]["post_data"] == "<redacted>"
    assert persisted_trace[0]["response_cookies"] == {"sid": "<redacted>"}
    assert persisted_trace[0]["redirect_chain"] == [
        {"url": "http://reddit.test/login?token=%3Credacted%3E", "status": 302}
    ]

    nav_trace = json.loads((tmp_path / "navigation_trace.json").read_text())
    assert nav_trace[0]["url"] == "http://reddit.test/f/news?token=%3Credacted%3E"

    har = json.loads((tmp_path / "network.har").read_text())
    assert har["log"]["pages"][0]["title"] == ("http://reddit.test/f/news?token=%3Credacted%3E")
    har_entry = har["log"]["entries"][0]
    har_request_headers = {item["name"]: item["value"] for item in har_entry["request"]["headers"]}
    har_response_headers = {
        item["name"]: item["value"] for item in har_entry["response"]["headers"]
    }
    assert har_request_headers["Referer"] == "http://reddit.test/ref?token=%3Credacted%3E"
    assert har_response_headers["Location"] == "http://reddit.test/done?token=%3Credacted%3E"


@pytest.mark.asyncio
async def test_reset_remote_browser_for_task_closes_old_pages_and_creates_fresh_target():
    agent = browser_use_agent.BrowserUseAgent(llm=object())
    agent._pvpo_cdp_url = "http://127.0.0.1:9222"
    old_one = SimpleNamespace(_target_id="old-1")
    old_two = SimpleNamespace(_target_id="old-2", goto=AsyncMock())
    cdp_session = object()

    class FakeSession:
        get_pages = AsyncMock(return_value=[old_one, old_two])
        get_current_page = AsyncMock(return_value=old_two)
        get_or_create_cdp_session = AsyncMock(side_effect=[cdp_session, cdp_session, cdp_session])
        clear_cookies = AsyncMock()
        close_page = AsyncMock()
        new_page = AsyncMock()
        cdp_client = SimpleNamespace(
            send=SimpleNamespace(Storage=SimpleNamespace(clearDataForOrigin=AsyncMock()))
        )

    session = FakeSession()
    await agent._reset_remote_browser_for_task(session)

    old_two.goto.assert_awaited_once_with("about:blank")
    session.close_page.assert_awaited_once_with(old_one)
    session.clear_cookies.assert_awaited_once()
    session.new_page.assert_not_awaited()
    session.get_or_create_cdp_session.assert_any_await(target_id="old-2", focus=True)
    assert agent._owned_target_ids == {"old-2"}
    assert agent._primary_target_id == "old-2"


@pytest.mark.asyncio
async def test_reset_remote_browser_for_task_closes_failed_retained_page_before_replacing():
    agent = browser_use_agent.BrowserUseAgent(llm=object())
    agent._pvpo_cdp_url = "http://127.0.0.1:9222"
    broken_page = SimpleNamespace(
        _target_id="old-1", goto=AsyncMock(side_effect=RuntimeError("boom"))
    )
    fresh = SimpleNamespace(_target_id="fresh-1")
    cdp_session = object()

    class FakeSession:
        get_pages = AsyncMock(return_value=[broken_page])
        get_or_create_cdp_session = AsyncMock(return_value=cdp_session)
        clear_cookies = AsyncMock()
        close_page = AsyncMock()
        new_page = AsyncMock(return_value=fresh)
        cdp_client = SimpleNamespace(
            send=SimpleNamespace(Storage=SimpleNamespace(clearDataForOrigin=AsyncMock()))
        )

    session = FakeSession()
    await agent._reset_remote_browser_for_task(session)

    session.close_page.assert_awaited_once_with(broken_page)
    session.new_page.assert_awaited_once_with("about:blank")
    session.get_or_create_cdp_session.assert_any_await(target_id="fresh-1", focus=True)
    assert agent._owned_target_ids == {"fresh-1"}
    assert agent._primary_target_id == "fresh-1"


@pytest.mark.asyncio
async def test_reset_remote_browser_for_task_preserves_storage_state_auth():
    agent = browser_use_agent.BrowserUseAgent(llm=object())
    agent._pvpo_cdp_url = "http://127.0.0.1:9222"
    agent._preserve_remote_auth_state = True
    focused = SimpleNamespace(_target_id="focused-1", goto=AsyncMock())
    extra = SimpleNamespace(_target_id="extra-1")
    cdp_session = object()
    clear_storage = AsyncMock()

    class FakeSession:
        get_pages = AsyncMock(return_value=[extra, focused])
        get_current_page = AsyncMock(return_value=focused)
        get_or_create_cdp_session = AsyncMock(return_value=cdp_session)
        clear_cookies = AsyncMock()
        close_page = AsyncMock()
        new_page = AsyncMock()
        cdp_client = SimpleNamespace(
            send=SimpleNamespace(Storage=SimpleNamespace(clearDataForOrigin=clear_storage))
        )

    session = FakeSession()
    await agent._reset_remote_browser_for_task(session)

    focused.goto.assert_awaited_once_with("about:blank")
    session.close_page.assert_awaited_once_with(extra)
    session.clear_cookies.assert_not_awaited()
    clear_storage.assert_not_awaited()
    assert agent._browser_runtime["reset_preserved_auth_state"] is True


@pytest.mark.asyncio
async def test_cleanup_external_cdp_state_clears_storage_state_auth_after_task():
    agent = browser_use_agent.BrowserUseAgent(llm=object())
    agent._pvpo_cdp_url = "http://127.0.0.1:9222"
    agent._preserve_remote_auth_state = True
    agent._task_origins = {"http://example.test"}
    page = SimpleNamespace(
        _target_id="target-1", get_url=AsyncMock(return_value="http://example.test/path")
    )
    clear_storage = AsyncMock()

    class FakeSession:
        get_pages = AsyncMock(return_value=[page])
        clear_cookies = AsyncMock()
        close_page = AsyncMock()
        cdp_client = SimpleNamespace(
            send=SimpleNamespace(Storage=SimpleNamespace(clearDataForOrigin=clear_storage))
        )

    session = FakeSession()
    await agent._cleanup_external_cdp_state(session)

    clear_storage.assert_awaited_once_with(
        params={"origin": "http://example.test", "storageTypes": "all"}
    )
    session.clear_cookies.assert_awaited_once()
    session.close_page.assert_awaited_once_with(page)
    assert agent._browser_runtime["cleanup_preserved_auth_state"] is False


@pytest.mark.asyncio
async def test_run_storage_state_remote_pvpo_reaches_session_start(monkeypatch, tmp_path):
    import browser_use

    class FakeHistory:
        history: ClassVar[list[object]] = []

        def save_to_file(self, path):
            path.write_text('{"history":[]}')

        def screenshot_paths(self):
            return []

        def is_done(self):
            return True

        def final_result(self):
            return "ok"

        def errors(self):
            return []

    class FakeAgent:
        def __init__(self, **kwargs):
            self.history = FakeHistory()

        async def run(self, max_steps=50):
            return self.history

    class FakeBrowserSession:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def start(self):
            return None

        async def kill(self):
            return None

    runner = browser_use_agent.BrowserUseAgent(llm=object())

    monkeypatch.setattr(browser_use, "Agent", FakeAgent)
    monkeypatch.setattr(browser_use, "BrowserSession", FakeBrowserSession)
    monkeypatch.setattr(
        browser_use_agent,
        "_resolve_auth",
        lambda *args, **kwargs: ({"storage_state": "/tmp/state.json"}, []),
    )
    monkeypatch.setattr(
        runner,
        "_reset_remote_browser_for_task",
        AsyncMock(),
    )
    monkeypatch.setattr(
        browser_use_agent._NetworkTraceRecorder,
        "start",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        browser_use_agent._NetworkTraceRecorder,
        "stop",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        runner,
        "_cleanup_external_cdp_state",
        AsyncMock(return_value=None),
    )

    result = await runner.run(
        task="task",
        server_url="http://example.test",
        task_dir=tmp_path / "task",
        auth_mechanism={"type": "storage_state"},
        pvpo_cdp_url="http://127.0.0.1:9222",
    )

    assert result.status == "success"
    runner._reset_remote_browser_for_task.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_writes_browser_runtime_after_cleanup(monkeypatch: pytest.MonkeyPatch, tmp_path):
    import browser_use

    observed: dict[str, object] = {}
    page = SimpleNamespace(
        _target_id="owned-1",
        goto=AsyncMock(),
        get_url=AsyncMock(return_value="https://example.test/path"),
    )
    cdp_session = object()

    class FakeHistory:
        history: ClassVar[list[object]] = [object()]

        def save_to_file(self, path):
            path.write_text('{"history":[{"step":1}]}')

        def screenshot_paths(self):
            return []

        def is_done(self):
            return True

        def final_result(self):
            return "ok"

        def errors(self):
            return []

    class FakeRecorder:
        def __init__(
            self,
            browser_session,
            task_dir,
            *,
            target_filter=None,
            sensitive_header_names=None,
        ):
            _ = browser_session, task_dir
            observed["target_filter"] = set(target_filter or ())
            observed["sensitive_header_names"] = set(sensitive_header_names or ())

        async def start(self):
            return None

        async def stop(self):
            return []

    class FakeAgent:
        def __init__(self, **kwargs):
            self.history = FakeHistory()

        async def run(self, max_steps: int = 1):
            _ = max_steps
            return self.history

    class FakeBrowserSession:
        def __init__(self, **kwargs):
            _ = kwargs
            self.cdp_client = SimpleNamespace(
                send=SimpleNamespace(Storage=SimpleNamespace(clearDataForOrigin=AsyncMock()))
            )
            self.clear_cookies = AsyncMock()
            self.close_page = AsyncMock()
            self.kill = AsyncMock()

        async def start(self):
            return None

        async def get_pages(self):
            return [page]

        async def get_current_page(self):
            return page

        async def get_or_create_cdp_session(self, *, target_id=None, focus=False):
            _ = target_id, focus
            return cdp_session

        async def new_page(self, url=None):
            _ = url
            return page

    monkeypatch.setattr(browser_use, "Agent", FakeAgent)
    monkeypatch.setattr(browser_use, "BrowserSession", FakeBrowserSession)
    monkeypatch.setattr(browser_use_agent, "_NetworkTraceRecorder", FakeRecorder)
    monkeypatch.setattr(
        browser_use_agent,
        "_make_pvpo_step_callback",
        lambda *args, **kwargs: AsyncMock(),
    )

    agent = browser_use_agent.BrowserUseAgent(llm=object())
    result = await agent.run(
        task="Open the page",
        server_url="https://example.test",
        task_dir=tmp_path / "task",
        start_urls=["https://example.test"],
        pvpo_cdp_url="http://127.0.0.1:9222",
    )

    assert result.status == "success"
    assert observed["target_filter"] == {"owned-1"}

    runtime = json.loads((tmp_path / "task" / "browser_runtime.json").read_text())
    assert runtime["cleanup_closed_targets"] == 1
    assert runtime["cleanup_target_ids"] == ["owned-1"]
    assert runtime["cleanup_origins"] == ["https://example.test"]


@pytest.mark.asyncio
async def test_run_rejects_authenticated_off_origin_start_url(tmp_path):
    agent = browser_use_agent.BrowserUseAgent(llm=object())
    with pytest.raises(
        browser_use_agent.AuthArtifactMissingError,
        match="off-origin start_urls",
    ):
        await agent.run(
            task="Open the page",
            server_url="https://trusted.test",
            task_dir=tmp_path / "task",
            start_urls=["https://evil.test"],
            auth_mechanism={
                "type": "http_basic",
                "http_basic": {"username": "admin", "password": "pw"},
            },
        )


@pytest.mark.asyncio
async def test_run_cleans_up_session_when_remote_reset_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    import browser_use

    observed: dict[str, object] = {}

    class FakeBrowserSession:
        def __init__(self, **kwargs):
            _ = kwargs
            self.kill = AsyncMock()
            observed["session"] = self

        async def start(self):
            return None

    monkeypatch.setattr(browser_use, "BrowserSession", FakeBrowserSession)
    monkeypatch.setattr(browser_use, "Agent", object)
    monkeypatch.setattr(
        browser_use_agent.BrowserUseAgent,
        "_reset_remote_browser_for_task",
        AsyncMock(side_effect=RuntimeError("reset boom")),
    )

    agent = browser_use_agent.BrowserUseAgent(llm=object())
    with pytest.raises(RuntimeError, match="reset boom"):
        await agent.run(
            task="Open the page",
            server_url="https://example.test",
            task_dir=tmp_path / "task",
            start_urls=["https://example.test"],
            pvpo_cdp_url="http://127.0.0.1:9222",
        )

    session = observed["session"]
    session.kill.assert_awaited_once()
