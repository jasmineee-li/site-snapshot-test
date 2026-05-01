"""Browser-Use PVPO integration against chrome-headless-shell.

This test covers the runtime path that previously broke in production:
BrowserUseAgent creates a real Browser-Use ``BrowserSession`` connected to
an instance-bound external chrome-headless-shell container over CDP, a
stub Agent drives one page load, and the PVPO callback must emit
deterministic artifacts.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen

import pytest

from worldsim.browser_use_agent import BrowserUseAgent

CDP_HOST = "127.0.0.1"
CDP_PORT = 9222


def _chrome_headless_shell_reachable() -> bool:
    try:
        with socket.create_connection((CDP_HOST, CDP_PORT), timeout=1):
            pass
    except OSError:
        return False
    try:
        with urlopen(f"http://{CDP_HOST}:{CDP_PORT}/json/version", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


class _FixtureServer:
    def __init__(self, html: str):
        self._html = html
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> str:
        html = self._html.encode("utf-8")

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)

            def log_message(self, format, *args):
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return f"http://127.0.0.1:{self._server.server_port}/"

    def __exit__(self, exc_type, exc, tb):
        assert self._server is not None
        self._server.shutdown()
        self._server.server_close()
        assert self._thread is not None
        self._thread.join(timeout=5)


class _FakeHistory:
    def __init__(self):
        self.history = [object()]

    def save_to_file(self, path: Path):
        path.write_text('{"history":[{"step":1}]}', encoding="utf-8")

    def screenshot_paths(self):
        return []

    def is_done(self):
        return True

    def final_result(self):
        return "ok"

    def errors(self):
        return []


pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    not _chrome_headless_shell_reachable(),
    reason="chrome-headless-shell container not reachable on 127.0.0.1:9222",
)
@pytest.mark.asyncio
async def test_browser_use_agent_emits_pvpo_artifacts_over_cdp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import browser_use

    payload = "PVPO visible payload 123"
    # This integration smoke may run against a manually-started local
    # chrome-headless-shell without Docker restart supervision. The dedicated
    # rigor-host preflight covers recycle supervision; keep this test focused
    # on artifact capture and avoid killing the developer's endpoint.
    monkeypatch.setenv("WORLDSIM_PVPO_BROWSER_RECYCLE", "0")
    html = f"""
<!doctype html>
<html>
  <head><meta charset="utf-8"></head>
  <body>
    <main>
      <p>{payload}</p>
    </main>
  </body>
</html>
"""

    with _FixtureServer(html) as url:
        class _FakeAgent:
            def __init__(self, **kwargs):
                self._session = kwargs["browser_session"]
                self._callback = kwargs["register_new_step_callback"]
                self._initial_actions = kwargs["initial_actions"]
                self.history = _FakeHistory()

            async def run(self, max_steps: int = 1):
                _ = max_steps
                current_page = await self._session.get_current_page()
                assert current_page is not None, "Browser-Use must expose a focused page"
                navigate_url = self._initial_actions[0]["navigate"]["url"]
                await self._session.navigate_to(navigate_url)
                await asyncio.sleep(0.5)
                await self._callback(object(), object(), 1)
                return self.history

        monkeypatch.setattr(browser_use, "Agent", _FakeAgent)
        runner = BrowserUseAgent(llm=object(), timeout=60)
        result = await runner.run(
            task="Open the fixture page",
            server_url=url,
            task_dir=tmp_path / "task",
            start_urls=[url],
            payload_text=payload,
            pvpo_cdp_url=f"http://{CDP_HOST}:{CDP_PORT}",
        )

    assert result.status == "success"

    task_dir = tmp_path / "task"
    capture_summary = json.loads((task_dir / "pvpo" / "capture_summary.json").read_text())
    assert capture_summary["steps_seen"] == 1
    assert capture_summary["steps_captured"] == 1

    step_json = json.loads((task_dir / "pvpo" / "step_1.json").read_text())
    assert step_json["visibility_vec"], "visible payload should produce a non-empty visibility vector"
    assert (task_dir / "screenshots" / "step_1.png").exists()
    runtime = json.loads((task_dir / "browser_runtime.json").read_text())
    assert runtime.get("pvpo_navigation_tick_navigations", 0) >= 1
    assert runtime.get("pvpo_navigation_tick_frames", 0) > 0


@pytest.mark.skipif(
    not _chrome_headless_shell_reachable(),
    reason="chrome-headless-shell container not reachable on 127.0.0.1:9222",
)
@pytest.mark.asyncio
async def test_browser_use_agent_pvpo_scroll_moves_beginframe_controlled_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import browser_use
    from browser_use.browser.watchdogs.default_action_watchdog import DefaultActionWatchdog

    monkeypatch.setenv("WORLDSIM_PVPO_BROWSER_RECYCLE", "0")
    monkeypatch.setenv("WORLDSIM_PVPO_SCROLL_ACTION_TIMEOUT_S", "2")
    html = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <style>
      body { margin: 0; font-family: sans-serif; }
      .spacer { height: 5000px; padding: 24px; }
    </style>
  </head>
  <body>
    <main class="spacer">PVPO scroll canary</main>
  </body>
</html>
"""

    async def _scroll_y(session) -> float:
        cdp_session = await session.get_or_create_cdp_session()
        result = await cdp_session.cdp_client.send.Runtime.evaluate(
            params={
                "expression": "Number(window.scrollY || document.scrollingElement.scrollTop || 0)",
                "returnByValue": True,
            },
            session_id=cdp_session.session_id,
        )
        return float(result["result"]["value"])

    with _FixtureServer(html) as url:
        class _FakeAgent:
            def __init__(self, **kwargs):
                self._session = kwargs["browser_session"]
                self._initial_actions = kwargs["initial_actions"]
                self.history = _FakeHistory()

            async def run(self, max_steps: int = 1):
                _ = max_steps
                current_page = await self._session.get_current_page()
                assert current_page is not None, "Browser-Use must expose a focused page"
                navigate_url = self._initial_actions[0]["navigate"]["url"]
                await self._session.navigate_to(navigate_url)
                await asyncio.sleep(0.5)

                before_y = await _scroll_y(self._session)
                scrolled = await DefaultActionWatchdog._scroll_with_cdp_gesture(
                    SimpleNamespace(browser_session=self._session),
                    720,
                )
                after_y = await _scroll_y(self._session)

                assert scrolled is True
                assert after_y > before_y
                return self.history

        monkeypatch.setattr(browser_use, "Agent", _FakeAgent)
        runner = BrowserUseAgent(llm=object(), timeout=60)
        result = await runner.run(
            task="Open the fixture page and scroll",
            server_url=url,
            task_dir=tmp_path / "task",
            start_urls=[url],
            pvpo_cdp_url=f"http://{CDP_HOST}:{CDP_PORT}",
        )

    assert result.status == "success"
    runtime = json.loads((tmp_path / "task" / "browser_runtime.json").read_text())
    assert runtime.get("pvpo_navigation_tick_navigations", 0) >= 1
    assert runtime.get("pvpo_navigation_tick_frames", 0) > 0
    assert any(
        runtime.get(key, 0) > 0
        for key in (
            "pvpo_scroll_wheel_successes",
            "pvpo_scroll_wheel_late_successes",
            "pvpo_scroll_js_fallbacks",
        )
    )
