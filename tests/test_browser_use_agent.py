from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from worldsim import browser_use_agent


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


@pytest.mark.asyncio
async def test_browser_use_agent_retries_startup_and_cleans_temp_profile(monkeypatch, tmp_path):
    screenshot = tmp_path / "source.png"
    screenshot.write_bytes(b"png")
    profile_dir = Path("/tmp") / f"browser-use-user-data-dir-{tmp_path.name}"
    profile_dir.mkdir()
    start_attempts = {"count": 0}
    created_args = {}

    class FakeSession:
        def __init__(self, *, headless, keep_alive, args):
            created_args["args"] = list(args)
            self.browser_profile = SimpleNamespace(user_data_dir=str(profile_dir))
            self.cdp_client = None

        async def start(self):
            start_attempts["count"] += 1
            if start_attempts["count"] < 3:
                raise RuntimeError("startup failed")

        async def kill(self):
            return None

    class FakeAgent:
        def __init__(self, **kwargs):
            self.history = _FakeHistory(str(screenshot))

        async def run(self):
            return self.history

    async def fake_sleep(delay):
        return None

    fake_module = types.ModuleType("browser_use")
    fake_module.Agent = FakeAgent
    fake_module.BrowserSession = FakeSession
    monkeypatch.setitem(sys.modules, "browser_use", fake_module)
    monkeypatch.setattr(browser_use_agent.asyncio, "sleep", fake_sleep)

    agent = browser_use_agent.BrowserUseAgent(llm=object())
    result = await agent.run("Check the order", "http://shopping.test", tmp_path / "task")

    assert start_attempts["count"] == 3
    assert "--no-sandbox" in created_args["args"]
    assert "--disable-software-rasterizer" in created_args["args"]
    assert result.status == "failure"
    assert not Path(profile_dir).exists()


@pytest.mark.asyncio
async def test_browser_use_agent_cleans_temp_profile_when_startup_never_succeeds(
    monkeypatch, tmp_path
):
    profile_dir = Path("/tmp") / f"browser-use-user-data-dir-fail-{tmp_path.name}"
    profile_dir.mkdir(exist_ok=True)
    kill_calls = {"count": 0}

    class FakeSession:
        def __init__(self, *, headless, keep_alive, args):
            self.browser_profile = SimpleNamespace(user_data_dir=str(profile_dir))
            self.cdp_client = None

        async def start(self):
            raise RuntimeError("startup failed")

        async def kill(self):
            kill_calls["count"] += 1

    fake_module = types.ModuleType("browser_use")
    fake_module.Agent = object
    fake_module.BrowserSession = FakeSession
    monkeypatch.setitem(sys.modules, "browser_use", fake_module)

    async def fake_sleep(delay):
        return None

    monkeypatch.setattr(browser_use_agent.asyncio, "sleep", fake_sleep)

    agent = browser_use_agent.BrowserUseAgent(llm=object())

    with pytest.raises(RuntimeError, match="startup failed"):
        await agent.run("Check the order", "http://shopping.test", tmp_path / "task")

    assert kill_calls["count"] == 1
    assert not profile_dir.exists()
