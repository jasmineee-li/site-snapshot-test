from __future__ import annotations

import json
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
