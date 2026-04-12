from __future__ import annotations

import json

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
