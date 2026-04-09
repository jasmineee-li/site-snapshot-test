from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from agentlab.benchmarks.redteam import agent_browser_runner


def test_localize_start_url_maps_external_origin_to_local_server() -> None:
    localized = agent_browser_runner.localize_start_url(
        "http://127.0.0.1:8123",
        "https://mail.example.com/mail/u/0/?tab=primary#inbox",
    )

    assert localized == "http://127.0.0.1:8123/mail/u/0/?tab=primary#inbox"


def test_run_agent_browser_task_uses_isolated_session_and_writes_transcript(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task_dir = tmp_path / "task_0"
    monkeypatch.setattr(agent_browser_runner.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(agent_browser_runner, "_load_chat_model", lambda agent_config: object())

    actions = iter(
        [
            agent_browser_runner.AgentBrowserAction(action="click", target="@e2", reason="open"),
            agent_browser_runner.AgentBrowserAction(action="done", result="complete", reason="done"),
        ]
    )
    monkeypatch.setattr(agent_browser_runner, "_plan_next_action", lambda **kwargs: next(actions))

    commands: list[dict[str, object]] = []

    def fake_run(cmd, cwd, capture_output, text, timeout, check):
        commands.append({"cmd": cmd, "cwd": cwd, "timeout": timeout})
        stdout = ""
        if "snapshot" in cmd:
            stdout = json.dumps({"data": {"snapshot": "@e2 button Open"}})
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(agent_browser_runner.subprocess, "run", fake_run)

    result = agent_browser_runner.run_agent_browser_task(
        server_url="http://127.0.0.1:8123",
        start_url="http://127.0.0.1:8123/inbox",
        instruction="Open the inbox",
        task_dir=task_dir,
        agent_config="fake.Agent",
        max_steps=3,
        timeout=17,
    )

    assert result.passed is True
    assert result.message == "complete"
    assert commands[0]["cmd"] == [
        "agent-browser",
        "--session",
        "redteam-task_0",
        "open",
        "http://127.0.0.1:8123/inbox",
    ]
    assert any(entry["cmd"][-2:] == ["click", "@e2"] for entry in commands)
    assert commands[-1]["cmd"][-1] == "close"

    transcript = json.loads((task_dir / "agent_browser_transcript.json").read_text(encoding="utf-8"))
    assert any(entry.get("type") == "model_action" for entry in transcript)
    assert any(entry.get("type") == "command" for entry in transcript)


def test_run_cli_records_failed_commands(tmp_path: Path, monkeypatch) -> None:
    transcript: list[dict[str, object]] = []

    def fake_run(cmd, cwd, capture_output, text, timeout, check):
        raise subprocess.CalledProcessError(2, cmd, output="out", stderr="err")

    monkeypatch.setattr(agent_browser_runner.subprocess, "run", fake_run)

    with pytest.raises(subprocess.CalledProcessError):
        agent_browser_runner._run_cli(
            "agent-browser",
            ["click", "@e1"],
            cwd=tmp_path,
            timeout=5,
            transcript=transcript,
            cli_prefix=["--session", "demo"],
        )

    assert len(transcript) == 1
    assert transcript[0]["command"] == ["agent-browser", "--session", "demo", "click", "@e1"]
    assert transcript[0]["stdout"] == "out"
    assert transcript[0]["stderr"] == "err"
    assert transcript[0]["returncode"] == 2
