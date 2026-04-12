from __future__ import annotations

import json
from argparse import Namespace

from worldsim import main as worldsim_main
from worldsim.state import get_state_dir, load_state, save_state


def test_state_dir_honors_runtime_env_override(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))

    save_state("phase_1", status="running", benchmark_path="/tmp/benchmark")

    assert get_state_dir() == tmp_path
    saved_state = json.loads((tmp_path / "pipeline_state.json").read_text())
    assert saved_state["step"] == "phase_1"
    assert saved_state["logs_dir"] == str(tmp_path)
    assert load_state()["status"] == "running"


def test_load_state_follows_resume_pointer_without_env_override(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    custom_logs = tmp_path / "custom-logs"
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(custom_logs))

    save_state("phase_3", status="running", instances_path="/tmp/instances.json")

    monkeypatch.delenv("WORLDSIM_STATE_DIR")

    state = load_state()

    assert state is not None
    assert state["step"] == "phase_3"
    assert state["instances_path"] == "/tmp/instances.json"
    assert state["logs_dir"] == str(custom_logs)


def test_dispatch_resume_restores_logs_dir_from_state(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    custom_logs = tmp_path / "custom-logs"
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(custom_logs))
    save_state("phase_3", status="running", instances_path="/tmp/instances.json")
    monkeypatch.delenv("WORLDSIM_STATE_DIR")

    captured = {}

    def fake_dispatch_phase(args):
        captured["phase"] = args.phase
        captured["instances"] = args.instances
        captured["logs_dir"] = get_state_dir()
        return 0

    monkeypatch.setattr(worldsim_main, "_dispatch_phase", fake_dispatch_phase)

    rc = worldsim_main._dispatch_resume(
        Namespace(
            benchmark=None,
            config=None,
            instances=None,
            agent_model="demo-model",
            agent_provider=None,
        )
    )

    assert rc == 0
    assert captured["phase"] == "3"
    assert str(captured["logs_dir"]) == str(custom_logs)
