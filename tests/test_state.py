from __future__ import annotations

import json

from worldsim.state import get_state_dir, load_state, save_state


def test_state_dir_honors_runtime_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))

    save_state("phase_1", status="running")

    assert get_state_dir() == tmp_path
    assert json.loads((tmp_path / "pipeline_state.json").read_text())["step"] == "phase_1"
    assert load_state()["status"] == "running"
