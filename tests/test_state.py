from __future__ import annotations

import json
from argparse import Namespace

from worldsim import main as worldsim_main
from worldsim.browser_use_agent import AgentResult
from worldsim.eval_worker_pool import load_completed_results
from worldsim.state import get_state_dir, load_state, save_state
from worldsim.trajectory import save_result


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


def test_load_state_rejects_valid_non_object_json(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
    (tmp_path / "pipeline_state.json").write_text('["not", "a", "state"]')

    assert load_state() is None


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
        captured["agent_model"] = args.agent_model
        captured["agent_provider"] = args.agent_provider
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
    assert captured["agent_model"] == "demo-model"
    assert captured["agent_provider"] is None
    assert str(captured["logs_dir"]) == str(custom_logs)


def test_dispatch_resume_restores_saved_agent_settings_when_not_overridden(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    custom_logs = tmp_path / "custom-logs"
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(custom_logs))
    save_state(
        "phase_4",
        status="running",
        instances_path="/tmp/instances.json",
        agent_model="claude-sonnet-4-6",
        agent_provider="anthropic",
    )
    monkeypatch.delenv("WORLDSIM_STATE_DIR")

    captured = {}

    def fake_dispatch_phase(args):
        captured["phase"] = args.phase
        captured["agent_model"] = args.agent_model
        captured["agent_provider"] = args.agent_provider
        return 0

    monkeypatch.setattr(worldsim_main, "_dispatch_phase", fake_dispatch_phase)

    rc = worldsim_main._dispatch_resume(Namespace())

    assert rc == 0
    assert captured["phase"] == "4"
    assert captured["agent_model"] == "claude-sonnet-4-6"
    assert captured["agent_provider"] == "anthropic"


def test_dispatch_resume_restores_saved_generate_novel_for_phase_1(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    custom_logs = tmp_path / "custom-logs"
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(custom_logs))
    save_state(
        "phase_1",
        status="running",
        benchmark_path="/tmp/benchmark",
        manifest_path="/tmp/manifest.json",
        generate_novel=True,
    )
    monkeypatch.delenv("WORLDSIM_STATE_DIR")

    captured = {}

    def fake_dispatch_phase(args):
        captured["phase"] = args.phase
        captured["generate_novel"] = args.generate_novel
        return 0

    monkeypatch.setattr(worldsim_main, "_dispatch_phase", fake_dispatch_phase)

    rc = worldsim_main._dispatch_resume(Namespace())

    assert rc == 0
    assert captured["phase"] == "1"
    assert captured["generate_novel"] is True


def test_dispatch_resume_handles_non_object_state_json(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
    (tmp_path / "pipeline_state.json").write_text('"bad-state"')

    rc = worldsim_main._dispatch_resume(Namespace())

    assert rc == 1
    assert "No pipeline state found" in capsys.readouterr().err


# ── Per-task resume (load_completed_results) ────────────────────────────


def test_load_completed_results_finds_valid_results(tmp_path):
    """Completed tasks with valid result.json are returned."""
    from worldsim.task_paths import safe_task_path_component

    task_dir = tmp_path / safe_task_path_component("42")
    task_dir.mkdir()
    (task_dir / "result.json").write_text(
        json.dumps({"task_id": "42", "passed": True, "message": "ok"})
    )
    task_dir2 = tmp_path / safe_task_path_component("99")
    task_dir2.mkdir()
    (task_dir2 / "result.json").write_text(
        json.dumps({"task_id": "99", "passed": False, "message": "reward mismatch"})
    )

    completed = load_completed_results(tmp_path)
    assert set(completed.keys()) == {"42", "99"}
    assert completed["42"]["passed"] is True
    assert completed["99"]["passed"] is False
    # trajectory_dir is reconstructed from the subdirectory path
    assert completed["42"]["trajectory_dir"] == str(task_dir)


def test_load_completed_results_skips_corrupt_files(tmp_path):
    """Corrupt result.json files are skipped, allowing re-run."""
    from worldsim.task_paths import safe_task_path_component

    task_dir = tmp_path / safe_task_path_component("7")
    task_dir.mkdir()
    (task_dir / "result.json").write_text("{truncated")

    completed = load_completed_results(tmp_path)
    assert completed == {}


def test_load_completed_results_ignores_incomplete_tasks(tmp_path):
    """Tasks with history.json but no result.json are not counted as complete."""
    from worldsim.task_paths import safe_task_path_component

    task_dir = tmp_path / safe_task_path_component("incomplete")
    task_dir.mkdir()
    (task_dir / "history.json").write_text(json.dumps({"steps": []}))

    completed = load_completed_results(tmp_path)
    assert completed == {}


def test_load_completed_results_empty_dir(tmp_path):
    """Empty task_dir_root returns an empty dict."""
    completed = load_completed_results(tmp_path)
    assert completed == {}


def test_load_completed_results_nonexistent_dir(tmp_path):
    """Nonexistent task_dir_root returns an empty dict."""
    completed = load_completed_results(tmp_path / "does_not_exist")
    assert completed == {}


def test_load_completed_results_skips_variant_and_rerun_dirs(tmp_path):
    """Variant/rerun result.json files are skipped to avoid dict-key collision."""
    from worldsim.task_paths import safe_task_path_component

    # Initial run dir (name matches canonical path for task_id "10")
    initial_dir = tmp_path / safe_task_path_component("10")
    initial_dir.mkdir()
    (initial_dir / "result.json").write_text(
        json.dumps({"task_id": "10", "passed": True, "message": "initial"})
    )
    # Variant dir (suffixed name, should be skipped)
    variant_dir = tmp_path / safe_task_path_component("10_variant_0")
    variant_dir.mkdir()
    (variant_dir / "result.json").write_text(
        json.dumps({"task_id": "10", "passed": False, "message": "variant"})
    )
    # Eco-fix dir (suffixed name, should be skipped)
    ecofix_dir = tmp_path / safe_task_path_component("10__ecoval_1")
    ecofix_dir.mkdir()
    (ecofix_dir / "result.json").write_text(
        json.dumps({"task_id": "10", "passed": False, "message": "ecofix"})
    )

    completed = load_completed_results(tmp_path)
    assert "10" in completed
    assert completed["10"]["message"] == "initial"
    assert completed["10"]["trajectory_dir"] == str(initial_dir)


def test_save_state_preserves_task_dir_root(monkeypatch, tmp_path):
    """task_dir_root is persisted in pipeline_state.json for resume."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))

    save_state(
        "phase_3",
        status="running",
        task_dir_root="/logs/phase_3/20260412_120000",
    )

    state = load_state()
    assert state["task_dir_root"] == "/logs/phase_3/20260412_120000"


# ── Null/missing task_id edge cases ──────────────────────────────────────


def test_load_completed_results_skips_result_without_task_id(tmp_path):
    """result.json with no task_id key is skipped (task will re-run)."""
    # Dir name won't match any task_id since task_id is empty/missing,
    # so it's skipped regardless of dir naming.
    task_dir = tmp_path / "some_dir"
    task_dir.mkdir()
    (task_dir / "result.json").write_text(json.dumps({"passed": True, "message": "ok"}))

    completed = load_completed_results(tmp_path)
    assert completed == {}


def test_load_completed_results_skips_result_with_empty_task_id(tmp_path):
    """result.json with an empty-string task_id is skipped."""
    task_dir = tmp_path / "some_dir"
    task_dir.mkdir()
    (task_dir / "result.json").write_text(
        json.dumps({"task_id": "", "passed": True, "message": "ok"})
    )

    completed = load_completed_results(tmp_path)
    assert completed == {}


def test_load_completed_results_skips_result_with_null_task_id(tmp_path):
    """result.json with null task_id is skipped (str(None or '') == '')."""
    task_dir = tmp_path / "some_dir"
    task_dir.mkdir()
    (task_dir / "result.json").write_text(
        json.dumps({"task_id": None, "passed": True, "message": "ok"})
    )

    completed = load_completed_results(tmp_path)
    assert completed == {}


# ── Atomic write safety (save_result) ────────────────────────────────────


def _make_agent_result(**overrides):
    defaults = dict(
        elapsed=1.5,
        steps=3,
        is_done=True,
        final_result="done",
        status="success",
        errors=[],
        network_trace=[],
    )
    defaults.update(overrides)
    return AgentResult(**defaults)


def test_save_result_atomic_write_produces_valid_json(tmp_path):
    """save_result writes valid result.json with no leftover .tmp files."""
    task_dir = tmp_path / "task_atomic"

    save_result(task_dir, {"id": "atomic-1"}, _make_agent_result(), True, "ok")

    target = task_dir / "result.json"
    assert target.exists()
    data = json.loads(target.read_text())
    assert data["task_id"] == "atomic-1"
    assert data["passed"] is True
    assert data["elapsed"] == 1.5

    tmp_files = list(task_dir.glob("*.tmp"))
    assert tmp_files == []


def test_save_result_creates_parent_dirs(tmp_path):
    """save_result creates the task directory if it doesn't exist."""
    task_dir = tmp_path / "deep" / "nested" / "task_dir"

    save_result(task_dir, {"id": "nested-1"}, _make_agent_result(), False, "failed")

    assert (task_dir / "result.json").exists()
    data = json.loads((task_dir / "result.json").read_text())
    assert data["task_id"] == "nested-1"
    assert data["passed"] is False


def test_save_result_extra_fields_persisted(tmp_path):
    """Extra keyword args (outcome, ecologically_valid) are written to result.json."""
    task_dir = tmp_path / "task_extra"

    save_result(
        task_dir,
        {"id": "extra-1"},
        _make_agent_result(),
        True,
        "ok",
        outcome="complied",
        ecologically_valid=True,
    )

    data = json.loads((task_dir / "result.json").read_text())
    assert data["outcome"] == "complied"
    assert data["ecologically_valid"] is True
