from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from warp_taskgen import main as worldsim_main
from warp_taskgen.browser_use_agent import AgentResult
from warp_taskgen.eval_worker_pool import load_completed_results
from warp_taskgen.resume_metadata import RESULT_FINGERPRINT_KEY
from warp_taskgen.run_transition import resolve_run_request
from warp_taskgen.state import bind_run_definition, get_state_dir, load_state, save_state
from warp_taskgen.trajectory import save_result


def test_state_dir_honors_runtime_env_override(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))

    save_state("phase_1", status="running", benchmark_path="/tmp/benchmark")

    assert get_state_dir() == tmp_path
    saved_state = json.loads((tmp_path / "pipeline_state.json").read_text())
    assert saved_state["step"] == "phase_1"
    assert saved_state["logs_dir"] == str(tmp_path)
    assert load_state()["status"] == "running"


def test_phase_2_process_pool_failed_text_fill_state_skips_pause_guard(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("WARP_TASKGEN_STATE_DIR", str(tmp_path))

    save_state(
        "phase_2",
        status="failed",
        phase_2_stage="text_fill",
        process_pool=True,
    )

    state = json.loads((tmp_path / "pipeline_state.json").read_text())
    assert state["status"] == "failed"
    assert state["phase_2_stage"] == "text_fill"
    assert state["process_pool"] is True


def test_state_dir_prefers_warp_taskgen_env_override(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    canonical_logs = tmp_path / "canonical"
    legacy_logs = tmp_path / "legacy"
    monkeypatch.setenv("WARP_TASKGEN_STATE_DIR", str(canonical_logs))
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(legacy_logs))

    save_state("phase_1", status="running", benchmark_path="/tmp/benchmark")

    assert get_state_dir() == canonical_logs
    assert (canonical_logs / "pipeline_state.json").is_file()
    assert not (legacy_logs / "pipeline_state.json").exists()


def test_run_definition_is_persisted_identically_and_remains_stable(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    state_dir = tmp_path / "run"
    monkeypatch.setenv("WARP_TASKGEN_STATE_DIR", str(state_dir))
    transition = resolve_run_request(
        {"agent_model": "model-a", "sites": ["gitlab"]},
        existing_state=None,
        new_run_id="run-stable",
    )
    assert transition.definition is not None

    with bind_run_definition(transition.definition):
        save_state("phase_1", status="running", agent_model="model-a", sites=["gitlab"])
        first = json.loads((state_dir / "pipeline_state.json").read_text())
        save_state("phase_2", status="running", agent_model="model-a", sites=["gitlab"])

    current = json.loads((state_dir / "pipeline_state.json").read_text())
    mirror = json.loads((tmp_path / "logs" / "last_run_state.json").read_text())
    assert first["run_definition"] == current["run_definition"]
    assert current["run_definition"] == mirror["run_definition"]
    assert current["run_definition"]["run_id"] == "run-stable"


def test_run_definition_context_is_root_scoped_and_reserved(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    primary = tmp_path / "primary"
    worker = tmp_path / "worker"
    monkeypatch.setenv("WARP_TASKGEN_STATE_DIR", str(primary))
    transition = resolve_run_request({}, existing_state=None, new_run_id="run-primary")
    assert transition.definition is not None

    with bind_run_definition(transition.definition):
        with pytest.raises(ValueError, match="reserved fields"):
            save_state("phase_1", status="running", run_definition={})
        monkeypatch.setenv("WARP_TASKGEN_STATE_DIR", str(worker))
        save_state("phase_4", status="running")

    worker_state = json.loads((worker / "pipeline_state.json").read_text())
    assert "run_definition" not in worker_state
    assert not (primary / "pipeline_state.json").exists()


def test_run_definition_context_rejects_existing_identity_mismatch(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WARP_TASKGEN_STATE_DIR", str(tmp_path))
    first = resolve_run_request({}, existing_state=None, new_run_id="run-one")
    second = resolve_run_request({}, existing_state=None, new_run_id="run-two")
    assert first.definition is not None and second.definition is not None
    with bind_run_definition(first.definition):
        save_state("phase_1", status="running")
    before = (tmp_path / "pipeline_state.json").read_bytes()

    with pytest.raises(ValueError, match="does not match"):
        with bind_run_definition(second.definition):
            save_state("phase_2", status="running")

    assert (tmp_path / "pipeline_state.json").read_bytes() == before


def test_fresh_cli_phase_binds_identity_before_first_checkpoint(monkeypatch, tmp_path):
    from warp_taskgen.cli import _impl as cli_impl

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WARP_TASKGEN_STATE_DIR", str(tmp_path / "fresh"))

    def fake_dispatch(_args):
        save_state("phase_1", status="running", agent_model="model-a")
        return 0

    monkeypatch.setattr(cli_impl, "_dispatch_phase_with_run_context", fake_dispatch)

    rc = cli_impl._dispatch_phase(
        Namespace(command="phase", phase="1", agent_model="model-a", sites="gitlab")
    )

    assert rc == 0
    state = json.loads((tmp_path / "fresh" / "pipeline_state.json").read_text())
    definition = state["run_definition"]
    assert definition["run_id"].startswith("run-")
    assert definition["legacy"] is False
    assert definition["contributions"]["phase_4"]["agent_model"] == "model-a"
    assert definition["contributions"]["phase_2"]["phase_2_text_model"]
    assert definition["contributions"]["pipeline"]["manifest_path"].endswith(
        "fresh/phase_0a/BENCHMARK_MANIFEST.json"
    )


def test_direct_phase_keeps_existing_legacy_state_without_logs_dir(monkeypatch, tmp_path):
    from warp_taskgen.cli import _impl as cli_impl

    monkeypatch.chdir(tmp_path)
    state_dir = tmp_path / "legacy"
    state_dir.mkdir()
    monkeypatch.setenv("WARP_TASKGEN_STATE_DIR", str(state_dir))
    (state_dir / "pipeline_state.json").write_text(
        json.dumps({"step": "phase_1", "status": "running"}),
        encoding="utf-8",
    )

    def fake_dispatch(_args):
        save_state("phase_1", status="running")
        return 0

    monkeypatch.setattr(cli_impl, "_dispatch_phase_with_run_context", fake_dispatch)

    assert cli_impl._dispatch_phase(Namespace(command="phase", phase="1")) == 0
    saved = json.loads((state_dir / "pipeline_state.json").read_text())
    assert "run_definition" not in saved


def test_top_level_identified_state_migrates_to_nested_envelope(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WARP_TASKGEN_STATE_DIR", str(tmp_path))
    projected = resolve_run_request({}, existing_state=None, new_run_id="run-top-level")
    assert projected.definition is not None
    top_level = {
        "step": "phase_1",
        "status": "running",
        "timestamp": "2026-08-11T12:00:00",
        "logs_dir": str(tmp_path),
        "run_definition_schema_version": 1,
        "run_id": "run-top-level",
        "source_run_id": None,
        "definition_digest": projected.definition.definition_digest,
    }
    (tmp_path / "pipeline_state.json").write_text(json.dumps(top_level), encoding="utf-8")
    transition = resolve_run_request({}, existing_state=top_level)
    assert transition.definition is not None

    with bind_run_definition(transition.definition):
        save_state("phase_2", status="running")

    saved = json.loads((tmp_path / "pipeline_state.json").read_text())
    assert saved["run_definition"]["run_id"] == "run-top-level"


def test_load_state_follows_resume_pointer_without_env_override(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    custom_logs = tmp_path / "custom-logs"
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(custom_logs))

    save_state("phase_3", status="running", instances_path="/tmp/instances.json")

    monkeypatch.delenv("WORLDSIM_STATE_DIR")

    state = load_state()

    assert state is not None
    assert state["step"] == "phase_3"
    assert state["instances_path"] == str(Path("/tmp/instances.json").resolve())
    assert state["logs_dir"] == str(custom_logs)


def test_load_state_reads_mirrored_state_without_status(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    custom_logs = tmp_path / "custom-logs"
    custom_logs.mkdir(parents=True, exist_ok=True)
    mirrored = tmp_path / "logs" / "last_run_state.json"
    mirrored.parent.mkdir(parents=True, exist_ok=True)
    mirrored.write_text(
        json.dumps(
            {
                "step": "phase_4",
                "timestamp": "2026-04-14T12:00:00",
                "logs_dir": str(custom_logs),
                "task_dir_root": str(custom_logs / "phase_4" / "run-1"),
            }
        )
    )

    state = load_state()

    assert state is not None
    assert state["step"] == "phase_4"
    assert state["status"] == "running"
    assert state["logs_dir"] == str(custom_logs)


def test_load_state_supports_legacy_resume_pointer(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    custom_logs = tmp_path / "custom-logs"
    custom_logs.mkdir(parents=True, exist_ok=True)
    (custom_logs / "pipeline_state.json").write_text(
        json.dumps(
            {
                "step": "phase_3",
                "status": "running",
                "timestamp": "2026-04-14T12:00:00",
                "logs_dir": str(custom_logs),
            }
        )
    )
    mirrored = tmp_path / "logs" / "last_run_state.json"
    mirrored.parent.mkdir(parents=True, exist_ok=True)
    mirrored.write_text(
        json.dumps(
            {
                "logs_dir": str(custom_logs),
                "state_file": str(custom_logs / "pipeline_state.json"),
                "timestamp": "2026-04-14T12:00:01",
            }
        )
    )

    state = load_state()

    assert state is not None
    assert state["step"] == "phase_3"
    assert state["status"] == "running"


def test_load_state_prefers_authoritative_state_file_over_stale_resume_mirror(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    custom_logs = tmp_path / "custom-logs"
    custom_logs.mkdir(parents=True, exist_ok=True)
    (custom_logs / "pipeline_state.json").write_text(
        json.dumps(
            {
                "step": "phase_4",
                "status": "running",
                "timestamp": "2026-04-14T12:05:00",
                "logs_dir": str(custom_logs),
            }
        )
    )
    mirrored = tmp_path / "logs" / "last_run_state.json"
    mirrored.parent.mkdir(parents=True, exist_ok=True)
    mirrored.write_text(
        json.dumps(
            {
                "step": "phase_3",
                "status": "running",
                "timestamp": "2026-04-14T12:00:00",
                "logs_dir": str(custom_logs),
            }
        )
    )

    state = load_state()

    assert state is not None
    assert state["step"] == "phase_4"


def test_load_state_prefers_newer_default_state_over_stale_resume_pointer(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    default_logs = tmp_path / "logs"
    default_logs.mkdir(parents=True, exist_ok=True)
    (default_logs / "pipeline_state.json").write_text(
        json.dumps(
            {
                "step": "phase_4",
                "status": "running",
                "timestamp": "2026-04-14T12:05:00",
                "logs_dir": str(default_logs),
            }
        )
    )
    custom_logs = tmp_path / "custom-logs"
    custom_logs.mkdir(parents=True, exist_ok=True)
    (custom_logs / "pipeline_state.json").write_text(
        json.dumps(
            {
                "step": "phase_3",
                "status": "running",
                "timestamp": "2026-04-14T12:00:00",
                "logs_dir": str(custom_logs),
            }
        )
    )
    (default_logs / "last_run_state.json").write_text(
        json.dumps(
            {
                "step": "phase_3",
                "status": "running",
                "timestamp": "2026-04-14T12:00:00",
                "logs_dir": str(custom_logs),
                "state_file": str(custom_logs / "pipeline_state.json"),
            }
        )
    )

    state = load_state()

    assert state is not None
    assert state["step"] == "phase_4"
    assert state["logs_dir"] == str(default_logs)


def test_load_state_uses_pointer_snapshot_when_custom_state_file_missing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    custom_logs = tmp_path / "custom-logs"
    custom_logs.mkdir(parents=True, exist_ok=True)
    mirrored = tmp_path / "logs" / "last_run_state.json"
    mirrored.parent.mkdir(parents=True, exist_ok=True)
    mirrored.write_text(
        json.dumps(
            {
                "step": "phase_2",
                "status": "running",
                "timestamp": "2026-04-14T12:10:00",
                "logs_dir": str(custom_logs),
                "state_file": str(custom_logs / "pipeline_state.json"),
                "sandbox_model": "claude-sonnet-4-6",
            }
        )
    )

    state = load_state()

    assert state is not None
    assert state["step"] == "phase_2"
    assert state["logs_dir"] == str(custom_logs)


def test_load_state_ignores_pointer_snapshot_when_referenced_logs_dir_missing(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    custom_logs = tmp_path / "deleted-logs"
    mirrored = tmp_path / "logs" / "last_run_state.json"
    mirrored.parent.mkdir(parents=True, exist_ok=True)
    mirrored.write_text(
        json.dumps(
            {
                "step": "phase_4",
                "status": "running",
                "timestamp": "2026-04-14T12:00:00",
                "logs_dir": str(custom_logs),
                "state_file": str(custom_logs / "pipeline_state.json"),
            }
        )
    )

    assert load_state() is None


def test_load_state_normalizes_matching_logs_dir_aliases(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WORLDSIM_STATE_DIR", "custom-logs")
    save_state("phase_1", status="running", benchmark_path="/tmp/benchmark")
    (tmp_path / "custom-logs" / "pipeline_state.json").unlink()
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path / "alias" / ".." / "custom-logs"))

    state = load_state()

    assert state is not None
    assert state["step"] == "phase_1"
    assert state["logs_dir"] == str(tmp_path / "custom-logs")


def test_load_state_rejects_pointer_target_outside_explicit_logs_dir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    expected_logs = tmp_path / "expected-logs"
    foreign_logs = tmp_path / "foreign-logs"
    expected_logs.mkdir(parents=True, exist_ok=True)
    foreign_logs.mkdir(parents=True, exist_ok=True)
    (expected_logs / "pipeline_state.json").write_text(
        json.dumps(
            {
                "step": "phase_2",
                "status": "running",
                "timestamp": "2026-04-14T12:00:00",
                "logs_dir": str(expected_logs),
            }
        )
    )
    (foreign_logs / "pipeline_state.json").write_text(
        json.dumps(
            {
                "step": "phase_4",
                "status": "running",
                "timestamp": "2026-04-14T12:10:00",
                "logs_dir": str(expected_logs),
                "instances_path": "/tmp/foreign.json",
            }
        )
    )
    mirrored = tmp_path / "logs" / "last_run_state.json"
    mirrored.parent.mkdir(parents=True, exist_ok=True)
    mirrored.write_text(
        json.dumps(
            {
                "step": "phase_4",
                "status": "running",
                "timestamp": "2026-04-14T12:10:00",
                "logs_dir": str(expected_logs),
                "state_file": str(foreign_logs / "pipeline_state.json"),
            }
        )
    )
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(expected_logs))

    state = load_state()

    assert state is not None
    assert state["step"] == "phase_2"
    assert state["logs_dir"] == str(expected_logs)


def test_load_state_rejects_valid_non_object_json(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
    (tmp_path / "pipeline_state.json").write_text('["not", "a", "state"]')

    assert load_state() is None


def test_load_state_ignores_unreadable_state_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "logs" / "pipeline_state.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "step": "phase_1",
                "status": "running",
                "timestamp": "2026-04-14T12:00:00",
                "logs_dir": str(tmp_path / "logs"),
            }
        )
    )
    real_read_text = Path.read_text

    def fake_read_text(self, *args, **kwargs):
        if self == target:
            raise OSError("boom")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read_text)

    assert load_state() is None


def test_load_state_rejects_authoritative_state_missing_status(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "logs" / "pipeline_state.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "step": "phase_1",
                "timestamp": "2026-04-14T12:00:00",
                "logs_dir": str(tmp_path / "logs"),
            }
        )
    )

    assert load_state() is None


def test_load_state_rejects_pointer_snapshot_when_target_state_is_corrupt(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    custom_logs = tmp_path / "custom-logs"
    custom_logs.mkdir(parents=True, exist_ok=True)
    (custom_logs / "pipeline_state.json").write_text("{bad-json")
    mirrored = tmp_path / "logs" / "last_run_state.json"
    mirrored.parent.mkdir(parents=True, exist_ok=True)
    mirrored.write_text(
        json.dumps(
            {
                "step": "phase_4",
                "status": "running",
                "timestamp": "2026-04-14T12:00:00",
                "logs_dir": str(custom_logs),
                "state_file": str(custom_logs / "pipeline_state.json"),
            }
        )
    )

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


def test_identity_aware_resume_refuses_definition_drift_without_writing(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.chdir(tmp_path)
    state_dir = tmp_path / "identified"
    monkeypatch.setenv("WARP_TASKGEN_STATE_DIR", str(state_dir))
    transition = resolve_run_request(
        {"agent_model": "source-model", "instances_path": "/tmp/instances.json"},
        existing_state=None,
        new_run_id="run-source",
    )
    assert transition.definition is not None
    with bind_run_definition(transition.definition):
        save_state(
            "phase_4",
            status="running",
            agent_model="source-model",
            instances_path="/tmp/instances.json",
        )
    state_path = state_dir / "pipeline_state.json"
    before = state_path.read_bytes()
    called = False

    def fake_dispatch_phase(_args):
        nonlocal called
        called = True
        return 0

    monkeypatch.setattr(worldsim_main, "_dispatch_phase", fake_dispatch_phase)

    rc = worldsim_main._dispatch_resume(Namespace(agent_model="changed-model"))

    assert rc == 2
    assert called is False
    assert state_path.read_bytes() == before
    assert "derive-and-resume" in capsys.readouterr().err


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
        agent_llm_timeout=240,
        agent_step_timeout=300,
    )
    monkeypatch.delenv("WORLDSIM_STATE_DIR")

    captured = {}

    def fake_dispatch_phase(args):
        captured["phase"] = args.phase
        captured["agent_model"] = args.agent_model
        captured["agent_provider"] = args.agent_provider
        captured["agent_llm_timeout"] = args.agent_llm_timeout
        captured["agent_step_timeout"] = args.agent_step_timeout
        return 0

    monkeypatch.setattr(worldsim_main, "_dispatch_phase", fake_dispatch_phase)

    rc = worldsim_main._dispatch_resume(Namespace())

    assert rc == 0
    assert captured["phase"] == "4"
    assert captured["agent_model"] == "claude-sonnet-4-6"
    assert captured["agent_provider"] == "anthropic"
    assert captured["agent_llm_timeout"] == 240
    assert captured["agent_step_timeout"] == 300


def test_dispatch_resume_treats_legacy_phase_4_state_as_strategy_variation(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    custom_logs = tmp_path / "custom-logs"
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(custom_logs))
    save_state(
        "phase_4",
        status="running",
        instances_path="/tmp/instances.json",
        agent_model="claude-sonnet-4-6",
    )
    monkeypatch.delenv("WORLDSIM_STATE_DIR")

    captured = {}

    def fake_dispatch_phase(args):
        captured["phase"] = args.phase
        captured["phase_4_variant_system"] = args.phase_4_variant_system
        return 0

    monkeypatch.setattr(worldsim_main, "_dispatch_phase", fake_dispatch_phase)

    rc = worldsim_main._dispatch_resume(Namespace())

    assert rc == 0
    assert captured["phase"] == "4"
    assert captured["phase_4_variant_system"] == "strategy-variation"


def test_dispatch_resume_overrides_saved_phase_4_timeouts(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    custom_logs = tmp_path / "custom-logs"
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(custom_logs))
    save_state(
        "phase_4",
        status="running",
        instances_path="/tmp/instances.json",
        agent_llm_timeout=120,
        agent_step_timeout=180,
    )
    monkeypatch.delenv("WORLDSIM_STATE_DIR")

    captured = {}

    def fake_dispatch_phase(args):
        captured["phase"] = args.phase
        captured["agent_llm_timeout"] = args.agent_llm_timeout
        captured["agent_step_timeout"] = args.agent_step_timeout
        return 0

    monkeypatch.setattr(worldsim_main, "_dispatch_phase", fake_dispatch_phase)

    rc = worldsim_main._dispatch_resume(Namespace(agent_llm_timeout=240, agent_step_timeout=300))

    assert rc == 0
    assert captured["phase"] == "4"
    assert captured["agent_llm_timeout"] == 240
    assert captured["agent_step_timeout"] == 300


def test_dispatch_resume_accepts_legacy_statusless_mirror(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    custom_logs = tmp_path / "custom-logs"
    custom_logs.mkdir(parents=True, exist_ok=True)
    mirrored = tmp_path / "logs" / "last_run_state.json"
    mirrored.parent.mkdir(parents=True, exist_ok=True)
    mirrored.write_text(
        json.dumps(
            {
                "step": "phase_4",
                "timestamp": "2026-04-14T12:00:00",
                "logs_dir": str(custom_logs),
                "instances_path": "/tmp/instances.json",
            }
        )
    )

    captured = {}

    def fake_dispatch_phase(args):
        captured["phase"] = args.phase
        captured["instances"] = args.instances
        return 0

    monkeypatch.setattr(worldsim_main, "_dispatch_phase", fake_dispatch_phase)

    rc = worldsim_main._dispatch_resume(Namespace())

    assert rc == 0
    assert captured["phase"] == "4"
    assert str(captured["instances"]) == str(Path("/tmp/instances.json").resolve())


def test_dispatch_resume_retries_failed_checkpoint(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    custom_logs = tmp_path / "custom-logs"
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(custom_logs))
    save_state(
        "phase_4",
        status="failed",
        reason="postprocess_exception",
        instances_path="/tmp/instances.json",
        agent_model="gpt-5.4",
    )
    stale_pause_marker = custom_logs / "pause_request.json"
    stale_pause_marker.write_text("{}", encoding="utf-8")
    monkeypatch.delenv("WORLDSIM_STATE_DIR")

    captured = {}

    def fake_dispatch_phase(args):
        captured["phase"] = args.phase
        captured["agent_model"] = args.agent_model
        return 0

    monkeypatch.setattr(worldsim_main, "_dispatch_phase", fake_dispatch_phase)

    rc = worldsim_main._dispatch_resume(Namespace())

    assert rc == 0
    assert captured["phase"] == "4"
    assert captured["agent_model"] == "gpt-5.4"
    assert not stale_pause_marker.exists()


@pytest.mark.parametrize("status", ["paused", "interrupted"])
def test_dispatch_resume_retries_cooperative_lifecycle_status(
    monkeypatch,
    tmp_path,
    status,
):
    monkeypatch.chdir(tmp_path)
    custom_logs = tmp_path / "custom-logs"
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(custom_logs))
    save_state(
        "phase_4",
        status=status,
        instances_path="/tmp/instances.json",
        agent_model="gpt-5.4",
    )
    monkeypatch.delenv("WORLDSIM_STATE_DIR")
    captured = {}

    def fake_dispatch_phase(args):
        captured["phase"] = args.phase
        captured["agent_model"] = args.agent_model
        return 0

    monkeypatch.setattr(worldsim_main, "_dispatch_phase", fake_dispatch_phase)

    rc = worldsim_main._dispatch_resume(Namespace())

    assert rc == 0
    assert captured == {"phase": "4", "agent_model": "gpt-5.4"}


def test_dispatch_resume_omits_saved_task_cap_by_default(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    custom_logs = tmp_path / "custom-logs"
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(custom_logs))
    save_state(
        "phase_4",
        status="failed",
        reason="postprocess_exception",
        instances_path="/tmp/instances.json",
        max_tasks_per_site=2,
    )
    monkeypatch.delenv("WORLDSIM_STATE_DIR")

    captured = {}

    def fake_dispatch_phase(args):
        captured["phase"] = args.phase
        captured["max_tasks_per_site"] = args.max_tasks_per_site
        return 0

    monkeypatch.setattr(worldsim_main, "_dispatch_phase", fake_dispatch_phase)

    rc = worldsim_main._dispatch_resume(Namespace())

    assert rc == 0
    assert captured["phase"] == "4"
    assert captured["max_tasks_per_site"] is None


def test_dispatch_resume_refuses_process_pool_root_and_prints_wrapper_command(
    monkeypatch,
    tmp_path,
    capsys,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WARP_TASKGEN_STATE_DIR", str(tmp_path))
    command = [
        "uv",
        "run",
        "python",
        "scripts/run_phase4_process_pool.py",
        "--resume",
        "--out-dir",
        str(tmp_path),
    ]
    (tmp_path / "pipeline_state.json").write_text(
        json.dumps(
            {
                "step": "phase_4",
                "status": "paused",
                "timestamp": "2026-08-11T00:00:00+00:00",
                "logs_dir": str(tmp_path),
                "process_pool": True,
                "process_pool_resume_argv": command,
            }
        ),
        encoding="utf-8",
    )
    called = False

    def fake_dispatch_phase(_args):
        nonlocal called
        called = True
        return 0

    monkeypatch.setattr(worldsim_main, "_dispatch_phase", fake_dispatch_phase)

    rc = worldsim_main._dispatch_resume(Namespace())

    assert rc == 2
    assert called is False
    assert "scripts/run_phase4_process_pool.py --resume" in capsys.readouterr().err


def test_dispatch_resume_allows_explicit_task_cap_override(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    custom_logs = tmp_path / "custom-logs"
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(custom_logs))
    save_state(
        "phase_4",
        status="failed",
        reason="postprocess_exception",
        instances_path="/tmp/instances.json",
        max_tasks_per_site=2,
    )
    monkeypatch.delenv("WORLDSIM_STATE_DIR")

    captured = {}

    def fake_dispatch_phase(args):
        captured["phase"] = args.phase
        captured["max_tasks_per_site"] = args.max_tasks_per_site
        return 0

    monkeypatch.setattr(worldsim_main, "_dispatch_phase", fake_dispatch_phase)

    rc = worldsim_main._dispatch_resume(Namespace(max_tasks_per_site=5))

    assert rc == 0
    assert captured["phase"] == "4"
    assert captured["max_tasks_per_site"] == 5


def test_dispatch_resume_restores_saved_phase_2_sites_filter(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    custom_logs = tmp_path / "custom-logs"
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(custom_logs))
    save_state(
        "phase_2",
        status="running",
        sandbox_model="claude-sonnet-4-6",
        sites="shopping,gitlab",
    )
    monkeypatch.delenv("WORLDSIM_STATE_DIR")

    captured = {}

    def fake_dispatch_phase(args):
        captured["phase"] = args.phase
        captured["sites"] = args.sites
        return 0

    monkeypatch.setattr(worldsim_main, "_dispatch_phase", fake_dispatch_phase)

    rc = worldsim_main._dispatch_resume(Namespace())

    assert rc == 0
    assert captured["phase"] == "2"
    assert captured["sites"] == "shopping,gitlab"


def test_dispatch_resume_treats_partial_complete_as_advanced_phase(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    custom_logs = tmp_path / "custom-logs"
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(custom_logs))
    save_state(
        "phase_2",
        status="partial_complete",
        sandbox_model="claude-sonnet-4-6",
        partial=True,
        generation_failures=["gitlab: sandbox did not produce adversarial_tasks.json"],
    )
    monkeypatch.delenv("WORLDSIM_STATE_DIR")

    captured = {}

    def fake_dispatch_phase(args):
        captured["phase"] = args.phase
        captured["sandbox_model"] = args.sandbox_model
        return 0

    monkeypatch.setattr(worldsim_main, "_dispatch_phase", fake_dispatch_phase)

    rc = worldsim_main._dispatch_resume(Namespace())

    assert rc == 0
    assert captured["phase"] == "3"
    assert captured["sandbox_model"] == "claude-sonnet-4-6"


def test_dispatch_resume_allows_explicit_phase_2_sites_override(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    custom_logs = tmp_path / "custom-logs"
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(custom_logs))
    save_state(
        "phase_2",
        status="running",
        sandbox_model="claude-sonnet-4-6",
        sites="shopping",
    )
    monkeypatch.delenv("WORLDSIM_STATE_DIR")

    captured = {}

    def fake_dispatch_phase(args):
        captured["phase"] = args.phase
        captured["sites"] = args.sites
        return 0

    monkeypatch.setattr(worldsim_main, "_dispatch_phase", fake_dispatch_phase)

    rc = worldsim_main._dispatch_resume(Namespace(sites="gitlab"))

    assert rc == 0
    assert captured["phase"] == "2"
    assert captured["sites"] == "gitlab"


def test_dispatch_resume_restores_saved_phase_3_sites_filter(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    custom_logs = tmp_path / "custom-logs"
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(custom_logs))
    save_state(
        "phase_3",
        status="running",
        instances_path="/tmp/instances.json",
        sites="gitlab",
    )
    monkeypatch.delenv("WORLDSIM_STATE_DIR")

    captured = {}

    def fake_dispatch_phase(args):
        captured["phase"] = args.phase
        captured["sites"] = args.sites
        return 0

    monkeypatch.setattr(worldsim_main, "_dispatch_phase", fake_dispatch_phase)

    rc = worldsim_main._dispatch_resume(Namespace())

    assert rc == 0
    assert captured["phase"] == "3"
    assert captured["sites"] == "gitlab"


def test_dispatch_resume_restores_saved_phase_4_sites_filter(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    custom_logs = tmp_path / "custom-logs"
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(custom_logs))
    save_state(
        "phase_4",
        status="running",
        instances_path="/tmp/instances.json",
        sites="shopping,gitlab",
    )
    monkeypatch.delenv("WORLDSIM_STATE_DIR")

    captured = {}

    def fake_dispatch_phase(args):
        captured["phase"] = args.phase
        captured["sites"] = args.sites
        return 0

    monkeypatch.setattr(worldsim_main, "_dispatch_phase", fake_dispatch_phase)

    rc = worldsim_main._dispatch_resume(Namespace())

    assert rc == 0
    assert captured["phase"] == "4"
    assert captured["sites"] == "shopping,gitlab"


def test_dispatch_resume_ignores_removed_phase_2a_modal_state(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    custom_logs = tmp_path / "custom-logs"
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(custom_logs))
    save_state(
        "phase_2",
        status="running",
        sandbox_model="claude-sonnet-4-6",
        phase_2a_runtime="modal",
        phase_2_sandbox_concurrency=3,
        phase_2_launch_jitter_ms=400,
    )
    monkeypatch.delenv("WORLDSIM_STATE_DIR")

    captured = {}

    def fake_dispatch_phase(args):
        captured["phase"] = args.phase
        captured["has_phase_2a_runtime"] = hasattr(args, "phase_2a_runtime")
        captured["has_phase_2_sandbox_concurrency"] = hasattr(args, "phase_2_sandbox_concurrency")
        captured["has_phase_2_launch_jitter_ms"] = hasattr(args, "phase_2_launch_jitter_ms")
        return 0

    monkeypatch.setattr(worldsim_main, "_dispatch_phase", fake_dispatch_phase)

    rc = worldsim_main._dispatch_resume(Namespace())

    assert rc == 0
    assert captured["phase"] == "2"
    assert captured["has_phase_2a_runtime"] is False
    assert captured["has_phase_2_sandbox_concurrency"] is False
    assert captured["has_phase_2_launch_jitter_ms"] is False


def test_dispatch_resume_restores_saved_phase_2_text_fill_flags(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    custom_logs = tmp_path / "custom-logs"
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(custom_logs))
    save_state(
        "phase_2",
        status="running",
        sandbox_model="claude-sonnet-4-6",
        phase_2b_texts_per_plan=3,
        phase_2_text_fill_concurrency=9,
        phase_2_text_model="anthropic/claude-sonnet-4-6",
    )
    monkeypatch.delenv("WORLDSIM_STATE_DIR")

    captured = {}

    def fake_dispatch_phase(args):
        captured["phase"] = args.phase
        captured["phase_2b_texts_per_plan"] = args.phase_2b_texts_per_plan
        captured["phase_2_text_fill_concurrency"] = args.phase_2_text_fill_concurrency
        captured["phase_2_text_model"] = args.phase_2_text_model
        return 0

    monkeypatch.setattr(worldsim_main, "_dispatch_phase", fake_dispatch_phase)

    rc = worldsim_main._dispatch_resume(Namespace())

    assert rc == 0
    assert captured["phase"] == "2"
    assert captured["phase_2b_texts_per_plan"] == 3
    assert captured["phase_2_text_fill_concurrency"] == 9
    assert captured["phase_2_text_model"] == "anthropic/claude-sonnet-4-6"


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


def test_main_resume_installs_proxy_after_restoring_saved_instances(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    custom_logs = tmp_path / "custom-logs"
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(custom_logs))
    save_state(
        "phase_4",
        status="running",
        instances_path="/tmp/instances.json",
    )
    monkeypatch.delenv("WORLDSIM_STATE_DIR")

    captured = {}

    def fake_install(args):
        captured["instances"] = args.instances
        captured["phase"] = args.phase

    monkeypatch.setattr(worldsim_main, "_install_verification_proxy_from_args", fake_install)
    monkeypatch.setattr(worldsim_main, "_dispatch_phase", lambda args: 0)

    rc = worldsim_main.main(["resume"])

    assert rc == 0
    assert captured["phase"] == "4"
    assert str(captured["instances"]) == str(Path("/tmp/instances.json").resolve())


def test_main_phase_rejects_invalid_verification_proxy(monkeypatch, tmp_path, capsys):
    instances_path = tmp_path / "instances.json"
    instances_path.write_text(
        json.dumps(
            {
                "instances": [
                    {"site_url": "http://shopping.test:7770"},
                ],
                "verification_proxy": {"token": "tok", "port_offset": "bad"},
            }
        )
    )

    rc = worldsim_main.main(["phase", "3", "--instances", str(instances_path)])

    assert rc == 2
    assert "verification_proxy_invalid" in capsys.readouterr().err


def test_install_verification_proxy_ignores_loopback_instances(monkeypatch, tmp_path):
    instances_path = tmp_path / "instances.json"
    instances_path.write_text(
        json.dumps(
            {
                "instances": [
                    {"site_url": "http://127.0.0.1:9999"},
                    {"site_url": "http://localhost:8023"},
                ],
                "verification_proxy": {"token": "tok", "port_offset": 10000},
            }
        )
    )
    called = False

    def fake_install_proxy(**kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr("warp_taskgen.http_proxy.install_proxy", fake_install_proxy)

    worldsim_main._install_verification_proxy_from_args(
        Namespace(instances=instances_path, feasibility_instances=None)
    )

    assert called is False


def test_install_verification_proxy_keeps_remote_instances(monkeypatch, tmp_path):
    instances_path = tmp_path / "instances.json"
    instances_path.write_text(
        json.dumps(
            {
                "instances": [
                    {"site_url": "http://203.0.113.10:9999"},
                ],
                "verification_proxy": {"token": "tok", "port_offset": 10000},
            }
        )
    )
    captured = {}

    def fake_install_proxy(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("warp_taskgen.http_proxy.install_proxy", fake_install_proxy)

    worldsim_main._install_verification_proxy_from_args(
        Namespace(instances=instances_path, feasibility_instances=None)
    )

    assert captured["token"] == "tok"
    assert captured["site_ports"] == {9999}


def test_install_verification_proxy_reads_token_env(monkeypatch, tmp_path):
    instances_path = tmp_path / "instances.json"
    instances_path.write_text(
        json.dumps(
            {
                "instances": [
                    {"site_url": "http://203.0.113.10:9999"},
                ],
                "verification_proxy": {
                    "token_env": "WORLDSIM_TEST_PROXY_TOKEN",
                    "port_offset": 10000,
                },
            }
        )
    )
    monkeypatch.setenv("WORLDSIM_TEST_PROXY_TOKEN", "env-token")
    captured = {}

    def fake_install_proxy(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("warp_taskgen.http_proxy.install_proxy", fake_install_proxy)

    worldsim_main._install_verification_proxy_from_args(
        Namespace(instances=instances_path, feasibility_instances=None)
    )

    assert captured["token"] == "env-token"


def test_install_verification_proxy_reads_token_file(monkeypatch, tmp_path):
    token_file = tmp_path / ".proxy_token"
    token_file.write_text("file-token\n")
    instances_path = tmp_path / "instances.json"
    instances_path.write_text(
        json.dumps(
            {
                "instances": [
                    {"site_url": "http://203.0.113.10:9999"},
                ],
                "verification_proxy": {
                    "token_file": ".proxy_token",
                    "port_offset": 10000,
                },
            }
        )
    )
    captured = {}

    def fake_install_proxy(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("warp_taskgen.http_proxy.install_proxy", fake_install_proxy)

    worldsim_main._install_verification_proxy_from_args(
        Namespace(instances=instances_path, feasibility_instances=None)
    )

    assert captured["token"] == "file-token"


def test_install_verification_proxy_missing_external_token_disables_proxy(monkeypatch, tmp_path):
    instances_path = tmp_path / "instances.json"
    instances_path.write_text(
        json.dumps(
            {
                "instances": [
                    {"site_url": "http://203.0.113.10:9999"},
                ],
                "verification_proxy": {
                    "token_file": ".missing_proxy_token",
                    "port_offset": 10000,
                },
            }
        )
    )
    called = False

    def fake_install_proxy(**kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr("warp_taskgen.http_proxy.install_proxy", fake_install_proxy)

    worldsim_main._install_verification_proxy_from_args(
        Namespace(instances=instances_path, feasibility_instances=None)
    )

    assert called is False


def test_phase4_run_lock_rejects_concurrent_run(tmp_path):
    with worldsim_main._phase4_run_lock(tmp_path):
        try:
            with worldsim_main._phase4_run_lock(tmp_path):
                raise AssertionError("nested lock should not be acquired")
        except worldsim_main.Phase4AlreadyRunning as exc:
            assert ".phase4_run.lock" in str(exc)


# ── Per-task resume (load_completed_results) ────────────────────────────


def test_load_completed_results_finds_valid_results(tmp_path):
    """Completed tasks with valid result.json are returned."""
    from warp_taskgen.task_paths import safe_task_path_component

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
    from warp_taskgen.task_paths import safe_task_path_component

    task_dir = tmp_path / safe_task_path_component("7")
    task_dir.mkdir()
    (task_dir / "result.json").write_text("{truncated")

    completed = load_completed_results(tmp_path)
    assert completed == {}


def test_load_completed_results_ignores_incomplete_tasks(tmp_path):
    """Tasks with history.json but no result.json are not counted as complete."""
    from warp_taskgen.task_paths import safe_task_path_component

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
    from warp_taskgen.task_paths import safe_task_path_component

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


def test_load_completed_results_filters_out_stale_fingerprints(tmp_path):
    task_dir = tmp_path / "task-10"
    task_dir.mkdir()
    (task_dir / "result.json").write_text(
        json.dumps(
            {
                "task_id": "task-10",
                "passed": True,
                "message": "initial",
                RESULT_FINGERPRINT_KEY: "old-fp",
            }
        )
    )

    completed = load_completed_results(
        tmp_path,
        expected_fingerprints={"task-10": "new-fp"},
    )

    assert completed == {}


def test_load_completed_results_round_trips_phase4_final_status_and_metrics(tmp_path):
    task_dir = tmp_path / "task-encounter"
    task_dir.mkdir()
    (task_dir / "result.json").write_text(
        json.dumps(
            {
                "task_id": "task-encounter",
                "outcome": "refused_or_ignored",
                "encounter": {"max_coverage": 0.0},
                "final_status": "injection_not_encountered",
                "transcript_purpose_classification": None,
                "verbalized_eval_awareness_classification": None,
            }
        )
    )
    (task_dir / "history.json").write_text(json.dumps({"history": []}))
    (task_dir / "final_response.json").write_text(json.dumps({"status": "FAILURE"}))

    completed = load_completed_results(tmp_path)

    assert completed["task-encounter"]["final_status"] == "injection_not_encountered"
    assert "transcript_purpose_classification" in completed["task-encounter"]


def test_load_completed_results_skips_reprocessable_phase4_result_missing_required_sidecars(
    tmp_path,
):
    task_dir = tmp_path / "task-missing-sidecars"
    task_dir.mkdir()
    (task_dir / "result.json").write_text(
        json.dumps(
            {
                "task_id": "task-missing-sidecars",
                "outcome": "refused_or_ignored",
                "encounter": {"max_coverage": 0.8},
            }
        )
    )

    completed = load_completed_results(tmp_path)

    assert completed == {}


def test_load_completed_results_keeps_seed_preflight_mismatch_without_sidecars(tmp_path):
    task_dir = tmp_path / "task-seed-preflight"
    task_dir.mkdir()
    (task_dir / "result.json").write_text(
        json.dumps(
            {
                "task_id": "task-seed-preflight",
                "outcome": "seed_preflight_mismatch",
                "error": "unsupported site",
            }
        )
    )

    completed = load_completed_results(tmp_path)

    assert completed["task-seed-preflight"]["outcome"] == "seed_preflight_mismatch"


def test_load_completed_results_skips_valid_json_with_non_object_top_level(tmp_path):
    task_dir = tmp_path / "task-wrong-shape"
    task_dir.mkdir()
    (task_dir / "result.json").write_text(json.dumps(["not", "an", "object"]))

    completed = load_completed_results(tmp_path)

    assert completed == {}


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
        **{RESULT_FINGERPRINT_KEY: "task-fp"},
    )

    data = json.loads((task_dir / "result.json").read_text())
    assert data["outcome"] == "complied"
    assert data["ecologically_valid"] is True
    assert data[RESULT_FINGERPRINT_KEY] == "task-fp"
