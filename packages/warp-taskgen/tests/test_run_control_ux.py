from __future__ import annotations

import json
from argparse import Namespace
from contextlib import nullcontext
from pathlib import Path

import pytest

from tests.phase_2._fixtures import _single_surface_profile
from warp_taskgen.cli import _impl as cli_impl
from warp_taskgen.cli.derived_run import dispatch_derived_resume
from warp_taskgen.cli.run_control import dispatch_phase_with_run_control
from warp_taskgen.cli_status import build_status_payload, format_status_payload
from warp_taskgen.phase_2.pause_control import write_planning_shard_checkpoint
from warp_taskgen.run_control import RunInterrupted, acknowledge_pause, request_pause
from warp_taskgen.run_control_history import load_transition_history
from warp_taskgen.run_control_status import build_run_control_projection
from warp_taskgen.run_control_wait import wait_for_pause
from warp_taskgen.run_definition import define_run, plan_resume
from warp_taskgen.run_transition import resolve_run_request


def _running(root: Path, *, stage: str = "planning") -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "pipeline_state.json").write_text(
        json.dumps(
            {
                "step": "phase_2",
                "status": "running",
                "timestamp": "2026-08-11T00:00:00+00:00",
                "logs_dir": str(root),
                "phase_2_stage": stage,
                "phase_2_planning_queued_count": 3,
                "phase_2_planning_admitted_count": 2,
                "phase_2_planning_completed_count": 1,
            }
        ),
        encoding="utf-8",
    )


def test_pause_wait_is_read_only_and_times_out_without_ack(tmp_path: Path) -> None:
    _running(tmp_path)
    request = request_pause(tmp_path)
    before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    result = wait_for_pause(tmp_path, request.request_id, timeout=0, poll_interval=0)

    assert result.status == "timed_out"
    assert result.reason_code == "pause_wait_timeout"
    assert {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    } == before


def test_pause_wait_requires_exact_ack_identity(tmp_path: Path) -> None:
    _running(tmp_path)
    request = request_pause(tmp_path)
    state = json.loads((tmp_path / "pipeline_state.json").read_text(encoding="utf-8"))
    state.update(status="paused", pause_request_id="pause-other")
    (tmp_path / "pipeline_state.json").write_text(json.dumps(state), encoding="utf-8")

    result = wait_for_pause(tmp_path, request.request_id, timeout=0)

    assert result.status == "rejected"
    assert result.reason_code == "pause_acknowledgement_identity_mismatch"


def test_pause_wait_rejects_same_request_id_from_a_different_run(tmp_path: Path) -> None:
    _running(tmp_path)
    request = request_pause(tmp_path)
    state = json.loads((tmp_path / "pipeline_state.json").read_text(encoding="utf-8"))
    state.update(
        status="paused",
        pause_request_id=request.request_id,
        pause_request_run_id=request.run_id,
        pause_request_definition_digest="different-run-definition",
    )
    (tmp_path / "pipeline_state.json").write_text(json.dumps(state), encoding="utf-8")

    result = wait_for_pause(tmp_path, request.request_id, timeout=0, expected_request=request)

    assert result.status == "rejected"
    assert result.reason_code == "pause_acknowledgement_definition_mismatch"


def test_pause_wait_reads_authoritative_ack_and_history_is_bounded(tmp_path: Path) -> None:
    _running(tmp_path)
    request = request_pause(tmp_path)
    acknowledge_pause(tmp_path)

    result = wait_for_pause(tmp_path, request.request_id, timeout=0)
    history = load_transition_history(tmp_path)

    assert result.status == "paused"
    assert [event["event"] for event in history] == ["pause_requested", "paused"]
    assert all("run_id" not in event for event in history)


def test_paused_status_projects_authoritative_request_metadata(tmp_path: Path) -> None:
    _running(tmp_path)
    request = request_pause(tmp_path)
    acknowledge_pause(tmp_path)
    state = json.loads((tmp_path / "pipeline_state.json").read_text(encoding="utf-8"))

    projection = build_run_control_projection(tmp_path, state)

    assert projection["lifecycle_status"] == "paused"
    assert projection["pause_request"]["request_id"] == request.request_id
    assert projection["pause_request"]["reason_code"] == "operator_requested_pause"
    assert projection["pause_request"]["source"] == "authoritative_pipeline_state"
    assert projection["pause_age_seconds"] >= 0


def test_interrupted_status_does_not_inherit_pause_reason(tmp_path: Path) -> None:
    state = {
        "step": "phase_2",
        "status": "interrupted",
        "timestamp": "2026-08-11T00:00:00+00:00",
        "logs_dir": str(tmp_path),
        "phase_2_stage": "planning",
        "pause_request_id": "pause-" + "b" * 32,
        "pause_requested_at": "2026-08-11T00:00:00+00:00",
    }

    projection = build_run_control_projection(tmp_path, state)

    assert projection["pause_request"]["reason_code"] == "abrupt_process_interruption"


def test_status_projection_uses_feature_owned_state_counts_and_next_action(tmp_path: Path) -> None:
    _running(tmp_path)
    request = request_pause(tmp_path)

    projection = build_run_control_projection(
        tmp_path,
        json.loads((tmp_path / "pipeline_state.json").read_text(encoding="utf-8")),
    )

    assert projection["lifecycle_status"] == "pausing"
    assert projection["supported_stage"] == "planning"
    assert projection["pause_request_id"] == request.request_id
    assert projection["checkpoint_counts"] == {
        "queued": 3,
        "admitted": 2,
        "completed": 1,
        "authority": "advisory:phase_2.planning.state_projection",
    }
    assert "pause --wait" in projection["next_action"]["command"]


def _identified_planning_status_fixture(root: Path) -> tuple[dict, object]:
    transition = resolve_run_request(
        {"sandbox_model": "model-a"},
        existing_state=None,
        new_run_id="run-status-planning",
    )
    state = {
        "step": "phase_2",
        "status": "running",
        "timestamp": "2026-08-11T00:00:00+00:00",
        "logs_dir": str(root),
        "phase_2_stage": "planning",
        "run_definition": transition.definition.to_dict(),
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "pipeline_state.json").write_text(json.dumps(state), encoding="utf-8")
    (root / "phase_1" / "benign_tasks.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "phase_1" / "benign_tasks.json").write_text(
        json.dumps(
            [
                {
                    "id": "benign-1",
                    "site": "shopping",
                    "benchmark": "webarena_verified",
                }
            ]
        ),
        encoding="utf-8",
    )
    (root / "phase_0c" / "BENCHMARK_PROFILE_shopping.json").parent.mkdir(
        parents=True, exist_ok=True
    )
    (root / "phase_0c" / "BENCHMARK_PROFILE_shopping.json").write_text(
        json.dumps(_single_surface_profile()),
        encoding="utf-8",
    )
    return state, transition.definition


def test_status_inspects_run_bound_planning_shards_when_state_counts_are_absent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("WARP_TASKGEN_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("WARP_TASKGEN_RESUME_POINTER", str(tmp_path / "pointer.json"))
    state, definition = _identified_planning_status_fixture(tmp_path)
    shard_path = tmp_path / "phase_2" / "shards" / "shopping.json"
    payload = [{"id": "adv-1", "benign_task_id": "benign-1", "site": "shopping"}]
    write_planning_shard_checkpoint(
        shard_path,
        payload,
        label="shopping",
        input_task_ids=["benign-1"],
    )
    before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    projection = build_run_control_projection(tmp_path, state)
    inspection = projection["planning_checkpoint_inspection"]

    assert inspection["status"] == "inspected"
    assert inspection["expected_count"] == 1
    assert inspection["compatible_count"] == 1
    assert inspection["pending_count"] == 0
    assert inspection["stale_count"] == 0
    assert inspection["malformed_count"] == 0
    assert inspection["not_inspected_count"] == 0
    assert inspection["shards"][0]["status"] == "compatible"
    assert inspection["shards"][0]["path"] == str(shard_path)

    payload = build_status_payload(tmp_path)
    assert payload["run_control"]["planning_checkpoint_inspection"] == inspection
    text = format_status_payload(payload)
    assert "Phase 2a planning checkpoints: status=inspected" in text
    assert "shopping: compatible" in text
    assert define_run(state) == definition
    assert {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    } == before


def test_status_does_not_invent_planning_denominator_without_phase1_inputs(
    tmp_path: Path,
) -> None:
    state = {
        "step": "phase_2",
        "status": "running",
        "timestamp": "2026-08-11T00:00:00+00:00",
        "logs_dir": str(tmp_path),
        "phase_2_stage": "planning",
        "run_definition": resolve_run_request(
            {"sandbox_model": "model-a"},
            existing_state=None,
            new_run_id="run-status-missing-inputs",
        ).definition.to_dict(),
    }

    projection = build_run_control_projection(tmp_path, state)
    inspection = projection["planning_checkpoint_inspection"]

    assert inspection["status"] == "not_inspected"
    assert inspection["expected_count"] is None
    assert inspection["compatible_count"] is None
    assert inspection["reason_code"] == "planning_inputs_missing"
    assert str(tmp_path / "phase_1" / "benign_tasks.json") == inspection["path"]


def test_status_fails_closed_when_required_profile_is_missing(tmp_path: Path) -> None:
    state, _ = _identified_planning_status_fixture(tmp_path)
    profile_path = tmp_path / "phase_0c" / "BENCHMARK_PROFILE_shopping.json"
    profile_path.unlink()

    inspection = build_run_control_projection(tmp_path, state)["planning_checkpoint_inspection"]

    assert inspection["status"] == "not_inspected"
    assert inspection["expected_count"] is None
    assert inspection["reason_code"] == "planning_profile_unavailable"
    assert inspection["path"] == str(profile_path)


def test_status_fails_closed_when_route_contracts_have_wrong_root_shape(
    tmp_path: Path,
) -> None:
    state, _ = _identified_planning_status_fixture(tmp_path)
    (tmp_path / "phase_1" / "TASK_ROUTE_CONTRACTS_shopping.json").write_text(
        json.dumps([]),
        encoding="utf-8",
    )
    inspection = build_run_control_projection(tmp_path, state)["planning_checkpoint_inspection"]

    assert inspection["status"] == "not_inspected"
    assert inspection["expected_count"] is None
    assert inspection["reason_code"] == "planning_profile_unavailable"
    assert "TASK_ROUTE_CONTRACTS_shopping.json" in inspection["path"]


def test_status_projection_uses_resume_plan_for_non_final_complete(tmp_path: Path) -> None:
    state = {
        "step": "phase_2",
        "status": "complete",
        "timestamp": "2026-08-11T00:00:00+00:00",
        "logs_dir": str(tmp_path),
        "phase_2_stage": "feasibility",
    }
    (tmp_path / "pipeline_state.json").write_text(json.dumps(state), encoding="utf-8")

    projection = build_run_control_projection(tmp_path, state)

    assert projection["next_action"]["command"].endswith("warp-taskgen resume")
    assert "phase_3" in projection["next_action"]["description"]


def test_status_and_wait_reject_unhashable_step_without_crashing(tmp_path: Path) -> None:
    state = {
        "step": [],
        "status": "running",
        "timestamp": "2026-08-11T00:00:00+00:00",
        "logs_dir": str(tmp_path),
        "phase_2_stage": "planning",
    }
    (tmp_path / "pipeline_state.json").write_text(json.dumps(state), encoding="utf-8")
    (tmp_path / "pause_request.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "request_id": "pause-" + "a" * 32,
                "requested_at": "2026-08-11T00:00:00+00:00",
                "run_id": None,
                "definition_digest": "0" * 64,
                "step": "phase_2",
            }
        ),
        encoding="utf-8",
    )

    projection = build_run_control_projection(tmp_path, state)
    result = wait_for_pause(tmp_path, "pause-" + "a" * 32, timeout=0)

    assert projection["supported"] is False
    assert result.status == "rejected"


def test_process_pool_stage_is_reflected_in_supported_stage_contract(tmp_path: Path) -> None:
    state = {
        "step": "phase_4",
        "status": "running",
        "timestamp": "2026-08-11T00:00:00+00:00",
        "logs_dir": str(tmp_path),
        "pause_stage": "process_pool_dispatch",
        "process_pool": True,
    }

    projection = build_run_control_projection(tmp_path, state)

    assert projection["supported_stage"] == "process_pool_dispatch"
    assert projection["supported"] is True
    assert "process_pool_dispatch" in projection["supported_stages"]["phase_4"]


def test_paused_process_pool_status_exposes_isolated_wrapper_command(tmp_path: Path) -> None:
    state = {
        "step": "phase_4",
        "status": "paused",
        "timestamp": "2026-08-11T00:00:00+00:00",
        "logs_dir": str(tmp_path),
        "pause_stage": "process_pool_paused",
        "process_pool": True,
        "process_pool_resume_argv": [
            "python",
            "scripts/run_phase4_process_pool.py",
            "--resume",
            "--source-state-dir",
            str(tmp_path),
        ],
    }

    projection = build_run_control_projection(tmp_path, state)

    assert projection["next_action"]["command"] == (
        "python scripts/run_phase4_process_pool.py --resume --source-state-dir " + str(tmp_path)
    )


def test_wait_rejects_zero_poll_interval_for_positive_timeout(tmp_path: Path) -> None:
    _running(tmp_path)
    request = request_pause(tmp_path)

    with pytest.raises(ValueError, match="positive"):
        wait_for_pause(tmp_path, request.request_id, timeout=1, poll_interval=0)


def test_resume_plan_canonicalizes_authoritative_logs_dir(tmp_path: Path) -> None:
    state = {
        "step": "phase_2",
        "status": "running",
        "timestamp": "2026-08-11T00:00:00+00:00",
        "logs_dir": str(tmp_path),
        "phase_2_stage": "planning",
    }
    (tmp_path / "pipeline_state.json").write_text(json.dumps(state), encoding="utf-8")

    plan = plan_resume(define_run(state), state, run_root=tmp_path)

    assert plan.checkpoint_decisions[0].action == "rerun"
    assert plan.checkpoint_decisions[0].reason_code == "pipeline_checkpoint_running"


def test_phase2_signal_transition_happens_after_lock_guard_owns_run(tmp_path: Path) -> None:
    _running(tmp_path)
    entered = False

    def guard():
        nonlocal entered
        entered = True
        return nullcontext()

    rc = dispatch_phase_with_run_control(
        phase="2",
        state_dir=tmp_path,
        operation=lambda: (_ for _ in ()).throw(RunInterrupted("SIGTERM")),
        lifecycle_guard=guard,
    )

    assert entered is True
    assert rc == 143
    state = json.loads((tmp_path / "pipeline_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "interrupted"
    assert state["interrupt_signal"] == "SIGTERM"


def test_phase2_keyboard_interrupt_is_persisted_after_lock_guard(tmp_path: Path) -> None:
    _running(tmp_path)
    entered = False

    def guard():
        nonlocal entered
        entered = True
        return nullcontext()

    def interrupt():
        raise KeyboardInterrupt

    rc = dispatch_phase_with_run_control(
        phase="2c",
        state_dir=tmp_path,
        operation=interrupt,
        lifecycle_guard=guard,
    )

    assert entered is True
    assert rc == 130
    state = json.loads((tmp_path / "pipeline_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "interrupted"
    assert state["interrupt_signal"] == "SIGINT"


def test_resume_plan_json_is_read_only(tmp_path: Path, capsys) -> None:
    state = {
        "step": "phase_2",
        "status": "running",
        "timestamp": "2026-08-11T00:00:00+00:00",
        "logs_dir": str(tmp_path),
        "phase_2_stage": "planning",
    }
    state_path = tmp_path / "pipeline_state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    before = state_path.read_bytes()

    rc = cli_impl._dispatch_resume_plan(
        Namespace(command="resume", plan=True, json=True),
        state,
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "legacy"
    assert payload["lifecycle_action"] == "rerun_phase"
    assert state_path.read_bytes() == before
    assert set(tmp_path.iterdir()) == {state_path}


def test_derive_resume_plan_json_does_not_materialize_child(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    state = {
        "step": "phase_2",
        "status": "running",
        "timestamp": "2026-08-11T00:00:00+00:00",
        "logs_dir": str(tmp_path),
        "run_definition_schema_version": 1,
        "run_id": "run-source",
        "agent_model": "source-model",
        "phase_2_stage": "planning",
    }
    state_path = tmp_path / "pipeline_state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr("warp_taskgen.cli.derived_run._load_source_state", lambda: state)

    rc = dispatch_derived_resume(
        Namespace(
            command="derive-and-resume",
            plan=True,
            json=True,
            agent_model="child-model",
        )
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "derived_required"
    assert payload["drift_fields"] == ["phase_4.agent_model"]
    assert list(tmp_path.iterdir()) == [state_path]
