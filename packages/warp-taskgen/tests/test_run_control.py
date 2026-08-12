from __future__ import annotations

import json
from contextlib import contextmanager, nullcontext
from pathlib import Path

import pytest

from worldsim import main as worldsim_main
from worldsim import run_control
from worldsim.atomic_io import write_json_atomic
from worldsim.cli import run_control as cli_run_control
from worldsim.cli.run_control import dispatch_phase_with_run_control
from worldsim.cli_status import build_status_payload, format_status_payload
from worldsim.run_control import (
    PauseBoundaryReached,
    acknowledge_pause,
    load_pause_request,
    mark_interrupted,
    pause_aware_map,
    request_pause,
)
from worldsim.state import save_state


def _write_running_phase_4(root: Path, *, stage: str = "initial_evaluation") -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "pipeline_state.json").write_text(
        json.dumps(
            {
                "step": "phase_4",
                "status": "running",
                "timestamp": "2026-08-11T00:00:00+00:00",
                "logs_dir": str(root),
                "agent_model": "model-a",
                "pause_stage": stage,
            }
        ),
        encoding="utf-8",
    )
    progress = root / "phase_4" / "progress.json"
    progress.parent.mkdir(parents=True, exist_ok=True)
    progress.write_text(json.dumps({"stage": stage}), encoding="utf-8")


def _write_running_phase_2(root: Path, *, stage: str = "planning") -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "pipeline_state.json").write_text(
        json.dumps(
            {
                "step": "phase_2",
                "status": "running",
                "timestamp": "2026-08-11T00:00:00+00:00",
                "logs_dir": str(root),
                "sandbox_model": "model-a",
                "phase_2_stage": stage,
            }
        ),
        encoding="utf-8",
    )


def test_pause_request_is_atomic_nonsecret_and_idempotent(tmp_path: Path) -> None:
    _write_running_phase_4(tmp_path)

    first = request_pause(tmp_path)
    second = request_pause(tmp_path)

    assert second == first
    payload = json.loads((tmp_path / "pause_request.json").read_text(encoding="utf-8"))
    assert payload == first.to_dict()
    assert set(payload) == {
        "schema_version",
        "request_id",
        "requested_at",
        "run_id",
        "definition_digest",
        "step",
    }
    assert "model-a" not in json.dumps(payload)


def test_pause_request_uses_state_owned_stage_not_progress_telemetry(tmp_path: Path) -> None:
    _write_running_phase_4(tmp_path)
    (tmp_path / "phase_4" / "progress.json").write_text(
        json.dumps({"stage": "stale_finalizing"}),
        encoding="utf-8",
    )

    request = request_pause(tmp_path)

    assert request.step == "phase_4"


def test_pause_request_loses_terminal_race_without_leaving_stale_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_running_phase_4(tmp_path)

    def finish_after_marker(path, payload, **kwargs):
        write_json_atomic(path, payload, **kwargs)
        state_path = tmp_path / "pipeline_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["status"] = "complete"
        write_json_atomic(state_path, state)

    monkeypatch.setattr(run_control, "write_json_atomic", finish_after_marker)

    with pytest.raises(ValueError, match="running pipeline phase"):
        request_pause(tmp_path)

    assert not (tmp_path / "pause_request.json").exists()


def test_pause_command_and_status_expose_pausing_without_mutating_checkpoint(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    _write_running_phase_4(tmp_path)
    before = (tmp_path / "pipeline_state.json").read_bytes()

    assert worldsim_main.main(["pause", "--state-dir", str(tmp_path)]) == 0
    payload = build_status_payload(tmp_path)

    assert payload["lifecycle_status"] == "pausing"
    assert payload["pause_request"]["step"] == "phase_4"
    assert "status=pausing" in format_status_payload(payload)
    assert (tmp_path / "pipeline_state.json").read_bytes() == before
    assert "Pause requested" in capsys.readouterr().out


def test_status_does_not_project_stale_terminal_marker_as_pausing(tmp_path: Path) -> None:
    _write_running_phase_4(tmp_path)
    request_pause(tmp_path)
    state_path = tmp_path / "pipeline_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["status"] = "complete"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    payload = build_status_payload(tmp_path)

    assert "lifecycle_status" not in payload
    assert payload["pipeline_state"]["status"] == "complete"
    assert "running pipeline phase" in payload["pause_request_error"]
    assert "status=complete" in format_status_payload(payload)


def test_status_rejects_marker_with_wrong_step(tmp_path: Path) -> None:
    _write_running_phase_4(tmp_path)
    request_pause(tmp_path)
    marker = tmp_path / "pause_request.json"
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["step"] = "phase_3"
    marker.write_text(json.dumps(payload), encoding="utf-8")

    status = build_status_payload(tmp_path)

    assert "lifecycle_status" not in status
    assert "step is unsupported" in status["pause_request_error"]


@pytest.mark.parametrize(
    ("step", "status", "stage", "process_pool", "message"),
    [
        ("phase_3", "running", "initial_evaluation", False, "not pause-aware"),
        ("phase_4", "complete", "initial_evaluation", False, "running pipeline phase"),
        ("phase_4", "running", "finalizing", False, "not pause-aware"),
        ("phase_4", "running", "initial_evaluation", True, "process-pool"),
    ],
)
def test_pause_request_rejects_unsupported_lifecycle(
    tmp_path: Path,
    step: str,
    status: str,
    stage: str,
    process_pool: bool,
    message: str,
) -> None:
    _write_running_phase_4(tmp_path, stage=stage)
    state_path = tmp_path / "pipeline_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update(
        step=step,
        status=status,
        process_pool=process_pool,
        pause_stage=stage,
    )
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        request_pause(tmp_path)


def test_pause_acknowledgement_preserves_checkpoint_and_definition(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pointer = tmp_path / "resume-pointer.json"
    monkeypatch.setenv("WARP_TASKGEN_RESUME_POINTER", str(pointer))
    _write_running_phase_4(tmp_path)
    request = request_pause(tmp_path)

    updated = acknowledge_pause(tmp_path)

    assert updated["status"] == "paused"
    assert updated["step"] == "phase_4"
    assert updated["agent_model"] == "model-a"
    assert updated["pause_request_id"] == request.request_id
    assert not (tmp_path / "pause_request.json").exists()
    assert json.loads(pointer.read_text(encoding="utf-8"))["status"] == "paused"
    assert json.loads((tmp_path / "pipeline_state.json").read_text())["status"] == "paused"


def test_phase_2_planning_pause_is_atomic_idempotent_and_visible(
    tmp_path: Path,
) -> None:
    _write_running_phase_2(tmp_path)
    before = (tmp_path / "pipeline_state.json").read_bytes()

    first = request_pause(tmp_path)
    second = request_pause(tmp_path)
    status = build_status_payload(tmp_path)

    assert first == second
    assert first.step == "phase_2"
    assert status["lifecycle_status"] == "pausing"
    assert status["pause_request"]["step"] == "phase_2"
    assert (tmp_path / "pipeline_state.json").read_bytes() == before


@pytest.mark.parametrize("stage", ["feasibility", "complete"])
def test_phase_2_pause_rejects_nonplanning_stages(tmp_path: Path, stage: str) -> None:
    _write_running_phase_2(tmp_path, stage=stage)

    with pytest.raises(ValueError, match="not pause-aware"):
        request_pause(tmp_path)

    assert not (tmp_path / "pause_request.json").exists()


def test_phase_2_running_checkpoint_raises_only_after_atomic_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("WARP_TASKGEN_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("WARP_TASKGEN_RESUME_POINTER", str(tmp_path / "pointer.json"))
    _write_running_phase_2(tmp_path)
    request_pause(tmp_path)

    with pytest.raises(PauseBoundaryReached):
        save_state(
            "phase_2",
            status="running",
            phase_2_stage="planning",
            checkpoint="safe",
        )

    persisted = json.loads((tmp_path / "pipeline_state.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "running"
    assert persisted["checkpoint"] == "safe"


def test_phase_2_pause_wins_before_text_fill_transition(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("WARP_TASKGEN_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("WARP_TASKGEN_RESUME_POINTER", str(tmp_path / "pointer.json"))
    _write_running_phase_2(tmp_path)
    request_pause(tmp_path)

    with pytest.raises(PauseBoundaryReached):
        save_state(
            "phase_2",
            status="running",
            phase_2_stage="text_fill",
            checkpoint="must-not-promote",
        )

    persisted = json.loads((tmp_path / "pipeline_state.json").read_text(encoding="utf-8"))
    assert persisted["phase_2_stage"] == "planning"
    assert "checkpoint" not in persisted


def test_running_checkpoint_raises_boundary_only_after_atomic_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("WARP_TASKGEN_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("WARP_TASKGEN_RESUME_POINTER", str(tmp_path / "pointer.json"))
    _write_running_phase_4(tmp_path)
    request_pause(tmp_path)

    with pytest.raises(PauseBoundaryReached):
        save_state("phase_4", status="running", agent_model="model-a", checkpoint="safe")

    persisted = json.loads((tmp_path / "pipeline_state.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "running"
    assert persisted["checkpoint"] == "safe"


def test_interruption_clears_pending_pause_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("WARP_TASKGEN_RESUME_POINTER", str(tmp_path / "pointer.json"))
    _write_running_phase_4(tmp_path)
    request_pause(tmp_path)

    updated = mark_interrupted(tmp_path, signal_name="SIGTERM")

    assert updated["status"] == "interrupted"
    assert updated["interrupt_signal"] == "SIGTERM"
    assert load_pause_request(tmp_path) is None


def test_phase_boundary_adapter_persists_pause_after_operation_unwinds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("WARP_TASKGEN_RESUME_POINTER", str(tmp_path / "pointer.json"))
    _write_running_phase_4(tmp_path)
    request_pause(tmp_path)
    unwound = False
    guard_held = False

    @contextmanager
    def lifecycle_guard():
        nonlocal guard_held
        guard_held = True
        try:
            yield
        finally:
            guard_held = False

    acknowledge = cli_run_control.acknowledge_pause

    def checked_acknowledge(state_dir):
        assert guard_held is True
        return acknowledge(state_dir)

    monkeypatch.setattr(cli_run_control, "acknowledge_pause", checked_acknowledge)

    def operation() -> int:
        nonlocal unwound
        try:
            raise PauseBoundaryReached()
        finally:
            unwound = True

    rc = dispatch_phase_with_run_control(
        phase="4",
        state_dir=tmp_path,
        operation=operation,
        lifecycle_guard=lifecycle_guard,
    )

    assert rc == 0
    assert unwound is True
    assert json.loads((tmp_path / "pipeline_state.json").read_text())["status"] == "paused"


def test_phase_2_boundary_adapter_persists_pause_after_operation_unwinds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("WARP_TASKGEN_RESUME_POINTER", str(tmp_path / "pointer.json"))
    _write_running_phase_2(tmp_path)
    request_pause(tmp_path)

    rc = dispatch_phase_with_run_control(
        phase="2",
        state_dir=tmp_path,
        operation=lambda: (_ for _ in ()).throw(PauseBoundaryReached()),
        lifecycle_guard=nullcontext,
    )

    assert rc == 0
    state = json.loads((tmp_path / "pipeline_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "paused"
    assert state["phase_2_stage"] == "planning"


def test_phase_boundary_adapter_records_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("WARP_TASKGEN_RESUME_POINTER", str(tmp_path / "pointer.json"))
    _write_running_phase_4(tmp_path)

    def operation() -> int:
        raise KeyboardInterrupt

    rc = dispatch_phase_with_run_control(
        phase="4",
        state_dir=tmp_path,
        operation=operation,
        lifecycle_guard=nullcontext,
    )

    assert rc == 130
    state = json.loads((tmp_path / "pipeline_state.json").read_text())
    assert state["status"] == "interrupted"
    assert state["interrupt_signal"] == "SIGINT"


@pytest.mark.asyncio
async def test_pause_aware_map_finishes_admitted_item_and_stops_dequeue(tmp_path: Path) -> None:
    _write_running_phase_4(tmp_path, stage="postprocessing")
    started: list[int] = []
    finished: list[int] = []

    async def operation(item: int) -> int:
        started.append(item)
        if item == 0:
            request_pause(tmp_path)
        finished.append(item)
        return item * 2

    with pytest.raises(PauseBoundaryReached):
        await pause_aware_map([0, 1, 2], operation, concurrency=1, state_dir=tmp_path)

    assert started == [0]
    assert finished == [0]
