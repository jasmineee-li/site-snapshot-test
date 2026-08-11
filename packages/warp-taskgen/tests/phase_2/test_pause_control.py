from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from worldsim.phase_2.pause_control import (
    planning_shard_checkpoint_matches,
    run_planning_shards,
    write_planning_shard_checkpoint,
)
from worldsim.phase_2.run_lock import Phase2AlreadyRunning, phase_2_run_lock
from worldsim.run_control import PauseBoundaryReached, request_pause
from worldsim.run_definition import define_run
from worldsim.run_transition import resolve_run_request


def _write_identified_planning_state(root: Path) -> object:
    transition = resolve_run_request(
        {"sandbox_model": "model-a"},
        existing_state=None,
        new_run_id="run-phase2-planning",
    )
    state = {
        "step": "phase_2",
        "status": "running",
        "timestamp": "2026-08-11T00:00:00+00:00",
        "logs_dir": str(root),
        "sandbox_model": "model-a",
        "phase_2_stage": "planning",
        "run_definition": transition.definition.to_dict(),
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "pipeline_state.json").write_text(json.dumps(state), encoding="utf-8")
    return transition.definition


@pytest.mark.asyncio
async def test_planning_scheduler_drains_active_shards_and_stops_claiming(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("WARP_TASKGEN_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("WARP_TASKGEN_RESUME_POINTER", str(tmp_path / "pointer.json"))
    _write_identified_planning_state(tmp_path)
    started: list[int] = []
    finished: list[int] = []
    second_started = asyncio.Event()
    pause_written = asyncio.Event()

    async def operation(item: int) -> int:
        started.append(item)
        if item == 1:
            second_started.set()
            await pause_written.wait()
        if item == 0:
            await second_started.wait()
            request_pause(tmp_path)
            pause_written.set()
        finished.append(item)
        return item

    with pytest.raises(PauseBoundaryReached):
        await run_planning_shards(
            [0, 1, 2],
            operation,
            concurrency=2,
            state_dir=tmp_path,
        )

    assert sorted(started) == [0, 1]
    assert sorted(finished) == [0, 1]


def test_planning_shard_checkpoint_is_atomic_and_run_bound(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("WARP_TASKGEN_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("WARP_TASKGEN_RESUME_POINTER", str(tmp_path / "pointer.json"))
    definition = _write_identified_planning_state(tmp_path)
    path = tmp_path / "phase_2" / "shards" / "gitlab-shard-0.json"
    payload = [{"id": "adv-1", "benign_task_id": "benign-1", "site": "gitlab"}]

    write_planning_shard_checkpoint(
        path,
        payload,
        label="gitlab-shard-0",
        input_task_ids=["benign-1"],
    )

    assert planning_shard_checkpoint_matches(path, payload, definition=definition)
    assert not planning_shard_checkpoint_matches(
        path,
        payload,
        definition=definition,
        expected_input_task_ids=["different-benign-task"],
    )
    assert not planning_shard_checkpoint_matches(
        path,
        [{**payload[0], "id": "tampered"}],
        definition=definition,
    )
    manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == "run-phase2-planning"
    assert "model-a" not in json.dumps(manifest)


def test_empty_planning_checkpoint_does_not_satisfy_nonempty_shard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from worldsim.phase_2.shards import _load_reusable_planning_shard

    monkeypatch.setenv("WARP_TASKGEN_STATE_DIR", str(tmp_path))
    definition = _write_identified_planning_state(tmp_path)
    path = tmp_path / "phase_2" / "shards" / "gitlab.json"
    write_planning_shard_checkpoint(
        path,
        [],
        label="gitlab",
        input_task_ids=["benign-1"],
    )

    assert (
        _load_reusable_planning_shard(
            path,
            expected_site="gitlab",
            expected_input_task_ids=["benign-1"],
            definition=definition,
            benign_by_id={},
            site_profiles={},
        )
        is None
    )


def test_planning_checkpoint_rejects_legacy_definition(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("WARP_TASKGEN_STATE_DIR", str(tmp_path))
    state = {
        "step": "phase_2",
        "status": "running",
        "logs_dir": str(tmp_path),
        "phase_2_stage": "planning",
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "pipeline_state.json").write_text(json.dumps(state), encoding="utf-8")
    path = tmp_path / "phase_2" / "shards" / "gitlab-shard-0.json"
    payload = [{"id": "adv-1"}]
    write_planning_shard_checkpoint(
        path,
        payload,
        label="gitlab-shard-0",
        input_task_ids=["benign-1"],
    )

    assert not planning_shard_checkpoint_matches(
        path,
        payload,
        definition=define_run(state),
    )


def test_phase_2_run_lock_rejects_concurrent_owner(tmp_path: Path) -> None:
    with phase_2_run_lock(tmp_path):
        with pytest.raises(Phase2AlreadyRunning):
            with phase_2_run_lock(tmp_path):
                pass
