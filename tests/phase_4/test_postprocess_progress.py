"""Coverage for Phase 4 postprocess heartbeat infrastructure.

Locks in the schema_version=1 progress.json shape that worldsim/cli_status.py
and scripts/remote_job_status.sh consume, and the callback chain wiring used
by worldsim/phase_4/runner.run().
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from worldsim.phase_4.postprocess_progress import (
    Phase4ProgressState,
    _phase_4_progress_path,
    compute_progress_extra,
    record_postprocess_result,
    record_postprocess_start,
    record_variant_progress,
    write_postprocess_progress,
)


def _new_state(tmp_path: Path) -> Phase4ProgressState:
    return Phase4ProgressState(
        state_dir=tmp_path,
        task_dir_root=tmp_path / "phase_4" / "20260505_120000",
        total_tasks=3,
        completed_initial_tasks=3,
    )


def _read_progress(state_dir: Path) -> dict:
    return json.loads(_phase_4_progress_path(state_dir).read_text())


def test_compute_progress_extra_emits_cli_status_schema_keys(tmp_path):
    state = _new_state(tmp_path)
    state.started_task_ids.update({"t1", "t2"})
    state.active_task_ids.add("t2")
    state.completed_task_ids.add("t1")
    state.variant_progress_by_task["t2"] = {
        "task_id": "t2",
        "event": "variant_evaluation_complete",
        "generation_attempted": 3,
        "generation_generated": 2,
        "generation_failed": 1,
        "evaluated": 2,
        "pvpo_valid": 1,
        "complied": 1,
    }

    extra = compute_progress_extra(state)

    assert extra["postprocess_started_tasks"] == 2
    assert extra["active_postprocess_tasks"] == 1
    assert extra["active_postprocess_task_ids"] == ["t2"]
    variant = extra["variant_progress"]
    assert variant["schema_version"] == 1
    for key in (
        "budget_preset",
        "budget_shape",
        "entered_tasks",
        "active_tasks",
        "generation_attempted",
        "generation_generated",
        "generation_failed",
        "evaluated",
        "pvpo_valid",
        "complied",
        "task_samples",
    ):
        assert key in variant, f"variant_progress missing {key!r}"
    assert variant["entered_tasks"] == 1
    assert variant["active_tasks"] == 1
    assert variant["generation_attempted"] == 3
    assert variant["generation_generated"] == 2
    assert variant["generation_failed"] == 1
    assert variant["evaluated"] == 2
    assert variant["pvpo_valid"] == 1
    assert variant["complied"] == 1


def test_write_postprocess_progress_persists_atomic_schema_v1(tmp_path):
    state = _new_state(tmp_path)
    state.started_task_ids.add("t1")
    state.active_task_ids.add("t1")

    asyncio.run(write_postprocess_progress(state))

    payload = _read_progress(tmp_path)
    assert payload["schema_version"] == 1
    assert payload["phase"] == "phase_4"
    assert payload["status"] == "running"
    assert payload["stage"] == "postprocessing"
    assert payload["total_tasks"] == 3
    assert payload["completed_initial_tasks"] == 3
    assert payload["postprocessed_tasks"] == 0
    assert payload["postprocess_attempted_tasks"] == 0
    assert payload["postprocess_failed_tasks"] == 0
    assert payload["postprocess_started_tasks"] == 1
    assert payload["active_postprocess_tasks"] == 1
    assert payload["active_postprocess_task_ids"] == ["t1"]


def test_record_postprocess_lifecycle_reflects_success_and_failure_counts(tmp_path):
    state = _new_state(tmp_path)

    async def _run() -> None:
        await record_postprocess_start(state, "t1")
        await record_postprocess_start(state, "t2")
        await record_postprocess_result(state, "t1")
        await record_postprocess_result(state, "t2", failed=True)

    asyncio.run(_run())

    payload = _read_progress(tmp_path)
    assert payload["postprocess_started_tasks"] == 2
    assert payload["active_postprocess_tasks"] == 0
    assert payload["postprocessed_tasks"] == 1
    assert payload["postprocess_failed_tasks"] == 1
    assert payload["postprocess_attempted_tasks"] == 2


def test_record_variant_progress_accumulates_event_payload_keys(tmp_path):
    state = _new_state(tmp_path)

    async def _run() -> None:
        await record_postprocess_start(state, "t1")
        await record_variant_progress(
            state,
            "t1",
            "judge_complete",
            {
                "judge_status": "judge_ok_actionable",
                "refusal_trigger": "policy",
                "recommended_strategy_count": 2,
            },
        )
        await record_variant_progress(
            state,
            "t1",
            "variant_generation_recorded",
            {
                "generation_attempted": 2,
                "generation_generated": 2,
                "generation_failed": 0,
            },
        )
        await record_variant_progress(
            state,
            "t1",
            "variant_evaluation_complete",
            {
                "evaluated": 2,
                "pvpo_valid": 2,
                "complied": 1,
            },
        )

    asyncio.run(_run())

    progress = state.variant_progress_by_task["t1"]
    assert progress["event"] == "variant_evaluation_complete"
    assert progress["judge_status"] == "judge_ok_actionable"
    assert progress["refusal_trigger"] == "policy"
    assert progress["recommended_strategy_count"] == 2
    assert progress["generation_attempted"] == 2
    assert progress["generation_generated"] == 2
    assert progress["generation_failed"] == 0
    assert progress["evaluated"] == 2
    assert progress["pvpo_valid"] == 2
    assert progress["complied"] == 1
    assert "updated_at" in progress

    payload = _read_progress(tmp_path)
    variant = payload["variant_progress"]
    assert variant["generation_attempted"] == 2
    assert variant["generation_generated"] == 2
    assert variant["evaluated"] == 2
    assert variant["pvpo_valid"] == 2
    assert variant["complied"] == 1


def test_concurrent_record_calls_preserve_all_updates(tmp_path):
    state = _new_state(tmp_path)
    state.total_tasks = 16

    async def _run() -> None:
        await asyncio.gather(*(record_postprocess_start(state, f"t{i}") for i in range(16)))
        await asyncio.gather(
            *(record_postprocess_result(state, f"t{i}", failed=(i % 4 == 0)) for i in range(16))
        )

    asyncio.run(_run())

    assert len(state.started_task_ids) == 16
    assert len(state.active_task_ids) == 0
    assert len(state.failed_task_ids) == 4
    assert len(state.completed_task_ids) == 12

    payload = _read_progress(tmp_path)
    assert payload["postprocess_started_tasks"] == 16
    assert payload["postprocessed_tasks"] == 12
    assert payload["postprocess_failed_tasks"] == 4


def test_record_variant_progress_jsonable_payload_keeps_dict_writeable(tmp_path):
    state = _new_state(tmp_path)

    async def _run() -> None:
        await record_postprocess_start(state, "t1")
        # Pass a Path to confirm _jsonable_payload coerces non-JSON-native values.
        await record_variant_progress(
            state,
            "t1",
            "judge_complete",
            {"profile_path": Path("/tmp/profile.json"), "judge_status": "judge_ok_actionable"},
        )

    asyncio.run(_run())

    progress = state.variant_progress_by_task["t1"]
    assert progress["profile_path"] == "/tmp/profile.json"
    payload = _read_progress(tmp_path)
    sample = payload["variant_progress"]["task_samples"][0]
    assert sample["profile_path"] == "/tmp/profile.json"


@pytest.mark.parametrize(
    "active, completed, failed, expected_attempted",
    [
        ([], [], [], 0),
        (["t1"], [], [], 0),
        ([], ["t1"], [], 1),
        ([], [], ["t1"], 1),
        ([], ["t1", "t2"], ["t3"], 3),
    ],
)
def test_postprocess_attempted_equals_completed_plus_failed(
    tmp_path, active, completed, failed, expected_attempted
):
    state = _new_state(tmp_path)
    state.active_task_ids.update(active)
    state.completed_task_ids.update(completed)
    state.failed_task_ids.update(failed)

    asyncio.run(write_postprocess_progress(state))

    payload = _read_progress(tmp_path)
    assert payload["postprocess_attempted_tasks"] == expected_attempted
    assert payload["postprocessed_tasks"] == len(completed)
    assert payload["postprocess_failed_tasks"] == len(failed)
