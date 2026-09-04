"""Phase 2a planning status and checkpoint-classification regressions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from warp_taskgen.phase_2 import shards


def test_build_planning_shard_specs_preserves_filter_cap_order_and_labels() -> None:
    from warp_taskgen.phase_2.planning_specs import build_planning_shard_specs

    tasks = [
        {"id": "shopping-a", "site": "shopping", "origin": "new_task"},
        {"id": "shopping-b", "site": "shopping", "origin": "new_task"},
        {"id": "reddit-a", "site": "reddit", "origin": "new_task"},
        {"id": "reddit-b", "site": "reddit", "origin": "new_task"},
        {"id": "shopping-existing", "site": "shopping", "origin": "existing_task"},
    ]

    plan = build_planning_shard_specs(
        tasks,
        task_origin="new_task",
        max_tasks_per_site=1,
        sites_filter="shopping,reddit",
    )

    assert [
        (spec["label"], [task["id"] for task in spec["site_tasks"]]) for spec in plan.specs
    ] == [("shopping", ["shopping-a"]), ("reddit", ["reddit-b"])]


def test_planning_checkpoint_inspection_classifies_pending_stale_and_malformed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from warp_taskgen.phase_2.pause_control import write_planning_shard_checkpoint
    from warp_taskgen.run_transition import resolve_run_request

    monkeypatch.setenv("WARP_TASKGEN_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("WARP_TASKGEN_RESUME_POINTER", str(tmp_path / "pointer.json"))
    transition = resolve_run_request(
        {"sandbox_model": "model-a"},
        existing_state=None,
        new_run_id="run-status-planning",
    )
    definition = transition.definition
    (tmp_path / "pipeline_state.json").write_text(
        json.dumps(
            {
                "step": "phase_2",
                "status": "running",
                "logs_dir": str(tmp_path),
                "phase_2_stage": "planning",
                "run_definition": definition.to_dict(),
            }
        ),
        encoding="utf-8",
    )
    shards_dir = tmp_path / "phase_2" / "shards"
    payload = [{"id": "adv-1", "benign_task_id": "benign-1", "site": "shopping"}]
    stale_path = shards_dir / "stale.json"
    write_planning_shard_checkpoint(
        stale_path,
        payload,
        label="stale",
        input_task_ids=["benign-1"],
    )
    manifest_path = stale_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["definition_digest"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    malformed_path = shards_dir / "malformed.json"
    write_planning_shard_checkpoint(
        malformed_path,
        payload,
        label="malformed",
        input_task_ids=["benign-1"],
    )
    malformed_path.with_suffix(".manifest.json").write_text("not-json", encoding="utf-8")

    invalid_output_path = shards_dir / "invalid-output.json"
    write_planning_shard_checkpoint(
        invalid_output_path,
        payload,
        label="invalid-output",
        input_task_ids=["benign-1"],
    )
    invalid_output_path.write_text(
        json.dumps([{**payload[0], "site": "reddit"}]),
        encoding="utf-8",
    )

    inspection = shards.inspect_planning_shard_checkpoints(
        shards_dir,
        definition=definition,
        expected_shards=[
            {"label": "stale", "site": "shopping", "input_task_ids": ["benign-1"]},
            {"label": "malformed", "site": "shopping", "input_task_ids": ["benign-1"]},
            {
                "label": "invalid-output",
                "site": "shopping",
                "input_task_ids": ["benign-1"],
            },
            {"label": "pending", "site": "shopping", "input_task_ids": ["benign-1"]},
        ],
        benign_by_id={"benign-1": {}},
        site_profiles={"shopping": {}},
    )

    assert inspection["status"] == "inspected"
    assert inspection["expected_count"] == 4
    assert inspection["compatible_count"] == 0
    assert inspection["pending_count"] == 1
    assert inspection["stale_count"] == 1
    assert inspection["malformed_count"] == 2
    assert {row["status"] for row in inspection["shards"]} == {
        "pending",
        "stale",
        "malformed",
    }
