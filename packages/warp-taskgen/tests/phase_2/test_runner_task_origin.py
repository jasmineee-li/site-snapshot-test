from __future__ import annotations

import json
from argparse import Namespace

import pytest

from warp_taskgen.phase_2 import runner

from ._fixtures import _benign_task


def test_phase_2_task_origin_filter_keeps_only_new_tasks():
    tasks = [
        {"id": "11", "origin": "existing_task"},
        {"id": "novel_gitlab_1", "origin": "new_task"},
        {"id": "novel_gitlab_2"},
    ]

    filtered = runner._filter_tasks_by_origin(
        tasks,
        "new_task",
        phase_label="Phase 2",
    )

    assert [task["id"] for task in filtered] == ["novel_gitlab_1", "novel_gitlab_2"]


def test_phase_2_task_origin_filter_rejects_unknown_origin():
    with pytest.raises(ValueError, match="--task-origin must be one of"):
        runner._filter_tasks_by_origin(
            [{"id": "novel_gitlab_1", "origin": "new_task"}],
            "paper_only",
            phase_label="Phase 2",
        )


def test_phase_2_adds_route_surface_overlay(tmp_path):
    phase_1 = tmp_path / "phase_1"
    phase_1.mkdir()
    (phase_1 / "TASK_ROUTE_CONTRACTS_gitlab.json").write_text(
        json.dumps(
            {
                "route_families": [
                    {
                        "id": "gitlab.issue_description.gitlab_search_result.create_issue_description",
                        "profile_surface_overlay": {
                            "id": "issue_description",
                            "source_field": "Issue.description",
                            "delivery_channels": [
                                {
                                    "mechanism": "api",
                                    "privileged_seed": False,
                                    "path_template": "/api/v4/projects/{project_id}/issues",
                                    "method": "POST",
                                    "body_field": "description",
                                }
                            ],
                            "compatible_concealments": ["plaintext"],
                            "controllable_by_tier": "any_user",
                        },
                    }
                ]
            }
        )
    )
    profile = {"site": "gitlab", "injection_surface": [{"id": "issue_note"}]}

    updated = runner._with_phase1_route_surface_overlays(
        "gitlab",
        profile,
        state_dir=tmp_path,
    )

    assert [surface["id"] for surface in updated["injection_surface"]] == [
        "issue_note",
        "issue_description",
    ]
    assert profile["injection_surface"] == [{"id": "issue_note"}]


def test_phase_2_planning_keeps_raw_site_key():
    from warp_taskgen.phase_2.planning_specs import build_planning_shard_specs

    plan = build_planning_shard_specs(
        [{"id": "task-1", "site": " shopping "}],
    )

    assert list(plan.tasks_by_site) == [" shopping "]
    assert plan.specs[0]["label"] == " shopping "


@pytest.mark.asyncio
async def test_phase_2_invalid_origin_persists_before_profiles_gate(monkeypatch, tmp_path):
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("WARP_TASKGEN_STATE_DIR", str(tmp_path))
    (tmp_path / "phase_1").mkdir(parents=True)
    (tmp_path / "phase_1" / "benign_tasks.json").write_text(json.dumps([_benign_task()]))

    rc = await runner.run(
        Namespace(
            skip_feasibility=True,
            sandbox_model="demo",
            task_origin="paper_only",
        )
    )

    assert rc == 1
    state = json.loads((tmp_path / "pipeline_state.json").read_text())
    assert state["status"] == "failed"
    assert state["reason"] == "invalid_task_origin"


@pytest.mark.asyncio
async def test_phase_2_planning_logs_filter_cap_and_site_selection(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("WARP_TASKGEN_STATE_DIR", str(tmp_path))
    (tmp_path / "phase_1").mkdir(parents=True)
    tasks = [
        {**_benign_task(), "id": f"novel-shopping-{index}", "origin": "new_task"}
        for index in range(3)
    ]
    (tmp_path / "phase_1" / "benign_tasks.json").write_text(json.dumps(tasks))
    (tmp_path / "phase_0c").mkdir()

    with caplog.at_level("INFO", logger="warp_taskgen.phase_2.runner"):
        rc = await runner.run(
            Namespace(
                skip_feasibility=True,
                sandbox_model="demo",
                task_origin="new_task",
                max_tasks_per_site=1,
                sites="shopping",
            )
        )

    assert rc == 1
    messages = [record.getMessage() for record in caplog.records]
    assert any("--task-origin filter active" in message for message in messages)
    assert any("task-origin filter kept 3/3 tasks" in message for message in messages)
    assert any(
        "capped at 1 tasks/site" in message and "(3 -> 1 tasks)" in message for message in messages
    )
    assert any(
        "--sites filter active, running only ['shopping']" in message for message in messages
    )
