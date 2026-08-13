from __future__ import annotations

import json

import pytest

from warp_taskgen.phase_2 import runner


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
