from __future__ import annotations

import json
from pathlib import Path

from scripts import summarize_phase_4_sweep


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_run(run_dir: Path, *, model: str, status: str) -> None:
    _write_json(
        run_dir / "pipeline_state.json",
        {
            "agent_provider": "openai",
            "agent_model": model,
            "agent_service_tier": "priority",
            "sandbox_model": "claude-sonnet-4-6",
        },
    )
    _write_json(
        run_dir / "phase_2" / "adversarial_tasks.json",
        [
            {
                "id": "adv_gitlab_1",
                "site": "gitlab",
                "origin": "new_task",
                "exposure_contract": {
                    "target_surface_id": "issue.title",
                    "editor_method": "create_issue_title",
                    "surface_route": {"route_variant": "project_issue_list"},
                },
            }
        ],
    )
    _write_json(
        run_dir / "phase_4" / "results.json",
        [
            {
                "task_id": "adv_gitlab_1",
                "final_status": status,
                "outcome_fine": status,
                "encounter": {"max_coverage": 1.0},
                "benign_passed": True,
                "adversarial_passed": status == "complied",
            }
        ],
    )


def test_summarize_sweep_groups_runs_by_model_and_pairs_tasks(tmp_path: Path) -> None:
    run_a = tmp_path / "run-a"
    run_b = tmp_path / "run-b"
    _write_run(run_a, model="gpt-5.2", status="complied")
    _write_run(run_b, model="gpt-5.4", status="resistant")

    summary = summarize_phase_4_sweep.summarize_sweep([run_a, run_b])

    assert summary["run_count"] == 2
    assert summary["paired_task_count"] == 1
    assert summary["runs"][0]["model_key"] == "openai|gpt-5.2|priority|claude-sonnet-4-6"
    assert summary["runs"][1]["model_key"] == "openai|gpt-5.4|priority|claude-sonnet-4-6"
    assert summary["by_model_status"]["openai|gpt-5.2|priority|claude-sonnet-4-6"] == {
        "complied": 1
    }
    assert summary["by_model_status"]["openai|gpt-5.4|priority|claude-sonnet-4-6"] == {
        "resistant": 1
    }
    assert summary["task_rows"] == [
        {
            "task_id": "adv_gitlab_1",
            "site": "gitlab",
            "origin": "new_task",
            "route_id": None,
            "route_variant": "project_issue_list",
            "target_surface_id": "issue.title",
            "editor_method": "create_issue_title",
            "models": {
                "openai|gpt-5.2|priority|claude-sonnet-4-6": {
                    "adversarial_passed": True,
                    "benign_passed": True,
                    "final_status": "complied",
                    "max_coverage": 1.0,
                    "outcome_fine": "complied",
                    "steps": None,
                },
                "openai|gpt-5.4|priority|claude-sonnet-4-6": {
                    "adversarial_passed": False,
                    "benign_passed": True,
                    "final_status": "resistant",
                    "max_coverage": 1.0,
                    "outcome_fine": "resistant",
                    "steps": None,
                },
            },
        }
    ]


def test_format_sweep_summary_is_operator_readable(tmp_path: Path) -> None:
    run_a = tmp_path / "run-a"
    _write_run(run_a, model="gpt-5.2", status="complied")

    text = summarize_phase_4_sweep.format_sweep_summary(
        summarize_phase_4_sweep.summarize_sweep([run_a])
    )

    assert "Phase 4 sweep: runs=1 paired_tasks=1" in text
    assert "openai|gpt-5.2|priority|claude-sonnet-4-6" in text
    assert "statuses={'complied': 1}" in text
