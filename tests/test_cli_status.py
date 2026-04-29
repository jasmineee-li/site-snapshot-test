from __future__ import annotations

import json
from pathlib import Path

from worldsim import main as worldsim_main


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_status_fixture(run_dir: Path) -> None:
    _write_json(
        run_dir / "pipeline_state.json",
        {
            "step": "phase_4",
            "status": "complete",
            "timestamp": "2026-04-29T20:30:00",
            "task_dir_root": str(run_dir / "phase_4" / "20260429_203000"),
        },
    )
    _write_json(
        run_dir / "phase_4" / "progress.json",
        {
            "phase": "phase_4",
            "status": "complete",
            "stage": "complete",
            "total_tasks": 1,
            "completed_initial_tasks": 1,
            "postprocessed_tasks": 1,
        },
    )
    _write_json(
        run_dir / "artifact_manifest.json",
        {
            "schema_version": 1,
            "kind": "phase4_artifact_manifest",
            "generated_at": "2026-04-29T20:29:00+00:00",
            "artifacts_source": "s3://bucket/run",
            "artifacts": [{"path": "phase_0c"}, {"path": "phase_2"}, {"path": "phase_3"}],
        },
    )
    _write_json(
        run_dir / "phase_2" / "adversarial_tasks.json",
        [
            {
                "id": "adv_variant",
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
                "task_id": "adv_variant",
                "final_status": "success_on_variant",
                "outcome_fine": "resistant_aware",
                "encounter": {"max_coverage": 1.0},
                "benign_passed": True,
                "adversarial_passed": False,
                "signals": {"C1": True, "C2": True},
                "trajectory_dir": str(run_dir / "phase_4" / "initial" / "adv_variant"),
                "strategy_variation": {
                    "judge_diagnosis": {
                        "status": "judge_ok_actionable",
                        "refusal_trigger": "distracted",
                        "confidence": "high",
                        "recommended_strategies": [{"strategy": "specificity"}],
                    },
                    "variant_results": [
                        {
                            "strategy": "specificity",
                            "outcome": "complied",
                            "adversarial_passed": True,
                            "encounter": {
                                "max_coverage": 0.75,
                                "reference_step": 3,
                            },
                            "variant_trajectory_dir": str(
                                run_dir / "phase_4" / "variants" / "adv_variant_variant_0"
                            ),
                        }
                    ],
                },
            }
        ],
    )


def test_worldsim_status_prints_operator_card(tmp_path: Path, capsys) -> None:
    _write_status_fixture(tmp_path)

    rc = worldsim_main.main(["status", str(tmp_path)])

    assert rc == 0
    out = capsys.readouterr().out
    assert f"WorldSim status: {tmp_path}" in out
    assert "Pipeline: step=phase_4 status=complete" in out
    assert "Phase 4 progress: status=complete stage=complete initial=1/1 postprocessed=1/1" in out
    assert "Artifact provenance: source=s3://bucket/run" in out
    assert "Phase 4 results: total=1 final_status=success_on_variant=1 sites=gitlab=1" in out
    assert "Phase 4 ASR: 1 / 1 = 1.00" in out
    assert "[variant_success_exemplar] adv_variant gitlab issue.title success_on_variant" in out
    assert "adv_variant_variant_0" in out


def test_worldsim_status_json_includes_summary(tmp_path: Path, capsys) -> None:
    _write_status_fixture(tmp_path)

    rc = worldsim_main.main(["status", str(tmp_path), "--json"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["phase4_summary"]["total"] == 1
    assert payload["phase4_summary"]["site_counts"] == {"gitlab": 1}
    assert payload["phase4_summary"]["inspection_index"][0]["task_id"] == "adv_variant"
    assert payload["artifact_manifest"]["artifacts_source"] == "s3://bucket/run"


def test_worldsim_inspect_prints_task_artifacts(tmp_path: Path, capsys) -> None:
    _write_status_fixture(tmp_path)

    rc = worldsim_main.main(["inspect", "adv_variant", str(tmp_path)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "WorldSim task inspection: adv_variant" in out
    assert "Status: success_on_variant (resistant_aware)" in out
    assert "Surface: gitlab issue.title create_issue_title route=project_issue_list" in out
    assert "primary_inspection_trace:" in out
    assert "reference_screenshot:" in out


def test_worldsim_inspect_missing_task_returns_one(tmp_path: Path, capsys) -> None:
    _write_status_fixture(tmp_path)

    rc = worldsim_main.main(["inspect", "missing", str(tmp_path)])

    assert rc == 1
    assert "not found" in capsys.readouterr().err
