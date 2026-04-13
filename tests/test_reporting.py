from __future__ import annotations

import json

from worldsim.reporting import load_phase_results


def test_load_phase_results_prefers_summary_file(tmp_path):
    results_dir = tmp_path / "phase_4"
    results_dir.mkdir()
    (results_dir / "results.json").write_text(
        json.dumps(
            [
                {"task_id": "task-a", "final_status": "complied"},
                {"task_id": "task-b", "final_status": "resistant"},
            ]
        )
    )

    task_dir = results_dir / "task-a"
    task_dir.mkdir()
    (task_dir / "result.json").write_text(json.dumps({"task_id": "task-a", "outcome": "complied"}))

    loaded = load_phase_results(results_dir)

    assert [item["task_id"] for item in loaded] == ["task-a", "task-b"]


def test_load_phase_results_ignores_nested_rerun_artifacts(tmp_path):
    results_dir = tmp_path / "phase_4_runs"
    results_dir.mkdir()

    canonical = results_dir / "task-a"
    canonical.mkdir()
    (canonical / "result.json").write_text(json.dumps({"task_id": "task-a", "outcome": "complied"}))

    variant = results_dir / "task-a_variant_0"
    variant.mkdir()
    (variant / "result.json").write_text(
        json.dumps({"task_id": "task-a", "outcome": "refused_or_ignored"})
    )

    ecoval = results_dir / "task-a__ecoval_1"
    ecoval.mkdir()
    (ecoval / "result.json").write_text(
        json.dumps({"task_id": "task-a", "outcome": "task_broke"})
    )

    loaded = load_phase_results(results_dir)

    assert loaded == [{"task_id": "task-a", "outcome": "complied"}]
