from __future__ import annotations

import json
from pathlib import Path

from eval_awareness_experiments.run_manifest import _aggregate_judge_results


def test_aggregate_judge_results_excludes_vea_judge_errors(tmp_path: Path) -> None:
    path = tmp_path / "trajectory_awareness_results.jsonl"
    rows = [
        {
            "sample_id": "sample_1",
            "experiment_type": "trajectory_verbalized_awareness",
            "classification": "Other",
            "metadata": {"error": "json_parse_error"},
        },
        {
            "sample_id": "sample_2",
            "experiment_type": "trajectory_verbalized_awareness",
            "classification": "Alignment Evaluation",
            "metadata": {},
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    result = _aggregate_judge_results(path)

    assert result["n_vea"] == 1
    assert result["n_vea_errors"] == 1
    assert result["vea_count"] == 1


def test_aggregate_judge_results_excludes_wasp_intermediate_asr_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trajectory_awareness_results.jsonl"
    rows = [
        {
            "sample_id": "sample_1",
            "experiment_type": "trajectory_verbalized_awareness",
            "classification": "Real",
            "metadata": {
                "attack_success": True,
                "attack_success_source": "wasp_post_hoc_action_classifier",
            },
        },
        {
            "sample_id": "sample_2",
            "experiment_type": "trajectory_verbalized_awareness",
            "classification": "Real",
            "metadata": {"attack_success": True},
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    result = _aggregate_judge_results(path)

    assert result["n_attack_samples"] == 1
    assert result["n_attacks"] == 1
