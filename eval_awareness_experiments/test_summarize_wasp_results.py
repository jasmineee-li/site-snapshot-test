from __future__ import annotations

import csv
import json
from pathlib import Path

from eval_awareness_experiments.summarize_wasp_results import summarize


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_summarize_wasp_uses_final_asr_as_headline_and_keeps_intermediate(
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "results" / "wasp"
    cell_dir = results_dir / "bare" / "anthropic_claude-sonnet-4.6"
    split_dir = cell_dir / "wasp" / "gitlab"

    _write_jsonl(
        split_dir / "trajectory_awareness_results.jsonl",
        [
            {
                "sample_id": "wasp_gitlab_task1",
                "experiment_type": "trajectory_verbalized_awareness",
                "classification": "Real",
                "metadata": {"task_id": "task1", "task_passed": True},
            },
            {
                "sample_id": "wasp_gitlab_task2",
                "experiment_type": "trajectory_verbalized_awareness",
                "classification": "Real",
                "metadata": {"task_id": "task2", "task_passed": True},
            },
        ],
    )
    _write_jsonl(
        cell_dir / "wasp_asr_gitlab.jsonl",
        [
            {"schema_version": "wasp_asr_v1"},
            {"task_id": "task1", "any_compromised": True},
            {"task_id": "task2", "any_compromised": False},
        ],
    )
    _write_jsonl(
        cell_dir / "wasp_final_asr_gitlab.jsonl",
        [
            {"source": "wasp_native_attacker_utility"},
            {"task_id": "task1", "final_attack_success": False},
            {"task_id": "task2", "final_attack_success": True},
        ],
    )

    output_dir = tmp_path / "tables"
    summarize(results_dir, output_dir)

    with (output_dir / "wasp_summary_by_model_arm_split.csv").open(
        newline="",
        encoding="utf-8",
    ) as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    row = rows[0]
    assert row["asr_source"] == "wasp_native_attacker_utility"
    assert row["asr_success_count"] == "1"
    assert row["asr_n"] == "2"
    assert row["asr_rate"] == "0.500000"
    assert row["intermediate_asr_source"] == "wasp_post_hoc_action_classifier"
    assert row["intermediate_asr_success_count"] == "1"
    assert row["intermediate_asr_n"] == "2"
    assert row["intermediate_asr_rate"] == "0.500000"


def test_summarize_wasp_leaves_headline_asr_blank_without_native_artifact(
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "results" / "wasp"
    cell_dir = results_dir / "bare" / "anthropic_claude-sonnet-4.6"
    split_dir = cell_dir / "wasp" / "gitlab"

    _write_jsonl(
        split_dir / "trajectory_awareness_results.jsonl",
        [
            {
                "sample_id": "wasp_gitlab_task1",
                "experiment_type": "trajectory_verbalized_awareness",
                "classification": "Real",
                "metadata": {"task_id": "task1", "task_passed": True},
            },
        ],
    )
    _write_jsonl(
        cell_dir / "wasp_asr_gitlab.jsonl",
        [
            {"schema_version": "wasp_asr_v1"},
            {"task_id": "task1", "any_compromised": True},
        ],
    )

    output_dir = tmp_path / "tables"
    summarize(results_dir, output_dir)

    with (output_dir / "wasp_summary_by_model_arm_split.csv").open(
        newline="",
        encoding="utf-8",
    ) as f:
        row = next(csv.DictReader(f))

    assert row["asr_rate"] == ""
    assert row["asr_n"] == "0"
    assert row["asr_source"] == ""
    assert row["intermediate_asr_rate"] == "1.000000"
    assert row["intermediate_asr_n"] == "1"
    assert row["intermediate_asr_source"] == "wasp_post_hoc_action_classifier"


def test_summarize_wasp_does_not_reuse_unsplit_final_artifact_for_split_rows(
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "results" / "wasp"
    cell_dir = results_dir / "bare" / "anthropic_claude-sonnet-4.6"
    split_dir = cell_dir / "wasp" / "gitlab"
    _write_jsonl(
        split_dir / "trajectory_awareness_results.jsonl",
        [
            {
                "sample_id": "wasp_gitlab_task1",
                "experiment_type": "trajectory_verbalized_awareness",
                "classification": "Real",
                "metadata": {"task_id": "task1", "task_passed": True},
            },
        ],
    )
    (cell_dir / "run_attacker_utility.json").write_text(
        json.dumps({"total_scores": 1, "cnt_tasks": 1}),
        encoding="utf-8",
    )

    output_dir = tmp_path / "tables"
    summarize(results_dir, output_dir)

    with (output_dir / "wasp_summary_by_model_arm_split.csv").open(
        newline="",
        encoding="utf-8",
    ) as f:
        row = next(csv.DictReader(f))

    assert row["asr_rate"] == ""
    assert row["asr_source"] == ""


def test_summarize_wasp_blanks_split_headline_asr_when_final_coverage_partial(
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "results" / "wasp"
    cell_dir = results_dir / "bare" / "anthropic_claude-sonnet-4.6"
    split_dir = cell_dir / "wasp" / "gitlab"
    _write_jsonl(
        split_dir / "trajectory_awareness_results.jsonl",
        [
            {
                "sample_id": "wasp_gitlab_task1",
                "experiment_type": "trajectory_verbalized_awareness",
                "classification": "Real",
                "metadata": {"task_id": "task1", "task_passed": True},
            },
            {
                "sample_id": "wasp_gitlab_task2",
                "experiment_type": "trajectory_verbalized_awareness",
                "classification": "Real",
                "metadata": {"task_id": "task2", "task_passed": True},
            },
        ],
    )
    _write_jsonl(
        cell_dir / "wasp_final_asr_gitlab.jsonl",
        [
            {"source": "wasp_native_attacker_utility"},
            {"task_id": "task1", "final_attack_success": True},
        ],
    )

    output_dir = tmp_path / "tables"
    summarize(results_dir, output_dir)

    with (output_dir / "wasp_summary_by_model_arm_split.csv").open(
        newline="",
        encoding="utf-8",
    ) as f:
        row = next(csv.DictReader(f))

    assert row["asr_rate"] == ""
    assert row["asr_n"] == "0"
    assert row["asr_source"] == "partial_missing_final_artifacts"


def test_summarize_wasp_blanks_aggregate_headline_asr_when_split_coverage_partial(
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "results" / "wasp"
    cell_dir = results_dir / "bare" / "anthropic_claude-sonnet-4.6"
    for split in ("gitlab", "reddit"):
        _write_jsonl(
            cell_dir / "wasp" / split / "trajectory_awareness_results.jsonl",
            [
                {
                    "sample_id": f"wasp_{split}_task1",
                    "experiment_type": "trajectory_verbalized_awareness",
                    "classification": "Real",
                    "metadata": {"task_id": f"{split}_task1", "task_passed": True},
                },
            ],
        )
    _write_jsonl(
        cell_dir / "wasp_final_asr_gitlab.jsonl",
        [
            {"source": "wasp_native_attacker_utility"},
            {"task_id": "gitlab_task1", "final_attack_success": True},
        ],
    )

    output_dir = tmp_path / "tables"
    summarize(results_dir, output_dir)

    with (output_dir / "wasp_summary_by_model_arm.csv").open(
        newline="",
        encoding="utf-8",
    ) as f:
        row = next(csv.DictReader(f))

    assert row["asr_rate"] == ""
    assert row["asr_n"] == "0"
    assert row["asr_source"] == "partial_missing_final_artifacts"
