from __future__ import annotations

from eval_awareness_experiments.generate_n200_analysis_tables import (
    JudgeRow,
    _bare_vea_tables,
    _is_vea,
)


def test_n200_tables_exclude_vea_judge_errors_from_vea_counts() -> None:
    row = JudgeRow(
        benchmark="wasp",
        arm="bare",
        model_slug="model-a",
        split="gitlab",
        sample_id="task-1",
        experiment_type="trajectory_verbalized_awareness",
        classification="Other",
        metadata={"error": "json_parse_error"},
    )

    assert not _is_vea(row)


def test_n200_bare_vea_tables_exclude_judge_errors_from_denominators() -> None:
    rows = [
        JudgeRow(
            benchmark="wasp",
            arm="bare",
            model_slug="model-a",
            split="gitlab",
            sample_id="task-1",
            experiment_type="trajectory_verbalized_awareness",
            classification="Alignment Evaluation",
            metadata={},
        ),
        JudgeRow(
            benchmark="wasp",
            arm="bare",
            model_slug="model-a",
            split="gitlab",
            sample_id="task-2",
            experiment_type="trajectory_verbalized_awareness",
            classification="Other",
            metadata={"error": "json_parse_error"},
        ),
    ]

    benchmark_table, model_table = _bare_vea_tables(rows)

    assert "| WASP | 1 | 1 | 100.00% |" in benchmark_table
    assert "| WASP | model-a | 1 | 1 | 100.00% |" in model_table
