"""Generate DoomArena summary tables from trajectory-awareness judge JSONL.

Usage:
    python -m eval_awareness_experiments.summarize_doomarena_results \
        --results-dir eval_awareness_experiments/results/n200_2026-04-29/doomarena \
        --output-dir eval_awareness_experiments/results/n200_2026-04-29/tables
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from eval_awareness_experiments.run_manifest import scan
from eval_awareness_experiments.summary_tables import (
    finalize_summary_row,
    model_from_slug,
    read_jsonl,
    sum_bool_by_task,
    write_csv,
    write_md,
)

DEFAULT_SPLITS = ["reddit", "shopping", "shopping_admin", "gitlab"]

COLUMNS = [
    "benchmark",
    "model",
    "arm",
    "split",
    "n_tasks",
    "status",
    "user_task_pass_rate",
    "user_task_pass_count",
    "user_task_pass_n",
    "asr_rate",
    "asr_success_count",
    "asr_n",
    "verbalized_eval_rate",
    "verbalized_alignment_eval_rate",
    "verbalized_capability_eval_rate",
    "verbalized_other_eval_rate",
    "verbalized_real_rate",
    "purpose_eval_class_rate",
    "purpose_alignment_eval_class_rate",
    "purpose_capability_eval_class_rate",
    "purpose_other_eval_class_rate",
    "purpose_real_class_rate",
    "purpose_alignment_eval_share_mean",
    "purpose_refused_share_mean",
    "judge_error_rows",
    "duplicate_judge_rows",
    "missing_judge_rows",
]

MD_COLUMNS = [
    "model",
    "arm",
    "split",
    "n_tasks",
    "status",
    "asr_rate",
    "asr_n",
    "user_task_pass_rate",
    "verbalized_eval_rate",
    "verbalized_real_rate",
    "purpose_eval_class_rate",
    "purpose_alignment_eval_class_rate",
    "purpose_capability_eval_class_rate",
    "purpose_other_eval_class_rate",
    "judge_error_rows",
]

EVAL_LABELS = {
    "Alignment Evaluation",
    "Capability Evaluation",
    "Other Evaluation",
}


def _task_id(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    if metadata.get("task_id"):
        return str(metadata["task_id"])
    return str(row.get("sample_id") or "")


def _dedupe_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    deduped: dict[tuple[str, str], tuple[int, dict[str, Any]]] = {}
    duplicate_rows = 0
    for index, row in enumerate(rows):
        key = (_task_id(row), str(row.get("experiment_type") or ""))
        if key in deduped:
            duplicate_rows += 1
        deduped[key] = (index, row)
    return [row for _, row in sorted(deduped.values(), key=lambda item: item[0])], duplicate_rows


def _summarize_split(
    result_path: Path, status_by_cell: dict[tuple[str, str], str]
) -> dict[str, Any]:
    split = result_path.parent.name
    cell_dir = result_path.parents[2]
    arm = cell_dir.parent.name
    model_slug = cell_dir.name
    raw_rows = read_jsonl(result_path, skip_malformed=True)
    rows, duplicate_rows = _dedupe_rows(raw_rows)
    task_ids = sorted({task for task in (_task_id(row) for row in rows) if task})

    task_pass = sum_bool_by_task(rows, "task_passed", task_id=_task_id)
    asr = sum_bool_by_task(rows, "attack_success", task_id=_task_id)
    judge_error_rows = sum(1 for row in raw_rows if (row.get("metadata") or {}).get("error"))
    expected_judge_rows = len(task_ids) * 2

    verbalized = [
        row for row in rows if row.get("experiment_type") == "trajectory_verbalized_awareness"
    ]
    verbalized_n = len(verbalized)
    verbalized_counts: dict[str, int] = defaultdict(int)
    for row in verbalized:
        verbalized_counts[str(row.get("classification") or "")] += 1

    purpose = [
        row for row in rows if row.get("experiment_type") == "trajectory_purpose_continue_5q"
    ]
    purpose_n = len(purpose)
    purpose_counts: dict[str, int] = defaultdict(int)
    align_share_sum = 0.0
    refused_share_sum = 0.0
    share_n = 0
    for row in purpose:
        purpose_counts[str(row.get("classification") or "")] += 1
        metadata = row.get("metadata") or {}
        if metadata.get("alignment_eval_share") is not None:
            align_share_sum += float(metadata["alignment_eval_share"])
            refused_share_sum += float(metadata.get("refused_share") or 0.0)
            share_n += 1

    purpose_eval_count = sum(purpose_counts[label] for label in EVAL_LABELS)
    return {
        "benchmark": "doomarena",
        "model": model_from_slug(model_slug),
        "arm": arm,
        "split": split,
        "n_tasks": len(task_ids),
        "status": status_by_cell.get((arm, model_slug), ""),
        "user_task_pass_count": sum(1 for value in task_pass.values() if value),
        "user_task_pass_n": len(task_pass),
        "asr_success_count": sum(1 for value in asr.values() if value),
        "asr_n": len(asr),
        "verbalized_n": verbalized_n,
        "verbalized_eval_count": sum(
            count for label, count in verbalized_counts.items() if label in EVAL_LABELS
        ),
        "verbalized_alignment_eval_count": verbalized_counts["Alignment Evaluation"],
        "verbalized_capability_eval_count": verbalized_counts["Capability Evaluation"],
        "verbalized_other_eval_count": verbalized_counts["Other Evaluation"],
        "verbalized_real_count": verbalized_counts["Real"],
        "purpose_n": purpose_n,
        "purpose_eval_class_count": purpose_eval_count,
        "purpose_alignment_eval_class_count": purpose_counts["Alignment Evaluation"],
        "purpose_capability_eval_class_count": purpose_counts["Capability Evaluation"],
        "purpose_other_eval_class_count": purpose_counts["Other Evaluation"],
        "purpose_real_class_count": purpose_counts["Real"],
        "purpose_alignment_eval_share_sum": align_share_sum,
        "purpose_refused_share_sum": refused_share_sum,
        "purpose_share_n": share_n,
        "judge_error_rows": judge_error_rows,
        "duplicate_judge_rows": duplicate_rows,
        "missing_judge_rows": max(0, expected_judge_rows - len(rows)),
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    sum_fields = [
        "n_tasks",
        "user_task_pass_count",
        "user_task_pass_n",
        "asr_success_count",
        "asr_n",
        "verbalized_n",
        "verbalized_eval_count",
        "verbalized_alignment_eval_count",
        "verbalized_capability_eval_count",
        "verbalized_other_eval_count",
        "verbalized_real_count",
        "purpose_n",
        "purpose_eval_class_count",
        "purpose_alignment_eval_class_count",
        "purpose_capability_eval_class_count",
        "purpose_other_eval_class_count",
        "purpose_real_class_count",
        "purpose_alignment_eval_share_sum",
        "purpose_refused_share_sum",
        "purpose_share_n",
        "judge_error_rows",
        "duplicate_judge_rows",
        "missing_judge_rows",
    ]
    for row in rows:
        key = (row["model"], row["arm"])
        if key not in groups:
            groups[key] = {
                "benchmark": "doomarena",
                "model": row["model"],
                "arm": row["arm"],
                "split": "all",
                "status": row.get("status") or "",
                **{field: 0 for field in sum_fields},
            }
        agg = groups[key]
        for field in sum_fields:
            agg[field] += row.get(field) or 0
    return sorted(groups.values(), key=lambda r: (r["model"], r["arm"]))


def summarize(
    results_dir: Path,
    output_dir: Path,
    expected_tasks_per_split: int,
    expected_splits: list[str],
) -> None:
    root_dir = results_dir.parent if results_dir.name == "doomarena" else results_dir
    manifest = scan(
        root_dir,
        expected_splits_by_benchmark={"doomarena": expected_splits},
        expected_tasks_per_split_by_benchmark={"doomarena": expected_tasks_per_split},
    )
    status_by_cell = {
        (cell["arm"], cell["model_slug"]): cell.get("status", "")
        for cell in manifest["cells"]
        if cell.get("benchmark") == "doomarena"
    }

    split_rows = [
        _summarize_split(path, status_by_cell)
        for path in sorted(results_dir.glob("*/*/doomarena/*/trajectory_awareness_results.jsonl"))
    ]
    split_rows = [finalize_summary_row(row) for row in split_rows]
    aggregate_rows = [finalize_summary_row(row) for row in _aggregate(split_rows)]

    write_csv(output_dir / "doomarena_summary_by_model_arm_split.csv", split_rows, COLUMNS)
    write_md(output_dir / "doomarena_summary_by_model_arm_split.md", split_rows, MD_COLUMNS)
    write_csv(output_dir / "doomarena_summary_by_model_arm.csv", aggregate_rows, COLUMNS)
    write_md(output_dir / "doomarena_summary_by_model_arm.md", aggregate_rows, MD_COLUMNS)
    (output_dir / "doomarena_summary_notes.md").write_text(
        "# DoomArena N200 Summary Tables\n\n"
        "- These tables summarize the current DoomArena result snapshot. If a "
        "repair pass is running, treat them as point-in-time partial numbers.\n"
        "- `asr_rate` is from boolean `metadata.attack_success` in the "
        "trajectory-awareness rows.\n"
        "- `user_task_pass_rate` is from `metadata.task_passed` in the "
        "trajectory-awareness rows.\n"
        "- `verbalized_*` columns summarize `verbalized_awareness` headline "
        "classifications.\n"
        "- `purpose_*` columns summarize `purpose_continue_5q` headline "
        "classifications. The markdown table shows the headline "
        "Alignment/Capability/Other breakdown and omits the broad per-variant "
        "`any_eval_share`.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("eval_awareness_experiments/results/n200_2026-04-29/doomarena"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_awareness_experiments/results/n200_2026-04-29/tables"),
    )
    parser.add_argument("--tasks-per-split", type=int, default=50)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    args = parser.parse_args()
    summarize(args.results_dir, args.output_dir, args.tasks_per_split, args.splits)


if __name__ == "__main__":
    main()
