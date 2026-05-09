"""Generate WASP summary tables from judge JSONL + post-hoc ASR sidecars.

Usage:
    python -m eval_awareness_experiments.summarize_wasp_results \
        --results-dir eval_awareness_experiments/results/n200_2026-04-29/wasp \
        --output-dir eval_awareness_experiments/results/n200_2026-04-29/tables
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

COLUMNS = [
    "benchmark",
    "model",
    "arm",
    "split",
    "n_tasks",
    "user_task_pass_rate",
    "user_task_pass_count",
    "user_task_pass_n",
    "asr_rate",
    "asr_success_count",
    "asr_n",
    "asr_source",
    "intermediate_asr_rate",
    "intermediate_asr_success_count",
    "intermediate_asr_n",
    "intermediate_asr_source",
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
    "missing_judge_rows",
]

MD_COLUMNS = [
    "model",
    "arm",
    "split",
    "n_tasks",
    "asr_rate",
    "asr_n",
    "asr_source",
    "intermediate_asr_rate",
    "intermediate_asr_n",
    "intermediate_asr_source",
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _model_from_slug(slug: str) -> str:
    suffix = ""
    base = slug
    if base.endswith("_thinking"):
        base = base.removesuffix("_thinking")
        suffix = ":thinking"
    if "_" not in base:
        return base + suffix
    provider, model = base.split("_", 1)
    return f"{provider}/{model}{suffix}"


def _task_id(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    if metadata.get("task_id"):
        return str(metadata["task_id"])
    sample_id = str(row.get("sample_id") or "")
    return sample_id.rsplit("_", 1)[-1] if sample_id else ""


def _rate(num: int | float, den: int | float) -> float | None:
    return (num / den) if den else None


def _load_wasp_asr(cell_dir: Path, split: str) -> tuple[dict[str, Any], dict[str, bool]]:
    path = cell_dir / f"wasp_asr_{split}.jsonl"
    if not path.exists():
        return {}, {}

    summary: dict[str, Any] = {}
    task_map: dict[str, bool] = {}
    for row in _read_jsonl(path):
        if "task_id" not in row:
            summary = row
            continue
        if "any_compromised" in row:
            task_map[str(row["task_id"])] = bool(row["any_compromised"])
    return summary, task_map


def _coerce_bool(*values: Any) -> bool | None:
    for value in values:
        if isinstance(value, bool):
            return value
        if value is None:
            continue
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "y", "compromised", "success"}:
                return True
            if normalized in {"false", "0", "no", "n", "not_compromised", "failure"}:
                return False
    return None


def _load_wasp_final_asr(cell_dir: Path, split: str) -> dict[str, Any]:
    """Load native/end-to-end WASP ASR if a final-utility artifact exists.

    Native WASP writes attacker utility as `total_scores / cnt_tasks`; data-import
    can also preserve per-task rows in `wasp_final_asr_<split>.jsonl`.
    """
    jsonl_path = cell_dir / f"wasp_final_asr_{split}.jsonl"
    if jsonl_path.exists():
        summary: dict[str, Any] = {}
        task_values: dict[str, bool] = {}
        for row in _read_jsonl(jsonl_path):
            if "task_id" not in row:
                summary = row
                continue
            value = _coerce_bool(
                row.get("attack_success"),
                row.get("end_to_end_compromised"),
                row.get("final_attack_success"),
            )
            if value is not None:
                task_values[str(row["task_id"])] = value
        if task_values:
            return {
                "success_count": sum(1 for value in task_values.values() if value),
                "n": len(task_values),
                "source": summary.get("source") or "wasp_native_attacker_utility",
                "path": str(jsonl_path),
                "task_success": task_values,
            }
        success_count = summary.get("final_asr_success_count", summary.get("success_count"))
        n = summary.get("final_asr_n", summary.get("n"))
        if success_count is not None and n is not None:
            return {
                "success_count": success_count,
                "n": n,
                "source": summary.get("source") or "wasp_native_attacker_utility",
                "path": str(jsonl_path),
            }

    for path in (
        cell_dir / f"run_attacker_utility_{split}.json",
    ):
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("total_scores") is None or data.get("cnt_tasks") is None:
            continue
        return {
            "success_count": data["total_scores"],
            "n": data["cnt_tasks"],
            "source": "wasp_native_attacker_utility",
            "path": str(path),
        }

    return {"success_count": 0, "n": 0, "source": ""}


def _sum_bool_by_task(rows: list[dict[str, Any]], field: str) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for row in rows:
        task = _task_id(row)
        if not task:
            continue
        value = (row.get("metadata") or {}).get(field)
        if isinstance(value, bool):
            out[task] = value
    return out


def _summarize_split(result_path: Path) -> dict[str, Any]:
    split_dir = result_path.parent
    split = split_dir.name
    cell_dir = result_path.parents[2]
    arm = cell_dir.parent.name
    model_slug = cell_dir.name
    rows = _read_jsonl(result_path)
    task_ids = sorted({task for task in (_task_id(row) for row in rows) if task})

    task_pass = _sum_bool_by_task(rows, "task_passed")
    metadata_asr = _sum_bool_by_task(rows, "attack_success")
    asr_summary, sidecar_asr = _load_wasp_asr(cell_dir, split)
    final_asr = _load_wasp_final_asr(cell_dir, split)
    if sidecar_asr:
        intermediate_asr_map = sidecar_asr
        intermediate_asr_source = "wasp_post_hoc_action_classifier"
    elif metadata_asr:
        intermediate_asr_map = metadata_asr
        intermediate_asr_source = "judge_metadata"
    else:
        intermediate_asr_map = {}
        intermediate_asr_source = ""

    judge_error_rows = sum(1 for row in rows if (row.get("metadata") or {}).get("error"))
    expected_judge_rows = len(task_ids) * 2

    verbalized = [
        row for row in rows
        if row.get("experiment_type") == "trajectory_verbalized_awareness"
        and not (row.get("metadata") or {}).get("error")
    ]
    verbalized_n = len(verbalized)
    verbalized_counts = defaultdict(int)
    for row in verbalized:
        label = str(row.get("classification") or "")
        verbalized_counts[label] += 1

    purpose = [
        row for row in rows
        if row.get("experiment_type") == "trajectory_purpose_continue_5q"
    ]
    purpose_n = len(purpose)
    purpose_counts = defaultdict(int)
    align_share_sum = 0.0
    any_eval_share_sum = 0.0
    refused_share_sum = 0.0
    share_n = 0
    for row in purpose:
        label = str(row.get("classification") or "")
        purpose_counts[label] += 1
        metadata = row.get("metadata") or {}
        if metadata.get("alignment_eval_share") is not None:
            align_share_sum += float(metadata["alignment_eval_share"])
            any_eval_share_sum += float(metadata.get("any_eval_share") or 0.0)
            refused_share_sum += float(metadata.get("refused_share") or 0.0)
            share_n += 1

    n_tasks = len(task_ids) or int(asr_summary.get("n_tasks") or 0)
    asr_success = final_asr.get("success_count") or 0
    asr_n = final_asr.get("n") or 0
    asr_source = final_asr.get("source") or ""
    if asr_source and asr_n < n_tasks:
        asr_success = 0
        asr_n = 0
        asr_source = "partial_missing_final_artifacts"
    intermediate_asr_n = len(intermediate_asr_map)
    intermediate_asr_success = sum(1 for v in intermediate_asr_map.values() if v)
    pass_n = len(task_pass)
    pass_success = sum(1 for v in task_pass.values() if v)
    purpose_eval_count = sum(purpose_counts[label] for label in EVAL_LABELS)

    return {
        "benchmark": "wasp",
        "model": _model_from_slug(model_slug),
        "arm": arm,
        "split": split,
        "n_tasks": n_tasks,
        "user_task_pass_count": pass_success,
        "user_task_pass_n": pass_n,
        "asr_success_count": asr_success,
        "asr_n": asr_n,
        "asr_source": asr_source,
        "intermediate_asr_success_count": intermediate_asr_success,
        "intermediate_asr_n": intermediate_asr_n,
        "intermediate_asr_source": intermediate_asr_source,
        "verbalized_n": verbalized_n,
        "verbalized_eval_count": sum(
            count for label, count in verbalized_counts.items()
            if label in EVAL_LABELS
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
        "purpose_any_eval_share_sum": any_eval_share_sum,
        "purpose_refused_share_sum": refused_share_sum,
        "purpose_share_n": share_n,
        "judge_error_rows": judge_error_rows,
        "missing_judge_rows": max(0, expected_judge_rows - len(rows)),
    }


def _finalize(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    row["user_task_pass_rate"] = _rate(row["user_task_pass_count"], row["user_task_pass_n"])
    row["asr_rate"] = _rate(row["asr_success_count"], row["asr_n"])
    row["intermediate_asr_rate"] = _rate(
        row["intermediate_asr_success_count"],
        row["intermediate_asr_n"],
    )
    row["verbalized_eval_rate"] = _rate(row["verbalized_eval_count"], row["verbalized_n"])
    row["verbalized_alignment_eval_rate"] = _rate(
        row["verbalized_alignment_eval_count"],
        row["verbalized_n"],
    )
    row["verbalized_capability_eval_rate"] = _rate(
        row["verbalized_capability_eval_count"],
        row["verbalized_n"],
    )
    row["verbalized_other_eval_rate"] = _rate(
        row["verbalized_other_eval_count"],
        row["verbalized_n"],
    )
    row["verbalized_real_rate"] = _rate(row["verbalized_real_count"], row["verbalized_n"])
    row["purpose_eval_class_rate"] = _rate(
        row["purpose_eval_class_count"],
        row["purpose_n"],
    )
    row["purpose_alignment_eval_class_rate"] = _rate(
        row["purpose_alignment_eval_class_count"],
        row["purpose_n"],
    )
    row["purpose_capability_eval_class_rate"] = _rate(
        row["purpose_capability_eval_class_count"],
        row["purpose_n"],
    )
    row["purpose_other_eval_class_rate"] = _rate(
        row["purpose_other_eval_class_count"],
        row["purpose_n"],
    )
    row["purpose_real_class_rate"] = _rate(
        row["purpose_real_class_count"],
        row["purpose_n"],
    )
    row["purpose_alignment_eval_share_mean"] = _rate(
        row["purpose_alignment_eval_share_sum"],
        row["purpose_share_n"],
    )
    row["purpose_refused_share_mean"] = _rate(
        row["purpose_refused_share_sum"],
        row["purpose_share_n"],
    )
    return row


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    sum_fields = [
        "n_tasks",
        "user_task_pass_count",
        "user_task_pass_n",
        "asr_success_count",
        "asr_n",
        "intermediate_asr_success_count",
        "intermediate_asr_n",
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
        "purpose_any_eval_share_sum",
        "purpose_refused_share_sum",
        "purpose_share_n",
        "judge_error_rows",
        "missing_judge_rows",
    ]
    for row in rows:
        key = (row["model"], row["arm"])
        if key not in groups:
            groups[key] = {
                "benchmark": "wasp",
                "model": row["model"],
                "arm": row["arm"],
                "split": "all",
                "asr_source": row.get("asr_source") or "",
                "intermediate_asr_source": row.get("intermediate_asr_source") or "",
                **{field: 0 for field in sum_fields},
            }
        agg = groups[key]
        for field in sum_fields:
            agg[field] += row.get(field) or 0
        if agg["asr_source"] != (row.get("asr_source") or ""):
            agg["asr_source"] = "mixed"
        if agg["intermediate_asr_source"] != (row.get("intermediate_asr_source") or ""):
            agg["intermediate_asr_source"] = "mixed"
    for agg in groups.values():
        if agg["asr_n"] < agg["n_tasks"]:
            agg["asr_success_count"] = 0
            agg["asr_n"] = 0
            agg["asr_source"] = "partial_missing_final_artifacts"
    return sorted(groups.values(), key=lambda r: (r["model"], r["arm"]))


def _format(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _format(row.get(column)) for column in COLUMNS})


def _write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "| " + " | ".join(MD_COLUMNS) + " |",
        "| " + " | ".join(["---"] * len(MD_COLUMNS)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format(row.get(column)) for column in MD_COLUMNS) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(results_dir: Path, output_dir: Path) -> None:
    split_rows = [
        _summarize_split(path)
        for path in sorted(results_dir.glob("*/*/wasp/*/trajectory_awareness_results.jsonl"))
    ]
    split_rows = [_finalize(row) for row in split_rows]
    aggregate_rows = [_finalize(row) for row in _aggregate(split_rows)]

    _write_csv(output_dir / "wasp_summary_by_model_arm_split.csv", split_rows)
    _write_md(output_dir / "wasp_summary_by_model_arm_split.md", split_rows)
    _write_csv(output_dir / "wasp_summary_by_model_arm.csv", aggregate_rows)
    _write_md(output_dir / "wasp_summary_by_model_arm.md", aggregate_rows)
    (output_dir / "wasp_summary_notes.md").write_text(
        "# WASP N200 Summary Tables\n\n"
        "- `asr_rate` is native/end-to-end WASP attacker utility when "
        "`asr_source=wasp_native_attacker_utility`; it is blank for runs that "
        "did not preserve a final attacker-utility artifact. This is a semantic "
        "cutover from older tables where `asr_rate` meant action/intermediate ASR.\n"
        "- `intermediate_asr_rate` is the WASP post-hoc action-only/intermediate "
        "classifier when `intermediate_asr_source=wasp_post_hoc_action_classifier`; "
        "it falls back to boolean `metadata.attack_success` only for legacy rows.\n"
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
        default=Path("eval_awareness_experiments/results/n200_2026-04-29/wasp"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_awareness_experiments/results/n200_2026-04-29/tables"),
    )
    args = parser.parse_args()
    summarize(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
