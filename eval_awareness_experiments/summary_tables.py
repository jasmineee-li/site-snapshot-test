"""Shared helpers for the summary tables under ``results/<run>/tables/``.

Single source for the JSONL reader, the model-slug display name, the rate
arithmetic, and the CSV/markdown writers that the DoomArena and WASP
summarizers, the N200 analysis-table generator, and the run manifest all
used to carry as private copies.

This is a leaf module: stdlib only, and it must not import ``run_manifest``
or anything that imports it, because ``run_manifest`` imports from here.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


def read_jsonl(path: Path, *, skip_malformed: bool = False) -> list[dict[str, Any]]:
    """Read one JSON object per line from ``path``.

    Blank lines are skipped and a missing file yields ``[]``. With
    ``skip_malformed=True`` a line that fails to parse is dropped instead of
    raising; the matrix runner reads judge JSONL while it is still being
    written, so its readers (the DoomArena summarizer and the run manifest)
    pass ``True``. The WASP summarizer and the N200 generator keep the strict
    default so a corrupt result file fails loudly.
    """
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if skip_malformed:
                    continue
                raise
    return rows


def model_from_slug(slug: str) -> str:
    """Turn a results-directory slug (``provider_model[_thinking]``) back into
    the ``provider/model[:thinking]`` display name."""
    suffix = ""
    base = slug
    if base.endswith("_thinking"):
        base = base.removesuffix("_thinking")
        suffix = ":thinking"
    if "_" not in base:
        return base + suffix
    provider, model = base.split("_", 1)
    return f"{provider}/{model}{suffix}"


def rate(num: int | float, den: int | float) -> float | None:
    """``num / den``, or ``None`` when the denominator is zero."""
    return (num / den) if den else None


def sum_bool_by_task(
    rows: list[dict[str, Any]],
    field: str,
    *,
    task_id: Callable[[dict[str, Any]], str],
) -> dict[str, bool]:
    """Map task id to the boolean ``metadata[field]`` of its last row.

    ``task_id`` is the caller's row-to-task-id function: the DoomArena and
    WASP summarizers derive the id differently, so that seam stays with them.
    Rows without a task id or without a boolean value are ignored.
    """
    out: dict[str, bool] = {}
    for row in rows:
        task = task_id(row)
        if not task:
            continue
        value = (row.get("metadata") or {}).get(field)
        if isinstance(value, bool):
            out[task] = value
    return out


def finalize_summary_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a count row with the derived ``*_rate`` and
    ``*_mean`` columns filled in from its ``*_count`` / ``*_n`` fields."""
    row = dict(row)
    row["user_task_pass_rate"] = rate(row["user_task_pass_count"], row["user_task_pass_n"])
    row["asr_rate"] = rate(row["asr_success_count"], row["asr_n"])
    row["verbalized_eval_rate"] = rate(row["verbalized_eval_count"], row["verbalized_n"])
    row["verbalized_alignment_eval_rate"] = rate(
        row["verbalized_alignment_eval_count"],
        row["verbalized_n"],
    )
    row["verbalized_capability_eval_rate"] = rate(
        row["verbalized_capability_eval_count"],
        row["verbalized_n"],
    )
    row["verbalized_other_eval_rate"] = rate(
        row["verbalized_other_eval_count"],
        row["verbalized_n"],
    )
    row["verbalized_real_rate"] = rate(row["verbalized_real_count"], row["verbalized_n"])
    row["purpose_eval_class_rate"] = rate(
        row["purpose_eval_class_count"],
        row["purpose_n"],
    )
    row["purpose_alignment_eval_class_rate"] = rate(
        row["purpose_alignment_eval_class_count"],
        row["purpose_n"],
    )
    row["purpose_capability_eval_class_rate"] = rate(
        row["purpose_capability_eval_class_count"],
        row["purpose_n"],
    )
    row["purpose_other_eval_class_rate"] = rate(
        row["purpose_other_eval_class_count"],
        row["purpose_n"],
    )
    row["purpose_real_class_rate"] = rate(
        row["purpose_real_class_count"],
        row["purpose_n"],
    )
    row["purpose_alignment_eval_share_mean"] = rate(
        row["purpose_alignment_eval_share_sum"],
        row["purpose_share_n"],
    )
    row["purpose_refused_share_mean"] = rate(
        row["purpose_refused_share_sum"],
        row["purpose_share_n"],
    )
    return row


def format_cell(value: Any) -> str:
    """Render one table cell: empty for ``None``, six decimals for floats."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]], columns: Sequence[str]) -> None:
    """Write ``rows`` to ``path`` as CSV with exactly ``columns`` in order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: format_cell(row.get(column)) for column in columns})


def write_md(path: Path, rows: list[dict[str, Any]], columns: Sequence[str]) -> None:
    """Write ``rows`` to ``path`` as a markdown pipe table with ``columns``."""
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(row.get(column)) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
