"""Summarize Phase 4 results with final-status and variant-aware ASR metrics.

Usage:

    uv run python scripts/summarize_phase_4_results.py logs/<run>
    uv run python scripts/summarize_phase_4_results.py logs/<run>/phase_4/results.json --json

The input may be a WorldSim state directory, a ``phase_4`` directory, or a
direct ``results.json`` path. Task metadata is loaded from
``phase_2/adversarial_tasks.json`` when available, and can be supplemented
with ``--tasks``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from worldsim.phase_4.result_summary import summarize_results


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _resolve_results_path(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = [
        path / "phase_4" / "results.json",
        path / "results.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"could not find Phase 4 results at {path}/phase_4/results.json or {path}/results.json"
    )


def _default_task_paths(results_path: Path) -> list[Path]:
    paths: list[Path] = []
    parent = results_path.parent
    if parent.name == "phase_4":
        paths.append(parent.parent / "phase_2" / "adversarial_tasks.json")
    paths.append(parent / "phase_2" / "adversarial_tasks.json")
    return [p for p in paths if p.exists()]


def _load_task_lookup(paths: list[Path]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for path in paths:
        data = _load_json(path)
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and isinstance(data.get("tasks"), list):
            items = data["tasks"]
        elif isinstance(data, dict):
            items = [data]
        else:
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            task_id = item.get("id")
            if isinstance(task_id, str) and task_id:
                lookup[task_id] = item
    return lookup


def _fmt_rate(rate: float | None) -> str:
    return f"{rate:.2f}" if rate is not None else "n/a"


def format_text_summary(summary: dict[str, Any], *, results_path: Path) -> str:
    lines = [f"Phase 4 results: {results_path}"]
    lines.append(f"Total: {summary['total']} tasks; scorable: {summary['scorable']}")
    lines.append(
        "Final Gate-1 ASR: "
        f"{summary['asr_valid_numerator']} / {summary['asr_valid_denominator']} = "
        f"{_fmt_rate(summary['asr_valid'])}"
    )
    lines.append(
        "Final compliant with adversarial reward: "
        f"{summary['final_compliant_with_adversarial_reward']} / "
        f"{summary['final_compliant_denominator']}"
    )
    lines.append(
        "Initial direct adversarial reward success (B2 ∧ D1): "
        f"{summary['initial_adversarial_reward_success_numerator']} / "
        f"{summary['initial_adversarial_reward_success_denominator']} = "
        f"{_fmt_rate(summary['initial_adversarial_reward_success_rate'])}"
    )
    lines.append(
        "Benign capability under attack (B1 ∧ D1): "
        f"{summary['benign_capability_numerator']} / "
        f"{summary['benign_capability_denominator']} = "
        f"{_fmt_rate(summary['benign_capability'])}"
    )
    lines.append(
        "Initial conditional ASR (complied | C1 ∧ C2 ∧ D1): "
        f"{summary['conditional_initial_asr_numerator']} / "
        f"{summary['conditional_initial_asr_denominator']} = "
        f"{_fmt_rate(summary['conditional_initial_asr'])}"
    )
    lines.append(f"Final status counts: {summary['final_status_counts']}")
    lines.append(f"Outcome fine counts: {summary['outcome_fine_counts']}")
    lines.append(f"Site counts: {summary['site_counts']}")
    lines.append(f"Origin counts: {summary['origin_counts']}")
    if summary.get("judge_trigger_counts"):
        lines.append(f"Judge trigger counts: {summary['judge_trigger_counts']}")

    rows = summary.get("by_site_surface_editor_status") or []
    if rows:
        lines.append("By site/surface/editor/status:")
        for row in rows:
            lines.append(
                "  "
                f"{row['count']} {row['site']} {row['surface']} {row['editor_method']} "
                f"{row['final_status']} {row['outcome_fine']}"
            )

    strategy_rows = summary.get("variant_strategy_outcomes") or []
    if strategy_rows:
        lines.append("Variant strategy outcomes:")
        for row in strategy_rows:
            lines.append(
                "  "
                f"{row['count']} {row['strategy']} {row['outcome']} {row['gate1']}"
            )

    successes = summary.get("variant_successes") or []
    if successes:
        lines.append("Variant successes:")
        for row in successes:
            lines.append(
                "  "
                f"{row['task_id']} {row['site']} {row['surface']} "
                f"{row['editor_method']} strategy={row['strategy']}"
            )

    errors = summary.get("variant_error_buckets") or []
    if errors:
        lines.append("Variant generation/error buckets:")
        for row in errors:
            lines.append(f"  {row['count']} {row['class']}: {row['reason']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize Phase 4 results with final-status and variant-aware metrics."
    )
    parser.add_argument("path", type=Path, help="State dir, phase_4 dir, or results.json path.")
    parser.add_argument(
        "--tasks",
        type=Path,
        nargs="*",
        default=[],
        help="Optional adversarial task JSON file(s) used for site/surface/origin breakdowns.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args(argv)

    try:
        results_path = _resolve_results_path(args.path)
        results = _load_json(results_path)
        if not isinstance(results, list):
            raise ValueError(f"{results_path} must contain a list of result objects")
        task_paths = [*args.tasks, *_default_task_paths(results_path)]
        task_lookup = _load_task_lookup(task_paths)
        summary = summarize_results(results, task_lookup=task_lookup)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        payload = {"results_path": str(results_path), **summary}
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(format_text_summary(summary, results_path=results_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
