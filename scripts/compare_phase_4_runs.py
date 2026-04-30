"""Compare two Phase 4 result sets by stable task id."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from worldsim.phase_4.artifact_audit import (
    compare_phase4_runs,
    default_task_paths,
    load_json,
    load_task_lookup,
    resolve_phase4_results_path,
)


def _fmt_count_map(values: dict[str, Any]) -> str:
    if not values:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in sorted(values.items()))


def _fmt_rate(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "n/a"


def _format_text_report(
    report: dict[str, Any],
    *,
    baseline_path: Path,
    candidate_path: Path,
    row_limit: int,
) -> str:
    baseline = report["baseline_summary"]
    candidate = report["candidate_summary"]
    lines = [
        "Phase 4 paired comparison:",
        f"  baseline:  {baseline_path}",
        f"  candidate: {candidate_path}",
    ]
    lines.append(
        "Method: task-id paired comparison; this highlights stochastic or code-driven "
        "outcome shifts without treating a single run as a carrier verdict."
    )
    lines.append(
        "ASR: "
        f"baseline {baseline['asr_valid_numerator']}/{baseline['asr_valid_denominator']}="
        f"{_fmt_rate(baseline['asr_valid'])}; "
        f"candidate {candidate['asr_valid_numerator']}/{candidate['asr_valid_denominator']}="
        f"{_fmt_rate(candidate['asr_valid'])}."
    )
    lines.append(
        "Coverage: "
        f"paired={report['paired_tasks']} "
        f"baseline_only={len(report['baseline_only_tasks'])} "
        f"candidate_only={len(report['candidate_only_tasks'])}; "
        f"success_gains={report['success_gains']} "
        f"success_losses={report['success_losses']}."
    )
    lines.append(f"Transitions: {_fmt_count_map(report['transition_counts'])}.")
    if report["baseline_only_tasks"]:
        lines.append(
            "Baseline-only tasks: " + ", ".join(report["baseline_only_tasks"][:10])
        )
    if report["candidate_only_tasks"]:
        lines.append(
            "Candidate-only tasks: " + ", ".join(report["candidate_only_tasks"][:10])
        )
    if report["rows"] and row_limit != 0:
        lines.append("Task transitions:")
        for row in report["rows"][: max(row_limit, 0)]:
            lines.append(
                "  "
                f"{row['task_id']} {row['site']} {row['surface']} "
                f"{row['editor_method']} route={row['route_variant']} "
                f"{row['transition']} "
                f"trigger={row['baseline_trigger']}->{row['candidate_trigger']} "
                f"strategy={row['baseline_successful_strategy'] or 'none'}->"
                f"{row['candidate_successful_strategy'] or 'none'}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two Phase 4 result sets.")
    parser.add_argument("baseline", type=Path, help="Baseline state dir, phase_4 dir, or results.")
    parser.add_argument("candidate", type=Path, help="Candidate state dir, phase_4 dir, or results.")
    parser.add_argument(
        "--baseline-tasks",
        type=Path,
        nargs="*",
        default=[],
        help="Optional baseline adversarial task JSON file(s).",
    )
    parser.add_argument(
        "--candidate-tasks",
        type=Path,
        nargs="*",
        default=[],
        help="Optional candidate adversarial task JSON file(s).",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument(
        "--row-limit",
        type=int,
        default=20,
        help="Number of task transition rows to print in text mode (0 disables).",
    )
    args = parser.parse_args(argv)

    try:
        baseline_path = resolve_phase4_results_path(args.baseline)
        candidate_path = resolve_phase4_results_path(args.candidate)
        baseline_results = load_json(baseline_path)
        candidate_results = load_json(candidate_path)
        if not isinstance(baseline_results, list):
            raise ValueError(f"{baseline_path} must contain a list of result objects")
        if not isinstance(candidate_results, list):
            raise ValueError(f"{candidate_path} must contain a list of result objects")
        baseline_lookup = load_task_lookup(
            [*args.baseline_tasks, *default_task_paths(baseline_path)]
        )
        candidate_lookup = load_task_lookup(
            [*args.candidate_tasks, *default_task_paths(candidate_path)]
        )
        report = compare_phase4_runs(
            baseline_results,
            candidate_results,
            baseline_task_lookup=baseline_lookup,
            candidate_task_lookup=candidate_lookup,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(
            json.dumps(
                {
                    "baseline_results_path": str(baseline_path),
                    "candidate_results_path": str(candidate_path),
                    **report,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(
            _format_text_report(
                report,
                baseline_path=baseline_path,
                candidate_path=candidate_path,
                row_limit=args.row_limit,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
