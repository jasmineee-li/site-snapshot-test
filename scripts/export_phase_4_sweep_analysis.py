#!/usr/bin/env python3
"""Export reproducible, report-only analysis for a Phase 4 model sweep."""

from __future__ import annotations

import argparse
from pathlib import Path

from worldsim.phase_4.sweep_analysis import analyze_sweep, write_analysis_outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-summary",
        type=Path,
        required=True,
        help="Path to sweep_summary.json from scripts/summarize_phase_4_sweep.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for analysis.json, CSVs, and Markdown reports.",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        type=Path,
        default=[],
        help=(
            "Optional compact run directory override. Repeatable. Matched by "
            "directory basename against run_dir entries in the sweep summary."
        ),
    )
    parser.add_argument(
        "--network-summary",
        type=Path,
        default=None,
        help="Optional targeted network_trace_summaries_manifest.json.",
    )
    args = parser.parse_args(argv)

    analysis = analyze_sweep(
        sweep_summary_path=args.sweep_summary,
        run_dirs=args.run_dir,
        network_summary_path=args.network_summary,
    )
    write_analysis_outputs(analysis, args.output_dir)
    for name in (
        "analysis.json",
        "model_summary.md",
        "task_matrix.csv",
        "failure_buckets.csv",
        "research_findings.md",
    ):
        print(args.output_dir / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
