#!/usr/bin/env python3
"""Export Phase 4 scenario-funnel evidence rows.

The export is observational. It reads existing Phase 2/4 artifacts and writes a
reviewer-facing projection of task instructions, payloads, action evidence,
created issue traces, benign utility, PVPO, and strict action success.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from warp_taskgen.phase_4.scenario_funnel_export import build_scenario_funnel_export
from warp_taskgen.phase_4.scenario_funnel_outputs import (
    write_scenario_funnel_csv,
    write_scenario_funnel_json,
    write_scenario_funnel_markdown,
)


def _formats(raw: str) -> set[str]:
    values = {item.strip().lower() for item in raw.split(",") if item.strip()}
    allowed = {"json", "csv", "md"}
    unknown = values - allowed
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown format(s): {', '.join(sorted(unknown))}; allowed: json,csv,md"
        )
    return values or {"json"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="State dir, phase_4 dir, or results.json path.")
    parser.add_argument(
        "--tasks",
        type=Path,
        nargs="*",
        default=[],
        help="Optional adversarial task JSON file(s). Defaults to sibling phase_2.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--formats", type=_formats, default={"json"})
    parser.add_argument("--commit-sha", default=None)
    parser.add_argument("--source-run-dir", default=None)
    parser.add_argument("--text-limit", type=int, default=900)
    parser.add_argument(
        "--payload-limit",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    text_limit = args.payload_limit if args.payload_limit is not None else args.text_limit

    export = build_scenario_funnel_export(
        args.path,
        task_paths=args.tasks,
        commit_sha=args.commit_sha,
        source_run_dir=args.source_run_dir,
        text_limit=text_limit,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if "json" in args.formats:
        path = args.output_dir / "scenario_funnel_evidence.json"
        write_scenario_funnel_json(export, path)
        print(path)
    if "csv" in args.formats:
        path = args.output_dir / "scenario_funnel_evidence.csv"
        write_scenario_funnel_csv(export, path)
        print(path)
    if "md" in args.formats:
        path = args.output_dir / "scenario_funnel_evidence.md"
        write_scenario_funnel_markdown(export, path)
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
