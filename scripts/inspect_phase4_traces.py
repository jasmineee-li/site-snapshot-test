#!/usr/bin/env python3
"""Inspect Phase 4 traces without dumping raw trajectory JSON.

Examples:
  uv run python scripts/inspect_phase4_traces.py logs/run summary --action create_issue_note
  uv run python scripts/inspect_phase4_traces.py logs/run slice --outcome resistant_unaware --limit 10
  uv run python scripts/inspect_phase4_traces.py logs/run task adv-123 --iterator --refs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from worldsim.phase_4.trace_inspection import (
    DEFAULT_FIELDS,
    build_summary,
    build_task_detail,
    filter_results,
    format_text,
    load_inspection,
    schema,
    task_row,
)


def _fields(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    fields = [item.strip() for item in raw.split(",") if item.strip()]
    return fields or None


def _add_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--site", help="Filter by site, e.g. gitlab or reddit.")
    parser.add_argument(
        "--action",
        help="Filter by adversarial action kind/family or task editor method.",
    )
    parser.add_argument("--status", help="Filter by final_status.")
    parser.add_argument("--outcome", help="Filter by outcome_fine.")
    parser.add_argument("--task-id", help="Filter by exact task id.")


def _json_or_text(payload: dict[str, Any], *, output: str, command: str) -> None:
    if output == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(format_text(payload, command=command))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="State dir, phase_4 dir, or results.json path.")
    parser.add_argument(
        "--tasks",
        type=Path,
        nargs="*",
        default=[],
        help="Optional adversarial_tasks.json path(s). Defaults to sibling phase_2.",
    )
    parser.add_argument(
        "--output",
        choices=("text", "json"),
        default="text",
        help="Output format. Default: text.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Shortcut for --output json.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary_cmd = subparsers.add_parser("summary", help="Compact aggregate counts.")
    _add_filters(summary_cmd)
    summary_cmd.add_argument("--limit", type=int, default=8, help="Sample rows to include.")

    slice_cmd = subparsers.add_parser("slice", help="Compact matching task table.")
    _add_filters(slice_cmd)
    slice_cmd.add_argument("--limit", type=int, default=20, help="Maximum rows.")
    slice_cmd.add_argument(
        "--fields",
        default=",".join(DEFAULT_FIELDS),
        help="Comma-separated fields for table/JSON rows.",
    )

    task_cmd = subparsers.add_parser("task", help="Explain one task.")
    task_cmd.add_argument("task_id", help="Task id to inspect.")
    task_cmd.add_argument("--iterator", action="store_true", help="Include compact iterator attempts.")
    task_cmd.add_argument("--refs", action="store_true", help="Include artifact paths.")

    subparsers.add_parser("schema", help="Print command schema for agents.")

    args = parser.parse_args(argv)
    output = "json" if args.json else args.output

    if args.command == "schema":
        _json_or_text(schema(), output=output, command="schema")
        return 0

    results_path, phase4_dir, results, task_lookup = load_inspection(
        args.path,
        task_paths=args.tasks,
    )

    if args.command == "task":
        filtered = filter_results(
            results,
            task_lookup,
            task_id=args.task_id,
        )
        if not filtered:
            raise SystemExit(f"task not found: {args.task_id}")
        payload = build_task_detail(
            filtered[0],
            task_lookup,
            phase4_dir=phase4_dir,
            include_iterator=args.iterator,
            include_refs=args.refs,
        )
        payload["results_path"] = str(results_path)
        _json_or_text(payload, output=output, command="task")
        return 0

    filtered = filter_results(
        results,
        task_lookup,
        site=args.site,
        action=args.action,
        status=args.status,
        outcome=args.outcome,
        task_id=args.task_id,
    )
    if args.command == "summary":
        payload = build_summary(
            results_path,
            phase4_dir,
            results,
            task_lookup,
            filtered=filtered,
            sample_limit=max(args.limit, 0),
        )
        _json_or_text(payload, output=output, command="summary")
        return 0

    fields = _fields(args.fields)
    rows = [
        task_row(result, task_lookup, phase4_dir=phase4_dir, fields=fields)
        for result in filtered[: max(args.limit, 0)]
    ]
    payload = {
        "schema_version": "phase4_trace_inspection_slice_v1",
        "results_path": str(results_path),
        "matched_results": len(filtered),
        "returned_rows": len(rows),
        "rows": rows,
    }
    _json_or_text(payload, output=output, command="slice")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
