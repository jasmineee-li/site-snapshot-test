"""CLI plumbing for the Phase 4 trace inspector."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from worldsim.phase_4.trace_inspection import (
    ALL_FIELDS,
    DEFAULT_FIELDS,
    build_summary,
    build_task_detail,
    build_timeline,
    filter_results,
    format_text,
    load_inspection,
    schema,
    task_row,
)


def add_trace_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    trace = subparsers.add_parser(
        "trace",
        help="Inspect Phase 4 results and traces with compact, read-only drill-downs.",
    )
    trace.set_defaults(func=_dispatch_trace)
    trace_sub = trace.add_subparsers(dest="trace_command", required=True)

    summary = trace_sub.add_parser("summary", help="Compact aggregate counts.")
    _add_path(summary)
    _add_common(summary)
    _add_filters(summary)
    summary.add_argument("--limit", type=int, default=8, help="Sample rows to include.")

    slice_cmd = trace_sub.add_parser("slice", help="Compact matching task table.")
    _add_path(slice_cmd)
    _add_common(slice_cmd, allow_jsonl=True)
    _add_filters(slice_cmd)
    slice_cmd.add_argument("--limit", type=int, default=20, help="Maximum rows.")
    slice_cmd.add_argument("--all", action="store_true", help="Return all matching rows.")
    slice_cmd.add_argument(
        "--fields",
        default=",".join(DEFAULT_FIELDS),
        help="Comma-separated fields. Use `trace fields` to list valid fields.",
    )
    slice_cmd.add_argument("--sort", choices=ALL_FIELDS, default=None, help="Sort rows by field.")
    slice_cmd.add_argument("--reverse", action="store_true", help="Reverse sort order.")

    task = trace_sub.add_parser("task", help="Explain one task.")
    _add_path(task)
    _add_common(task)
    task.add_argument("task_id", help="Task id to inspect.")
    task.add_argument("--iterator", action="store_true", help="Include compact iterator attempts.")
    task.add_argument("--refs", action="store_true", help="Include artifact manifest and paths.")

    timeline = trace_sub.add_parser("timeline", help="Show compact derived task event timeline.")
    _add_path(timeline)
    _add_common(timeline)
    timeline.add_argument("task_id", help="Task id to inspect.")

    fields = trace_sub.add_parser("fields", help="List selectable fields.")
    _add_common(fields)

    examples = trace_sub.add_parser("examples", help="Print example trace commands.")
    _add_common(examples)

    schema_cmd = trace_sub.add_parser("schema", help="Print machine-readable command schema.")
    _add_common(schema_cmd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect Phase 4 traces compactly.")
    subparsers = parser.add_subparsers(dest="trace_command", required=True)
    # Reuse the same subcommands without the top-level `trace` wrapper for the
    # compatibility script.
    shim = argparse.ArgumentParser(add_help=False)
    del shim
    add_trace_parser_for_script(subparsers)
    return _dispatch_trace(parser.parse_args(argv))


def add_trace_parser_for_script(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    # Kept separate from add_trace_parser because the script command omits the
    # top-level `trace` word.
    for name, help_text in (
        ("summary", "Compact aggregate counts."),
        ("slice", "Compact matching task table."),
        ("task", "Explain one task."),
        ("timeline", "Show compact derived task event timeline."),
        ("fields", "List selectable fields."),
        ("examples", "Print example trace commands."),
        ("schema", "Print machine-readable command schema."),
    ):
        cmd = subparsers.add_parser(name, help=help_text)
        cmd.set_defaults(trace_command=name)
        if name in {"summary", "slice", "task", "timeline"}:
            _add_path(cmd)
        if name == "summary":
            _add_common(cmd)
            _add_filters(cmd)
            cmd.add_argument("--limit", type=int, default=8)
        elif name == "slice":
            _add_common(cmd, allow_jsonl=True)
            _add_filters(cmd)
            cmd.add_argument("--limit", type=int, default=20)
            cmd.add_argument("--all", action="store_true")
            cmd.add_argument("--fields", default=",".join(DEFAULT_FIELDS))
            cmd.add_argument("--sort", choices=ALL_FIELDS, default=None)
            cmd.add_argument("--reverse", action="store_true")
        elif name == "task":
            _add_common(cmd)
            cmd.add_argument("task_id")
            cmd.add_argument("--iterator", action="store_true")
            cmd.add_argument("--refs", action="store_true")
        elif name == "timeline":
            _add_common(cmd)
            cmd.add_argument("task_id")
        else:
            _add_common(cmd)


def _add_path(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("path", type=Path, help="State dir, phase_4 dir, or results.json path.")
    parser.add_argument(
        "--tasks",
        type=Path,
        nargs="*",
        default=[],
        help="Optional adversarial_tasks.json path(s). Defaults to sibling phase_2.",
    )


def _add_common(parser: argparse.ArgumentParser, *, allow_jsonl: bool = False) -> None:
    outputs = ("text", "json", "jsonl") if allow_jsonl else ("text", "json")
    parser.add_argument("--output", choices=outputs, default="text", help="Output format.")
    parser.add_argument("--json", action="store_true", help="Shortcut for --output json.")


def _add_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--site")
    parser.add_argument("--action")
    parser.add_argument("--status")
    parser.add_argument("--outcome")
    parser.add_argument("--task-id")
    parser.add_argument("--surface")
    parser.add_argument("--origin")
    parser.add_argument("--route")
    parser.add_argument("--pvpo", choices=("encountered", "not_encountered", "unknown"))
    parser.add_argument("--coverage-min", type=float)
    parser.add_argument("--tp", choices=("aware", "unaware"))
    parser.add_argument("--vea", choices=("aware", "unaware"))
    parser.add_argument("--awareness", choices=("any", "both", "none"))
    parser.add_argument("--iterator-stop")
    parser.add_argument("--iterator-algorithm")
    parser.add_argument("--has-iterator", type=_bool_arg)
    parser.add_argument("--benign-passed", type=_bool_arg)
    parser.add_argument("--attack-attempted", type=_bool_arg)
    parser.add_argument("--attack-success", type=_bool_arg)
    parser.add_argument("--state-success", type=_bool_arg)
    parser.add_argument("--has-trace", type=_bool_arg)
    parser.add_argument("--missing-artifact", choices=("history", "result", "pvpo_summary"))
    parser.add_argument("--reward-contains")


def _dispatch_trace(args: argparse.Namespace) -> int:
    output = "json" if getattr(args, "json", False) else getattr(args, "output", "text")
    command = args.trace_command
    if command == "schema":
        return _emit(schema(), output=output, command="schema")
    if command == "fields":
        return _emit({"schema_version": "phase4_trace_fields_v1", "fields": ALL_FIELDS}, output=output, command="schema")
    if command == "examples":
        return _emit(
            {"schema_version": "phase4_trace_examples_v1", "examples": schema()["examples"]},
            output=output,
            command="schema",
        )

    try:
        results_path, phase4_dir, results, task_lookup = load_inspection(
            args.path,
            task_paths=getattr(args, "tasks", []),
        )
    except Exception as exc:
        print(f"trace failed: {exc}", file=sys.stderr)
        return 2

    if command in {"task", "timeline"}:
        filtered = filter_results(results, task_lookup, task_id=args.task_id)
        if not filtered:
            print(f"task not found: {args.task_id}", file=sys.stderr)
            return 1
        if command == "timeline":
            return _emit(
                build_timeline(filtered[0], phase4_dir=phase4_dir),
                output=output,
                command="timeline",
            )
        payload = build_task_detail(
            filtered[0],
            task_lookup,
            phase4_dir=phase4_dir,
            include_iterator=args.iterator,
            include_refs=args.refs,
        )
        payload["results_path"] = str(results_path)
        return _emit(payload, output=output, command="task")

    filtered = filter_results(results, task_lookup, **_filter_kwargs(args))
    if command == "summary":
        return _emit(
            build_summary(
                results_path,
                phase4_dir,
                results,
                task_lookup,
                filtered=filtered,
                sample_limit=max(args.limit, 0),
            ),
            output=output,
            command="summary",
        )

    fields = _parse_fields(args.fields)
    row_limit = len(filtered) if args.all else max(args.limit, 0)
    rows = [
        task_row(result, task_lookup, phase4_dir=phase4_dir, fields=fields)
        for result in filtered[:row_limit]
    ]
    if args.sort:
        rows.sort(key=lambda row: str(row.get(args.sort)), reverse=args.reverse)
    payload = {
        "schema_version": "phase4_trace_inspection_slice_v1",
        "results_path": str(results_path),
        "matched_results": len(filtered),
        "returned_rows": len(rows),
        "redaction_mode": "compact",
        "rows": rows,
    }
    return _emit(payload, output=output, command="slice")


def _filter_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    keys = (
        "site",
        "action",
        "status",
        "outcome",
        "task_id",
        "surface",
        "origin",
        "route",
        "pvpo",
        "coverage_min",
        "tp",
        "vea",
        "awareness",
        "iterator_stop",
        "iterator_algorithm",
        "has_iterator",
        "benign_passed",
        "attack_attempted",
        "attack_success",
        "state_success",
        "has_trace",
        "missing_artifact",
        "reward_contains",
    )
    return {key: getattr(args, key) for key in keys if hasattr(args, key)}


def _parse_fields(raw: str) -> list[str]:
    fields = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(fields) - set(ALL_FIELDS))
    if unknown:
        raise SystemExit(
            f"unknown field(s): {', '.join(unknown)}; valid fields: {', '.join(ALL_FIELDS)}"
        )
    return fields


def _emit(payload: dict[str, Any], *, output: str, command: str) -> int:
    if output == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif output == "jsonl":
        for row in payload.get("rows") or []:
            print(json.dumps(row, sort_keys=True))
    else:
        print(format_text(payload, command=command))
    return 0


def _bool_arg(raw: str) -> bool:
    value = raw.lower()
    if value in {"true", "yes", "1"}:
        return True
    if value in {"false", "no", "0"}:
        return False
    raise argparse.ArgumentTypeError("expected true/false")


__all__ = ["add_trace_parser", "main"]
