#!/usr/bin/env python3
"""Export a human-readable Phase 4 variant trace table.

This is report-only. It reads existing Phase 4 results and variant-generation
artifacts and writes JSON, CSV, and/or HTML projections.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from worldsim.phase_4.variant_trace_export import build_variant_trace_export
from worldsim.phase_4.variant_trace_outputs import (
    write_variant_trace_csv,
    write_variant_trace_html,
    write_variant_trace_json,
)


def _formats(raw: str) -> set[str]:
    values = {item.strip().lower() for item in raw.split(",") if item.strip()}
    allowed = {"json", "csv", "html"}
    unknown = values - allowed
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown format(s): {', '.join(sorted(unknown))}; allowed: json,csv,html"
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for variant_trace_table.{json,csv,html}.",
    )
    parser.add_argument(
        "--formats",
        type=_formats,
        default={"json"},
        help="Comma-separated output formats: json,csv,html. Default: json.",
    )
    parser.add_argument(
        "--include",
        choices=("all", "variant-entered", "success-on-variant"),
        default="all",
        help="Which task rows to include. Default: all.",
    )
    parser.add_argument(
        "--payload-limit",
        type=int,
        default=None,
        help="Optionally compact payload text in JSON/CSV/HTML to this many characters.",
    )
    parser.add_argument(
        "--html-title",
        default=None,
        help="Optional title for the HTML report.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if the export contains row-level warnings.",
    )
    args = parser.parse_args(argv)

    export = build_variant_trace_export(
        args.path,
        task_paths=args.tasks,
        include=args.include,
        payload_limit=args.payload_limit,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if "json" in args.formats:
        write_variant_trace_json(export, args.output_dir / "variant_trace_table.json")
        print(args.output_dir / "variant_trace_table.json")
    if "csv" in args.formats:
        write_variant_trace_csv(export, args.output_dir / "variant_trace_table.csv")
        print(args.output_dir / "variant_trace_table.csv")
    if "html" in args.formats:
        write_variant_trace_html(
            export,
            args.output_dir / "variant_trace_table.html",
            title=args.html_title,
        )
        print(args.output_dir / "variant_trace_table.html")
    if args.strict and export.get("warning_count"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
