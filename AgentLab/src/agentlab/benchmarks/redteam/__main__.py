"""Unified CLI for the redteam benchmark package.

Usage::

    uv run python -m agentlab.benchmarks.redteam generate --benchmark-file ...
    uv run python -m agentlab.benchmarks.redteam validate --platform-manifest ...
    uv run python -m agentlab.benchmarks.redteam status --output-dir apps/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _run_generate(args: argparse.Namespace) -> None:
    from agentlab.benchmarks.redteam.generate_apps import main as gen_main

    gen_main(args._remaining)


def _run_validate(args: argparse.Namespace) -> None:
    from agentlab.benchmarks.redteam.validate_platform_manifest import main as val_main

    val_main(args._remaining)


def _run_status(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Output directory not found: {output_dir}")
        sys.exit(1)

    app_dirs = sorted(
        d for d in output_dir.iterdir()
        if d.is_dir() and (d / "app_manifest.json").exists()
    )

    if not app_dirs:
        print(f"No generated apps found in {output_dir}")
        return

    ok_count = 0
    fail_count = 0
    progress_count = 0

    for app_dir in app_dirs:
        manifest = json.loads((app_dir / "app_manifest.json").read_text(encoding="utf-8"))
        gen = manifest.get("generation", {})
        status = gen.get("status", "unknown")
        phase = gen.get("last_completed_phase", "")
        app_id = manifest.get("app_id", app_dir.name)
        errors = manifest.get("errors", [])

        if status == "succeeded":
            marker = " OK "
            ok_count += 1
        elif status == "failed":
            marker = "FAIL"
            fail_count += 1
        elif status == "in_progress":
            marker = " .. "
            progress_count += 1
        else:
            marker = " ?  "

        line = f"  [{marker}] {app_id}: {status}"
        if phase:
            line += f" (phase: {phase})"
        if errors:
            line += f" — {len(errors)} error(s)"
        print(line)

    print(f"\nTotal: {len(app_dirs)} apps — {ok_count} ok, {fail_count} failed, {progress_count} in progress")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="agentlab.benchmarks.redteam",
        description="Redteam benchmark CLI",
    )
    subparsers = parser.add_subparsers(dest="command")

    # generate subcommand
    gen_sub = subparsers.add_parser(
        "generate",
        help="Generate web apps from behavior specs",
        add_help=False,
    )
    gen_sub.set_defaults(func=_run_generate)

    # validate subcommand
    val_sub = subparsers.add_parser(
        "validate",
        help="Validate platform manifest",
        add_help=False,
    )
    val_sub.set_defaults(func=_run_validate)

    # status subcommand
    status_sub = subparsers.add_parser("status", help="Show generation status")
    status_sub.add_argument("--output-dir", default="apps/", help="App output directory")
    status_sub.set_defaults(func=_run_status)

    # Parse only the subcommand, pass remaining to delegates
    args, remaining = parser.parse_known_args()
    args._remaining = remaining

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
