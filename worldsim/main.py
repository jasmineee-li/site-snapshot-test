"""WorldSim v5 CLI entrypoint.

See ``docs/worldsim-v5-technical-specifcation.md`` for the full pipeline
and ``README.md`` for prerequisites.

Usage::

    # Phase 0 reconnaissance against a benchmark codebase on disk
    uv run python -m worldsim.main phase 0 --benchmark vendors/webarena-infinity

    # Resume from the last saved checkpoint
    uv run python -m worldsim.main resume
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

from worldsim.state import load_state


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. See module docstring for usage."""
    load_dotenv()  # .env fills gaps; shell exports take precedence

    parser = argparse.ArgumentParser(
        prog="worldsim",
        description="WorldSim v5 adversarial evaluation pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    phase_cmd = subparsers.add_parser("phase", help="Run a specific phase")
    phase_cmd.add_argument(
        "phase",
        choices=["0", "0a", "0b", "0c", "1", "2", "3", "4"],
        help="Phase to run",
    )
    phase_cmd.add_argument(
        "--benchmark",
        type=Path,
        help="Path to the benchmark codebase (e.g. vendors/webarena-infinity). "
        "Required for Phase 0.",
    )
    phase_cmd.add_argument(
        "--config",
        type=Path,
        help="Path to a BENCHMARK_MANIFEST.json from Phase 0a. "
        "Required for Phases 0b+.",
    )

    subparsers.add_parser("resume", help="Resume from the last saved checkpoint")

    args = parser.parse_args(argv)

    if args.command == "resume":
        state = load_state()
        if state is None:
            print("No pipeline state found; run a phase first.", file=sys.stderr)
            return 1
        print(f"Last checkpoint: {state}")
        print("resume is not yet implemented", file=sys.stderr)
        return 1

    if args.command == "phase":
        return _dispatch_phase(args)

    return 0


def _dispatch_phase(args: argparse.Namespace) -> int:
    """Dispatch to the requested phase module."""
    from worldsim.phases import (
        phase_0_recon,
        phase_1_tasks,
        phase_2_injections,
        phase_3_benign,
        phase_4_adversarial,
    )

    phase = args.phase
    if phase in {"0", "0a", "0b", "0c"}:
        if not args.benchmark:
            print("--benchmark is required for Phase 0", file=sys.stderr)
            return 1
        return asyncio.run(phase_0_recon.run(benchmark=args.benchmark, sub=phase))
    elif phase == "1":
        return asyncio.run(phase_1_tasks.run(args))
    elif phase == "2":
        return asyncio.run(phase_2_injections.run(args))
    elif phase == "3":
        return asyncio.run(phase_3_benign.run(args))
    elif phase == "4":
        return asyncio.run(phase_4_adversarial.run(args))
    else:
        print(f"Unknown phase: {phase}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
