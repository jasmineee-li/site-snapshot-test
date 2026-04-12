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
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # Load .env before importing stateful modules.


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. See module docstring for usage."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

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
        help="Path to the benchmark codebase (e.g. vendors/webarena-verified). "
        "Required for Phase 0.",
    )
    phase_cmd.add_argument(
        "--config",
        type=Path,
        help="Path to a BENCHMARK_MANIFEST.json from Phase 0a. "
        "Required for Phases 0b+.",
    )
    phase_cmd.add_argument(
        "--instances",
        type=Path,
        help="JSON file with BenchmarkConfig (site_url, db_connection, "
        "reset_endpoint). Required for Phases 3-4.",
    )
    phase_cmd.add_argument(
        "--agent-model",
        default="gemini-3.1-pro-preview",
        help="LLM model name for Browser Use agent (default: gemini-3.1-pro-preview). "
        "Examples: gpt-5.4, claude-sonnet-4-6, gemini-3.1-pro-preview, gemini-3-flash-preview.",
    )
    phase_cmd.add_argument(
        "--agent-provider",
        default=None,
        choices=["google", "openai", "anthropic"],
        help="LLM provider (default: auto-detect from model name). "
        "Requires the corresponding env var: GOOGLE_API_KEY, OPENAI_API_KEY, "
        "or ANTHROPIC_API_KEY.",
    )

    resume_cmd = subparsers.add_parser("resume", help="Resume from the last saved checkpoint")
    resume_cmd.add_argument(
        "--benchmark",
        type=Path,
        help="Path to the benchmark codebase (overrides path from state).",
    )
    resume_cmd.add_argument(
        "--config",
        type=Path,
        help="Path to a BENCHMARK_MANIFEST.json (overrides path from state).",
    )
    resume_cmd.add_argument(
        "--instances",
        type=Path,
        help="JSON file with BenchmarkConfig (overrides path from state).",
    )
    resume_cmd.add_argument(
        "--agent-model",
        default="gemini-3.1-pro-preview",
        help="LLM model name for Browser Use agent (default: gemini-3.1-pro-preview).",
    )
    resume_cmd.add_argument(
        "--agent-provider",
        default=None,
        choices=["google", "openai", "anthropic"],
        help="LLM provider (default: auto-detect from model name).",
    )

    args = parser.parse_args(argv)

    if args.command == "resume":
        return _dispatch_resume(args)

    if args.command == "phase":
        return _dispatch_phase(args)

    return 0


# Ordered pipeline steps. Each entry maps step name -> (phase_id, sub) where
# sub is the sub-step for Phase 0, or None for later phases.
_PHASE_ORDER: list[str] = [
    "phase_0a", "phase_0b", "phase_0c",
    "phase_1", "phase_2", "phase_3", "phase_4",
]


def _next_step(step: str) -> str | None:
    """Return the step after ``step``, or None if ``step`` is the last."""
    try:
        idx = _PHASE_ORDER.index(step)
    except ValueError:
        return None
    if idx + 1 < len(_PHASE_ORDER):
        return _PHASE_ORDER[idx + 1]
    return None


def _dispatch_resume(args: argparse.Namespace) -> int:
    """Read last checkpoint and dispatch to the appropriate phase."""
    from worldsim.state import load_state

    state = load_state()
    if state is None:
        print("No pipeline state found; run a phase first.", file=sys.stderr)
        return 1

    last_step = state.get("step", "")
    status = state.get("status", "")
    logs_dir = state.get("logs_dir")

    if logs_dir and not os.environ.get("WORLDSIM_STATE_DIR"):
        os.environ["WORLDSIM_STATE_DIR"] = str(logs_dir)

    if status == "complete":
        target = _next_step(last_step)
        if target is None:
            print(f"Last checkpoint: {last_step} complete. Pipeline finished — nothing to resume.")
            return 0
        print(f"Last checkpoint: {last_step} complete. Resuming from {target}.")
    elif status == "running":
        target = last_step
        print(f"Last checkpoint: {last_step} was running (likely crashed). Re-running {target}.")
    else:
        print(f"Last checkpoint: {last_step} has unknown status {status!r}.", file=sys.stderr)
        return 1

    # Build a synthetic argparse.Namespace that _dispatch_phase understands.
    # CLI flags override state metadata; state metadata fills gaps.
    benchmark = getattr(args, "benchmark", None)
    config = getattr(args, "config", None)
    instances = getattr(args, "instances", None)

    # Fall back to paths stored in state metadata
    if benchmark is None and "benchmark_path" in state:
        benchmark = Path(state["benchmark_path"])
    if config is None and "manifest_path" in state:
        config = Path(state["manifest_path"])
    if instances is None and "instances_path" in state:
        instances = Path(state["instances_path"])

    # Map target step to phase ID for _dispatch_phase (e.g. "phase_0a" -> "0a")
    phase_id = target.replace("phase_", "")

    synthetic = argparse.Namespace(
        command="phase",
        phase=phase_id,
        benchmark=benchmark,
        config=config,
        instances=instances,
        agent_model=getattr(args, "agent_model", "gemini-3.1-pro-preview"),
        agent_provider=getattr(args, "agent_provider", None),
    )

    return _dispatch_phase(synthetic)


def _dispatch_phase(args: argparse.Namespace) -> int:
    """Dispatch to the requested phase module."""
    from worldsim.cost_tracker import tracker as cost_tracker
    from worldsim.phases import (
        phase_0_recon,
        phase_1_tasks,
        phase_2_injections,
        phase_3_benign,
        phase_4_adversarial,
    )
    from worldsim.state import get_state_dir

    # Load any previously saved cost data so cross-phase totals accumulate.
    cost_report_path = get_state_dir() / "cost_report.json"
    cost_tracker.load(cost_report_path)

    phase = args.phase
    if phase in {"0", "0a", "0b", "0c"}:
        if not args.benchmark:
            print("--benchmark is required for Phase 0", file=sys.stderr)
            return 1
        rc = asyncio.run(phase_0_recon.run(benchmark=args.benchmark, sub=phase))
    elif phase == "1":
        rc = asyncio.run(phase_1_tasks.run(args))
    elif phase == "2":
        rc = asyncio.run(phase_2_injections.run(args))
    elif phase == "3":
        rc = asyncio.run(phase_3_benign.run(args))
    elif phase == "4":
        rc = asyncio.run(phase_4_adversarial.run(args))
    else:
        print(f"Unknown phase: {phase}", file=sys.stderr)
        return 1

    # Log final pipeline cost summary if any sandbox calls were recorded.
    if cost_tracker.entries:
        logger = logging.getLogger(__name__)
        logger.info("--- Cost Summary ---\n%s", cost_tracker.summary_report())

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
