"""WARP Taskgen CLI parser."""

from __future__ import annotations

import argparse
from pathlib import Path

from warp_taskgen.cli.agentlab_arguments import add_agentlab_parser
from warp_taskgen.cli.argument_defaults import (
    AGENT_PROVIDER_CHOICES,
    DEFAULT_AGENT_MODEL,
    DEFAULT_SANDBOX_MODEL,
)
from warp_taskgen.cli.argument_types import (
    _non_negative_float,
    _non_negative_int,
    _positive_int,
)
from warp_taskgen.cli.phase_arguments import add_phase_parser
from warp_taskgen.cli.resume_arguments import add_resume_parser
from warp_taskgen.cli.task_bank_arguments import add_task_bank_parser


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser used by ``main()`` and tests."""
    parser = argparse.ArgumentParser(
        description="WARP Taskgen task-generation and admission pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    from warp_taskgen.cli.site_composition_check import add_site_composition_parser
    from warp_taskgen.phase_4.trace_inspection_cli import add_trace_parser

    add_site_composition_parser(subparsers)
    add_trace_parser(subparsers)

    add_phase_parser(subparsers)
    add_resume_parser(subparsers)
    rescore_cmd = subparsers.add_parser(
        "rescore-phase-3",
        help="Re-score an existing Phase 3 run with the agent-response transform.",
    )
    rescore_cmd.add_argument(
        "--phase-3-dir",
        type=Path,
        default=Path("logs/phase_3_gemini-3-flash"),
        help="Phase 3 output directory containing results.json and task trajectories.",
    )
    rescore_cmd.add_argument(
        "--instances",
        type=Path,
        default=None,
        help="Optional instances JSON, used to supply URL placeholders to the "
        "vendor evaluator's config-validation step.",
    )

    preflight_cmd = subparsers.add_parser(
        "preflight",
        help=(
            "Run the Phase 4 preflight gates (storage_state, PVPO endpoints, "
            "instance reachability) without spinning up Phase 4 itself. "
            "Cheap local sanity check — run after config or generator changes."
        ),
    )
    preflight_cmd.add_argument(
        "--host-config",
        type=Path,
        default=None,
        help="Benchmark host YAML, e.g. configs/benchmark_hosts/r8a.yaml.",
    )
    preflight_cmd.add_argument(
        "--instances",
        type=Path,
        default=Path("instances.scale.json"),
        help="Instances config to preflight (default: instances.scale.json).",
    )
    preflight_cmd.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to pytest (e.g. -k, -x, -v).",
    )

    status_cmd = subparsers.add_parser(
        "status",
        help="Show a read-only operator summary for a WARP Taskgen run.",
    )
    status_cmd.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=None,
        help=(
            "Run state dir, phase_4/results.json, or pipeline_state.json. "
            "Defaults to WARP_TASKGEN_STATE_DIR/logs, with WORLDSIM_STATE_DIR "
            "accepted as a legacy alias."
        ),
    )
    status_cmd.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    status_cmd.add_argument(
        "--inspect-limit",
        type=_non_negative_int,
        default=5,
        help="Number of ranked per-task inspection rows to print in text mode.",
    )

    inspect_cmd = subparsers.add_parser(
        "inspect",
        help="Inspect one Phase 4 task with trace and artifact pointers.",
    )
    inspect_cmd.add_argument("task_id", help="Phase 4 task id to inspect.")
    inspect_cmd.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=None,
        help=(
            "Run state dir, phase_4/results.json, or pipeline_state.json. "
            "Defaults to WARP_TASKGEN_STATE_DIR/logs, with WORLDSIM_STATE_DIR "
            "accepted as a legacy alias."
        ),
    )
    inspect_cmd.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    add_agentlab_parser(subparsers)
    add_task_bank_parser(subparsers)

    return parser


def _parse_cli_sites(raw_sites: str | None) -> set[str] | None:
    if raw_sites is None:
        return None
    parsed = {site.strip() for site in raw_sites.split(",") if site.strip()}
    return parsed or None


__all__ = [
    "AGENT_PROVIDER_CHOICES",
    "DEFAULT_AGENT_MODEL",
    "DEFAULT_SANDBOX_MODEL",
    "_non_negative_float",
    "_non_negative_int",
    "_parse_cli_sites",
    "_positive_int",
    "build_parser",
]
