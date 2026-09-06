"""Arguments for the WARP Taskgen ``agentlab`` command and its subcommands."""

from __future__ import annotations

import argparse
from pathlib import Path

from warp_taskgen.cli.argument_defaults import AGENT_PROVIDER_CHOICES
from warp_taskgen.cli.argument_types import _positive_int


def add_agentlab_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``agentlab`` command with its ``models`` and ``run`` subcommands."""
    agentlab_cmd = subparsers.add_parser(
        "agentlab",
        help="Run AgentLab/BrowserGym comparison tasks.",
    )
    agentlab_sub = agentlab_cmd.add_subparsers(dest="agentlab_command", required=True)
    agentlab_models = agentlab_sub.add_parser(
        "models",
        help="List named WARP Taskgen AgentLab model profiles.",
    )
    agentlab_models.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    agentlab_run = agentlab_sub.add_parser(
        "run",
        help="Run one AgentLab/BrowserGym comparison task through the isolated sidecar.",
    )
    agentlab_run.add_argument(
        "--task-json",
        type=Path,
        default=None,
        help="WARP Taskgen task JSON object. Lists are accepted only when they contain one task.",
    )
    agentlab_run.add_argument(
        "--browsergym-task-name",
        default=None,
        help="BrowserGym task name to run when --task-json is omitted or should be overridden.",
    )
    agentlab_run.add_argument(
        "--task-id",
        default=None,
        help="Optional task id for artifact paths and result.json.",
    )
    agentlab_run.add_argument(
        "--benchmark-name",
        default=None,
        help="Benchmark metadata for synthetic --browsergym-task-name tasks.",
    )
    agentlab_run.add_argument(
        "--instances",
        type=Path,
        required=True,
        help="Instances config supplying site URLs, reset endpoints, and auth metadata.",
    )
    agentlab_run.add_argument(
        "--site",
        default=None,
        help="Site/instance to bind. Required when the task and instances are ambiguous.",
    )
    agentlab_run.add_argument(
        "--replica-name",
        default=None,
        help="Optional same-site replica_name selector.",
    )
    agentlab_run.add_argument(
        "--replica-index",
        type=int,
        default=None,
        help="Optional same-site replica_index selector.",
    )
    agentlab_run.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Task artifact directory. Defaults under logs/agentlab_comparison/.",
    )
    agentlab_run.add_argument(
        "--agent-model",
        default="gpt52",
        help="Agent model/profile passed to AgentLab. Default: gpt52.",
    )
    agentlab_run.add_argument(
        "--agent-provider",
        choices=AGENT_PROVIDER_CHOICES,
        default="openrouter",
        help="Provider route for bare model names/profiles. Default: openrouter.",
    )
    agentlab_run.add_argument(
        "--agent-service-tier",
        default=None,
        help="Optional OpenAI service tier for OpenAI/OpenRouter routes.",
    )
    agentlab_run.add_argument(
        "--max-steps",
        type=_positive_int,
        default=30,
        help="AgentLab/BrowserGym max_steps.",
    )
    agentlab_run.add_argument(
        "--attack-mode",
        choices=("comparison", "seeded_comparison"),
        default="comparison",
        help=(
            "comparison runs BrowserGym-native tasks; seeded_comparison also applies "
            "the task data_seed before the AgentLab run."
        ),
    )
    agentlab_run.add_argument(
        "--benchmark-prefix",
        default="webarena_verified",
        help="Fallback BrowserGym task-name prefix for WebArena Verified task ids.",
    )
    agentlab_run.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
