"""Arguments for the WARP Taskgen ``resume``, ``derive-and-resume``, and ``pause`` commands."""

from __future__ import annotations

import argparse
from pathlib import Path

from warp_taskgen.adversarial_actions import ACTION_POLICIES
from warp_taskgen.cli.argument_defaults import AGENT_PROVIDER_CHOICES
from warp_taskgen.cli.argument_types import (
    _non_negative_float,
    _non_negative_int,
    _positive_int,
)
from warp_taskgen.phase_4.options import (
    phase_4_variant_budget_choices,
    phase_4_variant_system_choices,
)
from warp_taskgen.phases.phase_1_task_cards import task_capability_profile_choices
from warp_taskgen.runners import available_runners
from warp_taskgen.runtime_composition import RUNTIME_COMPOSITION_CHOICES


def add_resume_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``resume``, ``derive-and-resume``, and ``pause`` commands."""
    resume_cmd = subparsers.add_parser(
        "resume",
        help="Resume from the last saved checkpoint",
        description=(
            "Resume from the last saved checkpoint. When resuming Phase 2, WARP Taskgen "
            "re-enters the saved internal sub-stage automatically: 2a planning or "
            "2b text fill. It also re-enters 2c feasibility when that is the saved "
            "sub-stage. There are no separate "
            "--phase-2a-only or --phase-2b-only flags."
        ),
    )
    resume_cmd.add_argument(
        "--benchmark",
        type=Path,
        default=argparse.SUPPRESS,
        help="Override the benchmark path saved in pipeline state.",
    )
    resume_cmd.add_argument(
        "--config",
        type=Path,
        default=argparse.SUPPRESS,
        help="Override the manifest path saved in pipeline state.",
    )
    resume_cmd.add_argument(
        "--instances",
        type=Path,
        default=argparse.SUPPRESS,
        help="Override the instances path saved in pipeline state.",
    )
    resume_cmd.add_argument(
        "--agent-model",
        default=argparse.SUPPRESS,
        help="Override the saved Browser Use agent model for the resumed phase.",
    )
    resume_cmd.add_argument(
        "--runner",
        default=argparse.SUPPRESS,
        choices=available_runners(),
        help="Override the saved browser-agent harness for the resumed phase.",
    )
    resume_cmd.add_argument(
        "--sandbox-model",
        default=argparse.SUPPRESS,
        help="Override the saved Claude sandbox model for the resumed phase.",
    )
    resume_cmd.add_argument(
        "--agent-provider",
        default=argparse.SUPPRESS,
        choices=AGENT_PROVIDER_CHOICES,
        help="Override the saved Browser Use agent provider for the resumed phase.",
    )
    resume_cmd.add_argument(
        "--agent-service-tier",
        default=argparse.SUPPRESS,
        choices=("auto", "default", "flex", "priority"),
        help="Override the saved OpenAI service tier for the resumed phase.",
    )
    resume_cmd.add_argument(
        "--agent-llm-timeout",
        type=_positive_int,
        default=argparse.SUPPRESS,
        metavar="SECONDS",
        help="Override the saved Phase 4 Browser Use per-step LLM-call timeout.",
    )
    resume_cmd.add_argument(
        "--agent-step-timeout",
        type=_positive_int,
        default=argparse.SUPPRESS,
        metavar="SECONDS",
        help="Override the saved Phase 4 Browser Use action-step timeout.",
    )
    resume_cmd.add_argument(
        "--agent-task-timeout",
        type=_positive_int,
        default=argparse.SUPPRESS,
        metavar="SECONDS",
        help="Override the saved Phase 4 Browser Use task wall-clock timeout.",
    )
    resume_cmd.add_argument(
        "--phase-4-max-workers",
        type=_positive_int,
        default=argparse.SUPPRESS,
        metavar="N",
        help="Override the saved Phase 4 Browser Use worker concurrency cap.",
    )
    resume_cmd.add_argument(
        "--phase-4-variant-system",
        choices=phase_4_variant_system_choices(),
        default=argparse.SUPPRESS,
        help="Override the saved Phase 4 variant/iterator system.",
    )
    resume_cmd.add_argument(
        "--phase-4-eval-awareness-max-iterations",
        type=_non_negative_int,
        default=argparse.SUPPRESS,
        metavar="N",
        help="Override the saved Phase 4 eval-awareness iterator rewrite budget.",
    )
    resume_cmd.add_argument(
        "--phase-4-variant-budget",
        choices=phase_4_variant_budget_choices(),
        default=argparse.SUPPRESS,
        help="Override the saved legacy Phase 4 strategy-variation budget.",
    )
    resume_cmd.add_argument(
        "--skip-intermediate-asr",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Override the saved Phase 4 run to skip post-hoc intermediate ASR.",
    )
    resume_cmd.add_argument(
        "--intermediate-asr-max-steps-per-task",
        type=_positive_int,
        default=argparse.SUPPRESS,
        metavar="N",
        help="Override the saved Phase 4 intermediate-ASR step cap.",
    )
    resume_cmd.add_argument(
        "--generate-novel",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Override saved Phase 1 state to enable novel-task generation.",
    )
    resume_cmd.add_argument(
        "--novel-tasks-per-site",
        "--new-tasks-per-site",
        dest="novel_tasks_per_site",
        type=_positive_int,
        default=argparse.SUPPRESS,
        metavar="N",
        help="Override saved Phase 1 novel-task count per eligible site.",
    )
    resume_cmd.add_argument(
        "--task-card-plan",
        type=Path,
        default=argparse.SUPPRESS,
        help="Override saved Phase 1 task-card plan path.",
    )
    resume_cmd.add_argument(
        "--task-capability-profile",
        choices=task_capability_profile_choices(),
        default=argparse.SUPPRESS,
        help="Override saved Phase 1 compiled action-capability task-card profile.",
    )
    resume_cmd.add_argument(
        "--runtime-composition",
        choices=RUNTIME_COMPOSITION_CHOICES,
        default=argparse.SUPPRESS,
        help=(
            "Override the saved runtime Site composition for the resumed phase; "
            "'default' selects the default GitLab/Reddit composition."
        ),
    )
    resume_cmd.add_argument(
        "--phase-1-action-counts",
        default=argparse.SUPPRESS,
        metavar="KIND=N[,KIND=N...]",
        help="Override saved Phase 1 contract-bound action-kind counts.",
    )
    resume_cmd.add_argument(
        "--max-tasks-per-site",
        type=_positive_int,
        default=argparse.SUPPRESS,
        metavar="N",
        help="Override per-site task cap for the resumed phase. Omit to run all remaining tasks.",
    )
    resume_cmd.add_argument(
        "--task-origin",
        choices=("all", "existing_task", "new_task"),
        default=argparse.SUPPRESS,
        help="Resume: override Phase 2/4 task-origin filtering.",
    )
    resume_cmd.add_argument(
        "--sites",
        type=str,
        default=argparse.SUPPRESS,
        metavar="SITE[,SITE...]",
        help="Override the saved site filter on resume.",
    )
    resume_cmd.add_argument(
        "--phase-2b-texts-per-plan",
        type=_positive_int,
        default=argparse.SUPPRESS,
        metavar="N",
        help="Override the saved Phase 2b texts-per-plan on resume.",
    )
    resume_cmd.add_argument(
        "--phase-2-text-fill-concurrency",
        type=_positive_int,
        default=argparse.SUPPRESS,
        metavar="N",
        help="Override the saved Phase 2b text-fill concurrency on resume.",
    )
    resume_cmd.add_argument(
        "--phase-2-text-model",
        type=str,
        default=argparse.SUPPRESS,
        metavar="MODEL",
        help="Override the saved Phase 2b text-fill model on resume.",
    )
    resume_cmd.add_argument(
        "--phase-2a-action-policy",
        choices=ACTION_POLICIES,
        default=argparse.SUPPRESS,
        help="Override the saved Phase 2a adversarial-action policy on resume.",
    )
    resume_cmd.add_argument(
        "--allow-unknown-auth",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Override the saved gate for auth_mechanism.type='unknown' during resume.",
    )
    resume_cmd.add_argument(
        "--skip-host-bound-storage-state-auth",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Override the saved behavior for host-bound storage_state artifacts during resume.",
    )
    resume_cmd.add_argument(
        "--skip-feasibility",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Resume: force Phase 2 to skip 2c live verification on this run.",
    )
    resume_cmd.add_argument(
        "--feasibility-only",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Resume: re-run only Phase 2c (skip 2a planning + 2b text fill).",
    )
    resume_cmd.add_argument(
        "--feasibility-instances",
        type=str,
        default=argparse.SUPPRESS,
        metavar="PATH",
        help="Resume: override the Phase 2c instances file.",
    )
    resume_cmd.add_argument(
        "--feasibility-concurrency",
        type=_positive_int,
        default=argparse.SUPPRESS,
        metavar="N",
        help="Resume: override Phase 2c worker concurrency.",
    )
    resume_cmd.add_argument(
        "--feasibility-retry-count",
        type=_non_negative_int,
        default=argparse.SUPPRESS,
        metavar="N",
        help="Resume: override Phase 2c retry budget.",
    )
    resume_cmd.add_argument(
        "--feasibility-ttl-hours",
        type=float,
        default=argparse.SUPPRESS,
        metavar="HOURS",
        help="Resume: override Phase 2c TTL shortcut.",
    )
    resume_cmd.add_argument(
        "--force-reverify",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Resume: re-verify every Phase 2c task regardless of fingerprint.",
    )
    resume_cmd.add_argument(
        "--no-l3-l4",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Resume: force the saved Phase 2 run down the offline L1/L2-only "
        "resolver path instead of live L3/L4 enrichment.",
    )
    resume_cmd.add_argument(
        "--plan",
        action="store_true",
        help="Explain the read-only Resume Plan without dispatching a phase.",
    )
    resume_cmd.add_argument(
        "--json",
        action="store_true",
        help="With --plan, print structured JSON instead of operator prose.",
    )
    # Keep derivation as a separate, unmistakably state-changing operator
    # action. The parent parser reuses the exact resume override surface
    # without adding a second implementation of those options.
    subparsers.add_parser(
        "derive-and-resume",
        parents=[resume_cmd],
        add_help=False,
        help="Materialize and execute an isolated Derived Run for explicit drift.",
        description=(
            "Materialize or recover one isolated Derived Run for explicit "
            "result-affecting overrides, then resume that child only."
        ),
    )
    pause_cmd = subparsers.add_parser(
        "pause",
        help="Request a cooperative pause at the next supported Phase 2 or Phase 4 boundary.",
    )
    pause_cmd.add_argument(
        "--state-dir",
        type=Path,
        default=None,
        help="Run state directory. Defaults to the configured WARP state root.",
    )
    pause_cmd.add_argument(
        "--wait",
        action="store_true",
        help="Wait for authoritative paused, terminal, rejected, or timeout readback.",
    )
    pause_cmd.add_argument(
        "--timeout",
        type=_non_negative_float,
        default=300.0,
        metavar="SECONDS",
        help="Maximum bounded wait duration (default: 300 seconds).",
    )
    pause_cmd.add_argument(
        "--poll-interval",
        type=_non_negative_float,
        default=0.25,
        metavar="SECONDS",
        help="Readback polling interval while --wait is active (default: 0.25).",
    )
