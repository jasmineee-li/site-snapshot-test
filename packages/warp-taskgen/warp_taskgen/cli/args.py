"""WARP Taskgen CLI parser."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from warp_taskgen.adversarial_actions import ACTION_POLICIES
from warp_taskgen.agent_runtime import RUNNER_BROWSER_USE
from warp_taskgen.phase_4.options import (
    phase_4_variant_budget_choices,
    phase_4_variant_system_choices,
)
from warp_taskgen.phases.phase_1_task_cards import task_capability_profile_choices
from warp_taskgen.runners import available_runners
from warp_taskgen.runtime_composition import RUNTIME_COMPOSITION_CHOICES

DEFAULT_AGENT_MODEL = "claude-sonnet-4-6"
DEFAULT_SANDBOX_MODEL = "claude-sonnet-4-6"
AGENT_PROVIDER_CHOICES = ("google", "openai", "anthropic", "openrouter")


def _positive_int(value: str) -> int:
    """Argparse type for positive integer CLI flags."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    """Argparse type for 0-or-greater integer CLI flags."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _non_negative_float(value: str) -> float:
    """Argparse type for finite zero-or-greater durations."""

    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a finite number >= 0")
    return parsed


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

    phase_cmd = subparsers.add_parser(
        "phase",
        help="Run a specific phase",
        description=(
            "Run a specific pipeline phase. Phase 2 is one command with two "
            "internal model stages: 2a host-side API strategy planning, then "
            "2b host-side text fill. The command runs those stages sequentially; "
            "there are no separate --phase-2a-only or --phase-2b-only flags."
        ),
    )
    phase_cmd.add_argument(
        "phase",
        choices=["0", "0a", "0b", "0c", "0d", "1", "2", "2c", "3", "4"],
        help="Phase to run. '2c' is CLI sugar for 'phase 2 --feasibility-only'.",
    )
    phase_cmd.add_argument(
        "--benchmark",
        type=Path,
        help="Path to the benchmark codebase. Required for Phase 0. "
        "Phase 1 can infer it from the manifest when BENCHMARK_MANIFEST.json "
        "includes benchmark_codebase.",
    )
    phase_cmd.add_argument(
        "--config",
        type=Path,
        help="Path to BENCHMARK_MANIFEST.json from Phase 0a. Used by Phase 1 to "
        "override the default manifest path under logs/phase_0a/.",
    )
    phase_cmd.add_argument(
        "--generate-novel",
        action="store_true",
        help="For Phase 1, also generate novel tasks for sites with Phase 4-admissible carrier route families.",
    )
    phase_cmd.add_argument(
        "--novel-tasks-per-site",
        "--new-tasks-per-site",
        dest="novel_tasks_per_site",
        type=_positive_int,
        default=None,
        metavar="N",
        help="Phase 1 generate-new-tasks: generate N novel tasks per eligible site. Defaults to 30.",
    )
    phase_cmd.add_argument(
        "--task-card-plan",
        type=Path,
        default=None,
        help=(
            "Phase 1 generate-new-tasks: optional JSON task-card plan that constrains "
            "novel generation by behavior/archetype while route contracts remain authoritative."
        ),
    )
    phase_cmd.add_argument(
        "--task-capability-profile",
        choices=task_capability_profile_choices(),
        default=None,
        help=(
            "Phase 1 generate-new-tasks: compile a named host-owned action-capability "
            "task-card profile. Mutually exclusive with --task-card-plan."
        ),
    )
    phase_cmd.add_argument(
        "--runtime-composition",
        choices=RUNTIME_COMPOSITION_CHOICES,
        default=None,
        help=(
            "Explicit runtime Site composition for a bounded POC. Omit to preserve "
            "the default GitLab/Reddit runtime."
        ),
    )
    phase_cmd.add_argument(
        "--phase-1-action-counts",
        default=None,
        metavar="KIND=N[,KIND=N...]",
        help=(
            "Phase 1 contract-bound generation: explicit global action-kind counts, "
            "for example create_issue=20,create_post=20,create_issue_note=10. "
            "Omitted action kinds receive zero rows."
        ),
    )
    phase_cmd.add_argument(
        "--instances",
        type=Path,
        help="JSON file with BenchmarkConfig (site_url, db_connection, "
        "reset_endpoint). Required for Phase 4.",
    )
    phase_cmd.add_argument(
        "--runner",
        default=RUNNER_BROWSER_USE,
        choices=available_runners(),
        help=(
            "Browser-agent harness for execution phases. WebArena Verified "
            "Phase 4 supports Browser Use and the isolated AgentLab sidecar; "
            "non-WARP-Taskgen benchmarks use AgentLab only for comparison runs."
        ),
    )
    phase_cmd.add_argument(
        "--host-inventory-instances",
        type=Path,
        default=None,
        help=(
            "Phase 0/0c: optional host-local BenchmarkConfig used only for "
            "host-side inventory enrichment. On r5, use instances.smoke.json "
            "with --instances for Modal browser probes and instances.scale.json "
            "here for DB/API inventory reads."
        ),
    )
    phase_cmd.add_argument(
        "--agent-model",
        default=DEFAULT_AGENT_MODEL,
        help=f"LLM model name for Browser Use agent (default: {DEFAULT_AGENT_MODEL}). "
        "Examples: claude-sonnet-4-6, gpt-5.4, gpt-5.4-mini "
        "(with --agent-provider openrouter), gemini-3-flash-preview, "
        "gemini-3.1-pro-preview.",
    )
    phase_cmd.add_argument(
        "--sandbox-model",
        default=DEFAULT_SANDBOX_MODEL,
        help="Claude sandbox model for Phase 4 judge and variant steps "
        f"(default: {DEFAULT_SANDBOX_MODEL}).",
    )
    phase_cmd.add_argument(
        "--agent-provider",
        default=None,
        choices=AGENT_PROVIDER_CHOICES,
        help="LLM provider (default: auto-detect from model name). "
        "Requires the corresponding env var: GOOGLE_API_KEY, OPENAI_API_KEY, "
        "ANTHROPIC_API_KEY, or OPENROUTER_API_KEY.",
    )
    phase_cmd.add_argument(
        "--agent-service-tier",
        default=None,
        choices=("auto", "default", "flex", "priority"),
        help="OpenAI service tier for the Browser Use agent. "
        "'priority' = lowest + most consistent latency (costs more); "
        "'flex' = ~50%% cheaper but slower + variable; "
        "'default' / 'auto' = provider default. Applies to --agent-provider "
        "openai and openrouter (forwarded when pinned to OpenAI upstream); "
        "ignored with a warning for google/anthropic. Omit to use provider default.",
    )
    phase_cmd.add_argument(
        "--agent-llm-timeout",
        type=_positive_int,
        default=None,
        metavar="SECONDS",
        help=(
            "Phase 4: explicit Browser Use per-step LLM-call timeout. "
            "Omit to preserve Browser Use's provider/model default."
        ),
    )
    phase_cmd.add_argument(
        "--agent-step-timeout",
        type=_positive_int,
        default=None,
        metavar="SECONDS",
        help=(
            "Phase 4: explicit Browser Use action-step timeout. "
            "Omit to preserve Browser Use's default."
        ),
    )
    phase_cmd.add_argument(
        "--agent-task-timeout",
        type=_positive_int,
        default=None,
        metavar="SECONDS",
        help=(
            "Phase 4: explicit WARP Taskgen wall-clock timeout for one Browser Use "
            "task. Omit to preserve Browser Use's long-running default. This is "
            "an infrastructure guard for stuck sessions, not an action-step limit."
        ),
    )
    phase_cmd.add_argument(
        "--phase-4-max-workers",
        type=_positive_int,
        default=None,
        metavar="N",
        help=(
            "Phase 4: cap concurrent Browser Use worker sessions, including "
            "initial evaluation and variant/placement reruns. Omit to use the "
            "instance pool size or WORLDSIM_MAX_WORKERS fallback."
        ),
    )
    phase_cmd.add_argument(
        "--phase-4-task-id",
        type=str,
        default=None,
        metavar="ID[,ID...]",
        help=(
            "Phase 4: restrict admitted tasks to explicit task IDs. Intended for "
            "process-isolated worker orchestration; normal rigor runs should omit it."
        ),
    )
    phase_cmd.add_argument(
        "--phase-4-variant-system",
        choices=phase_4_variant_system_choices(),
        default=None,
        help=(
            "Phase 4 variant/iterator system. Defaults to eval-awareness-iterator, "
            "which records an envelope for every PVPO-valid baseline row and only "
            "rewrites when Transcript Purpose calls for it; use strategy-variation "
            "for the legacy post-resistance refusal-judge strategy fan-out, or none "
            "to stop after the baseline trajectory."
        ),
    )
    phase_cmd.add_argument(
        "--phase-4-eval-awareness-max-iterations",
        type=_non_negative_int,
        default=None,
        metavar="N",
        help=(
            "Phase 4 eval-awareness iterator: maximum sequential rewrite/rerun "
            "attempts after the baseline trajectory. Defaults to 3."
        ),
    )
    phase_cmd.add_argument(
        "--phase-4-variant-budget",
        choices=phase_4_variant_budget_choices(),
        default=None,
        help=(
            "Phase 4 post-resistance strategy budget. Use adaptive-3-3-1 for "
            "the full bounded adaptive ASR metric, or smoke-3-probe for fast "
            "diagnostic smokes that run only the first three variants."
        ),
    )
    phase_cmd.add_argument(
        "--skip-intermediate-asr",
        action="store_true",
        help="Phase 4: skip post-hoc structured intermediate ASR classification.",
    )
    phase_cmd.add_argument(
        "--intermediate-asr-max-steps-per-task",
        type=_positive_int,
        default=None,
        metavar="N",
        help=(
            "Phase 4 post-hoc intermediate ASR: classify at most N action steps "
            "per trajectory. Omit to classify all steps. Use 10 only for "
            "data-import default-cap compatibility."
        ),
    )
    phase_cmd.add_argument(
        "--max-tasks-per-site",
        type=_positive_int,
        default=None,
        metavar="N",
        help="Phase 4: cap tasks to at most N per site for smoke testing. "
        "Selection is deterministic (fixed seed). Omit for full runs. "
        "Use `resume --max-tasks-per-site N` to keep the cap, or omit it on "
        "resume to process all remaining tasks.",
    )
    phase_cmd.add_argument(
        "--task-origin",
        choices=("all", "existing_task", "new_task"),
        default=None,
        help="Phase 2/3/4: optionally restrict tasks by Phase 1 origin before "
        "per-site capping. Use new_task for novel-carrier smoke runs; default is all.",
    )
    phase_cmd.add_argument(
        "--adversarial-action-kind",
        type=str,
        default=None,
        metavar="KIND[,KIND...]",
        help=(
            "Phase 4: optionally restrict admitted tasks by adversarial action "
            "kind before per-site capping, for reproducible split-cell runs "
            "(for example create_issue or create_issue_note)."
        ),
    )
    phase_cmd.add_argument(
        "--sites",
        type=str,
        default=None,
        metavar="SITE[,SITE...]",
        help="Comma-separated site names to include (e.g. 'shopping_admin'). "
        "Phase 2 preserves unselected sites on merge; Phases 3-4 filter task selection "
        "to the listed sites only.",
    )
    phase_cmd.add_argument(
        "--phase-2b-texts-per-plan",
        type=_positive_int,
        default=None,
        metavar="N",
        help="Phase 2b text fill: generate N payload variants per validated 2a plan "
        "during the sequential `phase 2` run. Defaults to 1.",
    )
    phase_cmd.add_argument(
        "--phase-2-text-fill-concurrency",
        type=_positive_int,
        default=None,
        metavar="N",
        help="Phase 2b text fill: cap concurrent host-side text fill requests to at most N.",
    )
    phase_cmd.add_argument(
        "--phase-2-text-model",
        type=str,
        default=None,
        metavar="MODEL",
        help="Phase 2b text fill: model identifier for host-side text fill requests.",
    )
    phase_cmd.add_argument(
        "--phase-2a-action-policy",
        choices=ACTION_POLICIES,
        default=None,
        help=(
            "Phase 2a adversarial-action policy. Default preserves planner choice; "
            "semantic_only restricts compatible contracts to host-compiled binary "
            "answer hijack for minimum validation; "
            "mutation_when_available marks compatible exposure contracts with a "
            "host-owned mutation preference for a controlled pilot; "
            "mutation_only_when_available suppresses semantic/navigation options "
            "where a mutation action is host-ready; tier*_pilot policies restrict "
            "contracts to one impact tier for controlled action-objective pilots."
        ),
    )
    phase_cmd.add_argument(
        "--allow-unknown-auth",
        action="store_true",
        default=False,
        help="Phase 4: proceed even when a site's auth_mechanism.type is 'unknown'. "
        "Default behavior is to refuse unknown-auth tasks so humans review them first.",
    )
    phase_cmd.add_argument(
        "--skip-host-bound-storage-state-auth",
        action="store_true",
        default=False,
        help="Phase 4: when a storage_state artifact was minted for a different host "
        "(for example an old EC2 IP), skip agent auth for that site instead of failing. "
        "Default behavior is to fail fast and ask you to re-run Phase 0d.",
    )
    phase_cmd.add_argument(
        "--skip-feasibility",
        action="store_true",
        default=False,
        help="Phase 2c: skip live feasibility verification. Tasks are stamped "
        "feasibility.status='unverified'. Strict Phase 4 admission skips "
        "unverified tasks unless WORLDSIM_STRICT_FEASIBILITY=false is set as a "
        "development break-glass. Use only for fast dev iteration; shipping runs "
        "must not bypass 2c.",
    )
    phase_cmd.add_argument(
        "--feasibility-only",
        action="store_true",
        default=False,
        help="Phase 2c: re-verify an existing adversarial_tasks.json without "
        "re-running 2a planning or 2b text fill. Idempotent.",
    )
    phase_cmd.add_argument(
        "--feasibility-instances",
        type=str,
        default="instances.smoke.json",
        metavar="PATH",
        help="Phase 2c: per-site instances file (wrapper dict with "
        "'instances' key). Defaults to instances.smoke.json.",
    )
    phase_cmd.add_argument(
        "--feasibility-concurrency",
        type=_positive_int,
        default=10,
        metavar="N",
        help="Phase 2c: maximum parallel verification workers (default 10).",
    )
    phase_cmd.add_argument(
        "--feasibility-retry-count",
        type=_non_negative_int,
        default=1,
        metavar="N",
        help="Phase 2c: per-task retry budget for transient EditorError kinds "
        "(default 1). 4xx rejections are never retried.",
    )
    phase_cmd.add_argument(
        "--feasibility-ttl-hours",
        type=float,
        default=None,
        metavar="HOURS",
        help="Phase 2c: skip re-verify when verified_at is newer than N hours "
        "even if the fingerprint drifts. Opt-in dev convenience.",
    )
    phase_cmd.add_argument(
        "--force-reverify",
        action="store_true",
        default=False,
        help="Phase 2c: re-verify every task regardless of fingerprint or status.",
    )
    phase_cmd.add_argument(
        "--no-l3-l4",
        action="store_true",
        default=False,
        help="Phase 2a: force the L1/L2-only target-resolver path even when "
        "--feasibility-instances points at a live benchmark. Skips the "
        "async L3 intent-classifier + live-probe pass and the L4 listing "
        "expansion. Tasks that depend on L3/L4 enrichment will arrive at "
        "2a with stub anchors (and route to the contract-table skip "
        "notice for listing kinds). Opt-in for offline reproductions and "
        "snapshot tests tied to the pre-expansion dataset.",
    )
    # Phase 4 transcript-purpose classifier has no CLI knobs — Apollo's
    # single-path protocol; self-healing handles soft failures internally.

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
        help="Override the saved runtime Site composition for the resumed phase.",
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

    task_bank_cmd = subparsers.add_parser(
        "task-bank",
        help="Manage the append-only admitted-task bank.",
    )
    task_bank_cmd.add_argument(
        "--path",
        type=Path,
        default=None,
        help=(
            "Task-bank JSONL path. Defaults to "
            "WARP_TASKGEN_STATE_DIR/task_bank/events.jsonl, with "
            "WORLDSIM_STATE_DIR accepted as a legacy alias."
        ),
    )
    task_bank_sub = task_bank_cmd.add_subparsers(dest="task_bank_command", required=True)

    task_bank_append = task_bank_sub.add_parser(
        "append",
        help="Append admitted tasks from a verified Phase 2c run.",
    )
    task_bank_append.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="WARP Taskgen run dir containing phase_2/adversarial_tasks.json.",
    )
    task_bank_append.add_argument(
        "--source",
        choices=("phase2c",),
        default="phase2c",
        help="Source artifact to append from.",
    )
    task_bank_append.add_argument(
        "--json", action="store_true", help="Print machine-readable JSON."
    )

    task_bank_status = task_bank_sub.add_parser("status", help="Show task-bank coverage counts.")
    task_bank_status.add_argument(
        "--json", action="store_true", help="Print machine-readable JSON."
    )

    task_bank_export = task_bank_sub.add_parser(
        "export", help="Export task-bank events or summary."
    )
    task_bank_export.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON path.",
    )
    task_bank_export.add_argument(
        "--summary",
        action="store_true",
        help="Export summary JSON instead of raw event records.",
    )
    task_bank_export.add_argument(
        "--include-retired-carriers",
        action="store_true",
        help="Include retired title-carrier events in raw exports. Summary exports always report both counts.",
    )

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
