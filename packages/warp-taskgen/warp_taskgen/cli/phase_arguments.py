"""Arguments for the WARP Taskgen ``phase`` command."""

from __future__ import annotations

import argparse
from pathlib import Path

from warp_taskgen.adversarial_actions import ACTION_POLICIES
from warp_taskgen.agent_runtime import RUNNER_BROWSER_USE
from warp_taskgen.cli.argument_defaults import (
    AGENT_PROVIDER_CHOICES,
    DEFAULT_AGENT_MODEL,
    DEFAULT_SANDBOX_MODEL,
)
from warp_taskgen.cli.argument_types import _non_negative_int, _positive_int
from warp_taskgen.phase_4.options import (
    phase_4_variant_budget_choices,
    phase_4_variant_system_choices,
)
from warp_taskgen.phases.phase_1_task_cards import task_capability_profile_choices
from warp_taskgen.runners import available_runners
from warp_taskgen.runtime_composition import RUNTIME_COMPOSITION_CHOICES


def add_phase_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``phase`` command on the CLI subparsers."""
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
            "Runtime Site composition for this Run. Omitting it, or passing "
            "'default', resolves the default GitLab/Reddit composition; the "
            "other names are bounded POC compositions."
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
