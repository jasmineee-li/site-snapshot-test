"""WorldSim v5 CLI entrypoint.

See ``docs/worldsim-v5-technical-specifcation.md`` for the full pipeline
and ``README.md`` for prerequisites.

Usage::

    # Phase 0 reconnaissance against a benchmark codebase on disk
    uv run python -m worldsim.main phase 0 --benchmark vendors/webarena-verified

    # Resume from the last saved checkpoint
    uv run python -m worldsim.main resume
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import fcntl
import ipaddress
import logging
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from worldsim.adversarial_actions import ACTION_POLICIES
from worldsim.config import BenchmarkConfig, has_configured_agent_auth

load_dotenv(override=True)  # override=True: .env values win over empty-string shell vars.

logger = logging.getLogger(__name__)

DEFAULT_AGENT_MODEL = "claude-sonnet-4-6"
DEFAULT_SANDBOX_MODEL = "claude-sonnet-4-6"
AGENT_PROVIDER_CHOICES = ("google", "openai", "anthropic", "openrouter")
_PHASE4_ASYNC_SHUTDOWN_TIMEOUT_ENV = "WORLDSIM_PHASE4_ASYNC_SHUTDOWN_TIMEOUT_S"
_PHASE4_ASYNC_SHUTDOWN_TIMEOUT_DEFAULT_S = 10.0


class Phase4AlreadyRunning(RuntimeError):
    """Raised when the per-state-dir Phase 4 run lock is held."""


def _phase4_async_shutdown_timeout() -> float:
    raw = os.environ.get(_PHASE4_ASYNC_SHUTDOWN_TIMEOUT_ENV, "").strip()
    if not raw:
        return _PHASE4_ASYNC_SHUTDOWN_TIMEOUT_DEFAULT_S
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "Invalid %s=%r; using %.1fs",
            _PHASE4_ASYNC_SHUTDOWN_TIMEOUT_ENV,
            raw,
            _PHASE4_ASYNC_SHUTDOWN_TIMEOUT_DEFAULT_S,
        )
        return _PHASE4_ASYNC_SHUTDOWN_TIMEOUT_DEFAULT_S
    if value <= 0:
        logger.warning(
            "Invalid %s=%r; using %.1fs",
            _PHASE4_ASYNC_SHUTDOWN_TIMEOUT_ENV,
            raw,
            _PHASE4_ASYNC_SHUTDOWN_TIMEOUT_DEFAULT_S,
        )
        return _PHASE4_ASYNC_SHUTDOWN_TIMEOUT_DEFAULT_S
    return value


def _run_phase4_with_bounded_async_shutdown(coro: Any, *, shutdown_timeout_s: float) -> Any:
    """Run Phase 4 without letting third-party background tasks hang process exit.

    Browser Use can leave storage-state/watchdog tasks alive after WorldSim has
    written complete Phase 4 artifacts. ``asyncio.run`` waits indefinitely for
    cancellation during shutdown, which keeps registered r5 jobs marked
    ``running`` and prevents post-run exporters from executing. Phase 4 owns
    browser-agent lifecycle boundaries, so it gets a bounded shutdown wrapper
    while the rest of the CLI keeps standard ``asyncio.run`` behavior.
    """

    loop = asyncio.new_event_loop()
    try:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("bounded Phase 4 runner cannot be called from a running event loop")
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
        if pending:
            for task in pending:
                task.cancel()
            done, still_pending = loop.run_until_complete(
                asyncio.wait(pending, timeout=shutdown_timeout_s)
            )
            for task in done:
                with contextlib.suppress(asyncio.CancelledError):
                    task.result()
            if still_pending:
                for task in still_pending:
                    with contextlib.suppress(Exception):
                        task._log_destroy_pending = False  # type: ignore[attr-defined]
                logger.warning(
                    "Phase 4 async shutdown timed out after %.1fs with %d pending task(s); "
                    "closing loop after completed artifacts were returned",
                    shutdown_timeout_s,
                    len(still_pending),
                )
        loop.run_until_complete(loop.shutdown_asyncgens())
        try:
            loop.run_until_complete(loop.shutdown_default_executor(timeout=shutdown_timeout_s))
        except TypeError:
            try:
                loop.run_until_complete(
                    asyncio.wait_for(loop.shutdown_default_executor(), timeout=shutdown_timeout_s)
                )
            except TimeoutError:
                logger.warning(
                    "Phase 4 default executor shutdown timed out after %.1fs",
                    shutdown_timeout_s,
                )
        except TimeoutError:
            logger.warning(
                "Phase 4 default executor shutdown timed out after %.1fs",
                shutdown_timeout_s,
            )
        asyncio.set_event_loop(None)
        loop.close()


def _is_loopback_hostname(hostname: str | None) -> bool:
    if hostname is None:
        return False
    normalized = hostname.strip().lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


@contextlib.contextmanager
def _phase4_run_lock(state_dir: Path):
    """Prevent concurrent Phase 4 runs from resetting the same benchmark stack."""
    lock_dir = state_dir / "phase_4"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / ".phase4_run.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            existing = handle.read().strip()
            detail = f"; holder: {existing}" if existing else ""
            raise Phase4AlreadyRunning(
                f"another Phase 4 run already holds {lock_path}{detail}"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} cwd={Path.cwd()} cmd={' '.join(sys.argv)}\n")
        handle.flush()
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


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


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser used by ``main()`` and tests."""
    parser = argparse.ArgumentParser(
        prog="worldsim",
        description="WorldSim v5 adversarial evaluation pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

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
        "--instances",
        type=Path,
        help="JSON file with BenchmarkConfig (site_url, db_connection, "
        "reset_endpoint). Required for Phase 4.",
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
            "Phase 4: explicit WorldSim wall-clock timeout for one Browser Use "
            "task. Omit to preserve Browser Use's long-running default. This is "
            "an infrastructure guard for stuck sessions, not an action-step limit."
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
        "feasibility.status='unverified' and Phase 4 runs in grace mode. Use "
        "only for fast dev iteration; shipping runs must not bypass 2c.",
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
            "Resume from the last saved checkpoint. When resuming Phase 2, WorldSim "
            "re-enters the saved internal sub-stage automatically: 2a planning or "
            "2b text fill. There are no separate --phase-2a-only or "
            "--phase-2b-only flags."
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
        help="Benchmark host YAML, e.g. configs/benchmark_hosts/r5.yaml.",
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
        help="Show a read-only operator summary for a WorldSim run.",
    )
    status_cmd.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=None,
        help="Run state dir, phase_4/results.json, or pipeline_state.json. Defaults to WORLDSIM_STATE_DIR/logs.",
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
        help="Run state dir, phase_4/results.json, or pipeline_state.json. Defaults to WORLDSIM_STATE_DIR/logs.",
    )
    inspect_cmd.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    task_bank_cmd = subparsers.add_parser(
        "task-bank",
        help="Manage the append-only admitted-task bank.",
    )
    task_bank_cmd.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Task-bank JSONL path. Defaults to WORLDSIM_STATE_DIR/task_bank/events.jsonl.",
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
        help="WorldSim run dir containing phase_2/adversarial_tasks.json.",
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


def _install_verification_proxy_from_args(args: argparse.Namespace) -> None:
    """Install the verification-proxy adapter if the args point at a config that has one.

    The proxy config is optional; when ``instances.<name>.json`` declares a
    ``verification_proxy`` block with a non-empty token, every ``requests.Session``
    created in-process rewrites allowlisted site-port URLs to the proxy port and
    attaches the ``X-Worldsim-Token`` header. Ports are derived from the
    instances' ``site_url`` so the adapter works for any benchmark.
    """
    # ``--instances`` is the general flag (Phase 1/4); Phase 2c uses
    # ``--feasibility-instances`` instead. Prefer whichever is set and points
    # at an existing file.
    candidates = [
        getattr(args, "instances", None),
        getattr(args, "feasibility_instances", None),
    ]
    path: Path | None = None
    for candidate in candidates:
        if candidate is None:
            continue
        maybe_path = Path(candidate)
        if maybe_path.exists():
            path = maybe_path
            break
    if path is None:
        return
    import json
    from urllib.parse import urlsplit

    from worldsim.http_proxy import install_proxy

    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        raise RuntimeError(
            f"verification_proxy_invalid: could not parse instances config {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"verification_proxy_invalid: instances config {path} must contain a JSON object"
        )
    proxy_data = payload.get("verification_proxy")
    if proxy_data is None:
        return
    if not isinstance(proxy_data, dict):
        raise RuntimeError(
            f"verification_proxy_invalid: verification_proxy in {path} must be an object"
        )
    token = proxy_data.get("token")
    if token is None:
        raise RuntimeError(
            f"verification_proxy_invalid: verification_proxy.token missing in {path}"
        )
    if not isinstance(token, str):
        raise RuntimeError(
            f"verification_proxy_invalid: verification_proxy.token in {path} must be a string"
        )
    if not token.strip():
        return
    scheme = proxy_data.get("scheme", "http")
    if not isinstance(scheme, str) or scheme.strip().lower() not in {"http", "https"}:
        raise RuntimeError(
            f"verification_proxy_invalid: verification_proxy.scheme in {path} must be 'http' or 'https'"
        )
    port_offset = proxy_data.get("port_offset", 10000)
    if not isinstance(port_offset, int) or port_offset < 0:
        raise RuntimeError(
            f"verification_proxy_invalid: verification_proxy.port_offset in {path} must be a non-negative integer"
        )
    instances = payload.get("instances")
    if not isinstance(instances, list):
        raise RuntimeError(
            f"verification_proxy_invalid: instances list missing or invalid in {path}"
        )
    site_ports: set[int] = set()
    non_loopback_site_url_seen = False
    for index, instance in enumerate(instances):
        if not isinstance(instance, dict):
            raise RuntimeError(
                f"verification_proxy_invalid: instances[{index}] in {path} must be an object"
            )
        site_url = instance.get("site_url")
        if not isinstance(site_url, str):
            raise RuntimeError(
                f"verification_proxy_invalid: instances[{index}].site_url in {path} must be a string"
            )
        parsed = urlsplit(site_url)
        if not parsed.scheme or not parsed.hostname or parsed.port is None:
            raise RuntimeError(
                f"verification_proxy_invalid: instances[{index}].site_url {site_url!r} must include scheme, host, and explicit port"
            )
        site_ports.add(parsed.port)
        if not _is_loopback_hostname(parsed.hostname):
            non_loopback_site_url_seen = True
    if not site_ports:
        raise RuntimeError(
            f"verification_proxy_invalid: no proxy-eligible site_url ports found in {path}"
        )
    if not non_loopback_site_url_seen:
        logging.getLogger(__name__).info(
            "verification_proxy ignored for %s because all site_url hosts are loopback",
            path,
        )
        return
    install_proxy(
        token=token.strip(),
        port_offset=port_offset,
        site_ports=site_ports,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. See module docstring for usage."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = build_parser().parse_args(argv)

    if args.command == "resume":
        return _dispatch_resume(args)

    if args.command == "phase":
        try:
            _install_verification_proxy_from_args(args)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        return _dispatch_phase(args)

    if args.command == "rescore-phase-3":
        from worldsim.phases import phase_3_rescore

        return phase_3_rescore.run(args)

    if args.command == "preflight":
        return _dispatch_preflight(args)

    if args.command == "status":
        return _dispatch_status(args)

    if args.command == "inspect":
        return _dispatch_inspect(args)

    if args.command == "task-bank":
        return _dispatch_task_bank(args)

    return 0


def _parse_cli_sites(raw_sites: str | None) -> set[str] | None:
    if raw_sites is None:
        return None
    parsed = {site.strip() for site in raw_sites.split(",") if site.strip()}
    return parsed or None


def _task_bank_path_from_args(args: argparse.Namespace) -> Path:
    from worldsim.task_bank import default_task_bank_path

    return Path(getattr(args, "path", None) or default_task_bank_path())


def _format_task_bank_summary(summary: dict[str, Any], *, path: Path) -> str:
    def fmt_counts(values: Any) -> str:
        if not isinstance(values, dict) or not values:
            return "none"
        return ", ".join(f"{key}={value}" for key, value in sorted(values.items()))

    return "\n".join(
        [
            f"Task bank: {path}",
            (
                "Events: "
                f"total={summary.get('total_events', 0)} "
                f"admitted={summary.get('admitted_tasks', 0)} "
                f"active_admitted={summary.get('active_admitted_tasks', 0)} "
                f"retired_admitted={summary.get('retired_admitted_tasks', 0)} "
                f"phase4_results={summary.get('phase4_results', 0)}"
            ),
            f"By site: {fmt_counts(summary.get('by_site'))}",
            f"By origin: {fmt_counts(summary.get('by_origin'))}",
            f"By surface: {fmt_counts(summary.get('by_surface'))}",
            f"Active by surface: {fmt_counts(summary.get('active_by_surface'))}",
            f"Retired by surface: {fmt_counts(summary.get('retired_by_surface'))}",
            f"By archetype: {fmt_counts(summary.get('by_archetype'))}",
            (
                "Latest: "
                f"{summary.get('latest_created_at') or 'none'} "
                f"{summary.get('latest_event_id') or 'none'}"
            ),
        ]
    )


def _dispatch_task_bank(args: argparse.Namespace) -> int:
    import json

    from worldsim.task_bank import (
        TaskBankError,
        admitted_events_from_phase2c_run,
        append_task_bank_events,
        is_active_task_bank_event,
        load_task_bank,
        summarize_task_bank,
    )

    path = _task_bank_path_from_args(args)
    try:
        if args.task_bank_command == "append":
            events = admitted_events_from_phase2c_run(Path(args.run_dir))
            appended = append_task_bank_events(path, events)
            payload = {
                "task_bank_path": str(path),
                "run_dir": str(args.run_dir),
                "source": args.source,
                "appended": len(appended),
                "summary": summarize_task_bank(load_task_bank(path)),
            }
            if args.json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                print(
                    f"Task bank append: appended={payload['appended']} "
                    f"path={payload['task_bank_path']}"
                )
                print(_format_task_bank_summary(payload["summary"], path=path))
            return 0
        if args.task_bank_command == "status":
            summary = summarize_task_bank(load_task_bank(path))
            if args.json:
                print(
                    json.dumps(
                        {"task_bank_path": str(path), "summary": summary}, indent=2, sort_keys=True
                    )
                )
            else:
                print(_format_task_bank_summary(summary, path=path))
            return 0
        if args.task_bank_command == "export":
            events = load_task_bank(path)
            if args.summary:
                payload: Any = summarize_task_bank(events)
            else:
                payload = (
                    events
                    if args.include_retired_carriers
                    else [event for event in events if is_active_task_bank_event(event)]
                )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            print(f"Task bank export: wrote {args.output}")
            return 0
    except TaskBankError as exc:
        print(f"task-bank failed: {exc}", file=sys.stderr)
        return 2
    return 2


def _dispatch_status(args: argparse.Namespace) -> int:
    from worldsim.cli_status import build_status_payload, format_status_payload

    try:
        payload = build_status_payload(getattr(args, "path", None))
    except Exception as exc:
        print(f"status failed: {exc}", file=sys.stderr)
        return 2
    if getattr(args, "json", False):
        import json

        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(format_status_payload(payload, inspect_limit=getattr(args, "inspect_limit", 5)))
    return 0


def _dispatch_inspect(args: argparse.Namespace) -> int:
    from worldsim.cli_status import build_inspection_payload, format_inspection_payload

    try:
        payload = build_inspection_payload(args.task_id, getattr(args, "path", None))
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"inspect failed: {exc}", file=sys.stderr)
        return 2
    if getattr(args, "json", False):
        import json

        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(format_inspection_payload(payload))
    return 0


def _dispatch_preflight(args: argparse.Namespace) -> int:
    """Run Phase 4 preflight gates standalone — fast feedback loop for config changes.

    Wraps ``pytest -m preflight tests/preflight`` with the same
    ``WORLDSIM_PREFLIGHT_*`` env vars that ``scripts/setup_phase4_on_host.sh``
    step 7 uses. Extra args after ``--`` are forwarded to pytest.
    """
    import subprocess

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()

    if args.host_config is not None:
        host_config = args.host_config
        if not host_config.is_absolute():
            host_config = repo_root / host_config
        if not host_config.exists():
            print(f"host-config not found: {host_config}", file=sys.stderr)
            return 2
        env["WORLDSIM_PREFLIGHT_HOST_CONFIG"] = str(host_config)

    instances = args.instances
    if not instances.is_absolute():
        instances = repo_root / instances
    if not instances.exists():
        print(
            f"instances file not found: {instances}\n"
            f"Regenerate via scripts/generate_scale_r5.sh, "
            f"or pass --instances <path>.",
            file=sys.stderr,
        )
        return 2
    env["WORLDSIM_PREFLIGHT_INSTANCES"] = str(instances)

    cmd = [sys.executable, "-m", "pytest", "-m", "preflight", "tests/preflight"]
    extra = [arg for arg in (args.pytest_args or []) if arg != "--"]
    cmd.extend(extra)

    print(f"running: {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, cwd=repo_root, env=env)
    return result.returncode


# Ordered pipeline steps. Each entry maps step name -> (phase_id, sub) where
# sub is the sub-step for Phase 0, or None for later phases.
_PHASE_ORDER: list[str] = [
    "phase_0a",
    "phase_0b",
    "phase_0c",
    "phase_0d",
    "phase_1",
    "phase_2",
    "phase_3",
    "phase_4",
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

    if status in {"complete", "partial_complete"}:
        target = _next_step(last_step)
        if target is None:
            print(f"Last checkpoint: {last_step} complete. Pipeline finished — nothing to resume.")
            return 0
        qualifier = "partial and " if status == "partial_complete" else ""
        print(f"Last checkpoint: {last_step} {qualifier}complete. Resuming from {target}.")
    elif status == "running":
        target = last_step
        print(f"Last checkpoint: {last_step} was running (likely crashed). Re-running {target}.")
    elif status == "failed":
        target = last_step
        reason = state.get("reason")
        suffix = f" ({reason})" if reason else ""
        print(f"Last checkpoint: {last_step} failed{suffix}. Re-running {target}.")
    else:
        print(f"Last checkpoint: {last_step} has unknown status {status!r}.", file=sys.stderr)
        return 1

    # Build a synthetic argparse.Namespace that _dispatch_phase understands.
    # CLI flags override state metadata; state metadata fills gaps.
    benchmark = getattr(args, "benchmark", None)
    config = getattr(args, "config", None)
    instances = getattr(args, "instances", None)
    agent_model = getattr(args, "agent_model", None)
    sandbox_model = getattr(args, "sandbox_model", None)
    agent_provider = getattr(args, "agent_provider", None)
    agent_service_tier = getattr(args, "agent_service_tier", None)
    agent_llm_timeout = getattr(args, "agent_llm_timeout", None)
    agent_step_timeout = getattr(args, "agent_step_timeout", None)
    agent_task_timeout = getattr(args, "agent_task_timeout", None)
    generate_novel = getattr(args, "generate_novel", None)
    novel_tasks_per_site = getattr(args, "novel_tasks_per_site", None)
    max_tasks_per_site = getattr(args, "max_tasks_per_site", None)
    task_origin = getattr(args, "task_origin", None)
    sites = getattr(args, "sites", None)
    phase_2b_texts_per_plan = getattr(args, "phase_2b_texts_per_plan", None)
    phase_2_text_fill_concurrency = getattr(args, "phase_2_text_fill_concurrency", None)
    phase_2_text_model = getattr(args, "phase_2_text_model", None)
    phase_2a_action_policy = getattr(args, "phase_2a_action_policy", None)

    skip_feasibility = getattr(args, "skip_feasibility", None)
    feasibility_only = getattr(args, "feasibility_only", None)
    feasibility_instances = getattr(args, "feasibility_instances", None)
    feasibility_concurrency = getattr(args, "feasibility_concurrency", None)
    feasibility_retry_count = getattr(args, "feasibility_retry_count", None)
    feasibility_ttl_hours = getattr(args, "feasibility_ttl_hours", None)
    force_reverify = getattr(args, "force_reverify", None)
    no_l3_l4 = getattr(args, "no_l3_l4", None)

    # Fall back to paths stored in state metadata
    if benchmark is None and "benchmark_path" in state:
        benchmark = Path(state["benchmark_path"])
    if config is None and "manifest_path" in state:
        config = Path(state["manifest_path"])
    if instances is None and "instances_path" in state:
        instances = Path(state["instances_path"])
    if agent_model is None:
        agent_model = state.get("agent_model")
    if sandbox_model is None:
        sandbox_model = state.get("sandbox_model", DEFAULT_SANDBOX_MODEL)
    if agent_provider is None:
        agent_provider = state.get("agent_provider")
    if agent_service_tier is None:
        agent_service_tier = state.get("agent_service_tier")
    if agent_llm_timeout is None:
        agent_llm_timeout = state.get("agent_llm_timeout")
    if agent_step_timeout is None:
        agent_step_timeout = state.get("agent_step_timeout")
    if agent_task_timeout is None:
        agent_task_timeout = state.get("agent_task_timeout")
    if generate_novel is None:
        generate_novel = state.get("generate_novel", False)
    if novel_tasks_per_site is None:
        novel_tasks_per_site = state.get("novel_tasks_per_site")
    if task_origin is None:
        task_origin = state.get("task_origin")
    if sites is None:
        sites = state.get("sites")
    if phase_2b_texts_per_plan is None:
        phase_2b_texts_per_plan = state.get("phase_2b_texts_per_plan")
    if phase_2_text_fill_concurrency is None:
        phase_2_text_fill_concurrency = state.get("phase_2_text_fill_concurrency")
    if phase_2_text_model is None:
        phase_2_text_model = state.get("phase_2_text_model")
    if phase_2a_action_policy is None:
        phase_2a_action_policy = state.get("phase_2a_action_policy")
    if skip_feasibility is None:
        skip_feasibility = state.get("skip_feasibility", False)
    if feasibility_only is None:
        feasibility_only = state.get("feasibility_only", False)
    if feasibility_instances is None:
        feasibility_instances = state.get("feasibility_instances")
    if feasibility_concurrency is None:
        feasibility_concurrency = state.get("feasibility_concurrency", 10)
    if feasibility_retry_count is None:
        feasibility_retry_count = state.get("feasibility_retry_count", 1)
    if feasibility_ttl_hours is None:
        feasibility_ttl_hours = state.get("feasibility_ttl_hours")
    if force_reverify is None:
        force_reverify = state.get("force_reverify", False)
    if no_l3_l4 is None:
        resolution_sig = state.get("phase_2a_resolution_signature")
        if isinstance(resolution_sig, dict):
            no_l3_l4 = bool(resolution_sig.get("no_l3_l4", False))
        else:
            no_l3_l4 = False

    # Map target step to phase ID for _dispatch_phase (e.g. "phase_0a" -> "0a")
    phase_id = target.replace("phase_", "")

    allow_unknown_auth = getattr(args, "allow_unknown_auth", None)
    if allow_unknown_auth is None:
        allow_unknown_auth = state.get("allow_unknown_auth", False)
    skip_host_bound_storage_state_auth = getattr(args, "skip_host_bound_storage_state_auth", None)
    if skip_host_bound_storage_state_auth is None:
        skip_host_bound_storage_state_auth = state.get("skip_host_bound_storage_state_auth", False)

    synthetic = argparse.Namespace(
        command="phase",
        phase=phase_id,
        resume=True,
        benchmark=benchmark,
        config=config,
        instances=instances,
        agent_model=agent_model,
        sandbox_model=sandbox_model,
        agent_provider=agent_provider,
        agent_service_tier=agent_service_tier,
        agent_llm_timeout=agent_llm_timeout,
        agent_step_timeout=agent_step_timeout,
        agent_task_timeout=agent_task_timeout,
        generate_novel=generate_novel,
        novel_tasks_per_site=novel_tasks_per_site,
        max_tasks_per_site=max_tasks_per_site,
        task_origin=task_origin,
        sites=sites,
        phase_2b_texts_per_plan=phase_2b_texts_per_plan,
        phase_2_text_fill_concurrency=phase_2_text_fill_concurrency,
        phase_2_text_model=phase_2_text_model,
        phase_2a_action_policy=phase_2a_action_policy,
        allow_unknown_auth=allow_unknown_auth,
        skip_host_bound_storage_state_auth=skip_host_bound_storage_state_auth,
        skip_feasibility=skip_feasibility,
        feasibility_only=feasibility_only,
        feasibility_instances=feasibility_instances,
        feasibility_concurrency=feasibility_concurrency,
        feasibility_retry_count=feasibility_retry_count,
        feasibility_ttl_hours=feasibility_ttl_hours,
        force_reverify=force_reverify,
        no_l3_l4=no_l3_l4,
    )

    try:
        _install_verification_proxy_from_args(synthetic)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return _dispatch_phase(synthetic)


def _unknown_auth_sites(
    state_dir: Path, *, instances: list[dict[str, Any]] | None = None
) -> list[str]:
    """Return a list of site names whose auth is truly unknown.

    A site is *not* unknown if either:
    - Phase 0c declared ``auth_mechanism.type`` as something other than
      ``"unknown"``, OR
    - ``instances.json`` provides ``agent_auth`` for the site (the static,
      instances.json-driven path supersedes Phase 0c discovery).

    Handles both layouts Phase 0c can produce:
      - flat: ``<state_dir>/phase_0c/AGENT_CONTEXT_<site>.json`` (current)
      - nested: ``<state_dir>/phase_0c/<site>/AGENT_CONTEXT.json`` (future)

    Returns an empty list when the directory is absent or nothing has been
    profiled yet.
    """
    import json as _json

    # Sites with instance-level agent_auth are never unknown.
    instance_auth_sites: set[str] = set()
    if instances:
        for inst in instances:
            if isinstance(inst, dict) and has_configured_agent_auth(inst.get("agent_auth")):
                site_name = inst.get("site_name", "")
                if site_name:
                    instance_auth_sites.add(site_name)

    profiles_dir = state_dir / "phase_0c"
    if not profiles_dir.exists():
        return []

    parse_errors: list[str] = []

    def _check(ctx_path: Path, site_name: str) -> None:
        if site_name in instance_auth_sites:
            return
        if not ctx_path.exists():
            return
        try:
            data = _json.loads(ctx_path.read_text(encoding="utf-8"))
        except (OSError, _json.JSONDecodeError) as exc:
            parse_errors.append(f"{ctx_path}: {exc}")
            return
        if not isinstance(data, dict):
            parse_errors.append(f"{ctx_path}: expected JSON object")
            return
        mech = data.get("auth_mechanism")
        if isinstance(mech, dict) and mech.get("type") == "unknown":
            unknown.append(site_name)

    unknown: list[str] = []

    # Flat layout: AGENT_CONTEXT_<site>.json
    for ctx_path in sorted(profiles_dir.glob("AGENT_CONTEXT_*.json")):
        site_name = ctx_path.stem[len("AGENT_CONTEXT_") :]
        _check(ctx_path, site_name)

    # Nested layout: <site>/AGENT_CONTEXT.json
    for site_dir in sorted(profiles_dir.iterdir()):
        if not site_dir.is_dir():
            continue
        _check(site_dir / "AGENT_CONTEXT.json", site_dir.name)

    if parse_errors:
        raise RuntimeError(
            "Failed to read Phase 0c AGENT_CONTEXT artifacts required for the unknown-auth gate:\n"
            + "\n".join(f"  - {error}" for error in parse_errors)
        )

    return sorted(set(unknown))


def _dispatch_phase(args: argparse.Namespace) -> int:
    """Dispatch to the requested phase module."""
    from worldsim.cost_tracker import tracker as cost_tracker
    from worldsim.phases import (
        phase_0_recon,
        phase_0d_auth_bootstrap,
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
            print(
                f"--benchmark is required for Phase {phase}. "
                "--config overrides the Phase 1 manifest path; it is not a "
                "substitute for the benchmark codebase during Phase 0.",
                file=sys.stderr,
            )
            return 1
        rc = asyncio.run(
            phase_0_recon.run(
                benchmark=args.benchmark,
                sub=phase,
                sandbox_model=args.sandbox_model,
                instances_path=getattr(args, "instances", None),
                host_inventory_instances_path=getattr(args, "host_inventory_instances", None),
                site_filter=_parse_cli_sites(getattr(args, "sites", None)),
            )
        )
    elif phase == "0d":
        if not args.benchmark:
            print(
                "--benchmark is required for Phase 0d; generator_script paths "
                "in AGENT_CONTEXT resolve relative to the benchmark root.",
                file=sys.stderr,
            )
            return 1
        rc = asyncio.run(phase_0d_auth_bootstrap.run(args))
    elif phase == "1":
        rc = asyncio.run(phase_1_tasks.run(args))
    elif phase in {"2", "2c"}:
        if phase == "2c":
            args.feasibility_only = True
        rc = asyncio.run(phase_2_injections.run(args))
    elif phase == "3":
        rc = asyncio.run(phase_3_benign.run(args))
    elif phase == "4":
        allow_unknown = getattr(args, "allow_unknown_auth", False)
        instances_for_gate: list[dict[str, object]] | None = None
        instances_path = getattr(args, "instances", None)
        if instances_path is not None and Path(instances_path).exists():
            try:
                config = BenchmarkConfig.model_validate_json(Path(instances_path).read_text())
                instances_for_gate = [instance.model_dump() for instance in config.instances]
            except ValueError as exc:
                print(f"Failed to parse instances config {instances_path}: {exc}", file=sys.stderr)
                return 1
        try:
            unknown_sites = _unknown_auth_sites(get_state_dir(), instances=instances_for_gate)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        if unknown_sites and not allow_unknown:
            print(
                f"Phase {phase} refused: the following sites declare "
                "auth_mechanism.type='unknown' and need human review before "
                "they can be evaluated:\n  - "
                + "\n  - ".join(unknown_sites)
                + "\nRe-run with --allow-unknown-auth to proceed anyway "
                "(unknown-auth tasks will no-op auth injection and likely fail).",
                file=sys.stderr,
            )
            return 2
        try:
            with _phase4_run_lock(get_state_dir()):
                rc = _run_phase4_with_bounded_async_shutdown(
                    phase_4_adversarial.run(args),
                    shutdown_timeout_s=_phase4_async_shutdown_timeout(),
                )
        except Phase4AlreadyRunning as exc:
            print(
                f"Phase 4 refused to start because another run is active: {exc}",
                file=sys.stderr,
            )
            return 2
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
