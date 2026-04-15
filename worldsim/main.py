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
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from worldsim.config import BenchmarkConfig, has_configured_agent_auth

load_dotenv(override=True)  # override=True: .env values win over empty-string shell vars.

DEFAULT_AGENT_MODEL = "gemini-3-flash-preview"
DEFAULT_SANDBOX_MODEL = "claude-sonnet-4-6"
AGENT_PROVIDER_CHOICES = ("google", "openai", "anthropic", "openrouter")


def _positive_int(value: str) -> int:
    """Argparse type for positive integer CLI flags."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
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
            "internal stages: 2a planning in Modal sandboxes, then 2b host-side "
            "text fill. The command runs those stages sequentially; there are no "
            "separate --phase-2a-only or --phase-2b-only flags."
        ),
    )
    phase_cmd.add_argument(
        "phase",
        choices=["0", "0a", "0b", "0c", "0d", "1", "2", "3", "4"],
        help="Phase to run",
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
        help="For Phase 1, also generate Mode B novel tasks for eligible sites.",
    )
    phase_cmd.add_argument(
        "--instances",
        type=Path,
        help="JSON file with BenchmarkConfig (site_url, db_connection, "
        "reset_endpoint). Required for Phases 3-4.",
    )
    phase_cmd.add_argument(
        "--agent-model",
        default=DEFAULT_AGENT_MODEL,
        help=f"LLM model name for Browser Use agent (default: {DEFAULT_AGENT_MODEL}). "
        "Examples: gpt-5.4, claude-sonnet-4-6, gemini-3-flash-preview, gemini-3.1-pro-preview.",
    )
    phase_cmd.add_argument(
        "--sandbox-model",
        default=DEFAULT_SANDBOX_MODEL,
        help="Claude sandbox model for Phase 3-4 diagnosis/judge/fix steps "
        f"(default: {DEFAULT_SANDBOX_MODEL}).",
    )
    phase_cmd.add_argument(
        "--agent-provider",
        default=None,
        choices=AGENT_PROVIDER_CHOICES,
        help="LLM provider (default: auto-detect from model name). "
        "Requires the corresponding env var: GOOGLE_API_KEY, OPENAI_API_KEY, "
        "or ANTHROPIC_API_KEY.",
    )
    phase_cmd.add_argument(
        "--full-baseline",
        action="store_true",
        help="Phase 3: validate all benign tasks, not just adversarial-paired ones. "
        "Produces baseline capability metric.",
    )
    phase_cmd.add_argument(
        "--max-tasks-per-site",
        type=_positive_int,
        default=None,
        metavar="N",
        help="Phases 3-4: cap tasks to at most N per site for smoke testing. "
        "Selection is deterministic (fixed seed). Omit for full runs. "
        "Use `resume --max-tasks-per-site N` to keep the cap, or omit it on "
        "resume to process all remaining tasks.",
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
        "--phase-2-sandbox-concurrency",
        type=_positive_int,
        default=None,
        metavar="N",
        help="Phase 2a planning: cap concurrent sandbox shards to at most N launches "
        "while `phase 2` runs 2a then 2b sequentially. Omit to use the phase default.",
    )
    phase_cmd.add_argument(
        "--phase-2-launch-jitter-ms",
        type=_positive_int,
        default=None,
        metavar="MS",
        help="Phase 2a planning: add up to MS of deterministic per-shard launch "
        "jitter to smooth burst traffic before 2b text fill starts. Omit to use "
        "the phase default.",
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
        "--allow-unknown-auth",
        action="store_true",
        default=False,
        help="Phase 3-4: proceed even when a site's auth_mechanism.type is 'unknown'. "
        "Default behavior is to refuse unknown-auth tasks so humans review them first.",
    )

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
        "--generate-novel",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Override saved Phase 1 state to enable Mode B novel task generation.",
    )
    resume_cmd.add_argument(
        "--full-baseline",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Phase 3: validate all benign tasks, not just adversarial-paired ones. "
        "Produces baseline capability metric.",
    )
    resume_cmd.add_argument(
        "--max-tasks-per-site",
        type=_positive_int,
        default=argparse.SUPPRESS,
        metavar="N",
        help="Override per-site task cap for the resumed phase. Omit to run all remaining tasks.",
    )
    resume_cmd.add_argument(
        "--sites",
        type=str,
        default=argparse.SUPPRESS,
        metavar="SITE[,SITE...]",
        help="Override the saved site filter on resume.",
    )
    resume_cmd.add_argument(
        "--phase-2-sandbox-concurrency",
        type=_positive_int,
        default=argparse.SUPPRESS,
        metavar="N",
        help="Override the saved Phase 2a planning sandbox concurrency on resume.",
    )
    resume_cmd.add_argument(
        "--phase-2-launch-jitter-ms",
        type=_positive_int,
        default=argparse.SUPPRESS,
        metavar="MS",
        help="Override the saved Phase 2a planning launch jitter on resume.",
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
        "--allow-unknown-auth",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Override the saved gate for auth_mechanism.type='unknown' during resume.",
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

    return parser


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
        return _dispatch_phase(args)

    if args.command == "rescore-phase-3":
        from worldsim.phases import phase_3_rescore

        return phase_3_rescore.run(args)

    return 0


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
    generate_novel = getattr(args, "generate_novel", None)
    full_baseline = getattr(args, "full_baseline", None)
    max_tasks_per_site = getattr(args, "max_tasks_per_site", None)
    sites = getattr(args, "sites", None)
    phase_2_sandbox_concurrency = getattr(args, "phase_2_sandbox_concurrency", None)
    phase_2_launch_jitter_ms = getattr(args, "phase_2_launch_jitter_ms", None)
    phase_2b_texts_per_plan = getattr(args, "phase_2b_texts_per_plan", None)
    phase_2_text_fill_concurrency = getattr(args, "phase_2_text_fill_concurrency", None)
    phase_2_text_model = getattr(args, "phase_2_text_model", None)

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
    if generate_novel is None:
        generate_novel = state.get("generate_novel", False)
    if full_baseline is None:
        full_baseline = state.get("full_baseline", False)
    if sites is None:
        sites = state.get("sites")
    if phase_2_sandbox_concurrency is None:
        phase_2_sandbox_concurrency = state.get("phase_2_sandbox_concurrency")
    if phase_2_launch_jitter_ms is None:
        phase_2_launch_jitter_ms = state.get("phase_2_launch_jitter_ms")
    if phase_2b_texts_per_plan is None:
        phase_2b_texts_per_plan = state.get("phase_2b_texts_per_plan")
    if phase_2_text_fill_concurrency is None:
        phase_2_text_fill_concurrency = state.get("phase_2_text_fill_concurrency")
    if phase_2_text_model is None:
        phase_2_text_model = state.get("phase_2_text_model")

    # Map target step to phase ID for _dispatch_phase (e.g. "phase_0a" -> "0a")
    phase_id = target.replace("phase_", "")

    allow_unknown_auth = getattr(args, "allow_unknown_auth", None)
    if allow_unknown_auth is None:
        allow_unknown_auth = state.get("allow_unknown_auth", False)

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
        generate_novel=generate_novel,
        full_baseline=full_baseline,
        max_tasks_per_site=max_tasks_per_site,
        sites=sites,
        phase_2_sandbox_concurrency=phase_2_sandbox_concurrency,
        phase_2_launch_jitter_ms=phase_2_launch_jitter_ms,
        phase_2b_texts_per_plan=phase_2b_texts_per_plan,
        phase_2_text_fill_concurrency=phase_2_text_fill_concurrency,
        phase_2_text_model=phase_2_text_model,
        allow_unknown_auth=allow_unknown_auth,
    )

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
    elif phase == "2":
        rc = asyncio.run(phase_2_injections.run(args))
    elif phase in {"3", "4"}:
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
        if phase == "3":
            rc = asyncio.run(phase_3_benign.run(args))
        else:
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
