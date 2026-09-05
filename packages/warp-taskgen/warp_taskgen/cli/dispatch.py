"""WARP Taskgen CLI dispatch.

See ``docs/warp-taskgen-technical-spec.md`` for the full pipeline
and ``README.md`` for prerequisites.

Usage::

    # Phase 0 reconnaissance against a benchmark codebase on disk
    uv run warp-taskgen phase 0 --benchmark vendors/webarena-verified

    # Resume from the last saved checkpoint
    uv run warp-taskgen resume
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import sys
from pathlib import Path

from warp_taskgen.cli.args import _parse_cli_sites, build_parser
from warp_taskgen.cli.auth import _unknown_auth_sites
from warp_taskgen.cli.derived_run import dispatch_derived_resume
from warp_taskgen.cli.env import _normalize_compat_env_aliases
from warp_taskgen.cli.phase4_lock import (
    Phase4AlreadyRunning,
    _phase4_async_shutdown_timeout,
    _phase4_run_lock,
    _run_phase4_with_bounded_async_shutdown,
)
from warp_taskgen.cli.proxy import _install_verification_proxy_from_args
from warp_taskgen.cli.task_bank import _dispatch_task_bank
from warp_taskgen.config import load_benchmark_config

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. See module docstring for usage."""
    _normalize_compat_env_aliases()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = build_parser().parse_args(argv)

    if args.command == "resume":
        from warp_taskgen.cli.resume import _dispatch_resume

        return _dispatch_resume(args)

    if args.command == "derive-and-resume":
        from warp_taskgen.cli.derived_run import dispatch_derived_resume

        return dispatch_derived_resume(args)

    if args.command == "pause":
        from warp_taskgen.cli.run_control import dispatch_pause

        return dispatch_pause(args)

    if args.command == "phase":
        try:
            _install_verification_proxy_from_args(args)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        return _dispatch_phase(args)

    if args.command == "rescore-phase-3":
        from warp_taskgen.phases import phase_3_rescore

        return phase_3_rescore.run(args)

    if args.command == "preflight":
        return _dispatch_preflight(args)

    if args.command == "status":
        return _dispatch_status(args)

    if args.command == "inspect":
        return _dispatch_inspect(args)

    if args.command == "site":
        from warp_taskgen.cli.site_composition_check import dispatch_site_composition

        return dispatch_site_composition(args)

    if args.command == "trace":
        return args.func(args)

    if args.command == "agentlab":
        from warp_taskgen import agentlab_cli

        if args.agentlab_command == "run":
            return agentlab_cli.run(args)
        if args.agentlab_command == "models":
            return agentlab_cli.models(args)
        return 1

    if args.command == "task-bank":
        return _dispatch_task_bank(args)

    return 0


def _dispatch_status(args: argparse.Namespace) -> int:
    from warp_taskgen.cli.status import build_status_payload, format_status_payload

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
    from warp_taskgen.cli.status import build_inspection_payload, format_inspection_payload

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

    package_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()

    if args.host_config is not None:
        host_config = args.host_config
        if not host_config.is_absolute():
            host_config = package_root / host_config
        if not host_config.exists():
            print(f"host-config not found: {host_config}", file=sys.stderr)
            return 2
        env["WORLDSIM_PREFLIGHT_HOST_CONFIG"] = str(host_config)

    instances = args.instances
    if not instances.is_absolute():
        instances = package_root / instances
    if not instances.exists():
        print(
            f"instances file not found: {instances}\n"
            f"Regenerate via scripts/generate_scale.sh with the selected "
            f"--host-config/--scale-config, "
            f"or pass --instances <path>.",
            file=sys.stderr,
        )
        return 2
    env["WORLDSIM_PREFLIGHT_INSTANCES"] = str(instances)

    cmd = [sys.executable, "-m", "pytest", "-m", "preflight", "tests/preflight"]
    extra = [arg for arg in (args.pytest_args or []) if arg != "--"]
    cmd.extend(extra)

    print(f"running: {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, cwd=package_root, env=env)
    return result.returncode


def _dispatch_phase(args: argparse.Namespace) -> int:
    """Resolve immutable Run identity, then dispatch to the requested phase."""

    from warp_taskgen.state import (
        bind_run_definition,
        get_state_dir,
        validate_run_definition_binding,
    )

    phase = str(getattr(args, "phase", ""))
    # ``phase 2c`` is the CLI alias for Phase 2 feasibility-only execution.
    # Normalize the alias before projecting the immutable Run Definition so
    # later phases observe the same value that Phase 2 persists.
    if phase == "2c":
        args.feasibility_only = True

    from warp_taskgen.phase_1.run_lock import Phase1AlreadyRunning
    from warp_taskgen.phase_2.run_lock import Phase2AlreadyRunning

    def _dispatch_with_run_definition() -> int:
        from warp_taskgen.cli.run_identity import resolve_cli_run_transition

        transition = getattr(args, "_run_transition", None)
        if transition is None:
            try:
                transition = resolve_cli_run_transition(args)
            except ValueError as exc:
                print(f"Phase dispatch rejected by Run Definition: {exc}", file=sys.stderr)
                return 2
        if transition.kind == "derived_required":
            fields = ", ".join(transition.drift_fields) or "unknown inputs"
            print(
                "Phase dispatch requires an isolated Derived Run before execution "
                f"({transition.reason_code}; changed: {fields}).",
                file=sys.stderr,
            )
            return 2
        if transition.kind == "rejected":
            print(f"Phase dispatch rejected: {transition.reason_code}", file=sys.stderr)
            return 2
        definition = transition.definition if transition.kind in {"new", "exact"} else None
        try:
            validate_run_definition_binding(definition, state_dir=get_state_dir())
        except ValueError as exc:
            print(f"Phase dispatch rejected by persisted Run Definition: {exc}", file=sys.stderr)
            return 2
        from warp_taskgen.cli.run_control import dispatch_phase_with_run_control

        def _run_bound_phase() -> int:
            with bind_run_definition(definition, state_dir=get_state_dir()):
                return _dispatch_phase_with_run_context(args)

        @contextlib.contextmanager
        def _lifecycle_guard():
            if phase in {"2", "2c"}:
                from warp_taskgen.phase_2.run_lock import phase_2_run_lock

                with phase_2_run_lock(get_state_dir()):
                    yield
                return
            if phase != "4":
                yield
                return
            from warp_taskgen.phase_4.sweep_tag import sweep_in_progress

            with _phase4_run_lock(get_state_dir()), sweep_in_progress():
                yield

        return dispatch_phase_with_run_control(
            phase=phase,
            state_dir=get_state_dir(),
            operation=_run_bound_phase,
            lifecycle_guard=_lifecycle_guard,
        )

    try:
        if phase == "1":
            from warp_taskgen.phase_1.run_lock import phase_1_run_lock

            with phase_1_run_lock(get_state_dir()):
                return _dispatch_with_run_definition()
        return _dispatch_with_run_definition()
    except Phase4AlreadyRunning as exc:
        print(
            f"Phase 4 refused to start because another run is active: {exc}",
            file=sys.stderr,
        )
        return 2
    except Phase2AlreadyRunning as exc:
        print(
            f"Phase 2 refused to start because another run is active: {exc}",
            file=sys.stderr,
        )
        return 2
    except Phase1AlreadyRunning as exc:
        print(
            f"Phase 1 refused to start because another run is active: {exc}",
            file=sys.stderr,
        )
        return 2


def _dispatch_phase_with_run_context(args: argparse.Namespace) -> int:
    """Dispatch to the requested phase module under a resolved Run context."""
    from warp_taskgen.cost_tracker import tracker as cost_tracker
    from warp_taskgen.state import get_state_dir

    # Load any previously saved cost data so cross-phase totals accumulate.
    cost_report_path = get_state_dir() / "cost_report.json"
    cost_tracker.load(cost_report_path)

    phase = args.phase
    if phase in {"0", "0a", "0b", "0c"}:
        from warp_taskgen.phases import phase_0_recon

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
        from warp_taskgen.phases import phase_0d_auth_bootstrap

        if not args.benchmark:
            print(
                "--benchmark is required for Phase 0d; generator_script paths "
                "in AGENT_CONTEXT resolve relative to the benchmark root.",
                file=sys.stderr,
            )
            return 1
        rc = asyncio.run(phase_0d_auth_bootstrap.run(args))
    elif phase == "1":
        from warp_taskgen.phases import phase_1_tasks

        rc = asyncio.run(phase_1_tasks.run(args))
    elif phase in {"2", "2c"}:
        from warp_taskgen.phase_2 import runner as phase_2_injections

        if phase == "2c":
            args.feasibility_only = True
        rc = asyncio.run(phase_2_injections.run(args))
    elif phase == "3":
        from warp_taskgen.phases import phase_3_benign

        rc = asyncio.run(phase_3_benign.run(args))
    elif phase == "4":
        allow_unknown = getattr(args, "allow_unknown_auth", False)
        instances_for_gate: list[dict[str, object]] | None = None
        instances_path = getattr(args, "instances", None)
        if instances_path is not None and Path(instances_path).exists():
            try:
                config = load_benchmark_config(instances_path)
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
        from warp_taskgen.phase_4 import runner as phase_4_adversarial

        rc = _run_phase4_with_bounded_async_shutdown(
            phase_4_adversarial.run(args),
            shutdown_timeout_s=_phase4_async_shutdown_timeout(),
        )
    else:
        print(f"Unknown phase: {phase}", file=sys.stderr)
        return 1

    # Log final pipeline cost summary if any sandbox calls were recorded.
    if cost_tracker.entries:
        logger = logging.getLogger(__name__)
        logger.info("--- Cost Summary ---\n%s", cost_tracker.summary_report())

    return rc


__all__ = [
    "_dispatch_inspect",
    "_dispatch_phase",
    "_dispatch_phase_with_run_context",
    "_dispatch_preflight",
    "_dispatch_status",
    "_dispatch_task_bank",
    "dispatch_derived_resume",
    "main",
]
