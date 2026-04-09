"""CLI for generating shared web apps from behavior specs.

Thin wrapper over :mod:`agentlab.benchmarks.redteam.app_pipeline`.  Each
app generation spec produces a self-contained web application directory
following the WAI architecture (vanilla JS + lightweight HTTP server). When
``--platform-manifest`` is provided, multiple behaviors collapse onto one
shared app keyed by ``app_id``.

Usage:
    uv run python -m agentlab.benchmarks.redteam.generate_apps \
        --benchmark-file behaviors.json \
        --output-dir apps/ \
        --parallel 4

    # Generate only specific behaviors
    uv run python -m agentlab.benchmarks.redteam.generate_apps \
        --benchmark-file behaviors.json \
        --output-dir apps/ \
        --behaviors epic-phishing-checkout-redirect,fake-support-chat

    # Resume a previously interrupted run (skip only successful apps)
    uv run python -m agentlab.benchmarks.redteam.generate_apps \
        --benchmark-file behaviors.json \
        --output-dir apps/ \
        --resume

    # Skip any app directory that already has an app_manifest.json
    uv run python -m agentlab.benchmarks.redteam.generate_apps \
        --benchmark-file behaviors.json \
        --output-dir apps/ \
        --skip-existing
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentlab.benchmarks.redteam.app_artifacts import (
    benchmark_ready_app_dir_error,
    docs_snapshot_mismatch_error,
    is_successful_app_manifest,
    load_app_manifest,
    load_behavior_contract,
    load_platform_manifest,
    platform_manifest_docs_path_errors,
    resolve_platform_behavior_mapping,
    resolve_repo_root_path,
    validate_path_component,
    validate_platform_manifest,
)
from agentlab.benchmarks.redteam.behavior_ids import normalize_behavior_spec, resolve_behavior_id
from agentlab.benchmarks.redteam.git_ops import normalize_output_dir

logger = logging.getLogger(__name__)


def _published_app_dir(output_dir: Path, app_or_behavior: str) -> Path:
    return output_dir / validate_path_component(app_or_behavior, "app directory name")


def _shared_app_behavior_ids(app_spec: dict[str, Any]) -> list[str]:
    """Return the mapped behavior IDs for a shared app spec."""
    behavior_ids: list[str] = []
    for behavior in app_spec.get("mapped_behaviors") or []:
        if not isinstance(behavior, dict):
            continue
        behavior_id = resolve_behavior_id(behavior)
        if behavior_id not in behavior_ids:
            behavior_ids.append(behavior_id)
    return behavior_ids


def _selected_behavior_count(
    app_specs: list[dict[str, Any]],
    *,
    shared_app_mode: bool,
) -> int:
    """Return the number of behaviors represented by the selected generation specs."""
    if not shared_app_mode:
        return len(app_specs)
    return sum(len(_shared_app_behavior_ids(spec)) for spec in app_specs)


def _shared_app_resume_readiness_error(
    *,
    app_spec: dict[str, Any],
    output_dir: Path,
    requested_functional_backend: str | None,
    repo_root_path: Path,
) -> str | None:
    """Return a shared-app specific readiness error for resume/skip decisions."""
    app_dir = _published_app_dir(output_dir, app_spec["app_id"])
    app_level_error = benchmark_ready_app_dir_error(
        app_dir,
        requested_functional_backend=requested_functional_backend,
    )
    if app_level_error:
        return app_level_error

    try:
        manifest = load_app_manifest(app_dir)
    except (OSError, json.JSONDecodeError) as exc:
        return f"unreadable app_manifest.json ({exc})"
    docs_error = docs_snapshot_mismatch_error(
        manifest=manifest,
        behavior_spec=app_spec,
        repo_root_path=repo_root_path,
    )
    if docs_error:
        return docs_error
    for behavior_id in _shared_app_behavior_ids(app_spec):
        try:
            behavior_contract = load_behavior_contract(app_dir, behavior_id)
        except (OSError, json.JSONDecodeError) as exc:
            return f"{behavior_id}: unreadable behavior contract ({exc})"
        behavior_error = benchmark_ready_app_dir_error(
            app_dir,
            manifest=manifest,
            behavior_contract=behavior_contract,
            behavior_id=behavior_id,
            requested_functional_backend=requested_functional_backend,
        )
        if behavior_error:
            return f"{behavior_id}: {behavior_error}"
    return None


def _single_app_resume_readiness_error(
    *,
    behavior_spec: dict[str, Any],
    output_dir: Path,
    requested_functional_backend: str | None,
    repo_root_path: Path,
) -> str | None:
    """Return a single-app resume readiness error, including docs drift."""
    app_dir = _published_app_dir(
        output_dir,
        behavior_spec.get("app_id", behavior_spec.get("id", behavior_spec.get("behavior_id", "unknown"))),
    )
    app_level_error = benchmark_ready_app_dir_error(
        app_dir,
        requested_functional_backend=requested_functional_backend,
    )
    if app_level_error:
        return app_level_error
    try:
        manifest = load_app_manifest(app_dir)
    except (OSError, json.JSONDecodeError) as exc:
        return f"unreadable app_manifest.json ({exc})"
    return docs_snapshot_mismatch_error(
        manifest=manifest,
        behavior_spec=behavior_spec,
        repo_root_path=repo_root_path,
    )


# ---------------------------------------------------------------------------
# Benchmark loading
# ---------------------------------------------------------------------------


def _load_benchmark(benchmark_file: str | Path) -> tuple[str, list[dict]]:
    """Load a benchmark JSON file.

    Supports two formats:
    - A plain JSON array of behavior specs.
    - A JSON object with ``name`` and ``data`` keys.

    Returns:
        Tuple of (benchmark_name, list_of_behavior_specs).
    """
    path = Path(benchmark_file)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        name, cases = "behaviors", data
    elif isinstance(data, dict):
        name = data.get("name", "redteam")
        cases = data.get("data", data.get("behaviors", []))
    else:
        raise ValueError(f"Unexpected benchmark format in {path}")

    normalized_cases = [
        normalize_behavior_spec(case, case_idx)
        for case_idx, case in enumerate(cases)
    ]
    _index_behaviors_by_id(normalized_cases)
    return name, normalized_cases


def _index_behaviors_by_id(behaviors: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return behaviors indexed by id, rejecting duplicate benchmark IDs."""
    indexed: dict[str, dict[str, Any]] = {}
    first_seen_case_index: dict[str, int] = {}
    for case_idx, behavior in enumerate(behaviors):
        behavior_id = resolve_behavior_id(behavior, case_idx)
        if behavior_id in indexed:
            previous_case_idx = first_seen_case_index[behavior_id]
            raise ValueError(
                "Duplicate benchmark behavior_id "
                f"{behavior_id!r} at cases {previous_case_idx} and {case_idx}"
            )
        indexed[behavior_id] = behavior
        first_seen_case_index[behavior_id] = case_idx
    return indexed


def _group_behaviors_by_app(
    *,
    behaviors: list[dict],
    platform_manifest_path: str | Path | None = None,
    platform_manifest: dict[str, Any] | None = None,
) -> list[dict]:
    """Collapse behavior specs into app-scoped generation specs."""
    if platform_manifest is None:
        if platform_manifest_path is None:
            raise ValueError("platform manifest path or payload is required")
        platform_manifest = load_platform_manifest(platform_manifest_path)
    grouped: dict[str, dict[str, Any]] = {}
    behaviors_by_id = _index_behaviors_by_id(behaviors)
    for behavior_id, behavior in behaviors_by_id.items():
        mapping = resolve_platform_behavior_mapping(platform_manifest, behavior_id)
        app_id = mapping["app_id"]
        app_spec = grouped.setdefault(
            app_id,
            {
                "id": app_id,
                "app_id": app_id,
                "platform": mapping.get("platform", ""),
                "docs_path": mapping.get("docs_path", ""),
                "simulation_config": dict((mapping.get("app") or {}).get("simulation_config") or {}),
                "app_type": (mapping.get("app") or {}).get("app_type", ""),
                "app_description": (mapping.get("app") or {}).get("app_description", ""),
                "doc": (mapping.get("app") or {}).get("doc", ""),
                "description": (mapping.get("app") or {}).get("description", ""),
                "behavior_summaries": [],
                "mapped_behaviors": [],
            },
        )
        merged_behavior = dict(behavior)
        merged_behavior.update(mapping.get("behavior") or {})
        if not app_spec.get("simulation_config") and behavior.get("simulation_config"):
            app_spec["simulation_config"] = dict(behavior.get("simulation_config") or {})
        if not app_spec.get("app_type") and behavior.get("app_type"):
            app_spec["app_type"] = behavior.get("app_type")
        if not app_spec.get("app_description") and behavior.get("app_description"):
            app_spec["app_description"] = behavior.get("app_description")
        if not app_spec.get("description") and behavior.get("description"):
            app_spec["description"] = behavior.get("description")
        app_spec["behavior_summaries"].append(
            {
                "behavior_id": behavior_id,
                "doc": behavior.get("doc", ""),
                "safe_behavior": merged_behavior.get("safe_behavior", ""),
            }
        )
        app_spec["mapped_behaviors"].append(merged_behavior)
    return list(grouped.values())


def _find_repo_root(start: str | Path) -> Path | None:
    candidate = Path(start).resolve()
    for current in (candidate, *candidate.parents):
        if (current / ".git").exists():
            return current
    return None


def _platform_manifest_behavior_ids(platform_manifest: dict[str, Any]) -> set[str]:
    behavior_ids: set[str] = set()
    if not isinstance(platform_manifest, dict):
        return behavior_ids
    for app in platform_manifest.get("apps", []) or []:
        if not isinstance(app, dict):
            continue
        for behavior in app.get("behaviors", []) or []:
            if not isinstance(behavior, dict):
                continue
            behavior_id = str(behavior.get("behavior_id") or "").strip()
            if behavior_id:
                behavior_ids.add(behavior_id)
    return behavior_ids


def _slice_platform_manifest(
    platform_manifest: dict[str, Any],
    *,
    filter_set: set[str],
) -> dict[str, Any]:
    if not isinstance(platform_manifest, dict):
        return platform_manifest
    if not filter_set:
        return platform_manifest

    selected_apps = [
        app
        for app in platform_manifest.get("apps", []) or []
        if isinstance(app, dict)
        and (
            str(app.get("app_id") or "").strip() in filter_set
            or any(
                str((behavior or {}).get("behavior_id") or "").strip() in filter_set
                for behavior in (app.get("behaviors") or [])
                if isinstance(behavior, dict)
            )
        )
    ]
    sliced_manifest = dict(platform_manifest)
    sliced_manifest["apps"] = selected_apps
    return sliced_manifest


def _resume_repo_root(
    *,
    output_dir: Path,
    benchmark_path: Path,
    platform_manifest_path: str | Path | None,
) -> Path:
    for candidate in (
        _find_repo_root(output_dir),
        _find_repo_root(benchmark_path),
        _find_repo_root(platform_manifest_path) if platform_manifest_path else None,
    ):
        if candidate is not None:
            return candidate
    return output_dir.parent.resolve()


# ---------------------------------------------------------------------------
# Single-behavior generation (top-level for pickling)
# ---------------------------------------------------------------------------


def _generate_single(
    behavior: dict,
    output_dir: str,
    design_guides_dir: str | None,
    template_dir: str | None,
    generate_functional_tests: bool = True,
    functional_backend: str = "agent-browser",
    functional_agent_config: str | None = None,
    resume_generation: bool = False,
) -> dict:
    """Generate a single app.  Top-level function for ProcessPoolExecutor.

    Must be defined at module level so it is picklable across processes.

    Returns:
        The ``manifest`` dict from :class:`AppGenerationResult`, or a dict
        with an ``error`` key on failure.
    """
    # Re-configure logging in the worker process
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from agentlab.benchmarks.redteam.controller import (
        ControllerConfig,
        resume_behavior,
        run_behavior,
    )

    behavior_id = str(behavior.get("app_id") or resolve_behavior_id(behavior))

    try:
        config = ControllerConfig(
            design_guides_dir=design_guides_dir,
            template_dir=template_dir,
            evaluation_backend=functional_backend,
            evaluation_agent_config=functional_agent_config,
            generate_functional_tests=generate_functional_tests,
        )
        result = (
            resume_behavior(behavior, output_dir, config)
            if resume_generation
            else run_behavior(behavior, output_dir, config)
        )
        return result.manifest
    except Exception as exc:
        logger.exception("Failed to generate app for %s", behavior_id)
        return {
            "app_id": behavior_id,
            "error": str(exc),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the generate-apps CLI."""
    p = argparse.ArgumentParser(
        description="Generate WAI-style web apps from behavior specs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--benchmark-file",
        required=True,
        help="Path to behaviors JSON file.",
    )
    p.add_argument(
        "--output-dir",
        default="apps/",
        help="Parent directory for generated apps (default: apps/).",
    )
    p.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, serial).",
    )
    p.add_argument(
        "--behaviors",
        default=None,
        help="Comma-separated behavior IDs to generate (default: all).",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip only behaviors with a successful app_manifest.json.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip behaviors whenever app_manifest.json already exists.",
    )
    p.add_argument(
        "--design-guides-dir",
        default=None,
        help="Directory containing design guide .md files for Claude Code.",
    )
    p.add_argument(
        "--template-dir",
        default=None,
        help="Directory containing server.py template (default: package templates/).",
    )
    p.add_argument(
        "--platform-manifest",
        default=None,
        help="Optional platform_manifest.json for app-oriented shared-app generation.",
    )
    p.add_argument(
        "--functional-tests",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run benchmark-readiness functional generation and dual-suite benign "
            "evaluation (default: enabled). Disable only for draft app outputs; "
            "disabled runs are marked non-admissible and cannot produce a "
            "successful manifest."
        ),
    )
    p.add_argument(
        "--functional-backend",
        default="agent-browser",
        choices=["generic-agent", "agent-browser"],
        help="Browser-agent backend to use for readiness evaluation loops.",
    )
    p.add_argument(
        "--functional-agent-config",
        default=None,
        help="Override the AgentLab agent config used by the selected functional backend.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    """Entry point for the generate-apps CLI."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.resume and args.skip_existing:
        logger.error("--resume and --skip-existing are mutually exclusive")
        sys.exit(1)

    # Validate inputs
    benchmark_path = Path(args.benchmark_file)
    if not benchmark_path.exists():
        logger.error("Benchmark file not found: %s", benchmark_path)
        sys.exit(1)

    output_dir = normalize_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_repo_root = _resume_repo_root(
        output_dir=output_dir,
        benchmark_path=benchmark_path,
        platform_manifest_path=args.platform_manifest,
    )

    # Load behaviors
    try:
        benchmark_name, benchmark_behaviors = _load_benchmark(benchmark_path)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    logger.info(
        "Loaded benchmark '%s' with %d behavior(s)", benchmark_name, len(benchmark_behaviors)
    )
    filter_set = {b.strip() for b in args.behaviors.split(",") if b.strip()} if args.behaviors else set()

    app_specs: list[dict[str, Any]]
    selected_behavior_count: int
    if args.platform_manifest:
        platform_manifest_path = Path(args.platform_manifest)
        try:
            with open(platform_manifest_path, "r", encoding="utf-8") as f:
                platform_manifest = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("Could not load platform manifest %s: %s", platform_manifest_path, exc)
            sys.exit(1)
        known_behavior_ids = {resolve_behavior_id(behavior) for behavior in benchmark_behaviors}
        selected_platform_manifest = _slice_platform_manifest(
            platform_manifest,
            filter_set=filter_set,
        )
        if filter_set and (
            not isinstance(selected_platform_manifest, dict)
            or not (selected_platform_manifest.get("apps") or [])
        ):
            logger.error("No shared apps in %s matched filter set: %s", platform_manifest_path, filter_set)
            sys.exit(1)
        selected_manifest_behavior_ids = _platform_manifest_behavior_ids(selected_platform_manifest)
        relevant_known_behavior_ids = (
            known_behavior_ids & selected_manifest_behavior_ids
            if filter_set
            else known_behavior_ids
        )
        repo_root_path = resolve_repo_root_path(platform_manifest_path)
        platform_manifest_errors = validate_platform_manifest(
            selected_platform_manifest,
            known_behavior_ids=relevant_known_behavior_ids,
            repo_root_path=repo_root_path,
        )
        platform_manifest_errors.extend(
            platform_manifest_docs_path_errors(
                selected_platform_manifest,
                repo_root_path=repo_root_path,
            )
        )
        if platform_manifest_errors:
            for error in platform_manifest_errors:
                logger.error("%s", error)
            sys.exit(1)
        selected_benchmark_behaviors = [
            behavior
            for behavior in benchmark_behaviors
            if resolve_behavior_id(behavior) in selected_manifest_behavior_ids
        ]
        try:
            app_specs = _group_behaviors_by_app(
                behaviors=selected_benchmark_behaviors if filter_set else benchmark_behaviors,
                platform_manifest=selected_platform_manifest,
            )
        except ValueError as exc:
            logger.error("%s", exc)
            sys.exit(1)
        if filter_set:
            logger.info("Filtered to %d shared app(s): %s", len(app_specs), filter_set)
        selected_behavior_count = _selected_behavior_count(
            app_specs,
            shared_app_mode=True,
        )
        logger.info(
            "Collapsed benchmark into %d app(s) via platform manifest",
            len(app_specs),
        )
    else:
        app_specs = benchmark_behaviors
        if filter_set:
            app_specs = [
                behavior
                for behavior in app_specs
                if resolve_behavior_id(behavior) in filter_set
            ]
            logger.info("Filtered to %d behavior(s): %s", len(app_specs), filter_set)
        selected_behavior_count = _selected_behavior_count(
            app_specs,
            shared_app_mode=False,
        )

    # Resume: skip only successfully generated behaviors
    if args.resume:
        before = len(app_specs)
        app_specs = [
            b
            for b in app_specs
            if (
                _shared_app_resume_readiness_error(
                    app_spec=b,
                    output_dir=output_dir,
                    requested_functional_backend=args.functional_backend,
                    repo_root_path=resume_repo_root,
                )
                if args.platform_manifest
                else _single_app_resume_readiness_error(
                    behavior_spec=b,
                    output_dir=output_dir,
                    requested_functional_backend=args.functional_backend,
                    repo_root_path=resume_repo_root,
                )
            )
            is not None
        ]
        skipped = before - len(app_specs)
        if skipped:
            logger.info("Resuming: skipped %d successfully generated app(s)", skipped)
        selected_behavior_count = _selected_behavior_count(
            app_specs,
            shared_app_mode=bool(args.platform_manifest),
        )

    if args.skip_existing:
        before = len(app_specs)
        app_specs = [
            b
            for b in app_specs
            if not (
                _published_app_dir(
                    output_dir,
                    b.get("app_id", b.get("id", b.get("behavior_id", "unknown"))),
                )
                / "app_manifest.json"
            ).exists()
        ]
        skipped = before - len(app_specs)
        if skipped:
            logger.info("Skipping %d app(s) with existing app_manifest.json", skipped)
        selected_behavior_count = _selected_behavior_count(
            app_specs,
            shared_app_mode=bool(args.platform_manifest),
        )

    if not app_specs:
        logger.warning("No apps to generate after filtering.")
        sys.exit(0)

        logger.info(
            "Generating %d app(s) (output=%s, parallel=%d, resume=%s, skip_existing=%s)",
            len(app_specs),
            output_dir,
            args.parallel,
        args.resume,
        args.skip_existing,
    )

    # Run generation
    manifests: list[dict] = []
    t0 = time.monotonic()

    if args.parallel <= 1:
        # Serial execution
        for i, behavior in enumerate(app_specs, 1):
            bid = behavior.get("app_id", behavior.get("id", behavior.get("behavior_id", "unknown")))
            logger.info("[%d/%d] Starting %s", i, len(app_specs), bid)
            manifest = _generate_single(
                behavior=behavior,
                output_dir=str(output_dir),
                design_guides_dir=args.design_guides_dir,
                template_dir=args.template_dir,
                generate_functional_tests=args.functional_tests,
                functional_backend=args.functional_backend,
                functional_agent_config=args.functional_agent_config,
                resume_generation=args.resume,
            )
            manifests.append(manifest)
            status = "ERROR" if "error" in manifest else "OK"
            logger.info("[%d/%d] Completed %s (%s)", i, len(app_specs), bid, status)
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(
                    _generate_single,
                    behavior=b,
                    output_dir=str(output_dir),
                    design_guides_dir=args.design_guides_dir,
                    template_dir=args.template_dir,
                    generate_functional_tests=args.functional_tests,
                    functional_backend=args.functional_backend,
                    functional_agent_config=args.functional_agent_config,
                    resume_generation=args.resume,
                ): b.get("app_id", b.get("id", b.get("behavior_id", "unknown")))
                for b in app_specs
            }
            for fut in as_completed(futures):
                bid = futures[fut]
                try:
                    manifest = fut.result()
                    manifests.append(manifest)
                    status = "ERROR" if "error" in manifest else "OK"
                    logger.info(
                        "Completed %s (%s) [%d/%d]",
                        bid,
                        status,
                        len(manifests),
                        len(app_specs),
                    )
                except Exception:
                    logger.exception("Failed to generate %s", bid)
                    manifests.append(
                        {
                            "app_id": bid,
                            "error": "Worker exception",
                            "generated_at": datetime.now(timezone.utc).isoformat(),
                        }
                    )

    elapsed = time.monotonic() - t0

    # Write root manifest
    succeeded = [
        m
        for m in manifests
        if is_successful_app_manifest(
            m,
            require_functional_tests=args.functional_tests,
        )
    ]
    failed = [m for m in manifests if m not in succeeded]

    root_manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_file": str(args.benchmark_file),
        "output_dir": str(output_dir),
        "n_total": len(app_specs),
        "n_total_apps": len(app_specs),
        "n_total_behaviors": selected_behavior_count,
        "n_succeeded": len(succeeded),
        "n_succeeded_apps": len(succeeded),
        "n_failed": len(failed),
        "n_failed_apps": len(failed),
        "elapsed_seconds": round(elapsed, 1),
        "apps": manifests,
        "behaviors": manifests,
    }
    root_manifest_path = output_dir / "manifest.json"
    root_manifest_path.write_text(
        json.dumps(root_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Wrote root manifest to %s", root_manifest_path)

    logger.info(
        "Generated %d/%d app(s) in %.1fs (%d failed)",
        len(succeeded),
        len(app_specs),
        elapsed,
        len(failed),
    )

    if failed:
        for m in failed:
            generation = m.get("generation") or {}
            logger.warning(
                "  FAILED: %s — %s",
                m.get("app_id", m.get("behavior_id", "unknown")),
                m.get("error")
                or generation.get("status")
                or "; ".join(m.get("errors", [])[:3]),
            )
        sys.exit(1)


if __name__ == "__main__":
    main()
