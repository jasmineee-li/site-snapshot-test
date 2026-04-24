"""Phase 1 orchestration: Mode A wrapping plus optional Mode B generation."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from worldsim.benchmark_capabilities import get_benchmark_capabilities, infer_benchmark_name
from worldsim.cost_tracker import tracker as cost_tracker
from worldsim.phases.phase_1_mode_a import build_mode_a_tasks
from worldsim.phases.phase_1_mode_b import (
    MODE_B_RESUME_METADATA_PATH,
    EligibleSiteProfile,
    compute_mode_b_resume_fingerprint,
    compute_mode_b_shared_inputs_fingerprint,
    compute_site_cache_fingerprint,
    run_mode_b,
    validate_existing_novel_tasks,
)
from worldsim.phases.phase_1_mode_b import (
    load_cached_novel_tasks as _load_cached_novel_tasks,
)
from worldsim.phases.phase_1_mode_b import (
    load_existing_novel_tasks as _load_existing_novel_tasks,
)
from worldsim.phases.phase_1_mode_b import (
    load_mode_b_eligible_sites as _load_mode_b_eligible_sites,
)
from worldsim.phases.phase_1_mode_b_validation import (
    merge_benign_tasks as _merge_benign_tasks,
)
from worldsim.phases.phase_1_mode_b_validation import (
    sort_novel_tasks as _sort_novel_tasks,
)
from worldsim.state import get_state_dir, save_state

logger = logging.getLogger(__name__)


async def run(args: argparse.Namespace) -> int:
    """Phase 1 entrypoint.

    Reads the benchmark's task definitions, wraps them into benign bundles,
    and optionally generates Mode B novel tasks for eligible sites.
    """
    state_dir = get_state_dir()
    output_dir = state_dir / "phase_1"
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_novel = bool(getattr(args, "generate_novel", False))
    sandbox_model = getattr(args, "sandbox_model", None) or "claude-sonnet-4-6"

    manifest_path = args.config or (state_dir / "phase_0a" / "BENCHMARK_MANIFEST.json")
    try:
        manifest = _load_manifest(Path(manifest_path))
    except ValueError as exc:
        logger.error("%s", exc)
        _save_phase_1_failure_state(
            reason="invalid_manifest",
            benchmark_root=Path(getattr(args, "benchmark", ""))
            if getattr(args, "benchmark", None)
            else None,
            manifest_path=Path(manifest_path),
            generate_novel=generate_novel,
            sandbox_model=sandbox_model,
            error=str(exc),
        )
        return 1
    if manifest is None:
        logger.error("Manifest not found at %s — run phase 0a first", manifest_path)
        _save_phase_1_failure_state(
            reason="missing_manifest",
            benchmark_root=Path(getattr(args, "benchmark", ""))
            if getattr(args, "benchmark", None)
            else None,
            manifest_path=Path(manifest_path),
            generate_novel=generate_novel,
            sandbox_model=sandbox_model,
        )
        return 1

    benchmark_root = _resolve_benchmark_root(args, manifest)
    if benchmark_root is None:
        logger.error(
            "Benchmark root not found: %s — pass --benchmark",
            Path(getattr(args, "benchmark", "")) if getattr(args, "benchmark", None) else "",
        )
        _save_phase_1_failure_state(
            reason="missing_benchmark_root",
            benchmark_root=Path(getattr(args, "benchmark", ""))
            if getattr(args, "benchmark", None)
            else None,
            manifest_path=Path(manifest_path),
            generate_novel=generate_novel,
            sandbox_model=sandbox_model,
        )
        return 1
    try:
        benchmark_name = _manifest_benchmark_name(manifest)
    except ValueError as exc:
        logger.error("Phase 1 benchmark gate failed: %s", exc)
        _save_phase_1_failure_state(
            reason="unsupported_benchmark",
            benchmark_root=benchmark_root,
            manifest_path=Path(manifest_path),
            generate_novel=generate_novel,
            sandbox_model=sandbox_model,
            error=str(exc),
        )
        return 1

    output_path = output_dir / "benign_tasks.json"
    resume_metadata_path = output_dir / MODE_B_RESUME_METADATA_PATH

    _save_phase_1_running_state(
        benchmark_root=benchmark_root,
        manifest_path=Path(manifest_path),
        benchmark_name=benchmark_name,
        generate_novel=generate_novel,
        sandbox_model=sandbox_model,
    )

    profiles_dir = get_state_dir() / "phase_0c"
    mode_a_tasks = build_mode_a_tasks(
        manifest,
        benchmark_root,
        profiles_dir=profiles_dir,
        benchmark=benchmark_name,
    )
    if not mode_a_tasks:
        logger.error("No tasks found in benchmark — check manifest task_definition_paths")
        _save_phase_1_failure_state(
            reason="no_raw_tasks",
            benchmark_root=benchmark_root,
            manifest_path=Path(manifest_path),
            generate_novel=generate_novel,
            sandbox_model=sandbox_model,
        )
        return 1

    logger.info("Phase 1A: wrapped %d raw tasks from benchmark", len(mode_a_tasks))

    novel_tasks: list[dict[str, Any]] = []
    if generate_novel:
        novel_tasks = await _maybe_generate_mode_b_tasks(
            manifest=manifest,
            benchmark_root=benchmark_root,
            output_dir=output_dir,
            output_path=output_path,
            manifest_path=Path(manifest_path),
            mode_a_task_count=len(mode_a_tasks),
            benchmark_resume_metadata_path=resume_metadata_path,
            resume=bool(getattr(args, "resume", False)),
            sandbox_model=sandbox_model,
        )
        if novel_tasks is None:
            return 1
        novel_tasks = _stamp_benchmark_metadata(novel_tasks, benchmark_name)

    benign_tasks = _merge_benign_tasks(mode_a_tasks, novel_tasks)
    output_path.write_text(json.dumps(benign_tasks, indent=2))
    if generate_novel:
        _write_mode_b_resume_metadata(
            resume_metadata_path,
            benchmark_root=benchmark_root,
            manifest=manifest,
            sandbox_model=sandbox_model,
        )

    save_state(
        "phase_1",
        status="complete",
        tasks_path=str(output_path),
        task_count=len(benign_tasks),
        mode_a_task_count=len(mode_a_tasks),
        novel_task_count=len(novel_tasks),
        benchmark_path=str(benchmark_root),
        benchmark_name=benchmark_name,
        manifest_path=str(manifest_path),
        generate_novel=generate_novel,
        sandbox_model=sandbox_model,
    )
    cost_tracker.log_phase_summary("phase_1")
    cost_tracker.save(state_dir / "cost_report.json")
    logger.info(
        "Phase 1 complete — %d benign tasks written to %s (%d Mode A + %d novel)",
        len(benign_tasks),
        output_path,
        len(mode_a_tasks),
        len(novel_tasks),
    )
    return 0


def _load_manifest(manifest_path: Path) -> dict[str, Any] | None:
    """Return the Phase 0a manifest when present."""
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Manifest at {manifest_path} is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest at {manifest_path} must be a JSON object")
    return manifest


def _resolve_benchmark_root(
    args: argparse.Namespace,
    manifest: dict[str, Any],
) -> Path | None:
    """Resolve the benchmark root from CLI args or the Phase 0a manifest."""
    benchmark_root = (
        Path(args.benchmark)
        if getattr(args, "benchmark", None)
        else Path(manifest.get("benchmark_codebase", ""))
    )
    if not benchmark_root.is_dir():
        return None
    return benchmark_root


def _manifest_benchmark_name(manifest: dict[str, Any]) -> str:
    benchmark_name = infer_benchmark_name(
        manifest.get(key) for key in ("benchmark_name", "benchmark", "benchmark_adapter")
    )
    if benchmark_name is None:
        raise ValueError("Phase 0a manifest is missing benchmark metadata")
    capabilities = get_benchmark_capabilities(benchmark_name)
    if not capabilities.phase_1_supported:
        raise ValueError(f"benchmark {capabilities.canonical_name!r} does not support Phase 1")
    return capabilities.canonical_name


def _stamp_benchmark_metadata(
    tasks: list[dict[str, Any]],
    benchmark_name: str,
) -> list[dict[str, Any]]:
    stamped: list[dict[str, Any]] = []
    for task in tasks:
        item = json.loads(json.dumps(task))
        item["benchmark"] = benchmark_name
        stamped.append(item)
    return stamped


def _save_phase_1_running_state(
    *,
    benchmark_root: Path,
    manifest_path: Path,
    benchmark_name: str,
    generate_novel: bool,
    sandbox_model: str,
) -> None:
    """Persist the start of Phase 1 for resume support."""
    save_state(
        "phase_1",
        status="running",
        benchmark_path=str(benchmark_root),
        benchmark_name=benchmark_name,
        manifest_path=str(manifest_path),
        generate_novel=generate_novel,
        sandbox_model=sandbox_model,
    )


def _save_phase_1_failure_state(
    *,
    reason: str,
    benchmark_root: Path | None,
    manifest_path: Path,
    generate_novel: bool,
    sandbox_model: str,
    mode_a_task_count: int | None = None,
    error: str | None = None,
) -> None:
    """Persist a failed Phase 1 state."""
    payload: dict[str, Any] = {
        "status": "failed",
        "reason": reason,
        "manifest_path": str(manifest_path),
        "generate_novel": generate_novel,
        "sandbox_model": sandbox_model,
    }
    if benchmark_root is not None:
        payload["benchmark_path"] = str(benchmark_root)
    if mode_a_task_count is not None:
        payload["mode_a_task_count"] = mode_a_task_count
    if error is not None:
        payload["error"] = error
    save_state("phase_1", **payload)


async def _maybe_generate_mode_b_tasks(
    *,
    manifest: dict[str, Any],
    benchmark_root: Path,
    output_dir: Path,
    output_path: Path,
    manifest_path: Path,
    mode_a_task_count: int,
    benchmark_resume_metadata_path: Path,
    resume: bool,
    sandbox_model: str,
) -> list[dict[str, Any]] | None:
    """Return Mode B tasks, reusing merged output when already present."""
    existing_novel_tasks = _reuse_existing_novel_tasks_if_valid(
        manifest=manifest,
        benchmark_root=benchmark_root,
        output_path=output_path,
        resume_metadata_path=benchmark_resume_metadata_path,
        resume=resume,
        sandbox_model=sandbox_model,
    )
    if existing_novel_tasks is not None:
        return existing_novel_tasks

    try:
        return await run_mode_b(
            manifest=manifest,
            benchmark_root=benchmark_root,
            output_dir=output_dir,
            sandbox_model=sandbox_model,
        )
    except Exception as exc:
        _save_phase_1_failure_state(
            reason="mode_b_generation_failed",
            error=str(exc),
            benchmark_root=benchmark_root,
            manifest_path=manifest_path,
            generate_novel=True,
            sandbox_model=sandbox_model,
            mode_a_task_count=mode_a_task_count,
        )
        logger.error("Phase 1B failed: %s", exc)
        return None


def _reuse_existing_novel_tasks_if_valid(
    *,
    manifest: dict[str, Any],
    benchmark_root: Path,
    output_path: Path,
    resume_metadata_path: Path,
    resume: bool,
    sandbox_model: str,
) -> list[dict[str, Any]] | None:
    """Reuse merged Mode B output only on resume and only after provenance checks."""
    if not resume:
        return None

    existing_novel_tasks = _load_existing_novel_tasks(output_path)
    if existing_novel_tasks is None:
        return None

    eligible_sites = _load_mode_b_eligible_sites(
        profiles_dir=get_state_dir() / "phase_0c",
        manifest_eval_types=manifest.get("evaluation", {}).get("eval_types", []),
    )
    validation_errors = validate_existing_novel_tasks(
        existing_novel_tasks,
        eligible_sites=eligible_sites,
    )
    if validation_errors:
        logger.warning(
            "Phase 1B: ignoring merged novel-task output because cached tasks are invalid: %s",
            "; ".join(validation_errors),
        )
        return None

    shared_inputs_fingerprint = compute_mode_b_shared_inputs_fingerprint(
        benchmark_root=benchmark_root,
        manifest=manifest,
        sandbox_model=sandbox_model,
    )
    current_fingerprint = compute_mode_b_resume_fingerprint(
        shared_inputs_fingerprint=shared_inputs_fingerprint,
        eligible_sites=eligible_sites,
    )

    metadata = _load_mode_b_resume_metadata(resume_metadata_path)
    if metadata.get("fingerprint") == current_fingerprint:
        logger.info(
            "Phase 1B: merged output already contains %d valid novel tasks, skipping generation on resume",
            len(existing_novel_tasks),
        )
        return existing_novel_tasks

    if _merged_output_matches_current_site_caches(
        output_dir=output_path.parent,
        existing_novel_tasks=existing_novel_tasks,
        eligible_sites=eligible_sites,
        shared_inputs_fingerprint=shared_inputs_fingerprint,
    ):
        logger.info(
            "Phase 1B: merged output already contains %d valid novel tasks and matches current per-site caches, skipping generation on resume",
            len(existing_novel_tasks),
        )
        return existing_novel_tasks

    logger.warning(
        "Phase 1B: ignoring merged novel-task output because resume provenance does not match current inputs"
    )
    return None


def _write_mode_b_resume_metadata(
    metadata_path: Path,
    *,
    benchmark_root: Path,
    manifest: dict[str, Any],
    sandbox_model: str,
) -> None:
    eligible_sites = _load_mode_b_eligible_sites(
        profiles_dir=get_state_dir() / "phase_0c",
        manifest_eval_types=manifest.get("evaluation", {}).get("eval_types", []),
    )
    payload = {
        "fingerprint": compute_mode_b_resume_fingerprint(
            shared_inputs_fingerprint=compute_mode_b_shared_inputs_fingerprint(
                benchmark_root=benchmark_root,
                manifest=manifest,
                sandbox_model=sandbox_model,
            ),
            eligible_sites=eligible_sites,
        ),
        "benchmark_path": str(benchmark_root),
        "eligible_sites": [site.site_name for site in eligible_sites],
        "sandbox_model": sandbox_model,
    }
    metadata_path.write_text(json.dumps(payload, indent=2))


def _merged_output_matches_current_site_caches(
    *,
    output_dir: Path,
    existing_novel_tasks: list[dict[str, Any]],
    eligible_sites: list[EligibleSiteProfile],
    shared_inputs_fingerprint: str,
) -> bool:
    cached_tasks: list[dict[str, Any]] = []
    for site in eligible_sites:
        cached_result = _load_cached_novel_tasks(
            intermediate_path=output_dir / f"novel_tasks_{site.site_name}.json",
            site_name=site.site_name,
            profile=site.profile,
            cache_fingerprint=compute_site_cache_fingerprint(
                shared_inputs_fingerprint=shared_inputs_fingerprint,
                site=site,
            ),
        )
        if cached_result is None:
            return False
        cached_tasks.extend(cached_result.benign_tasks)

    return _sort_novel_tasks(cached_tasks) == _sort_novel_tasks(existing_novel_tasks)


def _load_mode_b_resume_metadata(metadata_path: Path) -> dict[str, Any]:
    if not metadata_path.exists():
        return {}
    try:
        payload = json.loads(metadata_path.read_text())
    except json.JSONDecodeError:
        logger.warning("Phase 1B: ignoring invalid resume metadata at %s", metadata_path)
        return {}
    return payload if isinstance(payload, dict) else {}
