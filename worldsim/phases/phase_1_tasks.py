"""Phase 1 orchestration: Mode A wrapping plus optional Mode B generation."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import logging
from pathlib import Path
from typing import Any

from worldsim.cost_tracker import tracker as cost_tracker
from worldsim.state import get_state_dir, save_state

from worldsim.phases.phase_1_mode_a import build_mode_a_tasks
from worldsim.phases.phase_1_mode_b import (
    DEFAULT_NOVEL_TASKS_PER_SITE,
    EligibleSiteProfile,
    MODE_B_RESUME_METADATA_PATH,
    NOVEL_TASK_OUTPUT_PATH,
    SiteNovelTaskResult,
    generate_novel_tasks_for_site as _generate_novel_tasks_for_site,
    load_existing_novel_tasks as _load_existing_novel_tasks,
    load_mode_b_eligible_sites as _load_mode_b_eligible_sites,
    run_mode_b,
    validate_existing_novel_tasks,
)
from worldsim.phases.phase_1_mode_b_validation import (
    merge_benign_tasks as _merge_benign_tasks,
    site_is_mode_b_eligible as _site_is_mode_b_eligible,
    sort_novel_tasks as _sort_novel_tasks,
    validate_generated_novel_task as _validate_generated_novel_task,
    validate_generated_novel_tasks as _validate_generated_novel_tasks,
)

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

    manifest_path = args.config or (state_dir / "phase_0a" / "BENCHMARK_MANIFEST.json")
    try:
        manifest = _load_manifest(Path(manifest_path))
    except ValueError as exc:
        logger.error("%s", exc)
        _save_phase_1_failure_state(
            reason="invalid_manifest",
            benchmark_root=Path(getattr(args, "benchmark", "")) if getattr(args, "benchmark", None) else None,
            manifest_path=Path(manifest_path),
            generate_novel=generate_novel,
            error=str(exc),
        )
        return 1
    if manifest is None:
        logger.error("Manifest not found at %s — run phase 0a first", manifest_path)
        _save_phase_1_failure_state(
            reason="missing_manifest",
            benchmark_root=Path(getattr(args, "benchmark", "")) if getattr(args, "benchmark", None) else None,
            manifest_path=Path(manifest_path),
            generate_novel=generate_novel,
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
            benchmark_root=Path(getattr(args, "benchmark", "")) if getattr(args, "benchmark", None) else None,
            manifest_path=Path(manifest_path),
            generate_novel=generate_novel,
        )
        return 1

    output_path = output_dir / "benign_tasks.json"
    resume_metadata_path = output_dir / MODE_B_RESUME_METADATA_PATH

    _save_phase_1_running_state(
        benchmark_root=benchmark_root,
        manifest_path=Path(manifest_path),
        generate_novel=generate_novel,
    )

    mode_a_tasks = build_mode_a_tasks(manifest, benchmark_root)
    if not mode_a_tasks:
        logger.error("No tasks found in benchmark — check manifest task_definition_paths")
        _save_phase_1_failure_state(
            reason="no_raw_tasks",
            benchmark_root=benchmark_root,
            manifest_path=Path(manifest_path),
            generate_novel=generate_novel,
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
        )
        if novel_tasks is None:
            return 1

    benign_tasks = _merge_benign_tasks(mode_a_tasks, novel_tasks)
    output_path.write_text(json.dumps(benign_tasks, indent=2))
    _write_mode_b_resume_metadata(
        resume_metadata_path,
        benchmark_root=benchmark_root,
        manifest_path=Path(manifest_path),
    )

    save_state(
        "phase_1",
        status="complete",
        tasks_path=str(output_path),
        task_count=len(benign_tasks),
        mode_a_task_count=len(mode_a_tasks),
        novel_task_count=len(novel_tasks),
        benchmark_path=str(benchmark_root),
        manifest_path=str(manifest_path),
        generate_novel=generate_novel,
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


def _save_phase_1_running_state(
    *,
    benchmark_root: Path,
    manifest_path: Path,
    generate_novel: bool,
) -> None:
    """Persist the start of Phase 1 for resume support."""
    save_state(
        "phase_1",
        status="running",
        benchmark_path=str(benchmark_root),
        manifest_path=str(manifest_path),
        generate_novel=generate_novel,
    )


def _save_phase_1_failure_state(
    *,
    reason: str,
    benchmark_root: Path | None,
    manifest_path: Path,
    generate_novel: bool,
    mode_a_task_count: int | None = None,
    error: str | None = None,
) -> None:
    """Persist a failed Phase 1 state."""
    payload: dict[str, Any] = {
        "status": "failed",
        "reason": reason,
        "manifest_path": str(manifest_path),
        "generate_novel": generate_novel,
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
) -> list[dict[str, Any]] | None:
    """Return Mode B tasks, reusing merged output when already present."""
    existing_novel_tasks = _reuse_existing_novel_tasks_if_valid(
        manifest=manifest,
        benchmark_root=benchmark_root,
        output_path=output_path,
        resume_metadata_path=benchmark_resume_metadata_path,
        manifest_path=manifest_path,
        resume=resume,
    )
    if existing_novel_tasks is not None:
        return existing_novel_tasks

    try:
        return await run_mode_b(
            manifest=manifest,
            benchmark_root=benchmark_root,
            output_dir=output_dir,
        )
    except Exception as exc:
        _save_phase_1_failure_state(
            reason="mode_b_generation_failed",
            error=str(exc),
            benchmark_root=benchmark_root,
            manifest_path=manifest_path,
            generate_novel=True,
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
    manifest_path: Path,
    resume: bool,
) -> list[dict[str, Any]] | None:
    """Reuse merged Mode B output only on resume and only after provenance checks."""
    if not resume:
        return None

    existing_novel_tasks = _load_existing_novel_tasks(output_path)
    if existing_novel_tasks is None:
        return None

    metadata = _load_mode_b_resume_metadata(resume_metadata_path)
    current_fingerprint = _mode_b_resume_fingerprint(
        benchmark_root=benchmark_root,
        manifest_path=manifest_path,
    )
    if metadata.get("fingerprint") != current_fingerprint:
        logger.warning(
            "Phase 1B: ignoring merged novel-task output because resume metadata does not match current inputs"
        )
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

    logger.info(
        "Phase 1B: merged output already contains %d valid novel tasks, skipping generation on resume",
        len(existing_novel_tasks),
    )
    return existing_novel_tasks


def _write_mode_b_resume_metadata(
    metadata_path: Path,
    *,
    benchmark_root: Path,
    manifest_path: Path,
) -> None:
    payload = {
        "fingerprint": _mode_b_resume_fingerprint(
            benchmark_root=benchmark_root,
            manifest_path=manifest_path,
        ),
        "benchmark_path": str(benchmark_root),
        "manifest_path": str(manifest_path),
    }
    metadata_path.write_text(json.dumps(payload, indent=2))


def _load_mode_b_resume_metadata(metadata_path: Path) -> dict[str, Any]:
    if not metadata_path.exists():
        return {}
    try:
        payload = json.loads(metadata_path.read_text())
    except json.JSONDecodeError:
        logger.warning("Phase 1B: ignoring invalid resume metadata at %s", metadata_path)
        return {}
    return payload if isinstance(payload, dict) else {}


def _mode_b_resume_fingerprint(*, benchmark_root: Path, manifest_path: Path) -> str:
    return sha256(
        json.dumps(
            {
                "benchmark_path": str(benchmark_root.resolve()),
                "manifest_path": str(manifest_path.resolve()),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
