"""Phase 1 orchestration: existing-task wrapping plus optional new-task generation."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from worldsim.benchmark_capabilities import get_benchmark_capabilities, infer_benchmark_name
from worldsim.cost_tracker import tracker as cost_tracker
from worldsim.phases.phase_1_existing_tasks import build_existing_task_wraps
from worldsim.phases.phase_1_generate_new_tasks import (
    DEFAULT_NOVEL_TASKS_PER_SITE,
    GENERATE_NEW_TASKS_RESUME_METADATA_PATH,
    EligibleSiteProfile,
    compute_generate_new_tasks_resume_fingerprint,
    compute_generate_new_tasks_shared_inputs_fingerprint,
    compute_site_cache_fingerprint,
    run_generate_new_tasks,
    validate_existing_novel_tasks,
)
from worldsim.phases.phase_1_generate_new_tasks import (
    load_cached_novel_tasks as _load_cached_novel_tasks,
)
from worldsim.phases.phase_1_generate_new_tasks import (
    load_existing_novel_tasks as _load_existing_novel_tasks,
)
from worldsim.phases.phase_1_generate_new_tasks import (
    load_generate_new_tasks_eligible_sites as _load_generate_new_tasks_eligible_sites,
)
from worldsim.phases.phase_1_generate_new_tasks_validation import (
    merge_benign_tasks as _merge_benign_tasks,
)
from worldsim.phases.phase_1_generate_new_tasks_validation import (
    sort_novel_tasks as _sort_novel_tasks,
)
from worldsim.phases.phase_1_task_cards import (
    load_or_compile_task_card_plan,
    task_card_plan_digest,
)
from worldsim.state import get_state_dir, save_state

logger = logging.getLogger(__name__)


async def run(args: argparse.Namespace) -> int:
    """Phase 1 entrypoint.

    Reads the benchmark's task definitions, wraps them into benign bundles,
    and optionally generates novel tasks for eligible sites.
    """
    state_dir = get_state_dir()
    output_dir = state_dir / "phase_1"
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_novel = bool(getattr(args, "generate_novel", False))
    sandbox_model = getattr(args, "sandbox_model", None) or "claude-sonnet-4-6"
    novel_tasks_per_site = (
        getattr(args, "novel_tasks_per_site", None) or DEFAULT_NOVEL_TASKS_PER_SITE
    )
    task_card_plan_arg = getattr(args, "task_card_plan", None)
    task_card_plan_path = Path(task_card_plan_arg) if task_card_plan_arg else None
    task_capability_profile = getattr(args, "task_capability_profile", None)
    try:
        action_counts = _parse_phase_1_action_counts(getattr(args, "phase_1_action_counts", None))
    except ValueError as exc:
        logger.error("Phase 1 action-count gate failed: %s", exc)
        return 1
    sites_filter_raw = getattr(args, "sites", None)
    sites_filter = _parse_sites_filter(sites_filter_raw)

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
            sites=sites_filter_raw,
            novel_tasks_per_site=novel_tasks_per_site,
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
            sites=sites_filter_raw,
            novel_tasks_per_site=novel_tasks_per_site,
        )
        return 1
    try:
        task_card_plan = load_or_compile_task_card_plan(
            path=task_card_plan_path,
            task_capability_profile=task_capability_profile,
            sites=sites_filter,
        )
    except ValueError as exc:
        logger.error("Phase 1 task-card plan gate failed: %s", exc)
        _save_phase_1_failure_state(
            reason="invalid_task_card_plan",
            benchmark_root=Path(getattr(args, "benchmark", ""))
            if getattr(args, "benchmark", None)
            else None,
            manifest_path=Path(manifest_path),
            generate_novel=generate_novel,
            sandbox_model=sandbox_model,
            sites=sites_filter_raw,
            novel_tasks_per_site=novel_tasks_per_site,
            task_card_plan_path=task_card_plan_path,
            task_capability_profile=task_capability_profile,
            task_card_plan_digest_value=None,
            error=str(exc),
        )
        return 1
    task_card_digest = task_card_plan_digest(task_card_plan)

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
            sites=sites_filter_raw,
            novel_tasks_per_site=novel_tasks_per_site,
            task_card_plan_path=task_card_plan_path,
            task_capability_profile=task_capability_profile,
            task_card_plan_digest_value=task_card_digest,
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
            sites=sites_filter_raw,
            novel_tasks_per_site=novel_tasks_per_site,
            task_card_plan_path=task_card_plan_path,
            task_capability_profile=task_capability_profile,
            task_card_plan_digest_value=task_card_digest,
            error=str(exc),
        )
        return 1

    output_path = output_dir / "benign_tasks.json"
    resume_metadata_path = output_dir / GENERATE_NEW_TASKS_RESUME_METADATA_PATH

    _save_phase_1_running_state(
        benchmark_root=benchmark_root,
        manifest_path=Path(manifest_path),
        benchmark_name=benchmark_name,
        generate_novel=generate_novel,
        sandbox_model=sandbox_model,
        sites=sites_filter_raw,
        novel_tasks_per_site=novel_tasks_per_site,
        task_card_plan_path=task_card_plan_path,
        task_capability_profile=task_capability_profile,
        task_card_plan_digest_value=task_card_digest,
    )

    profiles_dir = get_state_dir() / "phase_0c"
    existing_task_wraps = build_existing_task_wraps(
        manifest,
        benchmark_root,
        profiles_dir=profiles_dir,
        benchmark=benchmark_name,
    )
    if not existing_task_wraps:
        logger.error("No tasks found in benchmark — check manifest task_definition_paths")
        _save_phase_1_failure_state(
            reason="no_raw_tasks",
            benchmark_root=benchmark_root,
            manifest_path=Path(manifest_path),
            generate_novel=generate_novel,
            sandbox_model=sandbox_model,
            sites=sites_filter_raw,
            novel_tasks_per_site=novel_tasks_per_site,
            task_card_plan_path=task_card_plan_path,
            task_capability_profile=task_capability_profile,
            task_card_plan_digest_value=task_card_digest,
        )
        return 1

    logger.info("Phase 1A: wrapped %d raw tasks from benchmark", len(existing_task_wraps))

    novel_tasks: list[dict[str, Any]] = []
    if generate_novel:
        novel_tasks = await _maybe_generate_new_tasks(
            manifest=manifest,
            benchmark_root=benchmark_root,
            output_dir=output_dir,
            output_path=output_path,
            manifest_path=Path(manifest_path),
            existing_task_count=len(existing_task_wraps),
            benchmark_resume_metadata_path=resume_metadata_path,
            resume=bool(getattr(args, "resume", False)),
            sandbox_model=sandbox_model,
            site_filter=sites_filter,
            novel_tasks_per_site=novel_tasks_per_site,
            task_card_plan=task_card_plan,
            action_counts=action_counts,
        )
        if novel_tasks is None:
            return 1
        novel_tasks = _stamp_benchmark_metadata(novel_tasks, benchmark_name)

    benign_tasks = _merge_benign_tasks(existing_task_wraps, novel_tasks)
    output_path.write_text(json.dumps(benign_tasks, indent=2))
    if generate_novel:
        _write_generate_new_tasks_resume_metadata(
            resume_metadata_path,
            benchmark_root=benchmark_root,
            manifest=manifest,
            sandbox_model=sandbox_model,
            site_filter=sites_filter,
            novel_tasks_per_site=novel_tasks_per_site,
            task_card_plan=task_card_plan,
            task_capability_profile=task_capability_profile,
            action_counts=action_counts,
        )

    save_state(
        "phase_1",
        status="complete",
        tasks_path=str(output_path),
        task_count=len(benign_tasks),
        existing_task_count=len(existing_task_wraps),
        novel_task_count=len(novel_tasks),
        benchmark_path=str(benchmark_root),
        benchmark_name=benchmark_name,
        manifest_path=str(manifest_path),
        generate_novel=generate_novel,
        sandbox_model=sandbox_model,
        sites=sites_filter_raw,
        novel_tasks_per_site=novel_tasks_per_site,
        action_counts=action_counts,
        task_card_plan_path=str(task_card_plan_path) if task_card_plan_path else None,
        task_capability_profile=task_capability_profile,
        task_card_plan_digest=task_card_digest,
    )
    cost_tracker.log_phase_summary("phase_1")
    cost_tracker.save(state_dir / "cost_report.json")
    logger.info(
        "Phase 1 complete — %d benign tasks written to %s (%d existing-task + %d novel)",
        len(benign_tasks),
        output_path,
        len(existing_task_wraps),
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


def _parse_sites_filter(value: Any) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        parsed = {site.strip() for site in value.split(",") if site.strip()}
        return parsed or None
    parsed = {str(site).strip() for site in value if str(site).strip()}
    return parsed or None


def _parse_phase_1_action_counts(value: Any) -> dict[str, int] | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("--phase-1-action-counts must be a comma-separated string")
    text = value.strip()
    if not text:
        return None
    counts: dict[str, int] = {}
    for raw_part in text.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"invalid action count {part!r}; expected KIND=N")
        kind, raw_count = (item.strip() for item in part.split("=", 1))
        if not kind:
            raise ValueError(f"invalid action count {part!r}; missing action kind")
        if kind in counts:
            raise ValueError(f"duplicate action count for {kind!r}")
        try:
            count = int(raw_count)
        except ValueError as exc:
            raise ValueError(f"invalid count for {kind!r}: {raw_count!r}") from exc
        if count < 0:
            raise ValueError(f"action count for {kind!r} must be non-negative")
        counts[kind] = count
    if not counts or sum(counts.values()) <= 0:
        raise ValueError("--phase-1-action-counts must request at least one row")
    return counts


def _resolve_benchmark_root(
    args: argparse.Namespace,
    manifest: dict[str, Any],
) -> Path | None:
    """Resolve the benchmark root from CLI args or the Phase 0a manifest."""
    if getattr(args, "benchmark", None):
        benchmark_root = Path(args.benchmark)
    else:
        manifest_root = str(manifest.get("benchmark_codebase") or "").strip()
        if not manifest_root:
            return None
        benchmark_root = Path(manifest_root)
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
    sites: str | None = None,
    novel_tasks_per_site: int = DEFAULT_NOVEL_TASKS_PER_SITE,
    task_card_plan_path: Path | None = None,
    task_capability_profile: str | None = None,
    task_card_plan_digest_value: str | None = None,
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
        sites=sites,
        novel_tasks_per_site=novel_tasks_per_site,
        task_card_plan_path=str(task_card_plan_path) if task_card_plan_path else None,
        task_capability_profile=task_capability_profile,
        task_card_plan_digest=task_card_plan_digest_value,
    )


def _save_phase_1_failure_state(
    *,
    reason: str,
    benchmark_root: Path | None,
    manifest_path: Path,
    generate_novel: bool,
    sandbox_model: str,
    sites: str | None = None,
    novel_tasks_per_site: int = DEFAULT_NOVEL_TASKS_PER_SITE,
    task_card_plan_path: Path | None = None,
    task_capability_profile: str | None = None,
    task_card_plan_digest_value: str | None = None,
    existing_task_count: int | None = None,
    error: str | None = None,
) -> None:
    """Persist a failed Phase 1 state."""
    payload: dict[str, Any] = {
        "status": "failed",
        "reason": reason,
        "manifest_path": str(manifest_path),
        "generate_novel": generate_novel,
        "sandbox_model": sandbox_model,
        "novel_tasks_per_site": novel_tasks_per_site,
        "task_card_plan_path": str(task_card_plan_path) if task_card_plan_path else None,
        "task_capability_profile": task_capability_profile,
        "task_card_plan_digest": task_card_plan_digest_value,
    }
    if sites is not None:
        payload["sites"] = sites
    if benchmark_root is not None:
        payload["benchmark_path"] = str(benchmark_root)
    if existing_task_count is not None:
        payload["existing_task_count"] = existing_task_count
    if error is not None:
        payload["error"] = error
    save_state("phase_1", **payload)


async def _maybe_generate_new_tasks(
    *,
    manifest: dict[str, Any],
    benchmark_root: Path,
    output_dir: Path,
    output_path: Path,
    manifest_path: Path,
    existing_task_count: int,
    benchmark_resume_metadata_path: Path,
    resume: bool,
    sandbox_model: str,
    site_filter: set[str] | None,
    novel_tasks_per_site: int,
    task_card_plan: dict[str, Any] | None,
    action_counts: dict[str, int] | None = None,
) -> list[dict[str, Any]] | None:
    """Return generate-new-tasks output, reusing merged output when already present."""
    existing_novel_tasks = _reuse_existing_novel_tasks_if_valid(
        manifest=manifest,
        benchmark_root=benchmark_root,
        output_path=output_path,
        resume_metadata_path=benchmark_resume_metadata_path,
        resume=resume,
        sandbox_model=sandbox_model,
        site_filter=site_filter,
        novel_tasks_per_site=novel_tasks_per_site,
        task_card_plan=task_card_plan,
        action_counts=action_counts,
    )
    if existing_novel_tasks is not None:
        return existing_novel_tasks

    try:
        return await run_generate_new_tasks(
            manifest=manifest,
            benchmark_root=benchmark_root,
            output_dir=output_dir,
            sandbox_model=sandbox_model,
            site_filter=site_filter,
            novel_tasks_per_site=novel_tasks_per_site,
            task_card_plan=task_card_plan,
            action_counts=action_counts,
        )
    except Exception as exc:
        _save_phase_1_failure_state(
            reason="new_task_generation_failed",
            error=str(exc),
            benchmark_root=benchmark_root,
            manifest_path=manifest_path,
            generate_novel=True,
            sandbox_model=sandbox_model,
            novel_tasks_per_site=novel_tasks_per_site,
            existing_task_count=existing_task_count,
        )
        logger.error("Phase 1 (generate-new-tasks) failed: %s", exc)
        return None


def _reuse_existing_novel_tasks_if_valid(
    *,
    manifest: dict[str, Any],
    benchmark_root: Path,
    output_path: Path,
    resume_metadata_path: Path,
    resume: bool,
    sandbox_model: str,
    site_filter: set[str] | None,
    novel_tasks_per_site: int,
    task_card_plan: dict[str, Any] | None,
    action_counts: dict[str, int] | None = None,
) -> list[dict[str, Any]] | None:
    """Reuse merged generate-new-tasks output only on resume and only after provenance checks."""
    if not resume:
        return None

    existing_novel_tasks = _load_existing_novel_tasks(output_path)
    if existing_novel_tasks is None:
        return None

    eligible_sites = _load_generate_new_tasks_eligible_sites(
        profiles_dir=get_state_dir() / "phase_0c",
        manifest_eval_types=manifest.get("evaluation", {}).get("eval_types", []),
        site_filter=site_filter,
    )
    validation_errors = validate_existing_novel_tasks(
        existing_novel_tasks,
        eligible_sites=eligible_sites,
        expected_task_count=novel_tasks_per_site,
        task_card_plan=task_card_plan,
        action_counts=action_counts,
    )
    if validation_errors:
        logger.warning(
            "Phase 1 (generate-new-tasks): ignoring merged novel-task output because cached tasks are invalid: %s",
            "; ".join(validation_errors),
        )
        return None

    shared_inputs_fingerprint = compute_generate_new_tasks_shared_inputs_fingerprint(
        benchmark_root=benchmark_root,
        manifest=manifest,
        sandbox_model=sandbox_model,
        task_card_plan=task_card_plan,
        action_counts=action_counts,
    )
    current_fingerprint = compute_generate_new_tasks_resume_fingerprint(
        shared_inputs_fingerprint=shared_inputs_fingerprint,
        eligible_sites=eligible_sites,
        novel_tasks_per_site=novel_tasks_per_site,
        task_card_plan=task_card_plan,
        action_counts=action_counts,
    )

    metadata = _load_generate_new_tasks_resume_metadata(resume_metadata_path)
    if metadata.get("fingerprint") == current_fingerprint:
        logger.info(
            "Phase 1 (generate-new-tasks): merged output already contains %d valid novel tasks, skipping generation on resume",
            len(existing_novel_tasks),
        )
        return existing_novel_tasks

    if _merged_output_matches_current_site_caches(
        output_dir=output_path.parent,
        existing_novel_tasks=existing_novel_tasks,
        eligible_sites=eligible_sites,
        shared_inputs_fingerprint=shared_inputs_fingerprint,
        novel_tasks_per_site=novel_tasks_per_site,
        task_card_plan=task_card_plan,
        action_counts=action_counts,
    ):
        logger.info(
            "Phase 1 (generate-new-tasks): merged output already contains %d valid novel tasks and matches current per-site caches, skipping generation on resume",
            len(existing_novel_tasks),
        )
        return existing_novel_tasks

    logger.warning(
        "Phase 1 (generate-new-tasks): ignoring merged novel-task output because resume provenance does not match current inputs"
    )
    return None


def _write_generate_new_tasks_resume_metadata(
    metadata_path: Path,
    *,
    benchmark_root: Path,
    manifest: dict[str, Any],
    sandbox_model: str,
    site_filter: set[str] | None,
    novel_tasks_per_site: int,
    task_card_plan: dict[str, Any] | None,
    task_capability_profile: str | None = None,
    action_counts: dict[str, int] | None = None,
) -> None:
    eligible_sites = _load_generate_new_tasks_eligible_sites(
        profiles_dir=get_state_dir() / "phase_0c",
        manifest_eval_types=manifest.get("evaluation", {}).get("eval_types", []),
        site_filter=site_filter,
    )
    payload = {
        "fingerprint": compute_generate_new_tasks_resume_fingerprint(
            shared_inputs_fingerprint=compute_generate_new_tasks_shared_inputs_fingerprint(
                benchmark_root=benchmark_root,
                manifest=manifest,
                sandbox_model=sandbox_model,
                task_card_plan=task_card_plan,
                action_counts=action_counts,
            ),
            eligible_sites=eligible_sites,
            novel_tasks_per_site=novel_tasks_per_site,
            task_card_plan=task_card_plan,
            action_counts=action_counts,
        ),
        "benchmark_path": str(benchmark_root),
        "eligible_sites": [site.site_name for site in eligible_sites],
        "sandbox_model": sandbox_model,
        "novel_tasks_per_site": novel_tasks_per_site,
        "task_card_plan_digest": task_card_plan_digest(task_card_plan),
        "task_capability_profile": task_capability_profile,
        "action_counts": action_counts,
    }
    metadata_path.write_text(json.dumps(payload, indent=2))


def _merged_output_matches_current_site_caches(
    *,
    output_dir: Path,
    existing_novel_tasks: list[dict[str, Any]],
    eligible_sites: list[EligibleSiteProfile],
    shared_inputs_fingerprint: str,
    novel_tasks_per_site: int = DEFAULT_NOVEL_TASKS_PER_SITE,
    task_card_plan: dict[str, Any] | None = None,
    action_counts: dict[str, int] | None = None,
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
                novel_tasks_per_site=novel_tasks_per_site,
                task_card_plan=task_card_plan,
                action_counts=action_counts,
            ),
            expected_task_count=novel_tasks_per_site,
            task_card_plan=task_card_plan,
        )
        if cached_result is None:
            return False
        cached_tasks.extend(cached_result.benign_tasks)

    return _sort_novel_tasks(cached_tasks) == _sort_novel_tasks(existing_novel_tasks)


def _load_generate_new_tasks_resume_metadata(metadata_path: Path) -> dict[str, Any]:
    if not metadata_path.exists():
        return {}
    try:
        payload = json.loads(metadata_path.read_text())
    except json.JSONDecodeError:
        logger.warning(
            "Phase 1 (generate-new-tasks): ignoring invalid resume metadata at %s", metadata_path
        )
        return {}
    return payload if isinstance(payload, dict) else {}
