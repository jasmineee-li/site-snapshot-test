"""Phase 0: Benchmark reconnaissance.

Canonical source: ``docs/worldsim-v5-technical-specifcation.md`` "Phase 0: Benchmark
Reconnaissance" section.

Phase 0 has three sub-steps:

- **0a — Benchmark Discovery.** Single Modal Sandbox with the full benchmark
  source. Produces ``BENCHMARK_MANIFEST.json`` + ``.md``.
- **0b — Sandbox Filesystem Mapping.** Pure local Python (no LLM, no network).
  Computes the exact file list for each site's sandbox based on the manifest.
- **0c — Per-Site Profiling.** N parallel Modal Sandboxes, one per site,
  profiling verification capabilities, data model, injection surface, and
  existing task coverage.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from worldsim.cost_tracker import tracker as cost_tracker
from worldsim.modal_sandbox import run_claude_in_sandbox, upload_to_volume
from worldsim.prompt_loading import load_prompt
from worldsim.state import get_state_dir, save_state

logger = logging.getLogger(__name__)


async def run(benchmark: Path, sub: str = "0") -> int:
    """Phase 0 entrypoint.

    Args:
        benchmark: Path to the benchmark codebase (e.g.
            ``vendors/webarena-verified``).
        sub: One of ``"0"`` (full phase), ``"0a"``, ``"0b"``, or ``"0c"``.

    Returns:
        Process exit code.
    """
    output_base = get_state_dir()
    manifest = None
    sandbox_map = None

    if sub in {"0", "0a"}:
        save_state("phase_0a", status="running", benchmark_path=str(benchmark))
        manifest = await run_phase_0a(benchmark, output_base / "phase_0a")
        save_state(
            "phase_0a",
            status="complete",
            manifest_path=str(output_base / "phase_0a" / "BENCHMARK_MANIFEST.json"),
            benchmark_path=str(benchmark),
        )
        cost_tracker.log_phase_summary("phase_0a")
        cost_tracker.save(get_state_dir() / "cost_report.json")
        logger.info("Phase 0a complete — manifest written")
        if sub == "0a":
            return 0

    if sub in {"0", "0b"}:
        if manifest is None:
            manifest_path = output_base / "phase_0a" / "BENCHMARK_MANIFEST.json"
            if not manifest_path.exists():
                logger.error("Phase 0a output not found at %s — run phase 0a first", manifest_path)
                return 1
            manifest = json.loads(manifest_path.read_text())
        save_state("phase_0b", status="running", benchmark_path=str(benchmark))
        sandbox_map = compute_sandbox_maps(manifest, benchmark)
        out_dir = output_base / "phase_0b"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "SANDBOX_MAP.json").write_text(json.dumps(sandbox_map, indent=2))
        save_state("phase_0b", status="complete",
                   sandbox_map_path=str(out_dir / "SANDBOX_MAP.json"),
                   benchmark_path=str(benchmark))
        logger.info("Phase 0b complete — sandbox maps written for %d sites", len(sandbox_map))
        if sub == "0b":
            return 0

    if sub in {"0", "0c"}:
        if manifest is None:
            manifest_path = output_base / "phase_0a" / "BENCHMARK_MANIFEST.json"
            if not manifest_path.exists():
                logger.error("Phase 0a output not found at %s — run phase 0a first", manifest_path)
                return 1
            manifest = json.loads(manifest_path.read_text())
        if sandbox_map is None:
            sandbox_map_path = output_base / "phase_0b" / "SANDBOX_MAP.json"
            if not sandbox_map_path.exists():
                logger.error("Phase 0b output not found at %s — run phase 0b first", sandbox_map_path)
                return 1
            sandbox_map = json.loads(sandbox_map_path.read_text())
        save_state("phase_0c", status="running", benchmark_path=str(benchmark))
        try:
            await run_phase_0c(manifest, sandbox_map, benchmark, output_base / "phase_0c")
        except Exception as e:
            save_state(
                "phase_0c",
                status="failed",
                benchmark_path=str(benchmark),
                error=str(e),
            )
            logger.error("Phase 0c failed: %s", e)
            return 1
        save_state("phase_0c", status="complete",
                   profiles_dir=str(output_base / "phase_0c"),
                   benchmark_path=str(benchmark))
        cost_tracker.log_phase_summary("phase_0c")
        cost_tracker.save(get_state_dir() / "cost_report.json")
        logger.info("Phase 0c complete — per-site profiles written")

    return 0


# ---------------------------------------------------------------------------
# Phase 0a — Benchmark Discovery
# ---------------------------------------------------------------------------


async def run_phase_0a(benchmark_root: Path, output_dir: Path) -> dict:
    """Discover benchmark structure via a single Modal Sandbox.

    Claude Code explores the full benchmark codebase and produces
    ``BENCHMARK_MANIFEST.json`` (structured) and ``.md`` (human-readable).

    Returns:
        Parsed manifest dict.
    """
    benchmark_root = Path(benchmark_root).resolve()
    if not benchmark_root.is_dir():
        raise FileNotFoundError(f"Benchmark root does not exist: {benchmark_root}")

    # Upload benchmark to a Modal Volume once, then mount read-only.
    # This avoids re-hashing and re-uploading ~100MB on every sandbox creation.
    vol = await upload_to_volume(benchmark_root)
    prompt = load_prompt("discover-benchmark")

    logger.info("Phase 0a: launching discovery sandbox for %s", benchmark_root)
    outputs = await run_claude_in_sandbox(
        site_files={},
        prompt=prompt,
        output_paths=[
            "/workspace/output/BENCHMARK_MANIFEST.json",
            "/workspace/output/BENCHMARK_MANIFEST.md",
        ],
        volumes={"/workspace/benchmark": vol},
    )

    cost_tracker.record("phase_0a", outputs.get("_summary"))

    manifest_json = outputs.get("/workspace/output/BENCHMARK_MANIFEST.json")
    if not manifest_json:
        raise RuntimeError(
            "Phase 0a sandbox did not produce BENCHMARK_MANIFEST.json. "
            "Check sandbox logs for errors."
        )

    manifest = json.loads(manifest_json)
    missing_paths, unsafe_paths = _validate_manifest_paths(manifest, benchmark_root)
    if unsafe_paths:
        raise RuntimeError(
            "Phase 0a manifest contains unsafe paths:\n"
            + "\n".join(f"  - {error}" for error in unsafe_paths)
        )

    if missing_paths:
        logger.warning(
            "Phase 0a manifest has %d path errors — re-running with corrections",
            len(missing_paths),
        )
        correction = (
            "\n\n--- CORRECTION NEEDED ---\n"
            "The previous attempt produced a manifest with invalid paths. "
            "The following paths do NOT exist in the benchmark filesystem:\n"
            + "\n".join(f"  - {e}" for e in missing_paths)
            + "\n\nPlease re-explore and produce corrected output files. "
            "Only include paths you have verified exist."
        )
        outputs = await run_claude_in_sandbox(
            site_files={},
            prompt=prompt + correction,
            output_paths=[
                "/workspace/output/BENCHMARK_MANIFEST.json",
                "/workspace/output/BENCHMARK_MANIFEST.md",
            ],
            volumes={"/workspace/benchmark": vol},
        )
        cost_tracker.record("phase_0a", outputs.get("_summary"))
        manifest_json = outputs.get("/workspace/output/BENCHMARK_MANIFEST.json")
        if manifest_json:
            manifest = json.loads(manifest_json)
            missing_paths, unsafe_paths = _validate_manifest_paths(manifest, benchmark_root)
            if unsafe_paths:
                raise RuntimeError(
                    "Phase 0a retry produced unsafe paths:\n"
                    + "\n".join(f"  - {error}" for error in unsafe_paths)
                )
            if missing_paths:
                raise RuntimeError(
                    "Phase 0a retry still has invalid manifest paths:\n"
                    + "\n".join(f"  - {error}" for error in missing_paths)
                )

    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "BENCHMARK_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    manifest_md = outputs.get("/workspace/output/BENCHMARK_MANIFEST.md")
    if manifest_md:
        (output_dir / "BENCHMARK_MANIFEST.md").write_text(manifest_md)

    logger.info(
        "Phase 0a: manifest has %d sites, %d eval types",
        len(manifest.get("sites", [])),
        len(manifest.get("evaluation", {}).get("eval_types", [])),
    )
    return manifest


def _validate_manifest_paths(
    manifest: dict,
    benchmark_root: Path,
) -> tuple[list[str], list[str]]:
    """Check that every path referenced in the manifest exists on disk.

    Returns:
        ``(missing_paths, unsafe_paths)``.
    """
    missing: list[str] = []
    unsafe: list[str] = []
    root = Path(benchmark_root).resolve()

    def check(path_str: str, context: str) -> None:
        try:
            full = _resolve_manifest_path(root, path_str)
        except ValueError as exc:
            unsafe.append(f"{context}: {exc}")
            return
        if not full.exists():
            missing.append(f"{context}: {path_str}")

    # Evaluation harness paths
    for p in manifest.get("evaluation", {}).get("harness_paths", []):
        check(p, "evaluation.harness_paths")

    # Task definition paths
    for p in manifest.get("evaluation", {}).get("task_definition_paths", []):
        check(p, "evaluation.task_definition_paths")

    # Per-site paths
    for site in manifest.get("sites", []):
        name = site.get("name", "?")
        if "source_path" in site:
            check(site["source_path"], f"sites[{name}].source_path")
        for p in site.get("data_seeding", {}).get("paths", []):
            check(p, f"sites[{name}].data_seeding.paths")

    return missing, unsafe


# ---------------------------------------------------------------------------
# Phase 0b — Sandbox Filesystem Mapping
# ---------------------------------------------------------------------------


def compute_sandbox_maps(
    manifest: dict, benchmark_root: Path
) -> dict[str, list[str]]:
    """Compute the exact file list for each site's sandbox.

    Pure Python, no LLM, deterministic. Each site gets: shared eval harness
    files + site source + data seeding files + sampled task definitions.

    Returns:
        Dict mapping site name to sorted list of absolute file paths.
    """
    benchmark_root = Path(benchmark_root).resolve()
    sandbox_maps: dict[str, list[str]] = {}

    shared_files = _collect_files(
        manifest.get("evaluation", {}).get("harness_paths", []),
        benchmark_root,
    )

    for site in manifest.get("sites", []):
        site_name = site["name"]
        site_files = list(shared_files)

        if "source_path" in site:
            site_files.extend(_collect_files([site["source_path"]], benchmark_root))

        seeding_paths = site.get("data_seeding", {}).get("paths", [])
        site_files.extend(_collect_files(seeding_paths, benchmark_root))

        site_files.extend(
            _sample_tasks_for_site(manifest, site_name, benchmark_root, max_tasks=20)
        )

        sandbox_maps[site_name] = sorted(set(site_files))

    return sandbox_maps


def _collect_files(paths: list[str], root: Path) -> list[str]:
    """Resolve relative paths under root, walk directories, return absolute file paths."""
    result: list[str] = []
    for p in paths:
        full = _resolve_manifest_path(root, p)
        if full.is_file():
            result.append(str(full.resolve()))
        elif full.is_dir():
            for f in full.rglob("*"):
                if f.is_file():
                    result.append(str(_resolve_path_within_root(root, f)))
    return result


def _sample_tasks_for_site(
    manifest: dict, site_name: str, root: Path, max_tasks: int = 20
) -> list[str]:
    """Return file paths of task definitions relevant to a given site.

    Reads task definition files from the paths declared in the manifest,
    filters to tasks that reference this site, and returns up to max_tasks.

    Handles two known formats:
    - Single JSON array file (WebArena Verified: all tasks in one file)
    - Directory of per-task JSON files (original WebArena: config_files/)
    """
    task_paths = manifest.get("evaluation", {}).get("task_definition_paths", [])
    result: list[str] = []

    for tp in task_paths:
        full = _resolve_manifest_path(root, tp)
        if full.is_file() and full.suffix == ".json":
            # Single file containing all tasks — include the file itself
            # (each sandbox gets the same file; filtering happens in-memory)
            result.append(str(full.resolve()))
        elif full.is_dir():
            # Directory of task files — sample those referencing this site
            count = 0
            for f in sorted(full.rglob("*.json")):
                if count >= max_tasks:
                    break
                try:
                    safe_file = _resolve_path_within_root(root, f)
                    data = json.loads(safe_file.read_text())
                    # Handle both single-task and array-of-tasks files
                    tasks = data if isinstance(data, list) else [data]
                    for t in tasks:
                        sites = t.get("sites", [])
                        if site_name in sites or any(site_name in s for s in sites):
                            result.append(str(safe_file))
                            count += 1
                            break
                except (json.JSONDecodeError, KeyError):
                    continue

    return result[:max_tasks]


def _resolve_manifest_path(root: Path, path_str: str) -> Path:
    """Resolve a manifest path under ``root`` and reject escapes."""
    manifest_path = Path(path_str)
    if manifest_path.is_absolute():
        raise ValueError(f"Manifest path must be relative: {path_str}")
    if ".." in manifest_path.parts:
        raise ValueError(f"Manifest path must not traverse out of root: {path_str}")

    resolved_root = Path(root).resolve()
    resolved_path = (resolved_root / manifest_path).resolve(strict=False)
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Manifest path escapes benchmark root: {path_str}") from exc
    return resolved_path


def _resolve_path_within_root(root: Path, path: Path) -> Path:
    """Resolve a discovered filesystem path and ensure it stays under ``root``."""
    resolved_root = Path(root).resolve()
    resolved_path = Path(path).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Discovered file escapes benchmark root: {path}") from exc
    return resolved_path


# ---------------------------------------------------------------------------
# Phase 0c — Per-Site Profiling
# ---------------------------------------------------------------------------


async def run_phase_0c(
    manifest: dict,
    sandbox_map: dict[str, list[str]],
    benchmark_root: Path,
    output_dir: Path,
    timeout: int = 14400,
) -> dict[str, Any]:
    """Profile each site in parallel via Modal Sandboxes.

    One sandbox per site, each receiving only that site's files from the
    sandbox map. Produces ``BENCHMARK_PROFILE_{site}.json`` and ``.md``.
    Sites that already have a profile on disk are skipped.

    Args:
        timeout: Per-sandbox wall-clock timeout in seconds (default: 4 hours).

    Returns:
        Dict mapping site name to sandbox outputs dict.
    """
    benchmark_root = Path(benchmark_root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Skip sites that already have profiles on disk (supports re-runs).
    sites_to_profile = {}
    for name, files in sandbox_map.items():
        profile_path = output_dir / f"BENCHMARK_PROFILE_{name}.json"
        if profile_path.exists():
            logger.info("Phase 0c: skipping site %r (profile already exists at %s)", name, profile_path)
        else:
            sites_to_profile[name] = files

    if not sites_to_profile:
        logger.info("Phase 0c: all sites already profiled, nothing to do")
        return {}

    async def profile_one_site(
        site_name: str, file_list: list[str]
    ) -> tuple[str, dict[str, str | None]]:
        # Stage all files into a single temp dir mirroring /workspace/benchmark/
        # structure, then mount once via add_local_dir. This replaces N separate
        # add_local_file calls (each SHA-256 hashed individually) with one
        # add_local_dir call, significantly reducing sandbox build time.
        #
        # The inner dir must be named "benchmark" because modal_sandbox.py's
        # add_local_dir logic takes parent of the remote_path and places the
        # directory by name — so /workspace/benchmark -> parent=/workspace,
        # and the dir named "benchmark" lands at /workspace/benchmark/.
        staging_root = Path(tempfile.mkdtemp(prefix=f"worldsim_0c_{site_name}_"))
        staging_dir = staging_root / "benchmark"
        staging_dir.mkdir()
        try:
            for local_path in file_list:
                rel = os.path.relpath(local_path, benchmark_root)
                staged = staging_dir / rel
                staged.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(local_path, staged)

            site_files = {"/workspace/benchmark": str(staging_dir)}

            logger.info(
                "Phase 0c: launching profiling sandbox for site %r "
                "(%d files staged into 1 mount)",
                site_name, len(file_list),
            )

            outputs = await run_claude_in_sandbox(
                site_files=site_files,
                prompt=load_prompt("profile-site"),
                output_paths=[
                    "/workspace/output/BENCHMARK_PROFILE.json",
                    "/workspace/output/BENCHMARK_PROFILE.md",
                ],
                timeout=timeout,
            )
            cost_tracker.record("phase_0c", outputs.get("_summary"), site=site_name)
        finally:
            shutil.rmtree(staging_root, ignore_errors=True)

        for remote_path, content in outputs.items():
            if content and not remote_path.startswith("_"):
                suffix = Path(remote_path).suffix
                out_path = output_dir / f"BENCHMARK_PROFILE_{site_name}{suffix}"
                out_path.write_text(content)
                logger.info("Phase 0c: wrote %s", out_path)

        return site_name, outputs

    raw_results = await asyncio.gather(
        *[
            profile_one_site(name, files)
            for name, files in sites_to_profile.items()
        ],
        return_exceptions=True,
    )

    results: list[tuple[str, dict[str, str | None]]] = []
    failures: list[str] = []
    for r in raw_results:
        if isinstance(r, Exception):
            logger.error("Phase 0c site profiling failed: %s", r)
            failures.append(str(r))
        else:
            results.append(r)

    expected_sites = set(sites_to_profile)
    completed_sites = {site_name for site_name, _ in results}
    missing_sites = sorted(expected_sites - completed_sites)
    failures.extend(f"missing profile result for site {site}" for site in missing_sites)

    for site_name, outputs in results:
        profile_json = outputs.get("/workspace/output/BENCHMARK_PROFILE.json")
        if not profile_json:
            failures.append(f"site {site_name} did not produce BENCHMARK_PROFILE.json")
            continue
        try:
            profile = json.loads(profile_json)
        except json.JSONDecodeError as exc:
            failures.append(f"site {site_name} produced invalid profile JSON: {exc}")
            continue
        try:
            _validate_profile(
                site_name,
                profile,
                manifest.get("evaluation", {}).get("eval_types", []),
            )
        except ValueError as exc:
            failures.append(str(exc))

    if failures:
        raise RuntimeError(
            "Phase 0c did not complete all required site profiles:\n"
            + "\n".join(f"  - {failure}" for failure in failures)
        )

    return dict(results)


def _validate_profile(
    site_name: str,
    profile: dict,
    manifest_eval_types: list[str],
) -> None:
    """Validate cross-references within a site profile.

    Checks that injection surface source_fields reference data model fields,
    and that eval_types appear in verification capabilities.
    """
    # Collect known field names from data model
    known_fields: set[str] = set()
    for entity in profile.get("data_model", []):
        for field in entity.get("fields", []):
            known_fields.add(field.get("name", ""))
        # Also add entity-level identifiers
        storage = entity.get("storage", "")
        if storage:
            known_fields.add(storage)

    errors: list[str] = []

    # Check injection surface source_fields
    for surface in profile.get("injection_surface", []):
        source = surface.get("source_field", "")
        if source and "." in source:
            # Format is "table.column" — check the column part
            field_name = source.split(".")[-1]
            if field_name not in known_fields and known_fields:
                errors.append(
                    f"injection surface {surface.get('id', '?')!r} references "
                    f"unknown field {source!r}"
                )

    # Collect known eval types from verification capabilities
    known_eval_types: set[str] = set()
    for cap in profile.get("verification_capabilities", []):
        eval_type = cap.get("eval_type", "")
        if eval_type:
            known_eval_types.add(eval_type)

    missing_eval_types = sorted(
        eval_type
        for eval_type in known_eval_types
        if eval_type not in set(manifest_eval_types)
    )
    if missing_eval_types:
        errors.append(
            "verification capabilities reference eval types absent from manifest: "
            + ", ".join(missing_eval_types)
        )

    if errors:
        raise ValueError(
            f"Profile {site_name} failed validation:\n"
            + "\n".join(f"  - {error}" for error in errors)
        )

    logger.info(
        "Profile %s validated: %d data model entities, %d injection surfaces, %d eval types",
        site_name,
        len(profile.get("data_model", [])),
        len(profile.get("injection_surface", [])),
        len(known_eval_types),
    )
