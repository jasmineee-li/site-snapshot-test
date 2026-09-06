"""Phase 0a: Benchmark Discovery and the sandbox file map it produces.

Runs the single Modal Sandbox that writes ``BENCHMARK_MANIFEST.json`` + ``.md``,
repairs and validates the manifest's paths and eval types, and derives the
per-site sandbox file list that Phase 0c profiles.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from warp_taskgen.cost_tracker import tracker as cost_tracker
from warp_taskgen.modal_sandbox import run_claude_in_sandbox, upload_to_volume
from warp_taskgen.prompt_corrections import render_validation_feedback
from warp_taskgen.prompt_loading import load_prompt

logger = logging.getLogger(__name__)


def _read_only_volume(volume: Any) -> Any:
    """Return a read-only mount when the object supports it."""
    read_only = getattr(volume, "read_only", None)
    return read_only() if callable(read_only) else volume


async def run_phase_0a(
    benchmark_root: Path,
    output_dir: Path,
    *,
    sandbox_model: str = "claude-sonnet-4-6",
) -> dict:
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
    prompt = load_prompt("discover-benchmark", validation_command="manifest")

    logger.info("Phase 0a: launching discovery sandbox for %s", benchmark_root)
    outputs = await run_claude_in_sandbox(
        site_files={},
        prompt=prompt,
        output_paths=[
            "/workspace/output/BENCHMARK_MANIFEST.json",
            "/workspace/output/BENCHMARK_MANIFEST.md",
        ],
        model=sandbox_model,
        volumes={"/workspace/benchmark": _read_only_volume(vol)},
        label="0a-discovery",
    )

    cost_tracker.record("phase_0a", outputs.get("_summary"))

    manifest_json = outputs.get("/workspace/output/BENCHMARK_MANIFEST.json")
    if not manifest_json:
        raise RuntimeError(
            "Phase 0a sandbox did not produce BENCHMARK_MANIFEST.json. "
            "Check sandbox logs for errors."
        )

    manifest = json.loads(manifest_json)
    _repair_manifest_paths(manifest, benchmark_root)
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
        correction = render_validation_feedback(
            artifact_name="BENCHMARK_MANIFEST.json",
            errors=[
                {
                    "code": "MISSING_PATH",
                    "path": "$",
                    "message": error,
                    "repair_hint": "Only include paths verified to exist under /workspace/benchmark.",
                }
                for error in missing_paths
            ],
            summary="The manifest referenced paths that do not exist in the benchmark filesystem.",
            instruction="Re-explore and produce corrected output files. Only include paths you have verified exist.",
        )
        outputs = await run_claude_in_sandbox(
            site_files={},
            prompt=prompt + correction,
            output_paths=[
                "/workspace/output/BENCHMARK_MANIFEST.json",
                "/workspace/output/BENCHMARK_MANIFEST.md",
            ],
            model=sandbox_model,
            volumes={"/workspace/benchmark": _read_only_volume(vol)},
            label="0a-discovery-retry",
        )
        cost_tracker.record("phase_0a", outputs.get("_summary"))
        manifest_json = outputs.get("/workspace/output/BENCHMARK_MANIFEST.json")
        if manifest_json:
            manifest = json.loads(manifest_json)
            _repair_manifest_paths(manifest, benchmark_root)
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


def _repair_manifest_paths(manifest: dict, benchmark_root: Path) -> None:
    """Apply deterministic, verified repairs for common manifest path shapes.

    Phase 0a is intentionally exploratory, but path existence is a host-side
    contract. When a model emits a near-miss path and an unambiguous existing
    sibling path can be verified under the benchmark root, repair it before
    spending another sandbox attempt. This preserves fail-closed validation:
    unresolved, unsafe, or ambiguous paths still fail below.
    """
    root = Path(benchmark_root).resolve()
    _repair_task_definition_paths(manifest, root)
    for site in manifest.get("sites", []):
        if not isinstance(site, dict):
            continue
        source_path = site.get("source_path")
        site_name = site.get("name")
        if not isinstance(source_path, str) or not isinstance(site_name, str):
            continue
        repaired = _repair_site_source_path(source_path, site_name, root)
        if repaired is not None:
            site["source_path"] = repaired


def _repair_task_definition_paths(manifest: dict, root: Path) -> None:
    evaluation = manifest.get("evaluation")
    if not isinstance(evaluation, dict):
        return
    task_paths = evaluation.get("task_definition_paths")
    if isinstance(task_paths, list) and task_paths:
        return

    candidate = "assets/dataset/webarena-verified.json"
    if _resolve_manifest_path(root, candidate).is_file():
        evaluation["task_definition_paths"] = [candidate]


def _repair_site_source_path(source_path: str, site_name: str, root: Path) -> str | None:
    try:
        current = _resolve_manifest_path(root, source_path)
    except ValueError:
        return None
    if current.exists():
        return None

    raw = source_path.strip().strip("/")
    if not raw:
        return None
    candidates = [
        f"{raw}/sites/{site_name}",
        f"{raw}/site/{site_name}",
    ]
    for prefix in ("docker", "environments"):
        marker = f"{prefix}/"
        if marker in raw:
            before, after = raw.split(f"{prefix}/", 1)
            if after and not after.startswith("sites/"):
                candidates.append(f"{before}{prefix}/sites/{after}")

    existing: list[str] = []
    for candidate in candidates:
        try:
            resolved = _resolve_manifest_path(root, candidate)
        except ValueError:
            continue
        if resolved.exists():
            existing.append(candidate)
    unique = sorted(set(existing))
    if len(unique) == 1:
        return unique[0]
    return None


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


def compute_sandbox_maps(manifest: dict, benchmark_root: Path) -> dict[str, list[str]]:
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

        site_files.extend(_sample_tasks_for_site(manifest, site_name, benchmark_root, max_tasks=20))

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


def _validate_manifest_eval_types(
    verification_capabilities: object,
    manifest_eval_type_set: set[str],
) -> list[str]:
    """Reject verification capabilities that name eval types absent from the manifest."""
    if not manifest_eval_type_set or not isinstance(verification_capabilities, list):
        return []

    discovered = {
        str(item.get("eval_type"))
        for item in verification_capabilities
        if isinstance(item, dict) and item.get("eval_type")
    }
    unknown = sorted(discovered - manifest_eval_type_set)
    if not unknown:
        return []
    return [
        "verification capabilities reference eval types absent from manifest: " + ", ".join(unknown)
    ]
