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
import hashlib
import json
import logging
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from pydantic import ValidationError

from worldsim._sandbox_validator import (
    validate_agent_context,
    validate_data_model_profile,
    validate_injection_surface,
    validate_verification_capabilities,
)
from worldsim.config import BenchmarkConfig, BenchmarkInstance, VerificationProxy
from worldsim.cost_tracker import tracker as cost_tracker
from worldsim.failpoints import crash_if_enabled
from worldsim.modal_sandbox import (
    preflight_sandbox_environment,
    run_claude_in_sandbox,
    upload_to_volume,
)
from worldsim.placeholders import normalize_site_name
from worldsim.profile_validation import validate_profile
from worldsim.prompt_loading import load_prompt
from worldsim.state import get_state_dir, save_state

logger = logging.getLogger(__name__)

# Maximum number of correction retries for profile validation (initial attempt + this many).
PROFILE_FIX_MAX_ITERATIONS = 2
_PROFILE_METADATA_PREFIX = "PROFILE_METADATA_"


def _write_text_atomic(path: Path, text: str) -> None:
    """Atomically replace *path* with *text*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        crash_if_enabled("phase_0c.outputs.before_replace")
        os.replace(tmp_path, path)
        crash_if_enabled("phase_0c.outputs.after_replace")
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _profile_metadata_path(output_dir: Path, site_name: str) -> Path:
    return output_dir / f"{_PROFILE_METADATA_PREFIX}{site_name}.json"


def _read_only_volume(volume: Any) -> Any:
    """Return a read-only mount when the object supports it."""
    read_only = getattr(volume, "read_only", None)
    return read_only() if callable(read_only) else volume


def _phase_0_state_metadata(
    *,
    benchmark: Path,
    sandbox_model: str,
    instances_path: Path | None,
) -> dict[str, str]:
    metadata = {
        "benchmark_path": str(benchmark),
        "sandbox_model": sandbox_model,
    }
    if instances_path is not None:
        metadata["instances_path"] = str(Path(instances_path))
    return metadata


def _load_phase_0c_config(
    instances_path: Path,
) -> tuple[list[BenchmarkInstance], VerificationProxy | None]:
    """Load instances and optional verification proxy config from instances.json.

    Returns (instances, verification_proxy). The proxy is None when absent or
    when its token is empty (disabled).
    """

    path = Path(instances_path)
    try:
        config = BenchmarkConfig.model_validate_json(path.read_text())
    except (OSError, ValidationError) as exc:
        raise RuntimeError(f"invalid instances config at {path}: {exc}") from exc
    proxy = config.verification_proxy
    # Treat empty token as "proxy not configured".
    if proxy is not None and not proxy.token.strip():
        proxy = None
    return config.instances, proxy


def _sanitize_instance_site_url(site_url: str, *, site_name: str) -> str:
    """Return a normalized live-verification URL safe to pass into sandboxes."""
    parts = urlsplit(site_url.strip())
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError(
            f"site {site_name!r} has unsupported site_url {site_url!r}; expected http(s) base URL"
        )
    if parts.username is not None or parts.password is not None:
        raise ValueError(
            f"site {site_name!r} site_url must not include credentials when used for Phase 0c"
        )
    if parts.query or parts.fragment:
        raise ValueError(
            f"site {site_name!r} site_url must not include query or fragment when used for Phase 0c"
        )
    hostname = parts.hostname
    if not hostname:
        raise ValueError(
            f"site {site_name!r} has unsupported site_url {site_url!r}; expected http(s) base URL"
        )
    try:
        port = parts.port
    except ValueError as exc:
        raise ValueError(f"site {site_name!r} has invalid site_url port in {site_url!r}") from exc

    default_port = 80 if scheme == "http" else 443
    normalized_host = hostname.lower()
    host_display = f"[{normalized_host}]" if ":" in normalized_host else normalized_host
    normalized_netloc = host_display
    if port is not None and port != default_port:
        normalized_netloc = f"{host_display}:{port}"
    normalized_path = parts.path.rstrip("/")
    return urlunsplit((scheme, normalized_netloc, normalized_path, "", ""))


def _build_instance_site_url_map(instances: list[BenchmarkInstance] | None) -> dict[str, str]:
    """Return a stable representative site_url for each site for live verification."""
    if not instances:
        return {}

    by_site: dict[str, set[str]] = {}
    for instance in instances:
        normalized_site = normalize_site_name(instance.site_name)
        if not normalized_site:
            continue
        sanitized_url = _sanitize_instance_site_url(instance.site_url, site_name=instance.site_name)
        by_site.setdefault(normalized_site, set()).add(sanitized_url)

    result: dict[str, str] = {}
    for site_name, site_urls in by_site.items():
        selected = sorted(site_urls)[0]
        if len(site_urls) > 1:
            logger.info(
                "Phase 0c: site %r has %d configured instance URLs; using %s for live verification",
                site_name,
                len(site_urls),
                selected,
            )
        result[site_name] = selected
    return result


def _apply_proxy_to_url(site_url: str, port_offset: int, *, scheme: str | None = None) -> str:
    """Rewrite a site URL to use the proxy port (real_port + port_offset)."""
    parts = urlsplit(site_url)
    try:
        port = parts.port
    except ValueError:
        return site_url
    if port is None:
        # No explicit port, cannot apply offset meaningfully.
        return site_url
    proxy_port = port + port_offset
    hostname = parts.hostname or ""
    host_display = f"[{hostname}]" if ":" in hostname else hostname
    proxy_netloc = f"{host_display}:{proxy_port}"
    proxy_scheme = scheme or parts.scheme
    return urlunsplit((proxy_scheme, proxy_netloc, parts.path, parts.query, parts.fragment))


def _verification_proxy_metadata(
    verification_proxy: VerificationProxy | None,
) -> dict[str, Any] | None:
    """Return stable, non-secret proxy metadata for Phase 0c cache fingerprints."""
    if verification_proxy is None:
        return None
    token_digest = hashlib.sha256(verification_proxy.token.encode()).hexdigest()
    return {
        "scheme": verification_proxy.scheme,
        "port_offset": verification_proxy.port_offset,
        "token_sha256": token_digest,
    }


def _existing_site_outputs_are_reusable(
    *,
    output_dir: Path,
    site_name: str,
    benchmark_root: Path,
    sandbox_model: str,
    manifest_eval_type_set: set[str],
    instance_site_url: str | None,
    verification_proxy: VerificationProxy | None,
) -> bool:
    """Return True only when existing site outputs are complete and match this run."""
    profile_path = output_dir / f"BENCHMARK_PROFILE_{site_name}.json"
    context_path = output_dir / f"AGENT_CONTEXT_{site_name}.json"
    metadata_path = _profile_metadata_path(output_dir, site_name)
    if not (profile_path.exists() and context_path.exists() and metadata_path.exists()):
        return False
    try:
        profile = json.loads(profile_path.read_text())
        agent_context = json.loads(context_path.read_text())
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "Phase 0c: existing outputs for site %r are unreadable/corrupt, re-profiling: %s",
            site_name,
            exc,
        )
        return False

    if (
        not isinstance(profile, dict)
        or not isinstance(agent_context, dict)
        or not isinstance(metadata, dict)
    ):
        logger.warning(
            "Phase 0c: existing outputs for site %r have invalid JSON shape, re-profiling",
            site_name,
        )
        return False

    if profile.get("agent_context") != agent_context:
        logger.warning(
            "Phase 0c: existing profile/context for site %r disagree, re-profiling",
            site_name,
        )
        return False

    expected_metadata = {
        "site_name": site_name,
        "benchmark_root": str(benchmark_root),
        "sandbox_model": sandbox_model,
        "instance_site_url": instance_site_url,
        "verification_proxy": _verification_proxy_metadata(verification_proxy),
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            logger.info(
                "Phase 0c: existing outputs for site %r do not match current %s, re-profiling",
                site_name,
                key,
            )
            return False

    try:
        validate_agent_context(agent_context, site_name=site_name)
        validate_profile(
            site_name,
            profile,
            manifest_eval_types=manifest_eval_type_set,
        )
    except ValueError as exc:
        logger.warning(
            "Phase 0c: existing outputs for site %r failed validation, re-profiling: %s",
            site_name,
            exc,
        )
        return False

    return True


async def run(
    benchmark: Path,
    sub: str = "0",
    sandbox_model: str = "claude-sonnet-4-6",
    instances_path: Path | None = None,
) -> int:
    """Phase 0 entrypoint.

    Args:
        benchmark: Path to the benchmark codebase (e.g.
            ``vendors/webarena-verified``).
        sub: One of ``"0"`` (full phase), ``"0a"``, ``"0b"``, or ``"0c"``.
        instances_path: Optional path to instances.json. When provided,
            Phase 0c sandboxes receive instance connectivity info
            (site URLs only, no credentials).

    Returns:
        Process exit code.
    """
    output_base = get_state_dir()
    manifest = None
    sandbox_map = None
    state_metadata = _phase_0_state_metadata(
        benchmark=benchmark,
        sandbox_model=sandbox_model,
        instances_path=instances_path,
    )

    # Fail fast if sandbox auth or image setup is missing — 0a and 0c need sandboxes.
    if sub in {"0", "0a", "0c"}:
        try:
            await preflight_sandbox_environment()
        except RuntimeError as exc:
            logger.error("Phase 0 sandbox pre-flight failed:\n%s", exc)
            save_state(
                f"phase_{sub}",
                status="failed",
                reason="sandbox_preflight_failed",
                **state_metadata,
            )
            return 1

    if sub in {"0", "0a"}:
        save_state(
            "phase_0a",
            status="running",
            **state_metadata,
        )
        manifest = await run_phase_0a(
            benchmark,
            output_base / "phase_0a",
            sandbox_model=sandbox_model,
        )
        save_state(
            "phase_0a",
            status="complete",
            manifest_path=str(output_base / "phase_0a" / "BENCHMARK_MANIFEST.json"),
            **state_metadata,
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
        save_state(
            "phase_0b",
            status="running",
            **state_metadata,
        )
        sandbox_map = compute_sandbox_maps(manifest, benchmark)
        out_dir = output_base / "phase_0b"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "SANDBOX_MAP.json").write_text(json.dumps(sandbox_map, indent=2))
        save_state(
            "phase_0b",
            status="complete",
            sandbox_map_path=str(out_dir / "SANDBOX_MAP.json"),
            **state_metadata,
        )
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
                logger.error(
                    "Phase 0b output not found at %s — run phase 0b first", sandbox_map_path
                )
                return 1
            sandbox_map = json.loads(sandbox_map_path.read_text())
        save_state(
            "phase_0c",
            status="running",
            **state_metadata,
        )
        try:
            # Load instance configs and optional proxy for connectivity context.
            instances: list[BenchmarkInstance] | None = None
            verification_proxy = None
            if instances_path is not None:
                instances, verification_proxy = _load_phase_0c_config(instances_path)
            await run_phase_0c(
                manifest,
                sandbox_map,
                benchmark,
                output_base / "phase_0c",
                sandbox_model=sandbox_model,
                instances=instances,
                verification_proxy=verification_proxy,
            )
        except Exception as e:
            save_state(
                "phase_0c",
                status="failed",
                error=str(e),
                **state_metadata,
            )
            logger.error("Phase 0c failed: %s", e)
            return 1
        save_state(
            "phase_0c",
            status="complete",
            profiles_dir=str(output_base / "phase_0c"),
            **state_metadata,
        )
        cost_tracker.log_phase_summary("phase_0c")
        cost_tracker.save(get_state_dir() / "cost_report.json")
        logger.info("Phase 0c complete — per-site profiles written")

    return 0


# ---------------------------------------------------------------------------
# Phase 0a — Benchmark Discovery
# ---------------------------------------------------------------------------


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
            model=sandbox_model,
            volumes={"/workspace/benchmark": _read_only_volume(vol)},
            label="0a-discovery-retry",
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


# ---------------------------------------------------------------------------
# Phase 0c — Per-Site Profiling
# ---------------------------------------------------------------------------


async def run_phase_0c(
    manifest: dict,
    sandbox_map: dict[str, list[str]],
    benchmark_root: Path,
    output_dir: Path,
    timeout: int = 14400,
    sandbox_model: str = "claude-sonnet-4-6",
    instances: list[BenchmarkInstance] | None = None,
    verification_proxy: VerificationProxy | None = None,
) -> dict[str, Any]:
    """Profile each site via tiered parallel Modal Sandboxes.

    Two-tier structure per site:

    - **Tier 1** (parallel): Verification Capabilities (A), Data Model (B),
      Agent Context (C).
    - **Tier 2** (sequential, receives validated Tier 1 outputs): Injection
      Surface + Task Coverage (D+E).

    All sites are profiled in parallel. Sites that already have a complete
    profile on disk are skipped.

    Args:
        timeout: Per-sandbox wall-clock timeout in seconds (default: 4 hours).
        instances: Optional list of validated benchmark instances. When
            provided, Tier 2 sandboxes receive one representative site URL per
            site for live verification.
        verification_proxy: Optional proxy config. When present, site URLs
            are rewritten to proxy ports and an auth header is included in
            the INSTANCE_CONNECTIVITY.json staged into Tier 2 sandboxes.

    Returns:
        Dict mapping site name to merged profile outputs.
    """
    benchmark_root = Path(benchmark_root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_eval_type_set = {
        str(eval_type)
        for eval_type in manifest.get("evaluation", {}).get("eval_types", [])
        if eval_type
    }
    instance_urls = _build_instance_site_url_map(instances)

    # Skip sites that already have all outputs on disk (supports re-runs).
    sites_to_profile = {}
    for name, files in sandbox_map.items():
        instance_site_url = instance_urls.get(normalize_site_name(name))
        if _existing_site_outputs_are_reusable(
            output_dir=output_dir,
            site_name=name,
            benchmark_root=benchmark_root,
            sandbox_model=sandbox_model,
            manifest_eval_type_set=manifest_eval_type_set,
            instance_site_url=instance_site_url,
            verification_proxy=verification_proxy,
        ):
            logger.info("Phase 0c: skipping site %r (profile + agent context already exist)", name)
        else:
            sites_to_profile[name] = files

    if not sites_to_profile:
        logger.info("Phase 0c: all sites already profiled, nothing to do")
        return {}

    raw_results = await asyncio.gather(
        *[
            _profile_one_site_tiered(
                site_name=name,
                file_list=files,
                benchmark_root=benchmark_root,
                output_dir=output_dir,
                manifest=manifest,
                timeout=timeout,
                sandbox_model=sandbox_model,
                site_url=instance_urls.get(normalize_site_name(name)),
                verification_proxy=verification_proxy,
            )
            for name, files in sites_to_profile.items()
        ],
        return_exceptions=True,
    )

    results: dict[str, Any] = {}
    failures: list[str] = []
    for r in raw_results:
        if isinstance(r, Exception):
            logger.error("Phase 0c site profiling failed: %s", r)
            failures.append(str(r))
        elif isinstance(r, tuple) and len(r) == 2:
            site_name, site_outputs = r
            results[site_name] = site_outputs

    expected_sites = set(sites_to_profile)
    missing_sites = sorted(expected_sites - set(results))
    failures.extend(f"missing profile result for site {site}" for site in missing_sites)

    if failures:
        raise RuntimeError(
            "Phase 0c did not complete all required site profiles:\n"
            + "\n".join(f"  - {failure}" for failure in failures)
        )

    return results


def _stage_benchmark_files(
    file_list: list[str],
    benchmark_root: Path,
    site_name: str,
) -> tuple[Path, Path]:
    """Stage benchmark files into a temp dir for sandbox mounting.

    Returns ``(staging_root, staging_dir)`` where staging_dir is the inner
    "benchmark" directory suitable for mounting at ``/workspace/benchmark``.
    Caller is responsible for cleanup via ``shutil.rmtree(staging_root)``.
    """
    staging_root = Path(tempfile.mkdtemp(prefix=f"worldsim_0c_{site_name}_"))
    staging_dir = staging_root / "benchmark"
    staging_dir.mkdir()
    for local_path in file_list:
        rel = os.path.relpath(local_path, benchmark_root)
        staged = staging_dir / rel
        staged.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, staged)
    return staging_root, staging_dir


async def _run_tier_sandbox(
    *,
    site_name: str,
    site_files: dict[str, str],
    prompt: str,
    output_paths: list[str],
    timeout: int,
    label: str,
    sandbox_model: str,
    extra_inputs: dict[str, str] | None = None,
    volumes: dict[str, Any] | None = None,
) -> dict[str, str | None]:
    """Run a single profiling tier sandbox with the standard pattern.

    Loads the prompt, appends validation footer, runs the sandbox, records
    cost, and returns raw outputs.
    """
    all_files = dict(site_files)
    if extra_inputs:
        all_files.update(extra_inputs)

    outputs = await run_claude_in_sandbox(
        site_files=all_files,
        prompt=prompt,
        output_paths=output_paths,
        timeout=timeout,
        model=sandbox_model,
        volumes=volumes,
        label=label,
    )
    cost_tracker.record("phase_0c", outputs.get("_summary"), site=site_name)
    return outputs


def _render_tier_prompt(*, prompt_name: str, validation_command: str, site_name: str) -> str:
    """Load a profiling prompt and substitute the site name placeholder."""
    return load_prompt(
        prompt_name,
        validation_command=validation_command,
    ).replace("{site_name}", site_name)


def _render_correction_block(
    *,
    site_name: str,
    artifact_name: str,
    errors: list[str],
    extra_guidance: str | None = None,
) -> str:
    """Return a reusable prompt suffix for retrying a failed tier output."""
    guidance = (
        "\n\nAdditional guidance:\n" + extra_guidance.strip()
        if extra_guidance and extra_guidance.strip()
        else ""
    )
    return (
        "\n\n--- CORRECTION NEEDED ---\n"
        f"The previous attempt for site {site_name} produced an invalid {artifact_name}.\n"
        "Rewrite the output file completely so it satisfies the schema and all cross-reference checks.\n"
        "Validation errors:\n" + "\n".join(f"  - {error}" for error in errors) + guidance
    )


async def _run_tier_json_with_retries(
    *,
    site_name: str,
    site_files: dict[str, str],
    prompt_name: str,
    validation_command: str,
    output_path: str,
    timeout: int,
    label: str,
    sandbox_model: str,
    validate_parsed: Callable[[object], list[str]],
    extra_inputs: dict[str, str] | None = None,
    correction_guidance: str | None = None,
    volumes: dict[str, Any] | None = None,
) -> Any:
    """Run one profiling tier, retrying semantic validation failures in-place."""
    artifact_name = Path(output_path).name
    base_prompt = _render_tier_prompt(
        prompt_name=prompt_name,
        validation_command=validation_command,
        site_name=site_name,
    )
    prompt = base_prompt
    last_errors: list[str] = []

    for attempt in range(1 + PROFILE_FIX_MAX_ITERATIONS):
        attempt_label = label if attempt == 0 else f"{label}-retry{attempt}"
        outputs = await _run_tier_sandbox(
            site_name=site_name,
            site_files=site_files,
            prompt=prompt,
            output_paths=[output_path],
            timeout=timeout,
            label=attempt_label,
            sandbox_model=sandbox_model,
            extra_inputs=extra_inputs,
            volumes=volumes,
        )

        raw = outputs.get(output_path)
        parsed: object | None = None
        errors: list[str] = []
        if not raw:
            errors.append(f"{artifact_name} was not produced")
        else:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                errors.append(f"{artifact_name} contained invalid JSON: {exc}")

        if not errors and parsed is not None:
            errors.extend(validate_parsed(parsed))

        if not errors:
            return parsed

        last_errors = errors
        if attempt < PROFILE_FIX_MAX_ITERATIONS:
            logger.warning(
                "Phase 0c: site %r %s failed validation, retrying (%d/%d): %s",
                site_name,
                artifact_name,
                attempt + 1,
                PROFILE_FIX_MAX_ITERATIONS,
                "; ".join(errors),
            )
            prompt = base_prompt + _render_correction_block(
                site_name=site_name,
                artifact_name=artifact_name,
                errors=errors,
                extra_guidance=correction_guidance,
            )

    raise RuntimeError(
        f"{artifact_name} for site {site_name} failed validation:\n"
        + "\n".join(f"  - {error}" for error in last_errors)
    )


async def _profile_one_site_tiered(
    *,
    site_name: str,
    file_list: list[str],
    benchmark_root: Path,
    output_dir: Path,
    manifest: dict,
    timeout: int,
    sandbox_model: str,
    site_url: str | None = None,
    verification_proxy: VerificationProxy | None = None,
) -> tuple[str, dict[str, Any]]:
    """Profile one site using two-tier sandbox execution.

    Tier 1 runs three sandboxes in parallel (A: verification capabilities,
    B: data model, C: agent context). Tier 2 runs one sandbox (D+E: injection
    surface + task coverage) with validated Tier 1 outputs as inputs.

    When *site_url* is provided, a connectivity file is staged into the Tier 2
    sandbox so the LLM can verify mechanical claims against the live instance.
    When *verification_proxy* is also provided, the connectivity URL is
    rewritten to the proxy port and an ``auth_header`` field is included so
    the sandbox can pass the token in curl requests.
    """
    staging_root, staging_dir = _stage_benchmark_files(file_list, benchmark_root, site_name)
    try:
        benchmark_volume = await upload_to_volume(staging_dir)
        benchmark_mount = {"/workspace/benchmark": _read_only_volume(benchmark_volume)}
        site_files: dict[str, str] = {}
        manifest_eval_type_set = {
            str(eval_type)
            for eval_type in manifest.get("evaluation", {}).get("eval_types", [])
            if eval_type
        }
        logger.info(
            "Phase 0c: profiling site %r (%d files staged), tier 1 starting",
            site_name,
            len(file_list),
        )

        # ── Tier 1: parallel ────────────────────────────────────────────
        tier1_results = await asyncio.gather(
            _run_tier_json_with_retries(
                site_name=site_name,
                site_files=site_files,
                prompt_name="profile-verification-capabilities",
                validation_command=f"verification-capabilities --site-name {site_name}",
                output_path="/workspace/output/VERIFICATION_CAPABILITIES.json",
                timeout=timeout,
                label=f"0c-{site_name}-A-verify",
                sandbox_model=sandbox_model,
                validate_parsed=lambda data: (
                    validate_verification_capabilities(data, site_name=site_name)
                    + _validate_manifest_eval_types(data, manifest_eval_type_set)
                ),
                volumes=benchmark_mount,
                correction_guidance=(
                    "Only include evaluation methods that actually exist in the benchmark harness. "
                    "Each entry needs a string eval_type and description."
                ),
            ),
            _run_tier_json_with_retries(
                site_name=site_name,
                site_files=site_files,
                prompt_name="profile-data-model",
                validation_command=f"data-model --site-name {site_name}",
                output_path="/workspace/output/DATA_MODEL.json",
                timeout=timeout,
                label=f"0c-{site_name}-B-data",
                sandbox_model=sandbox_model,
                validate_parsed=lambda data: validate_data_model_profile(data, site_name=site_name),
                volumes=benchmark_mount,
                correction_guidance=(
                    "Every entity must declare a non-empty fields array, and every field name should "
                    "match the entity it belongs to."
                ),
            ),
            _run_tier_json_with_retries(
                site_name=site_name,
                site_files=site_files,
                prompt_name="profile-agent-context",
                validation_command=f"agent-context --site-name {site_name}",
                output_path="/workspace/output/AGENT_CONTEXT.json",
                timeout=timeout,
                label=f"0c-{site_name}-C-context",
                sandbox_model=sandbox_model,
                validate_parsed=lambda data: validate_agent_context(data, site_name=site_name),
                volumes=benchmark_mount,
                correction_guidance=(
                    "When structured output is required, output_schema must be a JSON object. "
                    "If agent_prompt_template is present, it must contain both {{INSTRUCTION}} "
                    "and {{START_URLS}} placeholders."
                ),
            ),
            return_exceptions=True,
        )

        # Extract validated Tier 1 outputs
        tier1_names = ["verification-capabilities", "data-model", "agent-context"]
        tier1_parsed: list[Any] = []
        for result, name in zip(tier1_results, tier1_names):
            if isinstance(result, Exception):
                raise RuntimeError(
                    f"Tier 1 sandbox {name} for site {site_name} failed: {result}"
                ) from result
            tier1_parsed.append(result)

        verify_caps, data_model, agent_context = tier1_parsed
        logger.info("Phase 0c: site %r tier 1 complete, starting tier 2", site_name)

        # ── Tier 2: receives Tier 1 outputs ─────────────────────────────
        # Stage Tier 1 outputs as input files for the D+E sandbox.
        inputs_dir = staging_root / "tier1_inputs"
        inputs_dir.mkdir()
        (inputs_dir / "VERIFICATION_CAPABILITIES.json").write_text(
            json.dumps(verify_caps, indent=2)
        )
        (inputs_dir / "DATA_MODEL.json").write_text(json.dumps(data_model, indent=2))
        (inputs_dir / "AGENT_CONTEXT.json").write_text(json.dumps(agent_context, indent=2))

        # Stage instance connectivity for live verification.
        # When a verification proxy is configured, rewrite the URL to the
        # proxy port and include the auth header for curl requests.
        if site_url:
            effective_url = site_url
            if verification_proxy is not None:
                effective_url = _apply_proxy_to_url(
                    site_url,
                    verification_proxy.port_offset,
                    scheme=verification_proxy.scheme,
                )
            connectivity: dict[str, str] = {
                "site_name": site_name,
                "site_url": effective_url,
            }
            if verification_proxy is not None:
                connectivity["auth_header"] = f"X-Worldsim-Token: {verification_proxy.token}"
            (inputs_dir / "INSTANCE_CONNECTIVITY.json").write_text(
                json.dumps(connectivity, indent=2)
            )
            logger.info(
                "Phase 0c: site %r tier 2 will have instance connectivity (%s%s)",
                site_name,
                effective_url,
                ", proxied" if verification_proxy else "",
            )

        tier2_extra_inputs = {
            "/workspace/inputs/VERIFICATION_CAPABILITIES.json": str(
                inputs_dir / "VERIFICATION_CAPABILITIES.json"
            ),
            "/workspace/inputs/DATA_MODEL.json": str(inputs_dir / "DATA_MODEL.json"),
            "/workspace/inputs/AGENT_CONTEXT.json": str(inputs_dir / "AGENT_CONTEXT.json"),
        }

        # Only stage connectivity file when an instance URL was provided.
        if site_url:
            tier2_extra_inputs["/workspace/inputs/INSTANCE_CONNECTIVITY.json"] = str(
                inputs_dir / "INSTANCE_CONNECTIVITY.json"
            )

        injection_surface = await _run_tier_json_with_retries(
            site_name=site_name,
            site_files=site_files,
            prompt_name="profile-injection-surface",
            validation_command=f"injection-surface --site-name {site_name}",
            output_path="/workspace/output/INJECTION_SURFACE.json",
            timeout=timeout,
            label=f"0c-{site_name}-DE-inject",
            sandbox_model=sandbox_model,
            extra_inputs=tier2_extra_inputs,
            volumes=benchmark_mount,
            validate_parsed=lambda data: validate_injection_surface(
                data,
                site_name=site_name,
                data_model=data_model,
                agent_context=agent_context,
            ),
            correction_guidance=(
                "Every source_field must use entity.field format and reference a real field on the "
                "matching entity in DATA_MODEL.json. existing_task_coverage may only reference ids "
                "declared in injection_surface."
            ),
        )

        logger.info("Phase 0c: site %r tier 2 complete, merging profile", site_name)

        # ── Merge into BENCHMARK_PROFILE ────────────────────────────────
        profile = {
            "site_name": site_name,
            "verification_capabilities": verify_caps,
            "data_model": data_model,
            "agent_context": agent_context,
            "injection_surface": injection_surface.get("injection_surface", []),
            "existing_task_coverage": injection_surface.get("existing_task_coverage", {}),
        }

        # Validate the merged profile before publishing anything to disk.
        validate_profile(
            site_name,
            profile,
            manifest_eval_types=manifest_eval_type_set,
        )

        # ── Write outputs ───────────────────────────────────────────────
        output_dir.mkdir(parents=True, exist_ok=True)

        profile_path = output_dir / f"BENCHMARK_PROFILE_{site_name}.json"
        _write_text_atomic(profile_path, json.dumps(profile, indent=2))
        logger.info("Phase 0c: wrote %s", profile_path)

        context_path = output_dir / f"AGENT_CONTEXT_{site_name}.json"
        _write_text_atomic(context_path, json.dumps(agent_context, indent=2))
        logger.info("Phase 0c: wrote %s", context_path)

        _write_text_atomic(
            _profile_metadata_path(output_dir, site_name),
            json.dumps(
                {
                    "site_name": site_name,
                    "benchmark_root": str(benchmark_root),
                    "sandbox_model": sandbox_model,
                    "instance_site_url": site_url,
                    "verification_proxy": _verification_proxy_metadata(verification_proxy),
                },
                indent=2,
            ),
        )

        # Write individual tier outputs for debugging/inspection
        for tier_name, tier_data in [
            ("VERIFICATION_CAPABILITIES", verify_caps),
            ("DATA_MODEL", data_model),
            ("INJECTION_SURFACE", injection_surface),
        ]:
            tier_path = output_dir / f"{tier_name}_{site_name}.json"
            _write_text_atomic(tier_path, json.dumps(tier_data, indent=2))

        return site_name, {
            "profile": profile,
            "agent_context": agent_context,
        }

    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


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
