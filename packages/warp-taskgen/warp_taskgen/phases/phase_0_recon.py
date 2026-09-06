"""Phase 0: Benchmark reconnaissance.

Canonical source: ``docs/warp-taskgen-technical-spec.md`` "Phase 0: Benchmark
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
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from warp_taskgen._sandbox_validator import (
    validate_agent_context,
    validate_data_model_profile,
    validate_injection_surface,
    validate_verification_capabilities,
)
from warp_taskgen.config import BenchmarkInstance, VerificationProxy, load_benchmark_config
from warp_taskgen.cost_tracker import tracker as cost_tracker
from warp_taskgen.modal_sandbox import preflight_sandbox_environment, upload_to_volume
from warp_taskgen.phases.phase_0_evidence_index import (
    benchmark_digest_from_evidence_payloads,
    build_phase_0c_evidence_payloads,
    hash_phase_0c_evidence_payloads,
    write_phase_0c_evidence_indexes,
)
from warp_taskgen.phases.phase_0a_benchmark_manifest import (
    _read_only_volume,
    _validate_manifest_eval_types,
    compute_sandbox_maps,
    run_phase_0a,
)
from warp_taskgen.phases.phase_0c_artifacts import (
    Phase0cTraceWriter,
    file_sha256,
    load_reusable_tier_output,
    phase_0c_timings_path,
)
from warp_taskgen.phases.phase_0c_instance_reachability import (
    _apply_proxy_to_url,
    _build_instance_groups,
    _build_instance_lookup,
    _build_instance_site_url_map,
    _phase_0c_redact_values,
    _site_reachability_record,
    _validate_phase_0c_modal_connectivity_urls,
    _verification_proxy_metadata,
    _write_reachability_report,
    _write_text_atomic,
)
from warp_taskgen.phases.phase_0c_profile_enrichment import (
    _enrich_agent_context_with_handles,
    _enrich_gitlab_profile_with_projects,
    _enrich_reddit_profile_with_forums,
)
from warp_taskgen.phases.phase_0c_profile_reuse import (
    _DATA_MODEL_SIDECARS,
    _INJECTION_SURFACE_SIDECARS,
    _existing_site_outputs_are_reusable,
    _expected_tier_metadata,
    _instance_inventory_fingerprint,
    _profile_metadata_path,
    _tier1_input_hashes,
)
from warp_taskgen.phases.phase_0c_tier_sandbox import (
    _run_tier_json_with_retries,
    _stage_benchmark_files,
    _tier_success_publisher,
)
from warp_taskgen.placeholders import normalize_site_name
from warp_taskgen.profile_validation import validate_profile
from warp_taskgen.state import get_state_dir, save_state

logger = logging.getLogger(__name__)


def _phase_0_state_metadata(
    *,
    benchmark: Path,
    sandbox_model: str,
    instances_path: Path | None,
    host_inventory_instances_path: Path | None = None,
) -> dict[str, str]:
    metadata = {
        "benchmark_path": str(benchmark),
        "sandbox_model": sandbox_model,
    }
    if instances_path is not None:
        metadata["instances_path"] = str(Path(instances_path))
    if host_inventory_instances_path is not None:
        metadata["host_inventory_instances_path"] = str(Path(host_inventory_instances_path))
        metadata["host_inventory_instances_sha256"] = _file_sha256(
            Path(host_inventory_instances_path)
        )
    return metadata


def _file_sha256(path: Path) -> str:
    return file_sha256(path)


def _load_phase_0c_config(
    instances_path: Path,
) -> tuple[list[BenchmarkInstance], VerificationProxy | None]:
    """Load instances and optional verification proxy config from instances.json.

    Returns (instances, verification_proxy). The proxy is None when absent or
    when its token is empty (disabled).
    """

    path = Path(instances_path)
    try:
        config = load_benchmark_config(path)
    except (OSError, ValueError, ValidationError) as exc:
        raise RuntimeError(f"invalid instances config at {path}: {exc}") from exc
    proxy = config.verification_proxy
    # Treat empty token as "proxy not configured".
    if proxy is not None and not proxy.token.strip():
        proxy = None
    return config.instances, proxy


async def run(
    benchmark: Path,
    sub: str = "0",
    sandbox_model: str = "claude-sonnet-4-6",
    instances_path: Path | None = None,
    host_inventory_instances_path: Path | None = None,
    site_filter: set[str] | None = None,
) -> int:
    """Phase 0 entrypoint.

    Args:
        benchmark: Path to the benchmark codebase (e.g.
            ``vendors/webarena-verified``).
        sub: One of ``"0"`` (full phase), ``"0a"``, ``"0b"``, or ``"0c"``.
        instances_path: Optional path to instances.json. When provided,
            Phase 0c sandboxes receive instance connectivity info
            (site URLs only, no credentials).
        host_inventory_instances_path: Optional host-local instances file for
            Phase 0c host-side inventory enrichment. On r5 this is normally
            ``instances.scale.json`` while ``instances_path`` remains
            ``instances.smoke.json`` for Modal browser probes.
        site_filter: Optional normalized site names to profile in Phase 0c.
            Phase 0a/0b still discover the full benchmark manifest and sandbox
            map so downstream phases can validate provenance.

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
        host_inventory_instances_path=host_inventory_instances_path,
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
            host_inventory_instances: list[BenchmarkInstance] | None = None
            verification_proxy = None
            if instances_path is not None:
                instances, verification_proxy = _load_phase_0c_config(instances_path)
            if host_inventory_instances_path is not None:
                host_inventory_instances, _ = _load_phase_0c_config(host_inventory_instances_path)
            await run_phase_0c(
                manifest,
                sandbox_map,
                benchmark,
                output_base / "phase_0c",
                sandbox_model=sandbox_model,
                instances=instances,
                host_inventory_instances=host_inventory_instances,
                verification_proxy=verification_proxy,
                site_filter=site_filter,
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


# ---------------------------------------------------------------------------
# Phase 0b — Sandbox Filesystem Mapping
# ---------------------------------------------------------------------------


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
    host_inventory_instances: list[BenchmarkInstance] | None = None,
    verification_proxy: VerificationProxy | None = None,
    site_filter: set[str] | None = None,
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
        host_inventory_instances: Optional list of validated benchmark
            instances used only by host-side inventory enrichment hooks. This
            lets r5 keep Modal-facing Phase 0c URLs public/proxied while
            enriching Reddit/GitLab from host-local scale topology.
        verification_proxy: Optional proxy config. When present, site URLs
            are rewritten to proxy ports and an auth header is included in
            the INSTANCE_CONNECTIVITY.json staged into Tier 2 sandboxes.
        site_filter: Optional normalized site names to profile. Existing
            outputs for unselected sites are ignored rather than deleted.

    Returns:
        Dict mapping site name to merged profile outputs.
    """
    benchmark_root = Path(benchmark_root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_writer = Phase0cTraceWriter(output_dir)
    trace_writer.record(
        "phase_0c_started",
        benchmark_root=str(benchmark_root),
        sandbox_model=sandbox_model,
        site_filter=sorted(site_filter or []),
    )
    manifest_eval_type_set = {
        str(eval_type)
        for eval_type in manifest.get("evaluation", {}).get("eval_types", [])
        if eval_type
    }
    instance_urls = _build_instance_site_url_map(instances)
    _validate_phase_0c_modal_connectivity_urls(instance_urls)
    instance_lookup = _build_instance_lookup(instances)
    host_inventory_lookup = (
        _build_instance_lookup(host_inventory_instances)
        if host_inventory_instances is not None
        else instance_lookup
    )
    host_inventory_groups = (
        _build_instance_groups(host_inventory_instances)
        if host_inventory_instances is not None
        else _build_instance_groups(instances)
    )

    normalized_site_filter = (
        {normalize_site_name(site) for site in site_filter if str(site).strip()}
        if site_filter is not None
        else None
    )

    # Skip sites that already have all outputs and tier provenance on disk.
    sites_to_profile: dict[str, dict[str, Any]] = {}
    reachability_records: list[dict[str, Any]] = []
    for name, files in sandbox_map.items():
        normalized_name = normalize_site_name(name)
        if normalized_site_filter is not None and normalized_name not in normalized_site_filter:
            logger.info("Phase 0c: skipping site %r (not selected by --sites)", name)
            continue
        instance_site_url = instance_urls.get(normalized_name)
        instance_record = instance_lookup.get(normalized_name)
        host_inventory_instance = host_inventory_lookup.get(normalized_name) or instance_record
        host_inventory_instances_for_site = host_inventory_groups.get(normalized_name, [])
        evidence_payloads = build_phase_0c_evidence_payloads(
            file_list=files,
            benchmark_root=benchmark_root,
            manifest=manifest,
            site_name=name,
        )
        benchmark_digest = benchmark_digest_from_evidence_payloads(evidence_payloads)
        evidence_index_digest = hash_phase_0c_evidence_payloads(evidence_payloads)
        if _existing_site_outputs_are_reusable(
            output_dir=output_dir,
            site_name=name,
            benchmark_root=benchmark_root,
            benchmark_digest=benchmark_digest,
            evidence_index_digest=evidence_index_digest,
            sandbox_model=sandbox_model,
            manifest_eval_type_set=manifest_eval_type_set,
            instance_site_url=instance_site_url,
            host_inventory_instance=host_inventory_instance,
            verification_proxy=verification_proxy,
        ):
            logger.info("Phase 0c: skipping site %r (profile + agent context already exist)", name)
            trace_writer.record("site_reused", site_name=name, instance_site_url=instance_site_url)
            injection_surface_path = output_dir / f"INJECTION_SURFACE_{name}.json"
            injection_surface = None
            if injection_surface_path.exists():
                try:
                    injection_surface = json.loads(injection_surface_path.read_text())
                except (OSError, json.JSONDecodeError):
                    injection_surface = None
            reachability_records.append(
                _site_reachability_record(
                    site_name=name,
                    site_url=instance_site_url,
                    verification_proxy=verification_proxy,
                    injection_surface=injection_surface,
                    cached=True,
                )
            )
        else:
            sites_to_profile[name] = {
                "files": files,
                "evidence_payloads": evidence_payloads,
                "benchmark_digest": benchmark_digest,
                "evidence_index_digest": evidence_index_digest,
                "site_url": instance_site_url,
                "instance": instance_record,
                "host_inventory_instance": host_inventory_instance,
                "host_inventory_instances": host_inventory_instances_for_site,
            }

    if not sites_to_profile:
        logger.info("Phase 0c: all sites already profiled, nothing to do")
        _write_reachability_report(output_dir, reachability_records)
        trace_writer.record(
            "phase_0c_completed", profiled_sites=0, cached_sites=len(reachability_records)
        )
        trace_writer.write_timings_summary()
        return {}

    raw_results = await asyncio.gather(
        *[
            _profile_one_site_tiered(
                site_name=name,
                file_list=site_plan["files"],
                benchmark_root=benchmark_root,
                output_dir=output_dir,
                manifest=manifest,
                timeout=timeout,
                sandbox_model=sandbox_model,
                site_url=site_plan["site_url"],
                verification_proxy=verification_proxy,
                instance=site_plan["instance"],
                host_inventory_instance=site_plan["host_inventory_instance"],
                host_inventory_instances=site_plan["host_inventory_instances"],
                trace_writer=trace_writer,
                evidence_payloads=site_plan["evidence_payloads"],
                benchmark_digest=site_plan["benchmark_digest"],
                evidence_index_digest=site_plan["evidence_index_digest"],
            )
            for name, site_plan in sites_to_profile.items()
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
            if isinstance(site_outputs, dict) and isinstance(
                site_outputs.get("reachability"), dict
            ):
                reachability_records.append(site_outputs["reachability"])

    expected_sites = set(sites_to_profile)
    missing_sites = sorted(expected_sites - set(results))
    failures.extend(f"missing profile result for site {site}" for site in missing_sites)

    if failures:
        trace_writer.write_timings_summary(failures=failures)
        raise RuntimeError(
            "Phase 0c did not complete all required site profiles:\n"
            + "\n".join(f"  - {failure}" for failure in failures)
        )

    _write_reachability_report(output_dir, reachability_records)
    trace_writer.record(
        "phase_0c_completed",
        profiled_sites=len(results),
        cached_sites=len(reachability_records) - len(results),
    )
    trace_writer.write_timings_summary()
    return results


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
    instance: BenchmarkInstance | None = None,
    host_inventory_instance: BenchmarkInstance | None = None,
    host_inventory_instances: list[BenchmarkInstance] | None = None,
    trace_writer: Phase0cTraceWriter | None = None,
    evidence_payloads: dict[str, object] | None = None,
    benchmark_digest: str | None = None,
    evidence_index_digest: str | None = None,
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
        if evidence_payloads is None:
            evidence_payloads = build_phase_0c_evidence_payloads(
                file_list=file_list,
                benchmark_root=benchmark_root,
                manifest=manifest,
                site_name=site_name,
            )
        if benchmark_digest is None:
            benchmark_digest = benchmark_digest_from_evidence_payloads(evidence_payloads)
        if evidence_index_digest is None:
            evidence_index_digest = hash_phase_0c_evidence_payloads(evidence_payloads)
        evidence_inputs = write_phase_0c_evidence_indexes(
            evidence_payloads,
            output_dir=staging_root / "phase0c_evidence",
        )
        proxy_metadata = _verification_proxy_metadata(verification_proxy)
        host_inventory_fingerprint = _instance_inventory_fingerprint(
            host_inventory_instance or instance
        )
        redact_values = _phase_0c_redact_values(verification_proxy)
        if trace_writer is not None:
            trace_writer.record(
                "site_started",
                site_name=site_name,
                file_count=len(file_list),
                benchmark_digest=benchmark_digest,
                evidence_index_digest=evidence_index_digest,
            )
        logger.info(
            "Phase 0c: profiling site %r (%d files staged), tier 1 starting",
            site_name,
            len(file_list),
        )

        def verify_validate(data: object) -> list[str]:
            return validate_verification_capabilities(
                data, site_name=site_name
            ) + _validate_manifest_eval_types(data, manifest_eval_type_set)

        def data_validate(data: object) -> list[str]:
            return validate_data_model_profile(data, site_name=site_name)

        def context_validate(data: object) -> list[str]:
            return validate_agent_context(data, site_name=site_name)

        verify_metadata = _expected_tier_metadata(
            site_name=site_name,
            tier_name="A_VERIFICATION_CAPABILITIES",
            prompt_name="profile-verification-capabilities",
            validation_command=f"verification-capabilities --site-name {site_name}",
            output_path="/workspace/output/VERIFICATION_CAPABILITIES.json",
            sandbox_model=sandbox_model,
            benchmark_digest=benchmark_digest,
            manifest_eval_type_set=manifest_eval_type_set,
            instance_site_url=None,
            verification_proxy_metadata=None,
            evidence_index_digest=evidence_index_digest,
            host_inventory_instance_fingerprint=host_inventory_fingerprint,
        )
        data_metadata = _expected_tier_metadata(
            site_name=site_name,
            tier_name="B_DATA_MODEL",
            prompt_name="profile-data-model",
            validation_command=f"data-model --site-name {site_name}",
            output_path="/workspace/output/DATA_MODEL.json",
            sandbox_model=sandbox_model,
            benchmark_digest=benchmark_digest,
            manifest_eval_type_set=manifest_eval_type_set,
            instance_site_url=None,
            verification_proxy_metadata=None,
            evidence_index_digest=evidence_index_digest,
            host_inventory_instance_fingerprint=host_inventory_fingerprint,
            required_sidecars=_DATA_MODEL_SIDECARS,
        )
        context_metadata = _expected_tier_metadata(
            site_name=site_name,
            tier_name="C_AGENT_CONTEXT",
            prompt_name="profile-agent-context",
            validation_command=f"agent-context --site-name {site_name}",
            output_path="/workspace/output/AGENT_CONTEXT.json",
            sandbox_model=sandbox_model,
            benchmark_digest=benchmark_digest,
            manifest_eval_type_set=manifest_eval_type_set,
            instance_site_url=None,
            verification_proxy_metadata=None,
            evidence_index_digest=evidence_index_digest,
            host_inventory_instance_fingerprint=host_inventory_fingerprint,
        )

        async def reuse_or_run_tier(
            *,
            tier_name: str,
            artifact_stem: str,
            prompt_name: str,
            validation_command: str,
            output_path: str,
            label: str,
            metadata: dict[str, Any],
            validate_parsed: Callable[[object], list[str]],
            correction_guidance: str,
            sidecar_outputs: dict[str, str] | None = None,
            extra_inputs_for_tier: dict[str, str] | None = None,
        ) -> Any:
            cached = load_reusable_tier_output(
                output_dir=output_dir,
                site_name=site_name,
                tier_name=tier_name,
                artifact_stem=artifact_stem,
                expected_metadata=metadata,
                validate_parsed=validate_parsed,
                required_sidecars=tuple((sidecar_outputs or {}).values()),
                redact_values=redact_values,
            )
            if cached is not None:
                logger.info("Phase 0c: site %r reusing tier %s", site_name, tier_name)
                if trace_writer is not None:
                    trace_writer.record(
                        "tier_reused",
                        site_name=site_name,
                        tier_name=tier_name,
                        artifact_stem=artifact_stem,
                    )
                return cached
            return await _run_tier_json_with_retries(
                site_name=site_name,
                site_files=site_files,
                prompt_name=prompt_name,
                validation_command=validation_command,
                output_path=output_path,
                timeout=timeout,
                label=label,
                sandbox_model=sandbox_model,
                validate_parsed=validate_parsed,
                extra_inputs=extra_inputs_for_tier
                if extra_inputs_for_tier is not None
                else evidence_inputs,
                volumes=benchmark_mount,
                correction_guidance=correction_guidance,
                side_output_paths=list(sidecar_outputs or {}),
                on_success_outputs=_tier_success_publisher(
                    output_dir=output_dir,
                    site_name=site_name,
                    tier_name=tier_name,
                    artifact_stem=artifact_stem,
                    output_path=output_path,
                    metadata=metadata,
                    sidecar_outputs=sidecar_outputs,
                    redact_values=redact_values,
                ),
                redact_values=redact_values,
                trace_writer=trace_writer,
                trace_context={"tier_name": tier_name, "artifact_stem": artifact_stem},
            )

        # ── Tier 1: parallel ────────────────────────────────────────────
        tier1_results = await asyncio.gather(
            reuse_or_run_tier(
                tier_name="A_VERIFICATION_CAPABILITIES",
                artifact_stem="VERIFICATION_CAPABILITIES",
                prompt_name="profile-verification-capabilities",
                validation_command=f"verification-capabilities --site-name {site_name}",
                output_path="/workspace/output/VERIFICATION_CAPABILITIES.json",
                label=f"0c-{site_name}-A-verify",
                metadata=verify_metadata,
                validate_parsed=verify_validate,
                correction_guidance=(
                    "Only include evaluation methods that actually exist in the benchmark harness. "
                    "Each entry needs a string eval_type and description."
                ),
            ),
            reuse_or_run_tier(
                tier_name="B_DATA_MODEL",
                artifact_stem="DATA_MODEL",
                prompt_name="profile-data-model",
                validation_command=f"data-model --site-name {site_name}",
                output_path="/workspace/output/DATA_MODEL.json",
                label=f"0c-{site_name}-B-data",
                metadata=data_metadata,
                validate_parsed=data_validate,
                correction_guidance=(
                    "Every entity must declare a non-empty fields array, and every field name should "
                    "match the entity it belongs to."
                ),
                sidecar_outputs={
                    "/workspace/output/DATA_MODEL_EVIDENCE.json": "DATA_MODEL_EVIDENCE"
                },
            ),
            reuse_or_run_tier(
                tier_name="C_AGENT_CONTEXT",
                artifact_stem="AGENT_CONTEXT_RAW",
                prompt_name="profile-agent-context",
                validation_command=f"agent-context --site-name {site_name}",
                output_path="/workspace/output/AGENT_CONTEXT.json",
                label=f"0c-{site_name}-C-context",
                metadata=context_metadata,
                validate_parsed=context_validate,
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
        for result, name in zip(tier1_results, tier1_names, strict=False):
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
        tier2_extra_inputs.update(evidence_inputs)

        # Only stage connectivity file when an instance URL was provided.
        if site_url:
            tier2_extra_inputs["/workspace/inputs/INSTANCE_CONNECTIVITY.json"] = str(
                inputs_dir / "INSTANCE_CONNECTIVITY.json"
            )

        def injection_validate(data: object) -> list[str]:
            return validate_injection_surface(
                data,
                site_name=site_name,
                data_model=data_model,
                agent_context=agent_context,
            )

        tier1_input_hashes = _tier1_input_hashes(
            verify_caps=verify_caps,
            data_model=data_model,
            agent_context=agent_context,
        )
        injection_metadata = _expected_tier_metadata(
            site_name=site_name,
            tier_name="DE_INJECTION_SURFACE",
            prompt_name="profile-injection-surface",
            validation_command=f"injection-surface --site-name {site_name}",
            output_path="/workspace/output/INJECTION_SURFACE.json",
            sandbox_model=sandbox_model,
            benchmark_digest=benchmark_digest,
            manifest_eval_type_set=manifest_eval_type_set,
            instance_site_url=site_url,
            verification_proxy_metadata=proxy_metadata,
            evidence_index_digest=evidence_index_digest,
            host_inventory_instance_fingerprint=host_inventory_fingerprint,
            tier_input_hashes=tier1_input_hashes,
            required_sidecars=_INJECTION_SURFACE_SIDECARS,
        )
        injection_surface = await reuse_or_run_tier(
            tier_name="DE_INJECTION_SURFACE",
            artifact_stem="INJECTION_SURFACE",
            prompt_name="profile-injection-surface",
            validation_command=f"injection-surface --site-name {site_name}",
            output_path="/workspace/output/INJECTION_SURFACE.json",
            label=f"0c-{site_name}-DE-inject",
            metadata=injection_metadata,
            validate_parsed=injection_validate,
            correction_guidance=(
                "Every source_field must use entity.field format and reference a real field on the "
                "matching entity in DATA_MODEL.json. existing_task_coverage may only reference ids "
                "declared in injection_surface."
            ),
            sidecar_outputs={
                "/workspace/output/SURFACE_DRAFT.json": "SURFACE_DRAFT",
                "/workspace/output/TASK_COVERAGE_DRAFT.json": "TASK_COVERAGE_DRAFT",
                "/workspace/output/LIVE_VERIFICATION_NOTES.json": "LIVE_VERIFICATION_NOTES",
            },
            extra_inputs_for_tier=tier2_extra_inputs,
        )
        reachability = _site_reachability_record(
            site_name=site_name,
            site_url=site_url,
            verification_proxy=verification_proxy,
            injection_surface=injection_surface if isinstance(injection_surface, dict) else None,
        )

        logger.info("Phase 0c: site %r tier 2 complete, merging profile", site_name)

        # ── Host-side handle enrichment (gitlab only) ───────────────────
        # Phase 2's URL-shape resolver disambiguates `/<segment>` gitlab
        # URLs as user_profile vs group via these handle lists. Best
        # effort: enrichment failure logs and continues; the resolver
        # gracefully degrades to kind=None for ambiguous segments.
        inventory_instance = host_inventory_instance or instance
        agent_context = _enrich_agent_context_with_handles(
            site_name=site_name,
            agent_context=agent_context,
            instance=inventory_instance,
        )

        # ── Merge into BENCHMARK_PROFILE ────────────────────────────────
        profile = {
            "site_name": site_name,
            "verification_capabilities": verify_caps,
            "data_model": data_model,
            "agent_context": agent_context,
            "injection_surface": injection_surface.get("injection_surface", []),
            "existing_task_coverage": injection_surface.get("existing_task_coverage", {}),
        }
        profile = _enrich_gitlab_profile_with_projects(
            site_name=site_name,
            profile=profile,
            instance=inventory_instance,
        )
        profile = _enrich_reddit_profile_with_forums(
            site_name=site_name,
            profile=profile,
            instance=inventory_instance,
            instances=host_inventory_instances,
        )

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
                    "provenance_schema_version": 1,
                    "site_name": site_name,
                    "benchmark_root": str(benchmark_root),
                    "sandbox_model": sandbox_model,
                    "benchmark_digest": benchmark_digest,
                    "evidence_index_digest": evidence_index_digest,
                    "instance_site_url": site_url,
                    "host_inventory_instance_fingerprint": _instance_inventory_fingerprint(
                        inventory_instance
                    ),
                    "verification_proxy": _verification_proxy_metadata(verification_proxy),
                    "trace_artifacts": {
                        "trace": "PHASE_0C_TRACE.jsonl",
                        "timings": phase_0c_timings_path(output_dir).name,
                    },
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

        if trace_writer is not None:
            trace_writer.record(
                "site_completed",
                site_name=site_name,
                injection_surface_count=len(profile.get("injection_surface", [])),
            )

        return site_name, {
            "profile": profile,
            "agent_context": agent_context,
            "reachability": reachability,
        }

    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
