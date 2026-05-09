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
import hashlib
import ipaddress
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
from worldsim.config import BenchmarkInstance, VerificationProxy, load_benchmark_config
from worldsim.cost_tracker import tracker as cost_tracker
from worldsim.modal_sandbox import (
    preflight_sandbox_environment,
    run_claude_in_sandbox,
    upload_to_volume,
)
from worldsim.phases.phase_0_evidence_index import (
    benchmark_digest_from_evidence_payloads,
    build_phase_0c_evidence_payloads,
    hash_phase_0c_evidence_payloads,
    write_phase_0c_evidence_indexes,
)
from worldsim.phases.phase_0c_artifacts import (
    Phase0cTraceWriter,
    build_tier_metadata,
    file_sha256,
    hash_json,
    load_reusable_tier_output,
    phase_0c_timings_path,
    profile_metadata_path,
    publish_tier_output,
    reachability_report_path,
    redact_json_secrets,
    text_sha256,
    write_text_atomic,
)
from worldsim.placeholders import normalize_site_name
from worldsim.profile_validation import validate_profile
from worldsim.prompt_corrections import render_validation_feedback
from worldsim.prompt_loading import load_prompt
from worldsim.state import get_state_dir, save_state

logger = logging.getLogger(__name__)

# Maximum number of correction retries for profile validation (initial attempt + this many).
PROFILE_FIX_MAX_ITERATIONS = 2
_DATA_MODEL_SIDECARS = ("DATA_MODEL_EVIDENCE",)
_INJECTION_SURFACE_SIDECARS = (
    "SURFACE_DRAFT",
    "TASK_COVERAGE_DRAFT",
    "LIVE_VERIFICATION_NOTES",
)


def _write_text_atomic(path: Path, text: str) -> None:
    """Atomically replace *path* with *text*."""
    write_text_atomic(path, text)


def _profile_metadata_path(output_dir: Path, site_name: str) -> Path:
    return profile_metadata_path(output_dir, site_name)


def _reachability_report_path(output_dir: Path) -> Path:
    return reachability_report_path(output_dir)


def _read_only_volume(volume: Any) -> Any:
    """Return a read-only mount when the object supports it."""
    read_only = getattr(volume, "read_only", None)
    return read_only() if callable(read_only) else volume


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


def _phase_0c_modal_unreachable_host_reason(site_url: str) -> str | None:
    parts = urlsplit(site_url)
    host = parts.hostname
    if not host:
        return "missing host"
    host_lower = host.lower()
    if host_lower in {"localhost"} or host_lower.endswith(".local"):
        return f"non-public hostname {host!r}"
    try:
        address = ipaddress.ip_address(host_lower)
    except ValueError:
        return None
    if (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    ):
        return f"non-public IP address {host!r}"
    return None


def _validate_phase_0c_modal_connectivity_urls(
    instance_urls: dict[str, str],
    *,
    instances_path: Path | None = None,
) -> None:
    """Fail early when Modal Phase 0c would receive host-local URLs."""
    invalid = []
    for site_name, site_url in sorted(instance_urls.items()):
        reason = _phase_0c_modal_unreachable_host_reason(site_url)
        if reason:
            invalid.append(f"{site_name}: {site_url} ({reason})")
    if not invalid:
        return
    path_hint = f" from {instances_path}" if instances_path is not None else ""
    raise RuntimeError(
        "Phase 0c live verification runs inside Modal sandboxes and requires "
        f"externally reachable instance URLs{path_hint}. The selected instance "
        "config uses host-local/on-host URLs that Modal cannot reach:\n"
        + "\n".join(f"  - {item}" for item in invalid)
        + "\nUse an externally reachable/proxied instance file for Phase 0c "
        "(for r5, instances.smoke.json), and use instances.scale.json only for "
        "on-host browser phases such as Phase 2c and Phase 4."
    )


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


def _build_instance_lookup(
    instances: list[BenchmarkInstance] | None,
) -> dict[str, BenchmarkInstance]:
    """Map normalized site name to a representative ``BenchmarkInstance``.

    Used by host-side enrichment hooks (e.g. gitlab handle enumeration)
    that need access to the instance's auth config, not just its URL.
    Picks the first instance per site; replicas share auth.
    """
    if not instances:
        return {}
    out: dict[str, BenchmarkInstance] = {}
    for instance in instances:
        normalized_site = normalize_site_name(instance.site_name)
        if not normalized_site or normalized_site in out:
            continue
        out[normalized_site] = instance
    return out


def _build_instance_groups(
    instances: list[BenchmarkInstance] | None,
) -> dict[str, list[BenchmarkInstance]]:
    """Map normalized site name to all configured instances for that site."""
    if not instances:
        return {}
    out: dict[str, list[BenchmarkInstance]] = {}
    for instance in instances:
        normalized_site = normalize_site_name(instance.site_name)
        if not normalized_site:
            continue
        out.setdefault(normalized_site, []).append(instance)
    return out


def _instance_inventory_fingerprint(instance: BenchmarkInstance | None) -> str | None:
    """Return a non-secret cache fingerprint for host-side inventory inputs."""
    if instance is None:
        return None
    payload = instance.model_dump(mode="json", exclude_none=True)
    # Store only a digest in profile metadata because auth/db fields can carry
    # credentials. The digest still invalidates stale profiles when the host
    # inventory topology changes.
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


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


def _phase_0c_redact_values(verification_proxy: VerificationProxy | None) -> tuple[str, ...]:
    if verification_proxy is None:
        return ()
    return (
        verification_proxy.token,
        f"X-Worldsim-Token: {verification_proxy.token}",
    )


def _delivery_channel_verification_status(channel: object) -> tuple[str, str | None]:
    if not isinstance(channel, dict):
        return "malformed_channel", None
    verified = channel.get("verified")
    note = channel.get("verification_notes")
    note_text = note.strip() if isinstance(note, str) and note.strip() else None
    if verified is True:
        return "verified", note_text
    if verified is False:
        return "discrepancy_corrected", note_text
    if verified is None:
        return "unverified", note_text
    return "malformed_verified_field", note_text


def _site_reachability_record(
    *,
    site_name: str,
    site_url: str | None,
    verification_proxy: VerificationProxy | None,
    injection_surface: dict[str, Any] | None,
    cached: bool = False,
) -> dict[str, Any]:
    """Summarize Phase 0c live verification health without gating profiles."""
    proxy_metadata = _verification_proxy_metadata(verification_proxy)
    if not site_url:
        return {
            "site": site_name,
            "status": "no_instance_config",
            "cached": cached,
            "site_url": None,
            "verification_proxy": proxy_metadata,
            "channel_counts": {},
            "notes": ["Phase 0c had no instance URL; profile is code-derived only."],
        }

    surfaces = []
    if isinstance(injection_surface, dict) and isinstance(
        injection_surface.get("injection_surface"), list
    ):
        surfaces = [
            item for item in injection_surface["injection_surface"] if isinstance(item, dict)
        ]

    channel_counts: dict[str, int] = {}
    notes: list[str] = []
    for surface in surfaces:
        channels = surface.get("delivery_channels")
        if not isinstance(channels, list):
            continue
        for channel in channels:
            status, note = _delivery_channel_verification_status(channel)
            channel_counts[status] = channel_counts.get(status, 0) + 1
            if note and len(notes) < 8:
                notes.append(f"{surface.get('id') or 'unknown'}: {note}")

    if not channel_counts:
        status = "no_channels"
    elif channel_counts.get("unverified") or channel_counts.get("malformed_verified_field"):
        status = "unverified"
    elif channel_counts.get("discrepancy_corrected"):
        status = "verified_with_corrections"
    elif channel_counts.get("verified"):
        status = "verified"
    else:
        status = "unknown"

    return {
        "site": site_name,
        "status": status,
        "cached": cached,
        "site_url": site_url,
        "verification_proxy": proxy_metadata,
        "channel_counts": channel_counts,
        "notes": notes,
    }


def _write_reachability_report(output_dir: Path, records: list[dict[str, Any]]) -> None:
    payload = {
        "schema_version": 1,
        "phase": "phase_0c",
        "sites": sorted(records, key=lambda item: str(item.get("site") or "")),
    }
    _write_text_atomic(_reachability_report_path(output_dir), json.dumps(payload, indent=2))


def _existing_site_outputs_are_reusable(
    *,
    output_dir: Path,
    site_name: str,
    benchmark_root: Path,
    sandbox_model: str,
    manifest_eval_type_set: set[str],
    instance_site_url: str | None,
    host_inventory_instance: BenchmarkInstance | None,
    verification_proxy: VerificationProxy | None,
    benchmark_digest: str | None = None,
    evidence_index_digest: str | None = None,
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
        "provenance_schema_version": 1,
        "site_name": site_name,
        "benchmark_root": str(benchmark_root),
        "sandbox_model": sandbox_model,
        "instance_site_url": instance_site_url,
        "host_inventory_instance_fingerprint": _instance_inventory_fingerprint(
            host_inventory_instance
        ),
        "verification_proxy": _verification_proxy_metadata(verification_proxy),
    }
    if benchmark_digest is not None:
        expected_metadata["benchmark_digest"] = benchmark_digest
    if evidence_index_digest is not None:
        expected_metadata["evidence_index_digest"] = evidence_index_digest
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            logger.info(
                "Phase 0c: existing outputs for site %r do not match current %s, re-profiling",
                site_name,
                key,
            )
            return False

    if benchmark_digest is None or evidence_index_digest is None:
        return False

    if not _existing_site_tiers_are_reusable(
        output_dir=output_dir,
        site_name=site_name,
        profile=profile,
        benchmark_digest=benchmark_digest,
        evidence_index_digest=evidence_index_digest,
        sandbox_model=sandbox_model,
        manifest_eval_type_set=manifest_eval_type_set,
        instance_site_url=instance_site_url,
        host_inventory_instance=host_inventory_instance,
        verification_proxy=verification_proxy,
    ):
        logger.info(
            "Phase 0c: existing tier provenance for site %r is stale or incomplete, re-profiling",
            site_name,
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


def _existing_site_tiers_are_reusable(
    *,
    output_dir: Path,
    site_name: str,
    profile: dict[str, Any],
    benchmark_digest: str,
    evidence_index_digest: str,
    sandbox_model: str,
    manifest_eval_type_set: set[str],
    instance_site_url: str | None,
    host_inventory_instance: BenchmarkInstance | None,
    verification_proxy: VerificationProxy | None,
) -> bool:
    """Return True only when all tier artifacts match the current provenance contract."""
    host_inventory_fingerprint = _instance_inventory_fingerprint(host_inventory_instance)
    proxy_metadata = _verification_proxy_metadata(verification_proxy)
    redact_values = _phase_0c_redact_values(verification_proxy)
    verify_caps = load_reusable_tier_output(
        output_dir=output_dir,
        site_name=site_name,
        tier_name="A_VERIFICATION_CAPABILITIES",
        artifact_stem="VERIFICATION_CAPABILITIES",
        expected_metadata=_expected_tier_metadata(
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
        ),
        validate_parsed=lambda data: (
            validate_verification_capabilities(data, site_name=site_name)
            + _validate_manifest_eval_types(data, manifest_eval_type_set)
        ),
        redact_values=redact_values,
    )
    data_model = load_reusable_tier_output(
        output_dir=output_dir,
        site_name=site_name,
        tier_name="B_DATA_MODEL",
        artifact_stem="DATA_MODEL",
        expected_metadata=_expected_tier_metadata(
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
        ),
        validate_parsed=lambda data: validate_data_model_profile(data, site_name=site_name),
        required_sidecars=_DATA_MODEL_SIDECARS,
        redact_values=redact_values,
    )
    agent_context_raw = load_reusable_tier_output(
        output_dir=output_dir,
        site_name=site_name,
        tier_name="C_AGENT_CONTEXT",
        artifact_stem="AGENT_CONTEXT_RAW",
        expected_metadata=_expected_tier_metadata(
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
        ),
        validate_parsed=lambda data: validate_agent_context(data, site_name=site_name),
        redact_values=redact_values,
    )
    if verify_caps is None or data_model is None or agent_context_raw is None:
        return False
    tier1_input_hashes = _tier1_input_hashes(
        verify_caps=verify_caps,
        data_model=data_model,
        agent_context=agent_context_raw,
    )
    injection_surface = load_reusable_tier_output(
        output_dir=output_dir,
        site_name=site_name,
        tier_name="DE_INJECTION_SURFACE",
        artifact_stem="INJECTION_SURFACE",
        expected_metadata=_expected_tier_metadata(
            site_name=site_name,
            tier_name="DE_INJECTION_SURFACE",
            prompt_name="profile-injection-surface",
            validation_command=f"injection-surface --site-name {site_name}",
            output_path="/workspace/output/INJECTION_SURFACE.json",
            sandbox_model=sandbox_model,
            benchmark_digest=benchmark_digest,
            manifest_eval_type_set=manifest_eval_type_set,
            instance_site_url=instance_site_url,
            verification_proxy_metadata=proxy_metadata,
            evidence_index_digest=evidence_index_digest,
            host_inventory_instance_fingerprint=host_inventory_fingerprint,
            tier_input_hashes=tier1_input_hashes,
            required_sidecars=_INJECTION_SURFACE_SIDECARS,
        ),
        validate_parsed=lambda data: validate_injection_surface(
            data,
            site_name=site_name,
            data_model=data_model,
            agent_context=agent_context_raw,
        ),
        required_sidecars=_INJECTION_SURFACE_SIDECARS,
        redact_values=redact_values,
    )
    if injection_surface is None or not isinstance(injection_surface, dict):
        return False
    return (
        profile.get("verification_capabilities") == verify_caps
        and profile.get("data_model") == data_model
        and profile.get("injection_surface") == injection_surface.get("injection_surface", [])
        and profile.get("existing_task_coverage")
        == injection_surface.get("existing_task_coverage", {})
    )


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
    return render_validation_feedback(
        artifact_name=artifact_name,
        errors=[
            {
                "code": "VALIDATION_ERROR",
                "path": "$",
                "message": error,
            }
            for error in errors
        ],
        summary=f"{artifact_name} for site {site_name} failed validation.",
        instruction=(
            "Rewrite the output file completely so it satisfies the schema and all "
            "cross-reference checks. Do not include markdown or commentary."
        ),
        extra_guidance=extra_guidance,
    )


def _expected_tier_metadata(
    *,
    site_name: str,
    tier_name: str,
    prompt_name: str,
    validation_command: str,
    output_path: str,
    sandbox_model: str,
    benchmark_digest: str,
    manifest_eval_type_set: set[str],
    instance_site_url: str | None,
    verification_proxy_metadata: dict[str, Any] | None,
    evidence_index_digest: str | None,
    host_inventory_instance_fingerprint: str | None,
    tier_input_hashes: dict[str, str] | None = None,
    required_sidecars: tuple[str, ...] = (),
) -> dict[str, Any]:
    prompt = _render_tier_prompt(
        prompt_name=prompt_name,
        validation_command=validation_command,
        site_name=site_name,
    )
    extra: dict[str, Any] = {}
    if tier_input_hashes is not None:
        extra["tier_input_sha256"] = tier_input_hashes
    if required_sidecars:
        extra["required_sidecars"] = list(required_sidecars)
    return build_tier_metadata(
        site_name=site_name,
        tier_name=tier_name,
        prompt_name=prompt_name,
        prompt_hash=text_sha256(prompt),
        validation_command=validation_command,
        output_path=output_path,
        sandbox_model=sandbox_model,
        benchmark_digest=benchmark_digest,
        manifest_eval_types=manifest_eval_type_set,
        instance_site_url=instance_site_url,
        host_inventory_instance_fingerprint=host_inventory_instance_fingerprint,
        verification_proxy=verification_proxy_metadata,
        evidence_index_digest=evidence_index_digest,
        extra=extra,
    )


def _tier1_input_hashes(
    *,
    verify_caps: object,
    data_model: object,
    agent_context: object,
) -> dict[str, str]:
    return {
        "VERIFICATION_CAPABILITIES.json": hash_json(verify_caps),
        "DATA_MODEL.json": hash_json(data_model),
        "AGENT_CONTEXT.json": hash_json(agent_context),
    }


def _tier_success_publisher(
    *,
    output_dir: Path,
    site_name: str,
    tier_name: str,
    artifact_stem: str,
    output_path: str,
    metadata: dict[str, Any],
    sidecar_outputs: dict[str, str] | None = None,
    redact_values: tuple[str, ...] = (),
) -> Callable[[dict[str, str | None]], None]:
    def publish(outputs: dict[str, str | None]) -> None:
        raw = outputs.get(output_path)
        if not raw:
            return
        payload = json.loads(raw)
        sidecars: dict[str, object] = {}
        for side_output_path, sidecar_stem in (sidecar_outputs or {}).items():
            side_raw = outputs.get(side_output_path)
            if not side_raw:
                raise ValueError(f"{Path(side_output_path).name} was not produced")
            try:
                sidecars[sidecar_stem] = json.loads(side_raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{Path(side_output_path).name} contained invalid JSON: {exc}"
                ) from exc
        publish_tier_output(
            output_dir=output_dir,
            site_name=site_name,
            tier_name=tier_name,
            artifact_stem=artifact_stem,
            payload=payload,
            metadata=metadata,
            sandbox_outputs=outputs,
            sidecars=sidecars,
            redact_values=redact_values,
        )

    return publish


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
    side_output_paths: list[str] | None = None,
    on_success_outputs: Callable[[dict[str, str | None]], None] | None = None,
    redact_values: tuple[str, ...] = (),
    trace_writer: Phase0cTraceWriter | None = None,
    trace_context: dict[str, Any] | None = None,
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
        if trace_writer is not None:
            trace_writer.record(
                "tier_attempt_started",
                site_name=site_name,
                label=attempt_label,
                attempt=attempt,
                output_path=output_path,
                **(trace_context or {}),
            )
        outputs = await _run_tier_sandbox(
            site_name=site_name,
            site_files=site_files,
            prompt=prompt,
            output_paths=[output_path, *list(side_output_paths or [])],
            timeout=timeout,
            label=attempt_label,
            sandbox_model=sandbox_model,
            extra_inputs=extra_inputs,
            volumes=volumes,
        )
        if trace_writer is not None:
            telemetry = outputs.get("_telemetry")
            trace_writer.record(
                "tier_attempt_finished",
                site_name=site_name,
                label=attempt_label,
                attempt=attempt,
                output_path=output_path,
                telemetry=telemetry,
                **(trace_context or {}),
            )

        raw = outputs.get(output_path)
        parsed: object | None = None
        errors: list[str] = []
        if not raw:
            errors.append(f"{artifact_name} was not produced")
        else:
            try:
                parsed = redact_json_secrets(
                    json.loads(raw),
                    redact_values=redact_values,
                )
            except json.JSONDecodeError as exc:
                errors.append(f"{artifact_name} contained invalid JSON: {exc}")

        for side_output_path in side_output_paths or []:
            side_name = Path(side_output_path).name
            side_raw = outputs.get(side_output_path)
            if not side_raw:
                errors.append(f"{side_name} was not produced")
                continue
            try:
                json.loads(side_raw)
            except json.JSONDecodeError as exc:
                errors.append(f"{side_name} contained invalid JSON: {exc}")

        if not errors and parsed is not None:
            errors.extend(validate_parsed(parsed))

        if not errors:
            if on_success_outputs is not None:
                try:
                    on_success_outputs(outputs)
                except ValueError as exc:
                    errors.append(str(exc))
            if not errors:
                if trace_writer is not None:
                    trace_writer.record(
                        "tier_generated",
                        site_name=site_name,
                        label=attempt_label,
                        attempt=attempt,
                        output_path=output_path,
                        **(trace_context or {}),
                    )
                return parsed

        last_errors = errors
        if trace_writer is not None:
            trace_writer.record(
                "tier_validation_failed",
                site_name=site_name,
                label=attempt_label,
                attempt=attempt,
                output_path=output_path,
                errors=errors,
                **(trace_context or {}),
            )
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

    if trace_writer is not None:
        trace_writer.record(
            "tier_failed",
            site_name=site_name,
            label=label,
            output_path=output_path,
            errors=last_errors,
            **(trace_context or {}),
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


def _enrich_agent_context_with_handles(
    *,
    site_name: str,
    agent_context: dict[str, Any],
    instance: BenchmarkInstance | None,
) -> dict[str, Any]:
    """For gitlab sites, enumerate user/group handles via the live API.

    Returns ``agent_context`` unchanged for non-gitlab sites or when the
    instance is unavailable / unauthenticated. Enrichment failures are
    logged at warning level — Phase 0c does not abort on a transient
    handle-enumeration outage. Phase 2's resolver categorizes downstream
    drops cleanly when the lists are absent.
    """
    if normalize_site_name(site_name) != "gitlab":
        return agent_context
    if instance is None:
        logger.info(
            "Phase 0c: site %r has no instance config; skipping handle enrichment", site_name
        )
        return agent_context
    auth_config = instance.api_auth or instance.auth
    if not auth_config:
        logger.info(
            "Phase 0c: site %r instance has no api_auth/auth; skipping handle enrichment",
            site_name,
        )
        return agent_context

    from worldsim.phases.phase_0c_handle_enrichment import (
        HandleEnrichmentError,
        enrich_gitlab_handles,
        merge_into_agent_context,
    )

    try:
        handles = enrich_gitlab_handles(
            instance.site_url,
            auth_config,
            runtime_web_host=_host_side_runtime_host(),
        )
    except HandleEnrichmentError as exc:
        logger.warning(
            "Phase 0c: gitlab handle enrichment for site %r failed: %s",
            site_name,
            exc,
        )
        return agent_context

    logger.info(
        "Phase 0c: site %r enriched with %d user_handles and %d group_handles",
        site_name,
        len(handles.get("user_handles", [])),
        len(handles.get("group_handles", [])),
    )
    return merge_into_agent_context(agent_context, handles)


def _enrich_gitlab_profile_with_projects(
    *,
    site_name: str,
    profile: dict[str, Any],
    instance: BenchmarkInstance | None,
) -> dict[str, Any]:
    """For gitlab sites, attach namespace-qualified project inventory."""
    if normalize_site_name(site_name) != "gitlab":
        return profile
    if instance is None:
        logger.info(
            "Phase 0c: site %r has no instance config; skipping gitlab project enrichment",
            site_name,
        )
        return profile
    auth_config = instance.api_auth or instance.auth
    if not auth_config:
        logger.info(
            "Phase 0c: site %r instance has no api_auth/auth; skipping gitlab project enrichment",
            site_name,
        )
        return profile

    from worldsim.phases.phase_0c_handle_enrichment import (
        HandleEnrichmentError,
        enrich_gitlab_projects,
        merge_gitlab_project_inventory_into_profile,
    )

    try:
        inventory = enrich_gitlab_projects(
            instance.site_url,
            auth_config,
            runtime_web_host=_host_side_runtime_host(),
        )
    except HandleEnrichmentError as exc:
        logger.warning(
            "Phase 0c: gitlab project enrichment for site %r failed: %s",
            site_name,
            exc,
        )
        return profile

    projects = inventory.get("projects", [])
    if not projects:
        logger.warning(
            "Phase 0c: gitlab project enrichment for site %r found no projects",
            site_name,
        )
        return profile
    logger.info(
        "Phase 0c: site %r enriched with %d gitlab projects",
        site_name,
        len(projects),
    )
    return merge_gitlab_project_inventory_into_profile(profile, inventory)


def _enrich_reddit_profile_with_forums(
    *,
    site_name: str,
    profile: dict[str, Any],
    instance: BenchmarkInstance | None,
    instances: list[BenchmarkInstance] | None = None,
) -> dict[str, Any]:
    """For reddit sites, attach live-reachable forum inventory to the profile."""
    if normalize_site_name(site_name) != "reddit":
        return profile
    inventory_instances = list(instances or ([instance] if instance is not None else []))
    inventory_instances = [
        item for item in inventory_instances if item is not None and item.db_connection
    ]
    if not inventory_instances:
        logger.info(
            "Phase 0c: site %r has no instance config; skipping reddit forum enrichment",
            site_name,
        )
        return profile

    from worldsim.phases.phase_0c_reddit_enrichment import (
        RedditInventoryEnrichmentError,
        common_reddit_forum_inventory,
        enrich_reddit_forums,
        merge_reddit_inventory_into_profile,
    )

    try:
        inventories = [
            enrich_reddit_forums(
                item.site_url,
                item.db_connection,
                runtime_db_host=_host_side_runtime_host(),
            )
            for item in inventory_instances
        ]
    except RedditInventoryEnrichmentError as exc:
        logger.warning(
            "Phase 0c: reddit forum enrichment for site %r failed: %s",
            site_name,
            exc,
        )
        return profile

    inventory = common_reddit_forum_inventory(inventories)
    forums = inventory.get("forums", [])
    if not forums:
        logger.warning(
            "Phase 0c: reddit forum enrichment for site %r found no forums common to %d replica(s)",
            site_name,
            len(inventory_instances),
        )
        return profile
    logger.info(
        "Phase 0c: site %r enriched with %d reachable reddit forums common to %d replica(s)",
        site_name,
        len(forums),
        len(inventory_instances),
    )
    return merge_reddit_inventory_into_profile(profile, inventory)


def _host_side_runtime_host() -> str | None:
    """Return an optional host-local hostname for Phase 0c enrichment.

    Modal receives public/proxied web URLs for Phase 0c live browsing, but the
    enrichment hooks run in the orchestrator process. Registered r5 jobs
    export ``WORLDSIM_ORCHESTRATOR_HOST`` so host-side DB/API queries can use
    the same local network view as Phase 2c/4 instead of trying to hairpin
    through the public EC2 address.
    """

    for name in ("WORLDSIM_ORCHESTRATOR_HOST", "WORLDSIM_REMOTE_ORCHESTRATOR_HOST"):
        value = os.environ.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


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
