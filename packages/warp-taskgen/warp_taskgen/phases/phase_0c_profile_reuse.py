"""Phase 0c: whether an existing site profile may be reused instead of re-run.

Owns the two reuse decisions, the tier metadata they compare against, the Tier 1
input hashes and prompt rendering that feed those hashes, and the sidecar name
tables the profiling loop publishes.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from warp_taskgen._sandbox_validator import (
    validate_agent_context,
    validate_data_model_profile,
    validate_injection_surface,
    validate_verification_capabilities,
)
from warp_taskgen.config import BenchmarkInstance, VerificationProxy
from warp_taskgen.phases.phase_0a_benchmark_manifest import _validate_manifest_eval_types
from warp_taskgen.phases.phase_0c_artifacts import (
    build_tier_metadata,
    hash_json,
    load_reusable_tier_output,
    profile_metadata_path,
    text_sha256,
)
from warp_taskgen.phases.phase_0c_instance_reachability import (
    _phase_0c_redact_values,
    _verification_proxy_metadata,
)
from warp_taskgen.profile_validation import validate_profile
from warp_taskgen.prompt_loading import load_prompt

logger = logging.getLogger(__name__)


_DATA_MODEL_SIDECARS = ("DATA_MODEL_EVIDENCE",)


_INJECTION_SURFACE_SIDECARS = (
    "SURFACE_DRAFT",
    "TASK_COVERAGE_DRAFT",
    "LIVE_VERIFICATION_NOTES",
)


def _profile_metadata_path(output_dir: Path, site_name: str) -> Path:
    return profile_metadata_path(output_dir, site_name)


def _instance_inventory_fingerprint(instance: BenchmarkInstance | None) -> str | None:
    """Return a non-secret cache fingerprint for host-side inventory inputs."""
    if instance is None:
        return None
    payload = instance.model_dump(mode="json", exclude_none=True)
    # Store only a digest in profile metadata because auth/db fields can carry
    # credentials. The digest still invalidates stale profiles when the host
    # inventory topology changes.
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


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


def _render_tier_prompt(*, prompt_name: str, validation_command: str, site_name: str) -> str:
    """Load a profiling prompt and substitute the site name placeholder."""
    return load_prompt(
        prompt_name,
        validation_command=validation_command,
    ).replace("{site_name}", site_name)


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
