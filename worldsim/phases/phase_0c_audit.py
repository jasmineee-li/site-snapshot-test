"""Deterministic audit helpers for Phase 0c profile artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from worldsim._sandbox_validator import (
    validate_agent_context,
    validate_data_model_profile,
    validate_injection_surface,
    validate_verification_capabilities,
)
from worldsim.phases.phase_0c_artifacts import text_sha256, tier_metadata_path
from worldsim.profile_validation import validate_profile

_TIER_ARTIFACTS = {
    "A_VERIFICATION_CAPABILITIES": "VERIFICATION_CAPABILITIES",
    "B_DATA_MODEL": "DATA_MODEL",
    "C_AGENT_CONTEXT": "AGENT_CONTEXT_RAW",
    "DE_INJECTION_SURFACE": "INJECTION_SURFACE",
}
_OPTIONAL_SIDECARS = (
    "DATA_MODEL_EVIDENCE",
    "SURFACE_DRAFT",
    "TASK_COVERAGE_DRAFT",
    "LIVE_VERIFICATION_NOTES",
)


def audit_phase_0c_profiles(
    profiles_dir: Path,
    *,
    benchmark_root: Path | None = None,
    manifest_eval_types: Iterable[str] = (),
) -> dict[str, Any]:
    """Audit Phase 0c profiles without making semantic research decisions."""
    profiles_dir = Path(profiles_dir)
    report: dict[str, Any] = {
        "schema_version": 1,
        "profiles_dir": str(profiles_dir),
        "benchmark_root": str(benchmark_root) if benchmark_root is not None else None,
        "sites": [],
        "errors": [],
        "warnings": [],
    }
    profile_paths = sorted(profiles_dir.glob("BENCHMARK_PROFILE_*.json"))
    if not profile_paths:
        report["errors"].append(
            {
                "site": None,
                "code": "NO_PROFILES",
                "message": f"no BENCHMARK_PROFILE_*.json files found in {profiles_dir}",
            }
        )
        return _finalize(report)

    for profile_path in profile_paths:
        site_name = profile_path.stem.removeprefix("BENCHMARK_PROFILE_")
        site_report = _audit_site(
            profiles_dir=profiles_dir,
            site_name=site_name,
            profile_path=profile_path,
            benchmark_root=benchmark_root,
            manifest_eval_types=manifest_eval_types,
        )
        report["sites"].append(site_report)
        report["errors"].extend(site_report["errors"])
        report["warnings"].extend(site_report["warnings"])
    return _finalize(report)


def _audit_site(
    *,
    profiles_dir: Path,
    site_name: str,
    profile_path: Path,
    benchmark_root: Path | None,
    manifest_eval_types: Iterable[str],
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    profile = _load_json(profile_path, site_name=site_name, errors=errors)
    if not isinstance(profile, dict):
        return {"site": site_name, "errors": errors, "warnings": warnings}

    _audit_profile_shape(
        profile=profile,
        profiles_dir=profiles_dir,
        site_name=site_name,
        manifest_eval_types=manifest_eval_types,
        errors=errors,
    )
    _audit_tier_artifacts(
        profile=profile,
        profiles_dir=profiles_dir,
        site_name=site_name,
        errors=errors,
        warnings=warnings,
    )
    _audit_optional_sidecars(
        profiles_dir=profiles_dir,
        site_name=site_name,
        benchmark_root=benchmark_root,
        warnings=warnings,
    )
    return {"site": site_name, "errors": errors, "warnings": warnings}


def _audit_profile_shape(
    *,
    profile: Mapping[str, Any],
    profiles_dir: Path,
    site_name: str,
    manifest_eval_types: Iterable[str],
    errors: list[dict[str, Any]],
) -> None:
    context_path = profiles_dir / f"AGENT_CONTEXT_{site_name}.json"
    context = _load_json(context_path, site_name=site_name, errors=errors)
    if isinstance(context, dict) and profile.get("agent_context") != context:
        errors.append(
            {
                "site": site_name,
                "code": "PROFILE_CONTEXT_MISMATCH",
                "path": str(context_path),
                "message": "BENCHMARK_PROFILE.agent_context does not match AGENT_CONTEXT sidecar",
            }
        )

    _extend_validation_errors(
        errors,
        site_name=site_name,
        artifact="verification_capabilities",
        messages=validate_verification_capabilities(
            profile.get("verification_capabilities"), site_name=site_name
        ),
    )
    _extend_validation_errors(
        errors,
        site_name=site_name,
        artifact="data_model",
        messages=validate_data_model_profile(profile.get("data_model"), site_name=site_name),
    )
    if isinstance(context, dict):
        _extend_validation_errors(
            errors,
            site_name=site_name,
            artifact="agent_context",
            messages=validate_agent_context(context, site_name=site_name),
        )
    injection_payload = {
        "injection_surface": profile.get("injection_surface"),
        "existing_task_coverage": profile.get("existing_task_coverage"),
    }
    _extend_validation_errors(
        errors,
        site_name=site_name,
        artifact="injection_surface",
        messages=validate_injection_surface(
            injection_payload,
            site_name=site_name,
            data_model=profile.get("data_model"),
            agent_context=profile.get("agent_context"),
        ),
    )
    try:
        validate_profile(site_name, dict(profile), manifest_eval_types=manifest_eval_types)
    except ValueError as exc:
        errors.append(
            {
                "site": site_name,
                "code": "PROFILE_VALIDATION",
                "message": str(exc),
            }
        )


def _audit_tier_artifacts(
    *,
    profile: Mapping[str, Any],
    profiles_dir: Path,
    site_name: str,
    errors: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> None:
    for tier_name, artifact_stem in _TIER_ARTIFACTS.items():
        artifact_path = profiles_dir / f"{artifact_stem}_{site_name}.json"
        metadata_path = tier_metadata_path(profiles_dir, site_name, tier_name)
        if not artifact_path.exists():
            warnings.append(
                {
                    "site": site_name,
                    "code": "MISSING_TIER_ARTIFACT",
                    "path": str(artifact_path),
                    "message": "tier artifact is absent; rerun Phase 0c to refresh provenance",
                }
            )
            continue
        artifact = _load_json(artifact_path, site_name=site_name, errors=errors)
        if artifact_stem == "AGENT_CONTEXT_RAW":
            _extend_validation_errors(
                errors,
                site_name=site_name,
                artifact="agent_context_raw",
                messages=validate_agent_context(artifact, site_name=site_name),
            )
        _check_artifact_matches_profile(
            profile=profile,
            artifact=artifact,
            artifact_stem=artifact_stem,
            site_name=site_name,
            errors=errors,
        )
        if not metadata_path.exists():
            warnings.append(
                {
                    "site": site_name,
                    "code": "MISSING_TIER_METADATA",
                    "path": str(metadata_path),
                    "message": "tier metadata is absent; artifact cannot be reused rigorously",
                }
            )
            continue
        metadata = _load_json(metadata_path, site_name=site_name, errors=errors)
        if not isinstance(metadata, dict):
            continue
        if metadata.get("artifact_path") != artifact_path.name:
            errors.append(
                {
                    "site": site_name,
                    "code": "TIER_METADATA_ARTIFACT_PATH",
                    "path": str(metadata_path),
                    "message": "tier metadata artifact_path does not name the artifact file",
                }
            )
        expected_hash = text_sha256(artifact_path.read_text(encoding="utf-8"))
        if metadata.get("artifact_sha256") != expected_hash:
            errors.append(
                {
                    "site": site_name,
                    "code": "TIER_ARTIFACT_HASH_MISMATCH",
                    "path": str(metadata_path),
                    "message": "tier metadata artifact_sha256 does not match artifact contents",
                }
            )


def _audit_optional_sidecars(
    *,
    profiles_dir: Path,
    site_name: str,
    benchmark_root: Path | None,
    warnings: list[dict[str, Any]],
) -> None:
    del benchmark_root
    for artifact_stem in _OPTIONAL_SIDECARS:
        path = profiles_dir / f"{artifact_stem}_{site_name}.json"
        if path.exists():
            continue
        warnings.append(
            {
                "site": site_name,
                "code": "MISSING_OPTIONAL_SIDECAR",
                "path": str(path),
                "message": (
                    f"{artifact_stem} sidecar is absent; profile remains valid but has weaker "
                    "review provenance"
                ),
            }
        )


def _check_artifact_matches_profile(
    *,
    profile: Mapping[str, Any],
    artifact: object,
    artifact_stem: str,
    site_name: str,
    errors: list[dict[str, Any]],
) -> None:
    expected: object
    if artifact_stem == "VERIFICATION_CAPABILITIES":
        expected = profile.get("verification_capabilities")
    elif artifact_stem == "DATA_MODEL":
        expected = profile.get("data_model")
    elif artifact_stem == "INJECTION_SURFACE":
        expected = {
            "injection_surface": profile.get("injection_surface"),
            "existing_task_coverage": profile.get("existing_task_coverage"),
        }
    elif artifact_stem == "AGENT_CONTEXT_RAW":
        return
    else:
        return
    if artifact != expected:
        errors.append(
            {
                "site": site_name,
                "code": "TIER_ARTIFACT_PROFILE_MISMATCH",
                "artifact": artifact_stem,
                "message": f"{artifact_stem} artifact does not match BENCHMARK_PROFILE",
            }
        )


def _load_json(
    path: Path,
    *,
    site_name: str,
    errors: list[dict[str, Any]],
) -> object | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        errors.append(
            {
                "site": site_name,
                "code": "MISSING_JSON",
                "path": str(path),
                "message": f"missing required JSON artifact: {path.name}",
            }
        )
    except json.JSONDecodeError as exc:
        errors.append(
            {
                "site": site_name,
                "code": "INVALID_JSON",
                "path": str(path),
                "message": str(exc),
            }
        )
    except OSError as exc:
        errors.append(
            {
                "site": site_name,
                "code": "UNREADABLE_JSON",
                "path": str(path),
                "message": str(exc),
            }
        )
    return None


def _extend_validation_errors(
    errors: list[dict[str, Any]],
    *,
    site_name: str,
    artifact: str,
    messages: Iterable[str],
) -> None:
    for message in messages:
        errors.append(
            {
                "site": site_name,
                "code": "SCHEMA_VALIDATION",
                "artifact": artifact,
                "message": message,
            }
        )


def _finalize(report: dict[str, Any]) -> dict[str, Any]:
    report["summary"] = {
        "sites": len(report.get("sites") or []),
        "errors": len(report.get("errors") or []),
        "warnings": len(report.get("warnings") or []),
    }
    return report
