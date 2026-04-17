"""Preflight checks for instance-level storage_state auth artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from worldsim.config import BenchmarkConfig, BenchmarkInstance


@dataclass(frozen=True)
class HostBoundStorageStateMismatch:
    """A storage_state artifact whose recorded hosts do not match live instances."""

    site_name: str
    declared_path: str
    artifact_path: Path
    recorded_hosts: tuple[str, ...]
    instance_hosts: tuple[str, ...]


@dataclass(frozen=True)
class StorageStatePreflightError:
    """A storage_state artifact that cannot be safely resolved or loaded."""

    site_name: str
    declared_path: str
    message: str


@dataclass(frozen=True)
class StorageStatePreflightReport:
    """Collected storage_state preflight issues for a set of instances."""

    mismatches: tuple[HostBoundStorageStateMismatch, ...]
    errors: tuple[StorageStatePreflightError, ...]


def _normalized_host(value: str) -> str:
    host = value.strip().lower().strip(".")
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    return host


def _cookie_domain_matches_host(domain: str, host: str) -> bool:
    normalized_domain = _normalized_host(domain)
    normalized_host = _normalized_host(host)
    if not normalized_domain or not normalized_host:
        return False
    if normalized_domain == normalized_host:
        return True
    if ":" in normalized_domain or ":" in normalized_host:
        return False
    return normalized_host.endswith(f".{normalized_domain}")


def _recorded_hosts(storage_state: dict[str, Any]) -> tuple[str, ...]:
    hosts: set[str] = set()

    cookies = storage_state.get("cookies")
    if isinstance(cookies, list):
        for cookie in cookies:
            if not isinstance(cookie, dict):
                continue
            domain = cookie.get("domain")
            if isinstance(domain, str) and domain.strip():
                hosts.add(_normalized_host(domain))

    origins = storage_state.get("origins")
    if isinstance(origins, list):
        for origin in origins:
            if not isinstance(origin, dict):
                continue
            origin_url = origin.get("origin")
            if not isinstance(origin_url, str) or not origin_url.strip():
                continue
            host = urlsplit(origin_url).hostname
            if host:
                hosts.add(_normalized_host(host))

    return tuple(sorted(host for host in hosts if host))


def _instance_host(instance: BenchmarkInstance) -> str | None:
    host = urlsplit(instance.site_url).hostname
    if host:
        return _normalized_host(host)
    return None


def _candidate_storage_state_paths(
    *,
    site_name: str,
    raw_path: str,
    benchmark_root: Path | None,
) -> list[Path]:
    declared = Path(raw_path)
    candidates: list[Path] = []
    if declared.is_absolute():
        candidates.append(declared.resolve())
    else:
        if benchmark_root is None:
            return []
        resolved_root = Path(benchmark_root).resolve()
        resolved = (resolved_root / declared).resolve()
        try:
            resolved.relative_to(resolved_root)
        except ValueError:
            return []
        candidates.append(resolved)

    try:
        from worldsim.phases.phase_0d_auth_bootstrap import phase_0d_artifact_path
    except ImportError:  # pragma: no cover
        phase0d = None
    else:
        phase0d = phase_0d_artifact_path(site_name)
    if phase0d is not None:
        candidates.append(phase0d)

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _declared_storage_state_path(instance: BenchmarkInstance) -> str | None:
    auth = instance.agent_auth
    if not isinstance(auth, dict) or str(auth.get("type", "")).strip() != "storage_state":
        return None
    storage_state = auth.get("storage_state")
    if not isinstance(storage_state, dict):
        return None
    raw_path = storage_state.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    return raw_path.strip()


def _resolve_storage_state_artifact_for_preflight(
    instance: BenchmarkInstance,
    *,
    benchmark_root: Path | None,
) -> tuple[Path | None, StorageStatePreflightError | None]:
    raw_path = _declared_storage_state_path(instance)
    if raw_path is None:
        return None, None

    declared = Path(raw_path)
    if declared.is_absolute():
        declared_path = declared.resolve()
    else:
        if benchmark_root is None:
            return None, StorageStatePreflightError(
                site_name=instance.site_name,
                declared_path=raw_path,
                message=(
                    "relative auth_mechanism.storage_state.path requires --benchmark so "
                    "the runtime can resolve the artifact safely"
                ),
            )
        resolved_root = Path(benchmark_root).resolve()
        declared_path = (resolved_root / declared).resolve()
        try:
            declared_path.relative_to(resolved_root)
        except ValueError:
            return None, StorageStatePreflightError(
                site_name=instance.site_name,
                declared_path=raw_path,
                message=(
                    f"storage_state path {raw_path!r} resolves outside benchmark root "
                    f"{resolved_root}"
                ),
            )

    if declared_path.exists():
        return declared_path, None

    try:
        from worldsim.phases.phase_0d_auth_bootstrap import phase_0d_artifact_path
    except ImportError:  # pragma: no cover
        bootstrap_path = None
    else:
        bootstrap_path = phase_0d_artifact_path(instance.site_name)

    if bootstrap_path is not None and bootstrap_path.exists():
        return bootstrap_path.resolve(), None

    return None, StorageStatePreflightError(
        site_name=instance.site_name,
        declared_path=raw_path,
        message=f"storage_state artifact missing at {declared_path}",
    )


def resolve_storage_state_artifact(
    instance: BenchmarkInstance,
    *,
    benchmark_root: Path | None,
) -> Path | None:
    """Return the on-disk storage_state artifact for an instance, when present."""
    artifact_path, _ = _resolve_storage_state_artifact_for_preflight(
        instance,
        benchmark_root=benchmark_root,
    )
    return artifact_path


def inspect_storage_state_preflight(
    instances: list[BenchmarkInstance],
    *,
    benchmark_root: Path | None = None,
) -> StorageStatePreflightReport:
    """Return storage_state mismatches plus resolution/load errors."""
    mismatches: list[HostBoundStorageStateMismatch] = []
    errors: list[StorageStatePreflightError] = []
    seen_mismatches: set[tuple[str, str, str]] = set()
    seen_errors: set[tuple[str, str, str]] = set()

    for instance in instances:
        raw_path = _declared_storage_state_path(instance)
        artifact_path, error = _resolve_storage_state_artifact_for_preflight(
            instance,
            benchmark_root=benchmark_root,
        )
        if error is not None:
            dedupe_key = (error.site_name, error.declared_path, error.message)
            if dedupe_key not in seen_errors:
                seen_errors.add(dedupe_key)
                errors.append(error)
            continue
        if artifact_path is None or raw_path is None:
            continue

        try:
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        except OSError as exc:
            error = StorageStatePreflightError(
                site_name=instance.site_name,
                declared_path=raw_path,
                message=f"unable to read storage_state artifact {artifact_path}: {exc}",
            )
            error_key = (error.site_name, error.declared_path, error.message)
            if error_key not in seen_errors:
                seen_errors.add(error_key)
                errors.append(error)
            continue
        except json.JSONDecodeError as exc:
            error = StorageStatePreflightError(
                site_name=instance.site_name,
                declared_path=raw_path,
                message=f"invalid JSON in storage_state artifact {artifact_path}: {exc}",
            )
            error_key = (error.site_name, error.declared_path, error.message)
            if error_key not in seen_errors:
                seen_errors.add(error_key)
                errors.append(error)
            continue
        if not isinstance(payload, dict):
            error = StorageStatePreflightError(
                site_name=instance.site_name,
                declared_path=raw_path,
                message=f"storage_state artifact {artifact_path} does not contain a JSON object",
            )
            error_key = (error.site_name, error.declared_path, error.message)
            if error_key not in seen_errors:
                seen_errors.add(error_key)
                errors.append(error)
            continue

        recorded_hosts = _recorded_hosts(payload)
        if not recorded_hosts:
            continue
        instance_host = _instance_host(instance)
        if not instance_host:
            continue
        dedupe_key = (instance.site_name, raw_path, str(artifact_path), instance_host)
        if dedupe_key in seen_mismatches:
            continue
        seen_mismatches.add(dedupe_key)

        if any(
            _cookie_domain_matches_host(recorded_host, instance_host)
            for recorded_host in recorded_hosts
        ):
            continue

        mismatches.append(
            HostBoundStorageStateMismatch(
                site_name=instance.site_name,
                declared_path=raw_path,
                artifact_path=artifact_path,
                recorded_hosts=recorded_hosts,
                instance_hosts=(instance_host,),
            )
        )

    return StorageStatePreflightReport(
        mismatches=tuple(mismatches),
        errors=tuple(errors),
    )


def find_host_bound_storage_state_mismatches(
    instances: list[BenchmarkInstance],
    *,
    benchmark_root: Path | None = None,
) -> list[HostBoundStorageStateMismatch]:
    """Return storage_state artifacts whose recorded hosts miss the live host."""
    return list(
        inspect_storage_state_preflight(
            instances,
            benchmark_root=benchmark_root,
        ).mismatches
    )


def apply_skip_auth_for_host_bound_storage_states(
    config: BenchmarkConfig,
    mismatches: list[HostBoundStorageStateMismatch],
) -> BenchmarkConfig:
    """Replace mismatched storage_state auth with explicit no-auth for runtime."""
    mismatch_paths = {
        (mismatch.site_name, mismatch.declared_path.strip())
        for mismatch in mismatches
        if mismatch.declared_path.strip()
    }
    if not mismatch_paths:
        return config

    payload = config.model_dump(mode="json")
    instances = payload.get("instances", [])
    for instance in instances:
        site_name = str(instance.get("site_name", "")).strip()
        agent_auth = instance.get("agent_auth")
        if not isinstance(agent_auth, dict) or str(agent_auth.get("type", "")).strip() != "storage_state":
            continue
        storage_state = agent_auth.get("storage_state")
        if not isinstance(storage_state, dict):
            continue
        declared_path = str(storage_state.get("path", "")).strip()
        if (site_name, declared_path) not in mismatch_paths:
            continue
        instance["agent_auth"] = {
            "type": "none",
            "notes": "Skipped due to host-bound storage_state artifact; re-run phase 0d for this host.",
        }
    return BenchmarkConfig.model_validate(payload)
