"""Preflight checks for instance-level storage_state auth artifacts."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from warp_taskgen.agent_auth import (
    _declared_path_references_phase_0d,
    _phase_0d_site_roots,
    _resolve_declared_storage_state_path,
)
from warp_taskgen.atomic_io import write_json_atomic
from warp_taskgen.benchmark_capabilities import normalize_benchmark_name
from warp_taskgen.config import BenchmarkConfig, BenchmarkInstance

logger = logging.getLogger(__name__)

# Playwright ``storage_state`` cookies are short-lived for many sites
# (GitLab's ``_gitlab_session`` especially). A 24h TTL is a conservative
# default: shorter re-mints too often and wastes Playwright launches;
# longer risks running a rigor run with an expired session that fails
# cryptically mid-task.
_STORAGE_STATE_TTL_SECONDS = 24 * 60 * 60

# Runtime auto-heal is allowed by default for the WebArena Verified
# benchmark (dummy creds already in repo). Other benchmarks must opt in
# via this env var.
_AUTO_MINT_ENV = "WORLDSIM_AUTO_MINT_STORAGE_STATE"


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


def _mixed_host_binding_error(
    *,
    artifact_path: Path,
    payload: dict[str, Any],
    instance_host: str,
) -> str | None:
    recorded_hosts = _recorded_hosts(payload)
    matching_hosts = [
        host for host in recorded_hosts if _cookie_domain_matches_host(host, instance_host)
    ]
    foreign_hosts = [
        host for host in recorded_hosts if not _cookie_domain_matches_host(host, instance_host)
    ]
    if not matching_hosts or not foreign_hosts:
        return None
    return (
        f"storage_state_mixed_hosts: artifact {artifact_path} mixes live host "
        f"{instance_host!r} records {sorted(matching_hosts)} with foreign recorded "
        f"hosts {sorted(foreign_hosts)}; re-run Phase 0d against the current "
        "generated instances file"
    )


def _artifact_is_structurally_usable(payload: dict[str, Any]) -> tuple[bool, str | None]:
    cookies = payload.get("cookies")
    origins = payload.get("origins")
    has_cookies = isinstance(cookies, list) and any(isinstance(cookie, dict) for cookie in cookies)
    has_origins = isinstance(origins, list) and any(isinstance(origin, dict) for origin in origins)
    if not has_cookies and not has_origins:
        return False, "storage_state_empty: artifact has no cookies or origins"
    recorded_hosts = _recorded_hosts(payload)
    if not recorded_hosts:
        return False, "storage_state_empty: artifact has no recorded hosts in cookies or origins"
    return True, None


def _candidate_storage_state_paths(
    *,
    site_name: str,
    raw_path: str,
    benchmark_root: Path | None,
) -> list[Path]:
    declared = Path(raw_path)
    candidates: list[Path] = []
    if declared.is_absolute():
        candidates.append(declared)
    else:
        if benchmark_root is None:
            return []
        root = Path(benchmark_root)
        joined = root / declared
        # Containment uses resolved paths (catches `../` escapes), but the
        # candidate returned is the unresolved logical path so symlinks in
        # benchmark_root don't divert existence checks away from where
        # Phase 0d actually writes.
        try:
            joined.resolve().relative_to(root.resolve())
        except ValueError:
            return []
        candidates.append(joined)

    try:
        from warp_taskgen.phases.phase_0d_auth_bootstrap import phase_0d_artifact_path
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

    references_phase_0d = _declared_path_references_phase_0d(
        raw_path,
        site_name=instance.site_name,
    )
    phase_0d_roots, phase_0d_roots_error = _phase_0d_site_roots(
        instance.site_name,
        include_canonical=references_phase_0d,
    )
    if phase_0d_roots_error is not None:
        return None, StorageStatePreflightError(
            site_name=instance.site_name,
            declared_path=raw_path,
            message=phase_0d_roots_error,
        )

    try:
        from warp_taskgen.phases.phase_0d_auth_bootstrap import phase_0d_instance_id
    except ImportError:  # pragma: no cover
        instance_id = None
    else:
        instance_id = phase_0d_instance_id(instance.model_dump())
        if references_phase_0d:
            for root in phase_0d_roots:
                per_instance_path = root / "instances" / instance_id / "storage_state.json"
                if per_instance_path.exists():
                    return per_instance_path, None

    declared_path, resolve_error = _resolve_declared_storage_state_path(
        raw_path,
        benchmark_root=benchmark_root,
        site_name=instance.site_name,
    )
    if resolve_error is not None or declared_path is None:
        return None, StorageStatePreflightError(
            site_name=instance.site_name,
            declared_path=raw_path,
            message=resolve_error or "storage_state path could not be resolved",
        )

    if declared_path.exists():
        return declared_path, None

    if instance_id is not None:
        # If runtime auto-heal minted a per-instance artifact after a declared
        # non-Phase-0d path went missing, consume the active state-dir artifact.
        # Canonical logs fallback remains limited to declared Phase 0d paths.
        for root in phase_0d_roots:
            per_instance_path = root / "instances" / instance_id / "storage_state.json"
            if per_instance_path.exists():
                return per_instance_path, None

    if references_phase_0d:
        for root in phase_0d_roots:
            bootstrap_path = root / "storage_state.json"
            completion_path = root / "completion.json"
            if bootstrap_path.exists() and completion_path.exists():
                return bootstrap_path, None

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

        usable, unusable_reason = _artifact_is_structurally_usable(payload)
        if not usable:
            error = StorageStatePreflightError(
                site_name=instance.site_name,
                declared_path=raw_path,
                message=f"{unusable_reason} for storage_state artifact {artifact_path}",
            )
            error_key = (error.site_name, error.declared_path, error.message)
            if error_key not in seen_errors:
                seen_errors.add(error_key)
                errors.append(error)
            continue
        recorded_hosts = _recorded_hosts(payload)
        instance_host = _instance_host(instance)
        if not instance_host:
            error = StorageStatePreflightError(
                site_name=instance.site_name,
                declared_path=raw_path,
                message=f"storage_state_empty: instance site_url {instance.site_url!r} has no resolvable host",
            )
            error_key = (error.site_name, error.declared_path, error.message)
            if error_key not in seen_errors:
                seen_errors.add(error_key)
                errors.append(error)
            continue
        dedupe_key = (instance.site_name, raw_path, str(artifact_path), instance_host)
        if dedupe_key in seen_mismatches:
            continue
        seen_mismatches.add(dedupe_key)

        mixed_error = _mixed_host_binding_error(
            artifact_path=artifact_path,
            payload=payload,
            instance_host=instance_host,
        )
        if mixed_error is not None:
            error = StorageStatePreflightError(
                site_name=instance.site_name,
                declared_path=raw_path,
                message=mixed_error,
            )
            error_key = (error.site_name, error.declared_path, error.message)
            if error_key not in seen_errors:
                seen_errors.add(error_key)
                errors.append(error)
            continue

        if all(
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
        if (
            not isinstance(agent_auth, dict)
            or str(agent_auth.get("type", "")).strip() != "storage_state"
        ):
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


# ---------------------------------------------------------------------------
# Auto-mint helpers (runtime auto-heal for missing/stale storage_state)
# ---------------------------------------------------------------------------


def _meta_sidecar_path(artifact_path: Path) -> Path:
    """Return the ``.meta.json`` sidecar companion for a storage_state file."""
    return artifact_path.with_name(artifact_path.name.replace(".json", ".meta.json"))


def storage_state_is_fresh(
    artifact_path: Path,
    *,
    ttl_seconds: int = _STORAGE_STATE_TTL_SECONDS,
    now_fn: Any | None = None,
) -> bool:
    """Return True when the sidecar says the artifact was minted within TTL.

    Missing sidecar → assume fresh (preserves behavior for pre-existing
    artifacts that were minted before this helper existed). Malformed
    sidecar → treat as stale so we re-mint rather than trusting garbage.
    """
    sidecar = _meta_sidecar_path(artifact_path)
    if not sidecar.exists():
        return True
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    try:
        from warp_taskgen.phases.phase_0d_auth_bootstrap import CURRENT_VALIDATOR_VERSION
    except ImportError:  # pragma: no cover
        CURRENT_VALIDATOR_VERSION = None
    if CURRENT_VALIDATOR_VERSION is not None:
        try:
            validator_version = int(payload.get("validator_version") or 0)
        except (TypeError, ValueError):
            return False
        if validator_version != CURRENT_VALIDATOR_VERSION:
            return False
    raw = payload.get("minted_at")
    if not isinstance(raw, str):
        return False
    try:
        minted_at = datetime.fromisoformat(raw)
    except ValueError:
        return False
    if minted_at.tzinfo is None:
        minted_at = minted_at.replace(tzinfo=UTC)
    now = now_fn() if now_fn is not None else datetime.now(UTC)
    age = (now - minted_at).total_seconds()
    return age < ttl_seconds


def write_storage_state_meta(
    artifact_path: Path,
    *,
    mechanism: str,
    now_fn: Any | None = None,
    last_validated_at: datetime | None = None,
    validator_version: int | None = None,
) -> None:
    """Write a ``.meta.json`` sidecar recording when + how the state was minted.

    ``last_validated_at`` and ``validator_version`` are optional Phase 0d
    liveness-cache fields. When omitted on initial mint they default to the
    mint timestamp and the current Phase 0d validator version respectively;
    callers re-validating an existing artifact pass them explicitly.
    """
    now = now_fn() if now_fn is not None else datetime.now(UTC)
    if validator_version is None:
        from warp_taskgen.phases.phase_0d_auth_bootstrap import CURRENT_VALIDATOR_VERSION

        validator_version = CURRENT_VALIDATOR_VERSION
    if last_validated_at is None:
        last_validated_at = now
    payload = {
        "minted_at": now.isoformat(),
        "mechanism": mechanism,
        "last_validated_at": last_validated_at.isoformat(),
        "validator_version": int(validator_version),
    }
    sidecar = _meta_sidecar_path(artifact_path)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_storage_state_meta(artifact_path: Path) -> dict[str, Any] | None:
    """Return the parsed ``.meta.json`` sidecar payload, or ``None`` when absent/malformed."""
    sidecar = _meta_sidecar_path(artifact_path)
    if not sidecar.exists():
        return None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def update_storage_state_meta_validation(
    artifact_path: Path,
    *,
    last_validated_at: datetime,
    validator_version: int,
) -> None:
    """Stamp a fresh ``last_validated_at`` + ``validator_version`` onto the sidecar.

    Preserves all existing fields (including ``minted_at`` and ``mechanism``)
    so older readers continue to function. If the sidecar does not exist,
    this is a no-op: a missing sidecar means the artifact was minted before
    the liveness cache existed and should be treated as fresh by readers.
    """
    sidecar = _meta_sidecar_path(artifact_path)
    if not sidecar.exists():
        return
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(payload, dict):
        return
    payload["last_validated_at"] = last_validated_at.isoformat()
    payload["validator_version"] = int(validator_version)
    sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _auto_mint_allowed(benchmark_name: str | None) -> bool:
    """Decide whether runtime auto-mint is permitted for this benchmark."""
    override = os.environ.get(_AUTO_MINT_ENV, "").strip().lower()
    if override in {"1", "true", "yes"}:
        return True
    if override in {"0", "false", "no"}:
        return False
    # Default: true only for WebArena Verified (dummy creds in repo).
    if normalize_benchmark_name(benchmark_name) == "webarena_verified":
        return True
    return False


async def ensure_storage_state(
    instance: BenchmarkInstance,
    *,
    benchmark_root: Path | None,
    benchmark_name: str | None,
) -> Path | None:
    """Resolve a storage_state artifact, auto-minting if missing or stale.

    Returns the resolved artifact path when available, or ``None`` when the
    instance has no ``storage_state`` auth configured. Raises ``RuntimeError``
    when the artifact is missing, stale, structurally unusable, or auto-mint
    fails or is unavailable.
    """
    artifact_path, error = _resolve_storage_state_artifact_for_preflight(
        instance,
        benchmark_root=benchmark_root,
    )
    if error is None and artifact_path is not None:
        try:
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise RuntimeError(
                f"storage_state_empty: unable to read storage_state for {instance.site_name}: {exc}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"storage_state_empty: invalid JSON in storage_state for {instance.site_name}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"storage_state_empty: storage_state for {instance.site_name} does not contain a JSON object"
            )
        usable, unusable_reason = _artifact_is_structurally_usable(payload)
        if not usable:
            raise RuntimeError(f"{unusable_reason} for site {instance.site_name}")
        if storage_state_is_fresh(artifact_path):
            return artifact_path
        logger.info(
            "storage_state for %s is stale per sidecar TTL; re-minting",
            instance.site_name,
        )
    elif error is not None and not error.message.startswith("storage_state artifact missing"):
        raise RuntimeError(error.message)

    # Either the artifact is missing, or the sidecar says it is stale.
    if not _auto_mint_allowed(benchmark_name):
        raise RuntimeError(
            f"storage_state_stale: storage_state missing or stale for site {instance.site_name} "
            f"and auto-mint is disabled (set {_AUTO_MINT_ENV}=true to opt in)"
        )

    try:
        from warp_taskgen.phases.phase_0d_auth_bootstrap import (
            phase_0d_completion_path,
            reacquire_storage_state,
        )
        from warp_taskgen.phases.phase_0d_site_auth_specs import _extract_form_login_recipe
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            f"storage_state_stale: Phase 0d helpers unavailable for auto-mint: {exc}"
        ) from exc

    auth = instance.agent_auth if isinstance(instance.agent_auth, dict) else {}
    form_login = _extract_form_login_recipe(auth) if auth else None
    if form_login is None:
        raise RuntimeError(
            f"storage_state_stale: site {instance.site_name} has no form_login recipe; "
            "cannot auto-mint storage_state"
        )

    try:
        output_path = await reacquire_storage_state(
            site_name=instance.site_name,
            instance=instance.model_dump(),
            benchmark_root=benchmark_root,
        )
    except Exception as exc:
        raise RuntimeError(
            f"storage_state_stale: auto-mint storage_state failed for {instance.site_name}: {exc}"
        ) from exc
    write_storage_state_meta(output_path, mechanism="form_login_auto_heal")
    write_json_atomic(
        phase_0d_completion_path(instance.site_name),
        {
            "site": instance.site_name,
            "input_hash": None,
            "artifact_path": str(output_path),
            "dispatch": "form_login_auto_heal",
            "generator_script": None,
            "form_login": form_login,
            "agent_context_source": str(output_path.parent / "runtime_reacquire_context.json"),
            "site_url": instance.site_url,
            "auto_minted_at": datetime.now(UTC).isoformat(),
        },
    )
    if not storage_state_is_fresh(output_path):
        raise RuntimeError(
            f"storage_state_stale: auto-minted storage_state for {instance.site_name} is still stale"
        )
    return output_path
