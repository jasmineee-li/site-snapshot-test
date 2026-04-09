"""Shared helpers for app-mode manifests, attack metadata, and state payloads."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

from agentlab.benchmarks.redteam.phase_ids import PHASE_4A, PHASE_4B, PHASE_5
from agentlab.benchmarks.redteam.utils import (
    app_start_page_contract_error,
    normalize_route_reference,
    resolve_external_app_start_url,
)

logger = logging.getLogger(__name__)

GENERATION_STATUS_IN_PROGRESS = "in_progress"
GENERATION_STATUS_SUCCEEDED = "succeeded"
GENERATION_STATUS_FAILED = "failed"
APP_MANIFEST_CONTRACT_VERSION = 3
_DEFAULT_REQUIRED_VARIANTS = ("benign",)
_REQUIRED_RUNTIME_FILES = ("server.py", "index.html")
_LEGACY_STATE_WRAPPER_KEYS = frozenset({"state", "app_state"})
_STATE_WRAPPER_METADATA_KEYS = frozenset(
    {"ok", "success", "status", "message", "error", "metadata", "version"}
)
_SUPPORTED_DOMAIN_BINDING_MODES = frozenset({"primary_spa", "shim", "mock", "blocked"})
USER_MANUALS_ROOT = PurePosixPath("apps/user-manuals")
ATTACK_METADATA_ARCHIVE_FILENAME = "attack_metadata.json"
_MOCK_NAMESPACE = PurePosixPath("/mock")
_SHIM_NAMESPACE = PurePosixPath("/shim")
_USABLE_ROUTE_BINDING_MODES = frozenset({"primary_spa", "shim"})
_REDACTED_STATE_VALUE = "[REDACTED]"
_SENSITIVE_STATE_PATH_MARKERS = frozenset(
    {
        "password",
        "passcode",
        "secret",
        "token",
        "credential",
        "credentials",
        "authorization",
        "cookie",
        "session",
        "api_key",
        "apikey",
        "access_key",
        "refresh_token",
        "private_key",
        "email_body",
        "body",
        "content",
        "message",
        "messages",
        "chat",
        "conversation",
        "prompt",
        "instruction",
    }
)
_SENSITIVE_STATE_VALUE_MARKERS = frozenset(
    {
        "test_secret_",
        "/mock/file/",
        "/mock/mail/submit",
        "authorization:",
        "bearer ",
    }
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def repo_root() -> Path:
    """Return the repository root for Browser-Sim."""
    return Path(__file__).resolve().parents[5]


def resolve_repo_root_path(reference: str | Path | None = None) -> Path:
    """Infer the repository root for *reference* or fall back to Browser-Sim."""
    if reference is None:
        return repo_root()

    candidate = Path(reference).resolve()
    start = candidate if candidate.is_dir() else candidate.parent
    for current in (start, *start.parents):
        if (current / ".git").exists():
            return current
    for current in (start, *start.parents):
        if (current / USER_MANUALS_ROOT).exists():
            return current
    return start


def _path_component_error(name: str | Path, field_name: str) -> str | None:
    normalized = str(name or "").strip()
    if not normalized:
        return f"{field_name} must be a non-empty path component"
    if normalized in {".", ".."}:
        return f"{field_name} must not be '.' or '..'"
    if Path(normalized).is_absolute():
        return f"{field_name} must not be an absolute path"
    parts = PurePosixPath(normalized).parts
    if len(parts) != 1 or "/" in normalized or "\\" in normalized:
        return f"{field_name} must be a single path component"
    return None


def _normalize_binding_namespace_path(
    raw_path: str,
    *,
    field_name: str,
    domain: str,
    namespace: PurePosixPath,
) -> str:
    normalized = str(raw_path or "").strip()
    if not normalized:
        raise ValueError(f"{field_name.split('_')[0]} binding for {domain!r} is missing {field_name}")

    path = PurePosixPath(normalized)
    if not path.is_absolute():
        raise ValueError(
            f"{field_name.split('_')[0]} binding for {domain!r} must use an absolute {field_name}"
        )
    if any(part in {".", ".."} for part in path.parts):
        raise ValueError(
            f"{field_name.split('_')[0]} binding for {domain!r} must not contain dot segments"
        )

    normalized_path = path.as_posix()
    namespace_str = namespace.as_posix()
    if normalized_path != namespace_str and namespace_str not in {
        parent.as_posix() for parent in path.parents
    }:
        raise ValueError(
            f"{field_name.split('_')[0]} binding for {domain!r} must stay under {namespace_str}"
        )
    return normalized_path


def normalize_domain_binding_mock_prefix(raw_path: str, *, domain: str) -> str:
    mock_prefix = str(raw_path or "/mock").strip() or "/mock"
    return _normalize_binding_namespace_path(
        mock_prefix,
        field_name="mock_prefix",
        domain=domain,
        namespace=_MOCK_NAMESPACE,
    )


def normalize_domain_binding_shim_path(raw_path: str, *, domain: str) -> str:
    return _normalize_binding_namespace_path(
        raw_path,
        field_name="shim_path",
        domain=domain,
        namespace=_SHIM_NAMESPACE,
    )


def validate_path_component(name: str | Path, field_name: str) -> str:
    """Return a normalized single path component or raise ``ValueError``."""
    normalized = str(name or "").strip()
    error = _path_component_error(normalized, field_name)
    if error:
        raise ValueError(error)
    return normalized


def _ensure_no_symlinks(path: str | Path, *, field_name: str) -> None:
    path_obj = Path(path)
    current = Path(path_obj.anchor) if path_obj.is_absolute() else Path()
    parts = path_obj.parts[1:] if path_obj.is_absolute() else path_obj.parts
    for part in parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ValueError(f"{field_name} may not traverse symlinked path segments")


def ensure_within_dir(root: str | Path, candidate: str | Path, field_name: str) -> Path:
    """Resolve *candidate* and ensure it remains within *root*."""
    root_path = Path(root).resolve()
    candidate_path = Path(candidate)
    resolved = candidate_path.resolve(strict=False)
    try:
        resolved.relative_to(root_path)
    except ValueError as exc:
        raise ValueError(f"{field_name} resolves outside {root_path}") from exc
    return resolved


def resolve_repo_relative_path(
    repo_root_path: str | Path,
    raw_path: str | Path,
    field_name: str,
    allowed_prefixes: tuple[str | PurePosixPath, ...] | None = None,
) -> Path:
    """Resolve a repository-relative path, rejecting absolute/out-of-root targets."""
    normalized = str(raw_path or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")

    candidate = Path(normalized).expanduser()
    if candidate.is_absolute():
        raise ValueError(f"{field_name} must be relative to the repository root")

    root_path = Path(repo_root_path).resolve()
    candidate = root_path / candidate
    _ensure_no_symlinks(candidate, field_name=field_name)
    resolved = ensure_within_dir(root_path, candidate, field_name)
    relative = resolved.relative_to(root_path)
    relative_posix = PurePosixPath(relative.as_posix())

    if allowed_prefixes:
        normalized_prefixes = tuple(PurePosixPath(str(prefix)) for prefix in allowed_prefixes)
        if not any(
            relative_posix == prefix or prefix in relative_posix.parents
            for prefix in normalized_prefixes
        ):
            allowed = ", ".join(prefix.as_posix() for prefix in normalized_prefixes)
            raise ValueError(f"{field_name} must stay under one of: {allowed}")
    return resolved


def attack_metadata_archive_path(exp_dir: str | Path) -> Path:
    """Return the archived attack-metadata path for an experiment directory."""
    return Path(exp_dir) / ATTACK_METADATA_ARCHIVE_FILENAME


def resolve_variant_data_path(
    app_dir: str | Path,
    variant_name: str,
    *,
    field_name: str = "variant",
) -> Path:
    """Return a validated ``<variant>/data.js`` path beneath *app_dir*."""
    normalized_variant = validate_path_component(variant_name, field_name)
    app_root = Path(app_dir).resolve()
    candidate = app_root / normalized_variant / "data.js"
    ensure_within_dir(app_root, candidate, f"{field_name} data.js")
    return candidate


def resolve_docs_source_path(
    *,
    repo_root_path: str | Path | None = None,
    docs_path: str,
) -> Path | None:
    """Resolve a docs corpus path relative to the provided repository root."""
    if not docs_path.strip():
        return None
    root = Path(repo_root_path) if repo_root_path is not None else repo_root()
    return resolve_repo_relative_path(
        root,
        docs_path,
        "docs_path",
        allowed_prefixes=(USER_MANUALS_ROOT,),
    )


def docs_snapshot_fingerprint(snapshot: dict[str, Any]) -> str:
    """Return a stable fingerprint for docs snapshot comparisons."""
    if not isinstance(snapshot, dict):
        return ""
    material = {
        "docs_path": str(snapshot.get("docs_path") or ""),
        "source_type": str(snapshot.get("source_type") or ""),
        "repo_relative_path": str(snapshot.get("repo_relative_path") or ""),
        "corpus_sha256": str(snapshot.get("corpus_sha256") or ""),
        "files": list(snapshot.get("files") or []),
    }
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def compute_docs_snapshot(
    spec: dict[str, Any],
    *,
    repo_root_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compute the normalized docs snapshot for a behavior or app spec."""
    provided = spec.get("docs_snapshot")
    docs_path = str(spec.get("docs_path") or "").strip()
    root = Path(repo_root_path) if repo_root_path is not None else repo_root()
    docs_source = resolve_docs_source_path(repo_root_path=root, docs_path=docs_path)
    if docs_source is None:
        if isinstance(provided, dict) and provided:
            return provided
        return {
            "docs_path": docs_path,
            "captured_at": _timestamp(),
            "source_type": "missing",
            "files": [],
            "corpus_sha256": "",
        }

    files: list[dict[str, str]] = []
    if docs_source.is_file():
        relative_path = docs_source.name
        files.append(
            {
                "path": relative_path,
                "sha256": hashlib.sha256(docs_source.read_bytes()).hexdigest(),
            }
        )
        source_type = "file"
    elif docs_source.is_dir():
        for file_path in sorted(path for path in docs_source.rglob("*") if path.is_file()):
            files.append(
                {
                    "path": file_path.relative_to(docs_source).as_posix(),
                    "sha256": hashlib.sha256(file_path.read_bytes()).hexdigest(),
                }
            )
        source_type = "directory"
    else:
        source_type = "missing"

    fingerprint_material = {
        "docs_path": docs_path,
        "files": files,
        "source_type": source_type,
    }
    try:
        repo_relative_path = (
            docs_source.relative_to(root).as_posix() if docs_source.exists() else docs_path
        )
    except ValueError:
        repo_relative_path = docs_source.as_posix()
    snapshot = {
        "docs_path": docs_path,
        "captured_at": _timestamp(),
        "source_type": source_type,
        "repo_relative_path": repo_relative_path,
        "files": files,
        "file_count": len(files),
        "corpus_sha256": hashlib.sha256(
            json.dumps(
                fingerprint_material,
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest(),
    }
    if isinstance(provided, dict) and provided:
        snapshot["provided_metadata"] = provided
    return snapshot


def docs_snapshot_mismatch_error(
    *,
    manifest: dict[str, Any],
    behavior_spec: dict[str, Any] | None = None,
    repo_root_path: str | Path | None = None,
) -> str | None:
    """Return a resume-time docs drift error, else ``None``."""
    expected = manifest.get("docs_snapshot")
    if not isinstance(expected, dict) or not expected:
        return None
    current_spec = {
        "docs_path": (
            str(
                (
                    (behavior_spec or {}).get("docs_path")
                    or manifest.get("docs_path")
                    or ""
                )
            ).strip()
        ),
        "docs_snapshot": (behavior_spec or {}).get("docs_snapshot"),
    }
    if not current_spec["docs_path"]:
        return None
    current = compute_docs_snapshot(current_spec, repo_root_path=repo_root_path)
    if docs_snapshot_fingerprint(expected) == docs_snapshot_fingerprint(current):
        return None
    return (
        "Manual corpus changed after generation started; docs_snapshot is stale. "
        "Regenerate the shared app from scratch instead of resuming."
    )


def default_platform_manifest_path() -> Path:
    """Return the default top-level platform manifest path."""
    return repo_root() / "platform_manifest.json"


def behavior_contract_path(app_dir: str | Path, behavior_id: str) -> Path:
    """Return the canonical behavior contract path for a shared app."""
    normalized_behavior_id = validate_path_component(behavior_id, "behavior_id")
    return Path(app_dir) / "behaviors" / f"{normalized_behavior_id}.json"


def load_behavior_contract(app_dir: str | Path, behavior_id: str) -> dict[str, Any]:
    """Load ``behaviors/{behavior_id}.json`` if present, else return an empty dict."""
    path = behavior_contract_path(app_dir, behavior_id)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_platform_manifest(path: str | Path) -> dict[str, Any]:
    """Load the top-level ``platform_manifest.json``."""
    manifest_path = Path(path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    errors = validate_platform_manifest(
        payload,
        repo_root_path=resolve_repo_root_path(manifest_path),
    )
    if errors:
        raise ValueError("; ".join(errors))
    return payload


def validate_platform_manifest(
    platform_manifest: dict[str, Any],
    *,
    known_behavior_ids: set[str] | None = None,
    repo_root_path: str | Path | None = None,
) -> list[str]:
    """Return structural validation errors for ``platform_manifest.json``."""
    if not isinstance(platform_manifest, dict):
        return ["platform_manifest.json must contain a JSON object"]

    apps = platform_manifest.get("apps")
    if not isinstance(apps, list):
        return ["platform_manifest.json missing apps list"]
    if not apps:
        return ["platform_manifest.json must declare at least one app entry"]

    errors: list[str] = []
    seen_app_ids: set[str] = set()
    seen_behavior_ids: dict[str, str] = {}
    mapped_behavior_ids: set[str] = set()
    resolved_repo_root = (
        Path(repo_root_path).resolve() if repo_root_path is not None else repo_root()
    )

    for app_index, app in enumerate(apps):
        app_label = f"apps[{app_index}]"
        if not isinstance(app, dict):
            errors.append(f"{app_label} must be a JSON object")
            continue

        app_id = str(app.get("app_id") or "").strip()
        if not app_id:
            errors.append(f"{app_label} missing non-empty app_id")
        elif component_error := _path_component_error(app_id, f"{app_label} app_id"):
            errors.append(component_error)
        elif app_id in seen_app_ids:
            errors.append(f"{app_label} duplicates app_id {app_id!r}")
        else:
            seen_app_ids.add(app_id)

        platform = str(app.get("platform") or "").strip()
        if not platform:
            errors.append(f"{app_label} missing non-empty platform")

        docs_path = str(app.get("docs_path") or "").strip()
        if not docs_path:
            errors.append(f"{app_label} missing non-empty docs_path")
        else:
            try:
                resolve_repo_relative_path(
                    resolved_repo_root,
                    docs_path,
                    f"{app_label} docs_path",
                    allowed_prefixes=(USER_MANUALS_ROOT,),
                )
            except ValueError as exc:
                errors.append(str(exc))

        behaviors = app.get("behaviors")
        if not isinstance(behaviors, list) or not behaviors:
            errors.append(f"{app_label} missing non-empty behaviors list")
            continue

        app_behavior_ids: set[str] = set()
        for behavior_index, behavior in enumerate(behaviors):
            behavior_label = f"{app_label}.behaviors[{behavior_index}]"
            if not isinstance(behavior, dict):
                errors.append(f"{behavior_label} must be a JSON object")
                continue

            behavior_id = str(behavior.get("behavior_id") or "").strip()
            if not behavior_id:
                errors.append(f"{behavior_label} missing non-empty behavior_id")
                continue
            component_error = _path_component_error(behavior_id, f"{behavior_label} behavior_id")
            if component_error:
                errors.append(component_error)
                continue
            if behavior_id in app_behavior_ids:
                errors.append(
                    f"{behavior_label} duplicates behavior_id {behavior_id!r} within app {app_id or app_label!r}"
                )
            elif behavior_id in seen_behavior_ids:
                errors.append(
                    f"{behavior_label} duplicates behavior_id {behavior_id!r} already mapped by app "
                    f"{seen_behavior_ids[behavior_id]!r}"
                )
            else:
                app_behavior_ids.add(behavior_id)
                seen_behavior_ids[behavior_id] = app_id or app_label
                mapped_behavior_ids.add(behavior_id)

            entry_route = normalize_route_reference(str(behavior.get("entry_route") or ""))
            if not entry_route:
                errors.append(f"{behavior_label} missing non-empty entry_route")

            allowed_routes = behavior.get("allowed_routes")
            if not isinstance(allowed_routes, list) or not allowed_routes:
                errors.append(f"{behavior_label} missing non-empty allowed_routes")
            else:
                normalized_allowed_routes = _normalize_routes(allowed_routes)
                if entry_route and entry_route not in normalized_allowed_routes:
                    errors.append(
                        f"{behavior_label} entry_route {entry_route!r} is not present in allowed_routes"
                    )

            domain_bindings = behavior.get("domain_bindings")
            if not isinstance(domain_bindings, list) or not domain_bindings:
                errors.append(f"{behavior_label} missing non-empty domain_bindings")
            else:
                binding_error = _domain_bindings_contract_error(domain_bindings)
                if binding_error:
                    errors.append(f"{behavior_label} invalid domain_bindings: {binding_error}")

            mapping_rationale = behavior.get("mapping_rationale")
            if mapping_rationale is not None and not str(mapping_rationale).strip():
                errors.append(f"{behavior_label} mapping_rationale must be non-empty when provided")

    if known_behavior_ids is not None:
        extra_behavior_ids = sorted(mapped_behavior_ids - known_behavior_ids)
        if extra_behavior_ids:
            errors.append(
                "platform_manifest.json references unknown behavior_ids: "
                + ", ".join(extra_behavior_ids)
            )
        missing_behavior_ids = sorted(known_behavior_ids - mapped_behavior_ids)
        if missing_behavior_ids:
            errors.append(
                "platform_manifest.json is missing mappings for behavior_ids: "
                + ", ".join(missing_behavior_ids)
            )

    return errors


def platform_manifest_docs_path_errors(
    platform_manifest: dict[str, Any],
    *,
    repo_root_path: str | Path | None = None,
) -> list[str]:
    """Return docs-path existence errors for ``platform_manifest.json``."""
    errors: list[str] = []
    root = Path(repo_root_path) if repo_root_path is not None else repo_root()
    for app_index, app in enumerate(platform_manifest.get("apps") or []):
        if not isinstance(app, dict):
            continue
        docs_path = str(app.get("docs_path") or "").strip()
        if not docs_path:
            continue
        try:
            docs_source = resolve_repo_relative_path(
                root,
                docs_path,
                f"apps[{app_index}] docs_path",
                allowed_prefixes=(USER_MANUALS_ROOT,),
            )
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if not docs_source.exists():
            errors.append(
                f"apps[{app_index}] docs_path {docs_path!r} does not exist relative to {root}"
            )
    return errors


def resolve_platform_behavior_mapping(
    platform_manifest: dict[str, Any],
    behavior_id: str,
) -> dict[str, Any]:
    """Resolve a behavior mapping entry from the loaded platform manifest."""
    for app in platform_manifest.get("apps", []) or []:
        if not isinstance(app, dict):
            continue
        app_id = str(app.get("app_id") or "").strip()
        if not app_id:
            continue
        for raw_behavior in app.get("behaviors", []) or []:
            if not isinstance(raw_behavior, dict):
                continue
            mapped_behavior_id = str(raw_behavior.get("behavior_id") or "").strip()
            if mapped_behavior_id != behavior_id:
                continue
            return {
                "app_id": app_id,
                "platform": app.get("platform", ""),
                "docs_path": app.get("docs_path", ""),
                "app_dir": app.get("app_dir", ""),
                "behavior": raw_behavior,
                "app": app,
            }
    raise KeyError(f"platform_manifest.json is missing mapping for behavior_id {behavior_id!r}")


def load_app_manifest(app_dir: str | Path) -> dict[str, Any]:
    """Load ``app_manifest.json`` if present, else return an empty dict."""
    path = Path(app_dir) / "app_manifest.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def app_manifest_contract_version(manifest: dict[str, Any]) -> int | None:
    """Return the declared manifest contract version, if present."""
    version = manifest.get("contract_version")
    return version if isinstance(version, int) else None


def is_supported_app_manifest_contract(manifest: dict[str, Any]) -> bool:
    """Return whether *manifest* uses a supported app contract version."""
    if not manifest:
        return False

    version = app_manifest_contract_version(manifest)
    if version is not None:
        return version >= APP_MANIFEST_CONTRACT_VERSION
    return False


def require_supported_app_manifest(
    manifest: dict[str, Any],
    *,
    context: str,
) -> None:
    """Raise when *manifest* is not supported for hardened app-mode flows."""
    if is_supported_app_manifest_contract(manifest):
        return

    version = app_manifest_contract_version(manifest)
    if version is None:
        raise ValueError(
            f"{context} requires app_manifest.json contract v"
            f"{APP_MANIFEST_CONTRACT_VERSION}+; regenerate the app directory."
        )
    raise ValueError(
        f"{context} requires app_manifest.json contract v"
        f"{APP_MANIFEST_CONTRACT_VERSION}+; found v{version}."
    )


def is_successful_app_manifest(
    manifest: dict[str, Any],
    *,
    require_functional_tests: bool = False,
) -> bool:
    """Return whether an app manifest represents a fully successful generation."""
    if not manifest:
        return False
    if not is_supported_app_manifest_contract(manifest):
        return False

    generation = manifest.get("generation") or {}
    status = generation.get("status")

    if status == GENERATION_STATUS_SUCCEEDED:
        return app_manifest_readiness_error(manifest) is None

    if status in {GENERATION_STATUS_FAILED, GENERATION_STATUS_IN_PROGRESS}:
        return False
    return False


def app_manifest_readiness_error(manifest: dict[str, Any]) -> str | None:
    """Return the first manifest-level readiness error, else ``None``."""
    if not manifest:
        return "missing app_manifest.json"
    if not is_supported_app_manifest_contract(manifest):
        version = app_manifest_contract_version(manifest)
        if version is None:
            return (
                "app_manifest.json requires contract "
                f"v{APP_MANIFEST_CONTRACT_VERSION}+; regenerate the app directory"
            )
        return (
            "app_manifest.json requires contract "
            f"v{APP_MANIFEST_CONTRACT_VERSION}+; found v{version}"
        )
    app_id = str(manifest.get("app_id") or "").strip()
    if not app_id:
        return "app_manifest.json missing non-empty app_id"
    platform = str(manifest.get("platform") or "").strip()
    if not platform:
        return "app_manifest.json missing non-empty platform"
    docs_path = str(manifest.get("docs_path") or "").strip()
    if not docs_path:
        return "app_manifest.json missing non-empty docs_path"
    try:
        resolve_repo_relative_path(
            repo_root(),
            docs_path,
            "app_manifest.json docs_path",
            allowed_prefixes=(USER_MANUALS_ROOT,),
        )
    except ValueError as exc:
        return str(exc)
    docs_snapshot = manifest.get("docs_snapshot")
    if not isinstance(docs_snapshot, dict) or not docs_snapshot:
        return "app_manifest.json missing docs_snapshot"
    docs_snapshot_hash = str(docs_snapshot.get("corpus_sha256") or "").strip()
    if not docs_snapshot_hash:
        return "app_manifest.json docs_snapshot missing corpus_sha256"
    if str(docs_snapshot.get("source_type") or "").strip() == "missing":
        return "app_manifest.json docs_snapshot source is missing"
    if not isinstance(manifest.get("shared_seed_version"), int):
        return "app_manifest.json missing integer shared_seed_version"
    shared_seed_hash = str(manifest.get("shared_seed_hash") or "").strip()
    if not shared_seed_hash:
        return "app_manifest.json missing non-empty shared_seed_hash"
    behavior_ids = manifest.get("behavior_ids")
    if not isinstance(behavior_ids, list) or not all(
        isinstance(value, str) and value.strip() for value in behavior_ids
    ):
        return "app_manifest.json missing behavior_ids list"
    normalized_behavior_ids = [str(value).strip() for value in behavior_ids]
    if len(set(normalized_behavior_ids)) != len(normalized_behavior_ids):
        return "app_manifest.json has duplicate behavior_ids"
    route_contract_error = app_start_page_contract_error(
        manifest.get("start_page", ""),
        manifest.get("pages", []),
    )
    if route_contract_error:
        return f"app_manifest.json invalid start_page/pages contract: {route_contract_error}"

    generation = manifest.get("generation") or {}
    if generation.get("status") != GENERATION_STATUS_SUCCEEDED:
        return "app manifest generation status is not succeeded"
    if (manifest.get("variant_generation") or {}).get("status") != "validated":
        return "app manifest variant generation status is not validated"
    if (manifest.get("validation") or {}).get("passed") is not True:
        return "app manifest validation did not pass"
    if generation.get("functional_tests_requested") is not True:
        return "app manifest did not request benchmark-readiness functional tests"
    if not functional_tests_complete(manifest):
        return "functional test metadata incomplete"
    if not functional_quality_gate_passed(manifest):
        return "functional readiness quality gate did not pass"
    if not real_task_quality_gate_passed(manifest):
        return "real-task benign readiness gate did not pass"
    phases = generation.get("phases") or {}
    hardening_requested = int(generation.get("hardening_rounds", 0) or 0) > 0
    hardening_phase_status = (phases.get(PHASE_4B) or {}).get("status") or (phases.get(PHASE_4A) or {}).get(
        "status"
    )
    if hardening_requested:
        if hardening_phase_status != "succeeded":
            return "requested hardening phase did not succeed"
        hardening_result = (manifest.get("functional_tests") or {}).get("task_hardening") or {}
        if hardening_result.get("ran") is not True or hardening_result.get("error") not in (None, ""):
            return "requested hardening artifacts are missing or failed"
    elif hardening_phase_status != "skipped":
        return "disabled hardening phase must be marked skipped"

    regression_requested = generation.get("run_final_regression") is True
    regression_phase_status = (phases.get(PHASE_5) or {}).get("status")
    if regression_requested:
        if regression_phase_status != "succeeded":
            return "requested final regression phase did not succeed"
        regression_result = (manifest.get("functional_tests") or {}).get("final_regression") or {}
        if regression_result.get("ran") is not True or regression_result.get("passed") is not True:
            return "requested final regression artifacts are missing or failed"
    elif regression_phase_status != "skipped":
        return "disabled final regression phase must be marked skipped"
    return None


def app_manifest_seed_signature(manifest: dict[str, Any]) -> tuple[int | None, str]:
    """Return the app seed version/hash tuple used to validate behavior compatibility."""
    version = manifest.get("shared_seed_version")
    normalized_version = version if isinstance(version, int) else None
    return normalized_version, str(manifest.get("shared_seed_hash") or "").strip()


def behavior_contract_compatibility_error(
    behavior_contract: dict[str, Any],
    *,
    manifest: dict[str, Any],
    behavior_id: str,
) -> str | None:
    """Return the first behavior-level compatibility error, else ``None``."""
    if not behavior_contract:
        return f"missing behaviors/{behavior_id}.json"

    contract_behavior_id = str(behavior_contract.get("behavior_id") or "").strip()
    if contract_behavior_id != behavior_id:
        return (
            f"behaviors/{behavior_id}.json has mismatched behavior_id "
            f"{contract_behavior_id!r}"
        )

    app_id = str(behavior_contract.get("app_id") or "").strip()
    manifest_app_id = str(manifest.get("app_id") or "").strip()
    if not app_id:
        return f"behaviors/{behavior_id}.json missing non-empty app_id"
    if manifest_app_id and app_id != manifest_app_id:
        return (
            f"behaviors/{behavior_id}.json app_id {app_id!r} does not match "
            f"app_manifest.json app_id {manifest_app_id!r}"
        )
    manifest_behavior_ids = {
        str(value).strip()
        for value in (manifest.get("behavior_ids") or [])
        if isinstance(value, str) and value.strip()
    }
    if manifest_behavior_ids and behavior_id not in manifest_behavior_ids:
        return (
            f"behaviors/{behavior_id}.json behavior_id {behavior_id!r} is not declared "
            "in app_manifest.json behavior_ids"
        )

    safe_behavior = str(behavior_contract.get("safe_behavior") or "").strip()
    if not safe_behavior:
        return f"behaviors/{behavior_id}.json missing non-empty safe_behavior"
    success_condition = str(behavior_contract.get("success_condition") or "").strip()
    if not success_condition:
        return f"behaviors/{behavior_id}.json missing non-empty success_condition"
    entry_route = normalize_route_reference(str(behavior_contract.get("entry_route") or ""))
    if not entry_route:
        return f"behaviors/{behavior_id}.json missing non-empty entry_route"

    allowed_routes = behavior_contract.get("allowed_routes")
    if not isinstance(allowed_routes, list) or not allowed_routes:
        return f"behaviors/{behavior_id}.json missing allowed_routes"
    normalized_allowed_routes = _normalize_routes(allowed_routes)
    if entry_route not in normalized_allowed_routes:
        return (
            f"behaviors/{behavior_id}.json entry_route {entry_route!r} is not present in "
            "allowed_routes"
        )

    domain_bindings = behavior_contract.get("domain_bindings")
    if not isinstance(domain_bindings, list) or not domain_bindings:
        return f"behaviors/{behavior_id}.json missing domain_bindings"
    binding_error = _domain_bindings_contract_error(domain_bindings)
    if binding_error:
        return f"behaviors/{behavior_id}.json invalid domain_bindings: {binding_error}"
    manifest_pages = manifest.get("pages")
    if not isinstance(manifest_pages, list) or not manifest_pages:
        return (
            f"behaviors/{behavior_id}.json cannot verify routes because "
            "app_manifest.json pages are missing"
        )
    route_error = _behavior_route_binding_error(
        entry_route,
        route_field="entry_route",
        behavior_id=behavior_id,
        manifest_pages=manifest_pages,
        domain_bindings=domain_bindings,
    )
    if route_error:
        return route_error
    for route in normalized_allowed_routes:
        route_error = _behavior_route_binding_error(
            route,
            route_field="allowed_routes entry",
            behavior_id=behavior_id,
            manifest_pages=manifest_pages,
            domain_bindings=domain_bindings,
        )
        if route_error:
            return route_error

    compatibility_status = str(behavior_contract.get("compatibility_status") or "").strip()
    if compatibility_status != "passed":
        return (
            f"behaviors/{behavior_id}.json compatibility_status must be 'passed'; "
            f"found {compatibility_status or 'missing'}"
        )
    evidence = behavior_contract.get("compatibility_evidence")
    if not isinstance(evidence, dict):
        return f"behaviors/{behavior_id}.json missing compatibility_evidence"
    app_seed_version, app_seed_hash = app_manifest_seed_signature(manifest)
    if evidence.get("checked_against_seed_version") != app_seed_version:
        return (
            f"behaviors/{behavior_id}.json checked_against_seed_version does not match "
            "app_manifest.json shared_seed_version"
        )
    if str(evidence.get("checked_against_seed_hash") or "").strip() != app_seed_hash:
        return (
            f"behaviors/{behavior_id}.json checked_against_seed_hash does not match "
            "app_manifest.json shared_seed_hash"
        )

    variants = behavior_contract.get("variants")
    if not isinstance(variants, list) or not variants:
        return f"behaviors/{behavior_id}.json missing variants"
    active_variant = str(behavior_contract.get("active_variant") or "").strip()
    if not active_variant:
        return f"behaviors/{behavior_id}.json missing active_variant"
    try:
        validate_path_component(active_variant, f"behaviors/{behavior_id}.json active_variant")
    except ValueError as exc:
        return str(exc)
    variants_by_name: dict[str, dict[str, Any]] = {}
    for item in variants:
        if not isinstance(item, dict):
            continue
        variant_name = str(item.get("name") or "").strip()
        if variant_name:
            variants_by_name[variant_name] = item
    for variant_name in variants_by_name:
        try:
            validate_path_component(variant_name, f"behaviors/{behavior_id}.json variant name")
        except ValueError as exc:
            return str(exc)
    if active_variant not in variants_by_name:
        return (
            f"behaviors/{behavior_id}.json active_variant {active_variant!r} is not present "
            "in variants lineage"
        )
    active_variant_status = str(
        (variants_by_name.get(active_variant) or {}).get("status") or ""
    ).strip()
    if active_variant_status != "validated":
        return (
            f"behaviors/{behavior_id}.json active_variant {active_variant!r} must reference "
            f"a validated lineage entry; found status {active_variant_status or 'missing'}"
        )
    return None


def functional_quality_gate_passed(manifest: dict[str, Any]) -> bool:
    """Return whether embedded functional-test quality-gate metadata passed."""
    functional_tests = manifest.get("functional_tests") or {}
    quality_gate = functional_tests.get("quality_gate") or {}
    return quality_gate.get("passed") is True


def real_task_quality_gate_passed(manifest: dict[str, Any]) -> bool:
    """Return whether embedded real-task readiness metadata passed."""
    functional_tests = manifest.get("functional_tests") or {}
    quality_gate = functional_tests.get("real_quality_gate") or {}
    return quality_gate.get("passed") is True


def functional_tests_complete(manifest: dict[str, Any]) -> bool:
    """Return whether manifest metadata records a complete dual-suite readiness run."""
    functional_tests = manifest.get("functional_tests") or {}
    if not functional_tests:
        return False
    if functional_tests.get("function_sanity_passed") is not True:
        return False
    if functional_tests.get("real_sanity_passed") is not True:
        return False
    return _suite_evaluation_complete(functional_tests.get("function_evaluation")) and _suite_evaluation_complete(
        functional_tests.get("real_evaluation")
    )


def _suite_evaluation_complete(summary: dict[str, Any] | None) -> bool:
    if not isinstance(summary, dict):
        return False
    if summary.get("ran") is not True:
        return False
    if summary.get("error") not in (None, ""):
        return False
    if not isinstance(summary.get("pass_rate"), (int, float)):
        return False
    if not isinstance(summary.get("total"), int):
        return False
    if not isinstance(summary.get("passed"), int):
        return False
    return True


def benchmark_ready_app_dir_error(
    app_dir: str | Path,
    *,
    manifest: dict[str, Any] | None = None,
    behavior_contract: dict[str, Any] | None = None,
    behavior_id: str | None = None,
    required_variant_subdirs: tuple[str, ...] | list[str] | None = None,
    functional_threshold: float | None = None,
    requested_functional_backend: str | None = None,
) -> str | None:
    """Return the first directory-level readiness error, else ``None``."""
    app_dir = Path(app_dir)
    if manifest is not None:
        effective_manifest = manifest
    else:
        manifest_path = app_dir / "app_manifest.json"
        if not manifest_path.exists():
            effective_manifest = {}
        else:
            try:
                effective_manifest = load_app_manifest(app_dir)
            except (OSError, json.JSONDecodeError) as exc:
                return f"unreadable app_manifest.json ({exc})"

    normalized_requested_backend = _normalize_functional_backend(requested_functional_backend)
    manifest_generation_backend = _normalize_functional_backend(
        (effective_manifest.get("generation") or {}).get("functional_backend")
    )

    manifest_error = app_manifest_readiness_error(effective_manifest)
    if manifest_error:
        return manifest_error
    if behavior_contract is not None:
        compatibility_error = behavior_contract_compatibility_error(
            behavior_contract,
            manifest=effective_manifest,
            behavior_id=str(behavior_id or behavior_contract.get("behavior_id") or "").strip(),
        )
        if compatibility_error:
            return compatibility_error
    if (
        normalized_requested_backend
        and manifest_generation_backend
        and manifest_generation_backend != normalized_requested_backend
    ):
        return (
            f"requested functional backend '{normalized_requested_backend}' does not match "
            f"app_manifest.json backend '{manifest_generation_backend}'"
        )

    functional_results_error = _functional_results_readiness_error(
        app_dir,
        backend=normalized_requested_backend or manifest_generation_backend,
        manifest=effective_manifest,
        functional_threshold=functional_threshold,
    )
    if functional_results_error:
        return functional_results_error

    return _required_app_assets_error(
        app_dir,
        behavior_contract=behavior_contract,
        required_variant_subdirs=required_variant_subdirs,
    )


def is_benchmark_ready_app_dir(
    app_dir: str | Path,
    *,
    manifest: dict[str, Any] | None = None,
    behavior_contract: dict[str, Any] | None = None,
    behavior_id: str | None = None,
    required_variant_subdirs: tuple[str, ...] | list[str] | None = None,
    functional_threshold: float | None = None,
    requested_functional_backend: str | None = None,
) -> bool:
    """Return whether ``app_dir`` is benchmark-ready for hardened app-mode use."""
    return benchmark_ready_app_dir_error(
        app_dir,
        manifest=manifest,
        behavior_contract=behavior_contract,
        behavior_id=behavior_id,
        required_variant_subdirs=required_variant_subdirs,
        functional_threshold=functional_threshold,
        requested_functional_backend=requested_functional_backend,
    ) is None


def _normalize_functional_backend(backend: str | None) -> str | None:
    if backend is None:
        return None
    from agentlab.benchmarks.redteam.eval_harness import normalize_functional_backend

    return normalize_functional_backend(backend)


def _functional_results_readiness_error(
    app_dir: Path,
    *,
    backend: str | None = None,
    manifest: dict[str, Any] | None = None,
    functional_threshold: float | None = None,
) -> str | None:
    path = app_dir / "functional_results.json"
    if not path.exists():
        return "missing functional_results.json"

    try:
        results = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"unreadable functional_results.json ({exc})"

    declared_backends = results.get("backends")
    if not isinstance(declared_backends, dict) or not declared_backends:
        return "functional_results.json missing declared backends"

    scoped_results = results
    if backend:
        if backend not in declared_backends:
            return f"functional_results.json missing declared backend block: {backend}"
        scoped_results = declared_backends[backend]
    elif len(declared_backends) == 1:
        scoped_results = next(iter(declared_backends.values()))
    else:
        return "functional_results.json missing declared backend selection"

    pass_rate = scoped_results.get("function_pass_rate")
    real_pass_rate = scoped_results.get("real_pass_rate")
    function_results = (scoped_results.get("function_tasks") or {}).get("results")
    real_results = (scoped_results.get("real_tasks") or {}).get("results")

    if not isinstance(pass_rate, (int, float)):
        return "functional_results.json missing numeric function_pass_rate"
    if not isinstance(real_pass_rate, (int, float)):
        return "functional_results.json missing numeric real_pass_rate"
    if not isinstance(function_results, list) or not isinstance(real_results, list):
        return "functional_results.json missing suite result lists"
    if functional_threshold is not None and pass_rate < functional_threshold:
        return (
            "functional_results.json function pass rate "
            f"{pass_rate * 100:.1f}% < {functional_threshold * 100:.0f}%"
        )
    if manifest:
        comparison_source = scoped_results
        comparison_label = "functional_results.json"
        readiness_baseline = scoped_results.get("readiness_baseline")
        if readiness_baseline is not None:
            if not isinstance(readiness_baseline, dict):
                return "functional_results.json readiness_baseline is not a JSON object"
            comparison_source = readiness_baseline
            comparison_label = "functional_results.json readiness_baseline"

        manifest_function = ((manifest.get("functional_tests") or {}).get("function_evaluation") or {})
        manifest_real = ((manifest.get("functional_tests") or {}).get("real_evaluation") or {})
        normalized_backend = _normalize_functional_backend(backend)
        manifest_generation_backend = _normalize_functional_backend(
            (manifest.get("generation") or {}).get("functional_backend")
        )
        manifest_function_backend = _normalize_functional_backend(manifest_function.get("backend"))
        manifest_real_backend = _normalize_functional_backend(manifest_real.get("backend"))
        manifest_backend = (
            normalized_backend
            or manifest_generation_backend
            or manifest_function_backend
            or manifest_real_backend
        )
        if normalized_backend and manifest_generation_backend and manifest_generation_backend != normalized_backend:
            return (
                f"requested functional backend '{normalized_backend}' does not match "
                f"app_manifest.json backend '{manifest_generation_backend}'"
            )
        if normalized_backend and manifest_function_backend != normalized_backend:
            return "app manifest function evaluation backend mismatch"
        if normalized_backend and manifest_real_backend != normalized_backend:
            return "app manifest real evaluation backend mismatch"
        scoped_results_backend = _normalize_functional_backend(scoped_results.get("backend"))
        comparison_backend = _normalize_functional_backend(comparison_source.get("backend"))
        if manifest_backend and scoped_results_backend not in (None, manifest_backend):
            return "functional_results.json backend mismatch"
        if manifest_backend and comparison_backend not in (None, manifest_backend):
            return f"{comparison_label} backend mismatch"
        comparison_function_rate = comparison_source.get("function_pass_rate")
        comparison_real_rate = comparison_source.get("real_pass_rate")
        comparison_function_results = (comparison_source.get("function_tasks") or {}).get("results")
        comparison_real_results = (comparison_source.get("real_tasks") or {}).get("results")
        if not isinstance(comparison_function_rate, (int, float)):
            return f"{comparison_label} missing numeric function_pass_rate"
        if not isinstance(comparison_real_rate, (int, float)):
            return f"{comparison_label} missing numeric real_pass_rate"
        if not isinstance(comparison_function_results, list) or not isinstance(comparison_real_results, list):
            return f"{comparison_label} missing suite result lists"
        for manifest_summary, suite_key, suite_name, rate_key in (
            (manifest_function, "function_tasks", "function", "function_pass_rate"),
            (manifest_real, "real_tasks", "real", "real_pass_rate"),
        ):
            disk_summary = comparison_source.get(suite_key) or {}
            if _normalize_functional_backend(manifest_summary.get("backend")) != manifest_backend:
                return f"app manifest {suite_name} evaluation backend mismatch"
            if manifest_summary.get("pass_rate") != comparison_source.get(rate_key):
                return f"app manifest {suite_name} evaluation pass_rate mismatch"
            if manifest_summary.get("total") != disk_summary.get("total"):
                return f"app manifest {suite_name} evaluation total mismatch"
            if manifest_summary.get("passed") != disk_summary.get("passed"):
                return f"app manifest {suite_name} evaluation passed mismatch"
    return None


def _required_app_assets_error(
    app_dir: Path,
    *,
    behavior_contract: dict[str, Any] | None = None,
    required_variant_subdirs: tuple[str, ...] | list[str] | None = None,
) -> str | None:
    for filename in _REQUIRED_RUNTIME_FILES:
        if not (app_dir / filename).exists():
            return f"missing required app asset: {filename}"

    derived_variants = list(required_variant_subdirs or _DEFAULT_REQUIRED_VARIANTS)
    if behavior_contract is not None:
        active_variant = str(behavior_contract.get("active_variant") or "").strip()
        if active_variant:
            derived_variants.append(active_variant)
    variants = tuple(dict.fromkeys(derived_variants))
    for variant in variants:
        try:
            variant_data = resolve_variant_data_path(app_dir, variant)
        except ValueError as exc:
            return str(exc)
        if not variant_data.exists():
            return f"missing required variant asset: {variant}/data.js"
    return None


def is_normalized_state_payload(payload: Any) -> bool:
    """Return whether ``payload`` matches the app-mode ``/api/state`` contract."""
    if not isinstance(payload, dict):
        return False

    return not any(isinstance(payload.get(key), dict) for key in _LEGACY_STATE_WRAPPER_KEYS)


def fetch_normalized_state(server_port: int, timeout: float = 5) -> dict[str, Any] | None:
    """Fetch structured state from ``/api/state``."""
    try:
        url = f"http://127.0.0.1:{server_port}/api/state"
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            payload = json.loads(resp.read())
        if not is_normalized_state_payload(payload):
            logger.warning(
                "Received unsupported /api/state payload shape from port %s",
                server_port,
            )
            return None
        return payload
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        logger.warning("Failed to fetch normalized app state from port %s: %s", server_port, exc)
        return None


def redact_state_payload(payload: Any) -> Any:
    """Redact likely-secret or message-body values from persisted state artifacts."""
    return _redact_state_value(payload, key_path=())


def is_sensitive_state_path(path: str | tuple[str, ...]) -> bool:
    """Return whether a state path likely contains secrets or message bodies."""
    for token in _state_path_tokens(path):
        if token in _SENSITIVE_STATE_PATH_MARKERS:
            return True
        if any(marker in token for marker in _SENSITIVE_STATE_PATH_MARKERS if len(marker) > 5):
            return True
    return False


def _redact_state_value(value: Any, *, key_path: tuple[str, ...]) -> Any:
    if is_sensitive_state_path(key_path):
        return _REDACTED_STATE_VALUE

    if isinstance(value, dict):
        return {
            key: _redact_state_value(item, key_path=(*key_path, str(key)))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_state_value(item, key_path=key_path) for item in value]
    if isinstance(value, str) and _state_value_contains_sensitive_marker(value):
        return _REDACTED_STATE_VALUE
    return value


def _state_path_tokens(path: str | tuple[str, ...]) -> list[str]:
    if isinstance(path, tuple):
        raw_tokens = [str(token) for token in path if str(token)]
    else:
        raw_tokens = re.findall(r"[A-Za-z0-9_]+", str(path))
    return [token.lower() for token in raw_tokens if token]


def _state_value_contains_sensitive_marker(value: str) -> bool:
    normalized = str(value or "").lower()
    return any(marker in normalized for marker in _SENSITIVE_STATE_VALUE_MARKERS)


def read_variant_data_js(path: Path) -> dict[str, Any]:
    """Parse a ``data.js`` variant file into Python objects."""
    from agentlab.benchmarks.redteam.data_validator import load_data_js_constants

    return load_data_js_constants(path)


def diff_variant_data(
    benign_data: dict[str, Any],
    variant_data: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Return items present in the variant but absent from the benign seed."""
    injections: dict[str, list[dict[str, Any]]] = {}

    for key, variant_value in variant_data.items():
        if key == "SEED_DATA_VERSION":
            continue
        benign_value = benign_data.get(key)
        if not isinstance(variant_value, list) or not isinstance(benign_value, list):
            continue

        benign_ids = {
            item["id"]
            for item in benign_value
            if isinstance(item, dict) and "id" in item
        }
        extra_items = [
            item
            for item in variant_value
            if isinstance(item, dict) and item.get("id") not in benign_ids
        ]
        if extra_items:
            injections[key] = extra_items

    return injections


def build_attack_metadata(
    behavior_spec: dict[str, Any],
    pages: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Persist stable app-mode attack metadata into ``app_manifest.json``."""
    adv_spec = behavior_spec.get("adversarial_data_spec") or {}
    explicit_routes = (
        adv_spec.get("display_routes")
        or adv_spec.get("routes")
        or adv_spec.get("injection_locations")
        or []
    )
    display_routes = _normalize_routes(explicit_routes)

    surface_summaries: list[str] = []
    if display_routes:
        route_descriptions = _route_descriptions(pages or [])
        for route in display_routes:
            description = route_descriptions.get(route)
            if description:
                surface_summaries.append(f"{route} - {description}")
            else:
                surface_summaries.append(route)
    else:
        display_routes, surface_summaries = _infer_attack_surfaces(
            pages or [],
            adv_spec,
        )

    return {
        "schema_version": 1,
        "injection_target": adv_spec.get("injection_target", ""),
        "injection_description": adv_spec.get("injection_description", ""),
        "benign_analog": adv_spec.get("benign_analog", ""),
        "display_routes": display_routes,
        "surface_summaries": surface_summaries,
        "exposure_known": bool(display_routes),
    }


def _domain_bindings_contract_error(domain_bindings: list[Any]) -> str | None:
    seen_domains: set[str] = set()
    for binding in domain_bindings:
        if not isinstance(binding, dict):
            return "every binding must be a JSON object"
        domain = str(binding.get("domain") or "").strip()
        mode = str(binding.get("mode") or "").strip()
        if not domain:
            return "binding missing non-empty domain"
        if domain in seen_domains:
            return f"duplicate binding for domain {domain!r}"
        seen_domains.add(domain)
        if mode not in _SUPPORTED_DOMAIN_BINDING_MODES:
            return f"unsupported mode {mode!r}"
        if mode == "mock":
            try:
                normalize_domain_binding_mock_prefix(
                    str(binding.get("mock_prefix") or "/mock"),
                    domain=domain,
                )
            except ValueError as exc:
                return str(exc)
        if mode == "shim":
            try:
                normalize_domain_binding_shim_path(
                    str(binding.get("shim_path") or ""),
                    domain=domain,
                )
            except ValueError as exc:
                return str(exc)
    return None


def _behavior_route_binding_error(
    route: str,
    *,
    route_field: str,
    behavior_id: str,
    manifest_pages: list[Any],
    domain_bindings: list[Any],
) -> str | None:
    try:
        resolved_url = resolve_external_app_start_url(_route_resolution_reference(route), manifest_pages)
    except ValueError:
        return (
            f"behaviors/{behavior_id}.json {route_field} {route!r} is not declared by "
            "app_manifest.json pages"
        )
    resolved_domain = urlparse(resolved_url).netloc

    binding_modes = _binding_modes_by_domain(domain_bindings)
    if binding_modes.get(resolved_domain) in _USABLE_ROUTE_BINDING_MODES:
        return None

    return (
        f"behaviors/{behavior_id}.json {route_field} {route!r} does not resolve to a usable "
        f"domain binding: {resolved_domain} ({binding_modes.get(resolved_domain, 'missing')})"
    )


def _route_resolution_reference(route: str) -> str:
    normalized_route = str(route or "").strip()
    if normalized_route.startswith("/#"):
        return normalized_route[1:]
    return normalized_route


def _binding_modes_by_domain(domain_bindings: list[Any]) -> dict[str, str]:
    binding_modes: dict[str, str] = {}
    for binding in domain_bindings or []:
        if not isinstance(binding, dict):
            continue
        domain = str(binding.get("domain") or "").strip()
        mode = str(binding.get("mode") or "").strip()
        if domain:
            binding_modes[domain] = mode
    return binding_modes


def load_attack_metadata(
    _exp_dir: str | Path,
    *,
    app_dir: str | Path = "",
    behavior_id: str = "",
    variant: str = "",
    test_mode: str = "unknown",
) -> dict[str, Any]:
    """Load normalized attack metadata for app-mode runs."""
    archive_path = attack_metadata_archive_path(_exp_dir)
    if archive_path.exists():
        with open(archive_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"Archived attack metadata at {archive_path} must contain a JSON object"
            )
        current_metadata = _load_app_attack_metadata(
            app_dir=app_dir,
            behavior_id=behavior_id,
            variant=variant,
            test_mode=test_mode,
        )
        expected_context = _attack_metadata_archive_context(
            app_dir=app_dir,
            behavior_id=behavior_id,
            variant=variant,
            test_mode=test_mode,
        )
        archived_context = payload.get("archive_context")
        if archived_context != expected_context:
            logger.warning(
                "Ignoring archived attack metadata at %s due to provenance mismatch",
                archive_path,
            )
            return current_metadata
        if _normalized_attack_metadata_payload(payload) != current_metadata:
            logger.warning(
                "Ignoring archived attack metadata at %s because it no longer matches app artifacts",
                archive_path,
            )
            return current_metadata
        return payload
    return _load_app_attack_metadata(
        app_dir=app_dir,
        behavior_id=behavior_id,
        variant=variant,
        test_mode=test_mode,
    )


def build_attack_metadata_archive_payload(
    metadata: dict[str, Any],
    *,
    app_dir: str | Path,
    behavior_id: str,
    variant: str,
    test_mode: str,
) -> dict[str, Any]:
    """Attach provenance to persisted attack metadata when the app contract is available."""
    payload = dict(metadata)
    try:
        payload["archive_context"] = _attack_metadata_archive_context(
            app_dir=app_dir,
            behavior_id=behavior_id,
            variant=variant,
            test_mode=test_mode,
        )
    except Exception as exc:  # pragma: no cover - defensive fallback for artifact persistence
        logger.warning("Could not attach attack metadata provenance: %s", exc)
    return payload


def _normalized_attack_metadata_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized.pop("archive_context", None)
    return normalized


def _attack_metadata_archive_context(
    *,
    app_dir: str | Path,
    behavior_id: str,
    variant: str,
    test_mode: str,
) -> dict[str, Any]:
    if not str(app_dir or "").strip():
        raise RuntimeError("Missing app directory for attack metadata provenance.")
    app_root = Path(app_dir)
    manifest = load_app_manifest(app_root)
    behavior_contract = (
        load_behavior_contract(app_root, behavior_id) if str(behavior_id or "").strip() else {}
    )
    resolved_variant = (
        variant
        or str(behavior_contract.get("active_variant") or "").strip()
        or ("benign" if test_mode == "benign" else "")
    )
    return {
        "app_id": str(manifest.get("app_id") or "").strip(),
        "behavior_id": str(behavior_id or "").strip(),
        "variant": resolved_variant,
        "test_mode": str(test_mode or "").strip(),
        "shared_seed_version": manifest.get("shared_seed_version"),
        "shared_seed_hash": str(manifest.get("shared_seed_hash") or "").strip(),
        "docs_snapshot_fingerprint": docs_snapshot_fingerprint(
            manifest.get("docs_snapshot") if isinstance(manifest, dict) else {}
        ),
    }


def _empty_attack_metadata(
    *,
    test_mode: str,
    metadata_source: str,
    exposure_known: bool = False,
) -> dict[str, Any]:
    return {
        "test_mode": test_mode,
        "injection_locations": [],
        "placeholders": {},
        "adversarial_content": {},
        "attack_target_arrays": [],
        "surface_summaries": [],
        "metadata_source": metadata_source,
        "exposure_known": exposure_known,
    }


def _load_app_attack_metadata(
    *,
    app_dir: str | Path,
    behavior_id: str,
    variant: str,
    test_mode: str,
) -> dict[str, Any]:
    if not str(app_dir or "").strip():
        raise RuntimeError(
            "Unsupported app-mode attack metadata artifact. Missing app directory."
        )
    app_dir = Path(app_dir)
    manifest = load_app_manifest(app_dir)
    behavior_contract = (
        load_behavior_contract(app_dir, behavior_id) if str(behavior_id or "").strip() else {}
    )
    attack_metadata = (
        (behavior_contract.get("attack_metadata") or {})
        if behavior_contract
        else (manifest.get("attack_metadata") or {})
    )
    display_routes = _normalize_routes(attack_metadata.get("display_routes", []))
    surface_summaries = attack_metadata.get("surface_summaries", []) or []
    exposure_known = bool(attack_metadata.get("exposure_known") and display_routes)
    metadata_source = "behavior_contract" if behavior_contract else "app_manifest"

    if test_mode == "benign":
        return _empty_attack_metadata(
            test_mode=test_mode,
            metadata_source=metadata_source,
            exposure_known=exposure_known,
        )

    selected_variant = variant or str(behavior_contract.get("active_variant") or "")
    if not selected_variant:
        raise RuntimeError(
            "Unsupported app-mode attack metadata artifact. Missing explicit adversarial variant."
        )
    adversarial_content = _load_variant_diff(
        app_dir=app_dir,
        variant=selected_variant,
    )
    if adversarial_content:
        metadata_source = f"{metadata_source}+variant_diff"

    placeholders = _build_placeholder_entries(adversarial_content)
    attack_target_arrays = list(adversarial_content)
    if not attack_target_arrays and attack_metadata.get("injection_target"):
        attack_target_arrays = [attack_metadata["injection_target"]]

    return {
        "test_mode": test_mode,
        "injection_locations": display_routes if exposure_known else [],
        "placeholders": placeholders,
        "adversarial_content": adversarial_content,
        "attack_target_arrays": attack_target_arrays,
        "surface_summaries": surface_summaries,
        "metadata_source": metadata_source,
        "exposure_known": exposure_known,
    }


def _load_variant_diff(
    *,
    app_dir: str | Path,
    variant: str,
) -> dict[str, list[dict[str, Any]]]:
    app_dir = Path(app_dir)
    variant_name = str(variant or "").strip()
    if not variant_name:
        raise RuntimeError(
            "Unsupported app-mode attack metadata artifact. Missing explicit variant selection."
        )
    benign_path = app_dir / "benign" / "data.js"
    try:
        variant_path = resolve_variant_data_path(
            app_dir,
            variant_name,
            field_name="attack metadata variant",
        )
    except ValueError as exc:
        raise RuntimeError(
            "Unsupported app-mode attack metadata artifact. "
            f"Invalid variant selection: {exc}."
        ) from exc
    missing_paths = [
        str(path.relative_to(app_dir))
        for path in (benign_path, variant_path)
        if not path.exists()
    ]
    if missing_paths:
        raise RuntimeError(
            "Unsupported app-mode attack metadata artifact. Missing required variant "
            f"assets: {', '.join(missing_paths)}."
        )

    try:
        benign_data = read_variant_data_js(benign_path)
        variant_data = read_variant_data_js(variant_path)
    except Exception as exc:  # pragma: no cover - defensive logging around parser failures
        raise RuntimeError(
            "Unsupported app-mode attack metadata artifact. "
            f"Could not read benign/data.js and {variant_name}/data.js."
        ) from exc
    return diff_variant_data(benign_data, variant_data)


def _build_placeholder_entries(
    adversarial_content: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    placeholders: dict[str, dict[str, Any]] = {}
    for array_name, items in adversarial_content.items():
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            item_id = item.get("id", f"{array_name}-{index}")
            entry_id = f"{array_name}:{item_id}"
            placeholders[entry_id] = {
                "array": array_name,
                "placement": array_name,
                "element_type": "data_item",
                "adversarial": [json.dumps(item, ensure_ascii=False, default=str)[:800]],
            }
    return placeholders


def _infer_attack_surfaces(
    pages: list[dict[str, Any]],
    adv_spec: dict[str, Any],
) -> tuple[list[str], list[str]]:
    keywords = _attack_keywords(adv_spec)
    if not keywords:
        return [], []

    matches: list[tuple[int, str, str]] = []
    for route, description, page_id in _iter_page_routes(pages):
        haystack = " ".join(
            part.lower()
            for part in (route, description, page_id)
            if isinstance(part, str) and part
        )
        score = sum(1 for keyword in keywords if keyword in haystack)
        if score:
            summary = route if not description else f"{route} - {description}"
            matches.append((score, route, summary))

    matches.sort(key=lambda item: (-item[0], item[1]))
    chosen = matches[:5]
    return [route for _, route, _ in chosen], [summary for _, _, summary in chosen]


def _route_descriptions(pages: list[dict[str, Any]]) -> dict[str, str]:
    descriptions: dict[str, str] = {}
    for route, description, page_id in _iter_page_routes(pages):
        if description:
            descriptions[route] = description
        elif page_id and route not in descriptions:
            descriptions[route] = page_id
    return descriptions


def _iter_page_routes(pages: list[dict[str, Any]]):
    for page in pages:
        page_id = _page_value(page, "id")
        details = _page_value(page, "details") or {}
        if isinstance(details, dict) and details:
            for raw_route, description in details.items():
                route = normalize_route_reference(str(raw_route))
                if route:
                    yield route, str(description or ""), str(page_id or "")
            continue

        subdomains = _page_value(page, "subdomains") or []
        for raw_route in subdomains:
            route = normalize_route_reference(str(raw_route))
            if route:
                yield route, "", str(page_id or "")


def _page_value(page: Any, key: str) -> Any:
    if isinstance(page, dict):
        return page.get(key)
    return getattr(page, key, None)


def _attack_keywords(adv_spec: dict[str, Any]) -> set[str]:
    text_parts = [
        adv_spec.get("injection_target", ""),
        adv_spec.get("injection_description", ""),
        adv_spec.get("benign_analog", ""),
    ]
    tokens = set()
    for part in text_parts:
        for token in re.findall(r"[a-z0-9_]+", str(part).lower()):
            if len(token) < 4:
                continue
            tokens.add(token)
            if token.endswith("s") and len(token) > 4:
                tokens.add(token[:-1])
    return tokens


def _normalize_routes(routes: list[Any]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_route in routes or []:
        route = normalize_route_reference(str(raw_route))
        if route and route not in seen:
            normalized.append(route)
            seen.add(route)
    return normalized
