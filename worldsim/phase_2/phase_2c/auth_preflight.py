"""Phase 2c auth and storage-state preflight helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from worldsim.agent_auth import (
    cookie_domain_matches_host,
    playwright_storage_state,
    playwright_storage_state_payload,
    read_storage_state_payload,
    resolve_agent_auth,
    resolve_agent_auth_headers,
    resolve_storage_state_path,
    storage_state_cookie_hosts,
    storage_state_origin_hosts,
    storage_state_preflight_error_for_payload,
    storage_state_recorded_hosts,
)


def _preflight_request_context_options(
    instance: dict[str, Any],
    *,
    benchmark_root: Path | None = None,
) -> tuple[dict[str, Any], str | None]:
    """Build Playwright APIRequestContext auth options for source-data preflight.

    Returns ``({}, reason)`` when declared auth is unusable. The caller then
    skips source-data quarantine for that instance instead of probing
    anonymously and falsely classifying private pages as source-data drops.
    """
    agent_auth = instance.get("agent_auth")
    resolved = resolve_agent_auth(
        agent_auth if isinstance(agent_auth, dict) else None,
        site_name=str(instance.get("site_name") or ""),
        site_url=str(instance.get("site_url") or ""),
        benchmark_root=benchmark_root,
        storage_state_override=instance.get("storage_state_path"),
    )
    if resolved.unusable_reason is not None:
        return {}, resolved.unusable_reason
    return dict(resolved.api_request_context_kwargs), None


def _agent_auth_type(instance: dict[str, Any]) -> str:
    agent_auth = instance.get("agent_auth")
    if not isinstance(agent_auth, dict):
        return ""
    return str(agent_auth.get("type") or "").strip()


def _resolve_agent_auth_headers(agent_auth: dict[str, Any]) -> dict[str, str]:
    return resolve_agent_auth_headers(agent_auth)


def _storage_state_preflight_error(path: str, instance: dict[str, Any]) -> str | None:
    payload, error = _read_storage_state_payload_for_preflight(path)
    if error is not None:
        return error
    return _storage_state_preflight_error_for_payload(Path(path), payload, instance)


def _read_storage_state_payload_for_preflight(
    path: str,
) -> tuple[dict[str, Any], str | None]:
    return read_storage_state_payload(path)


def _storage_state_preflight_error_for_payload(
    path_obj: Path,
    payload: dict[str, Any],
    instance: dict[str, Any],
) -> str | None:
    return storage_state_preflight_error_for_payload(
        path_obj,
        payload,
        str(instance.get("site_url") or ""),
    )


def _playwright_storage_state_for_preflight(
    path: str,
) -> tuple[str | dict[str, Any], str | None]:
    """Return a Playwright-compatible storage state for preflight.

    Phase 0d artifacts may come from non-Playwright browser APIs whose
    cookie ``sameSite`` values use CDP names such as ``no_restriction``.
    Normalize known equivalents in memory so auth remains usable. Unknown
    shapes keep the existing auth-unusable path, which makes preflight skip
    this instance instead of probing private surfaces anonymously.
    """
    return playwright_storage_state(path)


def _playwright_storage_state_payload_for_preflight(
    path_obj: Path,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    return playwright_storage_state_payload(path_obj, payload)


def _storage_state_recorded_hosts(payload: dict[str, Any]) -> set[str]:
    return storage_state_recorded_hosts(payload)


def _storage_state_cookie_hosts(payload: dict[str, Any]) -> set[str]:
    return storage_state_cookie_hosts(payload)


def _storage_state_origin_hosts(payload: dict[str, Any]) -> set[str]:
    return storage_state_origin_hosts(payload)


def _cookie_domain_matches_host(domain: str, host: str) -> bool:
    return cookie_domain_matches_host(domain, host)


def _resolve_benign_storage_state_path(instance: dict[str, Any]) -> str | None:
    """Return the Phase-0d-bootstrapped storage_state.json path for this site.

    Under Option A (alpha) identity the seed writer and the reachability
    probe both act as the benign user, so threading those cookies into
    Playwright lets the probe reach private projects + authed-only
    pages. Falls back to ``None`` when no artifact is present (public
    content still works in an anonymous context).
    """
    agent_auth = instance.get("agent_auth")
    if not isinstance(agent_auth, dict) or agent_auth.get("type") != "storage_state":
        return None
    path = resolve_storage_state_path(
        agent_auth,
        site_name=str(instance.get("site_name") or ""),
        storage_state_override=instance.get("storage_state_path"),
        benchmark_root=Path(str(instance["benchmark_root"]))
        if instance.get("benchmark_root")
        else None,
    )
    return str(path) if path is not None else None


def _resolve_benign_browser_context_auth(
    instance: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    """Return browser context auth kwargs for Phase 2c browser probes.

    No configured ``agent_auth`` preserves the legacy anonymous probe path.
    Declared-but-unusable auth returns an explicit reason so callers fail
    closed instead of silently probing as an anonymous visitor.
    """
    agent_auth = instance.get("agent_auth")
    if not isinstance(agent_auth, dict) or not str(agent_auth.get("type") or "").strip():
        return {}, None
    benchmark_root = (
        Path(str(instance["benchmark_root"])) if instance.get("benchmark_root") else None
    )
    resolved = resolve_agent_auth(
        agent_auth,
        site_name=str(instance.get("site_name") or ""),
        site_url=str(instance.get("site_url") or ""),
        benchmark_root=benchmark_root,
        storage_state_override=instance.get("storage_state_path"),
    )
    if resolved.unusable_reason is not None:
        return {}, resolved.unusable_reason
    return dict(resolved.browser_context_kwargs), None


def _auth_probe_failure_kind(reason: str) -> str:
    lowered = reason.lower()
    if "missing" in lowered or "not found" in lowered or "no usable artifact" in lowered:
        return "auth_missing"
    return "auth_unusable"


__all__ = [
    "_agent_auth_type",
    "_auth_probe_failure_kind",
    "_cookie_domain_matches_host",
    "_playwright_storage_state_for_preflight",
    "_playwright_storage_state_payload_for_preflight",
    "_preflight_request_context_options",
    "_read_storage_state_payload_for_preflight",
    "_resolve_agent_auth_headers",
    "_resolve_benign_browser_context_auth",
    "_resolve_benign_storage_state_path",
    "_storage_state_cookie_hosts",
    "_storage_state_origin_hosts",
    "_storage_state_preflight_error",
    "_storage_state_preflight_error_for_payload",
    "_storage_state_recorded_hosts",
]
