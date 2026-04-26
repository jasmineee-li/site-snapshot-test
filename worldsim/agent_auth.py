"""Shared agent-auth resolution for Phase 2c and Browser Use runtime."""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

_PLAYWRIGHT_COOKIE_SAMESITE_DEFAULT = "Lax"
_PLAYWRIGHT_COOKIE_SAMESITE_ALIASES: dict[str, str] = {
    "lax": "Lax",
    "none": "None",
    "no_restriction": "None",
    "no-restriction": "None",
    "strict": "Strict",
    "": _PLAYWRIGHT_COOKIE_SAMESITE_DEFAULT,
    "unspecified": _PLAYWRIGHT_COOKIE_SAMESITE_DEFAULT,
}
_PHASE_0D_SITE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class ResolvedAgentAuth:
    auth_type: str
    browser_context_kwargs: dict[str, Any]
    api_request_context_kwargs: dict[str, Any]
    unusable_reason: str | None = None
    storage_state_path: Path | None = None

    @property
    def usable(self) -> bool:
        return self.unusable_reason is None


def safe_phase_0d_site_name(site_name: object) -> tuple[str | None, str | None]:
    """Return a safe Phase 0d path segment for a benchmark site name.

    Site names come from config/task metadata, so reject path separators and
    traversal tokens before constructing ``logs/phase_0d/<site>`` paths.
    """
    site = str(site_name or "").strip()
    if not site:
        return None, "site_name must be non-empty"
    if _PHASE_0D_SITE_RE.fullmatch(site) is None:
        return None, f"site_name {site!r} is not safe for Phase 0d storage_state lookup"
    return site, None


def _http_auth_origin(site_url: str) -> tuple[str | None, str | None]:
    parsed = urlsplit(str(site_url or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None, f"site_url {site_url!r} is not a valid HTTP origin for http_basic auth"
    return f"{parsed.scheme}://{parsed.netloc}", None


def resolve_agent_auth(
    agent_auth: Mapping[str, Any] | None,
    *,
    site_name: str,
    site_url: str,
    task: Mapping[str, Any] | None = None,
    benchmark_root: Path | None = None,
    storage_state_override: str | Path | None = None,
) -> ResolvedAgentAuth:
    """Resolve instance ``agent_auth`` to browser/API context kwargs.

    ``unusable_reason`` is returned instead of raising for declared auth that
    cannot be safely represented. Source-data preflight callers must then skip
    quarantine rather than retrying anonymously.
    """
    if not isinstance(agent_auth, Mapping):
        return ResolvedAgentAuth("none", {}, {})

    auth_type = str(agent_auth.get("type") or "").strip()
    if auth_type in {"", "none"}:
        return ResolvedAgentAuth(auth_type or "none", {}, {})
    if auth_type == "unknown":
        return ResolvedAgentAuth(
            auth_type,
            {},
            {},
            "agent_auth type 'unknown' cannot be used for authenticated preflight",
        )

    if auth_type == "storage_state":
        path = resolve_storage_state_path(
            agent_auth,
            site_name=site_name,
            storage_state_override=storage_state_override,
            task=task,
            benchmark_root=benchmark_root,
        )
        if path is None:
            return ResolvedAgentAuth(
                auth_type,
                {},
                {},
                "storage_state auth declared but no usable artifact was found",
            )
        payload, error = read_storage_state_payload(path)
        if error is not None:
            return ResolvedAgentAuth(auth_type, {}, {}, error, path)
        error = storage_state_preflight_error_for_payload(path, payload, site_url)
        if error is not None:
            return ResolvedAgentAuth(auth_type, {}, {}, error, path)
        storage_state, error = playwright_storage_state_payload(path, payload)
        if error is not None:
            return ResolvedAgentAuth(auth_type, {}, {}, error, path)
        return ResolvedAgentAuth(
            auth_type,
            {"storage_state": storage_state},
            {"storage_state": storage_state},
            None,
            path,
        )

    if auth_type == "http_headers":
        try:
            headers = resolve_agent_auth_headers(agent_auth)
        except RuntimeError as exc:
            return ResolvedAgentAuth(auth_type, {}, {}, str(exc))
        return ResolvedAgentAuth(
            auth_type,
            {"extra_http_headers": headers},
            {"extra_http_headers": headers},
        )

    if auth_type == "http_basic":
        block = agent_auth.get("http_basic")
        if not isinstance(block, Mapping):
            return ResolvedAgentAuth(
                auth_type,
                {},
                {},
                "http_basic auth declared without http_basic block",
            )
        username = block.get("username")
        password = block.get("password")
        if not isinstance(username, str) or not username or not isinstance(password, str):
            return ResolvedAgentAuth(
                auth_type,
                {},
                {},
                "http_basic auth declared without username/password",
            )
        origin, error = _http_auth_origin(site_url)
        if error is not None:
            return ResolvedAgentAuth(auth_type, {}, {}, error)
        credentials = {"username": username, "password": password, "origin": origin}
        return ResolvedAgentAuth(
            auth_type,
            {"http_credentials": credentials},
            {"http_credentials": credentials},
        )

    return ResolvedAgentAuth(
        auth_type,
        {},
        {},
        f"agent_auth type {auth_type!r} is not supported",
    )


def resolve_storage_state_path(
    agent_auth: Mapping[str, Any],
    *,
    site_name: str,
    storage_state_override: str | Path | None = None,
    task: Mapping[str, Any] | None = None,
    benchmark_root: Path | None = None,
) -> Path | None:
    if storage_state_override is not None and str(storage_state_override).strip():
        path = Path(storage_state_override)
        if path.exists():
            error = _validate_storage_state_override_path(
                path,
                benchmark_root=benchmark_root,
                site_name=site_name,
            )
            if error is None:
                return path
            logger.warning("agent auth: %s", error)
            # Ignore an unsafe runtime override, but still allow the declared
            # agent_auth.storage_state.path below to go through the legacy
            # config path. This prevents an override from broadening access
            # while preserving existing validated configs.
        logger.warning(
            "agent auth: storage_state override %s not found; checking agent_auth/fallback paths",
            path,
        )

    storage_state = agent_auth.get("storage_state")
    if isinstance(storage_state, Mapping):
        nested = storage_state.get("path")
        if isinstance(nested, str) and nested.strip():
            path, error = _resolve_declared_storage_state_path(
                nested.strip(),
                benchmark_root=benchmark_root,
                site_name=site_name,
            )
            if error is not None:
                logger.warning("agent auth: %s", error)
                return None
            if path is not None and path.exists():
                return path
            logger.warning(
                "agent auth: declared storage_state %s not found; checking fallback path",
                path or nested,
            )

    fallback_site = site_name
    if not fallback_site and isinstance(task, Mapping):
        raw_site = task.get("site")
        fallback_site = raw_site if isinstance(raw_site, str) else ""
    fallback_site, site_error = safe_phase_0d_site_name(fallback_site)
    if site_error is not None:
        logger.warning("agent auth: %s", site_error)
        return None
    candidate = (
        Path(os.environ.get("WORLDSIM_STATE_DIR") or "logs")
        / "phase_0d"
        / fallback_site
        / "storage_state.json"
    )
    return candidate if candidate.exists() else None


def _validate_storage_state_override_path(
    path: Path,
    *,
    benchmark_root: Path | None,
    site_name: str,
) -> str | None:
    """Constrain instance-level storage_state overrides to expected artifact roots.

    Absolute declared ``agent_auth.storage_state.path`` values are legacy config
    and remain accepted, but the mutable runtime override is produced by Phase
    0d refresh or host config. When a benchmark root is available, do not let it
    point at arbitrary local JSON outside the benchmark or Phase 0d state dir.
    """
    fallback_site, site_error = safe_phase_0d_site_name(site_name)
    if site_error is not None and str(site_name or "").strip():
        return site_error
    if benchmark_root is None:
        return "storage_state override requires a benchmark root for containment checks"
    resolved = path.resolve()
    allowed_roots = [Path(benchmark_root).resolve()]
    if fallback_site:
        allowed_roots.append(
            (
                Path(os.environ.get("WORLDSIM_STATE_DIR") or "logs") / "phase_0d" / fallback_site
            ).resolve()
        )
    for root in allowed_roots:
        try:
            resolved.relative_to(root)
            return None
        except ValueError:
            continue
    roots = ", ".join(str(root) for root in allowed_roots)
    return f"storage_state override path {path} is outside allowed roots: {roots}"


def _phase_0d_site_root(site_name: str) -> tuple[Path | None, str | None]:
    safe_site, site_error = safe_phase_0d_site_name(site_name)
    if site_error is not None:
        return None, site_error
    return (
        Path(os.environ.get("WORLDSIM_STATE_DIR") or "logs") / "phase_0d" / safe_site
    ).resolve(), None


def _resolve_declared_storage_state_path(
    raw_path: str,
    *,
    benchmark_root: Path | None,
    site_name: str,
) -> tuple[Path | None, str | None]:
    """Resolve a declared ``agent_auth.storage_state.path``.

    Relative paths are anchored against the WorldSim state dir (where Phase 0d
    writes), not ``benchmark_root``. Run-output artifacts must resolve against
    a single output root so Phase 0d (writer) and Phase 4 (reader) cannot
    diverge into different files for the same config string. Absolute paths
    keep their dual-root containment (state dir or benchmark root) so legacy
    configs that point inside the benchmark tree still validate.
    """
    path = Path(raw_path)
    state_root, site_error = _phase_0d_site_root(site_name)
    if site_error is not None:
        return None, site_error
    state_dir_root = Path(os.environ.get("WORLDSIM_STATE_DIR") or "logs").expanduser().resolve()
    allowed_absolute_roots = [
        root
        for root in (
            Path(benchmark_root).resolve() if benchmark_root else None,
            state_root,
            state_dir_root,
        )
        if root is not None
    ]
    if path.is_absolute():
        resolved = path.resolve()
        for root in allowed_absolute_roots:
            try:
                resolved.relative_to(root)
                return path, None
            except ValueError:
                continue
        roots = ", ".join(str(root) for root in allowed_absolute_roots)
        return None, f"declared storage_state path {raw_path} is outside allowed roots: {roots}"
    candidate = state_dir_root / path
    try:
        candidate.resolve().relative_to(state_dir_root)
    except ValueError:
        return None, (
            f"declared storage_state path {raw_path} escapes WorldSim state dir {state_dir_root}"
        )
    return candidate, None


def resolve_agent_auth_headers(agent_auth: Mapping[str, Any]) -> dict[str, str]:
    block = agent_auth.get("http_headers")
    if not isinstance(block, Mapping):
        block = agent_auth
    headers = block.get("headers")
    if not isinstance(headers, Mapping) or not headers:
        raise RuntimeError("http_headers auth declared without a non-empty headers map")

    authentication = agent_auth.get("authentication")
    credentials = authentication.get("credentials") if isinstance(authentication, Mapping) else {}
    if not isinstance(credentials, Mapping):
        credentials = {}
    username = credentials.get("username")
    password = credentials.get("password")

    resolved: dict[str, str] = {}
    for key, value in headers.items():
        if not isinstance(key, str) or not key.strip() or not isinstance(value, str):
            raise RuntimeError("http_headers entries must be string keys and string values")
        text = value
        needs_username = "${credentials.username}" in text
        needs_password = "${credentials.password}" in text
        if needs_username and not isinstance(username, str):
            raise RuntimeError(
                "http_headers references ${credentials.username} without credentials"
            )
        if needs_password and not isinstance(password, str):
            raise RuntimeError(
                "http_headers references ${credentials.password} without credentials"
            )
        if needs_username:
            text = text.replace("${credentials.username}", username)
        if needs_password:
            text = text.replace("${credentials.password}", password)
        resolved[key] = text
    return resolved


def read_storage_state_payload(path: str | Path) -> tuple[dict[str, Any], str | None]:
    path_obj = Path(path)
    try:
        payload = json.loads(path_obj.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"storage_state {path_obj} is not readable JSON: {exc}"
    if not isinstance(payload, dict):
        return {}, f"storage_state {path_obj} does not contain a JSON object"
    return payload, None


def storage_state_preflight_error_for_payload(
    path: str | Path,
    payload: dict[str, Any],
    site_url: str,
) -> str | None:
    path_obj = Path(path)
    recorded_hosts = storage_state_recorded_hosts(payload)
    if not recorded_hosts:
        return f"storage_state {path_obj} has no recorded cookie/origin hosts"

    live_host = urlsplit(str(site_url or "")).hostname
    if not live_host:
        return f"instance site_url {site_url!r} has no host"

    matching_hosts = [
        host for host in recorded_hosts if cookie_domain_matches_host(host, live_host)
    ]
    foreign_hosts = [
        host for host in recorded_hosts if not cookie_domain_matches_host(host, live_host)
    ]
    if matching_hosts and foreign_hosts:
        return (
            f"storage_state {path_obj} mixes live host {live_host!r} records "
            f"{sorted(matching_hosts)} with foreign recorded hosts {sorted(foreign_hosts)}; "
            "re-run Phase 0d against the current generated instances file"
        )

    cookie_hosts = storage_state_cookie_hosts(payload)
    origin_hosts = storage_state_origin_hosts(payload)
    mismatched_cookie_hosts = [
        host for host in cookie_hosts if not cookie_domain_matches_host(host, live_host)
    ]
    if mismatched_cookie_hosts:
        return (
            f"storage_state {path_obj} cookie domains are host-bound to "
            f"{sorted(mismatched_cookie_hosts)} and do not match live host {live_host!r}"
        )
    mismatched_origin_hosts = [
        host for host in origin_hosts if not cookie_domain_matches_host(host, live_host)
    ]
    if mismatched_origin_hosts:
        return (
            f"storage_state {path_obj} origins include hosts outside live host "
            f"{live_host!r}: {sorted(mismatched_origin_hosts)}"
        )
    if cookie_hosts:
        return None
    if not any(cookie_domain_matches_host(recorded, live_host) for recorded in origin_hosts):
        return (
            f"storage_state {path_obj} is host-bound to {sorted(recorded_hosts)} "
            f"and does not match live host {live_host!r}"
        )
    return None


def playwright_storage_state(path: str | Path) -> tuple[str | dict[str, Any], str | None]:
    path_obj = Path(path)
    payload, error = read_storage_state_payload(path_obj)
    if error is not None:
        return str(path_obj), error
    return playwright_storage_state_payload(path_obj, payload)


def playwright_storage_state_payload(
    path: str | Path,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    path_obj = Path(path)
    cookies = payload.get("cookies")
    if not isinstance(cookies, list):
        return payload, None

    normalized_cookies: list[Any] = []
    changed = False
    for index, cookie in enumerate(cookies):
        if not isinstance(cookie, dict):
            return {}, f"storage_state {path_obj} cookie[{index}] is not an object"
        normalized_cookie = dict(cookie)
        raw_same_site = normalized_cookie.get("sameSite")
        if "sameSite" not in normalized_cookie or raw_same_site is None:
            normalized_cookie["sameSite"] = _PLAYWRIGHT_COOKIE_SAMESITE_DEFAULT
            changed = True
            normalized_cookies.append(normalized_cookie)
            continue
        if not isinstance(raw_same_site, str):
            return {}, f"storage_state {path_obj} cookie[{index}] has non-string sameSite"
        key = raw_same_site.strip().lower()
        if key not in _PLAYWRIGHT_COOKIE_SAMESITE_ALIASES:
            return (
                {},
                f"storage_state {path_obj} cookie[{index}] has unsupported sameSite {raw_same_site!r}",
            )
        normalized_same_site = _PLAYWRIGHT_COOKIE_SAMESITE_ALIASES[key]
        if normalized_same_site != raw_same_site:
            normalized_cookie["sameSite"] = normalized_same_site
            changed = True
        normalized_cookies.append(normalized_cookie)

    if not changed:
        return payload, None
    normalized_payload = dict(payload)
    normalized_payload["cookies"] = normalized_cookies
    return normalized_payload, None


def storage_state_recorded_hosts(payload: dict[str, Any]) -> set[str]:
    return storage_state_cookie_hosts(payload) | storage_state_origin_hosts(payload)


def storage_state_cookie_hosts(payload: dict[str, Any]) -> set[str]:
    hosts: set[str] = set()
    cookies = payload.get("cookies")
    if isinstance(cookies, list):
        for cookie in cookies:
            if not isinstance(cookie, dict):
                continue
            domain = cookie.get("domain")
            if isinstance(domain, str) and domain.strip():
                hosts.add(domain.strip().lower().strip("."))
    return hosts


def storage_state_origin_hosts(payload: dict[str, Any]) -> set[str]:
    hosts: set[str] = set()
    origins = payload.get("origins")
    if isinstance(origins, list):
        for origin in origins:
            if not isinstance(origin, dict):
                continue
            origin_url = origin.get("origin")
            if not isinstance(origin_url, str):
                continue
            host = urlsplit(origin_url).hostname
            if host:
                hosts.add(host.lower().strip("."))
    return hosts


def cookie_domain_matches_host(domain: str, host: str) -> bool:
    normalized_domain = domain.strip().lower().strip(".")
    normalized_host = host.strip().lower().strip(".")
    if not normalized_domain or not normalized_host:
        return False
    if normalized_domain == normalized_host:
        return True
    if ":" in normalized_domain or ":" in normalized_host:
        return False
    return normalized_host.endswith(f".{normalized_domain}")


__all__ = [
    "ResolvedAgentAuth",
    "cookie_domain_matches_host",
    "playwright_storage_state",
    "playwright_storage_state_payload",
    "read_storage_state_payload",
    "resolve_agent_auth",
    "resolve_agent_auth_headers",
    "resolve_storage_state_path",
    "safe_phase_0d_site_name",
    "storage_state_cookie_hosts",
    "storage_state_origin_hosts",
    "storage_state_preflight_error_for_payload",
    "storage_state_recorded_hosts",
]
