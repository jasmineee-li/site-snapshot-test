"""Shared PVPO CDP endpoint validation helpers."""

from __future__ import annotations

import ipaddress
import os
from urllib.parse import urlparse

REMOTE_PVPO_CDP_OVERRIDE_ENV = "WORLDSIM_ALLOW_REMOTE_PVPO_CDP_URL"


def _host_is_loopback(host: str | None) -> bool:
    if not host:
        return False
    normalized = host.strip().lower()
    if normalized in {"localhost", "::1", "[::1]"}:
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def validate_pvpo_cdp_url(
    raw_url: str | None,
    *,
    field_name: str = "pvpo_cdp_url",
    allow_empty: bool = True,
) -> str | None:
    """Validate and normalize one PVPO CDP endpoint URL."""
    if raw_url is None:
        if allow_empty:
            return None
        raise ValueError(f"{field_name} must be a non-empty string")

    url = str(raw_url).strip()
    if not url:
        if allow_empty:
            return None
        raise ValueError(f"{field_name} must be a non-empty string")

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https", "ws", "wss"}:
        raise ValueError(f"{field_name} must use http/https/ws/wss, got {parsed.scheme!r}")
    if not parsed.hostname:
        raise ValueError(f"{field_name} must include a hostname")

    if _host_is_loopback(parsed.hostname):
        return url

    if os.environ.get(REMOTE_PVPO_CDP_OVERRIDE_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return url

    raise ValueError(
        f"{field_name} must point to a loopback CDP endpoint by default; got remote host "
        f"{parsed.hostname!r}. Set {REMOTE_PVPO_CDP_OVERRIDE_ENV}=1 only if you intentionally "
        "trust that remote browser."
    )


def canonical_pvpo_endpoint_identity(raw_url: str) -> str:
    """Return a stable identity string for endpoint dedupe."""
    parsed = urlparse(raw_url)
    if not parsed.hostname:
        raise ValueError(f"pvpo_cdp_url must include a hostname: {raw_url!r}")
    host = parsed.hostname.strip().lower()
    if _host_is_loopback(host):
        host = "127.0.0.1"
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme in {"https", "wss"} else 80
    return f"{host}:{port}"
