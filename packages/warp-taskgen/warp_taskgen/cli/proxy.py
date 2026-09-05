"""Verification proxy installation for the WARP Taskgen CLI."""

from __future__ import annotations

import argparse
import ipaddress
import logging
from pathlib import Path


def _is_loopback_hostname(hostname: str | None) -> bool:
    if hostname is None:
        return False
    normalized = hostname.strip().lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _install_verification_proxy_from_args(args: argparse.Namespace) -> None:
    """Install the verification-proxy adapter if the args point at a config that has one.

    The proxy config is optional; when ``instances.<name>.json`` declares a
    ``verification_proxy`` block with a non-empty token, every ``requests.Session``
    created in-process rewrites allowlisted site-port URLs to the proxy port and
    attaches the ``X-Worldsim-Token`` header. Ports are derived from the
    instances' ``site_url`` so the adapter works for any benchmark.
    """
    # ``--instances`` is the general flag (Phase 1/4); Phase 2c uses
    # ``--feasibility-instances`` instead. Prefer whichever is set and points
    # at an existing file.
    candidates = [
        getattr(args, "instances", None),
        getattr(args, "feasibility_instances", None),
    ]
    path: Path | None = None
    for candidate in candidates:
        if candidate is None:
            continue
        maybe_path = Path(candidate)
        if maybe_path.exists():
            path = maybe_path
            break
    if path is None:
        return
    import json
    from urllib.parse import urlsplit

    from warp_taskgen.http_proxy import install_proxy

    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        raise RuntimeError(
            f"verification_proxy_invalid: could not parse instances config {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"verification_proxy_invalid: instances config {path} must contain a JSON object"
        )
    proxy_data = payload.get("verification_proxy")
    if proxy_data is None:
        return
    if not isinstance(proxy_data, dict):
        raise RuntimeError(
            f"verification_proxy_invalid: verification_proxy in {path} must be an object"
        )
    token = _resolve_verification_proxy_token(proxy_data, config_path=path)
    if not token.strip():
        return
    scheme = proxy_data.get("scheme", "http")
    if not isinstance(scheme, str) or scheme.strip().lower() not in {"http", "https"}:
        raise RuntimeError(
            f"verification_proxy_invalid: verification_proxy.scheme in {path} must be 'http' or 'https'"
        )
    port_offset = proxy_data.get("port_offset", 10000)
    if not isinstance(port_offset, int) or port_offset < 0:
        raise RuntimeError(
            f"verification_proxy_invalid: verification_proxy.port_offset in {path} must be a non-negative integer"
        )
    instances = payload.get("instances")
    if not isinstance(instances, list):
        raise RuntimeError(
            f"verification_proxy_invalid: instances list missing or invalid in {path}"
        )
    site_ports: set[int] = set()
    non_loopback_site_url_seen = False
    for index, instance in enumerate(instances):
        if not isinstance(instance, dict):
            raise RuntimeError(
                f"verification_proxy_invalid: instances[{index}] in {path} must be an object"
            )
        site_url = instance.get("site_url")
        if not isinstance(site_url, str):
            raise RuntimeError(
                f"verification_proxy_invalid: instances[{index}].site_url in {path} must be a string"
            )
        parsed = urlsplit(site_url)
        if not parsed.scheme or not parsed.hostname or parsed.port is None:
            raise RuntimeError(
                f"verification_proxy_invalid: instances[{index}].site_url {site_url!r} must include scheme, host, and explicit port"
            )
        site_ports.add(parsed.port)
        if not _is_loopback_hostname(parsed.hostname):
            non_loopback_site_url_seen = True
    if not site_ports:
        raise RuntimeError(
            f"verification_proxy_invalid: no proxy-eligible site_url ports found in {path}"
        )
    if not non_loopback_site_url_seen:
        logging.getLogger(__name__).info(
            "verification_proxy ignored for %s because all site_url hosts are loopback",
            path,
        )
        return
    install_proxy(
        token=token.strip(),
        port_offset=port_offset,
        site_ports=site_ports,
    )


def _resolve_verification_proxy_token(proxy_data: dict[str, object], *, config_path: Path) -> str:
    token = proxy_data.get("token", "")
    if token is not None and not isinstance(token, str):
        raise RuntimeError(
            f"verification_proxy_invalid: verification_proxy.token in {config_path} must be a string"
        )
    if isinstance(token, str) and token.strip():
        return token.strip()
    token_env = proxy_data.get("token_env")
    if token_env is not None and not isinstance(token_env, str):
        raise RuntimeError(
            f"verification_proxy_invalid: verification_proxy.token_env in {config_path} must be a string"
        )
    if isinstance(token_env, str) and token_env.strip():
        import os

        env_value = os.environ.get(token_env.strip(), "").strip()
        if env_value:
            return env_value
    token_file = proxy_data.get("token_file")
    if token_file is not None and not isinstance(token_file, str):
        raise RuntimeError(
            f"verification_proxy_invalid: verification_proxy.token_file in {config_path} must be a string"
        )
    if isinstance(token_file, str) and token_file.strip():
        candidate = Path(token_file)
        if not candidate.is_absolute():
            candidate = config_path.parent / candidate
        try:
            return candidate.read_text(encoding="utf-8").strip()
        except OSError:
            return ""
    return ""


__all__ = [
    "_install_verification_proxy_from_args",
    "_is_loopback_hostname",
    "_resolve_verification_proxy_token",
]
