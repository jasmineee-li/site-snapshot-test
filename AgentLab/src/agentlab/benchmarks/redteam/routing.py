"""Playwright route interception for app-mode redteam environments."""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse

from agentlab.benchmarks.redteam.app_artifacts import (
    normalize_domain_binding_mock_prefix,
    normalize_domain_binding_shim_path,
)

logger = logging.getLogger(__name__)

_PRIMARY_SPA_STATIC_PATHS = {
    "/",
    "/css",
    "/index.html",
    "/js",
}
_PRIMARY_SPA_STATIC_PREFIXES = ("/css/", "/js/")
_PRIMARY_SPA_API_METHODS = {
    ("GET", "/api/events"),
    ("GET", "/api/state"),
    ("PUT", "/api/state"),
}
_LOCALHOSTS = {"127.0.0.1", "localhost"}


def setup_route_interception(
    context,
    *,
    domain_bindings: list[dict],
    server_port: int = 0,
) -> None:
    """Install explicit shared-app routing with fail-closed defaults."""
    if server_port is None or server_port == 0:
        raise ValueError("server_port is required")
    if not isinstance(domain_bindings, list):
        raise ValueError("domain_bindings must be a list")

    bindings_by_domain: dict[str, dict] = {}
    for binding in domain_bindings:
        if not isinstance(binding, dict):
            raise ValueError("domain_bindings entries must be objects")
        domain = str(binding.get("domain") or "").strip()
        mode = str(binding.get("mode") or "").strip()
        if not domain:
            raise ValueError("domain binding is missing domain")
        if mode not in {"primary_spa", "shim", "mock", "blocked"}:
            raise ValueError(f"unsupported domain binding mode: {mode}")
        normalized_binding = dict(binding)
        if mode == "mock":
            normalized_binding["mock_prefix"] = normalize_domain_binding_mock_prefix(
                str(binding.get("mock_prefix") or "/mock"),
                domain=domain,
            )
        elif mode == "shim":
            normalized_binding["shim_path"] = normalize_domain_binding_shim_path(
                str(binding.get("shim_path") or ""),
                domain=domain,
            )
        bindings_by_domain[domain] = normalized_binding
        logger.info("Domain binding: %s -> %s", domain, mode)

    context.route(
        "**/*",
        _make_global_handler(
            bindings_by_domain=bindings_by_domain,
            server_port=server_port,
        ),
    )


def _make_global_handler(*, bindings_by_domain: dict[str, dict], server_port: int):
    """Create a catch-all route handler that blocks unlisted domains."""

    def handler(route):
        request = route.request
        parsed = urlparse(request.url)
        domain = parsed.netloc

        if parsed.scheme in {"data", "about"}:
            route.continue_()
            return

        if _is_loopback_request(parsed):
            if _is_allowed_loopback_request(parsed, server_port=server_port):
                route.continue_()
                return
            route.fulfill(
                status=451,
                body=f"Blocked loopback destination: {domain}".encode("utf-8"),
                content_type="text/plain",
            )
            return

        binding = bindings_by_domain.get(domain)
        if binding is None:
            route.fulfill(
                status=451,
                body=f"Blocked unlisted domain: {domain}".encode("utf-8"),
                content_type="text/plain",
            )
            return

        mode = str(binding.get("mode") or "").strip()
        if mode == "blocked":
            route.fulfill(
                status=451,
                body=f"Blocked domain binding: {domain}".encode("utf-8"),
                content_type="text/plain",
            )
            return
        if mode == "primary_spa":
            _proxy_to_loopback(route, server_port=server_port, mode="primary_spa")
            return
        if mode == "mock":
            mock_prefix = str(binding.get("mock_prefix") or "/mock").strip() or "/mock"
            path = parsed.path or "/"
            if not _path_matches_prefix(path, mock_prefix):
                route.fulfill(
                    status=451,
                    body=f"Blocked non-mock path for {domain}: {path}".encode("utf-8"),
                    content_type="text/plain",
                )
                return
            _proxy_to_loopback(
                route,
                server_port=server_port,
                mode="mock",
                mock_prefix=mock_prefix,
            )
            return
        if mode == "shim":
            shim_path = str(binding.get("shim_path") or "").strip()
            _proxy_to_shim(route, server_port=server_port, shim_path=shim_path)
            return

        raise ValueError(f"unsupported domain binding mode: {mode}")

    return handler


def _is_loopback_request(parsed) -> bool:
    hostname = (parsed.hostname or "").strip()
    return hostname in _LOCALHOSTS


def _is_allowed_loopback_request(parsed, *, server_port: int) -> bool:
    hostname = (parsed.hostname or "").strip()
    return hostname in _LOCALHOSTS and parsed.port == server_port


def _proxy_to_loopback(
    route,
    *,
    server_port: int,
    mode: str,
    mock_prefix: str | None = None,
) -> None:
    request = route.request
    parsed = urlparse(request.url)
    local_path = _proxy_loopback_path(
        parsed.path or "/",
        request.method,
        mode=mode,
        mock_prefix=mock_prefix,
    )
    if local_path is None:
        route.fulfill(
            status=451,
            body=f"Blocked path for {mode}: {parsed.path or '/'}".encode("utf-8"),
            content_type="text/plain",
        )
        return
    if parsed.query:
        local_path += f"?{parsed.query}"
    local_url = f"http://127.0.0.1:{server_port}{local_path}"

    try:
        route.continue_(url=local_url)
    except Exception as exc:
        logger.error("Proxy error %s -> %s: %s", request.url, local_url, exc)
        route.fulfill(status=502, body=f"Proxy error: {exc}".encode("utf-8"))


def _proxy_to_shim(route, *, server_port: int, shim_path: str) -> None:
    request = route.request
    parsed = urlparse(request.url)
    base = shim_path.rstrip("/") or "/"
    request_path = parsed.path or "/"
    if request_path in {"", "/"}:
        local_path = base
    else:
        local_path = f"{base}/{request_path.lstrip('/')}"
    if parsed.query:
        local_path += f"?{parsed.query}"
    local_url = f"http://127.0.0.1:{server_port}{local_path}"

    try:
        route.continue_(url=local_url)
    except Exception as exc:
        logger.error("Shim proxy error %s -> %s: %s", request.url, local_url, exc)
        route.fulfill(status=502, body=f"Shim proxy error: {exc}".encode("utf-8"))


def _path_matches_prefix(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(f"{prefix}/")


def _make_proxy_handler(domain: str, server_port: int):
    """Backward-compatible helper retained for focused tests."""

    def handler(route):
        request = route.request
        parsed = urlparse(request.url)
        if parsed.netloc != domain:
            route.fulfill(status=451, body=b"Blocked unexpected domain")
            return
        _proxy_to_loopback(route, server_port=server_port, mode="primary_spa")

    return handler


def _proxy_loopback_path(
    path: str,
    method: str,
    *,
    mode: str,
    mock_prefix: str | None = None,
) -> str | None:
    """Map an external request path to the local app-server path."""
    if mode == "mock":
        if mock_prefix and _path_matches_prefix(path, mock_prefix):
            return path
        return None

    if mode != "primary_spa":
        raise ValueError(f"unsupported loopback proxy mode: {mode}")

    if (method, path) in _PRIMARY_SPA_API_METHODS:
        return path

    if path == "/mock" or path.startswith("/mock/"):
        return None

    if path == "/api" or path.startswith("/api/"):
        return None

    if method not in {"GET", "HEAD"}:
        return None

    if path in _PRIMARY_SPA_STATIC_PATHS:
        return path

    if any(path.startswith(prefix) for prefix in _PRIMARY_SPA_STATIC_PREFIXES):
        return path

    if _looks_like_static_file(path):
        return path

    return "/"


def _looks_like_static_file(path: str) -> bool:
    """Return whether *path* names an explicit file instead of an SPA route."""
    name = Path(path).name
    return "." in name and not name.endswith(".")
