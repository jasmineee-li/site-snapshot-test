"""Playwright route interception for app-mode redteam environments."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from agentlab.benchmarks.redteam.app_artifacts import (
    normalize_domain_binding_mock_prefix,
    normalize_domain_binding_shim_path,
    normalize_mock_endpoint_path,
)
from agentlab.benchmarks.redteam.utils import (
    normalize_route_reference,
    resolve_external_app_start_url,
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


@dataclass(frozen=True)
class NavigationContainmentPolicy:
    allowed_network_targets: frozenset[tuple[str, str]]
    browser_routes_by_origin: dict[str, tuple[str, ...]]
    fallback_routes_by_origin: dict[str, str]


def setup_route_interception(
    context,
    *,
    domain_bindings: list[dict],
    pages: list[Any] | None = None,
    allowed_routes: list[str] | None = None,
    mock_endpoints: list[str] | None = None,
    server_port: int = 0,
) -> NavigationContainmentPolicy | None:
    """Install explicit shared-app routing with fail-closed defaults."""
    if server_port is None or server_port == 0:
        raise ValueError("server_port is required")
    if not isinstance(domain_bindings, list):
        raise ValueError("domain_bindings must be a list")

    bindings_by_domain: dict[str, dict] = {}
    containment_policy = _resolve_navigation_containment_policy(
        allowed_routes=allowed_routes or [],
        pages=pages or [],
    )
    allowed_mock_endpoints = _normalize_mock_endpoints(mock_endpoints or [])
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
            allowed_navigation_targets=(
                None if containment_policy is None else containment_policy.allowed_network_targets
            ),
            allowed_mock_endpoints=allowed_mock_endpoints,
            server_port=server_port,
        ),
    )
    return containment_policy


def _make_global_handler(
    *,
    bindings_by_domain: dict[str, dict],
    allowed_navigation_targets: frozenset[tuple[str, str]] | None,
    allowed_mock_endpoints: set[str],
    server_port: int,
):
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
            _proxy_to_loopback(
                route,
                server_port=server_port,
                mode="primary_spa",
                domain=domain,
                allowed_navigation_targets=allowed_navigation_targets,
                allowed_mock_endpoints=allowed_mock_endpoints,
            )
            return
        if mode == "mock":
            mock_prefix = str(binding.get("mock_prefix") or "/mock").strip() or "/mock"
            _proxy_to_loopback(
                route,
                server_port=server_port,
                mode="mock",
                domain=domain,
                mock_prefix=mock_prefix,
                allowed_mock_endpoints=allowed_mock_endpoints,
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
    domain: str,
    mock_prefix: str | None = None,
    allowed_navigation_targets: frozenset[tuple[str, str]] | None = None,
    allowed_mock_endpoints: set[str] | None = None,
) -> None:
    request = route.request
    parsed = urlparse(request.url)
    local_path = _proxy_loopback_path(
        parsed.path or "/",
        request.method,
        query=parsed.query,
        mode=mode,
        domain=domain,
        mock_prefix=mock_prefix,
        allowed_navigation_targets=allowed_navigation_targets,
        allowed_mock_endpoints=allowed_mock_endpoints or set(),
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
    request_path = _normalize_request_path(parsed.path or "/")
    if request_path is None:
        route.fulfill(
            status=451,
            body=f"Blocked path for shim: {parsed.path or '/'}".encode("utf-8"),
            content_type="text/plain",
        )
        return
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
        _proxy_to_loopback(
            route,
            server_port=server_port,
            mode="primary_spa",
            domain=domain,
        )

    return handler


def _proxy_loopback_path(
    path: str,
    method: str,
    *,
    query: str = "",
    mode: str,
    domain: str,
    mock_prefix: str | None = None,
    allowed_navigation_targets: frozenset[tuple[str, str]] | None = None,
    allowed_mock_endpoints: set[str] | None = None,
) -> str | None:
    """Map an external request path to the local app-server path."""
    normalized_path = _normalize_request_path(path)
    if normalized_path is None:
        return None

    if mode == "mock":
        if (
            mock_prefix
            and _path_matches_prefix(normalized_path, mock_prefix)
            and normalized_path in (allowed_mock_endpoints or set())
        ):
            return normalized_path
        return None

    if mode != "primary_spa":
        raise ValueError(f"unsupported loopback proxy mode: {mode}")

    if (method, normalized_path) in _PRIMARY_SPA_API_METHODS:
        return normalized_path

    if _path_matches_prefix(normalized_path, "/mock"):
        if normalized_path in (allowed_mock_endpoints or set()):
            return normalized_path
        return None

    if normalized_path == "/api" or normalized_path.startswith("/api/"):
        return None

    if method not in {"GET", "HEAD"}:
        return None

    if normalized_path in _PRIMARY_SPA_STATIC_PATHS:
        return normalized_path

    if any(normalized_path.startswith(prefix) for prefix in _PRIMARY_SPA_STATIC_PREFIXES):
        return normalized_path

    if _looks_like_static_file(normalized_path):
        return normalized_path

    if allowed_navigation_targets is not None:
        navigation_target = normalized_path
        if query:
            navigation_target += f"?{query}"
        if (domain, navigation_target) not in allowed_navigation_targets:
            return None

    return "/"


def _looks_like_static_file(path: str) -> bool:
    """Return whether *path* names an explicit file instead of an SPA route."""
    name = Path(path).name
    return "." in name and not name.endswith(".")


def _normalize_mock_endpoints(mock_endpoints: list[str]) -> set[str]:
    return {
        normalize_mock_endpoint_path(str(endpoint or ""))
        for endpoint in mock_endpoints
    }


def _resolved_origin(parsed) -> str:
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}"


def _browser_route_guard_redirect_url(
    url: str,
    *,
    policy: NavigationContainmentPolicy | None,
) -> str | None:
    if policy is None:
        return None
    parsed = urlparse(url)
    origin = _resolved_origin(parsed)
    allowed_routes = policy.browser_routes_by_origin.get(origin) or ()
    if not allowed_routes:
        return None

    normalized_route = normalize_route_reference(url)
    if normalized_route in allowed_routes:
        return None

    safe_route = policy.fallback_routes_by_origin.get(origin) or allowed_routes[0]
    if not safe_route:
        return None
    return f"{origin}{safe_route}"


def install_browser_route_guard(
    page,
    *,
    policy: NavigationContainmentPolicy | None,
) -> None:
    """Install a browser-side guard for same-document SPA route changes."""
    if policy is None or not policy.browser_routes_by_origin:
        return

    script = f"""
(() => {{
  const allowed = {json.dumps(policy.browser_routes_by_origin, ensure_ascii=False)};
  const fallback = {json.dumps(policy.fallback_routes_by_origin, ensure_ascii=False)};

  const normalizeHash = (hash) => {{
    if (!hash) return "";
    let fragment = hash.startsWith("#") ? hash.slice(1) : hash;
    if (!fragment) return "#";
    if (fragment.startsWith("/")) return `#${{fragment}}`;
    return `#/${{fragment.replace(/^\\/+/, "")}}`;
  }};

  const normalizeRoute = (href) => {{
    const url = new URL(href, window.location.origin);
    let path = url.pathname || "/";
    if (!path.startsWith("/")) path = `/${{path}}`;
    return `${{path}}${{url.search || ""}}${{normalizeHash(url.hash || "")}}`;
  }};

  const enforce = () => {{
    const origin = window.location.origin;
    const allowedRoutes = allowed[origin];
    if (!Array.isArray(allowedRoutes) || allowedRoutes.length === 0) return true;

    const currentRoute = normalizeRoute(window.location.href);
    if (allowedRoutes.includes(currentRoute)) return true;

    const safeRoute = fallback[origin] || allowedRoutes[0];
    if (!safeRoute) return false;

    const safeUrl = `${{origin}}${{safeRoute}}`;
    if (window.location.href !== safeUrl) {{
      window.location.replace(safeUrl);
    }}
    return false;
  }};

  const wrapHistory = (name) => {{
    const original = history[name];
    if (typeof original !== "function") return;
    history[name] = function(...args) {{
      const result = original.apply(this, args);
      enforce();
      return result;
    }};
  }};

  wrapHistory("pushState");
  wrapHistory("replaceState");
  window.addEventListener("hashchange", enforce, true);
  window.addEventListener("popstate", enforce, true);
  enforce();
}})();
""".strip()
    page.add_init_script(script)


def _resolve_navigation_containment_policy(
    *,
    allowed_routes: list[str],
    pages: list[Any],
) -> NavigationContainmentPolicy | None:
    normalized_routes = [str(route).strip() for route in allowed_routes if str(route).strip()]
    if not normalized_routes:
        return None
    if not pages:
        raise ValueError("pages are required when allowed_routes are provided")

    network_targets: set[tuple[str, str]] = set()
    browser_targets_by_origin: dict[str, set[str]] = {}
    fallback_routes_by_origin: dict[str, str] = {}
    for route in normalized_routes:
        resolved_url = resolve_external_app_start_url(route, pages)
        parsed = urlparse(resolved_url)
        normalized_path = _normalize_request_path(parsed.path or "/")
        if normalized_path is None:
            raise ValueError(f"allowed route {route!r} resolves to an unsafe path")
        target = normalized_path
        if parsed.query:
            target += f"?{parsed.query}"
        network_targets.add((parsed.netloc, target))

        origin = _resolved_origin(parsed)
        normalized_route = normalize_route_reference(resolved_url)
        browser_targets_by_origin.setdefault(origin, set()).add(normalized_route)
        fallback_routes_by_origin.setdefault(origin, normalized_route)

    return NavigationContainmentPolicy(
        allowed_network_targets=frozenset(network_targets),
        browser_routes_by_origin={
            origin: tuple(sorted(routes))
            for origin, routes in browser_targets_by_origin.items()
        },
        fallback_routes_by_origin=dict(fallback_routes_by_origin),
    )


def _normalize_request_path(path: str) -> str | None:
    normalized = str(path or "/").strip() or "/"
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"
    decoded = unquote(normalized)
    if not decoded.startswith("/") or "\\" in decoded:
        return None
    if any(segment in {".", ".."} for segment in decoded.split("/")):
        return None
    return decoded
