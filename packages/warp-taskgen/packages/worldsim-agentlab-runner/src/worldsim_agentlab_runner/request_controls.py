from __future__ import annotations

from typing import Any
from urllib.parse import urlparse, urlunparse

_URL_VALUE_HEADER_NAMES = {"referer", "referrer", "location", "content-location"}
_FORBIDDEN_CONTINUE_HEADER_NAMES = {"host", "content-length"}


def install_request_controls(context: Any, request: dict[str, Any]) -> dict[str, Any]:
    """Install origin rewrites and scoped auth before task navigation.

    Browser Use applies both controls through request interception instead of
    global Playwright headers. AgentLab has to do the same because Phase 4
    replicas can emit canonical absolute links, and auth headers must stay on
    the intended site origin.
    """

    rewrites = _normalize_origin_rewrites(request.get("url_origin_rewrites"))
    scoped_auth = _normalize_scoped_auth(request.get("scoped_auth"))
    installed = bool(rewrites or scoped_auth)
    telemetry = {
        "request_controls_installed": installed,
        "url_origin_rewrites": rewrites,
        "scoped_auth_origin": scoped_auth.get("origin") if scoped_auth else None,
        "scoped_auth_header_names": sorted((scoped_auth.get("headers") or {}).keys())
        if scoped_auth
        else [],
        "rewrite_hits": 0,
        "scoped_auth_hits": 0,
    }
    if not installed:
        return telemetry

    def handler(route: Any, req: Any) -> None:
        url = str(getattr(req, "url", "") or "")
        rewritten = _rewrite_url_origin(url, rewrites)
        headers = dict(getattr(req, "headers", {}) or {})
        headers, headers_changed = _headers_for_rewritten_request(
            headers,
            original_url=url,
            rewritten_url=rewritten,
        )
        auth_origin = scoped_auth.get("origin") if scoped_auth else ""
        auth_applied = False
        if auth_origin and _origin_from_url(rewritten or url) == auth_origin:
            headers.update(scoped_auth.get("headers") or {})
            telemetry["scoped_auth_hits"] += 1
            auth_applied = True
        headers = _strip_forbidden_continue_headers(headers)
        if rewritten != url:
            telemetry["rewrite_hits"] += 1
            route.continue_(url=rewritten, headers=headers)
            return
        if headers_changed or auth_applied:
            route.continue_(headers=headers)
            return
        route.continue_()

    context.route("**/*", handler)
    return telemetry


def _normalize_scoped_auth(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    origin = _origin_from_url(str(value.get("origin") or ""))
    headers = value.get("headers")
    if not origin or not isinstance(headers, dict) or not headers:
        return {}
    return {"origin": origin, "headers": {str(k): str(v) for k, v in headers.items()}}


def _normalize_origin_rewrites(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    rewrites: dict[str, str] = {}
    for source, target in value.items():
        source_origin = _origin_from_url(str(source))
        target_origin = _origin_from_url(str(target))
        if (
            source_origin
            and target_origin
            and source_origin != target_origin
            and urlparse(source_origin).scheme == urlparse(target_origin).scheme
        ):
            rewrites[source_origin] = target_origin
    return rewrites


def _rewrite_url_origin(url: str, rewrites: dict[str, str]) -> str:
    origin = _origin_from_url(url)
    target = rewrites.get(origin)
    if not target:
        return url
    parsed = urlparse(url)
    target_parsed = urlparse(target)
    return urlunparse(
        (
            target_parsed.scheme,
            target_parsed.netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )


def _headers_for_rewritten_request(
    headers: dict[str, str],
    *,
    original_url: str,
    rewritten_url: str,
) -> tuple[dict[str, str], bool]:
    if not headers:
        return {}, False
    old_origin = _origin_from_url(original_url)
    new_origin = _origin_from_url(rewritten_url)
    if not old_origin or not new_origin or old_origin == new_origin:
        return dict(headers), False
    out = dict(headers)
    changed = False
    for name, value in list(out.items()):
        lower = name.lower()
        if lower == "origin" and _origin_from_url(value) == old_origin:
            out[name] = new_origin
            changed = True
        elif lower in _URL_VALUE_HEADER_NAMES and _origin_from_url(value) == old_origin:
            rewritten_value = _rewrite_url_origin(value, {old_origin: new_origin})
            if rewritten_value != value:
                out[name] = rewritten_value
                changed = True
    return out, changed


def _strip_forbidden_continue_headers(headers: dict[str, str]) -> dict[str, str]:
    return {
        name: value
        for name, value in headers.items()
        if str(name).lower() not in _FORBIDDEN_CONTINUE_HEADER_NAMES
    }


def _origin_from_url(raw_url: str) -> str:
    parsed = urlparse(str(raw_url or ""))
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return ""
    default_port = 80 if parsed.scheme == "http" else 443
    if parsed.port and parsed.port != default_port:
        return f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"
    return f"{parsed.scheme}://{parsed.hostname}"
