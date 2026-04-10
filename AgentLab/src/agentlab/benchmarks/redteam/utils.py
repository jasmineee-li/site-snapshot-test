"""Shared pure-utility functions used by multiple redteam modules."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers (de-duplicated from controller, app_pipeline, eval_harness,
# app_artifacts)
# ---------------------------------------------------------------------------


def utc_timestamp() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def write_json(path: str | Path, payload: Any) -> None:
    """Atomically write *payload* as pretty-printed JSON, creating parents."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: str | Path, content: str) -> None:
    """Write *content* as UTF-8 text, creating parent directories."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def sha256_bytes(payload: bytes) -> str:
    """Return the hex SHA-256 digest of raw *payload* bytes."""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of the file at *path*."""
    return sha256_bytes(path.read_bytes())


def normalize_hash_fragment(fragment: str) -> str:
    """Normalize SPA fragments so equivalent inbox/settings routes compare equal."""
    if not fragment:
        return ""

    fragment = fragment[1:] if fragment.startswith("#") else fragment
    if not fragment:
        return "#"
    if fragment.startswith("/"):
        return f"#{fragment}"
    return f"#/{fragment.lstrip('/')}"


def normalize_route_reference(route_or_url: str) -> str:
    """Canonicalize a path/query/fragment reference from a URL or relative route."""
    if not route_or_url:
        return ""

    parsed = urlparse(route_or_url)
    if parsed.scheme and parsed.scheme not in {"http", "https"}:
        return ""

    path = parsed.path or "/"
    if not path.startswith("/"):
        path = f"/{path}"
    query = f"?{parsed.query}" if parsed.query else ""
    fragment = normalize_hash_fragment(parsed.fragment)

    return f"{path}{query}{fragment}"


def resolve_external_app_start_url(start_page: str, pages: list[Any]) -> str:
    """Resolve an app ``start_page`` onto a declared external page contract.

    Returns the external URL the browser should navigate to. Raises ``ValueError``
    when ``start_page`` cannot be reached through the declared ``pages`` contract.
    """

    declared_routes = list(_declared_page_routes(pages))
    if not declared_routes:
        raise ValueError("pages must include at least one routable page with base_site_url and subdomains")

    start_page = str(start_page or "").strip()
    if not start_page:
        raise ValueError("start_page must be a non-empty string")

    parsed = urlparse(start_page)
    if parsed.scheme and parsed.scheme not in {"http", "https"}:
        raise ValueError(f"unsupported start_page URL scheme: {parsed.scheme}")

    if parsed.scheme or parsed.netloc:
        start_domain = parsed.netloc
        normalized_start_route = normalize_route_reference(start_page)
        declared_domains = {route["domain"] for route in declared_routes}
        if start_domain not in declared_domains:
            declared_domains_rendered = ", ".join(sorted(declared_domains))
            raise ValueError(
                f"start_page domain {start_domain!r} is not declared by pages "
                f"({declared_domains_rendered})"
            )
        if not any(
            route["domain"] == start_domain and route["normalized_route"] == normalized_start_route
            for route in declared_routes
        ):
            raise ValueError(
                f"start_page route {normalized_start_route!r} is not declared by pages for domain {start_domain!r}"
            )
        return start_page

    normalized_start_route = _normalize_relative_start_route(start_page, declared_routes)
    matching_route = next(
        (
            route
            for route in declared_routes
            if route["normalized_route"] == normalized_start_route
        ),
        None,
    )
    if matching_route is None:
        raise ValueError(
            f"start_page route {normalized_start_route!r} is not declared by any page subdomain"
        )
    return f"{matching_route['base_site_url']}{normalized_start_route}"


def app_start_page_contract_error(start_page: str, pages: list[Any]) -> str | None:
    """Return a manifest-contract error for ``start_page``/``pages``, else ``None``."""

    try:
        resolve_external_app_start_url(start_page, pages)
    except ValueError as exc:
        return str(exc)
    return None


def strip_html(text: str) -> str:
    """Remove HTML tags from text and normalise whitespace."""
    clean = re.sub(r"<[^>]+>", " ", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def _page_field(page: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(page, dict):
        return page.get(field_name, default)
    return getattr(page, field_name, default)


def _declared_page_routes(pages: list[Any]):
    for page in pages or []:
        base_site_url = str(_page_field(page, "base_site_url", "") or "").strip().rstrip("/")
        parsed_base = urlparse(base_site_url)
        if parsed_base.scheme not in {"http", "https"} or not parsed_base.netloc:
            continue
        for subdomain in _page_field(page, "subdomains", []) or []:
            normalized_route = normalize_route_reference(str(subdomain or ""))
            if not normalized_route:
                continue
            yield {
                "base_site_url": base_site_url,
                "domain": parsed_base.netloc,
                "normalized_route": normalized_route,
            }


def _normalize_relative_start_route(start_page: str, declared_routes: list[dict[str, str]]) -> str:
    parsed = urlparse(start_page)
    if start_page.startswith("#") or (not parsed.path and parsed.fragment):
        fragment = normalize_hash_fragment(parsed.fragment or start_page)
        prefix = _default_declared_route_prefix(declared_routes)
        return f"{prefix}{fragment}"
    return normalize_route_reference(start_page)


def _default_declared_route_prefix(declared_routes: list[dict[str, str]]) -> str:
    default_route = declared_routes[0]["normalized_route"]
    parsed = urlparse(default_route)
    path = parsed.path or "/"
    if not path.startswith("/"):
        path = f"/{path}"
    return path
