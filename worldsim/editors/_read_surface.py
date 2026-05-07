"""Read-surface URL extraction helpers for editors.

The Phase 4 C1b signal matches editor-emitted URLs against document-type
navigations in the network trace (see docs/handoffs/codex-handoff-c1-read-surface.md
§5, §6). Editors capture authoritative URLs at seed time either from the
platform's API response (preferred — ``editor_api_response``) or by
constructing from platform conventions (``editor_constructed``).

Every editor method that carries a read surface emits BOTH a host-qualified
URL AND a path-only form per §12.8, so matches work across hosts in
cross-benchmark replays.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse


def _dotted_lookup(node: Any, path: str) -> Any:
    """Walk a dotted path (``_links.self``) into a nested mapping/list."""
    current = node
    for segment in path.split("."):
        if isinstance(current, dict):
            current = current.get(segment)
        elif isinstance(current, list):
            try:
                current = current[int(segment)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return current


def collect_platform_urls(response: Any, paths: list[str]) -> list[str]:
    """Pull URLs from ``response`` via dotted-path lookups.

    Returns the deduped union in first-occurrence order. Non-string / empty
    values are skipped; unknown paths return nothing silently.
    """
    seen: set[str] = set()
    out: list[str] = []
    if not isinstance(response, (dict, list)):
        return out
    for path in paths:
        value = _dotted_lookup(response, path)
        if isinstance(value, str) and value.strip() and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def host_and_path_forms(site_url: str, path_or_url: str) -> list[str]:
    """Return both host-qualified and path-only URL forms.

    ``path_or_url`` may be either a full URL (host-qualified) or a path-only
    string starting with ``/``. Returns ``[host_qualified, path_only]`` with
    duplicates removed. Either form may be absent if construction is
    impossible (e.g. missing site_url).
    """
    site = str(site_url or "").rstrip("/")
    raw = str(path_or_url or "").strip()
    if not raw:
        return []

    if raw.startswith("http://") or raw.startswith("https://"):
        parsed = urlparse(raw)
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        if parsed.fragment:
            path = f"{path}#{parsed.fragment}"
        host_form = raw
        path_form = path
    elif raw.startswith("/"):
        path_form = raw
        host_form = f"{site}{raw}" if site else ""
    else:
        # Relative path — rare for editor URLs; treat as path-only.
        path_form = f"/{raw}"
        host_form = f"{site}{path_form}" if site else ""

    forms: list[str] = []
    seen: set[str] = set()
    for form in (host_form, path_form):
        form = form.strip()
        if form and form not in seen:
            seen.add(form)
            forms.append(form)
    return forms


def normalize_surface_urls(urls: list[str]) -> list[str]:
    """Dedupe a list of URLs preserving first-occurrence order."""
    seen: set[str] = set()
    out: list[str] = []
    for url in urls or []:
        value = str(url or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out
