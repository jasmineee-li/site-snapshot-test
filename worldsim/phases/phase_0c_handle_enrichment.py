"""Phase 0c host-side handle enrichment.

The LLM-driven Phase 0c sandboxes produce ``AGENT_CONTEXT_<site>.json``
from benchmark source files alone. They cannot reliably enumerate live
benchmark state. Phase 2's URL-shape resolver, however, needs to
disambiguate ``/<segment>`` GitLab URLs as user_profile vs group vs
project_namespace, and the only sound source of truth is the live
benchmark instance.

This module runs after Tier 2 of :mod:`worldsim.phases.phase_0_recon`
finalizes ``agent_context``. For GitLab sites with available
seeding-time credentials, it enumerates ``/api/v4/users`` and
``/api/v4/groups`` and merges a top-level ``gitlab`` block into
``agent_context``::

    {
      "gitlab": {
        "user_handles": ["root", "byteblaze", ...],
        "group_handles": ["a11yproject", ...]
      }
    }

The Phase 2 resolver consumes this block via ``agent_context.gitlab`` —
see :mod:`worldsim.phases.phase_2_target_resolver._disambiguate_root_segment`.

Enrichment is best-effort. When credentials are missing or the live
instance is unreachable, the function logs a warning and returns the
input unchanged. The resolver gracefully degrades to ``kind=None`` for
ambiguous root-segment URLs in that case.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests

from worldsim.auth_tokens import acquire_token

logger = logging.getLogger(__name__)

_PER_PAGE = 100
_TIMEOUT_SECONDS = 30
_MAX_PAGES = 50


class HandleEnrichmentError(RuntimeError):
    """Raised when enrichment fails in a way the caller should surface."""


def enrich_gitlab_handles(
    site_url: str,
    auth_config: Mapping[str, Any] | None,
    *,
    runtime_web_host: str | None = None,
    timeout: int = _TIMEOUT_SECONDS,
) -> dict[str, list[str]]:
    """Enumerate gitlab user / top-level group handles via the v4 API.

    Returns a dict suitable for merging into ``agent_context["gitlab"]``::

        {"user_handles": [...sorted, deduped...],
         "group_handles": [...sorted, deduped...]}

    Raises :class:`HandleEnrichmentError` if the call cannot complete.
    Callers in Phase 0c catch and log this so a transient outage doesn't
    fail the entire profile run.
    """
    if not site_url:
        raise HandleEnrichmentError("site_url is required for handle enrichment")
    if not isinstance(auth_config, Mapping) or not auth_config:
        raise HandleEnrichmentError("auth_config is required for handle enrichment")

    errors: list[str] = []
    for candidate in _site_url_candidates(site_url, runtime_web_host):
        try:
            return _enrich_gitlab_handles_once(candidate, auth_config, timeout=timeout)
        except HandleEnrichmentError as exc:
            errors.append(str(exc))
    detail = "; ".join(errors) if errors else "no URL candidates"
    raise HandleEnrichmentError(detail)


def _enrich_gitlab_handles_once(
    site_url: str,
    auth_config: Mapping[str, Any],
    *,
    timeout: int,
) -> dict[str, list[str]]:
    base = site_url.rstrip("/")
    try:
        token = acquire_token(dict(auth_config), base)
    except RuntimeError as exc:
        raise HandleEnrichmentError(f"could not acquire gitlab token: {exc}") from exc

    header_name = str(auth_config.get("header_name") or "PRIVATE-TOKEN")
    headers = {header_name: token, "Accept": "application/json"}

    user_handles = _paginated_collect(
        f"{base}/api/v4/users",
        params={"per_page": _PER_PAGE},
        headers=headers,
        timeout=timeout,
        extract=lambda item: item.get("username") if isinstance(item, dict) else None,
    )
    # GitLab returns subgroups under their parent's full_path. We only
    # disambiguate top-level segments (`/<one_segment>`), so filter out
    # paths containing `/`.
    group_handles = _paginated_collect(
        f"{base}/api/v4/groups",
        params={"per_page": _PER_PAGE, "all_available": "true", "top_level_only": "true"},
        headers=headers,
        timeout=timeout,
        extract=lambda item: (
            item.get("full_path")
            if isinstance(item, dict)
            and isinstance(item.get("full_path"), str)
            and "/" not in item.get("full_path", "")
            else None
        ),
    )

    return {
        "user_handles": sorted(set(user_handles)),
        "group_handles": sorted(set(group_handles)),
    }


def _site_url_candidates(site_url: str, runtime_web_host: str | None) -> list[str]:
    candidates: list[str] = []
    host = str(runtime_web_host or "").strip()
    if host:
        rewritten = _replace_url_host(site_url, host)
        if rewritten != site_url:
            candidates.append(rewritten)
    candidates.append(site_url)
    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = candidate.rstrip("/")
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _replace_url_host(url: str, host: str) -> str:
    parsed = urlsplit(url)
    if not parsed.scheme or not parsed.netloc or not host:
        return url
    hostname = host.strip()
    host_display = f"[{hostname}]" if ":" in hostname and not hostname.startswith("[") else hostname
    netloc = host_display
    if parsed.port is not None:
        netloc = f"{host_display}:{parsed.port}"
    return urlunsplit(parsed._replace(netloc=netloc))


def merge_into_agent_context(
    agent_context: Mapping[str, Any],
    handles: Mapping[str, list[str]],
) -> dict[str, Any]:
    """Return a copy of agent_context with a top-level ``gitlab`` block."""
    merged = dict(agent_context)
    existing = merged.get("gitlab")
    gitlab_block: dict[str, Any] = dict(existing) if isinstance(existing, Mapping) else {}
    gitlab_block["user_handles"] = list(handles.get("user_handles", []))
    gitlab_block["group_handles"] = list(handles.get("group_handles", []))
    merged["gitlab"] = gitlab_block
    return merged


def _paginated_collect(
    url: str,
    *,
    params: dict[str, Any],
    headers: dict[str, str],
    timeout: int,
    extract,
) -> list[str]:
    """Walk paginated GitLab API responses and collect values via ``extract``.

    Stops at empty page or when the page count exceeds ``_MAX_PAGES``
    (a guard against runaway pagination on unbounded endpoints).
    """
    out: list[str] = []
    page = 1
    while page <= _MAX_PAGES:
        merged_params = dict(params)
        merged_params["page"] = page
        try:
            resp = requests.get(url, params=merged_params, headers=headers, timeout=timeout)
        except requests.RequestException as exc:
            raise HandleEnrichmentError(f"GET {url} page={page} failed: {exc}") from exc
        if resp.status_code == 401 or resp.status_code == 403:
            raise HandleEnrichmentError(
                f"GET {url} returned HTTP {resp.status_code}; token lacks scope"
            )
        if resp.status_code != 200:
            raise HandleEnrichmentError(f"GET {url} page={page} returned HTTP {resp.status_code}")
        try:
            payload = resp.json()
        except ValueError as exc:
            raise HandleEnrichmentError(f"GET {url} page={page} returned non-JSON: {exc}") from exc
        if not isinstance(payload, list):
            raise HandleEnrichmentError(
                f"GET {url} page={page} returned non-list JSON ({type(payload).__name__})"
            )
        if not payload:
            break
        for item in payload:
            value = extract(item)
            if isinstance(value, str) and value.strip():
                out.append(value.strip())
        # Trust the next-page header when present; fall back to length check.
        next_page_header = resp.headers.get("X-Next-Page", "").strip()
        if next_page_header:
            try:
                next_page = int(next_page_header)
            except ValueError:
                next_page = 0
            if next_page <= 0:
                break
            page = next_page
        else:
            if len(payload) < params.get("per_page", _PER_PAGE):
                break
            page += 1
    return out


__all__ = [
    "HandleEnrichmentError",
    "enrich_gitlab_handles",
    "merge_into_agent_context",
]
