"""Phase 2 target resolution http probes."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote as urlquote

from worldsim.phase_2.target_resolution.constants import (
    _POSTMILL_COMMENT_ID_RE,
    _POSTMILL_STRONG_COUNT_RE,
    DEFAULT_REDDIT_MAX_EXISTING_COMMENTS,
)
from worldsim.phase_2.target_resolution.encounter import _benign_user_handle
from worldsim.phase_2.target_resolution.reconstruction import (
    _anchors_from_gitlab_item,
    _anchors_from_reddit_submission,
)
from worldsim.phase_2.target_resolution.types import RedditCommentCountFn
from worldsim.phase_2.target_resolution.url_matching import (
    _canonicalize_project_path,
    _empty_record,
)

logger = logging.getLogger(__name__)


async def _probe_http_json(
    instance: Mapping[str, Any],
    path: str,
    *,
    params: Mapping[str, Any] | None = None,
    timeout: float = 15.0,
) -> Any:
    """GET ``path`` against ``instance.site_url`` as the benign user, JSON-decoded.

    Auth is assembled via :func:`worldsim.auth_tokens.build_auth_headers`
    but forced onto the benign-user auth lane. Phase 2a target resolution
    must not inherit privileged ``api_auth`` if a future host config adds
    it for other phases. This helper is read-only and sync-wrapped in
    ``asyncio.to_thread``.
    """
    # Lazy import: requests + seeding are heavy and L1/L2 tests don't need them.
    import requests

    from worldsim.auth_tokens import build_auth_headers

    site_url = str(instance.get("site_url") or "").rstrip("/")
    if not site_url:
        raise RuntimeError("instance has no site_url; cannot run L3 probe")
    url = f"{site_url}{path}"
    headers = build_auth_headers(_benign_probe_instance(instance), {}, mechanism="api")

    def _send() -> Any:
        response = requests.get(url, headers=headers, params=dict(params or {}), timeout=timeout)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        try:
            return response.json()
        except ValueError:
            return None

    return await asyncio.to_thread(_send)


def _benign_probe_instance(instance: Mapping[str, Any]) -> dict[str, Any]:
    """Return an instance view pinned to benign-user auth.

    L3/L4 are observational read probes over resources the benign agent can
    encounter. They must never escalate to a privileged ``api_auth`` lane.
    If a host config provides only ``api_auth`` and no benign ``auth``,
    fail closed so the caller excludes the task instead of resolving
    anchors against data the benign user cannot see.
    """
    probe_instance = dict(instance)
    auth = probe_instance.get("auth")
    api_auth = probe_instance.get("api_auth")
    if isinstance(api_auth, dict) and not isinstance(auth, dict):
        raise RuntimeError("instance has api_auth but no benign auth for L3/L4 probe")
    probe_instance.pop("api_auth", None)
    return probe_instance


def _normalise_sort_direction(raw: Any) -> str:
    """Map an LLM-emitted sort hint onto GitLab's ``asc|desc`` contract."""
    value = str(raw or "").strip().lower()
    if not value:
        return "desc"
    if value in ("asc", "ascending", "ascend"):
        return "asc"
    if value in ("desc", "descending", "descend"):
        return "desc"
    if value.endswith("_asc") or value.startswith("asc_"):
        return "asc"
    if value.endswith("_desc") or value.startswith("desc_"):
        return "desc"
    return "desc"


async def _default_probe(
    probe_query: Mapping[str, Any],
    task: Mapping[str, Any],
    instance: Mapping[str, Any],
    placeholders: Mapping[str, str],
) -> dict[str, Any] | None:
    """Execute a probe_query against the live benchmark instance.

    Returns an ``anchors`` dict on success, ``None`` on empty result or
    on a transport error (the caller decides whether to exclude the
    task or fall back).
    """
    api = str(probe_query.get("api") or "")
    limit = int(probe_query.get("limit") or 1)
    username = str(probe_query.get("username") or "") or _benign_user_handle(task) or ""
    sort_dir = _normalise_sort_direction(probe_query.get("sort"))

    if api == "none":
        return None

    if api in {"list_user_todos", "list_user_merge_requests", "list_user_issues"}:
        dashboard = {
            "list_user_todos": "todos",
            "list_user_merge_requests": "merge_requests",
            "list_user_issues": "issues",
        }[api]
        return {"dashboard": dashboard}

    if api in {"list_user_submitted", "list_user_comments"}:
        dashboard = {
            "list_user_submitted": "submitted",
            "list_user_comments": "comments",
        }[api]
        return {"dashboard": dashboard}

    if api in {"search_user_issues", "search_user_mrs"}:
        endpoint = "/api/v4/issues" if api == "search_user_issues" else "/api/v4/merge_requests"
        params: dict[str, Any] = {
            "scope": "created_by_me",
            "order_by": "updated_at",
            "sort": sort_dir,
            "per_page": limit,
        }
        query = str(probe_query.get("query") or "").strip()
        if query:
            params["search"] = query
        if username:
            params["author_username"] = username
        data = await _probe_http_json(instance, endpoint, params=params)
        if not isinstance(data, list) or not data:
            return None
        top = data[0]
        return _anchors_from_gitlab_item(top, kind_hint=api)

    if api == "search_project_issues" or api == "list_project_issues_recent":
        project_id = probe_query.get("project_id")
        project_path = str(probe_query.get("project_path") or "")
        if not project_id and project_path:
            project_id = await _resolve_project_id(instance, project_path)
        if not project_id:
            return None
        params = {
            "order_by": "updated_at",
            "sort": sort_dir,
            "per_page": limit,
        }
        query = str(probe_query.get("query") or "").strip()
        if query:
            params["search"] = query
        data = await _probe_http_json(
            instance, f"/api/v4/projects/{project_id}/issues", params=params
        )
        if not isinstance(data, list) or not data:
            return None
        return _anchors_from_gitlab_item(data[0], kind_hint="gitlab_issue")

    if api == "search_project_mrs" or api == "list_project_mrs_recent":
        project_id = probe_query.get("project_id")
        project_path = str(probe_query.get("project_path") or "")
        if not project_id and project_path:
            project_id = await _resolve_project_id(instance, project_path)
        if not project_id:
            return None
        params = {
            "order_by": "updated_at",
            "sort": sort_dir,
            "per_page": limit,
        }
        query = str(probe_query.get("query") or "").strip()
        if query:
            params["search"] = query
        data = await _probe_http_json(
            instance, f"/api/v4/projects/{project_id}/merge_requests", params=params
        )
        if not isinstance(data, list) or not data:
            return None
        return _anchors_from_gitlab_item(data[0], kind_hint="gitlab_mr")

    if api == "find_project_by_path":
        project_path = str(probe_query.get("project_path") or "").strip()
        if not project_path:
            return None
        canonical = _canonicalize_project_path(project_path)
        if not canonical:
            return None
        project_id = await _resolve_project_id(instance, canonical)
        if project_id is None:
            return None
        return {
            "project_id": str(project_id),
            "project_path": canonical,
        }

    if api == "find_submission_by_title":
        forum_name = str(probe_query.get("forum_name") or "")
        query = str(probe_query.get("query") or "").strip()
        if not forum_name or not query:
            return None
        # Postmill's search endpoint differs by instance; fall back to
        # listing the forum and grepping locally when we can.
        submissions = await _fetch_forum_submissions(instance, forum_name, limit=25)
        if not submissions:
            return None
        lowered = query.lower()
        for entry in submissions:
            title = str(entry.get("title") or "").lower()
            if lowered in title:
                if not await _reddit_submission_within_comment_budget(
                    instance, forum_name, str(entry.get("id") or "")
                ):
                    return None
                return _anchors_from_reddit_submission(entry, forum_name)
        return None

    if api == "list_forum_submissions_recent":
        forum_name = str(probe_query.get("forum_name") or "")
        if not forum_name:
            return None
        # Pull a wider window than the requested top-N so regeneration can
        # skip busy threads whose appended comments are predictably below fold.
        submissions = await _fetch_forum_submissions(instance, forum_name, limit=max(limit, 25))
        if not submissions:
            return None
        for entry in submissions:
            submission_id = str(entry.get("id") or entry.get("submission_id") or "")
            if await _reddit_submission_within_comment_budget(instance, forum_name, submission_id):
                return _anchors_from_reddit_submission(entry, forum_name)
        return None

    logger.warning("L3 probe_query.api %r not implemented; excluding task", api)
    return None


async def _resolve_project_id(instance: Mapping[str, Any], project_path: str) -> int | None:
    canonical = _canonicalize_project_path(project_path)
    if not canonical:
        return None
    data = await _probe_http_json(instance, f"/api/v4/projects/{urlquote(canonical, safe='')}")
    if isinstance(data, dict):
        pid = data.get("id")
        if isinstance(pid, int):
            return pid
    return None


async def _fetch_forum_submissions(
    instance: Mapping[str, Any], forum_name: str, *, limit: int = 3
) -> list[dict[str, Any]] | None:
    # Postmill doesn't expose a documented JSON API for forum listings;
    # the `/f/{forum}.json` URL 404s on the WebArena image. Fall back to
    # the GET /f/{forum} HTML and parse submission anchors. This is
    # best-effort — callers treat None as "probe failed, exclude".
    import re as _re

    import requests

    from worldsim.auth_tokens import build_auth_headers

    site_url = str(instance.get("site_url") or "").rstrip("/")
    if not site_url:
        return None
    url = f"{site_url}/f/{urlquote(forum_name, safe='')}"
    headers = build_auth_headers(_benign_probe_instance(instance), {}, mechanism="form")

    def _send() -> str | None:
        try:
            response = requests.get(url, headers=headers, timeout=15)
        except requests.RequestException:
            return None
        if response.status_code >= 400:
            return None
        return response.text

    html = await asyncio.to_thread(_send)
    if not html:
        return None
    # Submission links on Postmill are of the form /f/{forum}/{id}/slug.
    pattern = _re.compile(
        rf"/f/{_re.escape(forum_name)}/(?P<id>\d+)/[^\"'>]*[\"'][^>]*>(?P<title>[^<]+)"
    )
    seen: set[str] = set()
    results: list[dict[str, Any]] = []
    for match in pattern.finditer(html):
        sid = match.group("id")
        if sid in seen:
            continue
        seen.add(sid)
        results.append({"id": sid, "title": match.group("title").strip()})
        if len(results) >= limit:
            break
    return results or None


def _reddit_max_existing_comments_default() -> int:
    raw = os.environ.get("WORLDSIM_REDDIT_MAX_EXISTING_COMMENTS", "").strip()
    if raw.isdigit() and int(raw) >= 0:
        return int(raw)
    return DEFAULT_REDDIT_MAX_EXISTING_COMMENTS


def _postmill_submission_comment_count_from_html(html_text: str) -> int:
    """Derive an existing top-level comment count from Postmill submission HTML.

    Prefer counting ``id=comment_{id}`` / ``id=comment-{id}`` nodes; if that is
    zero (markup change), fall back to the submission header count.
    """
    by_id = len(_POSTMILL_COMMENT_ID_RE.findall(html_text))
    if by_id > 0:
        return by_id
    m = _POSTMILL_STRONG_COUNT_RE.search(html_text)
    if m:
        return int(m.group(1))
    return 0


async def _reddit_submission_existing_comment_count(
    instance: Mapping[str, Any], forum_name: str, submission_id: str
) -> int | None:
    """Count existing comments before using a Reddit thread as an IPI target.

    Postmill renders top-level comments by score in the default submission
    view. Any existing visible comment can displace a newly seeded carrier from
    the first painted comment region, which is a target-selection failure
    rather than a Phase 4 encounter signal.
    """
    import requests

    from worldsim.auth_tokens import build_auth_headers

    if not forum_name or not submission_id:
        return None
    site_url = str(instance.get("site_url") or "").rstrip("/")
    if not site_url:
        return None
    url = f"{site_url}/f/{urlquote(forum_name, safe='')}/{urlquote(str(submission_id), safe='')}"
    headers = build_auth_headers(_benign_probe_instance(instance), {}, mechanism="form")

    def _send() -> str | None:
        try:
            response = requests.get(url, headers=headers, timeout=15)
        except requests.RequestException:
            return None
        if response.status_code >= 400:
            return None
        return response.text

    html_text = await asyncio.to_thread(_send)
    if not html_text:
        return None
    return _postmill_submission_comment_count_from_html(html_text)


async def _reddit_submission_within_comment_budget(
    instance: Mapping[str, Any],
    forum_name: str,
    submission_id: str,
    *,
    comment_count_fn: RedditCommentCountFn | None = None,
) -> bool:
    count_fn = comment_count_fn or _reddit_submission_existing_comment_count
    count = await count_fn(instance, forum_name, submission_id)
    return count is not None and count <= _reddit_max_existing_comments_default()


async def _admission_filter_resolved_record(
    record: dict[str, Any],
    instance: Mapping[str, Any],
    *,
    reddit_comment_count_fn: RedditCommentCountFn | None = None,
) -> dict[str, Any]:
    if record.get("kind") != "reddit_submission":
        return record
    anchors = record.get("anchors") if isinstance(record.get("anchors"), Mapping) else {}
    forum_name = str(anchors.get("forum_name") or "")
    submission_id = str(anchors.get("submission_id") or "")
    count_fn = reddit_comment_count_fn or _reddit_submission_existing_comment_count
    comment_count = await count_fn(instance, forum_name, submission_id)
    limit = _reddit_max_existing_comments_default()
    if comment_count is None:
        filtered = _empty_record(
            "reddit_submission_visibility_probe_failed",
            pending_layer=None,
        )
    elif comment_count > limit:
        filtered = _empty_record(
            "reddit_submission_too_many_existing_comments_for_visible_seed",
            pending_layer=None,
        )
    else:
        requirements = dict(record.get("encounter_requirements") or {})
        requirements["max_existing_comments_for_comment_seed"] = limit
        requirements["existing_comment_count"] = comment_count
        record["encounter_requirements"] = requirements
        return record
    filtered["start_url_resolved"] = record.get("start_url_resolved")
    filtered["anchors"] = dict(anchors)
    filtered["layer"] = record.get("layer")
    filtered["reddit_existing_comment_count"] = comment_count
    filtered["reddit_existing_comment_limit"] = limit
    return filtered
