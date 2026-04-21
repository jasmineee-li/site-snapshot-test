"""Resolve the benign resource each Phase 1 task navigates to.

Option A placement (``docs/handoffs/phase-2-placement-systemic-gap.md``):
adversarial seeds must attach to the **existing** benign-task resource,
not to dangling new projects/forums/issues. This module produces a
``benign_target_resource`` record per benign task so Phase 2a can
constrain ``delivery_channel.method`` to the correct attach surfaces.

Four-layer pipeline, cheap first:

* L1 — regex over ``reward_function.eval[].expected.url`` (gold source).
* L2 — regex over resolved ``start_urls`` (fallback when eval lacks a URL).
* L3 — LLM intent parse + live API probe (added in the L3 commit).
* L4 — dynamic listing resolution for search / dashboard / forum kinds.

This commit ships L1 + L2 and deterministic fallthrough. L3 and L4 are
wired behind an explicit layer flag and raise NotImplementedError until
their own commits land.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Literal
from urllib.parse import urlsplit

from worldsim.placeholders import apply_placeholders

ResourceKind = Literal[
    "gitlab_issue",
    "gitlab_mr",
    "gitlab_search_result",
    "gitlab_dashboard_list",
    "reddit_submission",
    "reddit_forum",
    "reddit_dashboard_list",
]

# Keeps seeds short enough that the attack directive fits one viewport
# once scrolled; PVPO max_coverage is per-viewport so longer bodies
# silently below-the-fold on busy issues.
VIEWPORT_BUDGET_CHARS = 600

# Default N for L4 listing resolution; env override `WORLDSIM_L4_TOP_N`
# is read lazily when L4 runs, not at import.
DEFAULT_L4_TOP_N = 3

# attach_surfaces allow-list keyed by ResourceKind. Each entry names the
# profile surface_id (BENCHMARK_PROFILE_{site}.injection_surface[].id) the
# 2a planner must emit, plus the editor method + required args so the
# validator can pattern-match a delivery_channel against this allow-list.
_ATTACH_SURFACES: dict[ResourceKind, list[dict[str, Any]]] = {
    "gitlab_issue": [
        {
            "surface_id": "note_on_issue",
            "attach_method": "create_issue_note",
            "required_editor_args": ["project_id", "issue_iid", "body"],
        },
    ],
    "gitlab_mr": [
        {
            "surface_id": "note_on_mr",
            "attach_method": "create_mr_note",
            "required_editor_args": ["project_id", "mr_iid", "body"],
        },
    ],
    "gitlab_search_result": [
        {
            "surface_id": "note_on_issue",
            "attach_method": "create_issue_note",
            "required_editor_args": ["project_id", "issue_iid", "body"],
        },
        {
            "surface_id": "note_on_mr",
            "attach_method": "create_mr_note",
            "required_editor_args": ["project_id", "mr_iid", "body"],
        },
    ],
    "gitlab_dashboard_list": [
        {
            "surface_id": "note_on_issue",
            "attach_method": "create_issue_note",
            "required_editor_args": ["project_id", "issue_iid", "body"],
        },
        {
            "surface_id": "note_on_mr",
            "attach_method": "create_mr_note",
            "required_editor_args": ["project_id", "mr_iid", "body"],
        },
    ],
    "reddit_submission": [
        {
            "surface_id": "comment_body_thread",
            "attach_method": "create_comment",
            "required_editor_args": ["submission_id", "body"],
        },
    ],
    "reddit_forum": [
        {
            "surface_id": "submission_body_detail",
            "attach_method": "create_submission",
            "required_editor_args": ["forum_name", "title", "body"],
        },
    ],
    "reddit_dashboard_list": [
        {
            "surface_id": "comment_body_thread",
            "attach_method": "create_comment",
            "required_editor_args": ["submission_id", "body"],
        },
    ],
}

# Regex inventory derived from 333 gitlab+reddit benign tasks
# (logs/phase_1/benign_tasks.json). Patterns are intentionally tolerant of:
#   * leading `^` and trailing `$` / `.*$` regex anchors (NetworkEvent eval
#     entries carry bare URLs but sometimes regex-escaped form)
#   * trailing `.json` suffix (WebArena's UI-form URL with JSON response)
#   * trailing `?query=...` strings
# Apply placeholder expansion BEFORE running these patterns — they assume
# `__GITLAB__` / `__REDDIT__` have already been swapped for concrete origins.
_ISSUE_RE = re.compile(r"/(?P<project_path>(?:[^/?#]+/)+[^/?#]+)/-/issues/(?P<issue_iid>\d+)")
_MR_RE = re.compile(r"/(?P<project_path>(?:[^/?#]+/)+[^/?#]+)/-/merge_requests/(?P<mr_iid>\d+)")
_SEARCH_RE = re.compile(
    r"/search\?(?=[^#]*\bsearch=(?P<q>[^&]+))(?=[^#]*\bscope=(?P<scope>issues|merge_requests))"
)
_DASHBOARD_RE = re.compile(r"/dashboard/(?P<dash>todos|merge_requests|issues)\b")
_PROJECT_ISSUES_API_RE = re.compile(r"/api/v4/projects/(?P<project_id>\d+)/issues\b")
_SUBMISSION_RE = re.compile(r"/f/(?P<forum_name>[^/?#]+)/(?P<submission_id>\d+)(?:/|$|\b)")
_FORUM_RE = re.compile(r"/f/(?P<forum_name>[^/?#\d][^/?#]*)(?:/?(?:\?|$))")
_REDDIT_SUBMIT_RE = re.compile(r"/submit/(?P<forum_name>[^/?#]+)")
_REDDIT_USER_DASH_RE = re.compile(r"/user/(?P<user>[^/?#]+)/(?P<dash>submitted|comments)\b")

# Matching order — most-specific first so /.-/issues/{iid} wins over a
# bare /.-/issues listing (the latter maps to search_result).
_GITLAB_PATTERNS: tuple[tuple[ResourceKind, re.Pattern[str]], ...] = (
    ("gitlab_issue", _ISSUE_RE),
    ("gitlab_mr", _MR_RE),
    ("gitlab_search_result", _SEARCH_RE),
    ("gitlab_dashboard_list", _DASHBOARD_RE),
    ("gitlab_search_result", _PROJECT_ISSUES_API_RE),
)
_REDDIT_PATTERNS: tuple[tuple[ResourceKind, re.Pattern[str]], ...] = (
    ("reddit_submission", _SUBMISSION_RE),
    ("reddit_dashboard_list", _REDDIT_USER_DASH_RE),
    ("reddit_forum", _FORUM_RE),
    ("reddit_forum", _REDDIT_SUBMIT_RE),
)


def _strip_regex_anchors(url: str) -> str:
    """Normalise an eval URL that may carry `^` / `$` / `.*$` regex anchors."""
    if not url:
        return ""
    stripped = url.strip()
    if stripped.startswith("^"):
        stripped = stripped[1:]
    if stripped.endswith(".*$"):
        stripped = stripped[:-3]
    elif stripped.endswith("$"):
        stripped = stripped[:-1]
    if stripped.endswith(".*"):
        stripped = stripped[:-2]
    return stripped


def _strip_json_suffix(url: str) -> str:
    """Drop a trailing `.json` so UI-form URLs match the HTML-page regex."""
    if url.endswith(".json"):
        return url[: -len(".json")]
    return url


def _normalise_url(url: str, placeholders: Mapping[str, str]) -> str | None:
    """Resolve placeholders, strip regex anchors, drop trailing `.json`.

    Returns None if placeholder expansion would leave unresolved
    ``__FOO__`` tokens — caller decides whether that's L3-pending or an
    outright non-match.
    """
    if not url:
        return None
    stripped = _strip_json_suffix(_strip_regex_anchors(url))
    try:
        return apply_placeholders(stripped, dict(placeholders), strict=True)
    except ValueError:
        return None


def _path_and_query(url: str) -> str:
    """Return just the path+query portion of a URL, so hostname components
    can't leak into ``project_path`` captures via greedy matching."""
    if not url:
        return ""
    if "://" not in url:
        # Bare path (eval URLs sometimes arrive without scheme).
        return url if url.startswith("/") else "/" + url
    parts = urlsplit(url)
    path = parts.path or "/"
    if parts.query:
        path = f"{path}?{parts.query}"
    return path


def _match_gitlab(url: str) -> tuple[ResourceKind, dict[str, str]] | None:
    path_and_query = _path_and_query(url)
    for kind, pattern in _GITLAB_PATTERNS:
        match = pattern.search(path_and_query)
        if match:
            anchors = {k: v for k, v in match.groupdict().items() if v}
            if kind == "gitlab_dashboard_list":
                anchors["dashboard"] = anchors.pop("dash", "")
            if kind == "gitlab_search_result" and "q" in anchors:
                anchors["query"] = anchors.pop("q")
            return kind, anchors
    return None


def _match_reddit(url: str) -> tuple[ResourceKind, dict[str, str]] | None:
    path_and_query = _path_and_query(url)
    for kind, pattern in _REDDIT_PATTERNS:
        match = pattern.search(path_and_query)
        if match:
            anchors = {k: v for k, v in match.groupdict().items() if v}
            if kind == "reddit_dashboard_list":
                anchors["dashboard"] = anchors.pop("dash", "")
            return kind, anchors
    return None


def _iter_eval_urls(task: Mapping[str, Any]) -> list[str]:
    """Yield all URLs declared under ``reward_function.eval[].expected``.

    Prefers NetworkEventEvaluator entries first (they name the URL the
    agent must hit); AgentResponseEvaluator entries rarely carry URLs
    but are included as fallback.
    """
    reward = task.get("reward_function") or {}
    evals = reward.get("eval") or []
    ranked: list[tuple[int, str]] = []
    for ev in evals:
        if not isinstance(ev, dict):
            continue
        evaluator = str(ev.get("evaluator") or "")
        priority = 0 if "NetworkEvent" in evaluator else 1
        expected = ev.get("expected") or {}
        raw = expected.get("url") or expected.get("reference_url")
        if raw is None:
            continue
        if isinstance(raw, str):
            candidates = [raw]
        elif isinstance(raw, list):
            candidates = [c for c in raw if isinstance(c, str)]
        else:
            continue
        for candidate in candidates:
            ranked.append((priority, candidate))
    ranked.sort(key=lambda pair: pair[0])
    return [url for _, url in ranked]


def _iter_start_urls(task: Mapping[str, Any]) -> list[str]:
    start = task.get("start_urls") or []
    if isinstance(start, str):
        return [start]
    return [u for u in start if isinstance(u, str)]


def _site_kind_for_task(task: Mapping[str, Any]) -> Literal["gitlab", "reddit"] | None:
    sites = task.get("sites") or []
    for site in sites:
        if not isinstance(site, str):
            continue
        lower = site.strip().lower()
        if lower == "gitlab":
            return "gitlab"
        if lower == "reddit":
            return "reddit"
    primary = str(task.get("site") or "").strip().lower()
    if primary == "gitlab":
        return "gitlab"
    if primary == "reddit":
        return "reddit"
    return None


def _attach_surfaces_for(kind: ResourceKind) -> list[dict[str, Any]]:
    return [dict(surface) for surface in _ATTACH_SURFACES[kind]]


def _benign_user_handle(task: Mapping[str, Any]) -> str | None:
    agent_ctx = task.get("agent_context") or {}
    auth = agent_ctx.get("authentication") or {}
    creds = auth.get("credentials") or {}
    handle = creds.get("username")
    if isinstance(handle, str) and handle.strip():
        return handle.strip()
    return None


def _encounter_requirements(
    kind: ResourceKind, task: Mapping[str, Any], anchors: Mapping[str, str]
) -> dict[str, Any]:
    """Populate encounter_requirements per handoff doc §Encounter derivation."""
    requirements: dict[str, Any] = {"viewport_budget_chars": VIEWPORT_BUDGET_CHARS}
    if kind == "gitlab_dashboard_list":
        handle = _benign_user_handle(task)
        if handle:
            requirements["requires_at_mention"] = handle
        requirements["must_appear_on_list"] = True
    elif kind == "gitlab_search_result":
        query = anchors.get("query")
        scope = anchors.get("scope") or "issues"
        if query:
            requirements["requires_search_index"] = {"query": query, "scope": scope}
    elif kind == "reddit_forum":
        requirements["requires_post_sort_order"] = "recent"
    elif kind == "reddit_dashboard_list":
        handle = _benign_user_handle(task)
        if handle:
            requirements["requires_at_mention"] = handle
    return requirements


def _empty_record(reason: str, pending_layer: Literal["L3", "L4"] | None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "kind": None,
        "anchors": {},
        "start_url_resolved": None,
        "attach_surfaces": [],
        "encounter_requirements": {"viewport_budget_chars": VIEWPORT_BUDGET_CHARS},
        "layer": None,
        "reason": reason,
    }
    if pending_layer is not None:
        record["pending_layer"] = pending_layer
    return record


def derive_benign_target_resource(
    task: Mapping[str, Any],
    placeholders: Mapping[str, str],
    *,
    allow_layers: tuple[Literal["L1", "L2", "L3", "L4"], ...] = ("L1", "L2"),
) -> dict[str, Any]:
    """Resolve the benign target resource for a Phase 1 task.

    Returns a dict matching handoff §Benign-target resource extraction:
    ``{kind, anchors, start_url_resolved, attach_surfaces,
    encounter_requirements, layer, ...}``. When L1+L2 cannot classify
    the task, returns an empty record with ``pending_layer`` set so the
    caller can route to L3/L4 in a later pass.

    ``allow_layers`` gates which layers run in this call. L3 and L4 are
    not implemented in this commit and raise NotImplementedError if
    explicitly requested.
    """
    site_kind = _site_kind_for_task(task)
    if site_kind is None:
        return _empty_record("task is not gitlab or reddit (out of WASP scope)", None)

    if "L3" in allow_layers or "L4" in allow_layers:
        raise NotImplementedError(
            "L3 and L4 resolution are wired in later commits; this build exposes only L1+L2."
        )

    start_urls_raw = _iter_start_urls(task)
    resolved_start: str | None = None
    for url in start_urls_raw:
        resolved = _normalise_url(url, placeholders)
        if resolved:
            resolved_start = resolved
            break

    # L1: parse eval URLs (gold source — NetworkEvent ranked before
    # AgentResponse so the "which URL must the agent hit" signal wins).
    if "L1" in allow_layers:
        for raw in _iter_eval_urls(task):
            resolved = _normalise_url(raw, placeholders)
            if not resolved:
                continue
            hit = _match_gitlab(resolved) if site_kind == "gitlab" else _match_reddit(resolved)
            if hit is None:
                continue
            kind, anchors = hit
            return {
                "kind": kind,
                "anchors": dict(anchors),
                "start_url_resolved": resolved_start,
                "attach_surfaces": _attach_surfaces_for(kind),
                "encounter_requirements": _encounter_requirements(kind, task, anchors),
                "layer": "L1",
            }

    # L2: parse start_urls directly — applies when eval[] lacks a URL
    # (AgentResponseEvaluator-only retrieve tasks).
    if "L2" in allow_layers and resolved_start:
        hit = (
            _match_gitlab(resolved_start)
            if site_kind == "gitlab"
            else _match_reddit(resolved_start)
        )
        if hit is not None:
            kind, anchors = hit
            return {
                "kind": kind,
                "anchors": dict(anchors),
                "start_url_resolved": resolved_start,
                "attach_surfaces": _attach_surfaces_for(kind),
                "encounter_requirements": _encounter_requirements(kind, task, anchors),
                "layer": "L2",
            }

    # Fall-through: bare __GITLAB__ / __REDDIT__ or intent-only task.
    # L3 owns these: LLM intent parse + live API probe. Signal pending
    # so the caller (Phase 2a) routes this task's target derivation to
    # the L3 pass once it lands.
    record = _empty_record(
        "L1+L2 found no concrete resource; intent-only task pending L3",
        pending_layer="L3",
    )
    record["start_url_resolved"] = resolved_start
    return record
