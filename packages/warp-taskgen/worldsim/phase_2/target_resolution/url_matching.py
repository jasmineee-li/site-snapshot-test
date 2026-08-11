"""Phase 2 target resolution url matching."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from worldsim.phase_2.target_resolution.constants import (
    _REGEX_META_RE,
    VIEWPORT_BUDGET_CHARS,
)
from worldsim.phase_2.target_resolution.types import ResourceKind
from worldsim.sites.catalog import (
    TargetingContext,
)
from worldsim.sites.catalog import (
    _iter_eval_urls as _catalog_iter_eval_urls,
)
from worldsim.sites.catalog import (
    _iter_start_urls as _catalog_iter_start_urls,
)
from worldsim.sites.catalog import (
    _normalise_url as _catalog_normalise_url,
)
from worldsim.sites.catalog import (
    _path_and_query as _catalog_path_and_query,
)
from worldsim.sites.catalog import (
    _site_kind_for_task as _catalog_site_kind_for_task,
)
from worldsim.sites.catalog import (
    _strip_json_suffix as _catalog_strip_json_suffix,
)
from worldsim.sites.catalog import (
    _strip_regex_anchors as _catalog_strip_regex_anchors,
)
from worldsim.sites.catalog import (
    _url_with_expected_query_params as _catalog_url_with_expected_query_params,
)
from worldsim.sites.gitlab import GitLabSite
from worldsim.sites.reddit import RedditSite

_GITLAB_SITE = GitLabSite()
_REDDIT_SITE = RedditSite()


def _strip_regex_anchors(url: str) -> str:
    """Compatibility delegate for the generic Site Targeting normalizer."""

    return _catalog_strip_regex_anchors(url)


def _literalize_regex_value(value: str | None) -> str | None:
    """Convert a WebArena-style regex query value into a single literal.

    WebArena URL-match evaluators encode query values as regex alternations
    (e.g. ``^(opened|)$`` for "either ``opened`` or absent"). Forwarding
    those literally to the live API yields HTTP 400 because GitLab/Postmill
    expect concrete enum values. This helper picks a usable literal:

    * Plain literals (no regex metacharacters) pass through unchanged.
    * ``^(a|b|...)$`` (with optional whitespace) → first non-empty literal
      branch (mirrors the URL-match semantics: any branch satisfies the
      evaluator, so we pick one).
    * Anything still containing regex syntax → ``None`` (caller drops it).

    Returning ``None`` for residual regex is deliberate: dashboard listing
    APIs return 400 on unknown enum values, and silently sending a regex
    string masks the bug.
    """
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    if not _REGEX_META_RE.search(text):
        return text
    inner = text
    if inner.startswith("^"):
        inner = inner[1:]
    if inner.endswith("$"):
        inner = inner[:-1]
    inner = inner.strip()
    if inner.startswith("(") and inner.endswith(")"):
        inner = inner[1:-1]
    candidates = inner.split("|") if "|" in inner else [inner]
    for alt in candidates:
        alt = alt.strip()
        if alt and not _REGEX_META_RE.search(alt):
            return alt
    return None


def _strip_json_suffix(url: str) -> str:
    """Drop a trailing `.json` so UI-form URLs match the HTML-page regex."""
    return _catalog_strip_json_suffix(url)


def _normalise_url(url: str, placeholders: Mapping[str, str]) -> str | None:
    """Resolve placeholders, strip regex anchors, drop trailing `.json`.

    Returns None if placeholder expansion would leave unresolved
    ``__FOO__`` tokens — caller decides whether that's L3-pending or an
    outright non-match.
    """
    return _catalog_normalise_url(url, placeholders)


def _path_and_query(url: str) -> str:
    """Return just the path+query portion of a URL, so hostname components
    can't leak into ``project_path`` captures via greedy matching."""
    return _catalog_path_and_query(url)


def _is_listing_kind(kind: str) -> bool:
    if kind.startswith("gitlab_"):
        return _GITLAB_SITE.is_listing(kind)
    if kind.startswith("reddit_"):
        return _REDDIT_SITE.is_listing(kind)
    return False


def _disambiguate_root_segment(task: Mapping[str, Any], segment: str) -> str | None:
    """Resolve a bare ``/<segment>`` URL into a gitlab kind.

    Reads ``agent_context.gitlab.{user_handles,group_handles}`` populated
    by Phase 0c handle enrichment (see
    :mod:`worldsim.phases.phase_0c_handle_enrichment`). Returns ``"user"``
    or ``"group"`` for an unambiguous match; ``None`` when the segment is
    in both lists, neither list, or when the agent_context block is
    missing. The resolver does not guess: ambiguous cases fall through to
    ``kind=None`` with a categorized drop reason.
    """
    return _GITLAB_SITE.disambiguate_root_segment(task, segment)


def _listing_start_url(kind: str, resolved_url: str, fallback_url: str | None) -> str | None:
    if kind.startswith("gitlab_"):
        return _GITLAB_SITE.listing_start_url(kind, resolved_url, fallback_url)
    if kind.startswith("reddit_"):
        return _REDDIT_SITE.listing_start_url(kind, resolved_url, fallback_url)
    return fallback_url


def _match_gitlab(
    url: str,
    task: Mapping[str, Any] | None = None,
) -> tuple[ResourceKind, dict[str, str]] | None:
    hit = _GITLAB_SITE.match(
        url,
        task or {},
        TargetingContext(benchmark="webarena_verified", site="gitlab"),
    )
    if hit is None:
        return None
    kind, anchors = hit
    from worldsim.sites.gitlab import to_legacy_kind

    return to_legacy_kind(kind), anchors


def _match_reddit(url: str) -> tuple[ResourceKind, dict[str, str]] | None:
    hit = _REDDIT_SITE.match(
        url,
        {},
        TargetingContext(benchmark="webarena_verified", site="reddit"),
    )
    if hit is None:
        return None
    kind, anchors = hit
    from worldsim.sites.reddit import to_legacy_kind

    return to_legacy_kind(kind), anchors


def _iter_eval_urls(task: Mapping[str, Any]) -> list[str]:
    """Yield all URLs declared under ``reward_function.eval[].expected``.

    Prefers NetworkEventEvaluator entries first (they name the URL the
    agent must hit); AgentResponseEvaluator entries rarely carry URLs
    but are included as fallback.
    """
    return _catalog_iter_eval_urls(task)


def _url_with_expected_query_params(url: str, expected: Mapping[str, Any]) -> str:
    return _catalog_url_with_expected_query_params(url, expected)


def _iter_start_urls(task: Mapping[str, Any]) -> list[str]:
    return _catalog_iter_start_urls(task)


def _site_kind_for_task(task: Mapping[str, Any]) -> Literal["gitlab", "reddit"] | None:
    site = _catalog_site_kind_for_task(task)
    if site in {"gitlab", "reddit"}:
        return site  # type: ignore[return-value]
    return None


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


def _canonicalize_project_path(project_path: str) -> str:
    """Strip host prefix and surrounding slashes from a project_path.

    Returns the bare ``namespace/project`` (or ``namespace/subgroup/project``)
    form expected by GitLab's path-based API endpoints. The L3 LLM
    sometimes emits ``localhost:8023/foo/bar`` because the API probe's
    ``web_url`` puts the authority in the path; ``_anchors_from_gitlab_item``
    propagates that into ``project_path`` anchors. The bare form is what
    ``urllib.parse.quote(path, safe='')`` should percent-encode for
    ``GET /api/v4/projects/:id``.

    Idempotent: already-canonical inputs return unchanged. Empty input
    returns the empty string.
    """
    return _GITLAB_SITE.canonicalize_project_path(project_path)
