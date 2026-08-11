"""Phase 2 target resolution reconstruction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from worldsim.phase_2.target_resolution.constants import VIEWPORT_BUDGET_CHARS
from worldsim.phase_2.target_resolution.encounter import (
    _attach_surfaces_for,
    _encounter_requirements,
)
from worldsim.sites.catalog import TargetingContext
from worldsim.sites.gitlab import GitLabSite
from worldsim.sites.gitlab import to_local_kind as _gitlab_to_local
from worldsim.sites.reddit import RedditSite
from worldsim.sites.reddit import to_local_kind as _reddit_to_local

_GITLAB_SITE = GitLabSite()
_REDDIT_SITE = RedditSite()


def _reconstruct_start_url_from_anchors(
    site_kind: Literal["gitlab", "reddit"],
    kind: str,
    anchors: Mapping[str, Any],
    placeholders: Mapping[str, str],
) -> str | None:
    """Compatibility delegate to the Site-owned route reconstruction."""

    if site_kind == "gitlab":
        adapter = _GITLAB_SITE
        local_kind = _gitlab_to_local(kind)
    elif site_kind == "reddit":
        adapter = _REDDIT_SITE
        local_kind = _reddit_to_local(kind)
    else:
        return None
    context = TargetingContext(
        benchmark="webarena_verified",
        site=site_kind,
        placeholders=placeholders,
    )
    return adapter.reconstruct(local_kind, anchors, context)


def _anchors_from_gitlab_item(item: Mapping[str, Any], *, kind_hint: str) -> dict[str, Any]:
    """Project anchors out of a GitLab API item (issue or MR)."""
    return _GITLAB_SITE.anchors_from_item(item, kind_hint=kind_hint)


def _anchors_from_reddit_submission(entry: Mapping[str, Any], forum_name: str) -> dict[str, Any]:
    return _REDDIT_SITE.anchors_from_submission(entry, forum_name)


def _project_item_to_record(
    base: Mapping[str, Any],
    item: Mapping[str, Any],
    placeholders: Mapping[str, str] | None = None,
    *,
    benchmark: str = "webarena_verified",
) -> dict[str, Any] | None:
    item_kind = item.get("_item_kind")
    if item_kind not in ("gitlab_issue", "gitlab_mr", "reddit_submission"):
        return None
    source_listing_kind = base.get("kind")
    source_listing_url = base.get("start_url_resolved")
    record = dict(base)
    record["kind"] = item_kind
    record["layer"] = "L4"
    if isinstance(source_listing_kind, str) and source_listing_kind:
        record["source_listing_kind"] = source_listing_kind
    if isinstance(source_listing_url, str) and source_listing_url.strip():
        record["benign_read_url"] = source_listing_url
    site_kind: Literal["gitlab", "reddit"] = (
        "reddit" if item_kind == "reddit_submission" else "gitlab"
    )
    record["attach_surfaces"] = _attach_surfaces_for(item_kind, benchmark=benchmark, site=site_kind)

    anchors: dict[str, Any] = {}
    if item_kind in {"gitlab_issue", "gitlab_mr"}:
        anchors.update(_anchors_from_gitlab_item(item, kind_hint=item_kind))
        title = item.get("title")
        if isinstance(title, str) and title.strip():
            record["l4_title"] = title.strip()
        visible_href = item.get("_entry_visible_href")
        if isinstance(visible_href, str) and visible_href.strip():
            record["entry_visibility_evidence"] = {
                "entry_url": record.get("benign_read_url"),
                "href_path": visible_href.strip(),
                "source": "dashboard_dom_href",
            }
    else:
        forum_name = str(
            item.get("forum_name") or (base.get("anchors") or {}).get("forum_name") or ""
        )
        anchors.update(_anchors_from_reddit_submission(item, forum_name))
        if "submission_id" not in anchors:
            return None
        title = item.get("title")
        if isinstance(title, str) and title.strip():
            record["l4_title"] = title.strip()

    if not anchors:
        return None
    record["anchors"] = anchors
    # encounter_requirements are recomputed for the concrete item kind.
    record["encounter_requirements"] = _encounter_requirements(item_kind, {}, anchors)
    # Viewport budget stays constant.
    record["encounter_requirements"].setdefault("viewport_budget_chars", VIEWPORT_BUDGET_CHARS)
    # Keep both URLs. The concrete item URL is where the seed is attached;
    # the benign_read_url is the page the benign task actually asks the
    # agent to observe. Phase 2c's exposure contract must verify the latter.
    if placeholders is not None:
        reconstructed = _reconstruct_start_url_from_anchors(
            site_kind, item_kind, anchors, placeholders
        )
        if reconstructed:
            record["start_url_resolved"] = reconstructed
            record["seeded_detail_url"] = reconstructed
    return record


def _clean_project_path(project_path: str) -> str:
    """Strip an L4-prefixed ``localhost:NNNN/`` authority from ``project_path``.

    ``_project_item_to_record`` writes anchors with paths like
    ``localhost:8023/byteblaze/a11y-webring.club`` (the authority from
    the API probe's ``web_url``). For URL reconstruction we want just
    the group-slashed path suffix.
    """
    return _GITLAB_SITE.clean_project_path(project_path)
