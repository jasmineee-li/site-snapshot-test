"""Phase 2 target resolution reconstruction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal
from urllib.parse import quote as urlquote

from worldsim.phase_2.target_resolution.constants import (
    _ISSUE_RE,
    _MR_RE,
    VIEWPORT_BUDGET_CHARS,
)
from worldsim.phase_2.target_resolution.encounter import (
    _attach_surfaces_for,
    _encounter_requirements,
)


def _reconstruct_start_url_from_anchors(
    site_kind: Literal["gitlab", "reddit"],
    kind: str,
    anchors: Mapping[str, Any],
    placeholders: Mapping[str, str],
) -> str | None:
    """Build a synthetic-host URL pointing at the concrete resource."""
    if site_kind == "gitlab":
        origin = placeholders.get("__GITLAB__")
    elif site_kind == "reddit":
        origin = placeholders.get("__REDDIT__")
    else:
        return None
    if not origin:
        return None
    base = origin.rstrip("/")

    if kind == "gitlab_issue":
        project_path = anchors.get("project_path")
        iid = anchors.get("issue_iid")
        if project_path and iid:
            return f"{base}/{_clean_project_path(str(project_path))}/-/issues/{iid}"
        return None
    if kind == "gitlab_mr":
        project_path = anchors.get("project_path")
        iid = anchors.get("mr_iid")
        if project_path and iid:
            return f"{base}/{_clean_project_path(str(project_path))}/-/merge_requests/{iid}"
        return None
    if kind == "gitlab_search_result":
        query = anchors.get("query")
        scope = anchors.get("scope") or "issues"
        project_path = anchors.get("project_path")
        if project_path:
            return f"{base}/{_clean_project_path(str(project_path))}/-/{scope}"
        if query:
            encoded = urlquote(str(query), safe="+")
            return f"{base}/search?search={encoded}&scope={scope}"
        return None
    if kind == "gitlab_dashboard_list":
        dashboard = anchors.get("dashboard")
        if dashboard:
            return f"{base}/dashboard/{dashboard}"
        return None
    if kind == "gitlab_user_profile":
        username = anchors.get("username")
        if username:
            return f"{base}/{username}"
        return None
    if kind == "gitlab_group":
        group_path = anchors.get("group_path")
        if group_path:
            return f"{base}/{group_path}"
        return None
    if kind == "gitlab_snippet":
        snippet_id = anchors.get("snippet_id")
        if snippet_id:
            return f"{base}/-/snippets/{snippet_id}"
        return None
    if kind == "gitlab_snippets_index":
        return f"{base}/-/snippets"
    if kind == "gitlab_project_milestone":
        project_path = anchors.get("project_path")
        iid = anchors.get("milestone_iid")
        if project_path and iid:
            return f"{base}/{_clean_project_path(str(project_path))}/-/milestones/{iid}"
        return None
    if kind == "gitlab_project_labels":
        project_path = anchors.get("project_path")
        if project_path:
            return f"{base}/{_clean_project_path(str(project_path))}/-/labels"
        return None
    if kind == "reddit_submission":
        forum = anchors.get("forum_name")
        submission_id = anchors.get("submission_id")
        if forum and submission_id:
            return f"{base}/f/{forum}/{submission_id}"
        return None
    if kind == "reddit_forum":
        forum = anchors.get("forum_name")
        if forum:
            return f"{base}/f/{forum}"
        return None
    if kind == "reddit_dashboard_list":
        user = anchors.get("user") or anchors.get("username")
        dashboard = anchors.get("dashboard")
        if user and dashboard:
            return f"{base}/user/{user}/{dashboard}"
        return None
    return None


def _anchors_from_gitlab_item(item: Mapping[str, Any], *, kind_hint: str) -> dict[str, Any]:
    """Project anchors out of a GitLab API item (issue or MR)."""
    anchors: dict[str, Any] = {}
    project_id = item.get("project_id")
    if project_id is not None:
        anchors["project_id"] = str(project_id)
    iid = item.get("iid")
    if iid is not None:
        if "mr" in kind_hint:
            anchors["mr_iid"] = str(iid)
        else:
            anchors["issue_iid"] = str(iid)
    web_url = str(item.get("web_url") or "")
    if web_url:
        # Extract project_path from web_url tail.
        match = _ISSUE_RE.search(web_url) or _MR_RE.search(web_url)
        if match:
            anchors["project_path"] = match.group("project_path")
    return anchors


def _anchors_from_reddit_submission(entry: Mapping[str, Any], forum_name: str) -> dict[str, Any]:
    submission_id = entry.get("id") or entry.get("submission_id")
    anchors: dict[str, Any] = {"forum_name": forum_name}
    if submission_id is not None:
        anchors["submission_id"] = str(submission_id)
    return anchors


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
        project_id = item.get("project_id")
        if project_id is not None:
            anchors["project_id"] = str(project_id)
        iid = item.get("iid")
        if iid is not None:
            anchors["mr_iid" if item_kind == "gitlab_mr" else "issue_iid"] = str(iid)
        web_url = str(item.get("web_url") or "")
        match = _ISSUE_RE.search(web_url) if item_kind == "gitlab_issue" else _MR_RE.search(web_url)
        if match:
            anchors["project_path"] = match.group("project_path")
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
        submission_id = item.get("id") or item.get("submission_id")
        if submission_id is None:
            return None
        anchors["submission_id"] = str(submission_id)
        anchors["forum_name"] = str(
            item.get("forum_name") or (base.get("anchors") or {}).get("forum_name") or ""
        )
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
    path = project_path.strip().strip("/")
    if "/" in path and path.split("/", 1)[0].startswith("localhost:"):
        path = path.split("/", 1)[1]
    return path
