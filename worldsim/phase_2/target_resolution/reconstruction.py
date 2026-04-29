"""Phase 2 target resolution reconstruction."""
# ruff: noqa: F821

from __future__ import annotations

from worldsim.phase_2.target_resolution._context import install_context

install_context(globals())

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
    site_kind: Literal[gitlab, reddit] = (
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

