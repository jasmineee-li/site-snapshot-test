"""Phase 2 target resolution reconstruction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal
from urllib.parse import urlsplit

from warp_taskgen.phase_2.target_resolution.listing_records import compose_listing_record
from warp_taskgen.sites import ListingItemCandidate, SiteCatalog, TargetingFailure
from warp_taskgen.sites.catalog import TargetingContext
from warp_taskgen.sites.gitlab import GitLabSite
from warp_taskgen.sites.gitlab import to_local_kind as _gitlab_to_local
from warp_taskgen.sites.reddit import RedditSite
from warp_taskgen.sites.reddit import to_local_kind as _reddit_to_local

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
    if not isinstance(item, Mapping):
        return None
    item_kind = item.get("_item_kind")
    source_kind = base.get("kind")
    if not isinstance(item_kind, str) or not isinstance(source_kind, str):
        return None
    catalog = SiteCatalog()
    site_kind = catalog.site_for_kind(source_kind, benchmark=benchmark)
    if site_kind is None:
        return None
    source_url = base.get("start_url_resolved")
    seam_source_kind = source_kind
    payload: Mapping[str, Any] = item
    # Reddit forum expansion remains disabled in the live L4 dispatcher, but
    # historical direct callers of this facade may still project a row.  Use
    # the Site's existing dashboard-list route solely for that compatibility
    # call and restore the original artifact provenance below.
    if site_kind == "reddit" and source_kind == "reddit_forum":
        seam_source_kind = "reddit_dashboard_list"
        base_anchors = base.get("anchors")
        forum_name = base_anchors.get("forum_name") if isinstance(base_anchors, Mapping) else None
        if forum_name and "forum_name" not in item:
            payload = {**item, "forum_name": forum_name}
    origin = _origin_from_url(source_url)
    effective_placeholders = dict(placeholders or {})
    try:
        bound = catalog.bind(
            benchmark=benchmark,
            site=site_kind,
            origin=origin,
            placeholders=effective_placeholders,
        )
        candidate = ListingItemCandidate(
            source_kind=seam_source_kind,
            item_kind=item_kind,
            payload=payload,
            evidence_url=source_url if isinstance(source_url, str) else None,
        )
    except (TypeError, ValueError):
        return None
    target = bound.materialize_listing_entry(candidate)
    if isinstance(target, TargetingFailure):
        return None
    record = compose_listing_record(base, candidate, target, benchmark=benchmark)
    if record is None:
        return None
    # Preserve the old facade's artifact-only behavior for callers that did
    # not provide placeholder context.  The canonical L4 path always emits
    # the reconstructed detail URL; this compatibility wrapper does not.
    if placeholders is None:
        if isinstance(source_url, str):
            record["start_url_resolved"] = source_url
        record.pop("seeded_detail_url", None)
    if seam_source_kind != source_kind:
        record["source_listing_kind"] = source_kind
    return record


def _origin_from_url(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    parsed = urlsplit(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return f"{parsed.scheme}://{parsed.netloc}"


def _clean_project_path(project_path: str) -> str:
    """Strip an L4-prefixed ``localhost:NNNN/`` authority from ``project_path``.

    ``_project_item_to_record`` writes anchors with paths like
    ``localhost:8023/byteblaze/a11y-webring.club`` (the authority from
    the API probe's ``web_url``). For URL reconstruction we want just
    the group-slashed path suffix.
    """
    return _GITLAB_SITE.clean_project_path(project_path)
