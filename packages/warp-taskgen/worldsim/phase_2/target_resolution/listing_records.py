"""Phase 2 artifact composition for materialized listing entries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from worldsim.phase_2.target_resolution.constants import VIEWPORT_BUDGET_CHARS
from worldsim.phase_2.target_resolution.encounter import (
    _attach_surfaces_for,
    _encounter_requirements,
)
from worldsim.sites import ListingItemCandidate, ResolvedTarget


def compose_listing_record(
    base: Mapping[str, Any],
    candidate: ListingItemCandidate,
    target: ResolvedTarget,
    *,
    benchmark: str = "webarena_verified",
) -> dict[str, Any] | None:
    """Compose the legacy Phase 2 record around a canonical L4 target.

    The Site seam supplies only canonical item kind, anchors, and URL.  Phase
    2 retains artifact fields that describe benign traversal and exposure
    evidence, including title/DOM visibility, encounter requirements, and
    compatibility Resource Kinds.
    """

    route = target.canonical_route
    compatibility_kind = route.compatibility_kind if route is not None else target.kind
    if not isinstance(compatibility_kind, str) or not compatibility_kind.strip():
        return None
    site_kind = target.site
    record = dict(base)
    record["kind"] = compatibility_kind
    record["anchors"] = dict(target.anchors)
    record["start_url_resolved"] = target.start_url_resolved
    record["seeded_detail_url"] = target.start_url_resolved
    record["layer"] = "L4"
    source_listing_kind = base.get("kind")
    record["source_listing_kind"] = (
        source_listing_kind.strip()
        if isinstance(source_listing_kind, str) and source_listing_kind.strip()
        else candidate.source_kind
    )
    source_listing_url = base.get("start_url_resolved") or candidate.evidence_url
    if isinstance(source_listing_url, str) and source_listing_url.strip():
        record["benign_read_url"] = source_listing_url
    record["attach_surfaces"] = _attach_surfaces_for(
        compatibility_kind,
        benchmark=benchmark,
        site=site_kind,
    )
    record["encounter_requirements"] = _encounter_requirements(
        compatibility_kind,
        {},
        target.anchors,
    )
    record["encounter_requirements"].setdefault("viewport_budget_chars", VIEWPORT_BUDGET_CHARS)

    payload = candidate.payload
    title = payload.get("title")
    if isinstance(title, str) and title.strip():
        record["l4_title"] = title.strip()
    visible_href = payload.get("_entry_visible_href")
    if isinstance(visible_href, str) and visible_href.strip():
        record["entry_visibility_evidence"] = {
            "entry_url": record.get("benign_read_url"),
            "href_path": visible_href.strip(),
            "source": "dashboard_dom_href",
        }
    return record


__all__ = ["compose_listing_record"]
