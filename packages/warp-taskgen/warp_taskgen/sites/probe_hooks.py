"""Optional Site hooks that Phase 2 target resolution reaches through a bound Site.

Neither hook is part of :class:`~warp_taskgen.sites.contracts.SiteAdapter`.
A Site that lacks one simply yields ``None`` from the matching
:class:`~warp_taskgen.sites.bound_site.BoundSite` method, so a probe or listing
heuristic fails closed instead of guessing a route.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ProbeItemAnchorsCapability(Protocol):
    """Project one raw L3 probe row into the Site's route anchors."""

    def probe_item_anchors(
        self,
        item: Mapping[str, Any],
        *,
        kind_hint: str,
        forum_name: str | None = None,
    ) -> Mapping[str, Any] | None: ...


@runtime_checkable
class ListingTaskProjectPathCapability(Protocol):
    """Read the project path a listing-intent task names, if the Site has one."""

    def project_path_from_listing_task(
        self,
        instruction: str,
        *,
        resolved_start: str | None,
    ) -> str | None: ...


__all__ = ["ListingTaskProjectPathCapability", "ProbeItemAnchorsCapability"]
