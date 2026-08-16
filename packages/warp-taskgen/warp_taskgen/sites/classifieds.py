"""Classifieds Site feature contracts for VisualWebArena.

This module is intentionally feature-local.  It describes deterministic route
and identity behavior and delegates read-surface/readback interpretation to
the sibling capabilities; it does not register a Site, editor, policy, or
runtime branch globally.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from warp_taskgen.sites.classifieds_profile import ClassifiedsProfileIdentity
from warp_taskgen.sites.classifieds_read_surface import ClassifiedsReadSurfaceCapability
from warp_taskgen.sites.classifieds_readback import ClassifiedsReadbackCapability
from warp_taskgen.sites.classifieds_routes import (
    CLASSIFIEDS_BENCHMARK,
    CLASSIFIEDS_SITE,
    is_resource_kind,
    listing_ids_from_profile,
    listing_route,
    route_contract_facts,
    to_local_kind,
)
from warp_taskgen.sites.contracts import CanonicalRoute, TargetingContext


def _scalar_id(value: Any) -> str | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    text = str(value).strip()
    return text if text.isdigit() and int(text) > 0 else None


def _candidate_listing_ids(value: Mapping[str, Any]) -> list[str]:
    ids: list[str] = []
    direct = value.get("listing_id")
    scalar = _scalar_id(direct)
    if scalar is not None:
        ids.append(scalar)
    for key in ("benign_target_resource", "target_resource", "resource", "target", "anchors"):
        nested = value.get(key)
        if isinstance(nested, Mapping):
            nested_id = _scalar_id(nested.get("listing_id"))
            if nested_id is not None:
                ids.append(nested_id)
            anchors = nested.get("anchors")
            if isinstance(anchors, Mapping):
                anchor_id = _scalar_id(anchors.get("listing_id"))
                if anchor_id is not None:
                    ids.append(anchor_id)
    return ids


def _task_listing_id(task: Mapping[str, Any]) -> str | None:
    ids = _candidate_listing_ids(task)
    unique = set(ids)
    return next(iter(unique)) if len(unique) == 1 else None


def _task_has_conflicting_listing_ids(task: Mapping[str, Any]) -> bool:
    ids = _candidate_listing_ids(task)
    return bool(ids) and len(set(ids)) != 1


def _same_origin(url: str, context: TargetingContext) -> bool:
    origin = context.site_origin()
    if origin is None:
        return False
    try:
        candidate = urlsplit(url)
        expected = urlsplit(origin)
    except ValueError:
        return False
    if candidate.scheme or candidate.netloc:
        return (candidate.scheme, candidate.netloc) == (expected.scheme, expected.netloc)
    # Path-local evidence is safe only when it cannot smuggle a network-path
    # reference or an unresolved scheme into the adapter.
    return bool(url.startswith("/") and not url.startswith("//"))


def _match_listing_id(url: str) -> str | None:
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if parts.path != "/index.php" or parts.fragment:
        return None
    try:
        query = parse_qsl(parts.query, keep_blank_values=True, strict_parsing=True)
    except ValueError:
        return None
    if len(query) != 2 or query[0][0] != "page" or query[0][1] != "item" or query[1][0] != "id":
        return None
    return _scalar_id(query[1][1])


class ClassifiedsSite(
    ClassifiedsProfileIdentity,
    ClassifiedsReadSurfaceCapability,
    ClassifiedsReadbackCapability,
):
    """Deterministic adapter for the ordinary VisualWebArena classifieds page."""

    site = CLASSIFIEDS_SITE
    supported_benchmarks = frozenset({CLASSIFIEDS_BENCHMARK})
    resource_kinds = frozenset({"listing", "listing_reply"})

    def validate(self) -> None:
        route = listing_route()
        if route.id != "classifieds.listing" or route.kind != "listing":
            raise ValueError("Classifieds listing route definition drifted")

    def validate_task(self, task: Mapping[str, Any]) -> tuple[str, str] | None:
        if not isinstance(task, Mapping):
            return ("malformed_metadata", "task must be a mapping")
        if _task_has_conflicting_listing_ids(task):
            return ("malformed_metadata", "task listing identity fields disagree")
        for key in ("title", "description"):
            surface = task.get("target_surface_id")
            if isinstance(surface, str) and surface.strip().casefold() in {
                f"listing.{key}",
            }:
                return ("unsupported_surface", "Classifieds exposes only listing_reply.body")
        return None

    def routes(self, context: TargetingContext) -> tuple[CanonicalRoute, ...]:
        del context
        return (listing_route(),)

    def match(
        self,
        url: str,
        task: Mapping[str, Any],
        context: TargetingContext,
    ) -> tuple[str, dict[str, Any]] | None:
        if context.site != CLASSIFIEDS_SITE or context.benchmark != CLASSIFIEDS_BENCHMARK:
            return None
        if not isinstance(url, str) or not _same_origin(url, context):
            return None
        listing_id = _match_listing_id(url)
        if listing_id is None:
            return None
        expected = _task_listing_id(task) if isinstance(task, Mapping) else None
        if expected is not None and expected != listing_id:
            return None
        inventory = listing_ids_from_profile(context.profile)
        if inventory and listing_id not in inventory:
            return None
        return "listing", {"listing_id": listing_id}

    def reconstruct(
        self,
        kind: str,
        anchors: Mapping[str, Any],
        context: TargetingContext,
    ) -> str | None:
        if context.site != CLASSIFIEDS_SITE or context.benchmark != CLASSIFIEDS_BENCHMARK:
            return None
        if not is_resource_kind(kind) or not isinstance(anchors, Mapping):
            return None
        listing_id = _scalar_id(anchors.get("listing_id"))
        origin = context.site_origin()
        if listing_id is None or origin is None:
            return None
        inventory = listing_ids_from_profile(context.profile)
        if inventory and listing_id not in inventory:
            return None
        return f"{origin}/index.php?page=item&id={listing_id}"

    def is_listing(self, kind: str) -> bool:
        return to_local_kind(kind) == "listing"

    def listing_start_url(
        self,
        kind: str,
        resolved_url: str,
        fallback_url: str | None,
    ) -> str | None:
        del fallback_url
        return resolved_url if self.is_listing(kind) and _match_listing_id(resolved_url) else None

    def route_contract_facts(
        self,
        *,
        benchmark: str,
        profile: Mapping[str, Any],
        kind: str,
    ) -> Any:
        return route_contract_facts(benchmark=benchmark, profile=profile, kind=kind)


ClassifiedsAdapter = ClassifiedsSite


def classifieds_site() -> ClassifiedsSite:
    """Return a fresh feature adapter without mutating a catalog."""

    return ClassifiedsSite()


def classifieds_editor_specs() -> tuple[Any, ...]:
    """Return the pure editor spec projection for integration composition."""

    from warp_taskgen.sites.classifieds_editor import classifieds_editor_specs as build

    return build()


__all__ = [
    "CLASSIFIEDS_BENCHMARK",
    "CLASSIFIEDS_SITE",
    "ClassifiedsAdapter",
    "ClassifiedsSite",
    "classifieds_editor_specs",
    "classifieds_site",
]
