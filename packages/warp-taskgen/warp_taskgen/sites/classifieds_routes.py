"""Classifieds route grammar and inventory-backed route facts.

The VisualWebArena classifieds page exposes a listing and its comments on one
ordinary-reader route.  The route is deliberately narrow: ``/index.php`` is
not a valid target by itself, and query parameters other than ``page=item``
and the concrete listing id are not promoted to a target.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from warp_taskgen.sites.contracts import CanonicalRoute, SiteRouteContractFacts

CLASSIFIEDS_BENCHMARK = "visualwebarena"
CLASSIFIEDS_SITE = "classifieds"
CLASSIFIEDS_PLACEHOLDER = "__CLASSIFIEDS__"

ClassifiedsResourceKind = Literal["listing", "listing_reply"]
_RESOURCE_KINDS: frozenset[str] = frozenset({"listing", "listing_reply"})


def to_local_kind(kind: str) -> str:
    """Normalize a canonical Classifieds kind without introducing aliases."""

    return str(kind or "").strip()


def is_resource_kind(kind: str) -> bool:
    return to_local_kind(kind) in _RESOURCE_KINDS


def listing_route() -> CanonicalRoute:
    """Build the one canonical ordinary-reader listing route."""

    return CanonicalRoute(
        id="classifieds.listing",
        site=CLASSIFIEDS_SITE,
        kind="listing",
        allowed_start_url_patterns=("/index.php?page=item&id={listing_id}",),
        compatibility_kind=None,
        anchor_examples=({"listing_id": "17"},),
        route_variant="item_detail",
        requires_inventory_backed_start_url=True,
    )


def route_for_kind(kind: str) -> CanonicalRoute | None:
    """Resolve a listing/reply kind to the parent listing route."""

    return listing_route() if is_resource_kind(kind) else None


def route_contract_facts(
    *,
    benchmark: str,
    profile: Mapping[str, Any],
    kind: str,
) -> SiteRouteContractFacts:
    """Project only deterministic route/inventory facts for Phase 1."""

    if str(benchmark or "").strip().lower() != CLASSIFIEDS_BENCHMARK:
        return SiteRouteContractFacts()
    if route_for_kind(kind) is None:
        return SiteRouteContractFacts()
    route = listing_route()
    examples = tuple(
        {
            "listing_id": listing_id,
            "start_url": f"{CLASSIFIEDS_PLACEHOLDER}/index.php?page=item&id={listing_id}",
        }
        for listing_id in listing_ids_from_profile(profile)
    )
    return SiteRouteContractFacts(
        allowed_start_url_patterns=tuple(
            f"{CLASSIFIEDS_PLACEHOLDER}{pattern}" for pattern in route.allowed_start_url_patterns
        ),
        anchor_examples=examples,
        requires_inventory_backed_start_url=True,
        route_variant=route.route_variant,
    )


def listing_ids_from_profile(profile: Mapping[str, Any]) -> tuple[str, ...]:
    """Extract concrete listing ids from explicit profile inventory only.

    This helper never invents an id and does not inspect private/admin-shaped
    fields.  If the profile carries no inventory, the empty tuple means that
    route validation can still operate on a caller-supplied deterministic
    anchor; it does not authorize a fallback id.
    """

    if not isinstance(profile, Mapping):
        return ()
    candidates: list[Mapping[str, Any]] = []
    available = profile.get("available_entities")
    if isinstance(available, Mapping):
        for key in ("listings", "items", "classifieds"):
            values = available.get(key)
            if isinstance(values, list):
                candidates.extend(item for item in values if isinstance(item, Mapping))
    data_model = profile.get("data_model")
    if isinstance(data_model, list):
        for entity in data_model:
            if not isinstance(entity, Mapping):
                continue
            entity_name = str(entity.get("entity") or "").strip().casefold()
            if entity_name not in {"listing", "listings", "item", "items"}:
                continue
            values = entity.get("sample_values")
            if isinstance(values, list):
                candidates.extend(item for item in values if isinstance(item, Mapping))

    ids: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        value = candidate.get(
            "listing_id",
            candidate.get(
                "item_id",
                candidate.get("pk_i_id", candidate.get("item_pk_i_id", candidate.get("id"))),
            ),
        )
        text = _listing_id_text(value)
        if text and text not in seen:
            seen.add(text)
            ids.append(text)
    return tuple(ids)


def _listing_id_text(value: Any) -> str | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    text = str(value).strip()
    return text if text.isdigit() and int(text) > 0 else None


__all__ = [
    "CLASSIFIEDS_BENCHMARK",
    "CLASSIFIEDS_PLACEHOLDER",
    "CLASSIFIEDS_SITE",
    "ClassifiedsResourceKind",
    "is_resource_kind",
    "listing_ids_from_profile",
    "listing_route",
    "route_contract_facts",
    "route_for_kind",
    "to_local_kind",
]
