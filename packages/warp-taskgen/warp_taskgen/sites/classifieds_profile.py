"""Classifieds-owned profile surface vocabulary.

Classifieds has exactly one WARP carrier in this slice: the body of a reply
to an existing listing.  Listing title and description are intentionally not
aliases here; they remain unavailable even when a historical profile happens
to mention similarly named fields.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from warp_taskgen.benchmark_capabilities import normalize_benchmark_name
from warp_taskgen.sites.contracts import SurfaceResolution
from warp_taskgen.sites.profile_routes import (
    ProfileSurfaceMapping,
)
from warp_taskgen.sites.profile_routes import (
    canonicalize_surface_id as _canonicalize,
)
from warp_taskgen.sites.profile_routes import (
    resolve_profile_surface as _resolve,
)

CLASSIFIEDS_BENCHMARK = "visualwebarena"
CLASSIFIEDS_SITE = "classifieds"
CLASSIFIEDS_SURFACE = "listing_reply.body"

_PROFILE_ID_ALIASES = {
    "listing_reply.body": CLASSIFIEDS_SURFACE,
    "listing_reply_body": CLASSIFIEDS_SURFACE,
    "listing_reply": CLASSIFIEDS_SURFACE,
    "comment.body": CLASSIFIEDS_SURFACE,
    "comment_body": CLASSIFIEDS_SURFACE,
    "item_comment.body": CLASSIFIEDS_SURFACE,
    "item_comment_body": CLASSIFIEDS_SURFACE,
}
_SOURCE_FIELD_ALIASES = {
    "listing_reply.body": CLASSIFIEDS_SURFACE,
    "item_comment.body": CLASSIFIEDS_SURFACE,
    "comment.body": CLASSIFIEDS_SURFACE,
    "comments.body": CLASSIFIEDS_SURFACE,
}
_MAPPING = ProfileSurfaceMapping(
    benchmark=CLASSIFIEDS_BENCHMARK,
    profile_id_aliases=_PROFILE_ID_ALIASES,
    source_field_aliases=_SOURCE_FIELD_ALIASES,
)


def mapping_for(benchmark: str) -> ProfileSurfaceMapping | None:
    return _MAPPING if normalize_benchmark_name(benchmark or "") == CLASSIFIEDS_BENCHMARK else None


def canonicalize_surface_id(*, benchmark: str, raw_surface_id: str | None) -> str | None:
    """Canonicalize only the body of a listing reply."""

    candidate = _canonicalize(mapping=mapping_for(benchmark), raw_surface_id=raw_surface_id)
    return candidate if candidate == CLASSIFIEDS_SURFACE else None


def resolve_profile_surface(
    *,
    benchmark: str,
    profile: Mapping[str, Any],
    target_surface_id: str,
    kind: str | None = None,
    method: str | None = None,
    editor_surface_id: str | None = None,
) -> SurfaceResolution | None:
    canonical = canonicalize_surface_id(benchmark=benchmark, raw_surface_id=target_surface_id)
    if canonical != CLASSIFIEDS_SURFACE:
        return None
    return _resolve(
        mapping=mapping_for(benchmark),
        benchmark=benchmark,
        site=CLASSIFIEDS_SITE,
        profile=profile,
        target_surface_id=canonical,
        kind=kind,
        method=method,
        editor_surface_id=editor_surface_id,
    )


class ClassifiedsProfileIdentity:
    """Mixin implementing Classifieds profile-surface identity."""

    def canonicalize_surface_id(
        self,
        *,
        benchmark: str,
        raw_surface_id: str | None,
    ) -> str | None:
        return canonicalize_surface_id(benchmark=benchmark, raw_surface_id=raw_surface_id)

    def resolve_profile_surface(
        self,
        *,
        benchmark: str,
        profile: Mapping[str, Any],
        target_surface_id: str,
        kind: str | None = None,
        method: str | None = None,
        editor_surface_id: str | None = None,
    ) -> SurfaceResolution | None:
        return resolve_profile_surface(
            benchmark=benchmark,
            profile=profile,
            target_surface_id=target_surface_id,
            kind=kind,
            method=method,
            editor_surface_id=editor_surface_id,
        )


__all__ = [
    "CLASSIFIEDS_BENCHMARK",
    "CLASSIFIEDS_SITE",
    "CLASSIFIEDS_SURFACE",
    "ClassifiedsProfileIdentity",
    "canonicalize_surface_id",
    "mapping_for",
    "resolve_profile_surface",
]
