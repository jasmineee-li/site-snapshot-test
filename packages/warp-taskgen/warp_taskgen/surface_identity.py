"""Compatibility facade for Site-owned profile surface identity.

Profile aliases and ambiguity rules now live in ``warp_taskgen.sites``.  This
module remains import-compatible for one migration cycle; it intentionally
contains no benchmark/Site mapping tables or policy decisions.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from warp_taskgen.phases.phase_2_core_surfaces import canonical_core_surface
from warp_taskgen.sites import SurfaceResolution, default_catalog


def canonicalize_surface_id(
    *,
    benchmark: str,
    site: str,
    raw_surface_id: str | None,
) -> str | None:
    """Delegate profile/editor surface normalization to the bound Site."""

    try:
        bound = default_catalog().bind(benchmark=benchmark, site=site)
    except Exception:
        return None
    return bound.canonicalize_surface_id(raw_surface_id) or canonical_core_surface(
        site.strip().lower(), raw_surface_id
    )


def resolve_profile_surface(
    *,
    benchmark: str,
    site: str,
    profile: Mapping[str, Any],
    target_surface_id: str,
    kind: str | None = None,
    method: str | None = None,
    editor_surface_id: str | None = None,
) -> SurfaceResolution | None:
    """Delegate concrete profile-field resolution to the bound Site."""

    try:
        bound = default_catalog().bind(
            benchmark=benchmark,
            site=site,
            profile=profile,
        )
    except Exception:
        return None
    return bound.resolve_profile_surface(
        target_surface_id,
        kind=kind,
        method=method,
        editor_surface_id=editor_surface_id,
    )


def surface_resolution_dict(resolution: SurfaceResolution | None) -> dict[str, Any] | None:
    """Serialize a Site resolution using the historical artifact shape."""

    if resolution is None:
        return None
    payload: dict[str, Any] = {
        "benchmark": resolution.benchmark,
        "site": resolution.site,
        "canonical_surface_id": resolution.canonical_surface_id,
        "profile_surface_id": resolution.profile_surface_id,
        "evidence": resolution.evidence,
    }
    if resolution.source_field:
        payload["source_field"] = resolution.source_field
    if resolution.editor_surface_id:
        payload["editor_surface_id"] = resolution.editor_surface_id
    return payload


def has_surface_mapping(*, benchmark: str, site: str) -> bool:
    """Return whether the Site has opted into profile identity capability."""

    try:
        bound = default_catalog().bind(benchmark=benchmark, site=site)
    except Exception:
        return False
    return bound.supports_profile_routes()


__all__ = [
    "SurfaceResolution",
    "canonicalize_surface_id",
    "has_surface_mapping",
    "resolve_profile_surface",
    "surface_resolution_dict",
]
