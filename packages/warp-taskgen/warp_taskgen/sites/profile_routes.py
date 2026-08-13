"""Small reusable mechanics for Site-owned profile surface identity.

The alias tables and ambiguity decisions live beside each Site feature.  This
module only implements the mechanical matching shared by those features; it
does not import Phase 1/2 policy.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from warp_taskgen.benchmark_capabilities import normalize_benchmark_name
from warp_taskgen.sites.contracts import SurfaceResolution


@dataclass(frozen=True)
class ProfileSurfaceMapping:
    """Profile aliases and source-field aliases for one benchmark/Site."""

    benchmark: str
    profile_id_aliases: Mapping[str, str]
    source_field_aliases: Mapping[str, str]


def canonicalize_surface_id(
    *,
    mapping: ProfileSurfaceMapping | None,
    raw_surface_id: str | None,
) -> str | None:
    """Canonicalize a profile/editor surface without applying carrier policy."""

    raw = raw_surface_id.strip() if isinstance(raw_surface_id, str) else ""
    if not raw:
        return None
    if mapping is None:
        return None
    aliased = lookup_alias(mapping.profile_id_aliases, raw)
    if aliased:
        return aliased
    # Canonical IDs are stable dotted names.  Returning an unknown non-empty
    # value preserves the historical compatibility helper's normalization;
    # profile resolution still fails closed when the Site has no mapping.
    return raw


def resolve_profile_surface(
    *,
    mapping: ProfileSurfaceMapping | None,
    benchmark: str,
    site: str,
    profile: Mapping[str, Any],
    target_surface_id: str,
    kind: str | None = None,
    method: str | None = None,
    editor_surface_id: str | None = None,
    disambiguate: Callable[
        [list[SurfaceResolution], str, str | None, str | None, str | None],
        SurfaceResolution | None,
    ]
    | None = None,
) -> SurfaceResolution | None:
    """Resolve exactly one profile field through a Site's alias table."""

    benchmark_key = normalize_benchmark_name(benchmark or "webarena_verified")
    if mapping is None or mapping.benchmark != benchmark_key:
        return None
    target = canonicalize_surface_id(mapping=mapping, raw_surface_id=target_surface_id)
    if not target:
        return None
    surfaces = profile.get("injection_surface", [])
    if isinstance(surfaces, (str, bytes, bytearray)) or not isinstance(surfaces, Iterable):
        return None
    candidates: list[SurfaceResolution] = []
    for surface in surfaces:
        if not isinstance(surface, Mapping):
            continue
        surface_id = str(surface.get("id") or "").strip()
        source_field = str(surface.get("source_field") or "").strip() or None
        matches: list[str] = []
        if surface_id == target:
            matches.append("exact_profile_id")
        aliased = lookup_alias(mapping.profile_id_aliases, surface_id)
        if aliased == target:
            matches.append("adapter_profile_id_alias")
        source_alias = lookup_source_field_alias(mapping.source_field_aliases, source_field)
        if source_alias == target:
            matches.append("adapter_source_field_alias")
        # A canonical dotted ID is valid evidence even when a profile uses no
        # Site-specific alias (for example ``issue.description`` itself).
        if surface_key(surface_id) == surface_key(target):
            matches.append("core_surface_alias")
        if not matches:
            continue
        candidates.append(
            SurfaceResolution(
                benchmark=benchmark_key,
                site=site,
                canonical_surface_id=target,
                profile_surface_id=surface_id,
                profile_surface=surface,
                evidence="+".join(dict.fromkeys(matches)),
                source_field=source_field,
                editor_surface_id=editor_surface_id,
            )
        )
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    if disambiguate is None:
        return None
    return disambiguate(candidates, target, kind, method, editor_surface_id)


def lookup_alias(aliases: Mapping[str, str], value: str | None) -> str | None:
    key = surface_key(value)
    for raw, canonical in aliases.items():
        if surface_key(raw) == key:
            return canonical
    return None


def lookup_source_field_alias(aliases: Mapping[str, str], value: str | None) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().casefold()
    for raw, canonical in aliases.items():
        if str(raw).strip().casefold() == normalized:
            return canonical
    return None


def surface_key(value: str | None) -> str:
    return "".join(char for char in str(value or "").casefold() if char.isalnum())


__all__ = [
    "ProfileSurfaceMapping",
    "canonicalize_surface_id",
    "lookup_alias",
    "lookup_source_field_alias",
    "resolve_profile_surface",
    "surface_key",
]
