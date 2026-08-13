from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

from warp_taskgen.phase_2.text_fill.constants import (
    _CATEGORY_LENGTH_BUDGETS,
    _SITE_TO_SITE_KIND,
    _SUFFIX_TO_CATEGORY,
)
from warp_taskgen.phases.phase_2_core_surfaces import CORE_SURFACES, canonical_core_surface
from warp_taskgen.surface_identity import has_surface_mapping, resolve_profile_surface

logger = logging.getLogger(__name__)


def load_voice_registry(registry_path: Path | None = None) -> dict[str, Any]:
    path = registry_path or (
        Path(__file__).resolve().parents[2] / "voice_exemplars" / "registry.json"
    )
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"voice exemplar registry at {path} must be a JSON object")
    data["_registry_path"] = str(path)
    return data


def _classify_source_field(source_field: str) -> str:
    """Derive a semantic category from a source_field suffix.

    Accepts formats like ``Entity.field`` or ``entity.field_name``.  Falls
    back to ``long_body`` when no suffix pattern matches so that tasks are
    never dropped.
    """
    if "." not in source_field:
        logger.warning(
            "source_field %r has no dot-separated suffix, defaulting to long_body",
            source_field,
        )
        return "long_body"
    _, _, suffix = source_field.rpartition(".")
    suffix_lower = suffix.lower()
    # Check each suffix group; a field like "short_description" should match
    # "description" via endswith, while "title" matches exactly.
    for suffixes, category in _SUFFIX_TO_CATEGORY:
        for candidate in suffixes:
            if suffix_lower == candidate or suffix_lower.endswith(f"_{candidate}"):
                return category
    logger.warning(
        "source_field %r suffix %r matched no pattern, defaulting to long_body",
        source_field,
        suffix,
    )
    return "long_body"


def resolve_site_kind(
    registry: dict[str, Any],
    site: str,
    target_surface_id: str,
    *,
    source_field: str | None = None,
) -> str:
    """Resolve the voice exemplar site_kind for a given surface.

    Uses the ``source_field`` suffix to determine the semantic category, then
    maps the site to an existing voice exemplar bank.  Falls back gracefully
    when ``source_field`` is absent or has no dot separator.
    """
    # Determine which voice exemplar bank to use based on site
    site_kind = _SITE_TO_SITE_KIND.get(site)
    if site_kind is None:
        # Unknown site: pick a reasonable default
        logger.warning(
            "no site_kind mapping for site %r, defaulting to marketplace_review",
            site,
        )
        site_kind = "marketplace_review"
    # Validate it exists in registry
    site_kinds = registry.get("site_kinds")
    if isinstance(site_kinds, dict) and site_kind in site_kinds:
        return site_kind
    # Last-resort: return the first available site_kind
    if isinstance(site_kinds, dict) and site_kinds:
        fallback = next(iter(site_kinds))
        logger.warning(
            "site_kind %r missing from registry, using %r",
            site_kind,
            fallback,
        )
        return fallback
    raise ValueError("voice exemplar registry has no site_kinds entries")


def derive_length_budget(
    task: dict[str, Any],
    site_profile: dict[str, Any],
    registry: dict[str, Any],
) -> dict[str, Any]:
    source = "fallback_default"
    # Resolve source_field from the site profile surface, or use the task-level value
    resolved_source_field: str | None = task.get("source_field")
    surface = _find_surface_by_id(site_profile, task)
    if isinstance(surface, dict) and resolved_source_field is None:
        resolved_source_field = surface.get("source_field")
    exemplar_budget = _exemplar_length_budget(
        registry,
        site=str(task.get("site", "")),
        target_surface_id=str(task.get("target_surface_id", "")),
        source_field=resolved_source_field,
    )
    max_chars: int | None = None
    if isinstance(surface, dict):
        source_field = resolved_source_field
        if isinstance(source_field, str) and "." in source_field:
            entity_name, _, field_name = source_field.partition(".")
            field_type = _field_type(site_profile, entity_name, field_name)
            varchar_limit = _varchar_limit(field_type)
            if varchar_limit is not None:
                max_chars = varchar_limit
                source = "data_model_varchar"
            elif isinstance(field_type, str) and "text" in field_type.lower():
                source = "data_model_text"

    if exemplar_budget is not None:
        exemplar_min = int(exemplar_budget.get("min", 80))
        exemplar_max = int(exemplar_budget.get("max", 1500))
        if max_chars is None:
            max_chars = exemplar_max
            if source == "fallback_default":
                source = "voice_exemplar_stats"
        else:
            max_chars = min(max_chars, exemplar_max)
        min_chars = min(exemplar_min, max_chars)
    else:
        if max_chars is None:
            max_chars = 1500
        min_chars = min(80, max_chars)

    min_chars = max(16, min_chars)
    max_chars = max(min_chars, max_chars)
    return {"min": min_chars, "max": max_chars, "source": source}


def _select_exemplars(
    registry: dict[str, Any],
    *,
    site_kind: str,
    framing: str,
    seed_material: str,
    sample_size: int = 3,
) -> list[str]:
    payload = _load_site_kind_payload(registry, site_kind)
    framings = payload.get("framings")
    if not isinstance(framings, dict):
        raise ValueError(f"voice exemplar payload for {site_kind!r} missing framings object")
    exemplars = framings.get(framing)
    if not isinstance(exemplars, list) or len(exemplars) < sample_size:
        raise ValueError(
            f"voice exemplar bank for {(site_kind, framing)!r} must contain >= {sample_size} samples"
        )
    ordered = sorted(
        exemplars,
        key=lambda item: hashlib.sha256(f"{seed_material}::{item}".encode()).hexdigest(),
    )
    return [str(item) for item in ordered[:sample_size]]


def _exemplar_length_budget(
    registry: dict[str, Any],
    *,
    site: str,
    target_surface_id: str,
    source_field: str | None = None,
) -> dict[str, Any] | None:
    # Prefer category-based budget from source_field pattern matching
    if isinstance(source_field, str) and source_field.strip():
        category = _classify_source_field(source_field)
        cat_budget = _CATEGORY_LENGTH_BUDGETS.get(category)
        if cat_budget is not None:
            return dict(cat_budget)
    # Fall back to the exemplar payload's length_budget
    site_kind = resolve_site_kind(
        registry,
        site,
        target_surface_id,
        source_field=source_field,
    )
    payload = _load_site_kind_payload(registry, site_kind)
    budget = payload.get("length_budget")
    return budget if isinstance(budget, dict) else None


def _load_site_kind_payload(registry: dict[str, Any], site_kind: str) -> dict[str, Any]:
    site_kinds = registry.get("site_kinds")
    if not isinstance(site_kinds, dict):
        raise ValueError("voice exemplar registry missing site_kinds object")
    config = site_kinds.get(site_kind)
    if not isinstance(config, dict):
        raise ValueError(f"voice exemplar registry missing site_kind {site_kind!r}")
    rel_path = config.get("file")
    if not isinstance(rel_path, str) or not rel_path:
        raise ValueError(f"voice exemplar registry site_kind {site_kind!r} missing file path")
    registry_path = Path(str(registry["_registry_path"]))
    payload_path = registry_path.parent / rel_path
    data = json.loads(payload_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"voice exemplar payload at {payload_path} must be a JSON object")
    return data


def _find_surface_by_id(
    site_profile: dict[str, Any],
    task_or_surface_id: dict[str, Any] | str,
) -> dict[str, Any] | None:
    site = str(site_profile.get("site") or site_profile.get("site_name") or "").strip().lower()
    if isinstance(task_or_surface_id, dict):
        task = task_or_surface_id
        target_surface_id = str(task.get("target_surface_id") or "")
        resolution = resolve_profile_surface(
            benchmark=str(task.get("benchmark") or "webarena_verified"),
            site=site or str(task.get("site") or ""),
            profile=site_profile,
            target_surface_id=target_surface_id,
            kind=_route_kind_from_task(task),
            method=str(task.get("editor_method") or "") or _route_method_from_task(task),
            editor_surface_id=str(task.get("editor_surface_id") or "") or None,
        )
        if resolution is not None and isinstance(resolution.profile_surface, dict):
            return resolution.profile_surface
        if has_surface_mapping(
            benchmark=str(task.get("benchmark") or "webarena_verified"),
            site=site or str(task.get("site") or ""),
        ):
            return None
    else:
        target_surface_id = str(task_or_surface_id)
    sites = (site,) if site else tuple(CORE_SURFACES)
    canonical_targets = {canonical_core_surface(site_key, target_surface_id) for site_key in sites}
    for surface in site_profile.get("injection_surface", []):
        if not isinstance(surface, dict):
            continue
        surface_id = surface.get("id")
        if surface_id == target_surface_id:
            return surface
        if any(
            canonical_core_surface(site_key, str(surface_id or "")) in canonical_targets
            for site_key in sites
        ):
            return surface
    return None


def _route_kind_from_task(task: dict[str, Any]) -> str | None:
    route_id = str(task.get("route_id") or "").strip()
    parts = route_id.split(".")
    if len(parts) >= 4:
        return parts[-2] or None
    return None


def _route_method_from_task(task: dict[str, Any]) -> str | None:
    route_id = str(task.get("route_id") or "").strip()
    parts = route_id.split(".")
    if len(parts) >= 4:
        return parts[-1] or None
    return None


def _field_type(site_profile: dict[str, Any], entity_name: str, field_name: str) -> str | None:
    for entity in site_profile.get("data_model", []):
        if not isinstance(entity, dict) or entity.get("entity") != entity_name:
            continue
        for field in entity.get("fields", []):
            if isinstance(field, dict) and field.get("name") == field_name:
                field_type = field.get("type")
                return str(field_type) if isinstance(field_type, str) else None
    return None


def _varchar_limit(field_type: str | None) -> int | None:
    if not isinstance(field_type, str):
        return None
    match = re.search(r"varchar\((\d+)\)", field_type, re.IGNORECASE)
    if match is None:
        return None
    return int(match.group(1))
