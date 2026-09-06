"""Map seed context resolution."""

from __future__ import annotations

from typing import Any

import requests


def _derive_map_seed_context(
    task: dict[str, Any],
    instance: dict[str, Any],
    placeholders: set[str],
) -> dict[str, Any]:
    needs_way = "way_id" in placeholders
    needs_relation = "relation_id" in placeholders
    if not needs_way and not needs_relation:
        return {}

    instantiation = task.get("instantiation_dict")
    place = instantiation.get("place") if isinstance(instantiation, dict) else None
    if not isinstance(place, str) or not place.strip():
        return {}

    site_url = str(instance.get("site_url", "")).rstrip("/")
    if not site_url:
        return {}
    search_url = f"{site_url}/nominatim/search"
    try:
        response = requests.get(
            search_url,
            params={"q": place.strip(), "format": "jsonv2", "limit": 10},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError(f"failed to resolve map seed placeholders for {place!r}: {exc}") from exc

    if not isinstance(payload, list):
        raise RuntimeError(f"map placeholder lookup for {place!r} returned invalid JSON payload")

    context: dict[str, Any] = {}
    if needs_way:
        way = _pick_map_search_result(payload, osm_type="way", place=place)
        if way is not None:
            context["way_id"] = way
    if needs_relation:
        relation = _pick_map_search_result(payload, osm_type="relation", place=place)
        if relation is not None:
            context["relation_id"] = relation
    return context


def _pick_map_search_result(payload: list[Any], *, osm_type: str, place: str) -> Any:
    place_lower = place.lower()
    fallback = None
    for item in payload:
        if not isinstance(item, dict):
            continue
        if str(item.get("osm_type", "")).strip().lower() != osm_type:
            continue
        osm_id = item.get("osm_id")
        if fallback is None:
            fallback = osm_id
        haystack = " ".join(
            str(item.get(key, "")).lower()
            for key in ("display_name", "name")
            if item.get(key) is not None
        )
        if place_lower and place_lower in haystack:
            return osm_id
    return fallback


__all__ = [
    "_derive_map_seed_context",
    "_pick_map_search_result",
]
