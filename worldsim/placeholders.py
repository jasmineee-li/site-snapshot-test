"""Helpers for WebArena-style URL placeholder handling."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

PLACEHOLDER_TO_SITE: dict[str, str] = {
    "__SHOPPING__": "shopping",
    "__SHOPPING_ADMIN__": "shopping_admin",
    "__GITLAB__": "gitlab",
    "__REDDIT__": "reddit",
    "__WIKIPEDIA__": "wikipedia",
    "__MAP__": "map",
}

SITE_TO_PLACEHOLDER: dict[str, str] = {
    site_name: placeholder
    for placeholder, site_name in PLACEHOLDER_TO_SITE.items()
}

PLACEHOLDER_PATTERN = re.compile(r"__[A-Z0-9_]+__")


def normalize_site_name(site_name: str) -> str:
    """Normalize a benchmark site identifier for matching."""
    return str(site_name).strip().lower()


def placeholder_for_site(site_name: str) -> str | None:
    """Return the canonical placeholder token for *site_name* if known."""
    return SITE_TO_PLACEHOLDER.get(normalize_site_name(site_name))


def normalize_task_sites(task: dict[str, Any]) -> list[str]:
    """Return the normalized list of sites a task requires."""
    raw_sites = task.get("sites") or []
    sites: list[str] = []
    for raw in raw_sites:
        normalized = normalize_site_name(raw)
        if normalized and normalized not in sites:
            sites.append(normalized)

    primary = normalize_site_name(task.get("site", ""))
    if primary and primary not in sites:
        sites.insert(0, primary)

    return sites


def extract_placeholders(value: str) -> list[str]:
    """Return placeholder tokens present in *value* in first-seen order."""
    seen: set[str] = set()
    matches: list[str] = []
    for token in PLACEHOLDER_PATTERN.findall(value):
        if token not in seen:
            seen.add(token)
            matches.append(token)
    return matches


def apply_placeholders(
    value: str,
    placeholders: dict[str, str],
    *,
    strict: bool = True,
) -> str:
    """Apply URL placeholders to *value*.

    When ``strict`` is true, unresolved placeholder tokens raise ``ValueError``.
    """
    resolved = value
    for token, replacement in placeholders.items():
        resolved = resolved.replace(token, replacement)

    unresolved = extract_placeholders(resolved)
    if strict and unresolved:
        raise ValueError(
            "Unresolved URL placeholders: " + ", ".join(unresolved)
        )
    return resolved


def merge_placeholder_maps(*maps: dict[str, str] | None) -> dict[str, str]:
    """Merge placeholder maps, keeping the last non-empty value for each key."""
    merged: dict[str, str] = {}
    for mapping in maps:
        if not mapping:
            continue
        for token, value in mapping.items():
            if value:
                merged[str(token)] = str(value)
    return merged


def placeholders_for_site_urls(
    site_urls: Iterable[tuple[str, str]],
) -> dict[str, str]:
    """Build a placeholder map from ``(site_name, site_url)`` pairs."""
    placeholders: dict[str, str] = {}
    for site_name, site_url in site_urls:
        token = placeholder_for_site(site_name)
        if token and site_url:
            placeholders[token] = site_url
    return placeholders
