"""Helpers for loading and validating Phase 0c site profiles."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_and_validate_profile(
    site_name: str,
    profile_path: Path,
    *,
    manifest_eval_types: Iterable[str] = (),
) -> dict[str, Any]:
    """Load a site profile from disk and validate its internal references."""
    if not profile_path.exists():
        raise ValueError(f"{site_name}: missing {profile_path}")

    try:
        profile = json.loads(profile_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{site_name}: invalid profile JSON at {profile_path}: {exc}") from exc

    validate_profile(site_name, profile, manifest_eval_types=manifest_eval_types)
    return profile


def validate_profile(
    site_name: str,
    profile: dict[str, Any],
    *,
    manifest_eval_types: Iterable[str] = (),
) -> None:
    """Validate cross-references within a site profile."""
    # site_name mismatch: catch profiles that claim to be a different site
    profile_site = profile.get("site_name")
    if profile_site and profile_site != site_name:
        raise ValueError(
            f"Profile site_name mismatch: expected {site_name!r}, got {profile_site!r}"
        )

    entity_fields = _entity_field_index(profile.get("data_model"))

    errors: list[str] = []
    for surface in profile.get("injection_surface", []):
        source = surface.get("source_field", "")
        if source and "." in source:
            entity_name, _, field_name = source.partition(".")
            if entity_name not in entity_fields and entity_fields:
                errors.append(
                    f"injection surface {surface.get('id', '?')!r} references "
                    f"unknown entity {entity_name!r} in {source!r}"
                )
            elif entity_fields and field_name not in entity_fields.get(entity_name, set()):
                errors.append(
                    f"injection surface {surface.get('id', '?')!r} references "
                    f"unknown field {source!r}"
                )

    known_eval_types = {
        cap.get("eval_type", "")
        for cap in profile.get("verification_capabilities", [])
        if cap.get("eval_type")
    }
    manifest_eval_type_set = set(manifest_eval_types)
    if manifest_eval_type_set:
        missing_eval_types = sorted(known_eval_types - manifest_eval_type_set)
        if missing_eval_types:
            errors.append(
                "verification capabilities reference eval types absent from manifest: "
                + ", ".join(missing_eval_types)
            )

    if errors:
        raise ValueError(
            f"Profile {site_name} failed validation:\n"
            + "\n".join(f"  - {error}" for error in errors)
        )

    logger.info(
        "Profile %s validated: %d data model entities, %d injection surfaces, %d eval types",
        site_name,
        len(profile.get("data_model", [])),
        len(profile.get("injection_surface", [])),
        len(known_eval_types),
    )


def _entity_field_index(data_model: object) -> dict[str, set[str]]:
    """Return {entity_name: {field_name}} for a data model payload."""
    index: dict[str, set[str]] = {}
    if not isinstance(data_model, list):
        return index

    for entity in data_model:
        if not isinstance(entity, dict):
            continue
        entity_name = entity.get("entity")
        if not isinstance(entity_name, str) or not entity_name:
            continue
        fields = index.setdefault(entity_name, set())
        for field in entity.get("fields", []):
            if not isinstance(field, dict):
                continue
            field_name = field.get("name")
            if isinstance(field_name, str) and field_name:
                fields.add(field_name)
    return index
