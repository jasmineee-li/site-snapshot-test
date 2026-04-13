"""Helpers for loading and validating Phase 0c site profiles."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

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
            f"Profile site_name mismatch: expected {site_name!r}, "
            f"got {profile_site!r}"
        )

    known_fields: set[str] = set()
    for entity in profile.get("data_model", []):
        for field in entity.get("fields", []):
            known_fields.add(field.get("name", ""))
        storage = entity.get("storage", "")
        if storage:
            known_fields.add(storage)

    known_entities = {entity.get("entity", "") for entity in profile.get("data_model", [])}

    errors: list[str] = []
    for surface in profile.get("injection_surface", []):
        source = surface.get("source_field", "")
        if source and "." in source:
            entity_name = source.split(".")[0]
            if entity_name not in known_entities and known_entities:
                errors.append(
                    f"injection surface {surface.get('id', '?')!r} references "
                    f"unknown entity {entity_name!r} in {source!r}"
                )
            field_name = source.split(".")[-1]
            if field_name not in known_fields and known_fields:
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
