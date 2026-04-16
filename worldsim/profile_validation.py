"""Helpers for loading and validating Phase 0c site profiles."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CONTROLLABLE_TIERS = frozenset({"anon", "any_user", "authed_user", "admin", "none"})
_DELIVERY_MECHANISMS = frozenset({"api", "form", "upload"})
_RENDERING_FORMATS = frozenset({"raw_html", "markdown", "plaintext"})
_CONCEALMENTS = frozenset(
    {"plaintext", "offscreen_css", "markdown_fenced_system", "image_alt_text"}
)
_POSTCONDITION_TYPES = frozenset({"db_row_value"})
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
_PATH_PARAM_PATTERN = re.compile(r"\{([^}/]+)\}")


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
        errors.extend(_validate_injection_surface_labels(surface))

    errors.extend(
        _validate_agent_context_tier_consistency(
            profile.get("injection_surface"),
            profile.get("agent_context"),
        )
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


def profile_requires_agent_auth(profile: dict[str, Any]) -> bool:
    surfaces = profile.get("injection_surface")
    if not isinstance(surfaces, list):
        return False
    return any(
        isinstance(surface, dict) and surface.get("controllable_by_tier") == "authed_user"
        for surface in surfaces
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


def _validate_concealment_compatibility(
    rendering_format: object,
    concealments: object,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(concealments, list) or not concealments:
        return ["compatible_concealments must be a non-empty array"]

    for concealment in concealments:
        if concealment not in _CONCEALMENTS:
            errors.append(
                f"compatible_concealments contains invalid value {concealment!r}; "
                f"allowed: {sorted(_CONCEALMENTS)}"
            )
            continue
        if concealment == "markdown_fenced_system" and rendering_format != "markdown":
            errors.append(
                "markdown_fenced_system is only compatible with rendering_format='markdown'"
            )
        if concealment == "offscreen_css" and rendering_format != "raw_html":
            errors.append("offscreen_css is only compatible with rendering_format='raw_html'")

    return errors


def _validate_delivery_channel_entry(entry: object, index: int) -> list[str]:
    prefix = f"delivery_channels[{index}]"
    if not isinstance(entry, dict):
        return [f"{prefix} must be an object"]

    errors: list[str] = []
    mechanism = entry.get("mechanism")
    if mechanism not in _DELIVERY_MECHANISMS:
        errors.append(
            f"{prefix} mechanism must be one of {sorted(_DELIVERY_MECHANISMS)}, got {mechanism!r}"
        )

    if not isinstance(entry.get("privileged_seed"), bool):
        errors.append(f"{prefix} privileged_seed must be a boolean")

    path_template = entry.get("path_template")
    method = entry.get("method")
    body_field = entry.get("body_field")
    table = entry.get("table")
    column = entry.get("column")
    postcondition = entry.get("postcondition")

    if mechanism in {"api", "form", "upload"}:
        if not isinstance(path_template, str) or not path_template.startswith("/"):
            errors.append(f"{prefix} path_template must be a path string starting with '/'")
        if not isinstance(method, str) or not method.strip():
            errors.append(f"{prefix} method must be a non-empty string")
        if not isinstance(body_field, str) or not body_field.strip():
            errors.append(f"{prefix} body_field must be a non-empty string")
        if table is not None:
            errors.append(f"{prefix} table must be null when mechanism={mechanism!r}")
        if column is not None:
            errors.append(f"{prefix} column must be null when mechanism={mechanism!r}")
        errors.extend(
            _validate_delivery_channel_postcondition(
                postcondition,
                prefix=prefix,
                body_field=body_field,
                path_template=path_template,
                required=True,
            )
        )

    # Optional live-verification fields. Accepted but never required.
    verified = entry.get("verified")
    if verified is not None and not isinstance(verified, bool):
        errors.append(f"{prefix} verified must be true, false, or null")
    verification_notes = entry.get("verification_notes")
    if verification_notes is not None and not isinstance(verification_notes, str):
        errors.append(f"{prefix} verification_notes must be a string or null")

    return errors


def _validate_delivery_channel_postcondition(
    postcondition: object,
    *,
    prefix: str,
    body_field: object,
    path_template: object,
    required: bool,
) -> list[str]:
    if postcondition is None:
        if required:
            return [f"{prefix} postcondition is required for non-sql delivery channels"]
        return []
    if not isinstance(postcondition, dict):
        return [f"{prefix} postcondition must be an object"]

    errors: list[str] = []
    postcondition_type = postcondition.get("type")
    if postcondition_type not in _POSTCONDITION_TYPES:
        errors.append(
            f"{prefix} postcondition.type must be one of {sorted(_POSTCONDITION_TYPES)}, "
            f"got {postcondition_type!r}"
        )
        return errors

    table = postcondition.get("table")
    if not isinstance(table, str) or not _IDENTIFIER_PATTERN.match(table):
        errors.append(f"{prefix} postcondition.table must be a SQL identifier")

    value_column = postcondition.get("value_column")
    if not isinstance(value_column, str) or not _IDENTIFIER_PATTERN.match(value_column):
        errors.append(f"{prefix} postcondition.value_column must be a SQL identifier")

    where = postcondition.get("where")
    if not isinstance(where, dict) or not where:
        errors.append(f"{prefix} postcondition.where must be a non-empty object")
        return errors

    path_params = (
        {match.group(1) for match in _PATH_PARAM_PATTERN.finditer(path_template)}
        if isinstance(path_template, str)
        else set()
    )

    for column_name, source in where.items():
        if not isinstance(column_name, str) or not _IDENTIFIER_PATTERN.match(column_name):
            errors.append(f"{prefix} postcondition.where keys must be SQL identifiers")
            continue
        if not isinstance(source, dict) or len(source) != 1:
            errors.append(
                f"{prefix} postcondition.where[{column_name!r}] must declare exactly one source"
            )
            continue
        source_key, source_value = next(iter(source.items()))
        if source_key == "path_param":
            if not isinstance(source_value, str) or source_value not in path_params:
                errors.append(
                    f"{prefix} postcondition.where[{column_name!r}] path_param must match "
                    "a placeholder in path_template"
                )
        elif source_key == "body_field":
            if not isinstance(source_value, str) or not source_value.strip():
                errors.append(
                    f"{prefix} postcondition.where[{column_name!r}] body_field must be a non-empty string"
                )
            elif isinstance(body_field, str) and source_value == body_field:
                continue
        elif source_key == "literal":
            if not isinstance(source_value, (str, int, float, bool)) and source_value is not None:
                errors.append(
                    f"{prefix} postcondition.where[{column_name!r}] literal must be a JSON scalar"
                )
        else:
            errors.append(
                f"{prefix} postcondition.where[{column_name!r}] source must be one of "
                "path_param, body_field, literal"
            )
    return errors


def _validate_injection_surface_labels(surface: object) -> list[str]:
    if not isinstance(surface, dict):
        return ["injection surface entry must be an object"]

    errors: list[str] = []
    prefix = f"injection surface {surface.get('id', '?')!r}"
    rendering_format = surface.get("rendering_format")
    if rendering_format not in _RENDERING_FORMATS:
        errors.append(
            f"{prefix} has invalid rendering_format {rendering_format!r}; "
            f"allowed: {sorted(_RENDERING_FORMATS)}"
        )

    tier = surface.get("controllable_by_tier")
    if tier not in _CONTROLLABLE_TIERS:
        errors.append(
            f"{prefix} has invalid controllable_by_tier {tier!r}; "
            f"allowed: {sorted(_CONTROLLABLE_TIERS)}"
        )

    justification = surface.get("controllability_justification")
    if not isinstance(justification, str) or not justification.strip():
        errors.append(f"{prefix} missing non-empty controllability_justification")

    delivery_channels = surface.get("delivery_channels")
    if not isinstance(delivery_channels, list) or not delivery_channels:
        errors.append(f"{prefix} must declare non-empty delivery_channels")
    else:
        for index, channel in enumerate(delivery_channels):
            errors.extend(
                f"{prefix} {error}" for error in _validate_delivery_channel_entry(channel, index)
            )

    errors.extend(
        f"{prefix} {error}"
        for error in _validate_concealment_compatibility(
            rendering_format,
            surface.get("compatible_concealments"),
        )
    )
    return errors


def _validate_agent_context_tier_consistency(
    surfaces: object,
    agent_context: object,
) -> list[str]:
    if not isinstance(surfaces, list):
        return []
    requires_authenticated_context = any(
        isinstance(surface, dict) and surface.get("controllable_by_tier") == "authed_user"
        for surface in surfaces
    )
    if not requires_authenticated_context:
        return []
    if not isinstance(agent_context, dict):
        return ["injection surfaces with controllable_by_tier='authed_user' require agent_context"]

    # Accept if auth_mechanism is present in agent_context OR if the instance
    # provides agent_auth (the static, instances.json-driven path).
    auth_mechanism = agent_context.get("auth_mechanism")
    if not isinstance(auth_mechanism, dict):
        return []
    auth_type = auth_mechanism.get("type")
    if not isinstance(auth_type, str) or not auth_type.strip():
        return []
    if auth_type != "none":
        return []

    errors: list[str] = []
    for index, surface in enumerate(surfaces):
        if not isinstance(surface, dict):
            continue
        if surface.get("controllable_by_tier") == "authed_user":
            errors.append(
                f"injection_surface[{index}] controllable_by_tier='authed_user' requires "
                "agent_context auth_mechanism.type != 'none'"
            )
    return errors
