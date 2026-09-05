"""Seed delivery surface matching helpers."""

from __future__ import annotations

from typing import Any

from warp_taskgen.seed_contracts.delivery import (
    _REDDIT_COMMENT_BODY_FIELD_PATTERN,
    PAYLOAD_PLACEHOLDER,
    _call_body_field_entries,
    _call_body_field_value,
    _call_body_fields,
    _call_body_key,
    _call_delivery_mechanism,
    _call_delivery_path,
    _call_matches_delivery_entry,
    _call_method,
    _call_site,
    _editor_delivery_key,
    _normalize_delivery_path,
    _normalize_payload_value,
    _seed_calls,
)


def _find_surface_by_id(
    site_profile: dict[str, Any], target_surface_id: str
) -> dict[str, Any] | None:
    # Lazy: ``warp_taskgen.sites`` imports ``seeding.site_contracts``.
    from warp_taskgen.sites import SiteCarrierPolicy, SiteTargetingDefinitionError, default_catalog

    def carrier_policy(site_key: str) -> SiteCarrierPolicy:
        try:
            return default_catalog().bind(site=site_key).carrier_policy()
        except SiteTargetingDefinitionError:
            return SiteCarrierPolicy.closed("webarena_verified")

    site = str(site_profile.get("site") or site_profile.get("site_name") or "").strip().lower()
    sites = (site,) if site else default_catalog().sites
    policies = tuple(carrier_policy(site_key) for site_key in sites)
    canonical_targets = {policy.canonical_surface(target_surface_id) for policy in policies}
    for surface in site_profile.get("injection_surface", []):
        if not isinstance(surface, dict):
            continue
        surface_id = surface.get("id")
        if surface_id == target_surface_id:
            return surface
        if any(
            policy.canonical_surface(str(surface_id or "")) in canonical_targets
            for policy in policies
        ):
            return surface
    return None


def _format_delivery_channels(surface: dict[str, Any]) -> list[str]:
    formatted: list[str] = []
    for entry in surface.get("delivery_channels", []):
        if not isinstance(entry, dict):
            continue
        mechanism = entry.get("mechanism")
        formatted.append(
            f"{mechanism} {entry.get('method')} {_normalize_delivery_path(str(entry.get('path_template', '')))} field:{entry.get('body_field')}"
        )
    return formatted


def _extract_attack_write(seed: Any) -> dict[str, Any] | None:
    if not isinstance(seed, dict):
        return None
    calls = _seed_calls(seed)
    if not calls:
        return None
    matches: list[dict[str, Any]] = []
    for call in calls:
        path = _call_delivery_path(call)
        method = _call_method(call)
        call_mechanism = _call_delivery_mechanism(seed, call)
        if (
            not isinstance(path, str)
            or not isinstance(method, str)
            or not isinstance(call_mechanism, str)
        ):
            continue
        body_key = _call_body_key(seed, call)
        body = call.get(body_key)
        if not isinstance(body, dict):
            continue
        field_entries = _call_body_field_entries(call, body_key)
        field_values = _call_body_fields(call, body_key)
        placeholder_fields_by_source: dict[str, set[str]] = {}
        placeholder_counts_by_source: dict[str, int] = {}
        for field_name, value, source_name in field_entries:
            if isinstance(value, str) and PAYLOAD_PLACEHOLDER in value:
                placeholder_fields_by_source.setdefault(source_name, set()).add(field_name)
                placeholder_counts_by_source.setdefault(
                    source_name, value.count(PAYLOAD_PLACEHOLDER)
                )
        placeholder_fields: set[str] = set()
        for _source_name, field_names in placeholder_fields_by_source.items():
            placeholder_fields.update(field_names)
        placeholder_count = sum(placeholder_counts_by_source.values())
        if placeholder_count <= 0:
            continue
        matches.append(
            {
                "editor_key": _editor_delivery_key(call),
                "mechanism": call_mechanism,
                "resource": f"path:{method.strip().upper()} {_normalize_delivery_path(path)}",
                "fields": set(field_values.keys()),
                "placeholder_fields": placeholder_fields,
                "placeholder_count": placeholder_count,
            }
        )
    return matches[0] if len(matches) == 1 else None


def _entry_matches_attack_write(entry: dict[str, Any], attack_write: dict[str, Any]) -> bool:
    mechanism = entry.get("mechanism")
    if mechanism != attack_write.get("mechanism"):
        return False
    path_template = entry.get("path_template")
    method = entry.get("method")
    body_field = entry.get("body_field")
    if (
        not isinstance(path_template, str)
        or not isinstance(method, str)
        or not isinstance(body_field, str)
    ):
        return False
    expected_resource = f"path:{method.strip().upper()} {_normalize_delivery_path(path_template)}"
    attack_resource = attack_write.get("resource")
    editor_key = attack_write.get("editor_key")
    if attack_resource != expected_resource and not _editor_write_matches_profile_resource(
        editor_key if isinstance(editor_key, tuple) else None,
        attack_resource,
        expected_resource,
    ):
        return False
    placeholder_fields = attack_write.get("placeholder_fields")
    if editor_key == ("reddit", "create_submission_title") and "{submission_id}" in path_template:
        return False
    return isinstance(placeholder_fields, set) and _body_field_matches_placeholder(
        body_field,
        placeholder_fields,
        editor_key=editor_key if isinstance(editor_key, tuple) else None,
    )


def _editor_write_matches_profile_resource(
    editor_key: tuple[str, str] | None,
    attack_resource: Any,
    expected_resource: str,
) -> bool:
    """Bridge editor-method paths to live profile form paths.

    Postmill exposes the create-submission form at ``/submit`` while the editor
    contract carries the forum selector as ``/submit/{forum_name}``. They are
    the same write surface; edit-submission paths still require the concrete
    edit route and are not matched here.
    """
    if editor_key in {("reddit", "create_submission"), ("reddit", "create_submission_title")}:
        return (
            attack_resource == "path:POST /submit/{id}" and expected_resource == "path:POST /submit"
        )
    return False


def _body_field_matches_placeholder(
    body_field: str,
    placeholder_fields: set[str],
    *,
    editor_key: tuple[str, str] | None,
) -> bool:
    if body_field in placeholder_fields:
        return True
    if editor_key == ("reddit", "create_submission"):
        return body_field == "submission[body]" and "body" in placeholder_fields
    if editor_key == ("reddit", "create_submission_title"):
        return body_field == "submission[title]" and "title" in placeholder_fields
    if editor_key == ("reddit", "create_comment"):
        return bool(
            "body" in placeholder_fields and _REDDIT_COMMENT_BODY_FIELD_PATTERN.match(body_field)
        )
    return False


def _extract_target_field_values(
    seed: Any,
    surface: dict[str, Any],
) -> list[str]:
    if not isinstance(seed, dict):
        return []
    calls = _seed_calls(seed)
    if not calls:
        return []
    values: list[str] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        mechanism = _call_delivery_mechanism(seed, call)
        if mechanism not in {"api", "form", "upload"}:
            continue
        for entry in surface.get("delivery_channels", []):
            if not _call_matches_delivery_entry(call, mechanism=mechanism, entry=entry):
                continue
            body_field = entry.get("body_field")
            if not isinstance(body_field, str):
                continue
            body_key = _call_body_key(seed, call)
            value = _call_body_field_value(call, body_key, body_field)
            if value is not None:
                values.append(_normalize_payload_value(value))
    return values


def _seed_matches_surface_channel(seed: Any, surface: dict[str, Any]) -> bool:
    for write in _extract_seed_writes(seed):
        if _surface_matches_write(surface, write):
            return True
    return False


def _extract_seed_writes(seed: Any) -> list[dict[str, Any]]:
    if not isinstance(seed, dict):
        return []
    calls = _seed_calls(seed)
    if not calls:
        return []
    writes: list[dict[str, Any]] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        path = _call_delivery_path(call)
        method = _call_method(call)
        mechanism = _call_delivery_mechanism(seed, call)
        if (
            not isinstance(path, str)
            or not isinstance(method, str)
            or not isinstance(mechanism, str)
        ):
            continue
        body_key = _call_body_key(seed, call)
        fields: set[str] = set()
        fields.update(_call_body_fields(call, body_key).keys())
        writes.append(
            {
                "site": _call_site(call),
                "mechanism": mechanism,
                "resource": f"path:{method.strip().upper()} {_normalize_delivery_path(path)}",
                "fields": fields,
                "field_mode": "contains" if isinstance(call.get("args"), dict) else "exact",
            }
        )
    return writes


def _surface_matches_write(surface: dict[str, Any], write: dict[str, Any]) -> bool:
    for entry in surface.get("delivery_channels", []):
        if not isinstance(entry, dict) or entry.get("privileged_seed") is not False:
            continue
        delivery_site = entry.get("delivery_site")
        write_site = write.get("site")
        if isinstance(delivery_site, str) and delivery_site.strip():
            if write_site not in (None, "", delivery_site.strip()):
                continue
        mechanism = entry.get("mechanism")
        if mechanism != write.get("mechanism"):
            continue
        path_template = entry.get("path_template")
        method = entry.get("method")
        body_field = entry.get("body_field")
        if (
            not isinstance(path_template, str)
            or not isinstance(method, str)
            or not isinstance(body_field, str)
        ):
            continue
        if (
            write.get("resource")
            != f"path:{method.strip().upper()} {_normalize_delivery_path(path_template)}"
        ):
            continue
        fields = write.get("fields")
        if isinstance(fields, set):
            field_mode = write.get("field_mode")
            if field_mode == "contains" and body_field in fields:
                return True
            if field_mode != "contains" and fields == {body_field}:
                return True
    return False
