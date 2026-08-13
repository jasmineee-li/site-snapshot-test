"""Seed contract validation helpers.

The functions here validate that generated or finalized seeds stay inside the
admitted editor method, surface, path, and payload field selected by Phase 2.
"""

from __future__ import annotations

import json
import re
from typing import Any

from warp_taskgen.seed_contracts.delivery import (
    PAYLOAD_PLACEHOLDER,
    _call_body_field_value,
    _call_body_key,
    _call_delivery_path,
    _call_satisfies_path_param,
    _call_site,
    _contains_deferred_map_target,
    _has_conflicting_nested_review_body,
    _seed_calls,
)
from warp_taskgen.seed_contracts.surface import (
    _entry_matches_attack_write,
    _extract_attack_write,
    _extract_target_field_values,
    _find_surface_by_id,
    _format_delivery_channels,
    _seed_matches_surface_channel,
)
from warp_taskgen.seeding import self_contained_adversarial_seed_error

_UNRESOLVED_HTTP_TEMPLATE_TOKEN = re.compile(r"(?<![${])\{[A-Za-z_]\w*\}(?!\})")
_ELIGIBLE_CONTROLLABLE_TIERS = frozenset({"any_user", "authed_user"})


def _validate_finalized_http_seed_contract(
    seed: Any,
    delivery_channel: Any,
    *,
    sites: Any,
) -> str | None:
    if not isinstance(seed, dict):
        return None
    if not isinstance(delivery_channel, dict):
        return None
    if not _seed_calls(seed):
        return None

    if _contains_deferred_map_target(seed):
        return "target-based map seeds must be quarantined instead of validated for execution"

    unresolved = _find_unresolved_http_seed_reference(seed, delivery_channel)
    if unresolved is not None:
        return unresolved

    return None


def _find_unresolved_http_seed_reference(
    seed: dict[str, Any], delivery_channel: dict[str, Any]
) -> str | None:
    calls = _seed_calls(seed)
    if not calls:
        return None

    required_body_field = delivery_channel.get("body_field")
    for index, call in enumerate(calls):
        if not isinstance(call, dict):
            continue
        path = _call_delivery_path(call)
        if (
            isinstance(path, str)
            and not isinstance(call.get("args"), dict)
            and _UNRESOLVED_HTTP_TEMPLATE_TOKEN.search(path)
        ):
            return f"adversarial_data_seed api_calls[{index}].path contains unresolved placeholders"
        if _has_conflicting_nested_review_body(call, _call_body_key(seed, call)):
            return (
                f"adversarial_data_seed api_calls[{index}] mixes top-level review fields with "
                "body.review; use exactly one shopping review body shape"
            )

        if isinstance(required_body_field, str):
            value = _call_body_field_value(call, _call_body_key(seed, call), required_body_field)
            if isinstance(value, str) and _UNRESOLVED_HTTP_TEMPLATE_TOKEN.search(value):
                return (
                    "adversarial_data_seed contains unresolved placeholders in the required "
                    f"body field {required_body_field!r}"
                )

    postcondition = delivery_channel.get("postcondition")
    if not isinstance(postcondition, dict) or postcondition.get("type") != "db_row_value":
        return None
    where = postcondition.get("where")
    if not isinstance(where, dict) or not where:
        return "delivery_channel.postcondition.where must be a non-empty object"
    for column_name, source in where.items():
        if not isinstance(source, dict) or len(source) != 1:
            return f"delivery_channel.postcondition.where[{column_name!r}] must declare exactly one source"
        source_key, source_value = next(iter(source.items()))
        if source_key == "literal" and source_value in ("", None, -1):
            return (
                "delivery_channel.postcondition.where contains a placeholder literal and must "
                "identify the created row concretely"
            )
        if source_key == "body_field":
            if not isinstance(source_value, str) or not source_value.strip():
                return f"delivery_channel.postcondition.where[{column_name!r}] body_field must be non-empty"
            if all(
                not isinstance(call, dict)
                or _call_body_field_value(call, _call_body_key(seed, call), source_value) is None
                for call in calls
            ):
                return (
                    f"delivery_channel.postcondition.where[{column_name!r}] references missing "
                    f"body_field {source_value!r}"
                )
        if source_key == "path_param":
            if not isinstance(source_value, str) or not source_value.strip():
                return f"delivery_channel.postcondition.where[{column_name!r}] path_param must be non-empty"
            if all(
                not isinstance(call, dict) or not _call_satisfies_path_param(call, source_value)
                for call in calls
            ):
                return (
                    f"delivery_channel.postcondition.where[{column_name!r}] path_param {source_value!r} "
                    "is unresolved in adversarial_data_seed"
                )
    return None


def _site_profile_supports_attack(
    site_profile: dict[str, Any],
    target_surface_id: str,
    concealment: str,
    delivery_mechanism: str,
) -> bool:
    surface = _find_surface_by_id(site_profile, target_surface_id)
    if surface is None:
        return False
    if surface.get("controllable_by_tier") not in _ELIGIBLE_CONTROLLABLE_TIERS:
        return False
    concealments = surface.get("compatible_concealments")
    deliveries = surface.get("delivery_channels")
    if not isinstance(concealments, list) or not isinstance(deliveries, list):
        return False
    if concealment not in concealments:
        return False
    for entry in deliveries:
        if (
            isinstance(entry, dict)
            and entry.get("mechanism") == delivery_mechanism
            and entry.get("privileged_seed") is False
        ):
            return True
    return False


def _resolve_delivery_channel(
    site_profile: dict[str, Any],
    *,
    target_surface_id: str,
    delivery_mechanism: str,
    seed_template: Any,
) -> dict[str, Any]:
    surface = _find_surface_by_id(site_profile, target_surface_id)
    if surface is None:
        raise ValueError(f"target_surface_id {target_surface_id!r} not found in site profile")
    attack_write = _extract_attack_write(seed_template)
    if attack_write is None:
        raise ValueError(f"seed payload must contain exactly one {PAYLOAD_PLACEHOLDER} placeholder")
    matches: list[dict[str, Any]] = []
    for entry in surface.get("delivery_channels", []):
        if not isinstance(entry, dict) or entry.get("privileged_seed") is not False:
            continue
        if entry.get("mechanism") != delivery_mechanism:
            continue
        if not _entry_matches_attack_write(entry, attack_write):
            continue
        matches.append(json.loads(json.dumps(entry)))
    if len(matches) != 1:
        available = _format_delivery_channels(surface)
        raise ValueError(
            f"seed payload must resolve to exactly one non-privileged delivery_channel for "
            f"{target_surface_id!r}; available: {available}"
        )
    return matches[0]


def _validate_discriminating_payload(
    benign_seed: Any,
    adversarial_seed: Any,
    surface: dict[str, Any] | None = None,
) -> str | None:
    if not isinstance(adversarial_seed, dict):
        return None
    if not isinstance(benign_seed, dict):
        return None

    if surface is not None:
        adversarial_mechanism = adversarial_seed.get("mechanism")
        benign_values = _extract_target_field_values(benign_seed, surface)
        adversarial_values = _extract_target_field_values(adversarial_seed, surface)
        if (
            adversarial_mechanism in {"api", "form", "upload"}
            and _seed_matches_surface_channel(adversarial_seed, surface)
            and not adversarial_values
        ):
            return "adversarial_data_seed must set the declared target field on the target surface"
        if adversarial_values and benign_values == adversarial_values:
            return "adversarial_data_seed target field values must differ from benign data_seed beyond whitespace/case"

    if (
        json.dumps(benign_seed, sort_keys=True).lower()
        == json.dumps(adversarial_seed, sort_keys=True).lower()
    ):
        return "adversarial_data_seed must differ from the benign data_seed"

    return None


def _validate_self_contained_adversarial_seed(
    benign_seed: Any, adversarial_seed: Any
) -> str | None:
    """Require the adversarial seed to preserve the benign seed verbatim before extending it."""
    return self_contained_adversarial_seed_error(benign_seed, adversarial_seed)


def _validate_editor_seed_sites(
    seed: Any, *, expected_site: str, field_name: str = "adversarial_data_seed"
) -> str | None:
    if not isinstance(seed, dict) or not expected_site:
        return None
    for index, call in enumerate(_seed_calls(seed)):
        if not isinstance(call, dict) or not isinstance(call.get("args"), dict):
            continue
        site_name = _call_site(call)
        if site_name and site_name != expected_site:
            return (
                f"{field_name} editor_calls[{index}].site {site_name!r} "
                f"must match delivery site {expected_site!r}"
            )
    return None
