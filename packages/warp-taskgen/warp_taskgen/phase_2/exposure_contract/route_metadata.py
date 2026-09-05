"""Exposure route metadata."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _surface_route_metadata(
    *,
    resource: Mapping[str, Any],
    mode: str,
    kind: Any,
    target_surface_id: str | None,
    phase4_exposure: Mapping[str, Any],
) -> dict[str, Any]:
    """Describe why the selected surface is or is not on the agent route.

    This is explanatory metadata only. Eligibility remains owned by
    ``phase4_exposure`` so richer surfaces cannot become admissible just by
    carrying a friendly route label.
    """

    requires_transition = phase4_exposure.get("requires_transition") is True
    transition_forced = phase4_exposure.get("transition_forced_by_task") is True
    exact_comment_forced = phase4_exposure.get("exact_comment_region_forced_by_task") is True
    seeded_comment_visible = phase4_exposure.get("seeded_comment_visible_forced_by_task") is True
    runtime_hook = phase4_exposure.get("requires_runtime_hook") is True
    route: dict[str, Any] = {
        "schema_version": 1,
        "entry_surface": _entry_surface_label(resource, kind),
        "seed_surface": target_surface_id or "unknown",
        "mode": mode,
        "requires_transition": requires_transition,
        "transition_forced_by_task": transition_forced,
        "exact_comment_region_forced_by_task": exact_comment_forced,
        "seeded_comment_visible_forced_by_task": seeded_comment_visible,
        "runtime_hook_required": runtime_hook,
        "route_evidence": _route_evidence_label(
            phase4_exposure,
            requires_transition=requires_transition,
            transition_forced=transition_forced,
            exact_comment_forced=exact_comment_forced,
            seeded_comment_visible=seeded_comment_visible,
        ),
        "content_capacity": _content_capacity_for_surface(target_surface_id),
        "attack_fit": _attack_fit_for_surface(
            target_surface_id,
            phase4_exposure=phase4_exposure,
        ),
    }
    route_variant = _route_variant_label(resource)
    if route_variant is not None:
        route["route_variant"] = route_variant
    return route


def _route_variant_label(resource: Mapping[str, Any]) -> str | None:
    raw = resource.get("route_variant")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def _entry_surface_label(resource: Mapping[str, Any], kind: Any) -> str:
    source_kind = resource.get("source_listing_kind")
    if isinstance(source_kind, str) and source_kind.strip():
        return source_kind.strip()
    return str(kind or "unknown")


def _route_evidence_label(
    phase4_exposure: Mapping[str, Any],
    *,
    requires_transition: bool,
    transition_forced: bool,
    exact_comment_forced: bool,
    seeded_comment_visible: bool,
) -> str:
    encounter_surface = str(phase4_exposure.get("encounter_surface") or "unknown")
    if phase4_exposure.get("admissible") is not True:
        reason = str(phase4_exposure.get("reason") or "inadmissible")
        return f"{encounter_surface}: {reason}"
    if seeded_comment_visible:
        return f"{encounter_surface}: seeded comment visibility forced by task"
    if exact_comment_forced:
        return f"{encounter_surface}: exact discussion region forced by task"
    if requires_transition:
        forced = "forced" if transition_forced else "not forced"
        return f"{encounter_surface}: transition {forced}"
    return f"{encounter_surface}: visible on entry route"


def _content_capacity_for_surface(surface_id: str | None) -> str:
    if not isinstance(surface_id, str) or not surface_id.strip():
        return "unknown"
    suffix = surface_id.rsplit(".", 1)[-1].lower()
    if suffix in {"title", "name"}:
        return "short_title"
    if suffix in {"description", "body", "detail", "sidebar"}:
        return "long_body"
    if suffix in {"note", "comment", "message"}:
        return "comment"
    return "unknown"


def _attack_fit_for_surface(
    surface_id: str | None,
    *,
    phase4_exposure: Mapping[str, Any],
) -> str:
    capacity = _content_capacity_for_surface(surface_id)
    if phase4_exposure.get("admissible") is not True:
        return "not_admissible"
    if capacity == "short_title":
        return "compact_action_required"
    if capacity in {"long_body", "comment"}:
        return "rich_text_action_supported"
    return "unknown"


__all__ = [
    "_attack_fit_for_surface",
    "_content_capacity_for_surface",
    "_entry_surface_label",
    "_route_evidence_label",
    "_route_variant_label",
    "_surface_route_metadata",
]
