"""Exposure candidate selection."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from warp_taskgen.editors._registry import EditorMethodSpec
from warp_taskgen.phase_2.exposure_contract.editor_args import _editor_args_template
from warp_taskgen.phase_2.exposure_contract.modes import _effective_mode_for_seeded_surface
from warp_taskgen.phase_2.exposure_contract.phase4_exposure import (
    _eligibility_from_capabilities,
    _phase4_exposure_capability,
    _phase4_exposure_for_seeded_surface,
)
from warp_taskgen.phase_2.exposure_contract.route_metadata import _surface_route_metadata
from warp_taskgen.phase_2.exposure_contract.verification import _verification_contract
from warp_taskgen.phases.phase_2_core_surfaces import (
    active_carrier_ineligible_reason,
    canonical_core_surface,
    is_active_carrier_surface,
    is_core_surface,
)


def _contract_id(site: str, benign_task_id: str, kind: Any, anchors: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(
            {"site": site, "benign_task_id": benign_task_id, "kind": kind, "anchors": anchors},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:10]
    return f"{site}:{benign_task_id}:{digest}"


def _surface_candidate(
    *,
    resource: Mapping[str, Any],
    base_mode: str,
    benign_task_id: str,
    benign_read_url: str,
    kind: str,
    site: str,
    available: frozenset[str],
    spec: EditorMethodSpec,
) -> dict[str, Any] | None:
    template = _editor_args_template(
        spec,
        available=available,
        benign_task_id=benign_task_id,
    )
    if template is None:
        return None
    payload_arg = template.pop("__payload_arg__")
    editor_surface_id = spec.surface_id_per_kind.get(kind, spec.method)
    target_surface_id = canonical_core_surface(site, editor_surface_id)
    effective_mode = _effective_mode_for_seeded_surface(
        base_mode=base_mode,
        site=site,
        kind=kind,
        editor_method=spec.method,
        target_surface_id=target_surface_id,
    )
    if not is_core_surface(site, target_surface_id):
        phase4_exposure = _phase4_exposure_capability(
            "ineligible",
            reason="non_core_surface",
        )
        seed_capability: dict[str, Any] = {
            "status": "unsupported",
            "reason": "non_core_surface",
        }
        eligibility = {
            "status": "ineligible",
            "reason": "non_core_surface",
        }
        route = _surface_route_metadata(
            resource=resource,
            mode=effective_mode,
            kind=kind,
            target_surface_id=target_surface_id or editor_surface_id,
            phase4_exposure=phase4_exposure,
        )
        candidate: dict[str, Any] = {
            "mode": effective_mode,
            "editor_method": spec.method,
            "target_surface_id": target_surface_id or editor_surface_id,
            "editor_surface_id": editor_surface_id,
            "required_tokens": sorted(available),
            "seed_capability": seed_capability,
            "phase4_exposure": phase4_exposure,
            "surface_route": route,
            "eligibility": eligibility,
        }
        return candidate

    seed_capability = _seed_capability(spec, payload_arg, target_surface_id)
    if not is_active_carrier_surface(
        site,
        target_surface_id,
        kind=kind,
        method=spec.method,
    ):
        reason = (
            active_carrier_ineligible_reason(
                site,
                target_surface_id,
                kind=kind,
                method=spec.method,
            )
            or "inactive_carrier_surface"
        )
        phase4_exposure = _phase4_exposure_capability(
            "ineligible",
            reason=reason,
        )
        return {
            "mode": effective_mode,
            "editor_method": spec.method,
            "target_surface_id": target_surface_id,
            "editor_surface_id": editor_surface_id,
            "payload_arg": payload_arg,
            "editor_args_template": template,
            "required_tokens": sorted(available),
            "seed_capability": seed_capability,
            "phase4_exposure": phase4_exposure,
            "surface_route": _surface_route_metadata(
                resource=resource,
                mode=effective_mode,
                kind=kind,
                target_surface_id=target_surface_id,
                phase4_exposure=phase4_exposure,
            ),
            "eligibility": _eligibility_from_capabilities(seed_capability, phase4_exposure),
        }

    verification = _verification_contract(resource, effective_mode, benign_read_url, kind)
    phase4_exposure = _phase4_exposure_for_seeded_surface(
        mode=effective_mode,
        site=site,
        kind=kind,
        editor_method=spec.method,
        target_surface_id=target_surface_id,
        resource=resource,
    )
    return {
        "mode": effective_mode,
        "editor_method": spec.method,
        "target_surface_id": target_surface_id,
        "editor_surface_id": editor_surface_id,
        "payload_arg": payload_arg,
        "editor_args_template": template,
        "required_tokens": sorted(available),
        "verification": verification,
        "seed_capability": seed_capability,
        "phase4_exposure": phase4_exposure,
        "surface_route": _surface_route_metadata(
            resource=resource,
            mode=effective_mode,
            kind=kind,
            target_surface_id=target_surface_id,
            phase4_exposure=phase4_exposure,
        ),
        "eligibility": _eligibility_from_capabilities(seed_capability, phase4_exposure),
    }


def _candidate_selection_rank(candidate: Mapping[str, Any]) -> tuple[int, int, int, str]:
    eligibility = candidate.get("eligibility")
    is_eligible = isinstance(eligibility, Mapping) and eligibility.get("status") == "eligible"
    return (
        0 if is_eligible else 1,
        _surface_richness_rank(candidate) if is_eligible else 99,
        0 if candidate.get("phase4_exposure", {}).get("admissible") is True else 1,
        str(candidate.get("editor_method") or ""),
    )


def _surface_richness_rank(candidate: Mapping[str, Any]) -> int:
    route = candidate.get("surface_route")
    capacity = route.get("content_capacity") if isinstance(route, Mapping) else None
    if capacity in {"long_body", "comment"}:
        return 0
    if capacity == "short_title":
        return 1
    return 2


def _candidate_summary(candidate: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "mode": candidate.get("mode"),
        "editor_method": candidate.get("editor_method"),
        "target_surface_id": candidate.get("target_surface_id"),
        "payload_arg": candidate.get("payload_arg"),
        "eligibility": candidate.get("eligibility"),
        "phase4_exposure": candidate.get("phase4_exposure"),
        "surface_route": candidate.get("surface_route"),
    }
    editor_surface = candidate.get("editor_surface_id")
    if editor_surface != candidate.get("target_surface_id"):
        summary["editor_surface_id"] = editor_surface
    return summary


def _seed_capability(
    spec: EditorMethodSpec,
    payload_arg: str,
    target_surface_id: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "supported",
        "editor_method": spec.method,
        "target_surface_id": target_surface_id,
        "payload_arg": payload_arg,
        "seed_timing": "pre_task",
    }


__all__ = [
    "_candidate_selection_rank",
    "_candidate_summary",
    "_contract_id",
    "_seed_capability",
    "_surface_candidate",
    "_surface_richness_rank",
]
