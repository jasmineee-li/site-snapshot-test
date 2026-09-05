"""Exposure contract builder.

An exposure contract is a composed view over the resolver output and the
editor-method registry. It is not a new placement registry: editor
decorators remain the source of truth for methods, bindings, surface IDs,
and reachable tokens.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from warp_taskgen.editors._registry import available_tokens_for_kind
from warp_taskgen.phase_2.exposure_contract.candidates import (
    _candidate_selection_rank,
    _candidate_summary,
    _contract_id,
    _surface_candidate,
)
from warp_taskgen.phase_2.exposure_contract.editor_args import (
    _allowed_editor_methods,
    _viable_specs,
)
from warp_taskgen.phase_2.exposure_contract.modes import (
    _benign_read_url,
    _mode_for_resource,
)
from warp_taskgen.phase_2.exposure_contract.phase4_exposure import (
    _phase4_exposure_capability,
    _phase4_runtime_hook_available,
    _transition_forced_by_task,
)
from warp_taskgen.phase_2.exposure_contract.route_metadata import _route_variant_label
from warp_taskgen.sites import SiteCarrierPolicy, SiteTargetingDefinitionError, default_catalog


def build_exposure_contract(
    *,
    benign_task_id: str,
    site: str,
    benchmark: str,
    benign_target_resource: Mapping[str, Any] | None,
    surface_visibility_by_id: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build the exposure contract for one resolved benign target."""
    resource = dict(benign_target_resource or {})
    kind = resource.get("kind")
    anchors_raw = resource.get("anchors")
    anchors = dict(anchors_raw) if isinstance(anchors_raw, Mapping) else {}
    site = site.strip().lower()

    base: dict[str, Any] = {
        "contract_id": _contract_id(site, benign_task_id, kind, anchors),
        "benign_task_id": benign_task_id,
        "site": site,
        "kind": kind,
        "anchors": anchors,
        "benign_read_url": _benign_read_url(resource),
        "seed_capability": {
            "status": "unsupported",
            "reason": "unresolved_target_resource",
        },
        "phase4_exposure": _phase4_exposure_capability(
            "ineligible",
            reason="unresolved_target_resource",
        ),
        "eligibility": {"status": "ineligible", "reason": "unresolved_target_resource"},
    }
    route_variant = _route_variant_label(resource)
    if route_variant is not None:
        base["route_variant"] = route_variant

    if not isinstance(kind, str) or not kind:
        return base

    mode, ineligible_reason = _mode_for_resource(resource, kind)
    if mode == "ineligible":
        base["mode"] = "ineligible"
        reason = ineligible_reason or f"kind_not_supported_for_exposure:{kind}"
        base["phase4_exposure"] = _phase4_exposure_capability("ineligible", reason=reason)
        base["seed_capability"] = {"status": "unsupported", "reason": reason}
        base["eligibility"] = {
            "status": "ineligible",
            "reason": reason,
        }
        return base
    if not base["benign_read_url"]:
        base["mode"] = mode
        base["phase4_exposure"] = _phase4_exposure_capability(
            mode,
            reason="missing_benign_read_url",
        )
        base["seed_capability"] = {
            "status": "unsupported",
            "reason": "missing_benign_read_url",
        }
        base["eligibility"] = {
            "status": "ineligible",
            "reason": "missing_benign_read_url",
        }
        return base

    # Bind the Site once: its carrier policy is the exposure gate for every
    # candidate below. An unknown Site or benchmark binds a closed policy, so
    # every candidate reports ``non_core_surface`` rather than guessing.
    try:
        policy = default_catalog().bind(benchmark=benchmark, site=site).carrier_policy()
    except SiteTargetingDefinitionError:
        policy = SiteCarrierPolicy.closed(benchmark)
    available = available_tokens_for_kind(kind, anchors, benchmark=benchmark, site=site)
    allowed_editor_methods = _allowed_editor_methods(resource)
    candidates: list[dict[str, Any]] = []
    for spec in _viable_specs(
        kind,
        site=site,
        benchmark=benchmark,
        available=available,
        allowed_editor_methods=allowed_editor_methods,
        surface_visibility_by_id=surface_visibility_by_id,
        policy=policy,
    ):
        candidate = _surface_candidate(
            resource=resource,
            base_mode=mode,
            benign_task_id=benign_task_id,
            benign_read_url=base["benign_read_url"],
            kind=kind,
            site=site,
            available=available,
            spec=spec,
            policy=policy,
        )
        if candidate is not None:
            candidates.append(candidate)

    if candidates:
        selected = min(candidates, key=_candidate_selection_rank)
        base.update(selected)
        base["surface_candidates"] = [_candidate_summary(candidate) for candidate in candidates]
        if base.get("target_surface_id") != base.get("editor_surface_id"):
            base["editor_surface_id"] = base.get("editor_surface_id")
        elif "editor_surface_id" in base and base.get("target_surface_id") == base.get(
            "editor_surface_id"
        ):
            base.pop("editor_surface_id", None)
        return base

    base["mode"] = mode
    base["required_tokens"] = sorted(available)
    base["phase4_exposure"] = _phase4_exposure_capability(
        mode,
        transition_forced_by_task=_transition_forced_by_task(resource),
        runtime_hook_available=_phase4_runtime_hook_available(resource),
    )
    base["seed_capability"] = {
        "status": "unsupported",
        "reason": "no_viable_editor_method_under_anchors",
    }
    base["eligibility"] = {
        "status": "ineligible",
        "reason": "no_viable_editor_method_under_anchors",
    }
    return base


__all__ = [
    "build_exposure_contract",
]
