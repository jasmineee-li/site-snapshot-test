"""Phase 2 target resolution public L1/L2 resolver."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from warp_taskgen.phase_2.target_resolution.encounter import (
    _assert_anchor_contract_conformance,
    _attach_surfaces_for,
    _encounter_requirements,
    _route_evidence_flags,
)
from warp_taskgen.phase_2.target_resolution.listing_intent import _gitlab_issue_listing_intent
from warp_taskgen.phase_2.target_resolution.url_matching import (
    _empty_record,
    _site_kind_for_task,
)
from warp_taskgen.sites import SiteTargetingDefinitionError, TargetingFailure, default_catalog


def derive_benign_target_resource(
    task: Mapping[str, Any],
    placeholders: Mapping[str, str],
    *,
    allow_layers: tuple[Literal["L1", "L2", "L3", "L4"], ...] = ("L1", "L2"),
    benchmark: str = "webarena_verified",
) -> dict[str, Any]:
    """Resolve the benign target resource for a Phase 1 task (L1/L2 only).

    Returns a dict matching handoff §Benign-target resource extraction:
    ``{kind, anchors, start_url_resolved, attach_surfaces,
    encounter_requirements, layer, ...}``. When L1+L2 cannot classify
    the task, returns an empty record with ``pending_layer`` set so the
    caller can route to L3 via :func:`resolve_l3` in a later pass.

    L3 is async (it calls the Anthropic Messages API and the live
    benchmark instance), so this sync entrypoint refuses to dispatch
    L3/L4 directly — ``allow_layers`` containing either raises.
    """
    site_kind = _site_kind_for_task(task)
    if site_kind is None:
        return _empty_record("task is not gitlab or reddit (out of WASP scope)", None)

    if "L3" in allow_layers or "L4" in allow_layers:
        raise NotImplementedError(
            "L3 and L4 are async; call resolve_l3() / resolve_l4() explicitly."
        )

    try:
        bound = default_catalog().bind(
            benchmark=benchmark,
            site=site_kind,
            placeholders=placeholders,
        )
    except SiteTargetingDefinitionError:
        return _empty_record("task is not gitlab or reddit (out of WASP scope)", None)

    target = bound.resolve(task, allow_layers=allow_layers)
    resolved_start = target.evidence_url if isinstance(target, TargetingFailure) else None
    if not isinstance(target, TargetingFailure):
        resolved_start = target.evidence_url
        is_gitlab_issue_listing = (
            site_kind == "gitlab"
            and target.kind == "search_result"
            and "project_path" in target.anchors
            and "/-/issues" in str(target.evidence_url or "")
        )
        if is_gitlab_issue_listing:
            listing_record = (
                _gitlab_issue_listing_intent(
                    task,
                    resolved_start=resolved_start,
                    placeholders=placeholders,
                    benchmark=benchmark,
                )
                if "L2" in allow_layers
                else None
            )
            if listing_record is not None:
                return listing_record
            target = TargetingFailure(
                site_kind,
                "unresolved_evidence",
                "L1+L2 found no concrete resource; intent-only task pending L3",
                pending_layer="L3",
                evidence_url=resolved_start,
            )

    if not isinstance(target, TargetingFailure):
        kind = (
            target.canonical_route.compatibility_kind
            if target.canonical_route is not None
            else target.kind
        )
        anchors = dict(target.anchors)
        record = {
            "kind": kind,
            "anchors": anchors,
            "start_url_resolved": target.start_url_resolved,
            "attach_surfaces": _attach_surfaces_for(kind, benchmark=benchmark, site=site_kind),
            "encounter_requirements": _encounter_requirements(kind, task, anchors),
            "layer": target.layer,
        }
        record.update(_route_evidence_flags(kind, task))
        _assert_anchor_contract_conformance(record, benchmark=benchmark, site=site_kind)
        return record

    if "L2" in allow_layers and site_kind == "gitlab":
        listing_record = _gitlab_issue_listing_intent(
            task,
            resolved_start=resolved_start,
            placeholders=placeholders,
            benchmark=benchmark,
        )
        if listing_record is not None:
            return listing_record

    # Fall-through: bare __GITLAB__ / __REDDIT__ or intent-only task.
    # L3 owns these: LLM intent parse + live API probe. Signal pending
    # so the caller (Phase 2a) routes this task's target derivation to
    # the L3 pass once it lands.
    record = _empty_record(
        "L1+L2 found no concrete resource; intent-only task pending L3",
        pending_layer="L3",
    )
    record["start_url_resolved"] = resolved_start
    return record
