"""Deterministic route contracts for Phase 1 novel task generation.

This module turns adapter-owned facts into a compact prompt artifact. It is
intentionally not an LLM-authored catalog: editor decorators, core-surface
policy, and benchmark profiles remain the source of truth.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping
from typing import Any

import warp_taskgen.editors  # noqa: F401 - populate editor method registry
from warp_taskgen.editors._registry import iter_specs
from warp_taskgen.phase_1.route_contract_guidance import (
    _answer_stability_guidance,
    _evaluator_guidance,
    _instruction_requirements,
)
from warp_taskgen.phase_1.route_exposure_admissibility import (
    _phase2_admissible_start_patterns,
    _sample_editor_args,
)
from warp_taskgen.phase_2.target_resolution.constants import (
    DEFAULT_REDDIT_MAX_EXISTING_COMMENTS,
)
from warp_taskgen.sites import (
    BoundSite,
    SiteRouteContractFacts,
    SiteTargetingDefinitionError,
    default_catalog,
)

ROUTE_CONTRACTS_SCHEMA_VERSION = 1


def build_task_route_contracts(
    *,
    site_name: str,
    profile: Mapping[str, Any],
    benchmark: str = "webarena_verified",
) -> dict[str, Any]:
    """Build the route contracts a Phase 1 generator may target."""
    site = site_name.strip().lower()
    if not _valid_profile_shape(profile):
        return {
            "schema_version": ROUTE_CONTRACTS_SCHEMA_VERSION,
            "site": site,
            "benchmark": benchmark,
            "route_families": [],
        }
    try:
        # Bind once so every profile/route lookup crosses the same immutable
        # Site projection. Mismatched profile identity fails closed here.
        bound = default_catalog().bind(
            benchmark=benchmark,
            site=site,
            profile=profile,
        )
    except SiteTargetingDefinitionError:
        return {
            "schema_version": ROUTE_CONTRACTS_SCHEMA_VERSION,
            "site": site,
            "benchmark": benchmark,
            "route_families": [],
        }
    policy = bound.carrier_policy()
    uncovered = _uncovered_surface_ids(profile, bound=bound)
    covered = _covered_surface_ids(profile, bound=bound)
    route_families: list[dict[str, Any]] = []

    for spec in sorted(iter_specs(site=site, benchmark=benchmark), key=lambda item: item.method):
        for kind in sorted(spec.kinds):
            raw_surface = spec.surface_id_per_kind.get(kind, spec.method)
            canonical = bound.canonicalize_surface_id(raw_surface)
            if not canonical or not policy.is_core_surface(canonical):
                continue
            if not policy.is_active_carrier(canonical, kind=kind, method=spec.method):
                continue
            facts = bound.route_contract_facts(kind)
            profile_resolution = bound.resolve_profile_surface(
                canonical,
                kind=kind,
                method=spec.method,
                editor_surface_id=raw_surface,
            )
            if profile_resolution is None:
                profile_surface, surface_resolution = _profile_surface_fallback(
                    benchmark=benchmark,
                    site=site,
                    target_surface_id=canonical,
                    method=spec.method,
                    editor_surface_id=raw_surface,
                    facts=facts,
                )
                if profile_surface is None:
                    continue
            else:
                profile_surface = profile_resolution.profile_surface
                surface_resolution = profile_resolution.as_record()
            route = _route_family_for_spec(
                site=site,
                kind=kind,
                method=spec.method,
                raw_surface_id=raw_surface,
                canonical_surface_id=canonical,
                coverage_status=_coverage_status(canonical, raw_surface, uncovered, covered),
                profile_surface=profile_surface,
                surface_resolution=surface_resolution,
                facts=facts,
            )
            if route is not None:
                route_families.append(route)

    return {
        "schema_version": ROUTE_CONTRACTS_SCHEMA_VERSION,
        "site": site,
        "benchmark": benchmark,
        "route_families": route_families,
    }


def _valid_profile_shape(profile: object) -> bool:
    if not isinstance(profile, Mapping):
        return False
    for field in ("site_name", "site", "benchmark_name", "benchmark"):
        if field in profile and (not isinstance(profile[field], str) or not profile[field].strip()):
            return False
    for field in ("injection_surface", "data_model"):
        if field not in profile:
            continue
        value = profile[field]
        if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
            return False
    available = profile.get("available_entities")
    if "available_entities" in profile and not isinstance(available, Mapping):
        return False
    coverage = profile.get("existing_task_coverage")
    if "existing_task_coverage" in profile and not isinstance(coverage, Mapping):
        return False
    if isinstance(coverage, Mapping):
        for field in (
            "injection_surfaces_with_task_coverage",
            "injection_surfaces_without_task_coverage",
        ):
            if field in coverage and not isinstance(coverage[field], list):
                return False
    return True


def route_contracts_digest(route_contracts: Mapping[str, Any]) -> str:
    """Return a stable string representation suitable for existing hash helpers."""
    return json.dumps(route_contracts, sort_keys=True, separators=(",", ":"))


def _profile_surface_fallback(
    *,
    benchmark: str,
    site: str,
    target_surface_id: str,
    method: str,
    editor_surface_id: str,
    facts: SiteRouteContractFacts,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return a Site-declared profile-surface fallback for a known carrier.

    Phase 0c profiles are LLM-authored and can omit a known editor-backed body
    surface even when host inventory is present.  A Site may declare such a
    surface on its route facts; this module keeps the evidence envelope, so the
    route stays inventory-backed and must still pass Phase 2c render
    feasibility.
    """

    surface = facts.profile_surface_fallbacks.get((target_surface_id, method))
    if surface is None:
        return None, None
    overlay = copy.deepcopy(surface)
    resolution = {
        "benchmark": benchmark,
        "site": site,
        "canonical_surface_id": target_surface_id,
        "profile_surface_id": overlay["id"],
        "evidence": "editor_registry_active_carrier_fallback",
        "source_field": overlay["source_field"],
        "editor_surface_id": editor_surface_id,
        "reason": (
            "profile omitted a known non-appended WASP carrier; route remains "
            "inventory-backed and must pass Phase 2c render feasibility"
        ),
    }
    return overlay, resolution


def _route_family_for_spec(
    *,
    site: str,
    kind: str,
    method: str,
    raw_surface_id: str,
    canonical_surface_id: str,
    coverage_status: str,
    profile_surface: Mapping[str, Any] | None,
    surface_resolution: Mapping[str, Any] | None,
    facts: SiteRouteContractFacts,
) -> dict[str, Any] | None:
    anchor_examples = [dict(example) for example in facts.anchor_examples]
    _apply_anchor_policy(anchor_examples)
    requires_inventory_backed_start_url = facts.requires_inventory_backed_start_url
    if requires_inventory_backed_start_url and not anchor_examples:
        return None
    start_patterns = list(facts.allowed_start_url_patterns)
    start_patterns = _phase2_admissible_start_patterns(
        site=site,
        kind=kind,
        method=method,
        patterns=start_patterns,
        facts=facts,
    )
    if not start_patterns:
        return None
    route_id = f"{site}.{canonical_surface_id.replace('.', '_')}.{kind}.{method}"
    route = {
        "id": route_id,
        "site": site,
        "enabled": True,
        "eligible": True,
        "resource_kind": kind,
        "content_surface": canonical_surface_id,
        "coverage_status": coverage_status,
        "profile_surface_id": _profile_surface_id(profile_surface),
        "surface_resolution": dict(surface_resolution or {}),
        "allowed_start_url_patterns": start_patterns,
        "allowed_editor_methods": [method],
        "editor_arg_templates": {method: _sample_editor_args(method, facts=facts)},
        "instruction_requirements": _instruction_requirements(
            canonical_surface_id, site=site, facts=facts
        ),
        "evaluator_guidance": _evaluator_guidance(canonical_surface_id),
        "answer_stability_guidance": _answer_stability_guidance(
            surface_id=canonical_surface_id,
            method=method,
            facts=facts,
        ),
        "source_evidence": {
            "source": "editor_registry_and_core_surface_policy",
            "editor_surface_id": raw_surface_id,
            "profile_location_page": _profile_location_page(profile_surface),
        },
    }
    if (
        isinstance(surface_resolution, Mapping)
        and surface_resolution.get("evidence") == "editor_registry_active_carrier_fallback"
        and profile_surface is not None
    ):
        route["profile_surface_overlay"] = dict(profile_surface)
    route_variant = facts.route_variant or _route_variant_from_anchor_examples(anchor_examples)
    if route_variant is not None:
        route["route_variant"] = route_variant
    if requires_inventory_backed_start_url:
        route["requires_inventory_backed_start_url"] = True
        route["anchor_examples"] = anchor_examples
    return route


def _route_variant_from_anchor_examples(
    anchor_examples: list[Mapping[str, Any]],
) -> str | None:
    variants = {
        str(example.get("route_variant") or "").strip()
        for example in anchor_examples
        if isinstance(example, Mapping)
    }
    variants.discard("")
    if len(variants) == 1:
        return next(iter(variants))
    return None


def _apply_anchor_policy(examples: list[dict[str, Any]]) -> None:
    """Add Phase-owned seed/visibility policy to Site inventory facts.

    Only an anchor that reports an existing child count can be judged here, so
    a Site that does not publish one keeps its inventory example unchanged.
    """

    for example in examples:
        raw_count = example.get("existing_comment_count")
        if not isinstance(raw_count, str) or not raw_count.isdigit():
            continue
        if int(raw_count) > DEFAULT_REDDIT_MAX_EXISTING_COMMENTS:
            example.pop("existing_comment_count", None)
            continue
        example["max_existing_comments_for_comment_seed"] = str(
            DEFAULT_REDDIT_MAX_EXISTING_COMMENTS
        )
        example["seeded_comment_visibility_candidate"] = "true"


def _uncovered_surface_ids(
    profile: Mapping[str, Any],
    *,
    bound: BoundSite,
) -> set[str]:
    coverage = profile.get("existing_task_coverage")
    if not isinstance(coverage, Mapping):
        return set()
    uncovered = coverage.get("injection_surfaces_without_task_coverage")
    if not isinstance(uncovered, list):
        return set()
    out: set[str] = set()
    for item in uncovered:
        raw = str(item).strip()
        if not raw:
            continue
        key = _surface_key(raw)
        out.add(key)
        canonical = bound.canonicalize_surface_id(raw)
        if canonical:
            out.add(_surface_key(canonical))
    return out


def _covered_surface_ids(
    profile: Mapping[str, Any],
    *,
    bound: BoundSite,
) -> set[str]:
    coverage = profile.get("existing_task_coverage")
    if not isinstance(coverage, Mapping):
        return set()
    covered = coverage.get("injection_surfaces_with_task_coverage")
    if not isinstance(covered, list):
        return set()
    out: set[str] = set()
    for item in covered:
        raw = str(item).strip()
        if not raw:
            continue
        key = _surface_key(raw)
        out.add(key)
        canonical = bound.canonicalize_surface_id(raw)
        if canonical:
            out.add(_surface_key(canonical))
    return out


def _coverage_status(
    canonical: str,
    raw: str,
    uncovered: set[str],
    covered: set[str],
) -> str:
    candidates = {_surface_key(canonical), _surface_key(raw)}
    if candidates & uncovered:
        return "uncovered"
    if candidates & covered:
        return "covered"
    return "unknown"


def _surface_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _profile_surface_id(surface: Mapping[str, Any] | None) -> str | None:
    if not surface:
        return None
    value = surface.get("id")
    return str(value) if value is not None else None


def _profile_location_page(surface: Mapping[str, Any] | None) -> str | None:
    if not surface:
        return None
    value = surface.get("location_page")
    return str(value) if value is not None else None
