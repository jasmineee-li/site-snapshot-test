"""Assertions for the deterministic Site Targeting owner seam."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from warp_taskgen.sites import BoundSite, ResolvedTarget, TargetingFailure


def assert_site_targeting_behavior(
    bound_site: BoundSite,
    valid_task: Mapping[str, Any],
    *,
    expected_site: str,
    expected_kind: str,
    expected_anchors: Mapping[str, Any],
    expected_url: str,
    expected_profile_surface: str,
    expected_canonical_surface: str,
    expected_editor_method: str,
    foreign_origin_task: Mapping[str, Any],
    malformed_parent_task: Mapping[str, Any],
) -> None:
    """Check exact route resolution and fail-closed Site evidence handling."""

    resolved = bound_site.resolve(valid_task)
    assert isinstance(resolved, ResolvedTarget)
    assert resolved.site == expected_site
    assert resolved.kind == expected_kind
    assert dict(resolved.anchors) == dict(expected_anchors)
    assert resolved.start_url_resolved == expected_url
    assert resolved.canonical_route is not None
    assert resolved.canonical_route.kind == expected_kind

    profile = bound_site.resolve_profile_surface(
        expected_profile_surface,
        kind=expected_kind,
        method=expected_editor_method,
        editor_surface_id=expected_canonical_surface,
    )
    assert profile is not None
    assert profile.canonical_surface_id == expected_canonical_surface
    assert bound_site.resolve_profile_surface("unknown_surface") is None

    wrong_site = dict(valid_task)
    foreign_site = f"not_{expected_site}"
    wrong_site["site"] = foreign_site
    wrong_site["sites"] = [foreign_site]
    failure = bound_site.resolve(wrong_site)
    assert isinstance(failure, TargetingFailure)
    assert failure.reason == "unsupported_site"

    failure = bound_site.resolve(foreign_origin_task)
    assert isinstance(failure, TargetingFailure)
    assert failure.reason == "unresolved_evidence"

    failure = bound_site.resolve(malformed_parent_task)
    assert isinstance(failure, TargetingFailure)
    assert failure.reason == "unresolved_evidence"
