"""Reddit/Postmill-owned profile surface aliases."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from warp_taskgen.benchmark_capabilities import normalize_benchmark_name
from warp_taskgen.sites.carrier_policy import SiteCarrierPolicy
from warp_taskgen.sites.contracts import SurfaceResolution
from warp_taskgen.sites.profile_routes import (
    ProfileSurfaceMapping,
)
from warp_taskgen.sites.profile_routes import (
    canonicalize_surface_id as _canonicalize,
)
from warp_taskgen.sites.profile_routes import (
    resolve_profile_surface as _resolve,
)

_PROFILE_ID_ALIASES: dict[str, str] = {
    "submission_title": "submission.title",
    "submission_title_forum_listing": "submission.title",
    "submission_title_listing": "submission.title",
    "submission_title_feed": "submission.title",
    "submission_title_detail": "submission.title",
    "submission_body": "submission.body",
    "submission_body_post_detail": "submission.body",
    "submission_body_detail": "submission.body",
    "comment_body": "comment.body",
    "comment_body_post_detail": "comment.body",
    "comment_body_thread": "comment.body",
    "comment_body_detail": "comment.body",
}

_SOURCE_FIELD_ALIASES: dict[str, str] = {
    "submission.title": "submission.title",
    "submissions.title": "submission.title",
    "submission.body": "submission.body",
    "submissions.body": "submission.body",
    "comment.body": "comment.body",
    "comments.body": "comment.body",
}

_MAPPING = ProfileSurfaceMapping(
    benchmark="webarena_verified",
    profile_id_aliases=_PROFILE_ID_ALIASES,
    source_field_aliases=_SOURCE_FIELD_ALIASES,
)

_CARRIER_POLICY = SiteCarrierPolicy(
    benchmark="webarena_verified",
    surface_aliases=_PROFILE_ID_ALIASES,
    core_surfaces=frozenset({"submission.title", "submission.body", "comment.body"}),
    retired_carrier_surfaces=frozenset({"submission.title"}),
)


class RedditProfileIdentity:
    """Mixin implementing Reddit/Postmill's profile-surface vocabulary."""

    def canonicalize_surface_id(
        self,
        *,
        benchmark: str,
        raw_surface_id: str | None,
    ) -> str | None:
        return canonicalize_surface_id(
            benchmark=benchmark,
            raw_surface_id=raw_surface_id,
        )

    def resolve_profile_surface(
        self,
        *,
        benchmark: str,
        profile: Mapping[str, Any],
        target_surface_id: str,
        kind: str | None = None,
        method: str | None = None,
        editor_surface_id: str | None = None,
    ) -> SurfaceResolution | None:
        return resolve_profile_surface(
            benchmark=benchmark,
            profile=profile,
            target_surface_id=target_surface_id,
            kind=kind,
            method=method,
            editor_surface_id=editor_surface_id,
        )

    def carrier_policy(self, *, benchmark: str) -> SiteCarrierPolicy | None:
        if normalize_benchmark_name(benchmark or "") != _CARRIER_POLICY.benchmark:
            return None
        return _CARRIER_POLICY


def mapping_for(benchmark: str) -> ProfileSurfaceMapping | None:
    return _MAPPING if normalize_benchmark_name(benchmark or "") == _MAPPING.benchmark else None


def canonicalize_surface_id(*, benchmark: str, raw_surface_id: str | None) -> str | None:
    return _canonicalize(mapping=mapping_for(benchmark), raw_surface_id=raw_surface_id)


def resolve_profile_surface(
    *,
    benchmark: str,
    profile: Mapping[str, Any],
    target_surface_id: str,
    kind: str | None = None,
    method: str | None = None,
    editor_surface_id: str | None = None,
) -> SurfaceResolution | None:
    return _resolve(
        mapping=mapping_for(benchmark),
        benchmark=benchmark,
        site="reddit",
        profile=profile,
        target_surface_id=target_surface_id,
        kind=kind,
        method=method,
        editor_surface_id=editor_surface_id,
    )


__all__ = [
    "RedditProfileIdentity",
    "canonicalize_surface_id",
    "mapping_for",
    "resolve_profile_surface",
]
