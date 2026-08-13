"""GitLab-owned profile surface aliases and ambiguity rules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from warp_taskgen.benchmark_capabilities import normalize_benchmark_name
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
    "issue_title": "issue.title",
    "issue_title_list": "issue.title",
    "issue_list_title": "issue.title",
    "issue_title_in_list": "issue.title",
    "issue_description": "issue.description",
    "issue_description_detail": "issue.description",
    "issue_detail_description": "issue.description",
    "issue_note": "note.body",
    "issue_note_body": "note.body",
    "note_on_issue": "note.body",
    "note_body_on_issue": "note.body",
    "mr_title": "mr.title",
    "mr_title_list": "mr.title",
    "mr_list_title": "mr.title",
    "mr_title_in_list": "mr.title",
    "mr_description": "mr.description",
    "mr_description_detail": "mr.description",
    "mr_detail_description": "mr.description",
    "mr_note": "note.body",
    "mr_note_body": "note.body",
    "note_on_mr": "note.body",
    "note_body_on_mr": "note.body",
    "gitlab_issue_title": "issue.title",
    "gitlab_issue_description": "issue.description",
    "gitlab_note_body_on_issue": "note.body",
    "gitlab_note_body_on_mr": "note.body",
    "gitlab_mr_title": "mr.title",
    "gitlab_mr_description": "mr.description",
}

_SOURCE_FIELD_ALIASES: dict[str, str] = {
    "issue.title": "issue.title",
    "issues.title": "issue.title",
    "issue.description": "issue.description",
    "issues.description": "issue.description",
    "merge_request.title": "mr.title",
    "mergerequest.title": "mr.title",
    "merge_requests.title": "mr.title",
    "merge_request.description": "mr.description",
    "mergerequest.description": "mr.description",
    "merge_requests.description": "mr.description",
    "note.body": "note.body",
    "notes.body": "note.body",
}

_MAPPING = ProfileSurfaceMapping(
    benchmark="webarena_verified",
    profile_id_aliases=_PROFILE_ID_ALIASES,
    source_field_aliases=_SOURCE_FIELD_ALIASES,
)


class GitLabProfileIdentity:
    """Mixin implementing GitLab's profile-surface vocabulary."""

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
        site="gitlab",
        profile=profile,
        target_surface_id=target_surface_id,
        kind=kind,
        method=method,
        editor_surface_id=editor_surface_id,
        disambiguate=_disambiguate,
    )


def _disambiguate(
    candidates: list[SurfaceResolution],
    target: str,
    kind: str | None,
    method: str | None,
    editor_surface_id: str | None,
) -> SurfaceResolution | None:
    if target != "note.body":
        return None
    wanted = _note_context(kind=kind, method=method, editor_surface_id=editor_surface_id)
    if not wanted:
        return None
    narrowed = [
        candidate for candidate in candidates if wanted in candidate.profile_surface_id.casefold()
    ]
    return narrowed[0] if len(narrowed) == 1 else None


def _note_context(
    *, kind: str | None, method: str | None, editor_surface_id: str | None
) -> str | None:
    kind_key = (kind or "").strip().lower()
    method_key = (method or "").strip()
    surface_key = (editor_surface_id or "").strip().lower()
    if method_key == "create_issue_note" or kind_key == "gitlab_issue":
        return "issue"
    if method_key == "create_mr_note" or kind_key == "gitlab_mr":
        return "mr"
    if surface_key == "note_on_issue":
        return "issue"
    if surface_key == "note_on_mr":
        return "mr"
    return None


__all__ = [
    "GitLabProfileIdentity",
    "canonicalize_surface_id",
    "mapping_for",
    "resolve_profile_surface",
]
