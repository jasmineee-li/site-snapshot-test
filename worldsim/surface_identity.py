"""Benchmark-owned surface identity resolution.

WorldSim's methodology uses canonical carrier names such as
``issue.description`` and ``note.body``. Benchmark profiles use their own
observed IDs such as ``gitlab_issue_description``. This module is the
deterministic adapter boundary between those vocabularies.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from worldsim.benchmark_capabilities import normalize_benchmark_name
from worldsim.phases.phase_2_core_surfaces import canonical_core_surface


@dataclass(frozen=True)
class SurfaceResolution:
    benchmark: str
    site: str
    canonical_surface_id: str
    profile_surface_id: str
    profile_surface: Mapping[str, Any]
    evidence: str
    source_field: str | None = None
    editor_surface_id: str | None = None


@dataclass(frozen=True)
class SurfaceMapping:
    profile_id_aliases: Mapping[str, str]
    source_field_aliases: Mapping[str, str]


_WEBARENA_GITLAB_PROFILE_ALIASES: dict[str, str] = {
    # Historical/local profile names.
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
    # Fresh Phase 0c profile names from live WebArena Verified runs.
    "gitlab_issue_title": "issue.title",
    "gitlab_issue_description": "issue.description",
    "gitlab_note_body_on_issue": "note.body",
    "gitlab_note_body_on_mr": "note.body",
    "gitlab_mr_title": "mr.title",
    "gitlab_mr_description": "mr.description",
}

_WEBARENA_REDDIT_PROFILE_ALIASES: dict[str, str] = {
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

_MAPPINGS: dict[tuple[str, str], SurfaceMapping] = {
    ("webarena_verified", "gitlab"): SurfaceMapping(
        profile_id_aliases=_WEBARENA_GITLAB_PROFILE_ALIASES,
        source_field_aliases={
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
        },
    ),
    ("webarena_verified", "reddit"): SurfaceMapping(
        profile_id_aliases=_WEBARENA_REDDIT_PROFILE_ALIASES,
        source_field_aliases={
            "submission.title": "submission.title",
            "submissions.title": "submission.title",
            "submission.body": "submission.body",
            "submissions.body": "submission.body",
            "comment.body": "comment.body",
            "comments.body": "comment.body",
        },
    ),
}


def canonicalize_surface_id(
    *,
    benchmark: str,
    site: str,
    raw_surface_id: str | None,
) -> str | None:
    """Return the canonical WorldSim surface id for an adapter/raw surface id."""
    site_key = site.strip().lower()
    raw = raw_surface_id.strip() if isinstance(raw_surface_id, str) else ""
    if not raw:
        return None
    mapping = _mapping_for(benchmark, site_key)
    if mapping is not None:
        aliased = _lookup_alias(mapping.profile_id_aliases, raw)
        if aliased:
            return aliased
    canonical = canonical_core_surface(site_key, raw)
    return canonical


def resolve_profile_surface(
    *,
    benchmark: str,
    site: str,
    profile: Mapping[str, Any],
    target_surface_id: str,
    kind: str | None = None,
    method: str | None = None,
    editor_surface_id: str | None = None,
) -> SurfaceResolution | None:
    """Resolve a canonical carrier to a concrete profile surface.

    Unknown benchmark/site mappings fail closed. For ambiguous surfaces such as
    GitLab ``note.body``, caller context is required to choose the issue-note
    or MR-note profile surface.
    """
    benchmark_key = normalize_benchmark_name(benchmark or "webarena_verified")
    site_key = site.strip().lower()
    mapping = _mapping_for(benchmark_key, site_key)
    if mapping is None:
        return None
    target = canonicalize_surface_id(
        benchmark=benchmark_key,
        site=site_key,
        raw_surface_id=target_surface_id,
    )
    if not target:
        return None

    candidates: list[SurfaceResolution] = []
    for surface in profile.get("injection_surface", []):
        if not isinstance(surface, Mapping):
            continue
        resolution = _surface_resolution_for_candidate(
            benchmark=benchmark_key,
            site=site_key,
            mapping=mapping,
            target=target,
            surface=surface,
            editor_surface_id=editor_surface_id,
        )
        if resolution is not None:
            candidates.append(resolution)

    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    return _disambiguate_candidates(
        candidates,
        site=site_key,
        target=target,
        kind=kind,
        method=method,
        editor_surface_id=editor_surface_id,
    )


def surface_resolution_dict(resolution: SurfaceResolution | None) -> dict[str, Any] | None:
    if resolution is None:
        return None
    payload: dict[str, Any] = {
        "benchmark": resolution.benchmark,
        "site": resolution.site,
        "canonical_surface_id": resolution.canonical_surface_id,
        "profile_surface_id": resolution.profile_surface_id,
        "evidence": resolution.evidence,
    }
    if resolution.source_field:
        payload["source_field"] = resolution.source_field
    if resolution.editor_surface_id:
        payload["editor_surface_id"] = resolution.editor_surface_id
    return payload


def has_surface_mapping(*, benchmark: str, site: str) -> bool:
    return _mapping_for(benchmark, site) is not None


def _mapping_for(benchmark: str, site: str) -> SurfaceMapping | None:
    benchmark_key = normalize_benchmark_name(benchmark or "")
    if not benchmark_key:
        return None
    return _MAPPINGS.get((benchmark_key, site.strip().lower()))


def _surface_resolution_for_candidate(
    *,
    benchmark: str,
    site: str,
    mapping: SurfaceMapping,
    target: str,
    surface: Mapping[str, Any],
    editor_surface_id: str | None,
) -> SurfaceResolution | None:
    surface_id = str(surface.get("id") or "").strip()
    source_field = str(surface.get("source_field") or "").strip() or None
    matches: list[str] = []
    if surface_id == target:
        matches.append("exact_profile_id")
    aliased = _lookup_alias(mapping.profile_id_aliases, surface_id)
    if aliased == target:
        matches.append("adapter_profile_id_alias")
    source_alias = _lookup_source_field_alias(mapping.source_field_aliases, source_field)
    if source_alias == target:
        matches.append("adapter_source_field_alias")
    canonical = canonical_core_surface(site, surface_id)
    if canonical == target:
        matches.append("core_surface_alias")
    if not matches:
        return None
    return SurfaceResolution(
        benchmark=benchmark,
        site=site,
        canonical_surface_id=target,
        profile_surface_id=surface_id,
        profile_surface=surface,
        evidence="+".join(matches),
        source_field=source_field,
        editor_surface_id=editor_surface_id,
    )


def _disambiguate_candidates(
    candidates: list[SurfaceResolution],
    *,
    site: str,
    target: str,
    kind: str | None,
    method: str | None,
    editor_surface_id: str | None,
) -> SurfaceResolution | None:
    if site == "gitlab" and target == "note.body":
        wanted = _gitlab_note_context(kind=kind, method=method, editor_surface_id=editor_surface_id)
        if wanted:
            narrowed = [
                candidate
                for candidate in candidates
                if _surface_key(wanted) in _surface_key(candidate.profile_surface_id)
            ]
            if len(narrowed) == 1:
                return narrowed[0]
    return None


def _gitlab_note_context(
    *,
    kind: str | None,
    method: str | None,
    editor_surface_id: str | None,
) -> str | None:
    kind_key = (kind or "").strip().lower()
    method_key = (method or "").strip()
    editor_surface_key = (editor_surface_id or "").strip().lower()
    if method_key == "create_issue_note" or kind_key == "gitlab_issue":
        return "issue"
    if method_key == "create_mr_note" or kind_key == "gitlab_mr":
        return "mr"
    if editor_surface_key == "note_on_issue":
        return "issue"
    if editor_surface_key == "note_on_mr":
        return "mr"
    return None


def _lookup_alias(aliases: Mapping[str, str], value: str | None) -> str | None:
    key = _surface_key(value)
    for raw, canonical in aliases.items():
        if _surface_key(raw) == key:
            return canonical
    return None


def _lookup_source_field_alias(
    aliases: Mapping[str, str],
    value: str | None,
) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = _source_field_key(value)
    for raw, canonical in aliases.items():
        if _source_field_key(raw) == normalized:
            return canonical
    return None


def _surface_key(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _source_field_key(value: str | None) -> str:
    return str(value or "").strip().casefold()
