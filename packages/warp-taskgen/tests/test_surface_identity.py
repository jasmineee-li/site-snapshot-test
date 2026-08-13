from __future__ import annotations

from warp_taskgen.surface_identity import canonicalize_surface_id, resolve_profile_surface


def _profile(*surface_ids: str) -> dict:
    return {
        "site_name": "gitlab",
        "injection_surface": [
            {"id": surface_id, "source_field": source_field}
            for surface_id, source_field in [
                (
                    surface_id,
                    {
                        "gitlab_issue_description": "Issue.description",
                        "gitlab_note_body_on_issue": "Note.body",
                        "gitlab_note_body_on_mr": "Note.body",
                        "issue_description_detail": "issues.description",
                    }.get(surface_id, ""),
                )
                for surface_id in surface_ids
            ]
        ],
    }


def test_canonicalize_surface_id_uses_webarena_verified_profile_aliases():
    assert (
        canonicalize_surface_id(
            benchmark="webarena_verified",
            site="gitlab",
            raw_surface_id="gitlab_issue_description",
        )
        == "issue.description"
    )
    assert (
        canonicalize_surface_id(
            benchmark="webarena_verified",
            site="gitlab",
            raw_surface_id="gitlab_note_body_on_issue",
        )
        == "note.body"
    )


def test_resolve_profile_surface_accepts_fresh_gitlab_profile_ids():
    resolution = resolve_profile_surface(
        benchmark="webarena_verified",
        site="gitlab",
        profile=_profile("gitlab_issue_description"),
        target_surface_id="issue.description",
        kind="gitlab_search_result",
        method="create_issue_description",
    )

    assert resolution is not None
    assert resolution.canonical_surface_id == "issue.description"
    assert resolution.profile_surface_id == "gitlab_issue_description"
    assert "adapter_profile_id_alias" in resolution.evidence


def test_resolve_profile_surface_accepts_source_field_evidence():
    profile = {
        "site_name": "gitlab",
        "injection_surface": [
            {"id": "gitlab_live_surface_1", "source_field": "Issue.description"},
        ],
    }

    resolution = resolve_profile_surface(
        benchmark="webarena_verified",
        site="gitlab",
        profile=profile,
        target_surface_id="issue.description",
        kind="gitlab_search_result",
        method="create_issue_description",
    )

    assert resolution is not None
    assert resolution.profile_surface_id == "gitlab_live_surface_1"
    assert "adapter_source_field_alias" in resolution.evidence


def test_resolve_profile_surface_disambiguates_gitlab_issue_notes_by_method():
    profile = _profile("gitlab_note_body_on_issue", "gitlab_note_body_on_mr")

    issue_resolution = resolve_profile_surface(
        benchmark="webarena_verified",
        site="gitlab",
        profile=profile,
        target_surface_id="note.body",
        kind="gitlab_issue",
        method="create_issue_note",
    )
    mr_resolution = resolve_profile_surface(
        benchmark="webarena_verified",
        site="gitlab",
        profile=profile,
        target_surface_id="note.body",
        kind="gitlab_mr",
        method="create_mr_note",
    )

    assert issue_resolution is not None
    assert issue_resolution.profile_surface_id == "gitlab_note_body_on_issue"
    assert mr_resolution is not None
    assert mr_resolution.profile_surface_id == "gitlab_note_body_on_mr"


def test_resolve_profile_surface_fails_closed_on_ambiguous_gitlab_note_without_context():
    assert (
        resolve_profile_surface(
            benchmark="webarena_verified",
            site="gitlab",
            profile=_profile("gitlab_note_body_on_issue", "gitlab_note_body_on_mr"),
            target_surface_id="note.body",
        )
        is None
    )


def test_resolve_profile_surface_fails_closed_for_unknown_benchmark():
    assert (
        resolve_profile_surface(
            benchmark="unknown_benchmark",
            site="gitlab",
            profile=_profile("gitlab_issue_description"),
            target_surface_id="issue.description",
            kind="gitlab_search_result",
            method="create_issue_description",
        )
        is None
    )
