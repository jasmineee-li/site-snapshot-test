from __future__ import annotations

from worldsim.phases.phase_2_core_surfaces import (
    CORE_SURFACES,
    active_carrier_ineligible_reason,
    canonical_core_surface,
    is_active_carrier_surface,
    is_core_surface,
    retired_carrier_reason,
)


def test_core_surfaces_module_lists_only_user_controllable():
    """Path A core surfaces must stay inside non-admin user-writable UGC."""
    user_controllable_non_admin = {
        "gitlab": {
            "issue.title",
            "issue.description",
            "mr.title",
            "mr.description",
            "note.body",
        },
        "reddit": {
            "submission.title",
            "submission.body",
            "comment.body",
        },
    }

    assert set(CORE_SURFACES) == set(user_controllable_non_admin)
    for site, surfaces in CORE_SURFACES.items():
        assert set(surfaces) <= user_controllable_non_admin[site]


def test_legacy_editor_surface_ids_canonicalize_to_core_surfaces():
    assert canonical_core_surface("gitlab", "issue_list_title") == "issue.title"
    assert canonical_core_surface("gitlab", "issue_title_in_list") == "issue.title"
    assert canonical_core_surface("gitlab", "issue_description_detail") == "issue.description"
    assert canonical_core_surface("gitlab", "issue_detail_description") == "issue.description"
    assert canonical_core_surface("gitlab", "mr_list_title") == "mr.title"
    assert canonical_core_surface("gitlab", "mr_title_in_list") == "mr.title"
    assert canonical_core_surface("gitlab", "mr_description_detail") == "mr.description"
    assert canonical_core_surface("gitlab", "mr_detail_description") == "mr.description"
    assert canonical_core_surface("gitlab", "note_on_issue") == "note.body"
    assert canonical_core_surface("gitlab", "note_body_on_issue") == "note.body"
    assert canonical_core_surface("gitlab", "note_on_mr") == "note.body"
    assert canonical_core_surface("gitlab", "note_body_on_mr") == "note.body"
    assert canonical_core_surface("reddit", "submission_title") == "submission.title"
    assert canonical_core_surface("reddit", "submission_title_listing") == "submission.title"
    assert canonical_core_surface("reddit", "submission_title_detail") == "submission.title"
    assert canonical_core_surface("reddit", "submission_body") == "submission.body"
    assert canonical_core_surface("reddit", "submission_body_detail") == "submission.body"
    assert canonical_core_surface("reddit", "comment_body") == "comment.body"
    assert canonical_core_surface("reddit", "comment_body_thread") == "comment.body"
    assert canonical_core_surface("reddit", "comment_body_detail") == "comment.body"


def test_non_core_and_unknown_sites_fail_closed():
    assert not is_core_surface("gitlab", "label.description")
    assert not is_core_surface("gitlab", "milestone.description")
    assert not is_core_surface("reddit", "comment_form_placeholder")
    assert not is_core_surface("shopping", "review.body")
    assert not is_core_surface("gitlab", None)


def test_title_surfaces_are_core_metadata_but_retired_as_active_carriers():
    assert is_core_surface("gitlab", "issue.title")
    assert is_core_surface("gitlab", "mr.title")
    assert is_core_surface("reddit", "submission.title")

    assert retired_carrier_reason("gitlab", "issue.title") == "retired_title_carrier_surface"
    assert retired_carrier_reason("gitlab", "mr.title") == "retired_title_carrier_surface"
    assert (
        retired_carrier_reason("reddit", "submission.title")
        == "retired_title_carrier_surface"
    )

    assert not is_active_carrier_surface("gitlab", "issue.title")
    assert not is_active_carrier_surface("gitlab", "mr.title")
    assert not is_active_carrier_surface("reddit", "submission.title")


def test_body_and_comment_surfaces_remain_active_carriers():
    assert is_active_carrier_surface("gitlab", "issue.description")
    assert is_active_carrier_surface("gitlab", "note.body")
    assert is_active_carrier_surface("reddit", "submission.body")
    assert is_active_carrier_surface("reddit", "comment.body")


def test_gitlab_merge_request_routes_are_not_active_mainline_carriers():
    assert (
        active_carrier_ineligible_reason("gitlab", "mr.description")
        == "unsupported_merge_request_carrier_surface"
    )
    assert (
        active_carrier_ineligible_reason("gitlab", "note.body", kind="gitlab_mr")
        == "unsupported_merge_request_carrier_surface"
    )
    assert (
        active_carrier_ineligible_reason("gitlab", "note.body", method="create_mr_note")
        == "unsupported_merge_request_carrier_surface"
    )

    assert not is_active_carrier_surface("gitlab", "mr.description")
    assert not is_active_carrier_surface("gitlab", "note.body", kind="gitlab_mr")
    assert not is_active_carrier_surface("gitlab", "note.body", method="create_mr_note")

    assert is_active_carrier_surface("gitlab", "note.body", kind="gitlab_issue")
    assert is_active_carrier_surface("gitlab", "note.body", method="create_issue_note")
