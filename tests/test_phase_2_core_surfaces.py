from __future__ import annotations

from worldsim.phases.phase_2_core_surfaces import (
    CORE_SURFACES,
    canonical_core_surface,
    is_core_surface,
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
    assert canonical_core_surface("gitlab", "note_on_issue") == "note.body"
    assert canonical_core_surface("gitlab", "note_on_mr") == "note.body"
    assert canonical_core_surface("reddit", "submission_body_detail") == "submission.body"
    assert canonical_core_surface("reddit", "comment_body_thread") == "comment.body"


def test_non_core_and_unknown_sites_fail_closed():
    assert not is_core_surface("gitlab", "label.description")
    assert not is_core_surface("gitlab", "milestone.description")
    assert not is_core_surface("reddit", "comment_form_placeholder")
    assert not is_core_surface("shopping", "review.body")
    assert not is_core_surface("gitlab", None)
