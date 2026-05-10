"""Exposure-contract tests for the six gitlab new_task kinds.

The editor registry can write these kinds, but Path A core-surface
placement keeps them out of the Phase 4 cohort because they are not
naturally traversed issue/MR note/body surfaces.
"""

from __future__ import annotations

from worldsim.phases.phase_2_exposure_contract import build_exposure_contract


def _build(kind: str, anchors: dict, start_url: str) -> dict:
    return build_exposure_contract(
        benign_task_id="task-1",
        site="gitlab",
        benchmark="webarena_verified",
        benign_target_resource={
            "kind": kind,
            "anchors": anchors,
            "start_url_resolved": start_url,
        },
    )


def test_gitlab_user_profile_eligible_direct_detail():
    contract = _build(
        "gitlab_user_profile",
        {"username": "byteblaze"},
        "https://gitlab.local/byteblaze",
    )
    assert contract["eligibility"] == {"status": "ineligible", "reason": "non_core_surface"}
    assert contract["mode"] == "direct_detail"
    # Both bio and status methods address gitlab_user_profile; the
    # registry sort picks the first by method name. Ensure target surface
    # id is one of the two declared values.
    assert contract["target_surface_id"] in {
        "user_profile_bio",
        "user_status_message_profile",
    }


def test_gitlab_snippet_eligible_direct_detail():
    contract = _build(
        "gitlab_snippet",
        {"snippet_id": "42"},
        "https://gitlab.local/-/snippets/42",
    )
    assert contract["eligibility"] == {"status": "ineligible", "reason": "non_core_surface"}
    assert contract["mode"] == "direct_detail"
    assert contract["editor_method"] == "update_snippet"
    assert contract["target_surface_id"] == "snippet_content_view"


def test_gitlab_snippets_index_eligible_inline_listing():
    contract = _build(
        "gitlab_snippets_index",
        {},
        "https://gitlab.local/-/snippets",
    )
    assert contract["eligibility"] == {"status": "ineligible", "reason": "non_core_surface"}
    assert contract["mode"] == "inline_listing"
    assert contract["editor_method"] == "create_snippet"
    assert contract["target_surface_id"] == "snippet_title_list"


def test_gitlab_project_milestone_eligible_direct_detail():
    contract = _build(
        "gitlab_project_milestone",
        {"project_path": "byteblaze/dotfiles", "milestone_iid": "1"},
        "https://gitlab.local/byteblaze/dotfiles/-/milestones/1",
    )
    assert contract["eligibility"] == {"status": "ineligible", "reason": "non_core_surface"}
    assert contract["mode"] == "direct_detail"
    assert contract["editor_method"] == "update_milestone"
    assert contract["target_surface_id"] == "milestone_description_detail"


def test_gitlab_project_labels_eligible_inline_listing():
    contract = _build(
        "gitlab_project_labels",
        {"project_path": "primer/design"},
        "https://gitlab.local/primer/design/-/labels",
    )
    assert contract["eligibility"] == {"status": "ineligible", "reason": "non_core_surface"}
    assert contract["mode"] == "inline_listing"
    assert contract["editor_method"] == "create_label"
    assert contract["target_surface_id"] == "label_description_page"


def test_gitlab_group_eligible_direct_detail():
    contract = _build(
        "gitlab_group",
        {"group_path": "a11yproject"},
        "https://gitlab.local/a11yproject",
    )
    assert contract["eligibility"] == {"status": "ineligible", "reason": "non_core_surface"}
    assert contract["mode"] == "direct_detail"
    assert contract["editor_method"] == "update_group"
    assert contract["target_surface_id"] == "group_description_page"


def test_unresolved_kind_remains_ineligible():
    """Sanity guard: unknown kinds still surface as unresolved_target_resource."""
    contract = build_exposure_contract(
        benign_task_id="task-x",
        site="gitlab",
        benchmark="webarena_verified",
        benign_target_resource={
            "kind": None,
            "anchors": {},
            "start_url_resolved": None,
        },
    )
    assert contract["eligibility"]["status"] == "ineligible"
    assert contract["eligibility"]["reason"] == "unresolved_target_resource"


def test_signature_version_bumped_to_18():
    """Persisted Phase 2 plans built before comment ordering guards must invalidate."""
    from worldsim.phases.phase_2_exposure_contract import exposure_contract_signature

    sig = exposure_contract_signature()
    assert sig["version"] == 18


def test_impl_signature_observes_direct_preferred_token_patch(monkeypatch):
    from worldsim.phase_2.exposure_contract import _impl

    monkeypatch.setattr(_impl, "PREFERRED_TOKEN_ORDER", ("{patched_token}",))

    sig = _impl.exposure_contract_signature()

    assert sig["token_preference"] == ["{patched_token}"]


def test_package_signature_observes_impl_preferred_token_patch(monkeypatch):
    import worldsim.phase_2.exposure_contract as package
    from worldsim.phase_2.exposure_contract import _impl

    monkeypatch.setattr(_impl, "PREFERRED_TOKEN_ORDER", ("{package_patched_token}",))

    sig = package.exposure_contract_signature()

    assert sig["token_preference"] == ["{package_patched_token}"]


def test_legacy_signature_observes_facade_preferred_token_patch(monkeypatch):
    from worldsim.phases import phase_2_exposure_contract as legacy

    monkeypatch.setattr(legacy, "PREFERRED_TOKEN_ORDER", ("{legacy_patched_token}",))

    sig = legacy.exposure_contract_signature()

    assert sig["token_preference"] == ["{legacy_patched_token}"]
