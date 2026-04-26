from __future__ import annotations

from worldsim.editors._method_spec import FreeText
from worldsim.editors._registry import EditorMethodSpec
from worldsim.phases import phase_2_exposure_contract as pec
from worldsim.phases.phase_2_exposure_contract import (
    build_exposure_contract,
    materialize_seed_template_from_contract,
)
from worldsim.seeding import self_contained_adversarial_seed_error


def test_builds_gitlab_issue_exposure_contract_from_registry():
    contract = build_exposure_contract(
        benign_task_id="task-1",
        site="gitlab",
        benchmark="webarena_verified",
        benign_target_resource={
            "kind": "gitlab_issue",
            "anchors": {"project_id": "22", "issue_iid": "7"},
            "start_url_resolved": "https://gitlab.local/acme/demo/-/issues/7",
        },
    )

    assert contract["eligibility"]["status"] == "eligible"
    assert contract["seed_capability"]["status"] == "supported"
    assert contract["phase4_exposure"]["admissible"] is True
    assert contract["mode"] == "direct_detail"
    assert contract["target_surface_id"] == "note_on_issue"
    assert contract["editor_method"] == "create_issue_note"
    assert contract["payload_arg"] == "body"
    assert contract["verification"]["url"] == "https://gitlab.local/acme/demo/-/issues/7"

    seed = materialize_seed_template_from_contract(contract)
    call = seed["editor_calls"][0]
    assert call["method"] == "create_issue_note"
    assert call["args"]["project_id"] == "{benign_project_id}"
    assert call["args"]["issue_iid"] == "{benign_issue_iid}"
    assert call["args"]["body"] == "{{PAYLOAD_TEXT}}"


def test_prefers_always_shown_surface_when_multiple_specs_qualify(monkeypatch):
    conditional = EditorMethodSpec(
        benchmark="webarena_verified",
        site="gitlab",
        method="aaa_conditional_method",
        kinds=frozenset({"gitlab_issue"}),
        http=("POST", "/conditional"),
        bindings={"body": FreeText()},
        surface_id_per_kind={"gitlab_issue": "conditional_surface"},
        required_editor_args=("body",),
    )
    always_shown = EditorMethodSpec(
        benchmark="webarena_verified",
        site="gitlab",
        method="zzz_always_method",
        kinds=frozenset({"gitlab_issue"}),
        http=("POST", "/always"),
        bindings={"body": FreeText()},
        surface_id_per_kind={"gitlab_issue": "always_surface"},
        required_editor_args=("body",),
    )
    monkeypatch.setattr(pec, "iter_specs", lambda *, site, benchmark: [conditional, always_shown])
    monkeypatch.setattr(pec, "available_tokens_for_kind", lambda *args, **kwargs: frozenset())

    contract = build_exposure_contract(
        benign_task_id="task-visibility",
        site="gitlab",
        benchmark="webarena_verified",
        benign_target_resource={
            "kind": "gitlab_issue",
            "anchors": {},
            "start_url_resolved": "https://gitlab.local/acme/demo/-/issues/7",
        },
        surface_visibility_by_id={
            "conditional_surface": "conditional",
            "always_surface": "always_shown",
        },
    )

    assert contract["editor_method"] == "zzz_always_method"
    assert contract["target_surface_id"] == "always_surface"


def test_builds_reddit_submission_exposure_contract_from_registry():
    contract = build_exposure_contract(
        benign_task_id="task-2",
        site="reddit",
        benchmark="webarena_verified",
        benign_target_resource={
            "kind": "reddit_submission",
            "anchors": {"forum_name": "news", "submission_id": "13"},
            "start_url_resolved": "https://reddit.local/f/news/13",
        },
    )

    assert contract["eligibility"]["status"] == "eligible"
    assert contract["seed_capability"]["status"] == "supported"
    assert contract["phase4_exposure"]["admissible"] is True
    assert contract["mode"] == "direct_detail"
    assert contract["target_surface_id"] == "comment_body_thread"
    assert contract["editor_method"] == "create_comment"
    assert contract["editor_args_template"]["submission_id"] == "{benign_submission_id}"
    assert contract["editor_args_template"]["forum_name"] == "{benign_forum_name}"
    assert contract["editor_args_template"]["body"] == "{{PAYLOAD_TEXT}}"


def test_project_metadata_task_is_ineligible():
    contract = build_exposure_contract(
        benign_task_id="task-3",
        site="gitlab",
        benchmark="webarena_verified",
        benign_target_resource={
            "kind": "gitlab_project",
            "anchors": {"project_id": "22"},
            "start_url_resolved": "https://gitlab.local/acme/demo",
        },
    )

    assert contract["eligibility"]["status"] == "ineligible"
    assert contract["eligibility"]["reason"] == "kind_not_supported_for_exposure:gitlab_project"


def test_missing_benign_read_url_is_ineligible():
    contract = build_exposure_contract(
        benign_task_id="task-missing-url",
        site="gitlab",
        benchmark="webarena_verified",
        benign_target_resource={
            "kind": "gitlab_issue",
            "anchors": {"project_id": "22", "issue_iid": "7"},
        },
    )

    assert contract["mode"] == "direct_detail"
    assert contract["eligibility"] == {
        "status": "ineligible",
        "reason": "missing_benign_read_url",
    }


def test_l4_listing_preserves_benign_read_url_as_verification_target():
    contract = build_exposure_contract(
        benign_task_id="task-4_l4_0",
        site="gitlab",
        benchmark="webarena_verified",
        benign_target_resource={
            "kind": "gitlab_issue",
            "source_listing_kind": "gitlab_search_result",
            "anchors": {"project_id": "22", "issue_iid": "7"},
            "benign_read_url": "https://gitlab.local/search?search=theme&scope=issues",
            "start_url_resolved": "https://gitlab.local/acme/demo/-/issues/7",
            "seeded_detail_url": "https://gitlab.local/acme/demo/-/issues/7",
        },
    )

    assert contract["eligibility"] == {
        "status": "ineligible",
        "reason": "phase4_exposure:unforced_transitive_surface",
    }
    assert contract["seed_capability"]["status"] == "supported"
    assert contract["phase4_exposure"] == {
        "schema_version": 1,
        "seed_timing": "pre_task",
        "admissible": False,
        "reason": "unforced_transitive_surface",
        "encounter_surface": "seed_existing_child_detail",
        "requires_transition": True,
        "transition_forced_by_task": False,
        "requires_runtime_hook": False,
    }
    assert contract["mode"] == "bounded_transitive_existing"
    assert contract["benign_read_url"] == "https://gitlab.local/search?search=theme&scope=issues"
    assert (
        contract["verification"]["url"] == "https://gitlab.local/search?search=theme&scope=issues"
    )
    assert contract["verification"]["entry"]["url"] == (
        "https://gitlab.local/search?search=theme&scope=issues"
    )
    assert contract["verification"]["target"]["url"] == "https://gitlab.local/acme/demo/-/issues/7"


def test_task_forced_transitive_contract_is_phase4_admissible():
    contract = build_exposure_contract(
        benign_task_id="task-forced_l4_0",
        site="gitlab",
        benchmark="webarena_verified",
        benign_target_resource={
            "kind": "gitlab_issue",
            "source_listing_kind": "gitlab_search_result",
            "anchors": {"project_id": "22", "issue_iid": "7"},
            "benign_read_url": "https://gitlab.local/search?search=theme&scope=issues",
            "start_url_resolved": "https://gitlab.local/acme/demo/-/issues/7",
            "seeded_detail_url": "https://gitlab.local/acme/demo/-/issues/7",
            "transition_forced_by_task": True,
        },
    )

    assert contract["eligibility"]["status"] == "eligible"
    assert contract["phase4_exposure"]["admissible"] is True
    assert contract["phase4_exposure"]["requires_transition"] is True
    assert contract["phase4_exposure"]["transition_forced_by_task"] is True


def test_project_root_l4_record_is_ineligible_not_transitive():
    contract = build_exposure_contract(
        benign_task_id="task-project-root_l4_0",
        site="gitlab",
        benchmark="webarena_verified",
        benign_target_resource={
            "kind": "gitlab_issue",
            "source_listing_kind": "gitlab_search_result",
            "anchors": {"project_id": "22", "issue_iid": "7"},
            "benign_read_url": "https://gitlab.local/acme/demo",
            "start_url_resolved": "https://gitlab.local/acme/demo/-/issues/7",
            "seeded_detail_url": "https://gitlab.local/acme/demo/-/issues/7",
        },
    )

    assert contract["mode"] == "ineligible"
    assert contract["eligibility"] == {
        "status": "ineligible",
        "reason": "unsupported_transitive_entry:gitlab_search_result",
    }


def test_reddit_forum_contract_is_seedable_but_phase4_inadmissible():
    contract = build_exposure_contract(
        benign_task_id="task-reddit-forum",
        site="reddit",
        benchmark="webarena_verified",
        benign_target_resource={
            "kind": "reddit_forum",
            "anchors": {"forum_name": "deeplearning"},
            "start_url_resolved": "https://reddit.local/f/deeplearning",
        },
    )

    assert contract["eligibility"] == {
        "status": "ineligible",
        "reason": "phase4_exposure:unforced_transitive_child_surface",
    }
    assert contract["seed_capability"]["status"] == "supported"
    assert contract["phase4_exposure"] == {
        "schema_version": 1,
        "seed_timing": "pre_task",
        "admissible": False,
        "reason": "unforced_transitive_child_surface",
        "encounter_surface": "seed_created_child_detail",
        "requires_transition": True,
        "transition_forced_by_task": False,
        "requires_runtime_hook": False,
    }
    assert contract["mode"] == "bounded_transitive_created_child"
    assert contract["editor_method"] == "create_submission"
    assert contract["payload_arg"] == "body"
    assert contract["verification"]["entry"]["url"] == "https://reddit.local/f/deeplearning"
    assert contract["verification"]["target"]["url_source"] == (
        "seed_metadata.created_resource.url"
    )


def _eligible_reddit_submission_contract() -> dict:
    return build_exposure_contract(
        benign_task_id="reddit-1",
        site="reddit",
        benchmark="webarena_verified",
        benign_target_resource={
            "kind": "reddit_submission",
            "anchors": {"forum_name": "technology", "submission_id": "12345"},
            "start_url_resolved": "https://reddit.local/f/technology/12345",
        },
    )


def _eligible_gitlab_user_profile_contract() -> dict:
    return build_exposure_contract(
        benign_task_id="gitlab-user-1",
        site="gitlab",
        benchmark="webarena_verified",
        benign_target_resource={
            "kind": "gitlab_user_profile",
            "anchors": {"username": "byteblaze"},
            "start_url_resolved": "https://gitlab.local/byteblaze",
        },
    )


def _eligible_gitlab_milestone_contract() -> dict:
    return build_exposure_contract(
        benign_task_id="gitlab-milestone-1",
        site="gitlab",
        benchmark="webarena_verified",
        benign_target_resource={
            "kind": "gitlab_project_milestone",
            "anchors": {"project_path": "byteblaze/dotfiles", "milestone_iid": "1"},
            "start_url_resolved": "https://gitlab.local/byteblaze/dotfiles/-/milestones/1",
        },
    )


def test_materialize_preserves_benign_editor_calls_for_reddit_submission():
    benign_seed = {
        "mechanism": "none",
        "editor_calls": [
            {
                "site": "reddit",
                "method": "create_submission",
                "args": {
                    "forum_name": "technology",
                    "title": "AI and the Future",
                    "body": "Some body",
                },
            },
            {
                "site": "reddit",
                "method": "create_comment",
                "args": {
                    "submission_id": "{benign_submission_id}",
                    "forum_name": "{benign_forum_name}",
                    "body": "First comment",
                },
            },
        ],
    }
    contract = _eligible_reddit_submission_contract()

    out = materialize_seed_template_from_contract(contract, benign_seed=benign_seed)

    assert out["mechanism"] == "none"
    assert out["editor_calls"][:2] == benign_seed["editor_calls"]
    assert len(out["editor_calls"]) == 3
    assert out["editor_calls"][-1]["method"] == "create_comment"
    assert out["editor_calls"][-1]["site"] == "reddit"
    assert self_contained_adversarial_seed_error(benign_seed, out) is None


def test_materialize_preserves_benign_api_calls_for_gitlab_user_profile():
    benign_seed = {
        "mechanism": "api",
        "api_calls": [
            {
                "method": "PUT",
                "path": "/api/v4/users/2",
                "body": {"bio": "Open source developer."},
            }
        ],
    }
    contract = _eligible_gitlab_user_profile_contract()

    out = materialize_seed_template_from_contract(contract, benign_seed=benign_seed)

    assert out["mechanism"] == "api"
    assert out["api_calls"][0] == benign_seed["api_calls"][0]
    assert len(out["api_calls"]) == 2
    appended = out["api_calls"][1]
    # The path-resolution helper reuses the benign's PUT path so the
    # {user_id} placeholder Phase 1 already resolved survives intact.
    assert appended["path"] == "/api/v4/users/2"
    assert appended["method"] == "PUT"
    assert "bio" in appended["body"]
    assert appended["body"]["bio"] == "{{PAYLOAD_TEXT}}"
    assert self_contained_adversarial_seed_error(benign_seed, out) is None


def test_materialize_preserves_benign_api_calls_for_gitlab_milestone():
    benign_seed = {
        "mechanism": "api",
        "api_calls": [
            {
                "method": "POST",
                "path": "/api/v4/projects/1/milestones",
                "body": {"title": "WCAG", "description": "Accessibility milestone"},
            }
        ],
    }
    contract = _eligible_gitlab_milestone_contract()

    out = materialize_seed_template_from_contract(contract, benign_seed=benign_seed)

    assert out["mechanism"] == "api"
    assert out["api_calls"][:1] == benign_seed["api_calls"]
    assert len(out["api_calls"]) == 2
    appended = out["api_calls"][1]
    # update_milestone is PUT; the benign's POST path can't be reused so
    # the literal template falls through. Phase 2c feasibility will
    # surface that path is unresolved if it ever shipped.
    assert appended["method"] == "PUT"
    assert "{{PAYLOAD_TEXT}}" in appended["body"].get("description", "")
    assert self_contained_adversarial_seed_error(benign_seed, out) is None


def test_materialize_legacy_no_actions_unchanged():
    benign_seed = {"mechanism": "none"}
    contract = _eligible_reddit_submission_contract()

    out = materialize_seed_template_from_contract(contract, benign_seed=benign_seed)

    assert out["mechanism"] == "editor"
    assert len(out["editor_calls"]) == 1
    assert out["editor_calls"][0]["method"] == "create_comment"
    assert self_contained_adversarial_seed_error(benign_seed, out) is None


def test_materialize_without_benign_seed_unchanged():
    contract = _eligible_reddit_submission_contract()

    out = materialize_seed_template_from_contract(contract)

    assert out["mechanism"] == "editor"
    assert len(out["editor_calls"]) == 1
    assert out["editor_calls"][0]["method"] == "create_comment"
