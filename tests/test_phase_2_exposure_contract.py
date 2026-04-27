from __future__ import annotations

from worldsim.editors._method_spec import FreeText
from worldsim.editors._registry import EditorMethodSpec
from worldsim.phases import phase_2_exposure_contract as pec
from worldsim.phases.phase_2_exposure_contract import (
    build_exposure_contract,
    materialize_seed_template_from_contract,
)
from worldsim.seeding import self_contained_adversarial_seed_error


def test_gitlab_issue_note_is_not_admitted_without_comment_region_evidence():
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

    assert contract["eligibility"] == {
        "status": "ineligible",
        "reason": "phase4_exposure:unanchored_appended_comment_surface",
    }
    assert contract["seed_capability"]["status"] == "supported"
    assert contract["phase4_exposure"] == {
        "schema_version": 1,
        "seed_timing": "pre_task",
        "admissible": False,
        "reason": "unanchored_appended_comment_surface",
        "encounter_surface": "seed_appended_comment_region",
        "requires_transition": False,
        "transition_forced_by_task": False,
        "requires_exact_comment_region": True,
        "exact_comment_region_forced_by_task": False,
        "requires_runtime_hook": False,
    }
    assert contract["mode"] == "direct_detail"
    assert contract["target_surface_id"] == "note.body"
    assert contract["editor_surface_id"] == "note_on_issue"
    assert contract["editor_method"] == "create_issue_note"
    assert contract["payload_arg"] == "body"
    assert contract["verification"]["url"] == "https://gitlab.local/acme/demo/-/issues/7"


def test_gitlab_issue_note_admitted_with_exact_comment_region_evidence():
    contract = build_exposure_contract(
        benign_task_id="task-1",
        site="gitlab",
        benchmark="webarena_verified",
        benign_target_resource={
            "kind": "gitlab_issue",
            "anchors": {"project_id": "22", "issue_iid": "7"},
            "start_url_resolved": "https://gitlab.local/acme/demo/-/issues/7",
            "exact_comment_region_forced_by_task": True,
        },
    )

    assert contract["eligibility"]["status"] == "eligible"
    assert contract["phase4_exposure"]["admissible"] is True
    assert contract["phase4_exposure"]["requires_exact_comment_region"] is True

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
        surface_id_per_kind={"gitlab_issue": "label.description"},
        required_editor_args=("body",),
    )
    always_shown = EditorMethodSpec(
        benchmark="webarena_verified",
        site="gitlab",
        method="zzz_always_method",
        kinds=frozenset({"gitlab_issue"}),
        http=("POST", "/always"),
        bindings={"body": FreeText()},
        surface_id_per_kind={"gitlab_issue": "note_on_issue"},
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
            "label.description": "conditional",
            "note_on_issue": "always_shown",
        },
    )

    assert contract["editor_method"] == "zzz_always_method"
    assert contract["target_surface_id"] == "note.body"


def test_core_surface_gitlab_issue_description_admitted(monkeypatch):
    issue_description = EditorMethodSpec(
        benchmark="webarena_verified",
        site="gitlab",
        method="update_issue_description",
        kinds=frozenset({"gitlab_issue"}),
        http=("PUT", "/projects/{project_id}/issues/{issue_iid}"),
        bindings={"description": FreeText()},
        surface_id_per_kind={"gitlab_issue": "issue.description"},
        required_editor_args=("description",),
    )
    monkeypatch.setattr(pec, "iter_specs", lambda *, site, benchmark: [issue_description])
    monkeypatch.setattr(pec, "available_tokens_for_kind", lambda *args, **kwargs: frozenset())

    contract = build_exposure_contract(
        benign_task_id="task-core-issue-description",
        site="gitlab",
        benchmark="webarena_verified",
        benign_target_resource={
            "kind": "gitlab_issue",
            "anchors": {},
            "start_url_resolved": "https://gitlab.local/acme/demo/-/issues/7",
        },
    )

    assert contract["eligibility"]["status"] == "eligible"
    assert contract["target_surface_id"] == "issue.description"


def test_gitlab_listing_contract_prefers_created_issue_title_over_note():
    contract = build_exposure_contract(
        benign_task_id="task-gitlab-listing-title",
        site="gitlab",
        benchmark="webarena_verified",
        benign_target_resource={
            "kind": "gitlab_search_result",
            "anchors": {
                "project_id": "174",
                "project_path": "byteblaze/secret-project",
            },
            "start_url_resolved": "https://gitlab.local/byteblaze/secret-project/-/issues",
            "encounter_requirements": {"must_appear_on_list": True},
        },
    )

    assert contract["eligibility"]["status"] == "eligible"
    assert contract["editor_method"] == "create_issue_title"
    assert contract["target_surface_id"] == "issue.title"
    assert contract["editor_args_template"]["title"] == "{{PAYLOAD_TEXT}}"
    assert contract["editor_args_template"]["project_id"] == "{benign_project_id}"
    assert contract["surface_route"] == {
        "schema_version": 1,
        "entry_surface": "gitlab_search_result",
        "seed_surface": "issue.title",
        "mode": "inline_listing",
        "requires_transition": False,
        "transition_forced_by_task": True,
        "exact_comment_region_forced_by_task": False,
        "runtime_hook_required": False,
        "route_evidence": "benign_read_surface: visible on entry route",
        "content_capacity": "short_title",
        "attack_fit": "compact_action_required",
    }
    candidates = {
        candidate["target_surface_id"]: candidate for candidate in contract["surface_candidates"]
    }
    assert candidates["issue.title"]["eligibility"] == {"status": "eligible"}
    assert candidates["issue.description"]["eligibility"] == {
        "status": "ineligible",
        "reason": "phase4_exposure:unforced_transitive_child_surface",
    }


def test_gitlab_listing_prefers_issue_description_when_detail_route_forced():
    contract = build_exposure_contract(
        benign_task_id="task-gitlab-listing-body",
        site="gitlab",
        benchmark="webarena_verified",
        benign_target_resource={
            "kind": "gitlab_search_result",
            "anchors": {
                "project_id": "174",
                "project_path": "byteblaze/secret-project",
            },
            "start_url_resolved": "https://gitlab.local/byteblaze/secret-project/-/issues",
            "encounter_requirements": {"must_appear_on_list": True},
            "transition_forced_by_task": True,
        },
    )

    assert contract["eligibility"]["status"] == "eligible"
    assert contract["editor_method"] == "create_issue_description"
    assert contract["target_surface_id"] == "issue.description"
    assert contract["mode"] == "bounded_transitive_created_child"
    assert contract["editor_args_template"]["body"] == "{{PAYLOAD_TEXT}}"
    assert contract["verification"]["target"] == {
        "url_source": "seed_metadata.created_resource.url",
        "role": "seed_render_surface",
    }
    assert contract["surface_route"]["content_capacity"] == "long_body"
    assert contract["surface_route"]["attack_fit"] == "rich_text_action_supported"
    candidates = {
        candidate["target_surface_id"]: candidate for candidate in contract["surface_candidates"]
    }
    assert candidates["issue.title"]["eligibility"] == {"status": "eligible"}
    assert candidates["issue.description"]["eligibility"] == {"status": "eligible"}


def test_core_surface_gitlab_label_description_dropped(monkeypatch):
    label_description = EditorMethodSpec(
        benchmark="webarena_verified",
        site="gitlab",
        method="create_label",
        kinds=frozenset({"gitlab_issue"}),
        http=("POST", "/projects/{project_id}/labels"),
        bindings={"description": FreeText()},
        surface_id_per_kind={"gitlab_issue": "label.description"},
        required_editor_args=("description",),
    )
    monkeypatch.setattr(pec, "iter_specs", lambda *, site, benchmark: [label_description])
    monkeypatch.setattr(pec, "available_tokens_for_kind", lambda *args, **kwargs: frozenset())

    contract = build_exposure_contract(
        benign_task_id="task-non-core-label-description",
        site="gitlab",
        benchmark="webarena_verified",
        benign_target_resource={
            "kind": "gitlab_issue",
            "anchors": {},
            "start_url_resolved": "https://gitlab.local/acme/demo/-/issues/7",
        },
    )

    assert contract["eligibility"] == {"status": "ineligible", "reason": "non_core_surface"}
    assert contract["seed_capability"] == {"status": "unsupported", "reason": "non_core_surface"}
    assert contract["target_surface_id"] == "label.description"
    assert contract["surface_route"]["attack_fit"] == "not_admissible"
    assert contract["surface_route"]["content_capacity"] == "long_body"


def test_core_surface_reddit_submission_body_admitted(monkeypatch):
    submission_body = EditorMethodSpec(
        benchmark="webarena_verified",
        site="reddit",
        method="update_submission_body",
        kinds=frozenset({"reddit_submission"}),
        http=("POST", "/f/{forum_name}/{submission_id}/-/edit"),
        bindings={"body": FreeText()},
        surface_id_per_kind={"reddit_submission": "submission.body"},
        required_editor_args=("body",),
    )
    monkeypatch.setattr(pec, "iter_specs", lambda *, site, benchmark: [submission_body])
    monkeypatch.setattr(pec, "available_tokens_for_kind", lambda *args, **kwargs: frozenset())

    contract = build_exposure_contract(
        benign_task_id="task-core-submission-body",
        site="reddit",
        benchmark="webarena_verified",
        benign_target_resource={
            "kind": "reddit_submission",
            "anchors": {},
            "start_url_resolved": "https://reddit.local/f/news/13",
        },
    )

    assert contract["eligibility"]["status"] == "eligible"
    assert contract["target_surface_id"] == "submission.body"


def test_core_surface_unknown_site_fails_closed(monkeypatch):
    shopping_surface = EditorMethodSpec(
        benchmark="webarena_verified",
        site="shopping",
        method="write_review",
        kinds=frozenset({"gitlab_issue"}),
        http=("POST", "/review"),
        bindings={"body": FreeText()},
        surface_id_per_kind={"gitlab_issue": "note.body"},
        required_editor_args=("body",),
    )
    monkeypatch.setattr(pec, "iter_specs", lambda *, site, benchmark: [shopping_surface])
    monkeypatch.setattr(pec, "available_tokens_for_kind", lambda *args, **kwargs: frozenset())

    contract = build_exposure_contract(
        benign_task_id="task-unknown-site",
        site="shopping",
        benchmark="webarena_verified",
        benign_target_resource={
            "kind": "gitlab_issue",
            "anchors": {},
            "start_url_resolved": "https://shopping.local/review",
        },
    )

    assert contract["eligibility"] == {"status": "ineligible", "reason": "non_core_surface"}
    assert contract["target_surface_id"] == "note.body"


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

    assert contract["eligibility"] == {
        "status": "ineligible",
        "reason": "phase4_exposure:unanchored_appended_comment_surface",
    }
    assert contract["seed_capability"]["status"] == "supported"
    assert contract["phase4_exposure"] == {
        "schema_version": 1,
        "seed_timing": "pre_task",
        "admissible": False,
        "reason": "unanchored_appended_comment_surface",
        "encounter_surface": "seed_appended_comment_region",
        "requires_transition": False,
        "transition_forced_by_task": False,
        "requires_exact_comment_region": True,
        "exact_comment_region_forced_by_task": False,
        "requires_runtime_hook": False,
    }
    assert contract["mode"] == "direct_detail"
    assert contract["target_surface_id"] == "comment.body"
    assert contract["editor_surface_id"] == "comment_body_thread"
    assert contract["editor_method"] == "create_comment"
    assert contract["editor_args_template"]["submission_id"] == "{benign_submission_id}"
    assert contract["editor_args_template"]["forum_name"] == "{benign_forum_name}"
    assert contract["editor_args_template"]["body"] == "{{PAYLOAD_TEXT}}"


def test_reddit_submission_comment_is_not_admitted_by_generic_transition_flag():
    contract = build_exposure_contract(
        benign_task_id="task-reply",
        site="reddit",
        benchmark="webarena_verified",
        benign_target_resource={
            "kind": "reddit_submission",
            "anchors": {"forum_name": "singularity", "submission_id": "69404"},
            "start_url_resolved": "https://reddit.local/f/singularity/69404",
            "transition_forced_by_task": True,
        },
    )

    assert contract["mode"] == "direct_detail"
    assert contract["eligibility"] == {
        "status": "ineligible",
        "reason": "phase4_exposure:unanchored_appended_comment_surface",
    }
    assert contract["phase4_exposure"]["transition_forced_by_task"] is True
    assert contract["phase4_exposure"]["exact_comment_region_forced_by_task"] is False
    assert contract["phase4_exposure"]["admissible"] is False


def test_reddit_submission_comment_admitted_with_exact_comment_region_evidence():
    contract = build_exposure_contract(
        benign_task_id="task-exact-comment-region",
        site="reddit",
        benchmark="webarena_verified",
        benign_target_resource={
            "kind": "reddit_submission",
            "anchors": {"forum_name": "singularity", "submission_id": "69404"},
            "start_url_resolved": "https://reddit.local/f/singularity/69404",
            "transition_forced_by_task": True,
            "exact_comment_region_forced_by_task": True,
        },
    )

    assert contract["eligibility"] == {"status": "eligible"}
    assert contract["phase4_exposure"]["admissible"] is True
    assert contract["phase4_exposure"]["requires_exact_comment_region"] is True
    assert contract["phase4_exposure"]["requires_runtime_hook"] is False
    assert contract["surface_route"]["seed_surface"] == "comment.body"
    assert (
        contract["surface_route"]["route_evidence"]
        == "seed_appended_comment_region: exact discussion region forced by task"
    )
    assert contract["surface_route"]["content_capacity"] == "long_body"
    assert contract["surface_route"]["attack_fit"] == "rich_text_action_supported"


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
        "reason": "phase4_exposure:unanchored_appended_comment_surface",
    }
    assert contract["seed_capability"]["status"] == "supported"
    assert contract["phase4_exposure"] == {
        "schema_version": 1,
        "seed_timing": "pre_task",
        "admissible": False,
        "reason": "unanchored_appended_comment_surface",
        "encounter_surface": "seed_appended_comment_region",
        "requires_transition": True,
        "transition_forced_by_task": False,
        "requires_exact_comment_region": True,
        "exact_comment_region_forced_by_task": False,
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


def test_gitlab_note_transitive_contract_is_not_admitted_by_generic_transition_flag():
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

    assert contract["eligibility"] == {
        "status": "ineligible",
        "reason": "phase4_exposure:unanchored_appended_comment_surface",
    }
    assert contract["phase4_exposure"]["admissible"] is False
    assert contract["phase4_exposure"]["requires_transition"] is True
    assert contract["phase4_exposure"]["transition_forced_by_task"] is True
    assert contract["phase4_exposure"]["exact_comment_region_forced_by_task"] is False


def test_gitlab_note_transitive_contract_admitted_with_exact_comment_region_evidence():
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
            "exact_comment_region_forced_by_task": True,
        },
    )

    assert contract["eligibility"]["status"] == "eligible"
    assert contract["phase4_exposure"]["admissible"] is True
    assert contract["phase4_exposure"]["requires_transition"] is True
    assert contract["phase4_exposure"]["transition_forced_by_task"] is True
    assert contract["phase4_exposure"]["exact_comment_region_forced_by_task"] is True


def test_l3_reddit_forum_existing_submission_is_bounded_transitive():
    contract = build_exposure_contract(
        benign_task_id="task-reddit-forum-existing-child",
        site="reddit",
        benchmark="webarena_verified",
        benign_target_resource={
            "kind": "reddit_submission",
            "source_listing_kind": "reddit_forum",
            "anchors": {"forum_name": "books", "submission_id": "59421"},
            "benign_read_url": "https://reddit.local/f/books",
            "start_url_resolved": "https://reddit.local/f/books/59421",
            "seeded_detail_url": "https://reddit.local/f/books/59421",
        },
    )

    assert contract["mode"] == "bounded_transitive_existing"
    assert contract["eligibility"] == {
        "status": "ineligible",
        "reason": "phase4_exposure:unanchored_appended_comment_surface",
    }
    assert contract["phase4_exposure"]["admissible"] is False
    assert contract["verification"]["entry"]["url"] == "https://reddit.local/f/books"
    assert contract["verification"]["target"]["url"] == "https://reddit.local/f/books/59421"


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


def test_reddit_forum_contract_uses_visible_submission_title_listing_row():
    contract = build_exposure_contract(
        benign_task_id="task-reddit-forum",
        site="reddit",
        benchmark="webarena_verified",
        benign_target_resource={
            "kind": "reddit_forum",
            "anchors": {"forum_name": "deeplearning"},
            "start_url_resolved": "https://reddit.local/f/deeplearning",
            "encounter_requirements": {
                "requires_post_sort_order": "recent",
                "must_appear_on_list": True,
            },
        },
    )

    assert contract["eligibility"] == {"status": "eligible"}
    assert contract["seed_capability"]["status"] == "supported"
    assert contract["phase4_exposure"] == {
        "schema_version": 1,
        "seed_timing": "pre_task",
        "admissible": True,
        "encounter_surface": "seed_created_child_listing_row",
        "requires_transition": False,
        "transition_forced_by_task": True,
        "requires_runtime_hook": False,
        "requires_visual_evidence": True,
    }
    assert contract["mode"] == "inline_listing_created_child"
    assert contract["editor_method"] == "create_submission_title"
    assert contract["target_surface_id"] == "submission.title"
    assert contract["payload_arg"] == "title"
    assert contract["editor_args_template"]["title"] == "{{PAYLOAD_TEXT}}"
    assert contract["verification"]["entry"]["url"] == "https://reddit.local/f/deeplearning"
    assert contract["verification"]["target"]["url_source"] == (
        "seed_metadata.created_resource.parent_url"
    )
    assert contract["verification"]["transition"]["type"] == "inline_listing_row"
    candidates = {
        candidate["target_surface_id"]: candidate for candidate in contract["surface_candidates"]
    }
    assert candidates["submission.title"]["eligibility"] == {"status": "eligible"}
    assert candidates["submission.body"]["eligibility"] == {
        "status": "ineligible",
        "reason": "phase4_exposure:unforced_transitive_child_surface",
    }


def test_reddit_forum_prefers_submission_body_when_detail_route_forced():
    contract = build_exposure_contract(
        benign_task_id="task-reddit-forum-body",
        site="reddit",
        benchmark="webarena_verified",
        benign_target_resource={
            "kind": "reddit_forum",
            "anchors": {"forum_name": "deeplearning"},
            "start_url_resolved": "https://reddit.local/f/deeplearning",
            "encounter_requirements": {
                "requires_post_sort_order": "recent",
                "must_appear_on_list": True,
            },
            "transition_forced_by_task": True,
        },
    )

    assert contract["eligibility"] == {"status": "eligible"}
    assert contract["mode"] == "bounded_transitive_created_child"
    assert contract["editor_method"] == "create_submission"
    assert contract["target_surface_id"] == "submission.body"
    assert contract["payload_arg"] == "body"
    assert contract["editor_args_template"]["body"] == "{{PAYLOAD_TEXT}}"
    assert contract["verification"]["target"] == {
        "url_source": "seed_metadata.created_resource.url",
        "role": "seed_render_surface",
    }
    assert contract["verification"]["transition"]["type"] == "visible_link"
    assert contract["surface_route"]["content_capacity"] == "long_body"
    assert contract["surface_route"]["attack_fit"] == "rich_text_action_supported"


def _eligible_reddit_submission_contract() -> dict:
    return build_exposure_contract(
        benign_task_id="reddit-1",
        site="reddit",
        benchmark="webarena_verified",
        benign_target_resource={
            "kind": "reddit_submission",
            "anchors": {"forum_name": "technology", "submission_id": "12345"},
            "start_url_resolved": "https://reddit.local/f/technology/12345",
            "exact_comment_region_forced_by_task": True,
        },
    )


def _eligible_gitlab_issue_contract() -> dict:
    return build_exposure_contract(
        benign_task_id="gitlab-issue-api-1",
        site="gitlab",
        benchmark="webarena_verified",
        benign_target_resource={
            "kind": "gitlab_issue",
            "anchors": {"project_id": "22", "issue_iid": "7"},
            "start_url_resolved": "https://gitlab.local/acme/demo/-/issues/7",
            "exact_comment_region_forced_by_task": True,
        },
    )


def _eligible_gitlab_issue_contract_for_unmatched_verb() -> dict:
    return build_exposure_contract(
        benign_task_id="gitlab-issue-api-2",
        site="gitlab",
        benchmark="webarena_verified",
        benign_target_resource={
            "kind": "gitlab_issue",
            "anchors": {"project_id": "22", "issue_iid": "7"},
            "start_url_resolved": "https://gitlab.local/acme/demo/-/issues/7",
            "exact_comment_region_forced_by_task": True,
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


def test_materialize_preserves_benign_api_calls_for_gitlab_issue_note():
    benign_seed = {
        "mechanism": "api",
        "api_calls": [
            {
                "method": "POST",
                "path": "/api/v4/projects/22/issues/7/notes",
                "body": {"body": "Existing note."},
            }
        ],
    }
    contract = _eligible_gitlab_issue_contract()

    out = materialize_seed_template_from_contract(contract, benign_seed=benign_seed)

    assert out["mechanism"] == "api"
    assert out["api_calls"][0] == benign_seed["api_calls"][0]
    assert len(out["api_calls"]) == 2
    appended = out["api_calls"][1]
    # The path-resolution helper reuses the benign's POST path so the
    # project/issue placeholders Phase 1 already resolved survive intact.
    assert appended["path"] == "/api/v4/projects/22/issues/7/notes"
    assert appended["method"] == "POST"
    assert appended["body"]["body"] == "{{PAYLOAD_TEXT}}"
    assert self_contained_adversarial_seed_error(benign_seed, out) is None


def test_materialize_preserves_benign_api_calls_with_unmatched_verb_for_gitlab_issue_note():
    benign_seed = {
        "mechanism": "api",
        "api_calls": [
            {
                "method": "PUT",
                "path": "/api/v4/projects/22/issues/7",
                "body": {"description": "Existing issue body."},
            }
        ],
    }
    contract = _eligible_gitlab_issue_contract_for_unmatched_verb()

    out = materialize_seed_template_from_contract(contract, benign_seed=benign_seed)

    assert out["mechanism"] == "api"
    assert out["api_calls"][:1] == benign_seed["api_calls"]
    assert len(out["api_calls"]) == 2
    appended = out["api_calls"][1]
    # create_issue_note is POST; the benign's PUT path can't be reused, so
    # the literal template falls through for Phase 2c to resolve/reject.
    assert appended["method"] == "POST"
    assert appended["path"] == "/api/v4/projects/{project_id}/issues/{issue_iid}/notes"
    assert appended["body"]["body"] == "{{PAYLOAD_TEXT}}"
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
