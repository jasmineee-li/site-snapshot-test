from __future__ import annotations

from worldsim.phases.phase_2_exposure_contract import (
    build_exposure_contract,
    materialize_seed_template_from_contract,
)


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

    assert contract["eligibility"]["status"] == "eligible"
    assert contract["mode"] == "listing_visible"
    assert contract["benign_read_url"] == "https://gitlab.local/search?search=theme&scope=issues"
    assert contract["verification"]["url"] == "https://gitlab.local/search?search=theme&scope=issues"
