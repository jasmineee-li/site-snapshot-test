"""Feature-local Site Behavior Contract tests for the test-only discussion forum Site."""

from __future__ import annotations

from tests.sites.behavior_contract.action_cards import assert_action_card_behavior
from tests.sites.behavior_contract.exact_resource_evidence import (
    assert_exact_resource_evidence,
)
from tests.sites.behavior_contract.feasibility import assert_feasibility_behavior
from tests.sites.behavior_contract.final_state_evaluation import (
    assert_final_state_evaluation_behavior,
)
from tests.sites.behavior_contract.fresh_anonymous_reader import (
    assert_fresh_anonymous_reader_behavior,
)
from tests.sites.behavior_contract.regular_participant_writer import (
    assert_regular_participant_writer_behavior,
)
from tests.sites.behavior_contract.site_targeting import assert_site_targeting_behavior
from tests.sites.synthetic_discussion_forum.action_card import action_card
from tests.sites.synthetic_discussion_forum.cases import evidence_case
from tests.sites.synthetic_discussion_forum.evaluator import evaluator_catalog
from tests.sites.synthetic_discussion_forum.feasibility import feasibility_policy
from tests.sites.synthetic_discussion_forum.reader import (
    fresh_anonymous_observation,
    writer_context_observation,
)
from tests.sites.synthetic_discussion_forum.site import (
    bound_site,
    foreign_origin_task,
    malformed_parent_task,
    valid_task,
)
from tests.sites.synthetic_discussion_forum.writer import editor, editor_spec, failing_editor


def test_synthetic_discussion_forum_site_targeting() -> None:
    assert_site_targeting_behavior(
        bound_site,
        valid_task,
        expected_site="synthetic_discussion_forum",
        expected_kind="thread",
        expected_anchors={"thread_id": "17"},
        expected_url="https://forum.test/threads/17",
        expected_profile_surface="thread_reply",
        expected_canonical_surface="comment.body",
        expected_editor_method="create_comment",
        foreign_origin_task=foreign_origin_task,
        malformed_parent_task=malformed_parent_task,
    )


def test_synthetic_discussion_forum_regular_participant_writer() -> None:
    assert_regular_participant_writer_behavior(
        editor,
        thread_id="17",
        actor="participant",
        body="A regular participant reply. TRACE-SIGNATURE",
        expected_comment_id="42",
        expected_resource_kind="comment",
        expected_parent_path="/threads/17",
        expected_read_surface_provenance="regular_participant_writer",
        expected_editor_method="create_comment",
        editor_spec=editor_spec,
        expected_surface_id="comment.body",
        cleanup_failure_editor=failing_editor,
    )


def test_synthetic_discussion_forum_feasibility() -> None:
    assert_feasibility_behavior(
        feasibility_policy,
        valid_task,
        instance_site_url="https://forum.test",
        expected_probe_url="https://forum.test/threads/17",
        expected_parent_id="17",
    )


def test_synthetic_discussion_forum_fresh_anonymous_reader() -> None:
    assert_fresh_anonymous_reader_behavior(
        bound_site,
        fresh_anonymous_observation(),
        writer_context_observation=writer_context_observation(),
        expected_reason="exact_comment_visible",
    )


def test_synthetic_discussion_forum_exact_resource_evidence() -> None:
    assert_exact_resource_evidence(
        bound_site,
        evidence_case,
        foreign_read_surface_url="https://foreign.invalid/threads/17",
    )


def test_synthetic_discussion_forum_final_state_evaluation() -> None:
    assert_final_state_evaluation_behavior(
        evaluator_catalog,
        evidence_case,
        expected_message="exact synthetic discussion forum final state",
    )


def test_synthetic_discussion_forum_action_card() -> None:
    assert_action_card_behavior(
        action_card,
        expected_site="synthetic_discussion_forum",
        expected_benchmark="webarena_verified",
        expected_carrier="comment.body",
        expected_action_kind="submit_comment",
    )
