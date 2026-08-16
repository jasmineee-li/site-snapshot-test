"""Feature-local Site Behavior Contract tests for the test-only discussion forum Site."""

from __future__ import annotations

import pytest

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
from tests.sites.synthetic_discussion_forum.composition import (
    ACTION_KIND,
    BENCHMARK,
    CARRIER,
    SITE,
    USE_CASE,
    composition_check_request,
    static_composition_report,
)
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
from tests.sites.synthetic_discussion_forum.writer import (
    editor_spec,
    failing_editor,
    seed_registry,
)
from warp_taskgen.site_composition import (
    SiteCompositionCheckRequest,
    check_site_composition,
    default_site_compositions,
)
from warp_taskgen.sites import SiteCatalog


@pytest.fixture
def exact_request() -> SiteCompositionCheckRequest:
    request = composition_check_request()
    assert evidence_case.benchmark == request.benchmark
    assert evidence_case.site == request.site
    assert evidence_case.action_kind == request.action_kind
    return request


def test_synthetic_discussion_forum_static_composition(
    exact_request: SiteCompositionCheckRequest,
) -> None:
    request = exact_request
    report = static_composition_report()

    assert (request.benchmark, request.site, request.use_case) == (BENCHMARK, SITE, USE_CASE)
    assert (request.carrier, request.action_kind) == (CARRIER, ACTION_KIND)
    assert report.static_status == "complete"
    assert report.site_composition_digest is not None
    assert report.site_composition_digest.startswith("sha256:")
    assert report.finding("static_closure").outcome == "pass"
    report_payload = report.to_dict()
    assert report_payload["readiness_status"] == "blocked"
    assert report_payload["active_policy_checked"] is False
    assert report_payload["live_evidence_checked"] is False


def test_synthetic_discussion_forum_static_composition_does_not_activate_defaults(
    exact_request: SiteCompositionCheckRequest,
) -> None:
    removed = check_site_composition((), exact_request)

    assert removed.static_status == "invalid"
    assert removed.site_composition_digest is None
    assert SITE not in {item.site for item in default_site_compositions()}
    assert SiteCatalog().sites == ("gitlab", "reddit")


def test_synthetic_discussion_forum_site_targeting(
    exact_request: SiteCompositionCheckRequest,
) -> None:
    assert_site_targeting_behavior(
        bound_site,
        valid_task,
        expected_site=exact_request.site,
        expected_kind="thread",
        expected_anchors={"thread_id": "17"},
        expected_url="https://forum.test/threads/17",
        expected_profile_surface="thread_reply",
        expected_canonical_surface=str(exact_request.carrier),
        expected_editor_method="create_comment",
        foreign_origin_task=foreign_origin_task,
        malformed_parent_task=malformed_parent_task,
    )


def test_synthetic_discussion_forum_regular_participant_writer(
    exact_request: SiteCompositionCheckRequest,
) -> None:
    assert_regular_participant_writer_behavior(
        seed_registry,
        benchmark=exact_request.benchmark,
        site=exact_request.site,
        thread_id="17",
        actor="participant",
        body="A regular participant reply. TRACE-SIGNATURE",
        expected_comment_id="42",
        expected_resource_kind="comment",
        expected_parent_path="/threads/17",
        expected_read_surface_provenance="regular_participant_writer",
        expected_editor_method="create_comment",
        editor_spec=editor_spec,
        expected_surface_id=str(exact_request.carrier),
        cleanup_failure_editor=failing_editor,
    )


def test_synthetic_discussion_forum_feasibility(
    exact_request: SiteCompositionCheckRequest,
) -> None:
    assert feasibility_policy.benchmark == exact_request.benchmark
    assert feasibility_policy.site == exact_request.site
    assert_feasibility_behavior(
        feasibility_policy,
        valid_task,
        instance_site_url="https://forum.test",
        expected_probe_url="https://forum.test/threads/17",
        expected_parent_id="17",
    )


def test_synthetic_discussion_forum_fresh_anonymous_reader(
    exact_request: SiteCompositionCheckRequest,
) -> None:
    assert exact_request.site == SITE
    assert_fresh_anonymous_reader_behavior(
        bound_site,
        fresh_anonymous_observation(),
        writer_context_observation=writer_context_observation(),
        expected_reason="exact_comment_visible",
    )


def test_synthetic_discussion_forum_exact_resource_evidence(
    exact_request: SiteCompositionCheckRequest,
) -> None:
    assert evidence_case.site == exact_request.site
    registration = seed_registry.get(exact_request.benchmark, exact_request.site)
    assert registration is not None
    seeded_editor = registration.create({}, object())
    seed_result = seeded_editor.create_comment(
        thread_id=evidence_case.parent_id,
        actor=evidence_case.actor,
        body=evidence_case.body,
    )
    assert_exact_resource_evidence(
        bound_site,
        evidence_case,
        seed_result,
        foreign_read_surface_url="https://foreign.invalid/threads/17",
    )
    seeded_editor.cleanup()
    seeded_editor.cleanup()
    assert seeded_editor.cleaned is True


def test_synthetic_discussion_forum_final_state_evaluation(
    exact_request: SiteCompositionCheckRequest,
) -> None:
    assert evidence_case.action_kind == exact_request.action_kind
    assert_final_state_evaluation_behavior(
        evaluator_catalog,
        evidence_case,
        expected_message="exact synthetic discussion forum final state",
    )


def test_synthetic_discussion_forum_action_card(
    exact_request: SiteCompositionCheckRequest,
) -> None:
    assert_action_card_behavior(
        action_card,
        expected_site=exact_request.site,
        expected_benchmark=exact_request.benchmark,
        expected_carrier=str(exact_request.carrier),
        expected_action_kind=str(exact_request.action_kind),
    )
