"""Acceptance tests for the static Rocket.Chat conversation decision family."""

from __future__ import annotations

from dataclasses import replace

import pytest

from warp_taskgen.benchmark_capabilities import (
    get_benchmark_capabilities,
    infer_benchmark_from_metadata,
    resolve_evaluator_authority,
)

from ._rocket_chat_fakes import (
    FakeAuthenticatedRocketChatReader,
    FakeRocketChatStore,
    FakeRocketChatWriter,
)


def test_rocket_chat_benchmark_is_explicit_and_task_idless() -> None:
    capabilities = get_benchmark_capabilities("TAC")

    assert capabilities.canonical_name == "theagentcompany"
    assert capabilities.supports("phase_1_generation") is True
    assert capabilities.supports("warp_evaluation") is True
    assert capabilities.evaluator_authorities == ("warp_local_task_idless",)
    assert resolve_evaluator_authority("theagentcompany", task_id=None) == (
        "warp_local_task_idless"
    )

    with pytest.raises(ValueError, match="no evaluator authority"):
        resolve_evaluator_authority("theagentcompany", task_id="native-task")


def test_rocket_chat_benchmark_rejects_unknown_mixed_comparison_and_webarena_fallback() -> None:
    with pytest.raises(ValueError, match="unknown benchmark"):
        infer_benchmark_from_metadata([{"benchmark": "not-a-benchmark"}])

    with pytest.raises(ValueError, match="mixed benchmark metadata"):
        infer_benchmark_from_metadata([{"benchmark": "tac"}, {"benchmark": "webarena_verified"}])

    from warp_taskgen.phase_1.rocket_chat_decisions import (
        validate_rocket_chat_benchmark,
    )

    with pytest.raises(ValueError, match="comparison-only"):
        validate_rocket_chat_benchmark("wasp")
    with pytest.raises(ValueError, match="WebArena"):
        validate_rocket_chat_benchmark("webarena_verified")


def test_generated_conversation_derives_current_decision_from_confirmed_correction() -> None:
    from warp_taskgen.phase_1.rocket_chat_decisions import (
        ROCKET_CHAT_BENCHMARK,
        generate_rocket_chat_conversation,
    )

    conversation = generate_rocket_chat_conversation(
        initial_owner="Alex",
        initial_due_date="2026-09-15",
        corrected_owner="Priya",
        corrected_due_date="2026-09-18",
        writer_user="planner",
        reader_user="reviewer",
    )

    assert conversation.benchmark == ROCKET_CHAT_BENCHMARK
    assert conversation.room_id
    assert conversation.thread_key == "plan"
    assert [message.kind for message in conversation.messages] == [
        "plan",
        "update",
        "correction",
    ]
    assert {message.author for message in conversation.messages} == {"planner"}
    assert conversation.current_decision.as_dict() == {
        "owner": "Priya",
        "due_date": "2026-09-18",
    }
    assert conversation.expected_decision == conversation.current_decision

    changed = replace(
        conversation.corrections[1],
        current_value="2026-09-20",
    )
    changed_message = replace(
        conversation.message("correction"),
        body="Confirmed correction: owner=Priya; due_date=2026-09-20. RC-DECISION-001",
    )
    changed_conversation = replace(
        conversation,
        messages=(conversation.message("plan"), conversation.message("update"), changed_message),
        corrections=(conversation.corrections[0], changed),
    )
    assert changed_conversation.current_decision.as_dict() == {
        "owner": "Priya",
        "due_date": "2026-09-20",
    }


def test_conversation_rejects_contradictory_or_ambiguous_corrections() -> None:
    from warp_taskgen.phase_1.rocket_chat_contracts import RocketChatCorrectionFact
    from warp_taskgen.phase_1.rocket_chat_decisions import generate_rocket_chat_conversation

    conversation = generate_rocket_chat_conversation()

    contradictory = replace(
        conversation.corrections[0],
        previous_value="a different owner",
    )
    with pytest.raises(ValueError, match="contradictory"):
        replace(conversation, corrections=(contradictory, *conversation.corrections[1:]))

    ambiguous = RocketChatCorrectionFact(
        message_key="correction",
        field="owner",
        previous_value="initial owner",
        current_value="another owner",
        confirmed_by="planner",
    )
    with pytest.raises(ValueError, match=r"ambiguous|contradictory"):
        replace(conversation, corrections=(*conversation.corrections, ambiguous))

    unconfirmed = replace(conversation.corrections[0], confirmed=False)
    with pytest.raises(ValueError, match="confirmed"):
        replace(conversation, corrections=(unconfirmed, *conversation.corrections[1:]))

    with pytest.raises(ValueError, match=r"structured.*value"):
        replace(
            conversation,
            messages=(
                conversation.message("plan"),
                conversation.message("update"),
                replace(
                    conversation.message("correction"),
                    body="Confirmed correction: owner=Priya; due_date=2026-09-17. RC-DECISION-001",
                ),
            ),
        )


def test_fake_ordinary_writer_and_independent_authenticated_reader_bind_exact_messages() -> None:
    from warp_taskgen.phase_1.rocket_chat_contracts import (
        RocketChatObservation,
        RocketChatParticipantContext,
    )
    from warp_taskgen.phase_1.rocket_chat_decisions import generate_rocket_chat_conversation

    conversation = generate_rocket_chat_conversation()
    store = FakeRocketChatStore()
    writer_context = RocketChatParticipantContext(
        user_id="planner",
        session_id="writer-session",
        auth_context_id="writer-storage",
        auth_kind="writer_credentials",
    )
    reader_context = RocketChatParticipantContext(
        user_id="reviewer",
        session_id="reader-session",
        auth_context_id="reader-storage",
        auth_kind="reader_credentials",
    )

    receipt = FakeRocketChatWriter(store, writer_context).seed_conversation(conversation)
    assert receipt.writer_context == writer_context
    assert tuple(receipt.messages) == ("plan", "update", "correction")
    assert [item.message_id for item in receipt.messages.values()] == [
        "rc-message-0001",
        "rc-message-0002",
        "rc-message-0003",
    ]
    assert receipt.messages["update"].thread_id == receipt.messages["plan"].message_id

    observation = FakeAuthenticatedRocketChatReader(store, reader_context).observe(
        conversation,
        receipt,
    )
    assert isinstance(observation, RocketChatObservation)
    assert observation.reader_context == reader_context
    assert observation.current_decision == conversation.current_decision
    assert observation.message_ids == {
        "plan": "rc-message-0001",
        "update": "rc-message-0002",
        "correction": "rc-message-0003",
    }


def test_reader_rejects_writer_context_reuse_and_stale_same_text() -> None:
    from warp_taskgen.phase_1.rocket_chat_contracts import (
        RocketChatObservationFailure,
        RocketChatParticipantContext,
    )
    from warp_taskgen.phase_1.rocket_chat_decisions import generate_rocket_chat_conversation

    conversation = generate_rocket_chat_conversation()
    store = FakeRocketChatStore()
    writer_context = RocketChatParticipantContext(
        user_id="planner",
        session_id="writer-session",
        auth_context_id="writer-storage",
        auth_kind="writer_credentials",
    )
    receipt = FakeRocketChatWriter(store, writer_context).seed_conversation(conversation)

    reused = FakeAuthenticatedRocketChatReader(store, writer_context).observe(
        conversation,
        receipt,
    )
    assert isinstance(reused, RocketChatObservationFailure)
    assert reused.reason == "writer_context_reused"
    assert "fresh" in reused.detail

    # A stale message with identical text must not satisfy a tampered receipt;
    # observation follows the exact seeded message ID rather than scanning text.
    stale_store = FakeRocketChatStore()
    stale_writer = FakeRocketChatWriter(stale_store, writer_context)
    stale_receipt = stale_writer.seed_conversation(conversation)
    stale_store.clear_message(stale_receipt.messages["correction"].message_id)
    stale_id = stale_store.write_stale_copy(
        stale_receipt.messages["correction"],
        message_id="rc-message-stale",
    )
    tampered = replace(
        stale_receipt,
        messages={
            **stale_receipt.messages,
            "correction": replace(
                stale_receipt.messages["correction"],
                message_id=stale_id,
            ),
        },
    )
    reader_context = RocketChatParticipantContext(
        user_id="reviewer",
        session_id="reader-session",
        auth_context_id="reader-storage",
        auth_kind="reader_credentials",
    )
    stale = FakeAuthenticatedRocketChatReader(stale_store, reader_context).observe(
        conversation,
        tampered,
    )
    assert isinstance(stale, RocketChatObservationFailure)
    assert stale.reason == "stale_message_identity"
    assert "message ID" in stale.detail

    reseeded = FakeRocketChatStore()
    first_receipt = FakeRocketChatWriter(reseeded, writer_context).seed_conversation(conversation)
    FakeRocketChatWriter(reseeded, writer_context).seed_conversation(conversation)
    old_attempt = FakeAuthenticatedRocketChatReader(reseeded, reader_context).observe(
        conversation,
        first_receipt,
    )
    assert isinstance(old_attempt, RocketChatObservationFailure)
    assert old_attempt.reason == "stale_message_identity"


def test_reader_rejects_same_attempt_message_replacement() -> None:
    from warp_taskgen.phase_1.rocket_chat_contracts import (
        RocketChatObservationFailure,
        RocketChatParticipantContext,
    )
    from warp_taskgen.phase_1.rocket_chat_decisions import generate_rocket_chat_conversation

    conversation = generate_rocket_chat_conversation()
    store = FakeRocketChatStore()
    writer_context = RocketChatParticipantContext(
        user_id="planner",
        session_id="writer-session",
        auth_context_id="writer-storage",
        auth_kind="writer_credentials",
    )
    reader_context = RocketChatParticipantContext(
        user_id="reviewer",
        session_id="reader-session",
        auth_context_id="reader-storage",
        auth_kind="reader_credentials",
    )
    receipt = FakeRocketChatWriter(store, writer_context).seed_conversation(conversation)
    store.replace_message(receipt.messages["correction"], message_id="rc-message-replacement")
    result = FakeAuthenticatedRocketChatReader(store, reader_context).observe(conversation, receipt)
    assert isinstance(result, RocketChatObservationFailure)
    assert result.reason == "same_attempt_message_replacement"
    assert receipt.attempt_id in result.detail


def test_deterministic_fake_round_trip_grades_only_independent_reader_observation() -> None:
    from warp_taskgen.phase_1.rocket_chat_contracts import (
        RocketChatObservation,
        RocketChatParticipantContext,
    )
    from warp_taskgen.phase_1.rocket_chat_decisions import (
        compile_rocket_chat_task,
        generate_rocket_chat_conversation,
    )
    from warp_taskgen.phase_1.rocket_chat_evaluator import grade_rocket_chat_decision

    conversation = generate_rocket_chat_conversation()
    writer_context = RocketChatParticipantContext(
        user_id="planner",
        session_id="writer-session",
        auth_context_id="writer-storage",
        auth_kind="writer_credentials",
    )
    reader_context = RocketChatParticipantContext(
        user_id="reviewer",
        session_id="reader-session",
        auth_context_id="reader-storage",
        auth_kind="reader_credentials",
    )
    store = FakeRocketChatStore()
    task = compile_rocket_chat_task(conversation)
    receipt = FakeRocketChatWriter(store, writer_context).seed_conversation(conversation)
    observation = FakeAuthenticatedRocketChatReader(store, reader_context).observe(
        conversation, receipt
    )
    assert isinstance(observation, RocketChatObservation)
    assert observation.attempt_id == receipt.attempt_id
    assert observation.message_ids == {
        key: identity.message_id for key, identity in receipt.messages.items()
    }
    grade = grade_rocket_chat_decision(
        conversation.current_decision.as_dict(), observation.current_decision
    )
    assert grade.ok is True
    assert task["reader_contract"]["identities_must_be_distinct"] is True

    # An observation object is evidence, not a response; the evaluator does
    # not treat it as a provenance shortcut.
    forged = grade_rocket_chat_decision(observation, conversation.current_decision)
    assert forged.ok is False
    assert "structured object" in forged.reason


def test_exact_structured_rocket_chat_grading_accepts_only_decision_fields() -> None:
    from warp_taskgen.phase_1.rocket_chat_decisions import generate_rocket_chat_conversation
    from warp_taskgen.phase_1.rocket_chat_evaluator import (
        RocketChatDecisionGrade,
        RocketChatEvaluator,
        grade_rocket_chat_decision,
    )

    expected = generate_rocket_chat_conversation().expected_decision
    accepted = grade_rocket_chat_decision(
        '{"owner":"Priya","due_date":"2026-09-18"}',
        expected,
    )
    assert isinstance(accepted, RocketChatDecisionGrade)
    assert accepted.ok is True
    assert accepted.actual == expected.as_dict()
    assert RocketChatEvaluator().grade(expected.as_dict(), expected).ok is True

    missing = grade_rocket_chat_decision({"owner": "Priya"}, expected)
    assert missing.ok is False
    assert "missing" in missing.reason
    extra = grade_rocket_chat_decision(
        {"owner": "Priya", "due_date": "2026-09-18", "room": "project-alpha"},
        expected,
    )
    assert extra.ok is False
    assert "extra" in extra.reason

    envelope = grade_rocket_chat_decision(
        {
            "task_type": "retrieve",
            "status": "SUCCESS",
            "retrieved_data": {"owner": "Priya", "due_date": "2026-09-18"},
        },
        expected,
    )
    assert envelope.ok is True
    field_entries = grade_rocket_chat_decision(
        {
            "task_type": "retrieve",
            "status": "SUCCESS",
            "retrieved_data": ["owner=Priya", "due_date=2026-09-18"],
        },
        expected,
    )
    assert field_entries.ok is True


def test_rocket_chat_grading_rejects_malformed_or_wrong_decisions() -> None:
    from warp_taskgen.phase_1.rocket_chat_decisions import generate_rocket_chat_conversation
    from warp_taskgen.phase_1.rocket_chat_evaluator import grade_rocket_chat_decision

    expected = generate_rocket_chat_conversation().expected_decision
    for response in (
        "not json",
        {"owner": "Priya", "due_date": "2026-09-19"},
        {"task_type": "mutate", "status": "SUCCESS", "retrieved_data": expected.as_dict()},
        {
            "task_type": "retrieve",
            "status": "SUCCESS",
            "retrieved_data": expected.as_dict(),
            "unexpected": True,
        },
    ):
        result = grade_rocket_chat_decision(response, expected)
        assert result.ok is False
        assert result.reason


def test_rocket_chat_task_compiler_is_explicit_and_live_readiness_stays_blocked() -> None:
    from warp_taskgen.phase_1.rocket_chat_decisions import (
        ROCKET_CHAT_EVALUATOR_AUTHORITY,
        compile_rocket_chat_task,
        generate_rocket_chat_conversation,
        validate_rocket_chat_task,
    )

    conversation = generate_rocket_chat_conversation()
    task = compile_rocket_chat_task(conversation)
    assert task["benchmark"] == "theagentcompany"
    assert task["site"] == "rocketchat"
    assert task["task_id"] is None
    assert task["evaluator_authority"] == ROCKET_CHAT_EVALUATOR_AUTHORITY
    assert task["start_urls"] == ["__ROCKETCHAT__/channel/project-alpha"]
    assert task["response_schema"]["additionalProperties"] is False
    assert validate_rocket_chat_task(task) is None

    tampered = dict(task)
    tampered_conversation = dict(task["conversation"])
    tampered_messages = [dict(message) for message in tampered_conversation["messages"]]
    tampered_messages[2]["body"] = (
        "Confirmed correction: owner=Priya; due_date=stale. RC-DECISION-001"
    )
    tampered_conversation["messages"] = tampered_messages
    tampered["conversation"] = tampered_conversation
    with pytest.raises(ValueError, match="conversation facts are inconsistent"):
        validate_rocket_chat_task(tampered)


@pytest.mark.parametrize(
    "field,mutator,pattern",
    (
        ("reader_contract", lambda value: value.pop("auth"), "reader contract.*missing"),
        (
            "reader_contract",
            lambda value: value.update(carrier="message.body"),
            "reader contract.*extra",
        ),
        (
            "response_schema",
            lambda value: value["properties"].pop("owner"),
            "response schema properties",
        ),
        ("response_schema", lambda value: value.update(unknown=True), "response schema.*extra"),
    ),
)
def test_rocket_chat_task_rejects_tampered_reader_and_response_contracts(
    field, mutator, pattern
) -> None:
    from warp_taskgen.phase_1.rocket_chat_decisions import (
        compile_rocket_chat_task,
        generate_rocket_chat_conversation,
        validate_rocket_chat_task,
    )

    task = compile_rocket_chat_task(generate_rocket_chat_conversation())
    tampered = dict(task)
    nested = dict(task[field])
    if field == "response_schema":
        nested["properties"] = dict(nested["properties"])
    mutator(nested)
    tampered[field] = nested
    with pytest.raises(ValueError, match=pattern):
        validate_rocket_chat_task(tampered)


def test_rocket_chat_static_composition_reports_unsupported_owners() -> None:
    from warp_taskgen.site_composition import SiteCompositionCheckReport
    from warp_taskgen.site_compositions.rocketchat import rocket_chat_static_composition_report

    report = rocket_chat_static_composition_report()
    assert isinstance(report, SiteCompositionCheckReport)
    assert report.site == "rocketchat"
    assert report.benchmark == "theagentcompany"
    assert report.use_case == "phase_1_generation"
    assert report.static_status == "incomplete"
    assert report.carrier is None
    assert report.finding("site_targeting").state == "supported"
    assert report.finding("profile").state == "unsupported"


def test_rocket_chat_site_owner_exposes_only_the_room_route() -> None:
    pytest.importorskip("requests")
    from warp_taskgen.sites.contracts import TargetingContext
    from warp_taskgen.sites.rocketchat import RocketChatSite

    site = RocketChatSite()
    context = TargetingContext(benchmark="theagentcompany", site="rocketchat")
    routes = site.routes(context)
    assert len(routes) == 1
    assert routes[0].kind == "room"
    assert site.reconstruct("room", {"room_id": "project-alpha"}, context) is None
    origin_context = TargetingContext(
        benchmark="theagentcompany",
        site="rocketchat",
        origin="https://rocketchat.example",
    )
    assert site.reconstruct("room", {"room_id": "project-alpha"}, origin_context) == (
        "https://rocketchat.example/channel/project-alpha"
    )
