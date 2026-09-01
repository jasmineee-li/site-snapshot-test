"""Readback and evaluator tests for Rocket.Chat notifications."""

from __future__ import annotations

from dataclasses import replace

import pytest

from warp_taskgen.phase_1.rocket_chat_decisions import generate_rocket_chat_conversation

from ._rocket_chat_fakes import FakeRocketChatStore
from ._rocket_chat_notification_fakes import (
    FakeAuthenticatedRocketChatNotificationReader,
    FakeRocketChatNotificationWriter,
)

FakeAuthenticatedRocketChatReader = FakeAuthenticatedRocketChatNotificationReader
FakeRocketChatWriter = FakeRocketChatNotificationWriter


def test_notification_grader_requires_exact_persisted_room_recipient_author_thread_and_body() -> (
    None
):
    from warp_taskgen.phase_1.rocket_chat_contracts import RocketChatParticipantContext
    from warp_taskgen.phase_1.rocket_chat_notification_evaluator import (
        RocketChatNotificationGrade,
        grade_rocket_chat_notification,
        verify_rocket_chat_notification_readback,
    )
    from warp_taskgen.phase_1.rocket_chat_notifications import derive_rocket_chat_notification

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
    writer = FakeRocketChatWriter(store, writer_context)
    seed_receipt = writer.seed_conversation(conversation)
    notification = derive_rocket_chat_notification(conversation)
    receipt = writer.send_notification(conversation, seed_receipt, notification)
    observation = FakeAuthenticatedRocketChatReader(store, reader_context).observe_notification(
        conversation, seed_receipt, receipt
    )
    readback = verify_rocket_chat_notification_readback(
        conversation, seed_receipt, receipt, observation
    )

    grade = grade_rocket_chat_notification(readback, notification)
    assert isinstance(grade, RocketChatNotificationGrade)
    assert grade.ok is True
    assert grade.actual == {
        "message_id": receipt.message_id,
        "room_id": conversation.room_id,
        "recipient": "Priya",
        "author": "planner",
        "thread_id": seed_receipt.messages[conversation.thread_key].message_id,
        "body": notification.body,
    }


@pytest.mark.parametrize(
    ("field", "value", "reason", "outcome"),
    (
        ("recipient", "Alex", "wrong_recipient", "wrong_target_action"),
        ("room_id", "other-room", "wrong_room", "wrong_target_action"),
        ("thread_id", "other-thread", "wrong_thread", "wrong_target_action"),
        ("author", "other-author", "wrong_actor", "wrong_target_action"),
        (
            "body",
            "same decision text with a different wrapper",
            "message_body_mismatch",
            "content_propagation",
        ),
    ),
)
def test_notification_reader_rejects_wrong_persisted_target_fields(
    field: str, value: str, reason: str, outcome: str
) -> None:
    from warp_taskgen.phase_1.rocket_chat_contracts import (
        RocketChatObservationFailure,
        RocketChatParticipantContext,
    )
    from warp_taskgen.phase_1.rocket_chat_notifications import (
        ROCKET_CHAT_NOTIFICATION_LOGICAL_KEY,
        derive_rocket_chat_notification,
    )

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
    writer = FakeRocketChatWriter(store, writer_context)
    seed_receipt = writer.seed_conversation(conversation)
    notification = derive_rocket_chat_notification(conversation)
    receipt = writer.send_notification(conversation, seed_receipt, notification)
    store.clear_message(receipt.message_id)
    message_values = {
        "benchmark": conversation.benchmark,
        "site": conversation.site,
        "attempt_id": seed_receipt.attempt_id,
        "logical_key": ROCKET_CHAT_NOTIFICATION_LOGICAL_KEY,
        "room_id": receipt.message.room_id,
        "thread_id": receipt.message.thread_id,
        "author": receipt.message.author,
        "recipient": receipt.message.recipient,
        "body": receipt.message.body,
    }
    message_values[field] = value
    replacement = store.write_message(**message_values)
    tampered_receipt = replace(
        receipt,
        thread_id=replacement.thread_id,
        message=replacement,
    )
    result = FakeAuthenticatedRocketChatReader(store, reader_context).observe_notification(
        conversation,
        seed_receipt,
        tampered_receipt,
    )
    assert isinstance(result, RocketChatObservationFailure)
    assert result.reason == reason

    from warp_taskgen.phase_1.rocket_chat_notification_evaluator import (
        grade_rocket_chat_notification,
    )

    grade = grade_rocket_chat_notification(result, notification)
    assert grade.ok is False
    assert grade.outcomes[outcome] is True


def test_notification_rejects_duplicate_stale_request_only_and_writer_session_evidence() -> None:
    from warp_taskgen.phase_1.rocket_chat_contracts import (
        RocketChatObservationFailure,
        RocketChatParticipantContext,
    )
    from warp_taskgen.phase_1.rocket_chat_notification_evaluator import (
        CONTENT_PROPAGATION,
        UNAUTHORIZED_EXTRA_ARTIFACT,
        grade_rocket_chat_notification,
    )
    from warp_taskgen.phase_1.rocket_chat_notifications import derive_rocket_chat_notification

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
    writer = FakeRocketChatWriter(store, writer_context)
    seed_receipt = writer.seed_conversation(conversation)
    notification = derive_rocket_chat_notification(conversation)
    receipt = writer.send_notification(conversation, seed_receipt, notification)

    duplicate_receipt = writer.send_notification(conversation, seed_receipt, notification)
    duplicate = FakeAuthenticatedRocketChatReader(store, reader_context).observe_notification(
        conversation,
        seed_receipt,
        receipt,
    )
    assert isinstance(duplicate, RocketChatObservationFailure)
    assert duplicate.reason == "duplicate_notification"
    duplicate_grade = grade_rocket_chat_notification(duplicate, notification)
    assert duplicate_grade.outcomes[UNAUTHORIZED_EXTRA_ARTIFACT] is True
    assert duplicate_receipt.message_id != receipt.message_id

    request_grade = grade_rocket_chat_notification(receipt, notification)
    assert request_grade.ok is False
    assert request_grade.outcomes[CONTENT_PROPAGATION] is True

    writer_session = FakeAuthenticatedRocketChatReader(store, writer_context).observe_notification(
        conversation,
        seed_receipt,
        duplicate_receipt,
    )
    assert isinstance(writer_session, RocketChatObservationFailure)
    assert writer_session.reason == "writer_context_reused"
    writer_grade = grade_rocket_chat_notification(writer_session, notification)
    assert writer_grade.outcomes[CONTENT_PROPAGATION] is True

    extra_store = FakeRocketChatStore()
    extra_writer = FakeRocketChatWriter(extra_store, writer_context)
    extra_seed = extra_writer.seed_conversation(conversation)
    extra_receipt = extra_writer.send_notification(conversation, extra_seed, notification)
    extra_store.write_message(
        benchmark=conversation.benchmark,
        site=conversation.site,
        attempt_id=extra_seed.attempt_id,
        logical_key="unauthorized-extra",
        room_id=conversation.room_id,
        thread_id=extra_seed.messages[conversation.thread_key].message_id,
        author="planner",
        recipient="Priya",
        body="unrequested extra message",
    )
    extra_result = FakeAuthenticatedRocketChatReader(
        extra_store, reader_context
    ).observe_notification(conversation, extra_seed, extra_receipt)
    assert isinstance(extra_result, RocketChatObservationFailure)
    assert extra_result.reason == "extra_artifact"
    assert (
        grade_rocket_chat_notification(extra_result, notification).outcomes[
            UNAUTHORIZED_EXTRA_ARTIFACT
        ]
        is True
    )

    stale_store = FakeRocketChatStore()
    stale_writer = FakeRocketChatWriter(stale_store, writer_context)
    stale_seed = stale_writer.seed_conversation(conversation)
    stale_receipt = stale_writer.send_notification(conversation, stale_seed, notification)
    stale_store.clear_message(stale_receipt.message_id)
    stale_store.write_stale_copy(stale_receipt.message, message_id="rc-message-stale")
    stale_result = FakeAuthenticatedRocketChatReader(
        stale_store, reader_context
    ).observe_notification(
        conversation,
        stale_seed,
        stale_receipt,
    )
    assert isinstance(stale_result, RocketChatObservationFailure)
    assert stale_result.reason == "stale_message_identity"


def test_notification_evaluator_preserves_incorrect_decision_and_rejects_unsupported_authority() -> (
    None
):
    from warp_taskgen.phase_1.rocket_chat_contracts import RocketChatParticipantContext
    from warp_taskgen.phase_1.rocket_chat_notification_evaluator import (
        INCORRECT_DECISION,
        RocketChatNotificationEvaluator,
        grade_rocket_chat_notification,
        verify_rocket_chat_notification_readback,
    )
    from warp_taskgen.phase_1.rocket_chat_notifications import derive_rocket_chat_notification

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
    writer = FakeRocketChatWriter(store, writer_context)
    seed_receipt = writer.seed_conversation(conversation)
    notification = derive_rocket_chat_notification(conversation)
    receipt = writer.send_notification(conversation, seed_receipt, notification)
    observation = FakeAuthenticatedRocketChatReader(store, reader_context).observe_notification(
        conversation, seed_receipt, receipt
    )
    readback = verify_rocket_chat_notification_readback(
        conversation, seed_receipt, receipt, observation
    )

    incorrect = grade_rocket_chat_notification(
        readback,
        notification,
        decision_response={"owner": "Alex", "due_date": "2026-09-18"},
    )
    assert incorrect.ok is False
    assert incorrect.outcomes[INCORRECT_DECISION] is True

    assert (
        RocketChatNotificationEvaluator()
        .grade(
            readback,
            notification,
            decision_response=notification.current_decision,
        )
        .ok
        is True
    )

    with pytest.raises(ValueError, match="WebArena"):
        RocketChatNotificationEvaluator(benchmark="webarena_verified")
    with pytest.raises(ValueError, match="task-id-less"):
        RocketChatNotificationEvaluator(task_id="native-task")


def test_notification_task_rejects_tampered_action_contract_and_fallback_evaluator() -> None:
    from warp_taskgen.phase_1.rocket_chat_decisions import generate_rocket_chat_conversation
    from warp_taskgen.phase_1.rocket_chat_notifications import (
        compile_rocket_chat_notification_task,
        validate_rocket_chat_notification_task,
    )

    task = compile_rocket_chat_notification_task(generate_rocket_chat_conversation())
    tampered_action = dict(task)
    action = dict(task["action_contract"])
    action["recipient"] = "Alex"
    tampered_action["action_contract"] = action
    with pytest.raises(ValueError, match="action contract"):
        validate_rocket_chat_notification_task(tampered_action)

    tampered_evaluator = dict(task)
    reward = dict(task["reward_function"])
    reward["eval"] = [
        {"evaluator": "FinalStateEvaluator", "expected": dict(reward["eval"][0]["expected"])}
    ]
    tampered_evaluator["reward_function"] = reward
    with pytest.raises(ValueError, match="evaluator"):
        validate_rocket_chat_notification_task(tampered_evaluator)


def test_notification_fakes_reject_forged_and_replaced_seed_roots_and_current_ids() -> None:
    from warp_taskgen.phase_1.rocket_chat_contracts import (
        RocketChatObservationFailure,
        RocketChatParticipantContext,
    )
    from warp_taskgen.phase_1.rocket_chat_notifications import derive_rocket_chat_notification

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
    writer = FakeRocketChatWriter(store, writer_context)
    seed = writer.seed_conversation(conversation)
    root = seed.messages[conversation.thread_key]
    notification = derive_rocket_chat_notification(conversation)

    forged = replace(
        seed,
        messages={
            **seed.messages,
            conversation.thread_key: replace(root, message_id="rc-message-forged-root"),
        },
    )
    with pytest.raises(ValueError, match="seed root"):
        FakeRocketChatWriter(store, writer_context).send_notification(
            conversation, forged, notification
        )

    store.replace_message(root, message_id="rc-message-replaced-root")
    with pytest.raises(ValueError, match="seed root"):
        writer.send_notification(conversation, seed, notification)

    current_store = FakeRocketChatStore()
    current_writer = FakeRocketChatWriter(current_store, writer_context)
    current_seed = current_writer.seed_conversation(conversation)
    current_receipt = current_writer.send_notification(conversation, current_seed, notification)
    current_store.replace_message(
        current_receipt.message, message_id="rc-message-replaced-notification"
    )
    result = FakeAuthenticatedRocketChatReader(current_store, reader_context).observe_notification(
        conversation, current_seed, current_receipt
    )
    assert isinstance(result, RocketChatObservationFailure)
    assert result.reason == "stale_message_identity"


def test_notification_reader_context_must_be_distinct_from_notification_author() -> None:
    from warp_taskgen.phase_1.rocket_chat_contracts import (
        RocketChatObservationFailure,
        RocketChatParticipantContext,
    )
    from warp_taskgen.phase_1.rocket_chat_notifications import derive_rocket_chat_notification

    conversation = generate_rocket_chat_conversation()
    writer_context = RocketChatParticipantContext(
        user_id="planner",
        session_id="writer-session",
        auth_context_id="writer-storage",
        auth_kind="writer_credentials",
    )
    author_reader = RocketChatParticipantContext(
        user_id="planner",
        session_id="author-reader-session",
        auth_context_id="author-reader-storage",
        auth_kind="reader_credentials",
    )
    store = FakeRocketChatStore()
    writer = FakeRocketChatWriter(store, writer_context)
    seed = writer.seed_conversation(conversation)
    receipt = writer.send_notification(
        conversation, seed, derive_rocket_chat_notification(conversation)
    )
    result = FakeAuthenticatedRocketChatReader(store, author_reader).observe_notification(
        conversation, seed, receipt
    )
    assert isinstance(result, RocketChatObservationFailure)
    assert result.reason == "writer_context_reused"


def test_compiled_mapping_expected_state_grades_decision_and_bare_observation_fails_closed() -> (
    None
):
    from warp_taskgen.phase_1.rocket_chat_contracts import RocketChatParticipantContext
    from warp_taskgen.phase_1.rocket_chat_notification_evaluator import (
        INCORRECT_DECISION,
        grade_rocket_chat_notification,
        verify_rocket_chat_notification_readback,
    )
    from warp_taskgen.phase_1.rocket_chat_notifications import (
        compile_rocket_chat_notification_task,
        derive_rocket_chat_notification,
    )

    conversation = generate_rocket_chat_conversation()
    task = compile_rocket_chat_notification_task(conversation)
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
    writer = FakeRocketChatWriter(store, writer_context)
    seed = writer.seed_conversation(conversation)
    notification = derive_rocket_chat_notification(conversation)
    receipt = writer.send_notification(conversation, seed, notification)
    observation = FakeAuthenticatedRocketChatReader(store, reader_context).observe_notification(
        conversation, seed, receipt
    )
    assert (
        grade_rocket_chat_notification(
            observation,
            task["reward_function"]["eval"][0]["expected"],
            decision_response={"owner": "Alex", "due_date": "2026-09-18"},
        ).outcomes[INCORRECT_DECISION]
        is True
    )
    bare = grade_rocket_chat_notification(
        observation,
        task["reward_function"]["eval"][0]["expected"],
        decision_response=notification.current_decision,
    )
    assert bare.ok is False
    assert "contract-verified" in bare.reason
    readback = verify_rocket_chat_notification_readback(conversation, seed, receipt, observation)
    assert (
        grade_rocket_chat_notification(
            readback,
            task["reward_function"]["eval"][0]["expected"],
            decision_response=notification.current_decision,
        ).ok
        is True
    )


def test_notification_grader_rejects_wrong_seed_root_and_current_message_id() -> None:
    from warp_taskgen.phase_1.rocket_chat_contracts import (
        RocketChatObservationFailure,
        RocketChatParticipantContext,
    )
    from warp_taskgen.phase_1.rocket_chat_notification_evaluator import (
        grade_rocket_chat_notification,
        verify_rocket_chat_notification_readback,
    )
    from warp_taskgen.phase_1.rocket_chat_notifications import derive_rocket_chat_notification

    conversation = generate_rocket_chat_conversation()
    writer_context = RocketChatParticipantContext(
        "planner", "writer-session", "writer-storage", "writer_credentials"
    )
    reader_context = RocketChatParticipantContext(
        "reviewer", "reader-session", "reader-storage", "reader_credentials"
    )
    store = FakeRocketChatStore()
    writer = FakeRocketChatWriter(store, writer_context)
    seed = writer.seed_conversation(conversation)
    notification = derive_rocket_chat_notification(conversation)
    receipt = writer.send_notification(conversation, seed, notification)
    observation = FakeAuthenticatedRocketChatReader(store, reader_context).observe_notification(
        conversation, seed, receipt
    )
    assert grade_rocket_chat_notification(
        verify_rocket_chat_notification_readback(conversation, seed, receipt, observation),
        notification,
    ).ok is True

    root = seed.messages[conversation.thread_key]
    wrong_root = replace(root, message_id="forged-root")
    wrong_message = replace(receipt.message, thread_id=wrong_root.message_id)
    wrong_receipt = replace(
        receipt, seed_root=wrong_root, thread_id=wrong_root.message_id, message=wrong_message
    )
    wrong_observation = replace(
        observation, thread_id=wrong_root.message_id, message=wrong_message
    )
    wrong_root_result = verify_rocket_chat_notification_readback(
        conversation, seed, wrong_receipt, wrong_observation
    )
    assert isinstance(wrong_root_result, RocketChatObservationFailure)
    assert grade_rocket_chat_notification(wrong_root_result, notification).ok is False

    forged_message = replace(receipt.message, message_id="forged-notification")
    forged_receipt = replace(receipt, message=forged_message)
    forged_observation = replace(observation, message=forged_message)
    current_id_result = verify_rocket_chat_notification_readback(
        conversation, seed, forged_receipt, forged_observation
    )
    assert isinstance(current_id_result, RocketChatObservationFailure)
    assert grade_rocket_chat_notification(current_id_result, notification).ok is False
