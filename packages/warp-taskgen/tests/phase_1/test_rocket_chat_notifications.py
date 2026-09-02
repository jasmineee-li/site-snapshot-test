"""Action-contract tests for Rocket.Chat notifications."""

from __future__ import annotations

import pytest

from warp_taskgen.phase_1.rocket_chat_contracts import RocketChatContractError
from warp_taskgen.phase_1.rocket_chat_decisions import generate_rocket_chat_conversation

from ._rocket_chat_fakes import FakeRocketChatStore
from ._rocket_chat_notification_fakes import (
    FakeAuthenticatedRocketChatNotificationReader,
    FakeRocketChatNotificationWriter,
)

FakeAuthenticatedRocketChatReader = FakeAuthenticatedRocketChatNotificationReader
FakeRocketChatWriter = FakeRocketChatNotificationWriter


def test_notification_recipient_and_body_follow_the_current_decision() -> None:
    from warp_taskgen.phase_1.rocket_chat_notifications import (
        compile_rocket_chat_notification_task,
        derive_rocket_chat_notification,
    )

    conversation = generate_rocket_chat_conversation(
        corrected_owner="Priya",
        corrected_due_date="2026-09-18",
    )
    notification = derive_rocket_chat_notification(conversation)

    assert notification.recipient == "Priya"
    assert notification.room_id == conversation.room_id
    assert notification.thread_key == conversation.thread_key
    assert notification.author == conversation.writer_user
    assert notification.body == "Current decision: owner=Priya; due_date=2026-09-18."

    task = compile_rocket_chat_notification_task(conversation)
    assert task["task_kind"] == "rocket_chat_conversation_notification"
    assert task["notification"]["recipient"] == "Priya"
    assert task["notification"]["body"] == notification.body

    changed = generate_rocket_chat_conversation(
        corrected_owner="Jordan",
        corrected_due_date="2026-09-21",
    )
    changed_notification = derive_rocket_chat_notification(changed)
    assert changed_notification.recipient == "Jordan"
    assert changed_notification.body == "Current decision: owner=Jordan; due_date=2026-09-21."


def test_notification_generation_rejects_owner_that_is_not_a_safe_mention() -> None:
    conversation = generate_rocket_chat_conversation(corrected_owner="Priya Lee")

    from warp_taskgen.phase_1.rocket_chat_notifications import (
        compile_rocket_chat_notification_task,
    )

    with pytest.raises(RocketChatContractError, match="cannot be encoded"):
        compile_rocket_chat_notification_task(conversation)


def test_ordinary_writer_returns_exact_notification_and_reader_reads_shared_persisted_state() -> (
    None
):
    from warp_taskgen.phase_1.rocket_chat_contracts import RocketChatParticipantContext
    from warp_taskgen.phase_1.rocket_chat_notifications import (
        RocketChatNotificationObservation,
        derive_rocket_chat_notification,
    )

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
    writer = FakeRocketChatWriter(store, writer_context)
    seed_receipt = writer.seed_conversation(conversation)
    notification = derive_rocket_chat_notification(conversation)

    receipt = writer.send_notification(conversation, seed_receipt, notification)
    assert receipt.message_id == "rc-message-0004"
    assert receipt.message.recipient == "Priya"
    assert receipt.message.room_id == conversation.room_id
    assert receipt.message.author == conversation.writer_user
    assert receipt.message.thread_id == seed_receipt.messages[conversation.thread_key].message_id
    assert receipt.message.body == notification.body

    observation = FakeAuthenticatedRocketChatReader(store, reader_context).observe_notification(
        conversation,
        seed_receipt,
        receipt,
    )
    assert isinstance(observation, RocketChatNotificationObservation)
    assert observation.reader_context == reader_context
    assert observation.message_identity == receipt.message
    assert observation.message_id == receipt.message_id
