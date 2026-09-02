from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

import pytest

from warp_taskgen.phase_1.rocket_chat_contracts import RocketChatObservationFailure
from warp_taskgen.phase_1.rocket_chat_decisions import generate_rocket_chat_conversation
from warp_taskgen.phase_1.rocket_chat_notification_evaluator import (
    CONTENT_PROPAGATION,
    INCORRECT_DECISION,
    UNAUTHORIZED_EXTRA_ARTIFACT,
    WRONG_TARGET_ACTION,
    grade_rocket_chat_notification,
    verify_rocket_chat_notification_readback,
)
from warp_taskgen.phase_1.rocket_chat_notifications import (
    RocketChatNotificationObservation,
    derive_rocket_chat_notification,
)
from warp_taskgen.rewards import run_reward_function
from warp_taskgen.sites.rocketchat_notification_runtime import (
    RocketChatHttpNotificationReader,
    RocketChatHttpNotificationWriter,
)
from warp_taskgen.sites.rocketchat_runtime import (
    RocketChatAuthSession,
    RocketChatCredentials,
    RocketChatHttpWriter,
    RocketChatTransportError,
)


@dataclass
class MentioningTransport:
    rows: list[dict[str, Any]] | None = None
    sender_username: str = "planner"
    next_id: int = 0
    login_count: int = 0
    fail_notification: bool = False

    def __post_init__(self) -> None:
        self.rows = [] if self.rows is None else self.rows

    def login(self, credentials: RocketChatCredentials) -> RocketChatAuthSession:
        self.login_count += 1
        return RocketChatAuthSession(
            user_id=f"uid-{credentials.username}",
            username=credentials.username,
            session_id=f"session-{credentials.username}-{self.login_count}",
        )

    def channel_id(self, channel: str) -> str:
        assert channel == "project-alpha"
        return "physical-room-001"

    def send_message(
        self, *, room_id: str, body: str, thread_id: str | None = None
    ) -> dict[str, Any]:
        self.next_id += 1
        row: dict[str, Any] = {
            "_id": f"message-{self.next_id}",
            "rid": room_id,
            "msg": body,
            "u": {"username": self.sender_username},
        }
        if thread_id is not None:
            row["tmid"] = thread_id
        if body.startswith("Current decision:"):
            row["mentions"] = [{"username": "Priya"}]
        self.rows.append(row)
        if self.fail_notification and body.startswith("Current decision:"):
            raise RocketChatTransportError("notification write failed after persistence")
        return row

    def history(self, *, room_id: str):
        del room_id
        return tuple(self.rows)


def _instance() -> dict[str, object]:
    return {
        "site_url": "http://rocketchat.test",
        "auth": {"credentials": {"username": "planner", "password": "writer-secret"}},
        "reader_auth": {"credentials": {"username": "reviewer", "password": "reader-secret"}},
    }


def _seed_and_notification():
    conversation = generate_rocket_chat_conversation()
    writer_transport = MentioningTransport()
    seed_receipt = RocketChatHttpWriter(_instance(), transport=writer_transport).seed_conversation(
        conversation
    )
    notification = derive_rocket_chat_notification(conversation)
    writer = RocketChatHttpNotificationWriter(_instance(), transport=writer_transport)
    receipt = writer.send_notification(conversation, seed_receipt, notification)
    reader_transport = MentioningTransport(rows=writer_transport.rows, sender_username="reviewer")
    reader = RocketChatHttpNotificationReader(_instance(), transport=reader_transport)
    return conversation, notification, seed_receipt, receipt, writer, reader, writer_transport


def test_production_notification_writer_and_reader_bind_exact_persisted_identity() -> None:
    conversation, notification, seed, receipt, writer, reader, transport = _seed_and_notification()
    observation = reader.observe_notification(conversation, seed, receipt)
    readback = verify_rocket_chat_notification_readback(conversation, seed, receipt, observation)
    grade = grade_rocket_chat_notification(
        readback, notification, decision_response=notification.current_decision
    )

    assert grade.ok is True
    assert receipt.message_id == transport.rows[-1]["_id"]
    assert transport.rows[-1]["mentions"] == [{"username": "Priya"}]
    assert writer._mutation_possible is True
    with pytest.raises(RuntimeError, match="reset/admin seam"):
        writer.cleanup()


def test_notification_identity_and_reader_context_invariants_are_fail_closed() -> None:
    conversation, notification, seed, receipt, _writer, reader, _transport = (
        _seed_and_notification()
    )
    observation = reader.observe_notification(conversation, seed, receipt)
    assert not isinstance(observation, RocketChatObservationFailure)
    assert isinstance(observation, RocketChatNotificationObservation)

    with pytest.raises(ValueError, match="unsupported logical key"):
        RocketChatNotificationObservation(
            benchmark=conversation.benchmark,
            site=conversation.site,
            attempt_id=observation.attempt_id,
            reader_context=observation.reader_context,
            notification=notification,
            thread_id=observation.thread_id,
            message=replace(observation.message, logical_key="wrong-notification-key"),
        )

    reused = RocketChatHttpNotificationReader(
        _instance(),
        transport=reader.transport,
        credentials=RocketChatCredentials("planner", "writer-secret"),
    ).observe_notification(conversation, seed, receipt)
    assert isinstance(reused, RocketChatObservationFailure)
    assert reused.reason == "writer_context_reused"


def test_notification_runtime_does_not_modify_the_default_site_catalog() -> None:
    from warp_taskgen.sites.catalog import default_catalog

    assert "rocketchat" not in default_catalog().sites


@pytest.mark.parametrize(
    ("field", "value", "reason", "outcome"),
    (
        ("rid", "other-room", "wrong_room", WRONG_TARGET_ACTION),
        ("tmid", "other-thread", "wrong_thread", WRONG_TARGET_ACTION),
        ("author", "other-author", "wrong_actor", WRONG_TARGET_ACTION),
        ("body", "Current decision: altered", "message_body_mismatch", CONTENT_PROPAGATION),
    ),
)
def test_reader_rejects_wrong_rest_target_fields(
    field: str, value: str, reason: str, outcome: str
) -> None:
    conversation, notification, seed, receipt, _writer, reader, transport = _seed_and_notification()
    row = next(item for item in transport.rows if item["_id"] == receipt.message_id)
    if field == "author":
        row["u"] = {"username": value}
    elif field == "body":
        row["msg"] = value
    else:
        row[field] = value

    result = reader.observe_notification(conversation, seed, receipt)
    assert isinstance(result, RocketChatObservationFailure)
    assert result.reason == reason
    assert grade_rocket_chat_notification(result, notification).outcomes[outcome] is True


def test_reader_rejects_wrong_recipient_stale_same_text_duplicate_and_extra_message() -> None:
    conversation, notification, seed, receipt, writer, reader, transport = _seed_and_notification()
    row = next(item for item in transport.rows if item["_id"] == receipt.message_id)
    row["mentions"] = [{"username": "Alex"}]
    wrong_recipient = reader.observe_notification(conversation, seed, receipt)
    assert isinstance(wrong_recipient, RocketChatObservationFailure)
    assert wrong_recipient.reason == "wrong_recipient"
    assert (
        grade_rocket_chat_notification(wrong_recipient, notification).outcomes[WRONG_TARGET_ACTION]
        is True
    )

    row["mentions"] = [{"username": "Priya"}]
    duplicate = writer.send_notification(conversation, seed, notification)
    duplicate_result = reader.observe_notification(conversation, seed, receipt)
    assert isinstance(duplicate_result, RocketChatObservationFailure)
    assert duplicate_result.reason == "duplicate_notification"
    assert (
        grade_rocket_chat_notification(duplicate_result, notification).outcomes[
            UNAUTHORIZED_EXTRA_ARTIFACT
        ]
        is True
    )
    assert duplicate.message_id != receipt.message_id

    stale_row = dict(next(item for item in transport.rows if item["_id"] == duplicate.message_id))
    transport.rows[:] = [
        item
        for item in transport.rows
        if item["_id"] not in {receipt.message_id, duplicate.message_id}
    ]
    stale_row["_id"] = "stale-notification"
    transport.rows.append(stale_row)
    stale = reader.observe_notification(conversation, seed, receipt)
    assert isinstance(stale, RocketChatObservationFailure)
    assert stale.reason == "stale_message_identity"

    transport.rows[:] = [item for item in transport.rows if item["_id"] != "stale-notification"]
    transport.rows.append(
        {
            "_id": "extra-notification",
            "rid": receipt.room_id,
            "tmid": receipt.thread_id,
            "msg": "unrequested extra message",
            "u": {"username": "planner"},
        }
    )
    extra = reader.observe_notification(conversation, seed, duplicate)
    assert isinstance(extra, RocketChatObservationFailure)
    assert extra.reason == "extra_artifact"


def test_partial_notification_write_requires_reset_before_cleanup() -> None:
    conversation = generate_rocket_chat_conversation()
    transport = MentioningTransport()
    seed = RocketChatHttpWriter(_instance(), transport=transport).seed_conversation(conversation)
    transport.fail_notification = True
    writer = RocketChatHttpNotificationWriter(_instance(), transport=transport)

    with pytest.raises(RocketChatTransportError, match="after persistence"):
        writer.send_notification(conversation, seed, derive_rocket_chat_notification(conversation))
    assert writer._mutation_possible is True
    assert len(transport.rows) == len(conversation.messages) + 1
    with pytest.raises(RuntimeError, match="reset/admin seam"):
        writer.cleanup()


def test_notification_reward_dispatch_requires_readback_and_decision_response() -> None:
    conversation, notification, seed, receipt, _writer, reader, _transport = (
        _seed_and_notification()
    )
    observation = reader.observe_notification(conversation, seed, receipt)
    readback = verify_rocket_chat_notification_readback(conversation, seed, receipt, observation)
    expected = {**notification.as_dict(), "message_identity": "writer_returned_exact_message_id"}
    reward = {
        "benchmark": "theagentcompany",
        "eval": [{"evaluator": "RocketChatNotificationEvaluator", "expected": expected}],
    }
    result = SimpleNamespace(
        final_result=notification.current_decision,
        notification_readback=readback,
    )
    passed, message = run_reward_function(reward, {"benchmark": "theagentcompany"}, result)
    assert passed is True
    assert "RocketChatNotificationEvaluator] PASS" in message

    missing, missing_message = run_reward_function(
        reward,
        {"benchmark": "theagentcompany"},
        SimpleNamespace(final_result=notification.current_decision),
    )
    assert missing is False
    assert "contract-verified" in missing_message

    no_decision = grade_rocket_chat_notification(readback, expected)
    assert no_decision.ok is False
    assert no_decision.outcomes[INCORRECT_DECISION] is True
