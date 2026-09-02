from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

import pytest

from warp_taskgen.phase_1.rocket_chat_contracts import (
    RocketChatContractError,
    RocketChatObservationFailure,
)
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
    ROCKET_CHAT_NOTIFICATION_MESSAGE_IDENTITY,
    RocketChatNotificationObservation,
    derive_rocket_chat_notification,
)
from warp_taskgen.rewards import run_reward_function
from warp_taskgen.sites.rocketchat_notification_runtime import (
    RocketChatHttpNotificationReader,
    RocketChatHttpNotificationWriter,
    render_rocket_chat_notification_message,
)
from warp_taskgen.sites.rocketchat_runtime import (
    RequestsRocketChatTransport,
    RocketChatAuthSession,
    RocketChatCredentials,
    RocketChatHttpWriter,
    RocketChatTransportError,
)


@dataclass
class RowsTransport:
    """In-memory rows transport; response mentions are explicit fixtures."""

    rows: list[dict[str, Any]] | None = None
    sender_username: str = "planner"
    next_id: int = 0
    login_count: int = 0
    fail_notification: bool = False
    response_mentions: Mapping[str, list[dict[str, str]]] | None = None
    sent_messages: list[dict[str, Any]] | None = None
    history_calls: int = 0
    fail_on_history: int | None = None

    def __post_init__(self) -> None:
        self.rows = [] if self.rows is None else self.rows
        self.sent_messages = [] if self.sent_messages is None else self.sent_messages

    def login(self, credentials: RocketChatCredentials) -> RocketChatAuthSession:
        self.login_count += 1
        return RocketChatAuthSession(
            user_id=f"uid-{credentials.username}",
            username=credentials.username,
            session_id=f"session-{credentials.username}-{self.login_count}",
            roles=("user",),
        )

    def channel_id(self, channel: str) -> str:
        assert channel == "project-alpha"
        return "physical-room-001"

    def send_message(
        self, *, room_id: str, body: str, thread_id: str | None = None
    ) -> dict[str, Any]:
        self.next_id += 1
        self.sent_messages.append({"room_id": room_id, "body": body, "thread_id": thread_id})
        row: dict[str, Any] = {
            "_id": f"message-{self.next_id}",
            "rid": room_id,
            "msg": body,
            "u": {"username": self.sender_username},
        }
        if thread_id is not None:
            row["tmid"] = thread_id
        if self.response_mentions is not None and body in self.response_mentions:
            row["mentions"] = [dict(item) for item in self.response_mentions[body]]
        self.rows.append(row)
        if self.fail_notification and body.startswith("@"):
            raise RocketChatTransportError("notification write failed after persistence")
        return row

    def history(self, *, room_id: str):
        del room_id
        self.history_calls += 1
        if self.fail_on_history == self.history_calls:
            raise RocketChatTransportError("notification history read failed")
        return tuple(row for row in self.rows if row.get("tmid") in (None, ""))

    def thread_history(self, *, room_id: str, thread_id: str):
        del room_id, thread_id
        self.history_calls += 1
        if self.fail_on_history == self.history_calls:
            raise RocketChatTransportError("notification history read failed")
        # Deliberately return every threaded row. The production caller must
        # still reject an endpoint that leaks a wrong-room or wrong-thread row.
        return tuple(row for row in self.rows if row.get("tmid") not in (None, ""))


@dataclass
class RecordingSession:
    """Production-shaped REST server response with explicit mention fixtures."""

    mentions_by_body: Mapping[str, list[dict[str, str]]]
    rows: list[dict[str, Any]] | None = None
    calls: list[dict[str, Any]] | None = None
    next_id: int = 0
    response_bodies: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        self.rows = [] if self.rows is None else self.rows
        self.calls = [] if self.calls is None else self.calls

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"method": method, "url": url, **kwargs})
        if url.endswith("/api/v1/login"):
            username = kwargs["json"]["user"]
            return FakeResponse(
                {
                    "status": "success",
                    "data": {"authToken": f"token-{username}", "userId": f"uid-{username}"},
                }
            )
        if url.endswith("/api/v1/me"):
            user_id = kwargs["headers"]["X-User-Id"]
            username = user_id.removeprefix("uid-")
            return FakeResponse(
                {
                    "success": True,
                    "user": {"_id": user_id, "username": username, "roles": ["user"]},
                }
            )
        if url.endswith("/api/v1/channels.info"):
            return FakeResponse({"success": True, "channel": {"_id": "physical-room-001"}})
        if url.endswith("/api/v1/chat.sendMessage"):
            message = dict(kwargs["json"]["message"])
            self.next_id += 1
            row = {
                **message,
                "_id": f"message-{self.next_id}",
                "u": {"username": "planner"},
            }
            if self.response_bodies is not None and message["msg"] in self.response_bodies:
                row["msg"] = self.response_bodies[message["msg"]]
            mentions = self.mentions_by_body.get(message["msg"])
            if mentions is not None:
                row["mentions"] = [dict(item) for item in mentions]
            self.rows.append(row)
            return FakeResponse({"success": True, "message": row})
        if url.endswith("/api/v1/channels.history"):
            return FakeResponse(
                {
                    "success": True,
                    "messages": [row for row in self.rows if row.get("tmid") in (None, "")],
                }
            )
        if url.endswith("/api/v1/chat.getThreadMessages"):
            thread_id = kwargs["params"]["tmid"]
            return FakeResponse(
                {
                    "success": True,
                    "messages": [row for row in self.rows if row.get("tmid") == thread_id],
                }
            )
        raise AssertionError(url)


@dataclass
class FakeResponse:
    payload: Mapping[str, Any]
    status_code: int = 200

    def json(self) -> Mapping[str, Any]:
        return self.payload


def _instance() -> dict[str, object]:
    return {
        "site_url": "http://rocketchat.test",
        "auth": {"credentials": {"username": "planner", "password": "writer-secret"}},
        "reader_auth": {"credentials": {"username": "reviewer", "password": "reader-secret"}},
    }


def _seed_and_notification():
    conversation = generate_rocket_chat_conversation()
    notification = derive_rocket_chat_notification(conversation)
    wire_body = render_rocket_chat_notification_message(notification)
    writer_transport = RowsTransport(
        response_mentions={wire_body: [{"username": notification.recipient}]}
    )
    seed_receipt = RocketChatHttpWriter(_instance(), transport=writer_transport).seed_conversation(
        conversation
    )
    writer = RocketChatHttpNotificationWriter(_instance(), transport=writer_transport)
    receipt = writer.send_notification(conversation, seed_receipt, notification)
    reader_transport = RowsTransport(rows=writer_transport.rows, sender_username="reviewer")
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
    assert transport.sent_messages[-1]["body"] == (
        "@Priya Current decision: owner=Priya; due_date=2026-09-18."
    )
    assert receipt.message.room_id == "physical-room-001"
    assert observation.message.room_id == "physical-room-001"
    assert grade.actual["room_id"] == "physical-room-001"
    assert grade.expected["room_id"] == "physical-room-001"
    assert transport.rows[-1]["mentions"] == [{"username": "Priya"}]
    assert writer._mutation_possible is True
    with pytest.raises(RuntimeError, match="reset/admin seam"):
        writer.cleanup()


def test_requests_transport_sends_explicit_mention_and_reads_server_mentions() -> None:
    conversation = generate_rocket_chat_conversation()
    notification = derive_rocket_chat_notification(conversation)
    wire_body = render_rocket_chat_notification_message(notification)
    session = RecordingSession(mentions_by_body={wire_body: [{"username": notification.recipient}]})
    transport = RequestsRocketChatTransport("http://rocketchat.test", session=session)
    seed = RocketChatHttpWriter(_instance(), transport=transport).seed_conversation(conversation)
    receipt = RocketChatHttpNotificationWriter(_instance(), transport=transport).send_notification(
        conversation, seed, notification
    )

    send_calls = [
        call for call in session.calls if call["url"].endswith("/api/v1/chat.sendMessage")
    ]
    assert send_calls[-1]["json"]["message"]["msg"] == wire_body
    assert send_calls[-1]["json"]["message"]["msg"].startswith("@Priya ")
    assert session.rows[-1]["mentions"] == [{"username": "Priya"}]
    assert receipt.message.body == notification.body

    reader_transport = RequestsRocketChatTransport("http://rocketchat.test", session=session)
    observation = RocketChatHttpNotificationReader(
        _instance(), transport=reader_transport
    ).observe_notification(conversation, seed, receipt)
    assert not isinstance(observation, RocketChatObservationFailure)
    assert observation.message.room_id == "physical-room-001"

    session.rows[-1]["mentions"] = [{"username": "Alex"}]
    wrong_recipient = RocketChatHttpNotificationReader(
        _instance(), transport=reader_transport
    ).observe_notification(conversation, seed, receipt)
    assert isinstance(wrong_recipient, RocketChatObservationFailure)
    assert wrong_recipient.reason == "wrong_recipient"


def test_requests_transport_rejects_missing_server_mention() -> None:
    conversation = generate_rocket_chat_conversation()
    notification = derive_rocket_chat_notification(conversation)
    wire_body = render_rocket_chat_notification_message(notification)
    session = RecordingSession(mentions_by_body={})
    transport = RequestsRocketChatTransport("http://rocketchat.test", session=session)
    seed = RocketChatHttpWriter(_instance(), transport=transport).seed_conversation(conversation)

    with pytest.raises(RocketChatTransportError, match="exactly the intended recipient"):
        RocketChatHttpNotificationWriter(_instance(), transport=transport).send_notification(
            conversation, seed, notification
        )
    send_calls = [
        call for call in session.calls if call["url"].endswith("/api/v1/chat.sendMessage")
    ]
    assert send_calls[-1]["json"]["message"]["msg"] == wire_body


def test_requests_transport_rejects_plain_response_body_without_mention() -> None:
    conversation = generate_rocket_chat_conversation()
    notification = derive_rocket_chat_notification(conversation)
    wire_body = render_rocket_chat_notification_message(notification)
    session = RecordingSession(
        mentions_by_body={wire_body: [{"username": notification.recipient}]},
        response_bodies={wire_body: notification.body},
    )
    transport = RequestsRocketChatTransport("http://rocketchat.test", session=session)
    seed = RocketChatHttpWriter(_instance(), transport=transport).seed_conversation(conversation)

    with pytest.raises(RocketChatTransportError, match="wrong body"):
        RocketChatHttpNotificationWriter(_instance(), transport=transport).send_notification(
            conversation, seed, notification
        )
    send_calls = [
        call for call in session.calls if call["url"].endswith("/api/v1/chat.sendMessage")
    ]
    assert send_calls[-1]["json"]["message"]["msg"] == wire_body


def test_notification_message_rejects_unsafe_recipient_mentions() -> None:
    conversation = generate_rocket_chat_conversation(corrected_owner="Priya Lee")

    with pytest.raises(RocketChatContractError, match="cannot be encoded"):
        derive_rocket_chat_notification(conversation)


def test_writer_rejects_changed_logical_to_physical_room_before_send() -> None:
    conversation = generate_rocket_chat_conversation()
    notification = derive_rocket_chat_notification(conversation)

    class MappingDriftTransport(RowsTransport):
        channel_calls: int = 0

        def channel_id(self, channel: str) -> str:
            self.channel_calls += 1
            assert channel == conversation.room_id
            return "physical-room-001" if self.channel_calls == 1 else "foreign-room"

    wire_body = render_rocket_chat_notification_message(notification)
    transport = MappingDriftTransport(
        response_mentions={wire_body: [{"username": notification.recipient}]}
    )
    seed = RocketChatHttpWriter(_instance(), transport=transport).seed_conversation(conversation)

    with pytest.raises(RocketChatTransportError, match="mapping"):
        RocketChatHttpNotificationWriter(_instance(), transport=transport).send_notification(
            conversation, seed, notification
        )
    assert len(transport.rows) == len(conversation.messages)
    assert transport.channel_calls == 2


def test_reader_rejects_changed_logical_to_physical_room_before_history() -> None:
    conversation, _notification, seed, receipt, _writer, _reader, transport = (
        _seed_and_notification()
    )

    class ReaderMappingDriftTransport(RowsTransport):
        channel_calls: int = 0

        def channel_id(self, channel: str) -> str:
            self.channel_calls += 1
            assert channel == conversation.room_id
            return "foreign-room"

    reader_transport = ReaderMappingDriftTransport(
        rows=transport.rows,
        sender_username="reviewer",
    )
    reader = RocketChatHttpNotificationReader(_instance(), transport=reader_transport)

    result = reader.observe_notification(conversation, seed, receipt)

    assert isinstance(result, RocketChatObservationFailure)
    assert result.reason == "wrong_room"
    assert "mapping" in result.detail
    assert reader_transport.channel_calls == 1
    assert reader_transport.history_calls == 0


def test_second_notification_history_error_is_a_typed_reader_failure() -> None:
    conversation, _notification, seed, receipt, _writer, _reader, transport = (
        _seed_and_notification()
    )
    reader_transport = RowsTransport(
        rows=transport.rows,
        sender_username="reviewer",
        fail_on_history=2,
    )
    reader = RocketChatHttpNotificationReader(_instance(), transport=reader_transport)

    result = reader.observe_notification(conversation, seed, receipt)

    assert isinstance(result, RocketChatObservationFailure)
    assert result.reason == "reader_transport_failed"
    assert "history read failed" in result.detail


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
    notification = derive_rocket_chat_notification(conversation)
    wire_body = render_rocket_chat_notification_message(notification)
    transport = RowsTransport(response_mentions={wire_body: [{"username": notification.recipient}]})
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
    expected = {
        **notification.as_dict(),
        "message_identity": ROCKET_CHAT_NOTIFICATION_MESSAGE_IDENTITY,
    }
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
