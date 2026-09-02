"""Independent final-state readback tests for Rocket.Chat notifications."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from warp_taskgen.phase_1.rocket_chat_contracts import (
    RocketChatContractError,
    RocketChatObservationFailure,
)
from warp_taskgen.phase_1.rocket_chat_decisions import generate_rocket_chat_conversation
from warp_taskgen.phase_1.rocket_chat_notification_final_state import (
    RocketChatNotificationFinalStateReadback,
)
from warp_taskgen.phase_1.rocket_chat_notifications import (
    compile_rocket_chat_notification_task,
    derive_rocket_chat_notification,
)
from warp_taskgen.rewards import run_reward_function
from warp_taskgen.sites import rocketchat_notification_final_state as final_state
from warp_taskgen.sites.rocketchat_notification_runtime import (
    RocketChatHttpNotificationWriter,
    render_rocket_chat_notification_message,
)
from warp_taskgen.sites.rocketchat_runtime import (
    RocketChatAuthSession,
    RocketChatCredentials,
    RocketChatHttpWriter,
)


@dataclass
class _RowsTransport:
    rows: list[dict[str, Any]] = field(default_factory=list)
    sender_username: str = "planner"
    next_id: int = 0
    login_count: int = 0
    response_mentions: Mapping[str, list[dict[str, str]]] = field(default_factory=dict)

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
        row: dict[str, Any] = {
            "_id": f"message-{self.next_id}",
            "rid": room_id,
            "msg": body,
            "u": {"username": self.sender_username},
            "ts": datetime(2026, 9, 2, tzinfo=UTC).replace(microsecond=self.next_id).isoformat(),
        }
        if thread_id is not None:
            row["tmid"] = thread_id
        mentions = self.response_mentions.get(body)
        if mentions is not None:
            row["mentions"] = [dict(item) for item in mentions]
        self.rows.append(row)
        return row

    def history(self, *, room_id: str):
        del room_id
        return tuple(row for row in self.rows if row.get("tmid") in (None, ""))

    def thread_history(self, *, room_id: str, thread_id: str):
        del room_id, thread_id
        return tuple(row for row in self.rows if row.get("tmid") not in (None, ""))


@dataclass
class _ClosableRowsTransport(_RowsTransport):
    close_calls: int = 0

    def close(self) -> None:
        self.close_calls += 1


def _instance() -> dict[str, object]:
    return {
        "site_url": "http://rocketchat.test",
        "auth": {"credentials": {"username": "planner", "password": "writer-secret"}},
        "reader_auth": {"credentials": {"username": "reviewer", "password": "reader-secret"}},
    }


def _seed_and_notification() -> tuple[Any, Any, Any, Any, _RowsTransport]:
    conversation = generate_rocket_chat_conversation()
    notification = derive_rocket_chat_notification(conversation)
    wire_body = render_rocket_chat_notification_message(notification)
    transport = _RowsTransport(
        response_mentions={wire_body: [{"username": notification.recipient}]}
    )
    seed = RocketChatHttpWriter(_instance(), transport=transport).seed_conversation(conversation)
    receipt = RocketChatHttpNotificationWriter(_instance(), transport=transport).send_notification(
        conversation, seed, notification
    )
    return conversation, notification, seed, receipt, transport


def _seed_metadata(conversation: Any, seed: Any) -> dict[str, object]:
    root = seed.messages[conversation.thread_key]
    tokens = {
        "attempt_id": seed.attempt_id,
        "room_id": root.room_id,
        "room_name": conversation.room_id,
        "thread_id": root.message_id,
        "writer_user": conversation.writer_user,
        "reader_user_id": "uid-reviewer",
        "reader_auth_context_id": "reader-credentials-uid-reviewer",
    }
    for fact in conversation.messages:
        identity = seed.messages[fact.logical_key]
        tokens[f"{fact.logical_key}_message_id"] = identity.message_id
        tokens[f"{fact.logical_key}_body_sha256"] = hashlib.sha256(fact.body.encode()).hexdigest()
    return {
        "editor_call_results": [
            {
                "call_index": 0,
                "site": "rocketchat",
                "method": "seed_rocket_chat_conversation",
                "editor_method": "rocketchat.seed_rocket_chat_conversation",
                "benchmark": "theagentcompany",
                "write_tokens": tokens,
            }
        ]
    }


def _action_started_at(source: _RowsTransport, notification_id: str) -> datetime:
    row = next(item for item in source.rows if item.get("_id") == notification_id)
    return datetime.fromisoformat(str(row["ts"])) - timedelta(microseconds=1)


def test_loader_reads_current_notification_without_writer_receipt(monkeypatch) -> None:
    conversation, _notification, seed, receipt, source = _seed_and_notification()
    writer_transport = _ClosableRowsTransport()
    reader_transport = _ClosableRowsTransport(rows=source.rows, sender_username="reviewer")
    transports = [writer_transport, reader_transport]

    def fresh_transport(origin: str):
        assert origin == "http://rocketchat.test"
        return transports.pop(0)

    monkeypatch.setattr(final_state, "RequestsRocketChatTransport", fresh_transport)
    task = {"rocket_chat_contract": compile_rocket_chat_notification_task(conversation)}
    evidence = final_state.load_rocket_chat_notification_reward_evidence(
        task,
        _instance(),
        _seed_metadata(conversation, seed),
        _action_started_at(source, receipt.message_id),
    )

    assert isinstance(evidence, RocketChatNotificationFinalStateReadback)
    assert not hasattr(evidence, "notification_receipt")
    assert evidence.seed_receipt.attempt_id == seed.attempt_id
    assert evidence.message_id == receipt.message_id
    assert evidence.message.room_id == receipt.message.room_id
    assert evidence.message.thread_id == seed.messages[conversation.thread_key].message_id
    assert writer_transport.close_calls == 1
    assert reader_transport.close_calls == 1

    aliased_observation = replace(
        evidence.observation,
        message=replace(
            evidence.message,
            message_id=seed.messages[conversation.thread_key].message_id,
        ),
    )
    with pytest.raises(RocketChatContractError, match="aliases a seeded message"):
        RocketChatNotificationFinalStateReadback(
            seed_receipt=evidence.seed_receipt,
            observation=aliased_observation,
            action_started_at=evidence.action_started_at,
            persisted_at=evidence.persisted_at,
        )

    reward = task["rocket_chat_contract"]["reward_function"]
    result = SimpleNamespace(
        final_result=_notification.current_decision,
        runtime_reward_evidence=evidence,
    )
    passed, _message = run_reward_function(
        reward,
        {"benchmark": "theagentcompany"},
        result,
    )
    assert passed is True

    # Once the production loader ran, its failure is authoritative.  A
    # pre-existing/request-side readback must not bypass failed final-state
    # collection by falling back to the legacy result field.
    failed, _message = run_reward_function(
        reward,
        {"benchmark": "theagentcompany"},
        SimpleNamespace(
            final_result=_notification.current_decision,
            notification_readback=evidence,
            runtime_reward_evidence=None,
        ),
    )
    assert failed is False


def test_reconstructed_seed_drives_production_notification_writer(monkeypatch) -> None:
    conversation = generate_rocket_chat_conversation()
    notification = derive_rocket_chat_notification(conversation)
    wire_body = render_rocket_chat_notification_message(notification)
    source = _RowsTransport(response_mentions={wire_body: [{"username": notification.recipient}]})
    seed = RocketChatHttpWriter(_instance(), transport=source).seed_conversation(conversation)
    writer_transport = _ClosableRowsTransport()
    monkeypatch.setattr(
        final_state,
        "RequestsRocketChatTransport",
        lambda _origin: writer_transport,
    )

    rebuilt = final_state.load_rocket_chat_notification_seed_receipt(
        {"rocket_chat_contract": compile_rocket_chat_notification_task(conversation)},
        _instance(),
        _seed_metadata(conversation, seed),
    )

    assert not isinstance(rebuilt, RocketChatObservationFailure)
    receipt = RocketChatHttpNotificationWriter(
        _instance(),
        transport=source,
    ).send_notification(conversation, rebuilt, notification)
    assert receipt.seed_root == seed.messages[conversation.thread_key]
    assert receipt.message.thread_id == seed.messages[conversation.thread_key].message_id
    assert writer_transport.close_calls == 1


def test_loader_requires_per_call_seed_tokens() -> None:
    conversation = generate_rocket_chat_conversation()
    result = final_state.load_rocket_chat_notification_reward_evidence(
        {"rocket_chat_contract": compile_rocket_chat_notification_task(conversation)},
        _instance(),
        {"write_tokens": {"attempt_id": "stale"}},
        datetime.now(UTC),
    )

    assert isinstance(result, RocketChatObservationFailure)
    assert result.reason == "seed_metadata_invalid"


def test_loader_reports_missing_notification_without_propagation(monkeypatch) -> None:
    conversation, _notification, seed, _receipt, source = _seed_and_notification()
    notification_id = _receipt.message_id
    action_started_at = _action_started_at(source, notification_id)
    source.rows[:] = [row for row in source.rows if row.get("_id") != notification_id]
    transports = [_RowsTransport(), _RowsTransport(rows=source.rows, sender_username="reviewer")]
    monkeypatch.setattr(
        final_state, "RequestsRocketChatTransport", lambda _origin: transports.pop(0)
    )
    result = final_state.load_rocket_chat_notification_reward_evidence(
        {"rocket_chat_contract": compile_rocket_chat_notification_task(conversation)},
        _instance(),
        _seed_metadata(conversation, seed),
        action_started_at,
    )

    assert isinstance(result, RocketChatObservationFailure)
    assert result.reason == "missing_notification"


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("rid", "other-room", "wrong_room"),
        ("tmid", "other-thread", "wrong_thread"),
        ("u", {"username": "other-author"}, "wrong_actor"),
        ("msg", "wrong body", "message_body_mismatch"),
        ("mentions", [{"username": "other-recipient"}], "wrong_recipient"),
    ],
)
def test_loader_rejects_wrong_persisted_fields(
    monkeypatch, field: str, value: object, reason: str
) -> None:
    conversation, _notification, seed, receipt, source = _seed_and_notification()
    row = next(item for item in source.rows if item.get("_id") == receipt.message_id)
    row[field] = value
    transports = [_RowsTransport(), _RowsTransport(rows=source.rows, sender_username="reviewer")]
    monkeypatch.setattr(
        final_state, "RequestsRocketChatTransport", lambda _origin: transports.pop(0)
    )

    result = final_state.load_rocket_chat_notification_reward_evidence(
        {"rocket_chat_contract": compile_rocket_chat_notification_task(conversation)},
        _instance(),
        _seed_metadata(conversation, seed),
        _action_started_at(source, receipt.message_id),
    )

    assert isinstance(result, RocketChatObservationFailure)
    assert result.reason == reason


def test_loader_classifies_wrong_recipient_before_wrong_body(monkeypatch) -> None:
    conversation, _notification, seed, receipt, source = _seed_and_notification()
    row = next(item for item in source.rows if item.get("_id") == receipt.message_id)
    row["msg"] = "@other-recipient wrong body"
    row["mentions"] = [{"username": "other-recipient"}]
    transports = [_RowsTransport(), _RowsTransport(rows=source.rows, sender_username="reviewer")]
    monkeypatch.setattr(
        final_state, "RequestsRocketChatTransport", lambda _origin: transports.pop(0)
    )

    result = final_state.load_rocket_chat_notification_reward_evidence(
        {"rocket_chat_contract": compile_rocket_chat_notification_task(conversation)},
        _instance(),
        _seed_metadata(conversation, seed),
        _action_started_at(source, receipt.message_id),
    )

    assert isinstance(result, RocketChatObservationFailure)
    assert result.reason == "wrong_recipient"


def test_loader_rejects_duplicate_and_stale_current_attempt_state(monkeypatch) -> None:
    conversation, _notification, seed, receipt, source = _seed_and_notification()
    duplicate = dict(next(item for item in source.rows if item.get("_id") == receipt.message_id))
    duplicate["_id"] = "duplicate-notification"
    source.rows.append(duplicate)
    transports = [_RowsTransport(), _RowsTransport(rows=source.rows, sender_username="reviewer")]
    monkeypatch.setattr(
        final_state, "RequestsRocketChatTransport", lambda _origin: transports.pop(0)
    )
    task = {"rocket_chat_contract": compile_rocket_chat_notification_task(conversation)}
    metadata = _seed_metadata(conversation, seed)

    duplicate_result = final_state.load_rocket_chat_notification_reward_evidence(
        task,
        _instance(),
        metadata,
        _action_started_at(source, receipt.message_id),
    )
    assert isinstance(duplicate_result, RocketChatObservationFailure)
    assert duplicate_result.reason == "duplicate_notification"

    source.rows.pop()
    metadata["editor_call_results"][0]["write_tokens"]["thread_id"] = "stale-root"
    metadata["editor_call_results"][0]["write_tokens"][f"{conversation.thread_key}_message_id"] = (
        "stale-root"
    )
    transports[:] = [_RowsTransport(), _RowsTransport(rows=source.rows, sender_username="reviewer")]
    stale_result = final_state.load_rocket_chat_notification_reward_evidence(
        task,
        _instance(),
        metadata,
        _action_started_at(source, receipt.message_id),
    )
    assert isinstance(stale_result, RocketChatObservationFailure)
    assert stale_result.reason == "stale_message_identity"


def test_loader_rejects_seed_tokens_for_a_cloned_physical_room(monkeypatch) -> None:
    conversation, _notification, seed, receipt, source = _seed_and_notification()
    for row in source.rows:
        row["rid"] = "cloned-physical-room"
    metadata = _seed_metadata(conversation, seed)
    metadata["editor_call_results"][0]["write_tokens"]["room_id"] = "cloned-physical-room"
    transports = [
        _RowsTransport(),
        _RowsTransport(rows=source.rows, sender_username="reviewer"),
    ]
    monkeypatch.setattr(
        final_state,
        "RequestsRocketChatTransport",
        lambda _origin: transports.pop(0),
    )

    result = final_state.load_rocket_chat_notification_reward_evidence(
        {"rocket_chat_contract": compile_rocket_chat_notification_task(conversation)},
        _instance(),
        metadata,
        _action_started_at(source, receipt.message_id),
    )

    assert isinstance(result, RocketChatObservationFailure)
    assert result.reason == "wrong_room"


def test_loader_rejects_writer_identity_as_reader(monkeypatch) -> None:
    conversation, _notification, seed, _receipt, source = _seed_and_notification()
    transports = [_RowsTransport(), _RowsTransport(rows=source.rows)]
    monkeypatch.setattr(
        final_state, "RequestsRocketChatTransport", lambda _origin: transports.pop(0)
    )
    instance = _instance()
    instance["reader_auth"] = instance["auth"]

    result = final_state.load_rocket_chat_notification_reward_evidence(
        {"rocket_chat_contract": compile_rocket_chat_notification_task(conversation)},
        instance,
        _seed_metadata(conversation, seed),
        _action_started_at(source, _receipt.message_id),
    )

    assert isinstance(result, RocketChatObservationFailure)
    assert result.reason == "writer_context_reused"


def test_loader_rejects_same_text_message_older_than_current_action(monkeypatch) -> None:
    conversation, _notification, seed, receipt, source = _seed_and_notification()
    action_started_at = _action_started_at(source, receipt.message_id)
    row = next(item for item in source.rows if item.get("_id") == receipt.message_id)
    row["_id"] = "stale-same-text-notification"
    row["ts"] = (action_started_at - timedelta(seconds=1)).isoformat()
    transports = [_RowsTransport(), _RowsTransport(rows=source.rows, sender_username="reviewer")]
    monkeypatch.setattr(
        final_state, "RequestsRocketChatTransport", lambda _origin: transports.pop(0)
    )

    result = final_state.load_rocket_chat_notification_reward_evidence(
        {"rocket_chat_contract": compile_rocket_chat_notification_task(conversation)},
        _instance(),
        _seed_metadata(conversation, seed),
        action_started_at,
    )

    assert isinstance(result, RocketChatObservationFailure)
    assert result.reason == "stale_message_identity"
