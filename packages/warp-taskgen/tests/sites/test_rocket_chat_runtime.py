from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from warp_taskgen import seeding
from warp_taskgen.phase_1.rocket_chat_contracts import RocketChatObservationFailure
from warp_taskgen.phase_1.rocket_chat_decisions import generate_rocket_chat_conversation
from warp_taskgen.rewards import run_reward_function
from warp_taskgen.runtime_composition import RequiredSeedCleanupError
from warp_taskgen.seeding.site_contracts import (
    EditorSeedResult,
    SeedSiteRegistration,
    SeedSiteRegistry,
)
from warp_taskgen.sites.catalog import SiteCatalog
from warp_taskgen.sites.rocketchat_runtime import (
    RequestsRocketChatTransport,
    RocketChatAuthSession,
    RocketChatCredentials,
    RocketChatHttpEditor,
    RocketChatHttpReader,
    RocketChatHttpWriter,
    RocketChatRuntimeSite,
    RocketChatTransportError,
    preflight_rocket_chat_reader,
)


@dataclass
class FakeRocketChatTransport:
    username: str = "planner"
    rows: list[dict[str, object]] | None = None
    next_id: int = 0
    session_ids: list[str] | None = None
    physical_room_id: str = "physical-room-001"
    resolved_channels: list[str] | None = None
    closed: bool = False

    def __post_init__(self) -> None:
        self.rows = [] if self.rows is None else self.rows
        self.session_ids = [] if self.session_ids is None else self.session_ids
        self.resolved_channels = [] if self.resolved_channels is None else self.resolved_channels

    def login(self, credentials: RocketChatCredentials) -> RocketChatAuthSession:
        self.session_ids.append(f"session-{credentials.username}-{len(self.session_ids)}")
        return RocketChatAuthSession(
            user_id=f"uid-{credentials.username}",
            username=credentials.username,
            session_id=self.session_ids[-1],
        )

    def channel_id(self, channel: str) -> str:
        self.resolved_channels.append(channel)
        return self.physical_room_id

    def send_message(self, *, room_id: str, body: str, thread_id: str | None = None):
        self.next_id += 1
        row: dict[str, object] = {
            "_id": f"message-{self.next_id}",
            "rid": room_id,
            "msg": body,
            "u": {"username": self.username},
        }
        if thread_id is not None:
            row["tmid"] = thread_id
        self.rows.append(row)
        return row

    def history(self, *, room_id: str):
        return tuple(row for row in self.rows if row.get("rid") == room_id)

    def close(self) -> None:
        self.closed = True


class FailingAfterFirstWriteTransport(FakeRocketChatTransport):
    def send_message(self, *, room_id: str, body: str, thread_id: str | None = None):
        row = super().send_message(room_id=room_id, body=body, thread_id=thread_id)
        if len(self.rows) == 1:
            raise RocketChatTransportError("writer failed after first POST")
        return row


def _instance() -> dict[str, object]:
    return {
        "site_url": "http://rocketchat.test",
        "auth": {"credentials": {"username": "planner", "password": "writer-secret"}},
        "reader_auth": {
            "credentials": {"username": "reviewer", "password": "reader-secret"}
        },
    }


def _seed_result(receipt):
    tokens = {
        "attempt_id": receipt.attempt_id,
        "room_id": receipt.messages["plan"].room_id,
        "room_name": "project-alpha",
        "thread_id": receipt.messages["plan"].message_id,
        "reader_user_id": "reviewer",
        "reader_auth_context_id": "reader-credentials-uid-reviewer",
    }
    for key, identity in receipt.messages.items():
        tokens[f"{key}_message_id"] = identity.message_id
        tokens[f"{key}_body_sha256"] = hashlib.sha256(identity.body.encode()).hexdigest()
    return EditorSeedResult.from_mapping(
        {
            "identity_tokens": tokens,
            "read_surface_urls": ["/channel/project-alpha"],
        },
        editor_method="rocketchat.seed_rocket_chat_conversation",
    )


def test_http_writer_and_fresh_reader_bind_exact_rest_identities() -> None:
    conversation = generate_rocket_chat_conversation()
    transport = FakeRocketChatTransport()
    receipt = RocketChatHttpWriter(_instance(), transport=transport).seed_conversation(conversation)

    assert transport.resolved_channels == ["project-alpha"]
    assert receipt.messages["plan"].room_id == "physical-room-001"
    assert receipt.messages["plan"].room_id != conversation.room_id
    assert receipt.writer_context.user_id == "planner"
    assert receipt.messages["update"].thread_id == receipt.messages["plan"].message_id
    observation = RocketChatHttpReader(_instance(), transport=transport).observe(
        conversation, receipt
    )
    assert not isinstance(observation, RocketChatObservationFailure)
    assert observation.reader_context.user_id == "reviewer"
    assert observation.reader_context.auth_context_id != receipt.writer_context.auth_context_id
    assert observation.message_ids == {
        key: identity.message_id for key, identity in receipt.messages.items()
    }


def test_reader_ignores_ambient_history_rows_when_order_is_chronological() -> None:
    conversation = generate_rocket_chat_conversation()
    transport = FakeRocketChatTransport()
    receipt = RocketChatHttpWriter(_instance(), transport=transport).seed_conversation(conversation)
    transport.rows.insert(
        1,
        {
            "_id": "ambient-message",
            "rid": receipt.messages["plan"].room_id,
            "msg": "unrelated ambient message",
            "u": {"username": "other-user"},
        },
    )

    observation = RocketChatHttpReader(_instance(), transport=transport).observe(
        conversation, receipt
    )

    assert not isinstance(observation, RocketChatObservationFailure)


def test_reader_rejects_swapped_receipt_rows_even_when_all_ids_are_present() -> None:
    conversation = generate_rocket_chat_conversation()
    transport = FakeRocketChatTransport()
    receipt = RocketChatHttpWriter(_instance(), transport=transport).seed_conversation(conversation)
    transport.rows[:] = [transport.rows[1], transport.rows[0], transport.rows[2]]

    observation = RocketChatHttpReader(_instance(), transport=transport).observe(
        conversation, receipt
    )

    assert isinstance(observation, RocketChatObservationFailure)
    assert observation.reason == "message_order_mismatch"


@pytest.mark.parametrize(
    "mutator,reason",
    (
        (lambda transport: transport.rows[1].update({"rid": "wrong-room"}), "stale_message_identity"),
        (lambda transport: transport.rows[1].update({"tmid": "wrong-thread"}), "message_identity_mismatch"),
        (lambda transport: transport.rows[1]["u"].update({"username": "other-user"}), "message_identity_mismatch"),
    ),
)
def test_reader_rejects_wrong_room_thread_and_author(mutator, reason: str) -> None:
    conversation = generate_rocket_chat_conversation()
    transport = FakeRocketChatTransport()
    receipt = RocketChatHttpWriter(_instance(), transport=transport).seed_conversation(conversation)
    mutator(transport)

    observation = RocketChatHttpReader(_instance(), transport=transport).observe(
        conversation, receipt
    )
    assert isinstance(observation, RocketChatObservationFailure)
    assert observation.reason == reason


def test_reader_rejects_reused_writer_context_and_missing_reader_credentials() -> None:
    conversation = generate_rocket_chat_conversation()
    transport = FakeRocketChatTransport()
    receipt = RocketChatHttpWriter(_instance(), transport=transport).seed_conversation(conversation)
    reader = RocketChatHttpReader(
        _instance(), transport=transport, credentials=RocketChatCredentials("planner", "writer-secret")
    )
    observation = reader.observe(conversation, receipt)
    assert isinstance(observation, RocketChatObservationFailure)
    assert observation.reason == "writer_context_reused"

    missing = dict(_instance())
    missing.pop("reader_auth")
    result = preflight_rocket_chat_reader(missing)
    assert result.ok is False
    assert result.reason == "missing_reader_auth"


def test_runtime_site_requires_all_message_ids_for_rest_identity_planning() -> None:
    conversation = generate_rocket_chat_conversation()
    transport = FakeRocketChatTransport()
    receipt = RocketChatHttpWriter(_instance(), transport=transport).seed_conversation(conversation)
    site = SiteCatalog((RocketChatRuntimeSite(),)).bind(
        benchmark="theagentcompany", site="rocketchat", origin="http://rocketchat.test"
    )
    plan = site.read_surface_plan(
        seed_result=_seed_result(receipt), signature="Confirmed correction"
    )
    assert plan.verification_mode == "body_text"
    assert plan.identity_tokens["correction_message_id"] == receipt.messages["correction"].message_id
    broken_tokens = dict(_seed_result(receipt).write_tokens)
    broken_tokens.pop("update_message_id")
    broken = EditorSeedResult.from_mapping(
        {"identity_tokens": broken_tokens, "read_surface_urls": ["/channel/project-alpha"]},
        editor_method="rocketchat.seed_rocket_chat_conversation",
    )
    failure = site.read_surface_plan(seed_result=broken, signature="Confirmed correction")
    assert getattr(failure, "reason", "") == "missing_message_identity"


def test_runtime_reader_preflight_never_reuses_writer_auth() -> None:
    result = preflight_rocket_chat_reader(
        {
            "site_url": "http://rocketchat.test",
            "auth": {"credentials": {"username": "planner", "password": "pw"}},
            "reader_auth": {"type": "credentials", "credentials": {"username": "reviewer", "password": "pw"}},
        }
    )
    assert result.ok is False
    assert result.reason == "reader_browser_auth_unavailable"


def test_transport_response_shape_failures_are_explicit() -> None:
    class BrokenTransport(FakeRocketChatTransport):
        def send_message(self, *, room_id: str, body: str, thread_id: str | None = None):
            return {"success": True, "message": {"rid": room_id}}

    with pytest.raises(RocketChatTransportError, match=r"missing _id/rid/msg/u.username"):
        RocketChatHttpWriter(_instance(), transport=BrokenTransport()).seed_conversation(
            generate_rocket_chat_conversation()
        )


@dataclass
class FakeResponse:
    payload: Mapping[str, Any]
    status_code: int = 200

    def json(self) -> Mapping[str, Any]:
        return self.payload


class RecordingSession:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"method": method, "url": url, **kwargs})
        if url.endswith("/api/v1/login"):
            return FakeResponse(
                {"status": "success", "data": {"authToken": "secret", "userId": "uid-planner"}}
            )
        if url.endswith("/api/v1/channels.info"):
            return FakeResponse({"success": True, "channel": {"_id": "physical-room-001"}})
        if url.endswith("/api/v1/chat.sendMessage"):
            message = dict(kwargs["json"]["message"])
            message.update({"_id": "message-1", "u": {"username": "planner"}})
            return FakeResponse({"success": True, "message": message})
        if url.endswith("/api/v1/channels.history"):
            return FakeResponse({"success": True, "messages": []})
        raise AssertionError(url)


def test_requests_transport_uses_rocket_chat_rest_shapes_and_physical_room() -> None:
    session = RecordingSession()
    transport = RequestsRocketChatTransport("http://rocketchat.test", session=session)
    credentials = RocketChatCredentials("planner", "writer-secret")

    transport.login(credentials)
    assert transport.channel_id("project-alpha") == "physical-room-001"
    transport.send_message(room_id="physical-room-001", body="hello", thread_id="message-0")
    transport.history(room_id="physical-room-001")

    login, channel, send, history = session.calls
    assert login["url"].endswith("/api/v1/login")
    assert login["json"] == {"user": "planner", "password": "writer-secret"}
    assert channel["params"] == {"roomName": "project-alpha"}
    assert send["json"] == {
        "message": {"rid": "physical-room-001", "msg": "hello", "tmid": "message-0"}
    }
    assert history["params"] == {
        "roomId": "physical-room-001",
        "sort": '{"ts":1}',
        "showThreadMessages": True,
    }
    assert history["headers"] == {"X-Auth-Token": "secret", "X-User-Id": "uid-planner"}


def test_editor_requires_independent_reader_observation_before_emitting_seed_result() -> None:
    conversation = generate_rocket_chat_conversation()
    writer_transport = FakeRocketChatTransport()
    reader_transport = FakeRocketChatTransport(username="reviewer", rows=writer_transport.rows)
    editor = RocketChatHttpEditor(
        _instance(),
        object(),
        transport=writer_transport,
        reader_transport=reader_transport,
    )

    result = editor.seed_rocket_chat_conversation(
        conversation={**conversation.as_dict(), "expected_decision": conversation.expected_decision.as_dict()}
    )

    assert result["identity_tokens"]["room_id"] == "physical-room-001"
    assert result["identity_tokens"]["room_id"] != conversation.room_id
    assert result["identity_tokens"]["reader_user_id"] == "reviewer"
    assert result["identity_tokens"]["reader_auth_context_id"] == "reader-credentials-uid-reviewer"
    assert reader_transport.session_ids == ["session-reviewer-0"]
    with pytest.raises(RuntimeError, match="reset/admin seam"):
        editor.cleanup()
    assert reader_transport.closed is True


def test_editor_fails_closed_when_independent_reader_cannot_observe_seed() -> None:
    conversation = generate_rocket_chat_conversation()
    writer_transport = FakeRocketChatTransport()
    reader_transport = FakeRocketChatTransport(username="reviewer", rows=[])
    editor = RocketChatHttpEditor(
        _instance(),
        object(),
        transport=writer_transport,
        reader_transport=reader_transport,
    )

    with pytest.raises(RocketChatTransportError, match="independent reader observation failed"):
        editor.seed_rocket_chat_conversation(
            conversation={**conversation.as_dict(), "expected_decision": conversation.expected_decision.as_dict()}
        )
    assert reader_transport.closed is True


def test_partial_writer_post_arms_strict_cleanup_and_closes_reader() -> None:
    conversation = generate_rocket_chat_conversation()
    writer_transport = FailingAfterFirstWriteTransport()
    reader_transport = FakeRocketChatTransport(username="reviewer", rows=writer_transport.rows)

    def editor_factory(instance, session):
        return RocketChatHttpEditor(
            instance,
            session,
            transport=writer_transport,
            reader_transport=reader_transport,
        )

    registry = SeedSiteRegistry.from_registrations(
        (SeedSiteRegistration("theagentcompany", "rocketchat", editor_factory),)
    )
    seed = {
        "mechanism": "editor",
        "editor_calls": [
            {
                "benchmark": "theagentcompany",
                "site": "rocketchat",
                "method": "seed_rocket_chat_conversation",
                "args": {
                    "conversation": {
                        **conversation.as_dict(),
                        "expected_decision": conversation.expected_decision.as_dict(),
                    }
                },
            }
        ],
    }

    with pytest.raises(RequiredSeedCleanupError) as raised:
        seeding.apply_data_seed(
            seed,
            {**_instance(), "site_name": "rocketchat", "benchmark": "theagentcompany"},
            seed_registry=registry,
            strict_cleanup=True,
        )

    assert len(writer_transport.rows) == 1
    assert reader_transport.closed is True
    assert isinstance(raised.value.primary_error, RocketChatTransportError)
    assert "reset/admin seam" in str(raised.value.cleanup_error)


def test_warp_local_rocket_chat_evaluator_dispatches_exact_decision() -> None:
    expected = {"owner": "Priya", "due_date": "2026-09-18"}
    passed, message = run_reward_function(
        {"benchmark": "theagentcompany", "eval": [{"evaluator": "RocketChatEvaluator", "expected": expected}]},
        {"benchmark": "theagentcompany"},
        SimpleNamespace(final_result=expected),
    )

    assert passed is True
    assert "[RocketChatEvaluator] PASS" in message


def test_rocket_chat_evaluator_rejects_native_task_id_authority() -> None:
    passed, message = run_reward_function(
        {
            "benchmark": "theagentcompany",
            "task_id": "native-task",
            "eval": [{"evaluator": "RocketChatEvaluator", "expected": {"owner": "Priya", "due_date": "2026-09-18"}}],
        },
        {"benchmark": "theagentcompany"},
        SimpleNamespace(final_result={"owner": "Priya", "due_date": "2026-09-18"}),
    )

    assert passed is False
    assert "benchmark contract" in message
