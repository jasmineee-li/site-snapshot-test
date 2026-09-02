from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pytest

from tests.sites.test_rocket_chat_runtime import (
    FailingAfterFirstWriteTransport,
    FakeRocketChatResetter,
    FakeRocketChatTransport,
    _instance,
)
from warp_taskgen import seeding
from warp_taskgen.phase_1.rocket_chat_contracts import RocketChatObservationFailure
from warp_taskgen.phase_1.rocket_chat_decisions import generate_rocket_chat_conversation
from warp_taskgen.runtime_composition import RequiredSeedCleanupError
from warp_taskgen.seeding.site_contracts import SeedSiteRegistration, SeedSiteRegistry
from warp_taskgen.sites.rocketchat_reset import RequestsRocketChatResetter
from warp_taskgen.sites.rocketchat_runtime import (
    RequestsRocketChatTransport,
    RocketChatCredentials,
    RocketChatHttpEditor,
    RocketChatHttpReader,
    RocketChatHttpWriter,
    RocketChatTransportError,
    preflight_rocket_chat_reader,
)


def test_http_writer_and_fresh_reader_bind_exact_rest_identities() -> None:
    conversation = generate_rocket_chat_conversation()
    transport = FakeRocketChatTransport()
    receipt = RocketChatHttpWriter(_instance(), transport=transport).seed_conversation(conversation)

    assert transport.resolved_channels == ["project-alpha"]
    assert receipt.messages["plan"].room_id == "physical-room-001"
    assert receipt.messages["plan"].room_id != conversation.room_id
    assert receipt.writer_context.user_id == "uid-planner"
    assert receipt.writer_context.username == "planner"
    assert receipt.messages["update"].thread_id == receipt.messages["plan"].message_id
    observation = RocketChatHttpReader(_instance(), transport=transport).observe(
        conversation, receipt
    )
    assert not isinstance(observation, RocketChatObservationFailure)
    assert observation.reader_context.user_id == "uid-reviewer"
    assert observation.reader_context.username == "reviewer"
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


def test_reader_rejects_root_only_room_history_without_thread_endpoint_rows() -> None:
    conversation = generate_rocket_chat_conversation()
    writer_transport = FakeRocketChatTransport()
    receipt = RocketChatHttpWriter(_instance(), transport=writer_transport).seed_conversation(
        conversation
    )

    class RootOnlyTransport(FakeRocketChatTransport):
        def history(self, *, room_id: str):
            return tuple(
                row
                for row in self.rows
                if row.get("rid") == room_id
                and row.get("_id") == receipt.messages["plan"].message_id
            )

        def thread_history(self, *, room_id: str, thread_id: str):
            del room_id, thread_id
            return ()

    reader_transport = RootOnlyTransport(username="reviewer", rows=writer_transport.rows)
    observation = RocketChatHttpReader(_instance(), transport=reader_transport).observe(
        conversation, receipt
    )

    assert isinstance(observation, RocketChatObservationFailure)
    assert observation.reason == "stale_message_identity"


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
        (
            lambda transport: transport.rows[1].update({"rid": "wrong-room"}),
            "stale_message_identity",
        ),
        (
            lambda transport: transport.rows[1].update({"tmid": "wrong-thread"}),
            "message_identity_mismatch",
        ),
        (
            lambda transport: transport.rows[1]["u"].update({"username": "other-user"}),
            "message_identity_mismatch",
        ),
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
        _instance(),
        transport=transport,
        credentials=RocketChatCredentials("planner", "writer-secret"),
    )
    observation = reader.observe(conversation, receipt)
    assert isinstance(observation, RocketChatObservationFailure)
    assert observation.reason == "writer_context_reused"

    missing = dict(_instance())
    missing.pop("reader_auth")
    result = preflight_rocket_chat_reader(missing)
    assert result.ok is False
    assert result.reason == "missing_reader_auth"


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
        if url == "http://reset.test:7771/init":
            return FakeResponse({"ok": True})
        if url.endswith("/api/v1/login"):
            return FakeResponse(
                {"status": "success", "data": {"authToken": "secret", "userId": "uid-planner"}}
            )
        if url.endswith("/api/v1/me"):
            return FakeResponse(
                {
                    "success": True,
                    "user": {"_id": "uid-planner", "username": "planner", "roles": ["user"]},
                }
            )
        if url.endswith("/api/v1/channels.info"):
            return FakeResponse({"success": True, "channel": {"_id": "physical-room-001"}})
        if url.endswith("/api/v1/chat.sendMessage"):
            message = dict(kwargs["json"]["message"])
            message.update({"_id": "message-1", "u": {"username": "planner"}})
            return FakeResponse({"success": True, "message": message})
        if url.endswith("/api/v1/channels.history"):
            return FakeResponse({"success": True, "messages": []})
        if url.endswith("/api/v1/chat.getThreadMessages"):
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
    transport.thread_history(room_id="physical-room-001", thread_id="message-0")

    login, me, channel, send, history, thread = session.calls
    assert login["url"].endswith("/api/v1/login")
    assert login["json"] == {"user": "planner", "password": "writer-secret"}
    assert me["url"].endswith("/api/v1/me")
    assert channel["params"] == {"roomName": "project-alpha"}
    assert send["json"] == {
        "message": {"rid": "physical-room-001", "msg": "hello", "tmid": "message-0"}
    }
    assert history["params"] == {
        "roomId": "physical-room-001",
        "count": 100,
    }
    assert history["headers"] == {"X-Auth-Token": "secret", "X-User-Id": "uid-planner"}
    assert thread["params"] == {"tmid": "message-0", "count": 100, "offset": 0}
    assert thread["headers"] == {"X-Auth-Token": "secret", "X-User-Id": "uid-planner"}


def test_requests_login_accepts_pinned_rocket_chat_53_top_level_me_shape() -> None:
    class TopLevelMeSession(RecordingSession):
        def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
            response = super().request(method, url, **kwargs)
            if url.endswith("/api/v1/me"):
                return FakeResponse(
                    {
                        "success": True,
                        "_id": "uid-planner",
                        "username": "planner",
                        "roles": ["user"],
                    }
                )
            return response

    auth = RequestsRocketChatTransport(
        "http://rocketchat.test", session=TopLevelMeSession()
    ).login(RocketChatCredentials("planner", "writer-secret"))

    assert auth.user_id == "uid-planner"
    assert auth.username == "planner"
    assert auth.roles == ("user",)


@pytest.mark.parametrize(
    "user_patch,pattern",
    (
        ({"roles": ["user", "admin"]}, "ordinary user"),
        ({"roles": ["bot"]}, "ordinary user"),
        ({"username": "someone-else"}, "username does not match"),
        ({"_id": "uid-other"}, "user ID does not match"),
    ),
)
def test_requests_login_rejects_nonordinary_or_mismatched_me_identity(
    user_patch: Mapping[str, object], pattern: str
) -> None:
    class MeMutationSession(RecordingSession):
        def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
            response = super().request(method, url, **kwargs)
            if url.endswith("/api/v1/me"):
                payload = dict(response.payload)
                user = dict(payload["user"])
                user.update(user_patch)
                payload["user"] = user
                return FakeResponse(payload, response.status_code)
            return response

    transport = RequestsRocketChatTransport("http://rocketchat.test", session=MeMutationSession())
    with pytest.raises(RocketChatTransportError, match=pattern):
        transport.login(RocketChatCredentials("planner", "writer-secret"))


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
        conversation={
            **conversation.as_dict(),
            "expected_decision": conversation.expected_decision.as_dict(),
        }
    )

    assert result["identity_tokens"]["room_id"] == "physical-room-001"
    assert result["identity_tokens"]["room_id"] != conversation.room_id
    assert result["identity_tokens"]["reader_user_id"] == "uid-reviewer"
    assert result["identity_tokens"]["reader_auth_context_id"] == "reader-credentials-uid-reviewer"
    assert result["read_surface_urls"] == [
        f"/channel/project-alpha/thread/{result['identity_tokens']['thread_id']}"
    ]
    assert reader_transport.session_ids == ["session-reviewer-0"]
    with pytest.raises(RuntimeError, match="reset/admin seam"):
        editor.cleanup()
    assert reader_transport.closed is True


def test_editor_cleanup_calls_injected_reset_synchronously_and_is_idempotent() -> None:
    conversation = generate_rocket_chat_conversation()
    writer_transport = FakeRocketChatTransport()
    reader_transport = FakeRocketChatTransport(username="reviewer", rows=writer_transport.rows)
    resetter = FakeRocketChatResetter()
    editor = RocketChatHttpEditor(
        _instance(),
        object(),
        transport=writer_transport,
        reader_transport=reader_transport,
        resetter=resetter,
    )

    editor.seed_rocket_chat_conversation(
        conversation={
            **conversation.as_dict(),
            "expected_decision": conversation.expected_decision.as_dict(),
        }
    )
    editor.cleanup()
    editor.cleanup()

    assert resetter.calls == 1
    assert reader_transport.closed is True


def test_editor_cleanup_reports_reset_failure_after_possible_mutation() -> None:
    conversation = generate_rocket_chat_conversation()
    writer_transport = FakeRocketChatTransport()
    reader_transport = FakeRocketChatTransport(username="reviewer", rows=writer_transport.rows)
    resetter = FakeRocketChatResetter(fail=True)
    editor = RocketChatHttpEditor(
        _instance(),
        object(),
        transport=writer_transport,
        reader_transport=reader_transport,
        resetter=resetter,
    )
    editor.seed_rocket_chat_conversation(
        conversation={
            **conversation.as_dict(),
            "expected_decision": conversation.expected_decision.as_dict(),
        }
    )

    with pytest.raises(RuntimeError, match="reset owner failed"):
        editor.cleanup()
    assert resetter.calls == 1


def test_requests_resetter_does_not_reuse_writer_transport_auth() -> None:
    session = RecordingSession()
    resetter = RequestsRocketChatResetter(
        "http://reset.test:7771/init",
        session=session,
        headers={"X-Reset-Owner": "host"},
    )

    resetter.reset()

    call = session.calls[-1]
    assert call["method"] == "POST"
    assert call["url"] == "http://reset.test:7771/init"
    assert call["headers"] == {"X-Reset-Owner": "host"}
    assert "X-Auth-Token" not in call["headers"]
    assert call["timeout"] == 120.0


def test_requests_resetter_rejects_async_202_without_terminal_completion() -> None:
    class AsyncResetSession(RecordingSession):
        def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
            self.calls.append({"method": method, "url": url, **kwargs})
            return FakeResponse({"accepted": True}, status_code=202)

    resetter = RequestsRocketChatResetter(
        "http://reset.test:7771/init",
        session=AsyncResetSession(),
    )
    with pytest.raises(RuntimeError, match="asynchronous reset is not complete"):
        resetter.reset()


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
            conversation={
                **conversation.as_dict(),
                "expected_decision": conversation.expected_decision.as_dict(),
            }
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
