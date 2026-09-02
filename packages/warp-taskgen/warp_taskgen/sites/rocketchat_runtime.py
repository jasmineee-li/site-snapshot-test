"""Feature-local, opt-in Rocket.Chat runtime seams for TAC decision transfer."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Protocol
from urllib.parse import urljoin, urlsplit

import requests

from warp_taskgen.phase_1.rocket_chat_contracts import (
    ROCKET_CHAT_BENCHMARK,
    ROCKET_CHAT_SITE,
    RocketChatContractError,
    RocketChatConversation,
    RocketChatMessageFact,
    RocketChatMessageIdentity,
    RocketChatObservation,
    RocketChatObservationFailure,
    RocketChatParticipantContext,
    RocketChatSeedReceipt,
)
from warp_taskgen.seeding.site_contracts import EditorSeedResult
from warp_taskgen.sites.read_surface import (
    ReadSurfacePlanFailure,
    ReadSurfaceVerificationPlan,
    build_read_surface_plan,
)
from warp_taskgen.sites.rocketchat import RocketChatSite

TAC_SOURCE_COMMIT = "98b68ef82a47690c316f42fddb05baafaab56851"
ROCKET_CHAT_RUNTIME_COMPOSITION = "rocket_chat_conversation_decision_poc"
ROCKET_CHAT_EDITOR_METHOD = "seed_rocket_chat_conversation"
_ID_MAX = 128
class RocketChatTransportError(RuntimeError):
    pass
@dataclass(frozen=True)
class RocketChatCredentials:
    username: str
    password: str

    def __post_init__(self) -> None:
        if not isinstance(self.username, str) or not self.username.strip():
            raise RocketChatContractError("Rocket.Chat username is required")
        if not isinstance(self.password, str) or not self.password:
            raise RocketChatContractError("Rocket.Chat password is required")
        object.__setattr__(self, "username", self.username.strip())
@dataclass(frozen=True)
class RocketChatAuthSession:
    user_id: str
    username: str
    session_id: str

    def __post_init__(self) -> None:
        for field in ("user_id", "username", "session_id"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip() or len(value.strip()) > _ID_MAX:
                raise RocketChatContractError(f"Rocket.Chat auth session {field} is invalid")
            object.__setattr__(self, field, value.strip())
class RocketChatTransport(Protocol):
    def login(self, credentials: RocketChatCredentials) -> RocketChatAuthSession: ...
    def channel_id(self, channel: str) -> str: ...
    def send_message(
        self, *, room_id: str, body: str, thread_id: str | None = None
    ) -> Mapping[str, Any]: ...

    def history(self, *, room_id: str) -> Sequence[Mapping[str, Any]]: ...
def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise RocketChatContractError(f"Rocket.Chat {field} must be a string")
    result = value.strip()
    if not result or len(result) > _ID_MAX or any(char.isspace() for char in result):
        raise RocketChatContractError(f"Rocket.Chat {field} must be a bounded identifier")
    return result
def _origin(instance: Mapping[str, Any]) -> str:
    raw = str(instance.get("site_url") or "").strip().rstrip("/")
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.query or parsed.fragment:
        raise RocketChatTransportError("Rocket.Chat requires an explicit HTTP(S) site_url origin")
    return raw
def _credentials(instance: Mapping[str, Any], role: str) -> RocketChatCredentials:
    if role not in {"writer", "reader"}:
        raise RocketChatContractError("Rocket.Chat credential role is unsupported")
    keys = ("writer_auth", "auth", "agent_auth") if role == "writer" else ("reader_auth",)
    for key in keys:
        block = instance.get(key)
        if not isinstance(block, Mapping):
            continue
        sources: list[Mapping[str, Any]] = [block]
        for nested_key in ("credentials", "authentication"):
            nested = block.get(nested_key)
            if isinstance(nested, Mapping):
                sources.append(nested)
        for source in sources:
            username = source.get("username", source.get("login"))
            password = source.get("password")
            if isinstance(username, str) and username.strip() and isinstance(password, str) and password:
                return RocketChatCredentials(username, password)
    raise RocketChatContractError(
        f"Rocket.Chat {role} credentials are required in explicit {role}_auth/auth configuration"
    )
class RequestsRocketChatTransport:
    def __init__(self, site_url: str, session: requests.Session | None = None, *, timeout_s: float = 30.0) -> None:
        parsed = urlsplit(str(site_url or "").strip().rstrip("/"))
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.query or parsed.fragment:
            raise RocketChatTransportError("Rocket.Chat transport requires an HTTP(S) origin")
        self.site_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"
        self.session = session or requests.Session()
        self.timeout_s = float(timeout_s)
        if self.timeout_s <= 0 or self.timeout_s > 120:
            raise RocketChatTransportError("Rocket.Chat transport timeout is out of bounds")
        self._auth_headers: dict[str, str] = {}
    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
        authenticated: bool = True,
    ) -> Mapping[str, Any]:
        if not path.startswith("/") or "//" in path:
            raise RocketChatTransportError("Rocket.Chat REST path is invalid")
        try:
            response = self.session.request(
                method.upper(),
                f"{self.site_url}{path}",
                headers=dict(self._auth_headers) if authenticated else {},
                json=dict(json_body) if isinstance(json_body, Mapping) else None,
                params=dict(params) if isinstance(params, Mapping) else None,
                timeout=self.timeout_s,
                allow_redirects=False,
            )
        except requests.RequestException as exc:
            raise RocketChatTransportError(f"Rocket.Chat REST {method.upper()} {path} failed: {exc.__class__.__name__}") from exc
        if 300 <= response.status_code < 400:
            raise RocketChatTransportError(f"Rocket.Chat REST {path} returned an unexpected redirect")
        if response.status_code in {401, 403}:
            raise RocketChatTransportError(f"Rocket.Chat REST {path} rejected authentication (HTTP {response.status_code})")
        if response.status_code < 200 or response.status_code >= 300:
            raise RocketChatTransportError(f"Rocket.Chat REST {path} returned HTTP {response.status_code}")
        try:
            payload = response.json()
        except (TypeError, ValueError) as exc:
            raise RocketChatTransportError(f"Rocket.Chat REST {path} returned non-JSON data") from exc
        if not isinstance(payload, Mapping):
            raise RocketChatTransportError(f"Rocket.Chat REST {path} returned a non-object")
        return payload
    @staticmethod
    def _require_success(payload: Mapping[str, Any], path: str) -> None:
        if payload.get("success") is not True:
            raise RocketChatTransportError(f"Rocket.Chat REST {path} did not report success")
    def login(self, credentials: RocketChatCredentials) -> RocketChatAuthSession:
        payload = self._request(
            "POST", "/api/v1/login",
            json_body={"user": credentials.username, "password": credentials.password},
            authenticated=False,
        )
        if payload.get("status") != "success":
            raise RocketChatTransportError("Rocket.Chat login did not report status=success")
        data = payload.get("data")
        if not isinstance(data, Mapping):
            raise RocketChatTransportError("Rocket.Chat login response is missing data")
        token, user_id = data.get("authToken"), data.get("userId")
        if not isinstance(token, str) or not token or not isinstance(user_id, str) or not user_id:
            raise RocketChatTransportError("Rocket.Chat login response is missing authToken/userId")
        self._auth_headers = {"X-Auth-Token": token, "X-User-Id": user_id}
        return RocketChatAuthSession(user_id=user_id, username=credentials.username, session_id=f"rc-session-{user_id}")
    def channel_id(self, channel: str) -> str:
        channel = _identifier(channel, "channel")
        payload = self._request("GET", "/api/v1/channels.info", params={"roomName": channel})
        self._require_success(payload, "/api/v1/channels.info")
        room = payload.get("channel")
        if not isinstance(room, Mapping):
            data = payload.get("data")
            room = data.get("channel") if isinstance(data, Mapping) else None
        return _identifier(room.get("_id") if isinstance(room, Mapping) else None, "room id")
    def send_message(self, *, room_id: str, body: str, thread_id: str | None = None) -> Mapping[str, Any]:
        room_id = _identifier(room_id, "room id")
        if not isinstance(body, str) or not body.strip():
            raise RocketChatTransportError("Rocket.Chat message body is required")
        message: dict[str, str] = {"rid": room_id, "msg": body}
        if thread_id is not None:
            message["tmid"] = _identifier(thread_id, "thread id")
        payload = self._request("POST", "/api/v1/chat.sendMessage", json_body={"message": message})
        self._require_success(payload, "/api/v1/chat.sendMessage")
        result = payload.get("message")
        if not isinstance(result, Mapping):
            raise RocketChatTransportError("Rocket.Chat chat.sendMessage response is missing message")
        return result
    def history(self, *, room_id: str) -> Sequence[Mapping[str, Any]]:
        room_id = _identifier(room_id, "room id")
        payload = self._request(
            "GET",
            "/api/v1/channels.history",
            params={"roomId": room_id, "sort": '{"ts":1}', "showThreadMessages": True},
        )
        self._require_success(payload, "/api/v1/channels.history")
        messages = payload.get("messages")
        if not isinstance(messages, list) or any(not isinstance(item, Mapping) for item in messages):
            raise RocketChatTransportError("Rocket.Chat channels.history response has invalid messages")
        return tuple(messages)
def _context(auth: RocketChatAuthSession, *, role: str, username: str) -> RocketChatParticipantContext:
    if auth.username != username:
        raise RocketChatContractError("Rocket.Chat login username does not match participant")
    return RocketChatParticipantContext(
        user_id=username,
        session_id=f"{role}-session-{auth.session_id}",
        auth_context_id=f"{role}-credentials-{auth.user_id}",
        auth_kind="writer_credentials" if role == "writer" else "reader_credentials",
        role="ordinary",
    )
def _identity(
    row: Mapping[str, Any],
    conversation: RocketChatConversation,
    fact: RocketChatMessageFact,
    attempt_id: str,
    expected_thread_id: str | None,
    expected_room_id: str,
) -> RocketChatMessageIdentity:
    message_id, room_id, body, user = row.get("_id"), row.get("rid"), row.get("msg"), row.get("u")
    author = user.get("username") if isinstance(user, Mapping) else None
    thread_id = row.get("tmid")
    if not all(isinstance(value, str) for value in (message_id, room_id, body, author)):
        raise RocketChatTransportError("Rocket.Chat message response is missing _id/rid/msg/u.username")
    if room_id != expected_room_id or body != fact.body or author != fact.author:
        raise RocketChatTransportError(f"Rocket.Chat message {fact.logical_key!r} identity/body mismatch")
    if expected_thread_id is None:
        if thread_id not in (None, ""):
            raise RocketChatTransportError("Rocket.Chat root message unexpectedly carries a thread id")
        normalized_thread = None
    elif thread_id != expected_thread_id:
        raise RocketChatTransportError(f"Rocket.Chat message {fact.logical_key!r} thread mismatch")
    else:
        normalized_thread = thread_id
    return RocketChatMessageIdentity(
        benchmark=conversation.benchmark, site=conversation.site, attempt_id=attempt_id,
        logical_key=fact.logical_key, room_id=room_id, message_id=message_id,
        thread_id=normalized_thread, author=author, body=body,
    )
class RocketChatHttpWriter:
    def __init__(self, instance: Mapping[str, Any], *, transport: RocketChatTransport, credentials: RocketChatCredentials | None = None) -> None:
        self.instance = dict(instance)
        self.transport = transport
        self.credentials = credentials or _credentials(self.instance, "writer")
    def seed_conversation(self, conversation: RocketChatConversation) -> RocketChatSeedReceipt:
        auth = self.transport.login(self.credentials)
        context = _context(auth, role="writer", username=conversation.writer_user)
        physical_room_id = _identifier(
            self.transport.channel_id(conversation.room_id), "resolved room id"
        )
        identities: dict[str, RocketChatMessageIdentity] = {}
        for fact in conversation.messages:
            parent = identities.get(fact.thread_key) if fact.thread_key else None
            if fact.thread_key and parent is None:
                raise RocketChatContractError(f"message {fact.logical_key!r} references an unseeded thread")
            identities[fact.logical_key] = _identity(
                self.transport.send_message(room_id=physical_room_id, body=fact.body, thread_id=parent.message_id if parent else None),
                conversation, fact, "pending", parent.message_id if parent else None, physical_room_id,
            )
        digest = hashlib.sha256("|".join(item.message_id for item in identities.values()).encode()).hexdigest()[:24]
        attempt_id = f"rc-attempt-{digest}"
        normalized = {
            key: RocketChatMessageIdentity(
                benchmark=item.benchmark, site=item.site, attempt_id=attempt_id,
                logical_key=item.logical_key, room_id=item.room_id, message_id=item.message_id,
                thread_id=item.thread_id, author=item.author, body=item.body,
            )
            for key, item in identities.items()
        }
        return RocketChatSeedReceipt(conversation.benchmark, conversation.site, attempt_id, context, normalized)
class RocketChatHttpReader:
    def __init__(self, instance: Mapping[str, Any], *, transport: RocketChatTransport, credentials: RocketChatCredentials | None = None) -> None:
        self.instance = dict(instance)
        self.transport = transport
        self.credentials = credentials or _credentials(self.instance, "reader")
    def observe(self, conversation: RocketChatConversation, receipt: RocketChatSeedReceipt) -> RocketChatObservation | RocketChatObservationFailure:
        try:
            if not isinstance(conversation, RocketChatConversation):
                return RocketChatObservationFailure("invalid_conversation", "reader requires a typed conversation")
            if not isinstance(receipt, RocketChatSeedReceipt):
                return RocketChatObservationFailure("invalid_seed_receipt", "reader requires a typed seed receipt")
            auth = self.transport.login(self.credentials)
            writer = receipt.writer_context
            if self.credentials.username == writer.user_id:
                return RocketChatObservationFailure("writer_context_reused", "reader must use fresh independent credentials")
            reader = _context(auth, role="reader", username=conversation.reader_user)
            if reader.user_id == writer.user_id or reader.session_id == writer.session_id or reader.auth_context_id == writer.auth_context_id:
                return RocketChatObservationFailure("writer_context_reused", "reader must use fresh independent credentials")
            root = receipt.messages.get(conversation.thread_key)
            if root is None:
                return RocketChatObservationFailure("missing_seed_identity", "receipt omits the conversation thread root")
            rows = self.transport.history(room_id=root.room_id)
            by_id: dict[str, Mapping[str, Any]] = {}
            for row in rows:
                message_id = row.get("_id")
                if isinstance(message_id, str):
                    if message_id in by_id:
                        return RocketChatObservationFailure("ambiguous_message_identity", "history returned duplicate message IDs")
                    by_id[message_id] = row
            expected_ids = [receipt.messages[fact.logical_key].message_id for fact in conversation.messages if fact.logical_key in receipt.messages]
            if all(message_id in by_id for message_id in expected_ids) and [row.get("_id") for row in rows if row.get("_id") in expected_ids] != expected_ids:
                return RocketChatObservationFailure("message_order_mismatch", "history did not expose receipt IDs in conversation order")
            observed: dict[str, RocketChatMessageIdentity] = {}
            for fact in conversation.messages:
                expected = receipt.messages.get(fact.logical_key)
                if expected is None or expected.message_id not in by_id:
                    return RocketChatObservationFailure("stale_message_identity", f"history omitted {fact.logical_key} message identity")
                parent = receipt.messages.get(fact.thread_key) if fact.thread_key else None
                try:
                    observed[fact.logical_key] = _identity(
                        by_id[expected.message_id],
                        conversation,
                        fact,
                        receipt.attempt_id,
                        parent.message_id if parent else None,
                        root.room_id,
                    )
                except (RocketChatContractError, RocketChatTransportError) as exc:
                    return RocketChatObservationFailure("message_identity_mismatch", str(exc))
            if {item.message_id for item in observed.values()} != {item.message_id for item in receipt.messages.values()}:
                return RocketChatObservationFailure("stale_message_identity", "history did not expose exactly the receipt IDs")
            return RocketChatObservation(
                conversation.benchmark, conversation.site, receipt.attempt_id,
                root.room_id, receipt.messages[conversation.thread_key].message_id,
                reader, observed, conversation.current_decision,
            )
        except (RocketChatContractError, RocketChatTransportError) as exc:
            return RocketChatObservationFailure("reader_transport_failed", str(exc))
@dataclass(frozen=True)
class RocketChatReaderPreflight:
    ok: bool
    reason: str | None = None
    detail: str = ""
    browser_context_kwargs: Mapping[str, Any] = MappingProxyType({})
    def __post_init__(self) -> None:
        if self.ok and self.reason is not None:
            raise ValueError("successful Rocket.Chat preflight cannot have a reason")
        if not self.ok and not self.reason:
            raise ValueError("failed Rocket.Chat preflight needs a reason")
        object.__setattr__(self, "browser_context_kwargs", MappingProxyType(dict(self.browser_context_kwargs)))

    def to_metadata(self) -> dict[str, Any]:
        return {"reader_auth": "independent_authenticated_ordinary_reader", "fresh_context_required": True, "writer_context_reuse_forbidden": True}


def preflight_rocket_chat_reader(instance: Mapping[str, Any]) -> RocketChatReaderPreflight:
    if not isinstance(instance, Mapping):
        return RocketChatReaderPreflight(False, "invalid_instance", "reader instance must be a mapping")
    declared = instance.get("reader_auth")
    if not isinstance(declared, Mapping):
        return RocketChatReaderPreflight(False, "missing_reader_auth", "Rocket.Chat requires explicit reader_auth")
    try:
        _credentials(instance, "reader")
    except RocketChatContractError as exc:
        return RocketChatReaderPreflight(False, "reader_credentials_missing", str(exc))
    auth_type = str(declared.get("type") or "credentials").strip().lower()
    if auth_type == "storage_state":
        state = declared.get("storage_state", declared.get("path"))
        path = state.get("path") if isinstance(state, Mapping) else state
        if not isinstance(path, str) or not path.strip() or not os.path.exists(path):
            return RocketChatReaderPreflight(False, "reader_storage_missing", "reader storage_state path is unavailable")
        return RocketChatReaderPreflight(True, browser_context_kwargs={"storage_state": path})
    if auth_type == "http_headers":
        headers = declared.get("headers")
        if not isinstance(headers, Mapping) or not all(isinstance(k, str) and isinstance(v, str) and v for k, v in headers.items()):
            return RocketChatReaderPreflight(False, "reader_headers_invalid", "reader headers must be non-empty strings")
        if not {"X-Auth-Token", "X-User-Id"}.issubset(headers):
            return RocketChatReaderPreflight(False, "reader_headers_incomplete", "reader headers require X-Auth-Token and X-User-Id")
        return RocketChatReaderPreflight(True, browser_context_kwargs={"extra_http_headers": dict(headers)})
    if auth_type == "credentials":
        return RocketChatReaderPreflight(False, "reader_browser_auth_unavailable", "REST credentials need reader storage_state or auth headers for browser rendering")
    return RocketChatReaderPreflight(False, "reader_auth_unsupported", f"unsupported reader_auth type {auth_type!r}")

class RocketChatRuntimeSite(RocketChatSite):
    """Source-only TAC Site wiring; browser checks remain body-signature based."""

    def build_read_surface_plan(self, *, seed_result: EditorSeedResult, signature: str, origin: str) -> ReadSurfaceVerificationPlan | ReadSurfacePlanFailure:
        required = ("room_id", "room_name", "thread_id", "plan_message_id", "update_message_id", "correction_message_id", "reader_user_id", "reader_auth_context_id")
        missing = [key for key in required if seed_result.write_tokens.get(key) in (None, "")]
        if missing:
            return ReadSurfacePlanFailure("rocketchat", "missing_message_identity", "Rocket.Chat readback requires " + ", ".join(missing))
        plan = build_read_surface_plan(
            site="rocketchat", seed_result=seed_result, signature=signature, origin=origin,
            identity_keys=("attempt_id", *required, "plan_body_sha256", "update_body_sha256", "correction_body_sha256"),
        )
        if isinstance(plan, ReadSurfaceVerificationPlan):
            # IDs document REST only; no DOM observer means body text, not exact painted identity.
            return replace(plan, verification_mode="body_text")
        return plan

@dataclass(frozen=True)
class RocketChatFeasibilityPolicy:
    benchmark: str = ROCKET_CHAT_BENCHMARK
    site: str = ROCKET_CHAT_SITE

    def auth_self_test_path(self) -> str | None:
        return "/api/v1/me"

    def requires_authenticated_preflight(self) -> bool:
        return True

    def probe_targets(self, task: dict[str, Any], instance_site_url: str) -> list[Any]:
        from warp_taskgen.phase_2.phase_2c.policy import ProbeTarget

        starts = task.get("start_urls")
        if isinstance(starts, list):
            return [ProbeTarget(url=urljoin(instance_site_url.rstrip("/") + "/", value), source="start_url") for value in starts if isinstance(value, str) and value.startswith("/")]
        conversation = task.get("conversation")
        room_id = conversation.get("room_id") if isinstance(conversation, Mapping) else None
        return [ProbeTarget(url=urljoin(instance_site_url.rstrip("/") + "/", f"/channel/{room_id}"), source="conversation.room_id")] if isinstance(room_id, str) and room_id else []

    def classify_probe(self, *, status: int | None, headers: dict[str, str] | None, body_snippet: str, exception_name: str | None) -> Any:
        from warp_taskgen.phase_2.phase_2c.policy import PreflightClassification

        del headers, body_snippet
        if exception_name:
            return PreflightClassification("host_unreachable", False, status, f"Rocket.Chat probe raised {exception_name}")
        if status in {401, 403}:
            return PreflightClassification("auth_missing", True, status, f"Rocket.Chat probe returned HTTP {status}")
        if status == 404:
            return PreflightClassification("not_found", True, status, "Rocket.Chat room was not found")
        if status is not None and 200 <= status < 300:
            return PreflightClassification("reachable", False, status, "Rocket.Chat probe reachable")
        return PreflightClassification("unexpected_status", False, status, f"Rocket.Chat probe returned HTTP {status}")

    def decide_source_data(self, *, task: dict[str, Any], classifications_by_target: dict[int, list[Any]], target_audit: dict[int, Any], candidate_replica_count: int, login_redirect_count: int, probed_count: int, bailout_ratio: float) -> Any:
        from warp_taskgen.phase_2.phase_2c.policy import SourceDataDecision

        del task, candidate_replica_count, login_redirect_count, probed_count, bailout_ratio
        for index, classifications in classifications_by_target.items():
            for classification in classifications:
                if classification.quarantine:
                    return SourceDataDecision("drop", classification=classification, target=target_audit[index])
        return SourceDataDecision("keep")

    def counts_toward_run_bailout(self, classification: Any) -> bool:
        return classification.kind == "auth_missing"

    def should_bailout_source_data_run(self, *, bailout_count: int, probed_count: int, bailout_ratio: float) -> bool:
        return bool(probed_count) and bailout_count / probed_count > bailout_ratio

    def restore_drop_on_run_bailout(self, issue: dict[str, Any]) -> bool:
        return issue.get("kind") == "auth_missing"

class RocketChatHttpEditor:
    site_name = ROCKET_CHAT_SITE
    supported_methods = frozenset({ROCKET_CHAT_EDITOR_METHOD})

    def __init__(
        self,
        instance: dict[str, Any],
        session: requests.Session,
        *,
        transport: RocketChatTransport | None = None,
        reader_transport: RocketChatTransport | None = None,
    ) -> None:
        self.instance = dict(instance)
        self.session = session
        self.transport = transport or RequestsRocketChatTransport(_origin(instance), session)
        self.reader_transport = reader_transport or RequestsRocketChatTransport(
            _origin(instance), requests.Session()
        )
        self._reader_closed = False
        self._mutation_possible = False

    @classmethod
    def probe_base_state(cls, instance: dict[str, Any]) -> None:
        RequestsRocketChatTransport(_origin(instance)).login(_credentials(instance, "writer"))

    def probe_authenticated(self) -> bool:
        try:
            self.transport.login(_credentials(self.instance, "writer"))
            return True
        except (RocketChatContractError, RocketChatTransportError):
            return False

    def validate_args(self, method_name: str, args: dict[str, Any]) -> None:
        if method_name != ROCKET_CHAT_EDITOR_METHOD:
            raise RuntimeError(f"unsupported Rocket.Chat editor method {method_name!r}")
        if not isinstance(args.get("conversation"), Mapping):
            raise ValueError("Rocket.Chat seed conversation must be a mapping")

    def preview_context(self, method_name: str, args: dict[str, Any]) -> dict[str, Any]:
        del method_name, args
        return {}

    def seed_rocket_chat_conversation(self, *, conversation: Mapping[str, Any]) -> dict[str, Any]:
        from warp_taskgen.phase_1.rocket_chat_contracts import RocketChatDecision
        from warp_taskgen.phase_1.rocket_chat_decisions import _validate_conversation

        expected = conversation.get("expected_decision") if isinstance(conversation, Mapping) else None
        if not isinstance(expected, Mapping):
            raise ValueError("Rocket.Chat seed conversation expected_decision is required")
        typed = _validate_conversation(conversation, RocketChatDecision.from_mapping(expected))
        # Arm strict cleanup before the first writer POST, which may mutate before returning.
        self._mutation_possible = True
        receipt = RocketChatHttpWriter(self.instance, transport=self.transport).seed_conversation(typed)
        observation = RocketChatHttpReader(
            self.instance, transport=self.reader_transport
        ).observe(typed, receipt)
        if isinstance(observation, RocketChatObservationFailure):
            self._close_reader_transport()
            raise RocketChatTransportError(
                f"independent reader observation failed: {observation.reason}: {observation.detail}"
            )
        if not isinstance(observation, RocketChatObservation):
            self._close_reader_transport()
            raise RocketChatTransportError("independent reader observation returned an unsupported result")
        tokens: dict[str, str] = {
            "attempt_id": receipt.attempt_id,
            "room_id": observation.room_id,
            "room_name": typed.room_id,
            "thread_id": observation.thread_id,
            "reader_user_id": observation.reader_context.user_id,
            "reader_auth_context_id": observation.reader_context.auth_context_id,
        }
        for key, identity in receipt.messages.items():
            tokens[f"{key}_message_id"] = identity.message_id
            tokens[f"{key}_body_sha256"] = hashlib.sha256(identity.body.encode()).hexdigest()
        return {
            "identity_tokens": tokens,
            "read_surface_urls": [f"/channel/{typed.room_id}"],
            "read_surface_provenance_source": "editor_api_response",
            "created_resource": {"url": f"/channel/{typed.room_id}", "kind": "message", "id": receipt.messages[typed.thread_key].message_id},
        }

    def cleanup(self) -> None:
        self._close_reader_transport()
        if self._mutation_possible:
            raise RuntimeError("Rocket.Chat cleanup requires an explicit disposable TAC reset/admin seam; ordinary writer credentials are not used for reset")

    def _close_reader_transport(self) -> None:
        if self._reader_closed:
            return
        close = getattr(self.reader_transport, "close", None)
        if not callable(close):
            close = getattr(getattr(self.reader_transport, "session", None), "close", None)
        if callable(close):
            close()
        self._reader_closed = True

def rocket_chat_credentials(instance: Mapping[str, Any], *, role: str) -> RocketChatCredentials:
    return _credentials(instance, role)
