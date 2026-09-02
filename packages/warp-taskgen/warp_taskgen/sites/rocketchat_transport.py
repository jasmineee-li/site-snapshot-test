"""Feature-local Rocket.Chat REST transport, auth, and identity seams.

This module owns the authenticated writer and independent reader transport
used by the E1 Rocket.Chat decision composition. It intentionally leaves
thread history as a narrow optional protocol: exact conversation and
notification readback fail closed when a dedicated thread endpoint is
unavailable.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlsplit

import requests

from warp_taskgen.phase_1.rocket_chat_contracts import (
    RocketChatContractError,
    RocketChatConversation,
    RocketChatMessageFact,
    RocketChatMessageIdentity,
    RocketChatParticipantContext,
    RocketChatSeedReceipt,
)

_THREAD_PAGE_SIZE = 100
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
    roles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field in ("user_id", "username", "session_id"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip() or len(value.strip()) > _ID_MAX:
                raise RocketChatContractError(f"Rocket.Chat auth session {field} is invalid")
            object.__setattr__(self, field, value.strip())
        if not isinstance(self.roles, (tuple, list)):
            raise RocketChatContractError("Rocket.Chat auth session roles must be a sequence")
        roles: list[str] = []
        for role in self.roles:
            if not isinstance(role, str) or not role.strip() or len(role.strip()) > _ID_MAX:
                raise RocketChatContractError("Rocket.Chat auth session role is invalid")
            normalized = role.strip().lower()
            if normalized not in roles:
                roles.append(normalized)
        object.__setattr__(self, "roles", tuple(roles))


class RocketChatTransport(Protocol):
    def login(self, credentials: RocketChatCredentials) -> RocketChatAuthSession: ...
    def channel_id(self, channel: str) -> str: ...
    def send_message(
        self, *, room_id: str, body: str, thread_id: str | None = None
    ) -> Mapping[str, Any]: ...

    def history(self, *, room_id: str) -> Sequence[Mapping[str, Any]]: ...


class RocketChatThreadHistoryTransport(Protocol):
    """Feature-local extension for the dedicated threaded-history REST call.

    ``RocketChatHttpReader`` and notification readback discover this optional
    method with ``getattr`` and fail closed when a deployment cannot provide it.
    """

    def thread_history(self, *, room_id: str, thread_id: str) -> Sequence[Mapping[str, Any]]: ...


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise RocketChatContractError(f"Rocket.Chat {field} must be a string")
    result = value.strip()
    if not result or len(result) > _ID_MAX or any(char.isspace() for char in result):
        raise RocketChatContractError(f"Rocket.Chat {field} must be a bounded identifier")
    return result


def _declared_reader_user_id(instance: Mapping[str, Any]) -> str | None:
    declared = instance.get("reader_auth")
    if not isinstance(declared, Mapping):
        return None
    candidates = [
        declared.get("user_id"),
        declared.get("expected_reader_user_id"),
    ]
    values: list[str] = []
    for candidate in candidates:
        if candidate in (None, ""):
            continue
        values.append(_identifier(candidate, "reader browser user id"))
    if len(set(values)) > 1:
        raise RocketChatContractError(
            "Rocket.Chat reader_auth user_id and expected_reader_user_id disagree"
        )
    return values[0] if values else None


def _origin(instance: Mapping[str, Any]) -> str:
    raw = str(instance.get("site_url") or "").strip().rstrip("/")
    parsed = urlsplit(raw)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
    ):
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
            if (
                isinstance(username, str)
                and username.strip()
                and isinstance(password, str)
                and password
            ):
                return RocketChatCredentials(username, password)
    raise RocketChatContractError(
        f"Rocket.Chat {role} credentials are required in explicit {role}_auth/auth configuration"
    )


class RequestsRocketChatTransport:
    def __init__(
        self, site_url: str, session: requests.Session | None = None, *, timeout_s: float = 30.0
    ) -> None:
        parsed = urlsplit(str(site_url or "").strip().rstrip("/"))
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.query
            or parsed.fragment
        ):
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
            raise RocketChatTransportError(
                f"Rocket.Chat REST {method.upper()} {path} failed: {exc.__class__.__name__}"
            ) from exc
        if 300 <= response.status_code < 400:
            raise RocketChatTransportError(
                f"Rocket.Chat REST {path} returned an unexpected redirect"
            )
        if response.status_code in {401, 403}:
            raise RocketChatTransportError(
                f"Rocket.Chat REST {path} rejected authentication (HTTP {response.status_code})"
            )
        if response.status_code < 200 or response.status_code >= 300:
            raise RocketChatTransportError(
                f"Rocket.Chat REST {path} returned HTTP {response.status_code}"
            )
        try:
            payload = response.json()
        except (TypeError, ValueError) as exc:
            raise RocketChatTransportError(
                f"Rocket.Chat REST {path} returned non-JSON data"
            ) from exc
        if not isinstance(payload, Mapping):
            raise RocketChatTransportError(f"Rocket.Chat REST {path} returned a non-object")
        return payload

    @staticmethod
    def _require_success(payload: Mapping[str, Any], path: str) -> None:
        if payload.get("success") is not True:
            raise RocketChatTransportError(f"Rocket.Chat REST {path} did not report success")

    def login(self, credentials: RocketChatCredentials) -> RocketChatAuthSession:
        payload = self._request(
            "POST",
            "/api/v1/login",
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
        me = self._request("GET", "/api/v1/me")
        self._require_success(me, "/api/v1/me")
        me_user: Mapping[str, Any] | None = None
        raw_user = me.get("user")
        if isinstance(raw_user, Mapping):
            me_user = raw_user
        elif "_id" in me or "username" in me:
            # Rocket.Chat 5.3 returns the authenticated user directly at the
            # top level; newer deployments may wrap it under user/data.
            me_user = me
        else:
            data_user = me.get("data")
            if isinstance(data_user, Mapping):
                nested_user = data_user.get("user")
                if isinstance(nested_user, Mapping):
                    me_user = nested_user
                elif "_id" in data_user or "username" in data_user:
                    me_user = data_user
        if me_user is None:
            raise RocketChatTransportError("Rocket.Chat /api/v1/me response is missing user")
        me_id = me_user.get("_id")
        me_username = me_user.get("username")
        raw_roles = me_user.get("roles")
        if (
            not isinstance(me_id, str)
            or not me_id.strip()
            or not isinstance(me_username, str)
            or not me_username.strip()
            or not isinstance(raw_roles, (list, tuple))
            or any(not isinstance(role, str) or not role.strip() for role in raw_roles)
        ):
            raise RocketChatTransportError(
                "Rocket.Chat /api/v1/me response is missing _id/username/roles"
            )
        roles = tuple(dict.fromkeys(role.strip().lower() for role in raw_roles))
        if me_id.strip() != user_id.strip():
            raise RocketChatTransportError(
                "Rocket.Chat /api/v1/me user ID does not match login userId"
            )
        if me_username.strip() != credentials.username:
            raise RocketChatTransportError(
                "Rocket.Chat /api/v1/me username does not match login credentials"
            )
        if "user" not in roles or "admin" in roles:
            raise RocketChatTransportError(
                "Rocket.Chat authenticated participant is not an ordinary user"
            )
        return RocketChatAuthSession(
            user_id=me_id,
            username=me_username,
            session_id=f"rc-session-{me_id}",
            roles=roles,
        )

    def channel_id(self, channel: str) -> str:
        channel = _identifier(channel, "channel")
        payload = self._request("GET", "/api/v1/channels.info", params={"roomName": channel})
        self._require_success(payload, "/api/v1/channels.info")
        room = payload.get("channel")
        if not isinstance(room, Mapping):
            data = payload.get("data")
            room = data.get("channel") if isinstance(data, Mapping) else None
        return _identifier(room.get("_id") if isinstance(room, Mapping) else None, "room id")

    def send_message(
        self, *, room_id: str, body: str, thread_id: str | None = None
    ) -> Mapping[str, Any]:
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
            raise RocketChatTransportError(
                "Rocket.Chat chat.sendMessage response is missing message"
            )
        return result

    def history(self, *, room_id: str) -> Sequence[Mapping[str, Any]]:
        room_id = _identifier(room_id, "room id")
        payload = self._request(
            "GET",
            "/api/v1/channels.history",
            params={"roomId": room_id, "count": _THREAD_PAGE_SIZE},
        )
        self._require_success(payload, "/api/v1/channels.history")
        messages = payload.get("messages")
        if not isinstance(messages, list) or any(
            not isinstance(item, Mapping) for item in messages
        ):
            raise RocketChatTransportError(
                "Rocket.Chat channels.history response has invalid messages"
            )
        return tuple(messages)

    def thread_history(self, *, room_id: str, thread_id: str) -> Sequence[Mapping[str, Any]]:
        """Read replies separately from the room history surface.

        Rocket.Chat deployments can omit ``tmid`` replies from
        ``channels.history`` even when ``showThreadMessages`` is requested.
        The dedicated endpoint is therefore part of the exact reader seam;
        callers must not treat a root-only room history as a complete
        conversation.
        """

        room_id = _identifier(room_id, "room id")
        thread_id = _identifier(thread_id, "thread id")
        payload = self._request(
            "GET",
            "/api/v1/chat.getThreadMessages",
            params={"tmid": thread_id, "count": _THREAD_PAGE_SIZE, "offset": 0},
        )
        self._require_success(payload, "/api/v1/chat.getThreadMessages")
        messages = payload.get("messages")
        if not isinstance(messages, list):
            data = payload.get("data")
            messages = data.get("messages") if isinstance(data, Mapping) else None
        if not isinstance(messages, list) or any(
            not isinstance(item, Mapping) for item in messages
        ):
            raise RocketChatTransportError(
                "Rocket.Chat chat.getThreadMessages response has invalid messages"
            )
        return tuple(messages)


def _context(
    auth: RocketChatAuthSession, *, role: str, username: str
) -> RocketChatParticipantContext:
    if auth.username != username:
        raise RocketChatContractError("Rocket.Chat login username does not match participant")
    if "user" not in auth.roles or "admin" in auth.roles:
        raise RocketChatContractError(
            "Rocket.Chat participant must have the ordinary user role without admin"
        )
    return RocketChatParticipantContext(
        user_id=auth.user_id,
        username=auth.username,
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
        raise RocketChatTransportError(
            "Rocket.Chat message response is missing _id/rid/msg/u.username"
        )
    if room_id != expected_room_id or body != fact.body or author != fact.author:
        raise RocketChatTransportError(
            f"Rocket.Chat message {fact.logical_key!r} identity/body mismatch"
        )
    if expected_thread_id is None:
        if thread_id not in (None, ""):
            raise RocketChatTransportError(
                "Rocket.Chat root message unexpectedly carries a thread id"
            )
        normalized_thread = None
    elif thread_id != expected_thread_id:
        raise RocketChatTransportError(f"Rocket.Chat message {fact.logical_key!r} thread mismatch")
    else:
        normalized_thread = thread_id
    return RocketChatMessageIdentity(
        benchmark=conversation.benchmark,
        site=conversation.site,
        attempt_id=attempt_id,
        logical_key=fact.logical_key,
        room_id=room_id,
        message_id=message_id,
        thread_id=normalized_thread,
        author=author,
        body=body,
    )


class RocketChatHttpWriter:
    def __init__(
        self,
        instance: Mapping[str, Any],
        *,
        transport: RocketChatTransport,
        credentials: RocketChatCredentials | None = None,
    ) -> None:
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
                raise RocketChatContractError(
                    f"message {fact.logical_key!r} references an unseeded thread"
                )
            identities[fact.logical_key] = _identity(
                self.transport.send_message(
                    room_id=physical_room_id,
                    body=fact.body,
                    thread_id=parent.message_id if parent else None,
                ),
                conversation,
                fact,
                "pending",
                parent.message_id if parent else None,
                physical_room_id,
            )
        digest = hashlib.sha256(
            "|".join(item.message_id for item in identities.values()).encode()
        ).hexdigest()[:24]
        attempt_id = f"rc-attempt-{digest}"
        normalized = {
            key: RocketChatMessageIdentity(
                benchmark=item.benchmark,
                site=item.site,
                attempt_id=attempt_id,
                logical_key=item.logical_key,
                room_id=item.room_id,
                message_id=item.message_id,
                thread_id=item.thread_id,
                author=item.author,
                body=item.body,
            )
            for key, item in identities.items()
        }
        return RocketChatSeedReceipt(
            conversation.benchmark, conversation.site, attempt_id, context, normalized
        )
