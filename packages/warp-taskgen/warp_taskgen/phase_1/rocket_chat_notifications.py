"""Rocket.Chat conversation-to-notification task contracts.

The response-only conversation family owns the generated facts and current
decision rule.  This module adds only the dependent notification action and
its exact identity contract; transports and readers are supplied by callers.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from warp_taskgen.phase_1.rocket_chat_contracts import (
    ROCKET_CHAT_BENCHMARK,
    ROCKET_CHAT_EVALUATOR_AUTHORITY,
    ROCKET_CHAT_SITE,
    RocketChatContractError,
    RocketChatConversation,
    RocketChatDecision,
    RocketChatMessageIdentity,
    RocketChatObservationFailure,
    RocketChatParticipantContext,
    RocketChatSeedReceipt,
    _identity,
    _text,
    validate_rocket_chat_benchmark,
)
from warp_taskgen.phase_1.rocket_chat_decisions import (
    _validate_conversation,
    derive_rocket_chat_decision,
    resolve_rocket_chat_evaluator_authority,
)

ROCKET_CHAT_NOTIFICATION_TASK_KIND = "rocket_chat_conversation_notification"
ROCKET_CHAT_NOTIFICATION_ACTION_KIND = "send_notification"
ROCKET_CHAT_NOTIFICATION_EVALUATOR_NAME = "RocketChatNotificationEvaluator"
ROCKET_CHAT_NOTIFICATION_LOGICAL_KEY = "notification"
ROCKET_CHAT_NOTIFICATION_BODY_PREFIX = "Current decision:"
ROCKET_CHAT_NOTIFICATION_MESSAGE_IDENTITY = "independent_reader_discovered_exact_message_id"
_MENTION_USERNAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_NOTIFICATION_FIELDS = frozenset(
    "benchmark site room_id thread_key author recipient body decision".split()
)
_ACTION_FIELDS = frozenset(
    "action_kind room_id thread_key recipient_source recipient author ordinary_participant message_identity".split()
)
_READER_FIELDS = frozenset(
    "auth writer_identity reader_identity identities_must_be_distinct resource_kind message_identity".split()
)


def render_rocket_chat_notification_body(decision: Mapping[str, object]) -> str:
    """Render the exact bounded message sent for the current decision."""

    if not isinstance(decision, Mapping):
        raise RocketChatContractError("notification decision must be a mapping")
    try:
        owner = _text(decision["owner"], field="notification decision owner")
        due_date = _text(decision["due_date"], field="notification decision due date")
    except KeyError as exc:
        raise RocketChatContractError("notification decision requires owner and due_date") from exc
    return f"{ROCKET_CHAT_NOTIFICATION_BODY_PREFIX} owner={owner}; due_date={due_date}."


def validate_rocket_chat_notification_recipient(value: object) -> str:
    """Return one recipient that can be encoded as an exact Rocket.Chat mention."""

    recipient = _text(value, field="notification recipient", max_length=128)
    if _MENTION_USERNAME_RE.fullmatch(recipient) is None:
        raise RocketChatContractError(
            "notification recipient cannot be encoded as a Rocket.Chat mention"
        )
    return recipient


@dataclass(frozen=True)
class RocketChatNotification:
    """The action selected from one validated generated conversation."""

    benchmark: str
    site: str
    room_id: str
    thread_key: str
    author: str
    recipient: str
    body: str
    decision: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "benchmark", validate_rocket_chat_benchmark(self.benchmark))
        if _identity(self.site, field="notification Site") != ROCKET_CHAT_SITE:
            raise RocketChatContractError("notification Site identity does not match Rocket.Chat")
        object.__setattr__(self, "site", ROCKET_CHAT_SITE)
        object.__setattr__(self, "room_id", _identity(self.room_id, field="notification room"))
        object.__setattr__(
            self, "thread_key", _identity(self.thread_key, field="notification thread")
        )
        object.__setattr__(self, "author", _identity(self.author, field="notification author"))
        object.__setattr__(
            self,
            "recipient",
            validate_rocket_chat_notification_recipient(self.recipient),
        )
        object.__setattr__(
            self,
            "body",
            _text(self.body, field="notification body", max_length=2000),
        )
        if not isinstance(self.decision, Mapping):
            raise RocketChatContractError("notification decision must be a mapping")
        decision = dict(self.decision)
        if set(decision) != {"owner", "due_date"}:
            raise RocketChatContractError("notification decision fields must be exact")
        owner = _text(decision["owner"], field="notification decision owner")
        due_date = _text(decision["due_date"], field="notification decision due date")
        normalized = {"owner": owner, "due_date": due_date}
        if self.recipient != owner:
            raise RocketChatContractError(
                "notification recipient must be derived from the current decision owner"
            )
        if self.body != render_rocket_chat_notification_body(normalized):
            raise RocketChatContractError("notification body does not match current decision")
        object.__setattr__(self, "decision", normalized)

    @property
    def current_decision(self) -> dict[str, str]:
        return dict(self.decision)  # type: ignore[return-value]

    def as_dict(self) -> dict[str, object]:
        return {
            "benchmark": self.benchmark,
            "site": self.site,
            "room_id": self.room_id,
            "thread_key": self.thread_key,
            "author": self.author,
            "recipient": self.recipient,
            "body": self.body,
            "decision": dict(self.decision),
        }


def derive_rocket_chat_notification(
    conversation: RocketChatConversation,
) -> RocketChatNotification:
    """Derive one recipient and body using the response family's rule."""

    if not isinstance(conversation, RocketChatConversation):
        raise TypeError("derive_rocket_chat_notification requires a RocketChatConversation")
    decision = derive_rocket_chat_decision(conversation)
    return RocketChatNotification(
        benchmark=conversation.benchmark,
        site=conversation.site,
        room_id=conversation.room_id,
        thread_key=conversation.thread_key,
        author=conversation.writer_user,
        recipient=decision.owner,
        body=render_rocket_chat_notification_body(decision.as_dict()),
        decision=decision.as_dict(),
    )


@dataclass(frozen=True)
class RocketChatNotificationReceipt:
    """Writer-returned binding for the exact persisted notification message."""

    benchmark: str
    site: str
    attempt_id: str
    writer_context: RocketChatParticipantContext
    notification: RocketChatNotification
    thread_id: str
    message: RocketChatMessageIdentity
    seed_root: RocketChatMessageIdentity
    current_message: RocketChatMessageIdentity

    def __post_init__(self) -> None:
        object.__setattr__(self, "benchmark", validate_rocket_chat_benchmark(self.benchmark))
        if _identity(self.site, field="notification receipt Site") != ROCKET_CHAT_SITE:
            raise RocketChatContractError("notification receipt Site identity does not match")
        object.__setattr__(self, "site", ROCKET_CHAT_SITE)
        object.__setattr__(
            self, "attempt_id", _identity(self.attempt_id, field="notification attempt")
        )
        if not isinstance(self.writer_context, RocketChatParticipantContext):
            raise RocketChatContractError("notification receipt requires a typed writer context")
        if self.writer_context.auth_kind != "writer_credentials":
            raise RocketChatContractError("notification receipt writer must use writer credentials")
        if self.writer_context.role != "ordinary":
            raise RocketChatContractError(
                "notification receipt writer must be an ordinary participant"
            )
        if not isinstance(self.notification, RocketChatNotification):
            raise RocketChatContractError("notification receipt requires a typed notification")
        if self.notification.benchmark != self.benchmark or self.notification.site != self.site:
            raise RocketChatContractError("notification and receipt Benchmark/Site must match")
        object.__setattr__(
            self, "thread_id", _identity(self.thread_id, field="notification thread ID")
        )
        if not isinstance(self.message, RocketChatMessageIdentity):
            raise RocketChatContractError(
                "notification receipt requires writer-returned message identity"
            )
        if self.message.benchmark != self.benchmark or self.message.site != self.site:
            raise RocketChatContractError(
                "notification message Benchmark/Site do not match receipt"
            )
        if self.message.attempt_id != self.attempt_id:
            raise RocketChatContractError("notification message belongs to a different attempt")
        if self.message.logical_key != ROCKET_CHAT_NOTIFICATION_LOGICAL_KEY:
            raise RocketChatContractError("notification message has an unsupported logical key")
        if self.message.thread_id != self.thread_id:
            raise RocketChatContractError(
                "notification receipt thread ID must bind the writer-returned message"
            )
        if not isinstance(self.seed_root, RocketChatMessageIdentity):
            raise RocketChatContractError(
                "notification receipt requires the typed current seed root identity"
            )
        if self.seed_root.benchmark != self.benchmark or self.seed_root.site != self.site:
            raise RocketChatContractError(
                "notification receipt seed root Benchmark/Site do not match"
            )
        if self.seed_root.attempt_id != self.attempt_id:
            raise RocketChatContractError(
                "notification receipt seed root belongs to a different attempt"
            )
        if self.seed_root.thread_id is not None:
            raise RocketChatContractError("notification receipt seed root must not be threaded")
        if not isinstance(self.current_message, RocketChatMessageIdentity):
            raise RocketChatContractError(
                "notification receipt requires the typed current message identity"
            )

    @property
    def room_id(self) -> str:
        return self.message.room_id

    @property
    def recipient(self) -> str | None:
        return self.message.recipient

    @property
    def author(self) -> str:
        return self.message.author

    @property
    def body(self) -> str:
        return self.message.body

    @property
    def thread_key(self) -> str:
        return self.notification.thread_key

    @property
    def message_id(self) -> str:
        return self.message.message_id


@dataclass(frozen=True)
class RocketChatNotificationObservation:
    """Fresh-reader observation of the exact notification message."""

    benchmark: str
    site: str
    attempt_id: str
    reader_context: RocketChatParticipantContext
    notification: RocketChatNotification
    thread_id: str
    message: RocketChatMessageIdentity

    def __post_init__(self) -> None:
        object.__setattr__(self, "benchmark", validate_rocket_chat_benchmark(self.benchmark))
        if _identity(self.site, field="notification observation Site") != ROCKET_CHAT_SITE:
            raise RocketChatContractError("notification observation Site identity does not match")
        object.__setattr__(self, "site", ROCKET_CHAT_SITE)
        object.__setattr__(
            self, "attempt_id", _identity(self.attempt_id, field="notification observation attempt")
        )
        if not isinstance(self.reader_context, RocketChatParticipantContext):
            raise RocketChatContractError(
                "notification observation requires a typed reader context"
            )
        if self.reader_context.auth_kind != "reader_credentials":
            raise RocketChatContractError("notification observation requires reader credentials")
        if self.reader_context.role != "ordinary":
            raise RocketChatContractError("notification observation reader must be ordinary")
        if not isinstance(self.notification, RocketChatNotification):
            raise RocketChatContractError("notification observation requires a typed notification")
        if self.notification.benchmark != self.benchmark or self.notification.site != self.site:
            raise RocketChatContractError("notification and observation Benchmark/Site must match")
        reader_username = self.reader_context.username or self.reader_context.user_id
        if reader_username == self.notification.author:
            raise RocketChatContractError(
                "notification observation reader must be distinct from the notification author"
            )
        object.__setattr__(
            self, "thread_id", _identity(self.thread_id, field="notification observation thread ID")
        )
        if not isinstance(self.message, RocketChatMessageIdentity):
            raise RocketChatContractError("notification observation requires a message identity")
        if self.message.benchmark != self.benchmark or self.message.site != self.site:
            raise RocketChatContractError(
                "notification observation message Benchmark/Site do not match"
            )
        if self.message.attempt_id != self.attempt_id:
            raise RocketChatContractError(
                "notification observation message belongs to a different attempt"
            )
        if self.message.logical_key != ROCKET_CHAT_NOTIFICATION_LOGICAL_KEY:
            raise RocketChatContractError(
                "notification observation message has an unsupported logical key"
            )
        expected = {
            "thread_id": self.thread_id,
            "author": self.notification.author,
            "recipient": self.notification.recipient,
            "body": self.notification.body,
        }
        actual = {
            "thread_id": self.message.thread_id,
            "author": self.message.author,
            "recipient": self.message.recipient,
            "body": self.message.body,
        }
        # ``notification.room_id`` is the generated logical channel.  A Site
        # adapter may resolve it to a different physical REST room; the
        # adapter/evaluator binds that physical identity to the seed root.
        if actual != expected:
            raise RocketChatContractError("persisted notification identity is inconsistent")

    @property
    def message_id(self) -> str:
        return self.message.message_id

    @property
    def message_identity(self) -> RocketChatMessageIdentity:
        return self.message


@runtime_checkable
class RocketChatNotificationWriter(Protocol):
    """Ordinary-participant action boundary supplied by a Site adapter."""

    def send_notification(
        self,
        conversation: RocketChatConversation,
        seed_receipt: RocketChatSeedReceipt,
        notification: RocketChatNotification,
    ) -> RocketChatNotificationReceipt: ...


@runtime_checkable
class RocketChatNotificationReader(Protocol):
    """Independent authenticated reader boundary for persisted notifications."""

    def observe_notification(
        self,
        conversation: RocketChatConversation,
        seed_receipt: RocketChatSeedReceipt,
        notification_receipt: RocketChatNotificationReceipt,
    ) -> RocketChatNotificationObservation | RocketChatObservationFailure: ...


def compile_rocket_chat_notification_task(
    conversation: RocketChatConversation,
) -> dict[str, object]:
    """Compile a task that asks for the current decision and one notification."""

    if not isinstance(conversation, RocketChatConversation):
        raise TypeError("compile_rocket_chat_notification_task requires a RocketChatConversation")
    notification = derive_rocket_chat_notification(conversation)
    authority = resolve_rocket_chat_evaluator_authority(conversation.benchmark)
    expected = {
        **notification.as_dict(),
        "message_identity": ROCKET_CHAT_NOTIFICATION_MESSAGE_IDENTITY,
    }
    task: dict[str, object] = {
        "benchmark": conversation.benchmark,
        "site": conversation.site,
        "task_kind": ROCKET_CHAT_NOTIFICATION_TASK_KIND,
        "task_id": None,
        "evaluator_authority": authority,
        "start_urls": [f"__ROCKETCHAT__/channel/{conversation.room_id}"],
        "conversation": conversation.as_dict(),
        "response_schema": {
            "type": "object",
            "required": ["owner", "due_date"],
            "additionalProperties": False,
            "properties": {"owner": {"type": "string"}, "due_date": {"type": "string"}},
        },
        "expected_decision": conversation.expected_decision.as_dict(),
        "notification": notification.as_dict(),
        "action_contract": {
            "action_kind": ROCKET_CHAT_NOTIFICATION_ACTION_KIND,
            "room_id": notification.room_id,
            "thread_key": notification.thread_key,
            "recipient_source": "current_decision.owner",
            "recipient": notification.recipient,
            "author": notification.author,
            "ordinary_participant": True,
            "message_identity": ROCKET_CHAT_NOTIFICATION_MESSAGE_IDENTITY,
        },
        "reward_function": {
            "eval": [{"evaluator": ROCKET_CHAT_NOTIFICATION_EVALUATOR_NAME, "expected": expected}]
        },
        "reader_contract": {
            "auth": "independent_authenticated_ordinary_reader",
            "writer_identity": conversation.writer_user,
            "reader_identity": conversation.reader_user,
            "identities_must_be_distinct": True,
            "resource_kind": "message",
            "message_identity": "exact_room_recipient_author_thread_body_message_id",
        },
    }
    validate_rocket_chat_notification_task(task)
    return task


_NOTIFICATION_TASK_KEYS = frozenset(
    "benchmark site task_kind task_id evaluator_authority start_urls conversation response_schema expected_decision notification action_contract reward_function reader_contract".split()
)


def _require_exact_keys(value: Mapping[str, object], expected: frozenset[str], label: str) -> None:
    missing = expected - set(value)
    extra = set(value) - expected
    if missing:
        raise RocketChatContractError(
            f"{label} is missing fields: {', '.join(sorted(str(item) for item in missing))}"
        )
    if extra:
        raise RocketChatContractError(
            f"{label} has extra fields: {', '.join(sorted(str(item) for item in extra))}"
        )


def validate_rocket_chat_notification_task(task: Mapping[str, object]) -> None:
    """Validate the notification task's static action and evaluator contract."""

    if not isinstance(task, Mapping):
        raise RocketChatContractError("Rocket.Chat notification task must be a mapping")
    _require_exact_keys(task, _NOTIFICATION_TASK_KEYS, "Rocket.Chat notification task")
    if validate_rocket_chat_benchmark(task["benchmark"]) != ROCKET_CHAT_BENCHMARK:
        raise RocketChatContractError("Rocket.Chat notification Benchmark must be TAC")
    if task["site"] != ROCKET_CHAT_SITE:
        raise RocketChatContractError("Rocket.Chat notification Site must be rocketchat")
    if task["task_kind"] != ROCKET_CHAT_NOTIFICATION_TASK_KIND:
        raise RocketChatContractError("Rocket.Chat notification task kind is unsupported")
    if task["task_id"] is not None:
        raise RocketChatContractError("Rocket.Chat notification task must remain task-id-less")
    if task["evaluator_authority"] != ROCKET_CHAT_EVALUATOR_AUTHORITY:
        raise RocketChatContractError(
            "Rocket.Chat notification evaluator authority is not WARP-local"
        )
    conversation_value = task["conversation"]
    if not isinstance(conversation_value, Mapping):
        raise RocketChatContractError("Rocket.Chat notification conversation is required")
    expected_decision_value = task["expected_decision"]
    if not isinstance(expected_decision_value, Mapping):
        raise RocketChatContractError("Rocket.Chat notification expected decision is required")
    try:
        expected_conversation = _validate_conversation(
            conversation_value,
            RocketChatDecision.from_mapping(expected_decision_value),
        )
    except (TypeError, ValueError) as exc:
        raise RocketChatContractError(
            f"Rocket.Chat notification conversation facts are inconsistent: {exc}"
        ) from exc

    notification_value = task["notification"]
    if not isinstance(notification_value, Mapping):
        raise RocketChatContractError("Rocket.Chat notification facts are required")
    _require_exact_keys(notification_value, _NOTIFICATION_FIELDS, "notification")
    decision = dict(expected_decision_value)
    if notification_value["decision"] != decision:
        raise RocketChatContractError("notification decision is inconsistent")
    if notification_value["recipient"] != decision.get("owner"):
        raise RocketChatContractError("notification recipient is not derived from current decision")
    notification = RocketChatNotification(**dict(notification_value))  # type: ignore[arg-type]
    if notification.current_decision != decision:
        raise RocketChatContractError("notification decision is inconsistent")
    if notification.room_id != expected_conversation.room_id:
        raise RocketChatContractError("notification room is inconsistent with conversation")
    if notification.thread_key != expected_conversation.thread_key:
        raise RocketChatContractError("notification thread is inconsistent with conversation")
    if notification.author != expected_conversation.writer_user:
        raise RocketChatContractError("notification author is inconsistent with conversation")
    if task["start_urls"] != [f"__ROCKETCHAT__/channel/{notification.room_id}"]:
        raise RocketChatContractError("Rocket.Chat notification task must target its exact room")

    action_contract = task["action_contract"]
    if not isinstance(action_contract, Mapping):
        raise RocketChatContractError("Rocket.Chat notification action contract is required")
    _require_exact_keys(action_contract, _ACTION_FIELDS, "notification action contract")
    expected_action = {
        "action_kind": ROCKET_CHAT_NOTIFICATION_ACTION_KIND,
        "room_id": notification.room_id,
        "thread_key": notification.thread_key,
        "recipient_source": "current_decision.owner",
        "recipient": notification.recipient,
        "author": notification.author,
        "ordinary_participant": True,
        "message_identity": ROCKET_CHAT_NOTIFICATION_MESSAGE_IDENTITY,
    }
    if dict(action_contract) != expected_action:
        raise RocketChatContractError("notification action contract is inconsistent")

    reward = task["reward_function"]
    if not isinstance(reward, Mapping):
        raise RocketChatContractError("Rocket.Chat notification reward function is required")
    if set(reward) != {"eval"} or not isinstance(reward["eval"], list) or len(reward["eval"]) != 1:
        raise RocketChatContractError(
            "Rocket.Chat notification requires one feature-owned evaluator"
        )
    evaluator = reward["eval"][0]
    if not isinstance(evaluator, Mapping) or set(evaluator) != {"evaluator", "expected"}:
        raise RocketChatContractError("Rocket.Chat notification evaluator entry is invalid")
    if evaluator["evaluator"] != ROCKET_CHAT_NOTIFICATION_EVALUATOR_NAME:
        raise RocketChatContractError("Rocket.Chat notification evaluator is unsupported")
    if evaluator["expected"] != {
        **notification.as_dict(),
        "message_identity": ROCKET_CHAT_NOTIFICATION_MESSAGE_IDENTITY,
    }:
        raise RocketChatContractError(
            "Rocket.Chat notification evaluator expected state is inconsistent"
        )

    reader = task["reader_contract"]
    if not isinstance(reader, Mapping):
        raise RocketChatContractError("Rocket.Chat notification reader contract is required")
    _require_exact_keys(reader, _READER_FIELDS, "notification reader contract")
    if reader["auth"] != "independent_authenticated_ordinary_reader":
        raise RocketChatContractError("notification reader must be independent and authenticated")
    conversation_mapping = conversation_value
    if reader["writer_identity"] != conversation_mapping["writer_user"]:
        raise RocketChatContractError("notification reader writer identity is inconsistent")
    if reader["reader_identity"] != conversation_mapping["reader_user"]:
        raise RocketChatContractError("notification reader identity is inconsistent")
    if reader["identities_must_be_distinct"] is not True:
        raise RocketChatContractError("notification reader identities must be distinct")
    if (
        reader["resource_kind"] != "message"
        or reader["message_identity"] != "exact_room_recipient_author_thread_body_message_id"
    ):
        raise RocketChatContractError(
            "notification reader message identity contract is inconsistent"
        )


__all__ = [
    "ROCKET_CHAT_NOTIFICATION_ACTION_KIND",
    "ROCKET_CHAT_NOTIFICATION_BODY_PREFIX",
    "ROCKET_CHAT_NOTIFICATION_EVALUATOR_NAME",
    "ROCKET_CHAT_NOTIFICATION_LOGICAL_KEY",
    "ROCKET_CHAT_NOTIFICATION_MESSAGE_IDENTITY",
    "ROCKET_CHAT_NOTIFICATION_TASK_KIND",
    "RocketChatNotification",
    "RocketChatNotificationObservation",
    "RocketChatNotificationReader",
    "RocketChatNotificationReceipt",
    "RocketChatNotificationWriter",
    "compile_rocket_chat_notification_task",
    "derive_rocket_chat_notification",
    "render_rocket_chat_notification_body",
    "validate_rocket_chat_notification_recipient",
    "validate_rocket_chat_notification_task",
]
