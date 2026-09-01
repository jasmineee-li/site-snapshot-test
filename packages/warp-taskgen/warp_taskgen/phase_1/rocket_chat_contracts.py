"""Typed records and narrow protocols for the Rocket.Chat response family.

This module contains no storage, browser, authentication, or deployment
implementation.  Those boundaries are represented by the two protocols and
are supplied by a caller (the focused tests use local fakes).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Protocol, runtime_checkable

from warp_taskgen.benchmark_capabilities import (
    get_benchmark_capabilities,
    infer_benchmark_from_metadata,
)

ROCKET_CHAT_BENCHMARK = "theagentcompany"
ROCKET_CHAT_SITE = "rocketchat"
ROCKET_CHAT_RESOURCE_KIND = "message"
ROCKET_CHAT_TASK_KIND = "rocket_chat_conversation_decision"
ROCKET_CHAT_EVALUATOR_AUTHORITY = "warp_local_task_idless"
ROCKET_CHAT_EVALUATOR_NAME = "RocketChatEvaluator"

DECISION_FIELDS = ("owner", "due_date")
MESSAGE_KINDS = frozenset({"plan", "update", "correction"})
READER_AUTH_KINDS = frozenset({"reader_credentials"})
WRITER_AUTH_KINDS = frozenset({"writer_credentials"})

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class RocketChatContractError(ValueError):
    """Raised when the bounded Rocket.Chat contract cannot be established."""


def _text(value: object, *, field: str, max_length: int = 240) -> str:
    if not isinstance(value, str):
        raise RocketChatContractError(f"{field} must be a string")
    text = value.strip()
    if not text or len(text) > max_length or "\n" in text or "\r" in text:
        raise RocketChatContractError(f"{field} must be bounded single-line text")
    return text


def _identity(value: object, *, field: str) -> str:
    text = _text(value, field=field, max_length=128)
    if _ID_RE.fullmatch(text) is None:
        raise RocketChatContractError(f"{field} must be a stable semantic identity")
    return text


def validate_rocket_chat_benchmark(value: object) -> str:
    """Resolve TAC identity and reject comparison or fallback Benchmarks."""

    try:
        capabilities = get_benchmark_capabilities(value)
    except ValueError as exc:
        raise RocketChatContractError("unknown benchmark for Rocket.Chat response") from exc
    if capabilities.is_comparison_only:
        raise RocketChatContractError(
            "comparison-only Benchmark cannot enter the Rocket.Chat response family"
        )
    if capabilities.canonical_name != ROCKET_CHAT_BENCHMARK:
        raise RocketChatContractError(
            "WebArena benchmark fallback is forbidden for Rocket.Chat response tasks"
        )
    capabilities.require("phase_1_generation")
    capabilities.require("warp_evaluation")
    return capabilities.canonical_name


def infer_rocket_chat_benchmark(sources: Sequence[Mapping[str, object]]) -> str:
    """Infer one explicit TAC Benchmark from metadata and fail closed."""

    try:
        benchmark = infer_benchmark_from_metadata(sources)
    except ValueError as exc:
        raise RocketChatContractError(str(exc)) from exc
    if benchmark is None:
        raise RocketChatContractError("Rocket.Chat response requires explicit Benchmark metadata")
    return validate_rocket_chat_benchmark(benchmark)


@dataclass(frozen=True)
class RocketChatDecision:
    """The finite fields exposed to the legitimate reader."""

    owner: str
    due_date: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner", _text(self.owner, field="decision.owner"))
        object.__setattr__(self, "due_date", _text(self.due_date, field="decision.due_date"))

    def as_dict(self) -> dict[str, str]:
        return {"owner": self.owner, "due_date": self.due_date}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> RocketChatDecision:
        if not isinstance(value, Mapping):
            raise RocketChatContractError("current decision must be a mapping")
        keys = set(value)
        expected = set(DECISION_FIELDS)
        missing = sorted(expected - keys)
        extra = sorted(str(item) for item in keys - expected)
        if missing:
            raise RocketChatContractError(
                f"current decision is missing fields: {', '.join(missing)}"
            )
        if extra:
            raise RocketChatContractError(f"current decision has extra fields: {', '.join(extra)}")
        return cls(owner=value["owner"], due_date=value["due_date"])  # type: ignore[arg-type]


@dataclass(frozen=True)
class RocketChatMessageFact:
    """One generated message with a logical key and thread relation."""

    logical_key: str
    room_id: str
    thread_key: str | None
    author: str
    body: str
    kind: Literal["plan", "update", "correction"]

    def __post_init__(self) -> None:
        object.__setattr__(self, "logical_key", _identity(self.logical_key, field="message key"))
        object.__setattr__(self, "room_id", _identity(self.room_id, field="message room"))
        if self.thread_key is not None:
            object.__setattr__(self, "thread_key", _identity(self.thread_key, field="thread key"))
        object.__setattr__(self, "author", _identity(self.author, field="message author"))
        object.__setattr__(self, "body", _text(self.body, field="message body", max_length=2000))
        if self.kind not in MESSAGE_KINDS:
            raise RocketChatContractError(f"unsupported message kind {self.kind!r}")


@dataclass(frozen=True)
class RocketChatCorrectionFact:
    """One explicitly confirmed correction to a finite decision field."""

    message_key: str
    field: str
    previous_value: str
    current_value: str
    confirmed_by: str
    confirmed: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "message_key", _identity(self.message_key, field="correction message")
        )
        object.__setattr__(
            self, "field", _text(self.field, field="correction field", max_length=64)
        )
        object.__setattr__(
            self, "previous_value", _text(self.previous_value, field="correction previous value")
        )
        object.__setattr__(
            self, "current_value", _text(self.current_value, field="correction current value")
        )
        object.__setattr__(
            self, "confirmed_by", _identity(self.confirmed_by, field="correction author")
        )
        if not isinstance(self.confirmed, bool):
            raise RocketChatContractError("correction confirmation must be boolean")
        if self.field not in DECISION_FIELDS:
            raise RocketChatContractError(f"unsupported correction field {self.field!r}")
        if self.previous_value == self.current_value:
            raise RocketChatContractError("correction must change its field value")


@dataclass(frozen=True)
class RocketChatConversation:
    """Generated static conversation facts and its decision rule."""

    room_id: str
    thread_key: str
    messages: tuple[RocketChatMessageFact, ...]
    corrections: tuple[RocketChatCorrectionFact, ...]
    initial_decision: RocketChatDecision
    writer_user: str
    reader_user: str
    benchmark: str = ROCKET_CHAT_BENCHMARK
    site: str = ROCKET_CHAT_SITE

    def __post_init__(self) -> None:
        object.__setattr__(self, "benchmark", validate_rocket_chat_benchmark(self.benchmark))
        site = _identity(self.site, field="Rocket.Chat Site")
        if site != ROCKET_CHAT_SITE:
            raise RocketChatContractError("Rocket.Chat conversation Site identity is unsupported")
        object.__setattr__(self, "site", site)
        object.__setattr__(self, "room_id", _identity(self.room_id, field="conversation room"))
        object.__setattr__(
            self, "thread_key", _identity(self.thread_key, field="conversation thread")
        )
        object.__setattr__(
            self, "writer_user", _identity(self.writer_user, field="writer identity")
        )
        object.__setattr__(
            self, "reader_user", _identity(self.reader_user, field="reader identity")
        )
        if self.writer_user == self.reader_user:
            raise RocketChatContractError("writer and reader identities must be distinct")
        if not isinstance(self.initial_decision, RocketChatDecision):
            raise RocketChatContractError("conversation initial decision must be typed")
        messages = tuple(self.messages)
        corrections = tuple(self.corrections)
        if not messages:
            raise RocketChatContractError("conversation requires messages")
        if any(not isinstance(message, RocketChatMessageFact) for message in messages):
            raise RocketChatContractError("conversation messages must be typed facts")
        if any(not isinstance(correction, RocketChatCorrectionFact) for correction in corrections):
            raise RocketChatContractError("conversation corrections must be typed facts")
        keys = [message.logical_key for message in messages]
        if len(keys) != len(set(keys)):
            raise RocketChatContractError("conversation message logical keys must be unique")
        if self.thread_key not in keys:
            raise RocketChatContractError("conversation thread must identify one message")
        if any(message.room_id != self.room_id for message in messages):
            raise RocketChatContractError("conversation messages must use the exact room")
        if any(
            message.thread_key is not None and message.thread_key not in keys
            for message in messages
        ):
            raise RocketChatContractError("conversation message references an unknown thread")
        kinds = {message.kind for message in messages}
        if not {"plan", "update", "correction"}.issubset(kinds):
            raise RocketChatContractError("conversation requires a plan, update, and correction")
        by_key = {message.logical_key: message for message in messages}
        if any(correction.message_key not in by_key for correction in corrections):
            raise RocketChatContractError("correction references an unknown message")
        if any(by_key[c.message_key].kind != "correction" for c in corrections):
            raise RocketChatContractError("correction must be carried by a correction message")
        if not corrections:
            raise RocketChatContractError("conversation requires an explicit correction")
        if any(c.confirmed_by != by_key[c.message_key].author for c in corrections):
            raise RocketChatContractError("correction confirmer must match message author")

        # Validate semantic chain before rendered text to preserve actionable
        # contradiction diagnostics.
        self._derive_decision()
        plan_body = by_key[self.thread_key].body
        for field, value in self.initial_decision.as_dict().items():
            if f"{field}={value}" not in plan_body:
                raise RocketChatContractError(
                    f"plan message does not carry the structured initial {field} value"
                )
        for correction in corrections:
            correction_body = by_key[correction.message_key].body
            if f"{correction.field}={correction.current_value}" not in correction_body:
                raise RocketChatContractError(
                    f"correction message does not carry the structured {correction.field} value"
                )
            if "confirmed correction" not in correction_body.casefold():
                raise RocketChatContractError(
                    "correction message must state that the correction is confirmed"
                )
        object.__setattr__(self, "messages", messages)
        object.__setattr__(self, "corrections", corrections)

    def _derive_decision(self) -> RocketChatDecision:
        values = self.initial_decision.as_dict()
        seen_fields: dict[str, str] = {}
        for correction in self.corrections:
            if not correction.confirmed:
                raise RocketChatContractError(
                    f"correction for {correction.field!r} is not explicitly confirmed"
                )
            previous = values[correction.field]
            if previous != correction.previous_value:
                if correction.field in seen_fields:
                    raise RocketChatContractError(
                        f"ambiguous or contradictory correction for {correction.field!r}"
                    )
                raise RocketChatContractError(
                    f"contradictory correction for {correction.field!r}: "
                    f"expected {previous!r}, got {correction.previous_value!r}"
                )
            values[correction.field] = correction.current_value
            seen_fields[correction.field] = correction.current_value
        return RocketChatDecision.from_mapping(values)

    @property
    def current_decision(self) -> RocketChatDecision:
        return self._derive_decision()

    @property
    def expected_decision(self) -> RocketChatDecision:
        return self.current_decision

    def message(self, logical_key: str) -> RocketChatMessageFact:
        for message in self.messages:
            if message.logical_key == logical_key:
                return message
        raise KeyError(logical_key)

    def as_dict(self) -> dict[str, object]:
        return {
            "benchmark": self.benchmark,
            "site": self.site,
            "room_id": self.room_id,
            "thread_key": self.thread_key,
            "writer_user": self.writer_user,
            "reader_user": self.reader_user,
            "initial_decision": self.initial_decision.as_dict(),
            "messages": [
                {
                    "logical_key": message.logical_key,
                    "room_id": message.room_id,
                    "thread_key": message.thread_key,
                    "author": message.author,
                    "body": message.body,
                    "kind": message.kind,
                }
                for message in self.messages
            ],
            "corrections": [
                {
                    "message_key": correction.message_key,
                    "field": correction.field,
                    "previous_value": correction.previous_value,
                    "current_value": correction.current_value,
                    "confirmed_by": correction.confirmed_by,
                    "confirmed": correction.confirmed,
                }
                for correction in self.corrections
            ],
            "expected_decision": self.expected_decision.as_dict(),
        }


@dataclass(frozen=True)
class RocketChatParticipantContext:
    """Non-secret identity metadata for one authenticated context."""

    user_id: str
    session_id: str
    auth_context_id: str
    auth_kind: Literal[
        "writer_credentials", "reader_credentials", "writer_cookies", "writer_storage"
    ]
    role: Literal["ordinary", "admin"] = "ordinary"

    def __post_init__(self) -> None:
        object.__setattr__(self, "user_id", _identity(self.user_id, field="context user"))
        object.__setattr__(self, "session_id", _identity(self.session_id, field="context session"))
        object.__setattr__(
            self, "auth_context_id", _identity(self.auth_context_id, field="context auth identity")
        )
        if self.auth_kind not in READER_AUTH_KINDS | WRITER_AUTH_KINDS | {
            "writer_cookies",
            "writer_storage",
        }:
            raise RocketChatContractError(
                f"unsupported Rocket.Chat auth context {self.auth_kind!r}"
            )
        if self.role not in {"ordinary", "admin"}:
            raise RocketChatContractError("Rocket.Chat context role is invalid")


@dataclass(frozen=True)
class RocketChatMessageIdentity:
    """Exact physical identity returned by one writer call."""

    benchmark: str
    site: str
    attempt_id: str
    logical_key: str
    room_id: str
    message_id: str
    thread_id: str | None
    author: str
    body: str
    recipient: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "benchmark", validate_rocket_chat_benchmark(self.benchmark))
        if _identity(self.site, field="message Site") != ROCKET_CHAT_SITE:
            raise RocketChatContractError("message Site identity does not match Rocket.Chat")
        object.__setattr__(self, "site", ROCKET_CHAT_SITE)
        for field_name in ("attempt_id", "logical_key", "room_id", "message_id", "author"):
            object.__setattr__(
                self,
                field_name,
                _identity(getattr(self, field_name), field=f"message {field_name}"),
            )
        if self.thread_id is not None:
            object.__setattr__(self, "thread_id", _identity(self.thread_id, field="message thread"))
        object.__setattr__(self, "body", _text(self.body, field="message body", max_length=2000))
        if self.recipient is not None:
            object.__setattr__(
                self,
                "recipient",
                _text(self.recipient, field="message recipient", max_length=128),
            )


@dataclass(frozen=True)
class RocketChatSeedReceipt:
    """Writer-owned binding of logical keys to exact physical message IDs."""

    benchmark: str
    site: str
    attempt_id: str
    writer_context: RocketChatParticipantContext
    messages: Mapping[str, RocketChatMessageIdentity]

    def __post_init__(self) -> None:
        object.__setattr__(self, "benchmark", validate_rocket_chat_benchmark(self.benchmark))
        if _identity(self.site, field="seed Site") != ROCKET_CHAT_SITE:
            raise RocketChatContractError("seed Site identity does not match Rocket.Chat")
        object.__setattr__(self, "site", ROCKET_CHAT_SITE)
        object.__setattr__(self, "attempt_id", _identity(self.attempt_id, field="seed attempt"))
        if not isinstance(self.writer_context, RocketChatParticipantContext):
            raise RocketChatContractError("seed receipt requires a typed writer context")
        if self.writer_context.auth_kind not in WRITER_AUTH_KINDS:
            raise RocketChatContractError("seed receipt writer must use writer credentials")
        if self.writer_context.role != "ordinary":
            raise RocketChatContractError("seed receipt writer must be an ordinary participant")
        if not isinstance(self.messages, Mapping) or not self.messages:
            raise RocketChatContractError("seed receipt requires exact message identities")
        normalized: dict[str, RocketChatMessageIdentity] = {}
        for raw_key, identity in self.messages.items():
            key = _identity(raw_key, field="seed logical key")
            if not isinstance(identity, RocketChatMessageIdentity):
                raise RocketChatContractError("seed receipt messages must be typed identities")
            if key != identity.logical_key:
                raise RocketChatContractError("seed logical key does not match physical identity")
            if identity.benchmark != self.benchmark or identity.site != self.site:
                raise RocketChatContractError(
                    "seed message Benchmark or Site does not match receipt"
                )
            if identity.attempt_id != self.attempt_id:
                raise RocketChatContractError("seed message belongs to a different attempt")
            normalized[key] = identity
        if len({item.message_id for item in normalized.values()}) != len(normalized):
            raise RocketChatContractError("seed receipt contains duplicate physical message IDs")
        object.__setattr__(self, "messages", MappingProxyType(normalized))


@dataclass(frozen=True)
class RocketChatObservation:
    """Exact independent-reader observation tied to one seed attempt."""

    benchmark: str
    site: str
    attempt_id: str
    room_id: str
    thread_id: str
    reader_context: RocketChatParticipantContext
    messages: Mapping[str, RocketChatMessageIdentity]
    current_decision: RocketChatDecision

    def __post_init__(self) -> None:
        object.__setattr__(self, "benchmark", validate_rocket_chat_benchmark(self.benchmark))
        if _identity(self.site, field="observation Site") != ROCKET_CHAT_SITE:
            raise RocketChatContractError("observation Site identity does not match Rocket.Chat")
        object.__setattr__(self, "site", ROCKET_CHAT_SITE)
        object.__setattr__(
            self, "attempt_id", _identity(self.attempt_id, field="observation attempt")
        )
        object.__setattr__(self, "room_id", _identity(self.room_id, field="observation room"))
        object.__setattr__(self, "thread_id", _identity(self.thread_id, field="observation thread"))
        if not isinstance(self.reader_context, RocketChatParticipantContext):
            raise RocketChatContractError("observation requires a typed reader context")
        if self.reader_context.auth_kind not in READER_AUTH_KINDS:
            raise RocketChatContractError("observation requires independent reader credentials")
        if self.reader_context.role != "ordinary":
            raise RocketChatContractError("observation reader must be an ordinary participant")
        if not isinstance(self.current_decision, RocketChatDecision):
            raise RocketChatContractError("observation decision must be typed")
        if not isinstance(self.messages, Mapping) or not self.messages:
            raise RocketChatContractError("observation requires exact message identities")
        normalized: dict[str, RocketChatMessageIdentity] = {}
        for key, identity in self.messages.items():
            if not isinstance(key, str) or not isinstance(identity, RocketChatMessageIdentity):
                raise RocketChatContractError("observation messages must be typed identities")
            normalized[key] = identity
        object.__setattr__(self, "messages", MappingProxyType(normalized))

    @property
    def message_ids(self) -> dict[str, str]:
        return {key: item.message_id for key, item in self.messages.items()}


@dataclass(frozen=True)
class RocketChatObservationFailure:
    """Actionable fail-closed reader diagnostic."""

    reason: str
    detail: str
    ok: bool = False


@runtime_checkable
class RocketChatConversationWriter(Protocol):
    """Feature-owned writer boundary; implementations live outside production."""

    def seed_conversation(self, conversation: RocketChatConversation) -> RocketChatSeedReceipt: ...


@runtime_checkable
class RocketChatAuthenticatedReader(Protocol):
    """Feature-owned independent-reader boundary; implementations live outside production."""

    def observe(
        self,
        conversation: RocketChatConversation,
        receipt: RocketChatSeedReceipt,
    ) -> RocketChatObservation | RocketChatObservationFailure: ...


__all__ = [
    "DECISION_FIELDS",
    "MESSAGE_KINDS",
    "ROCKET_CHAT_BENCHMARK",
    "ROCKET_CHAT_EVALUATOR_AUTHORITY",
    "ROCKET_CHAT_EVALUATOR_NAME",
    "ROCKET_CHAT_RESOURCE_KIND",
    "ROCKET_CHAT_SITE",
    "ROCKET_CHAT_TASK_KIND",
    "RocketChatAuthenticatedReader",
    "RocketChatContractError",
    "RocketChatConversation",
    "RocketChatConversationWriter",
    "RocketChatCorrectionFact",
    "RocketChatDecision",
    "RocketChatMessageFact",
    "RocketChatMessageIdentity",
    "RocketChatObservation",
    "RocketChatObservationFailure",
    "RocketChatParticipantContext",
    "RocketChatSeedReceipt",
    "infer_rocket_chat_benchmark",
    "validate_rocket_chat_benchmark",
]
