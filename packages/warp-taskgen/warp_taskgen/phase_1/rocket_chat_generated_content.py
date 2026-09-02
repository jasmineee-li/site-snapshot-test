"""Strict semantic slots for generated Rocket.Chat conversations.

The Phase 1 model is allowed to supply *facts* and natural language only.  It
does not get to choose the channel, thread graph, actor identities, evaluator,
or seed operation.  This module keeps that boundary small and typed so the
host-side compiler can project the slots into the existing
``RocketChatConversation`` contract.

The wire shape deliberately uses a short, enumerable ``messages`` array.  The
array is not a message model: ``slot`` is a semantic slot name and the host
assigns every logical key, kind, author, room and thread relation later.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from warp_taskgen.phase_1.rocket_chat_contracts import (
    RocketChatContractError,
    RocketChatDecision,
)

ROCKET_CHAT_GENERATED_CONTENT_KEY = "generated_rocket_chat"
ROCKET_CHAT_MESSAGE_SLOTS = ("plan", "update", "correction")

_GENERATED_KEYS = frozenset({"initial_decision", "corrected_decision", "messages"})
_DECISION_KEYS = frozenset({"owner", "due_date"})
_MESSAGE_KEYS = frozenset({"slot", "text"})
_STRUCTURAL_KEYS = frozenset(
    {
        "action_contract",
        "author",
        "benchmark",
        "body",
        "correction",
        "data_seed",
        "evaluator",
        "evaluator_authority",
        "kind",
        "logical_key",
        "message_id",
        "notification",
        "reader_user",
        "recipient",
        "reward_function",
        "room",
        "room_id",
        "seed",
        "site",
        "start_urls",
        "thread",
        "thread_id",
        "thread_key",
        "writer_user",
    }
)
_PLACEHOLDER_RE = re.compile(r"\{[^{}]+\}")


def _require_exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    missing = expected - set(value)
    extra = set(value) - expected
    if missing:
        raise ValueError(
            f"{label} is missing semantic slot(s): "
            + ", ".join(sorted(str(key) for key in missing))
        )
    if extra:
        structural = extra.intersection(_STRUCTURAL_KEYS)
        if structural:
            raise ValueError(
                f"{label} contains host-owned structural slot(s): "
                + ", ".join(sorted(str(key) for key in structural))
            )
        raise ValueError(
            f"{label} contains unsupported slot(s): " + ", ".join(sorted(str(key) for key in extra))
        )


def _semantic_text(value: object, *, field: str, minimum_words: int = 3) -> str:
    if not isinstance(value, str):
        raise ValueError(f"generated Rocket.Chat {field} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"generated Rocket.Chat {field} must be non-empty")
    if len(text) > 800:
        raise ValueError(f"generated Rocket.Chat {field} exceeds the 800-character limit")
    if "\n" in text or "\r" in text:
        raise ValueError(f"generated Rocket.Chat {field} must be single-line text")
    if _PLACEHOLDER_RE.search(text):
        raise ValueError(f"generated Rocket.Chat {field} must not contain template placeholders")
    # A one-word label is not an instance.  Keep this deliberately modest so
    # short but meaningful generated notes remain valid.
    if len(text.split()) < minimum_words:
        raise ValueError(f"generated Rocket.Chat {field} must contain substantive natural language")
    return text


def _decision(value: object, *, field: str) -> RocketChatDecision:
    if not isinstance(value, Mapping):
        raise ValueError(f"generated Rocket.Chat {field} must be an object")
    try:
        _require_exact_keys(value, _DECISION_KEYS, f"generated Rocket.Chat {field}")
        return RocketChatDecision(owner=value["owner"], due_date=value["due_date"])  # type: ignore[arg-type]
    except (KeyError, TypeError, RocketChatContractError) as exc:
        raise ValueError(f"generated Rocket.Chat {field} is invalid: {exc}") from exc


@dataclass(frozen=True)
class RocketChatSemanticMessage:
    """One model-authored prose slot, without runtime message identity."""

    slot: Literal["plan", "update", "correction"]
    text: str

    def __post_init__(self) -> None:
        if self.slot not in ROCKET_CHAT_MESSAGE_SLOTS:
            raise ValueError(f"unsupported Rocket.Chat semantic message slot {self.slot!r}")
        object.__setattr__(
            self,
            "text",
            _semantic_text(self.text, field=f"messages[{self.slot!r}].text"),
        )

    def as_mapping(self) -> dict[str, str]:
        return {"slot": self.slot, "text": self.text}


@dataclass(frozen=True)
class RocketChatGeneratedContent:
    """Validated semantic facts accepted from one Phase 1 model output."""

    initial_decision: RocketChatDecision
    corrected_decision: RocketChatDecision
    messages: tuple[RocketChatSemanticMessage, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.initial_decision, RocketChatDecision):
            raise ValueError("generated Rocket.Chat initial_decision must be typed")
        if not isinstance(self.corrected_decision, RocketChatDecision):
            raise ValueError("generated Rocket.Chat corrected_decision must be typed")
        if self.initial_decision.owner == self.corrected_decision.owner:
            raise ValueError("generated Rocket.Chat correction must change owner")
        if self.initial_decision.due_date == self.corrected_decision.due_date:
            raise ValueError("generated Rocket.Chat correction must change due_date")
        messages = tuple(self.messages)
        if len(messages) != len(ROCKET_CHAT_MESSAGE_SLOTS):
            raise ValueError(
                "generated Rocket.Chat content requires exactly three semantic message slots"
            )
        if any(not isinstance(message, RocketChatSemanticMessage) for message in messages):
            raise ValueError("generated Rocket.Chat messages must be typed semantic slots")
        slots = [message.slot for message in messages]
        if len(slots) != len(set(slots)):
            raise ValueError("generated Rocket.Chat semantic message slots must be unique")
        if set(slots) != set(ROCKET_CHAT_MESSAGE_SLOTS):
            raise ValueError(
                "generated Rocket.Chat semantic message slots must include plan, update, and correction"
            )
        texts = [message.text.casefold() for message in messages]
        if len(texts) != len(set(texts)):
            raise ValueError("generated Rocket.Chat semantic message text must be distinct")
        # Keep a deterministic canonical ordering regardless of model ordering.
        by_slot = {message.slot: message for message in messages}
        object.__setattr__(
            self,
            "messages",
            tuple(by_slot[slot] for slot in ROCKET_CHAT_MESSAGE_SLOTS),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RocketChatGeneratedContent:
        """Parse and fail closed on one model-produced semantic payload."""

        if not isinstance(value, Mapping):
            raise ValueError("generated Rocket.Chat content must be an object")
        _require_exact_keys(value, _GENERATED_KEYS, "generated Rocket.Chat content")

        initial = _decision(value.get("initial_decision"), field="initial_decision")
        corrected = _decision(value.get("corrected_decision"), field="corrected_decision")

        raw_messages = value.get("messages")
        if not isinstance(raw_messages, list):
            raise ValueError("generated Rocket.Chat content messages must be an array")
        if len(raw_messages) != len(ROCKET_CHAT_MESSAGE_SLOTS):
            raise ValueError(
                "generated Rocket.Chat content requires exactly three semantic message slots"
            )
        messages: list[RocketChatSemanticMessage] = []
        for index, raw_message in enumerate(raw_messages):
            if not isinstance(raw_message, Mapping):
                raise ValueError(f"generated Rocket.Chat messages[{index}] must be an object")
            _require_exact_keys(
                raw_message, _MESSAGE_KEYS, f"generated Rocket.Chat messages[{index}]"
            )
            raw_slot = raw_message.get("slot")
            if not isinstance(raw_slot, str) or raw_slot not in ROCKET_CHAT_MESSAGE_SLOTS:
                raise ValueError(
                    f"generated Rocket.Chat messages[{index}].slot must be one of "
                    f"{ROCKET_CHAT_MESSAGE_SLOTS!r}"
                )
            messages.append(
                RocketChatSemanticMessage(slot=raw_slot, text=raw_message.get("text"))  # type: ignore[arg-type]
            )
        try:
            return cls(
                initial_decision=initial, corrected_decision=corrected, messages=tuple(messages)
            )
        except RocketChatContractError as exc:
            raise ValueError(f"generated Rocket.Chat content is invalid: {exc}") from exc

    @classmethod
    def from_task(cls, task: Mapping[str, Any]) -> RocketChatGeneratedContent:
        """Read exactly one supported generated-content field from a task."""

        if ROCKET_CHAT_GENERATED_CONTENT_KEY not in task:
            raise ValueError(
                "Rocket.Chat task requires one generated semantic content field "
                f"({ROCKET_CHAT_GENERATED_CONTENT_KEY})"
            )
        return cls.from_mapping(task[ROCKET_CHAT_GENERATED_CONTENT_KEY])  # type: ignore[arg-type]

    def as_mapping(self) -> dict[str, Any]:
        return {
            "initial_decision": self.initial_decision.as_dict(),
            "corrected_decision": self.corrected_decision.as_dict(),
            "messages": [message.as_mapping() for message in self.messages],
        }

    @property
    def by_slot(self) -> Mapping[str, RocketChatSemanticMessage]:
        return {message.slot: message for message in self.messages}


__all__ = [
    "ROCKET_CHAT_GENERATED_CONTENT_KEY",
    "ROCKET_CHAT_MESSAGE_SLOTS",
    "RocketChatGeneratedContent",
    "RocketChatSemanticMessage",
]
