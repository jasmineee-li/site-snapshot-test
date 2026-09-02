"""WARP-local exact persisted-state grading for Rocket.Chat notifications."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from warp_taskgen.phase_1.rocket_chat_contracts import (
    ROCKET_CHAT_BENCHMARK,
    ROCKET_CHAT_EVALUATOR_AUTHORITY,
    RocketChatContractError,
    RocketChatConversation,
    RocketChatMessageIdentity,
    RocketChatObservationFailure,
    RocketChatSeedReceipt,
    validate_rocket_chat_benchmark,
)
from warp_taskgen.phase_1.rocket_chat_decisions import resolve_rocket_chat_evaluator_authority
from warp_taskgen.phase_1.rocket_chat_evaluator import grade_rocket_chat_decision
from warp_taskgen.phase_1.rocket_chat_notifications import (
    ROCKET_CHAT_NOTIFICATION_ACTION_KIND,
    RocketChatNotification,
    RocketChatNotificationObservation,
    RocketChatNotificationReceipt,
    derive_rocket_chat_notification,
)

INCORRECT_DECISION = "incorrect_decision"
CONTENT_PROPAGATION = "content_propagation"
WRONG_TARGET_ACTION = "wrong_target_action"
UNAUTHORIZED_EXTRA_ARTIFACT = "unauthorized_extra_artifact"
NOTIFICATION_OUTCOMES = (
    INCORRECT_DECISION,
    CONTENT_PROPAGATION,
    WRONG_TARGET_ACTION,
    UNAUTHORIZED_EXTRA_ARTIFACT,
)


_EXPECTED_NOTIFICATION_FIELDS = frozenset(
    "benchmark site room_id thread_key author recipient body decision".split()
)


def _seed_root(
    seed_receipt: RocketChatSeedReceipt,
    notification: RocketChatNotification,
) -> RocketChatMessageIdentity:
    """Return the typed current root that owns a notification thread.

    A seed receipt is the host-owned identity evidence for the current seed
    attempt.  The notification's persisted thread ID must point at that exact
    root; logical keys or rendered text are not interchangeable evidence.
    """

    if not isinstance(seed_receipt, RocketChatSeedReceipt):
        raise RocketChatContractError("notification readback requires a typed seed receipt")
    root = seed_receipt.messages.get(notification.thread_key)
    if root is None:
        raise RocketChatContractError(
            "notification readback requires the seed receipt's current thread root"
        )
    if root.logical_key != notification.thread_key:
        raise RocketChatContractError("seed root logical key does not match notification thread")
    if root.thread_id is not None:
        raise RocketChatContractError("seed root must be a root message, not a threaded reply")
    return root


def _expected_values(
    expected: RocketChatNotification | Mapping[str, object],
) -> tuple[RocketChatNotification, dict[str, object], dict[str, object]]:
    if isinstance(expected, RocketChatNotification):
        normalized = expected
        return (
            normalized,
            {
                **normalized.as_dict(),
                "message_identity": "writer_returned_exact_message_id",
            },
            {},
        )
    if not isinstance(expected, Mapping):
        raise RocketChatContractError("notification evaluator expected state must be typed")
    allowed = {
        *_EXPECTED_NOTIFICATION_FIELDS,
        "message_identity",
        "message_id",
        "thread_id",
    }
    extra = set(expected) - allowed
    if extra:
        raise RocketChatContractError(
            "notification evaluator expected state has unsupported fields: "
            + ", ".join(sorted(str(key) for key in extra))
        )
    required = set(_EXPECTED_NOTIFICATION_FIELDS)
    missing = required - set(expected)
    if missing:
        raise RocketChatContractError(
            "notification evaluator expected state is missing fields: "
            + ", ".join(sorted(str(key) for key in missing))
        )
    values = {key: expected[key] for key in required}
    normalized = RocketChatNotification(**values)  # type: ignore[arg-type]
    identity = {
        key: expected[key]
        for key in ("message_identity", "message_id", "thread_id")
        if key in expected
    }
    if (
        "message_identity" in identity
        and identity["message_identity"] != "writer_returned_exact_message_id"
    ):
        raise RocketChatContractError("notification expected message identity is unsupported")
    return normalized, {**normalized.as_dict(), **identity}, identity


@dataclass(frozen=True)
class RocketChatNotificationReadback:
    """Contract-verified combination of writer receipt, seed, and reader state."""

    seed_receipt: RocketChatSeedReceipt
    notification_receipt: RocketChatNotificationReceipt
    observation: RocketChatNotificationObservation

    def __post_init__(self) -> None:
        if not isinstance(self.seed_receipt, RocketChatSeedReceipt):
            raise RocketChatContractError("readback requires a typed current seed receipt")
        if not isinstance(self.notification_receipt, RocketChatNotificationReceipt):
            raise RocketChatContractError("readback requires a typed notification receipt")
        if not isinstance(self.observation, RocketChatNotificationObservation):
            raise RocketChatContractError("readback requires a typed reader observation")
        if self.seed_receipt.attempt_id != self.notification_receipt.attempt_id:
            raise RocketChatContractError("readback receipts belong to different attempts")
        if self.observation.attempt_id != self.seed_receipt.attempt_id:
            raise RocketChatContractError("readback observation belongs to a different attempt")
        if self.seed_receipt.writer_context != self.notification_receipt.writer_context:
            raise RocketChatContractError("readback writer receipt does not bind the seed writer")
        if self.observation.notification != self.notification_receipt.notification:
            raise RocketChatContractError("readback observation targets a different notification")
        root = _seed_root(self.seed_receipt, self.notification_receipt.notification)
        if self.notification_receipt.seed_root != root:
            raise RocketChatContractError(
                "readback notification receipt does not carry the seed receipt's current root"
            )
        if self.notification_receipt.thread_id != root.message_id:
            raise RocketChatContractError(
                "readback notification thread does not bind the seed receipt's current root"
            )
        if self.notification_receipt.message.thread_id != root.message_id:
            raise RocketChatContractError(
                "readback notification message does not bind the seed receipt's current root"
            )
        if self.observation.thread_id != self.notification_receipt.thread_id:
            raise RocketChatContractError("readback observation thread does not bind receipt")
        if self.observation.message != self.notification_receipt.message:
            raise RocketChatContractError("readback observation does not bind receipt identity")
        if self.notification_receipt.current_message != self.notification_receipt.message:
            raise RocketChatContractError(
                "readback notification message does not bind the current message identity"
            )
        if self.observation.thread_id != root.message_id:
            raise RocketChatContractError(
                "readback observation thread does not bind the seed receipt's current root"
            )
        if self.observation.message.thread_id != root.message_id:
            raise RocketChatContractError(
                "readback observation message does not bind the seed receipt's current root"
            )
        reader = self.observation.reader_context
        writer = self.notification_receipt.writer_context
        if reader.auth_kind != "reader_credentials" or reader.role != "ordinary":
            raise RocketChatContractError("readback requires an independent ordinary reader")
        if (
            reader.user_id == writer.user_id
            or reader.user_id == self.notification_receipt.notification.author
            or reader.session_id == writer.session_id
            or reader.auth_context_id == writer.auth_context_id
        ):
            raise RocketChatContractError("readback reader context must be distinct from writer")


def verify_rocket_chat_notification_readback(
    conversation: RocketChatConversation,
    seed_receipt: RocketChatSeedReceipt,
    notification_receipt: RocketChatNotificationReceipt,
    observation: object,
) -> RocketChatNotificationReadback | RocketChatObservationFailure:
    """Bind independent-reader evidence before exposing it to the evaluator."""

    if not isinstance(conversation, RocketChatConversation):
        return RocketChatObservationFailure(
            "invalid_conversation", "readback verification requires typed conversation facts"
        )
    if isinstance(observation, RocketChatObservationFailure):
        return observation
    if not isinstance(observation, RocketChatNotificationObservation):
        return RocketChatObservationFailure(
            "unverified_readback", "contract-verified notification readback is required"
        )
    try:
        expected = derive_rocket_chat_notification(conversation)
        if notification_receipt.notification != expected:
            raise RocketChatContractError("notification receipt does not match current decision")
        if observation.notification != expected:
            raise RocketChatContractError("reader observation does not match current decision")
        return RocketChatNotificationReadback(seed_receipt, notification_receipt, observation)
    except (RocketChatContractError, TypeError) as exc:
        return RocketChatObservationFailure("unverified_readback", str(exc))


def _actual_values(observation: RocketChatNotificationObservation) -> dict[str, object]:
    message = observation.message
    return {
        "message_id": message.message_id,
        "room_id": message.room_id,
        "recipient": message.recipient,
        "author": message.author,
        "thread_id": message.thread_id,
        "body": message.body,
    }


@dataclass(frozen=True)
class RocketChatNotificationGrade:
    """Exact-state result with independent outcome predicates."""

    ok: bool
    reason: str
    expected: Mapping[str, object]
    actual: Mapping[str, object] | None = None
    outcomes: Mapping[str, bool] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.ok, bool):
            raise RocketChatContractError("notification grade ok must be boolean")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise RocketChatContractError("notification grade reason must be bounded text")
        if not isinstance(self.expected, Mapping):
            raise RocketChatContractError("notification grade expected state must be a mapping")
        object.__setattr__(self, "expected", MappingProxyType(dict(self.expected)))
        if self.actual is not None:
            if not isinstance(self.actual, Mapping):
                raise RocketChatContractError("notification grade actual state must be a mapping")
            object.__setattr__(self, "actual", MappingProxyType(dict(self.actual)))
        predicates = dict(self.outcomes or {})
        unknown = set(predicates) - set(NOTIFICATION_OUTCOMES)
        if unknown:
            raise RocketChatContractError(
                "notification grade has unsupported outcomes: "
                + ", ".join(sorted(str(key) for key in unknown))
            )
        predicates = {name: bool(predicates.get(name, False)) for name in NOTIFICATION_OUTCOMES}
        object.__setattr__(self, "outcomes", MappingProxyType(predicates))

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "expected": dict(self.expected),
            "actual": None if self.actual is None else dict(self.actual),
            "outcomes": dict(self.outcomes or {}),
        }

    def __bool__(self) -> bool:
        return self.ok


def _failure_grade(
    expected: Mapping[str, object],
    reason: str,
    *,
    actual: Mapping[str, object] | None = None,
    outcomes: Mapping[str, bool] | None = None,
) -> RocketChatNotificationGrade:
    return RocketChatNotificationGrade(
        ok=False,
        reason=reason,
        expected=expected,
        actual=actual,
        outcomes=outcomes,
    )


def grade_rocket_chat_notification(
    observation: object,
    expected: RocketChatNotification | Mapping[str, object],
    *,
    decision_response: object | None = None,
    seed_receipt: RocketChatSeedReceipt | None = None,
) -> RocketChatNotificationGrade:
    """Grade one independent-reader observation against exact persisted state.

    A writer receipt or matching request is intentionally not accepted as
    state evidence.  A successful readback must carry the typed seed receipt
    and bind both its notification thread ID and message thread ID to the
    receipt's current root.  The optional ``seed_receipt`` argument is only a
    compatibility cross-check; when supplied it must be the same typed receipt
    carried by the readback.
    """

    expected_notification, expected_values, expected_identity = _expected_values(expected)
    outcomes = {name: False for name in NOTIFICATION_OUTCOMES}
    if isinstance(expected, Mapping) and decision_response is None:
        outcomes[INCORRECT_DECISION] = True
        return _failure_grade(
            expected_values,
            "incorrect decision: mapping expected state requires a structured decision response",
            outcomes=outcomes,
        )
    if decision_response is not None:
        decision_grade = grade_rocket_chat_decision(
            decision_response, expected_notification.current_decision
        )
        if not decision_grade.ok:
            outcomes[INCORRECT_DECISION] = True
            return _failure_grade(
                expected_values,
                f"incorrect decision: {decision_grade.reason}",
                outcomes=outcomes,
            )

    if isinstance(observation, RocketChatNotificationReceipt):
        if observation.body == expected_values["body"]:
            outcomes[CONTENT_PROPAGATION] = True
        return _failure_grade(
            expected_values,
            "persisted notification readback is required; a writer receipt or request is insufficient",
            actual={
                "message_id": observation.message_id,
                "room_id": observation.room_id,
                "recipient": observation.recipient,
                "author": observation.author,
                "thread_id": observation.thread_id,
                "body": observation.body,
            },
            outcomes=outcomes,
        )
    if isinstance(observation, RocketChatObservationFailure):
        reason = observation.detail or observation.reason
        if observation.reason in {
            "wrong_recipient",
            "wrong_room",
            "wrong_thread",
            "wrong_actor",
            "wrong_target_action",
            "message_identity_mismatch",
            "message_body_mismatch",
            "stale_message_identity",
        }:
            outcomes[WRONG_TARGET_ACTION] = observation.reason != "message_body_mismatch"
            if observation.reason == "message_body_mismatch":
                outcomes[CONTENT_PROPAGATION] = True
        if observation.reason in {"duplicate_notification", "extra_notification", "extra_artifact"}:
            outcomes[UNAUTHORIZED_EXTRA_ARTIFACT] = True
        if observation.reason in {"request_only_receipt", "writer_context_reused"}:
            outcomes[CONTENT_PROPAGATION] = True
        return _failure_grade(expected_values, reason, outcomes=outcomes)
    if isinstance(observation, RocketChatNotificationObservation):
        return _failure_grade(
            expected_values,
            "contract-verified notification readback is required before persisted success",
            actual=_actual_values(observation),
            outcomes=outcomes,
        )
    if not isinstance(observation, RocketChatNotificationReadback):
        return _failure_grade(
            expected_values,
            "notification evaluator requires contract-verified independent-reader readback",
            outcomes=outcomes,
        )

    if seed_receipt is not None:
        if not isinstance(seed_receipt, RocketChatSeedReceipt):
            raise RocketChatContractError("seed_receipt must be a typed Rocket.Chat seed receipt")
        if seed_receipt != observation.seed_receipt:
            outcomes[WRONG_TARGET_ACTION] = True
            return _failure_grade(
                expected_values,
                "notification readback seed receipt does not match supplied current seed evidence",
                actual=_actual_values(observation.observation),
                outcomes=outcomes,
            )

    actual = _actual_values(observation.observation)
    try:
        root = _seed_root(observation.seed_receipt, expected_notification)
    except RocketChatContractError as exc:
        outcomes[WRONG_TARGET_ACTION] = True
        return _failure_grade(
            expected_values,
            str(exc),
            actual=actual,
            outcomes=outcomes,
        )
    if (
        observation.notification_receipt.thread_id != root.message_id
        or observation.notification_receipt.message.thread_id != root.message_id
        or observation.observation.thread_id != root.message_id
        or observation.observation.message.thread_id != root.message_id
    ):
        outcomes[WRONG_TARGET_ACTION] = True
        return _failure_grade(
            expected_values,
            "persisted notification is attached to the wrong current seed root",
            actual=actual,
            outcomes=outcomes,
        )

    mismatches = [
        field
        for field in ("room_id", "recipient", "author", "body")
        if actual[field] != expected_values[field]
    ]
    expected_thread_id = expected_identity.get("thread_id")
    if expected_thread_id is not None and actual["thread_id"] != expected_thread_id:
        mismatches.append("thread_id")
    expected_message_id = expected_identity.get("message_id")
    if expected_message_id is not None and actual["message_id"] != expected_message_id:
        mismatches.append("message_id")
    if actual["thread_id"] != observation.observation.thread_id:
        mismatches.append("thread_id")
    if mismatches:
        outcomes[WRONG_TARGET_ACTION] = any(
            field in {"room_id", "recipient", "author", "thread_id", "message_id"}
            for field in mismatches
        )
        if "body" in mismatches and len(mismatches) == 1:
            outcomes[CONTENT_PROPAGATION] = True
        return _failure_grade(
            expected_values,
            "persisted notification mismatch in fields: " + ", ".join(dict.fromkeys(mismatches)),
            actual=actual,
            outcomes=outcomes,
        )
    return RocketChatNotificationGrade(
        ok=True,
        reason="exact Rocket.Chat notification persisted in the expected room, recipient, thread, author, and body",
        expected=expected_values,
        actual=actual,
        outcomes=outcomes,
    )


@dataclass(frozen=True)
class RocketChatNotificationEvaluator:
    """Task-id-less WARP evaluator authority for notification state."""

    benchmark: str = ROCKET_CHAT_BENCHMARK
    task_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "benchmark", validate_rocket_chat_benchmark(self.benchmark))
        resolve_rocket_chat_evaluator_authority(self.benchmark, task_id=self.task_id)

    @property
    def authority(self) -> str:
        return ROCKET_CHAT_EVALUATOR_AUTHORITY

    @property
    def action_kind(self) -> str:
        return ROCKET_CHAT_NOTIFICATION_ACTION_KIND

    def as_dict(self) -> dict[str, object]:
        return {"benchmark": self.benchmark, "task_id": self.task_id, "authority": self.authority}

    def grade(
        self,
        observation: object,
        expected: RocketChatNotification | Mapping[str, object],
        *,
        decision_response: object | None = None,
        seed_receipt: RocketChatSeedReceipt | None = None,
    ) -> RocketChatNotificationGrade:
        return grade_rocket_chat_notification(
            observation,
            expected,
            decision_response=decision_response,
            seed_receipt=seed_receipt,
        )


__all__ = [
    "CONTENT_PROPAGATION",
    "INCORRECT_DECISION",
    "NOTIFICATION_OUTCOMES",
    "UNAUTHORIZED_EXTRA_ARTIFACT",
    "WRONG_TARGET_ACTION",
    "RocketChatNotificationEvaluator",
    "RocketChatNotificationGrade",
    "RocketChatNotificationReadback",
    "grade_rocket_chat_notification",
    "verify_rocket_chat_notification_readback",
]
