"""Independent Rocket.Chat notification final-state readback.

This feature-local module reconstructs the per-call seed and discovers
persisted notification state through a fresh reader. It reuses the notification
runtime's canonical wire renderer and parser without creating a reverse import.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from warp_taskgen.phase_1.rocket_chat_contracts import (
    ROCKET_CHAT_SEED_METHOD,
    ROCKET_CHAT_SITE,
    RocketChatContractError,
    RocketChatConversation,
    RocketChatDecision,
    RocketChatMessageIdentity,
    RocketChatObservation,
    RocketChatObservationFailure,
    RocketChatParticipantContext,
    RocketChatSeedReceipt,
)
from warp_taskgen.phase_1.rocket_chat_decisions import _validate_conversation
from warp_taskgen.phase_1.rocket_chat_notification_final_state import (
    RocketChatNotificationFinalStateReadback,
)
from warp_taskgen.phase_1.rocket_chat_notifications import (
    ROCKET_CHAT_NOTIFICATION_LOGICAL_KEY,
    RocketChatNotification,
    RocketChatNotificationObservation,
    derive_rocket_chat_notification,
    validate_rocket_chat_notification_task,
)
from warp_taskgen.phase_1.rocket_chat_task_envelope import project_rocket_chat_static_contract
from warp_taskgen.sites.rocketchat_notification_runtime import (
    _notification_identity,
    render_rocket_chat_notification_message,
)
from warp_taskgen.sites.rocketchat_runtime import (
    RocketChatHttpReader,
    RocketChatTransport,
    RocketChatTransportError,
    _context,
    _credentials,
)
from warp_taskgen.sites.rocketchat_transport import RequestsRocketChatTransport, _origin

_SEED_WRITE_TOKEN_FIXED_KEYS = frozenset(
    {
        "attempt_id",
        "room_id",
        "room_name",
        "thread_id",
        "writer_user",
        "reader_user_id",
        "reader_auth_context_id",
    }
)


def _notification_task_facts(
    task: Mapping[str, object],
) -> tuple[RocketChatConversation, RocketChatNotification] | RocketChatObservationFailure:
    """Project one notification task into its typed generated facts."""

    try:
        static = project_rocket_chat_static_contract(task)
        validate_rocket_chat_notification_task(static)
        raw_conversation = static.get("conversation")
        raw_decision = static.get("expected_decision")
        if not isinstance(raw_conversation, Mapping) or not isinstance(raw_decision, Mapping):
            raise RocketChatContractError(
                "Rocket.Chat notification contract requires conversation and expected decision"
            )
        conversation = _validate_conversation(
            raw_conversation,
            RocketChatDecision.from_mapping(raw_decision),
        )
        notification = derive_rocket_chat_notification(conversation)
        return conversation, notification
    except (RocketChatContractError, TypeError, ValueError, KeyError) as exc:
        return RocketChatObservationFailure("task_contract_invalid", str(exc))


def _notification_seed_tokens(
    seed_metadata: Mapping[str, object], conversation: RocketChatConversation
) -> Mapping[str, str] | RocketChatObservationFailure:
    """Select exactly one Rocket.Chat editor call and its complete token set."""

    if not isinstance(seed_metadata, Mapping):
        return RocketChatObservationFailure(
            "seed_metadata_invalid", "Rocket.Chat seed metadata must be a mapping"
        )
    raw_records = seed_metadata.get("editor_call_results")
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
        return RocketChatObservationFailure(
            "seed_metadata_invalid",
            "Rocket.Chat final-state readback requires editor_call_results",
        )
    if len(raw_records) != 1:
        return RocketChatObservationFailure(
            "seed_metadata_invalid",
            "Rocket.Chat final-state readback requires exactly one editor call result",
        )
    record = raw_records[0]
    if not isinstance(record, Mapping):
        return RocketChatObservationFailure(
            "seed_metadata_invalid", "Rocket.Chat editor call result must be a mapping"
        )
    if (
        record.get("call_index") != 0
        or record.get("site") != ROCKET_CHAT_SITE
        or record.get("method") != ROCKET_CHAT_SEED_METHOD
        or record.get("editor_method") != f"{ROCKET_CHAT_SITE}.{ROCKET_CHAT_SEED_METHOD}"
        or record.get("benchmark") != conversation.benchmark
    ):
        return RocketChatObservationFailure(
            "seed_metadata_invalid",
            "Rocket.Chat editor call result does not bind the current seed method",
        )
    raw_tokens = record.get("write_tokens")
    if not isinstance(raw_tokens, Mapping):
        return RocketChatObservationFailure(
            "seed_metadata_invalid", "Rocket.Chat seed editor result is missing write_tokens"
        )
    expected_keys = _SEED_WRITE_TOKEN_FIXED_KEYS | frozenset(
        key
        for fact in conversation.messages
        for key in (
            f"{fact.logical_key}_message_id",
            f"{fact.logical_key}_body_sha256",
        )
    )
    if set(raw_tokens) != expected_keys:
        missing = sorted(str(key) for key in expected_keys - set(raw_tokens))
        extra = sorted(str(key) for key in set(raw_tokens) - expected_keys)
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if extra:
            detail.append("extra " + ", ".join(extra))
        return RocketChatObservationFailure(
            "seed_metadata_invalid",
            "Rocket.Chat seed write_tokens are not exact: " + "; ".join(detail),
        )
    tokens: dict[str, str] = {}
    for key in expected_keys:
        value = raw_tokens.get(key)
        if not isinstance(value, str) or not value.strip():
            return RocketChatObservationFailure(
                "seed_metadata_invalid",
                f"Rocket.Chat seed write token {key!r} must be non-empty text",
            )
        tokens[key] = value.strip()
    return tokens


def _reconstruct_notification_seed_receipt(
    conversation: RocketChatConversation,
    tokens: Mapping[str, str],
    writer_context: RocketChatParticipantContext,
) -> RocketChatSeedReceipt | RocketChatObservationFailure:
    """Rebuild only the current seed receipt from editor-owned identity tokens."""

    try:
        if not isinstance(writer_context, RocketChatParticipantContext):
            raise RocketChatContractError("Rocket.Chat writer context must be typed")
        if tokens.get("room_name") != conversation.room_id:
            raise RocketChatContractError("Rocket.Chat seed room token does not match the task")
        if tokens.get("writer_user") != conversation.writer_user:
            raise RocketChatContractError(
                "Rocket.Chat seed writer token does not match the task participant"
            )
        if writer_context.username != conversation.writer_user:
            raise RocketChatContractError(
                "Rocket.Chat fresh writer login does not match the task participant"
            )
        identities: dict[str, RocketChatMessageIdentity] = {}
        room_id = tokens["room_id"]
        for fact in conversation.messages:
            message_id = tokens[f"{fact.logical_key}_message_id"]
            body_digest = tokens[f"{fact.logical_key}_body_sha256"]
            expected_digest = hashlib.sha256(fact.body.encode()).hexdigest()
            if body_digest != expected_digest:
                raise RocketChatContractError(
                    f"Rocket.Chat seed body digest does not match {fact.logical_key!r}"
                )
            parent = identities.get(fact.thread_key) if fact.thread_key else None
            if fact.thread_key and parent is None:
                raise RocketChatContractError(
                    f"Rocket.Chat seed thread parent is missing for {fact.logical_key!r}"
                )
            identities[fact.logical_key] = RocketChatMessageIdentity(
                benchmark=conversation.benchmark,
                site=conversation.site,
                attempt_id=tokens["attempt_id"],
                logical_key=fact.logical_key,
                room_id=room_id,
                message_id=message_id,
                thread_id=parent.message_id if parent else None,
                author=fact.author,
                body=fact.body,
            )
        root = identities.get(conversation.thread_key)
        if root is None or root.message_id != tokens["thread_id"]:
            raise RocketChatContractError(
                "Rocket.Chat seed thread token does not match the current root identity"
            )
        if len({item.message_id for item in identities.values()}) != len(identities):
            raise RocketChatContractError(
                "Rocket.Chat seed write_tokens contain duplicate message IDs"
            )
        return RocketChatSeedReceipt(
            conversation.benchmark,
            conversation.site,
            tokens["attempt_id"],
            writer_context,
            identities,
        )
    except (RocketChatContractError, TypeError, KeyError) as exc:
        return RocketChatObservationFailure("seed_metadata_invalid", str(exc))


def _close_notification_transport(transport: object | None) -> None:
    """Close a reader/writer session when the transport exposes one."""

    if transport is None:
        return
    close = getattr(transport, "close", None)
    if not callable(close):
        close = getattr(getattr(transport, "session", None), "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            # A best-effort close must not replace the evidence diagnostic.
            pass


def _load_current_notification_seed_receipt(
    instance: Mapping[str, object],
    conversation: RocketChatConversation,
    tokens: Mapping[str, str],
) -> RocketChatSeedReceipt | RocketChatObservationFailure:
    """Bind per-call seed tokens to one fresh ordinary writer identity."""

    writer_transport: RocketChatTransport | None = None
    try:
        writer_transport = RequestsRocketChatTransport(_origin(instance))
        writer_auth = writer_transport.login(_credentials(instance, "writer"))
        writer_context = _context(
            writer_auth,
            role="writer",
            username=conversation.writer_user,
        )
    except (RocketChatContractError, RocketChatTransportError, TypeError, ValueError) as exc:
        return RocketChatObservationFailure("writer_transport_failed", str(exc))
    finally:
        _close_notification_transport(writer_transport)
    return _reconstruct_notification_seed_receipt(
        conversation,
        tokens,
        writer_context,
    )


def load_rocket_chat_notification_seed_receipt(
    task: Mapping[str, object],
    instance: Mapping[str, object],
    seed_metadata: Mapping[str, object],
) -> RocketChatSeedReceipt | RocketChatObservationFailure:
    """Reconstruct the current seed for the production notification action."""

    if not isinstance(task, Mapping):
        return RocketChatObservationFailure(
            "task_contract_invalid", "Rocket.Chat task must be a mapping"
        )
    if not isinstance(instance, Mapping):
        return RocketChatObservationFailure(
            "instance_invalid", "Rocket.Chat instance must be a mapping"
        )
    facts = _notification_task_facts(task)
    if isinstance(facts, RocketChatObservationFailure):
        return facts
    conversation, _notification = facts
    tokens = _notification_seed_tokens(seed_metadata, conversation)
    if isinstance(tokens, RocketChatObservationFailure):
        return tokens
    return _load_current_notification_seed_receipt(
        instance,
        conversation,
        tokens,
    )


def _notification_from_thread_rows(
    rows: Sequence[Mapping[str, Any]],
    conversation: RocketChatConversation,
    notification: RocketChatNotification,
    seed_receipt: RocketChatSeedReceipt,
    reader_context: RocketChatParticipantContext,
    action_started_at: datetime,
) -> tuple[RocketChatNotificationObservation, datetime] | RocketChatObservationFailure:
    """Select exactly one persisted notification from a fresh thread read."""

    if not isinstance(reader_context, RocketChatParticipantContext):
        return RocketChatObservationFailure(
            "reader_transport_failed", "Rocket.Chat reader context must be typed"
        )
    try:
        materialized = tuple(rows)
    except TypeError:
        return RocketChatObservationFailure(
            "reader_transport_failed", "Rocket.Chat thread history is not iterable"
        )
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in materialized:
        if not isinstance(row, Mapping):
            return RocketChatObservationFailure(
                "reader_transport_failed", "Rocket.Chat thread history returned a non-object row"
            )
        message_id = row.get("_id")
        if not isinstance(message_id, str) or not message_id.strip():
            return RocketChatObservationFailure(
                "reader_transport_failed", "Rocket.Chat notification row is missing _id"
            )
        if message_id in by_id:
            return RocketChatObservationFailure(
                "ambiguous_message_identity", "history returned duplicate message IDs"
            )
        by_id[message_id] = row

    seed_ids = {item.message_id for item in seed_receipt.messages.values()}
    candidates = [row for message_id, row in by_id.items() if message_id not in seed_ids]
    if not candidates:
        return RocketChatObservationFailure(
            "missing_notification", "the current seed thread contains no notification message"
        )
    try:
        wire_body = render_rocket_chat_notification_message(notification)
    except (RocketChatContractError, TypeError) as exc:
        return RocketChatObservationFailure("wrong_recipient", str(exc))
    matching_body = [row for row in candidates if row.get("msg") == wire_body]
    if len(matching_body) > 1:
        return RocketChatObservationFailure(
            "duplicate_notification", "exactly one current notification is allowed"
        )
    if len(candidates) > 1:
        return RocketChatObservationFailure(
            "extra_artifact", "the current seed thread contains an unauthorized extra message"
        )
    row = matching_body[0] if matching_body else candidates[0]
    root = seed_receipt.messages.get(conversation.thread_key)
    if root is None:
        return RocketChatObservationFailure("missing_seed_identity", "seed root is missing")
    row_attempt_id = row.get("attempt_id")
    if row_attempt_id not in (None, seed_receipt.attempt_id):
        return RocketChatObservationFailure(
            "stale_message_identity",
            "notification row belongs to a different seed attempt",
        )
    row_logical_key = row.get("logical_key")
    if row_logical_key not in (None, ROCKET_CHAT_NOTIFICATION_LOGICAL_KEY):
        return RocketChatObservationFailure(
            "wrong_target_action", "notification row has an unsupported logical key"
        )
    raw_persisted_at = row.get("ts")
    if not isinstance(raw_persisted_at, str) or not raw_persisted_at.strip():
        return RocketChatObservationFailure(
            "stale_message_identity",
            "notification row is missing its persisted timestamp",
        )
    try:
        persisted_at = datetime.fromisoformat(raw_persisted_at.strip().replace("Z", "+00:00"))
    except ValueError:
        return RocketChatObservationFailure(
            "stale_message_identity",
            "notification row has an invalid persisted timestamp",
        )
    if persisted_at.tzinfo is None or persisted_at.utcoffset() is None:
        return RocketChatObservationFailure(
            "stale_message_identity",
            "notification row has a timezone-naive persisted timestamp",
        )
    persisted_at = persisted_at.astimezone(UTC)
    if persisted_at < action_started_at:
        return RocketChatObservationFailure(
            "stale_message_identity",
            "notification row predates the current action attempt",
        )
    try:
        message = _notification_identity(
            row,
            conversation,
            notification,
            seed_receipt.attempt_id,
            root.message_id,
            root.room_id,
        )
    except (RocketChatContractError, RocketChatTransportError) as exc:
        text = str(exc)
        reason = "message_identity_mismatch"
        if "wrong room" in text:
            reason = "wrong_room"
        elif "wrong thread" in text:
            reason = "wrong_thread"
        elif "wrong author" in text:
            reason = "wrong_actor"
        elif "wrong body" in text:
            reason = "message_body_mismatch"
        elif "recipient mention" in text or "invalid mentions" in text:
            reason = "wrong_recipient"
        return RocketChatObservationFailure(reason, text)
    try:
        observation = RocketChatNotificationObservation(
            benchmark=conversation.benchmark,
            site=conversation.site,
            attempt_id=seed_receipt.attempt_id,
            reader_context=reader_context,
            notification=notification,
            thread_id=root.message_id,
            message=message,
        )
        return observation, persisted_at
    except (RocketChatContractError, TypeError) as exc:
        return RocketChatObservationFailure("message_identity_mismatch", str(exc))


def load_rocket_chat_notification_reward_evidence(
    task: Mapping[str, object],
    instance: Mapping[str, object],
    seed_metadata: Mapping[str, object],
    action_started_at: datetime,
) -> RocketChatNotificationFinalStateReadback | RocketChatObservationFailure:
    """Read the current notification with independent ordinary credentials.

    The writer transport is used for one fresh login solely to bind the seed's
    ordinary writer context.  The notification message ID is discovered from
    a separate reader transport and is never synthesized as a writer receipt.
    """

    if not isinstance(task, Mapping):
        return RocketChatObservationFailure(
            "task_contract_invalid", "Rocket.Chat task must be a mapping"
        )
    if not isinstance(instance, Mapping):
        return RocketChatObservationFailure(
            "instance_invalid", "Rocket.Chat instance must be a mapping"
        )
    if (
        not isinstance(action_started_at, datetime)
        or action_started_at.tzinfo is None
        or action_started_at.utcoffset() is None
    ):
        return RocketChatObservationFailure(
            "action_context_invalid",
            "Rocket.Chat final-state readback requires an aware action-start timestamp",
        )
    action_started_at = action_started_at.astimezone(UTC)
    facts = _notification_task_facts(task)
    if isinstance(facts, RocketChatObservationFailure):
        return facts
    conversation, notification = facts
    tokens = _notification_seed_tokens(seed_metadata, conversation)
    if isinstance(tokens, RocketChatObservationFailure):
        return tokens

    seed_receipt = _load_current_notification_seed_receipt(
        instance,
        conversation,
        tokens,
    )
    if isinstance(seed_receipt, RocketChatObservationFailure):
        return seed_receipt

    reader_transport: RocketChatTransport | None = None
    try:
        reader_transport = RequestsRocketChatTransport(_origin(instance))
        try:
            reader_transport.login(_credentials(instance, "reader"))
            resolved_room_id = reader_transport.channel_id(conversation.room_id)
        except (RocketChatContractError, RocketChatTransportError) as exc:
            return RocketChatObservationFailure("reader_transport_failed", str(exc))
        root = seed_receipt.messages.get(conversation.thread_key)
        if root is None:
            return RocketChatObservationFailure("missing_seed_identity", "seed root is missing")
        if resolved_room_id != root.room_id:
            return RocketChatObservationFailure(
                "wrong_room",
                "reader channel mapping does not match the current seed receipt room",
            )
        reader = RocketChatHttpReader(instance, transport=reader_transport)
        seed_observation = reader.observe(conversation, seed_receipt)
        if isinstance(seed_observation, RocketChatObservationFailure):
            return seed_observation
        if not isinstance(seed_observation, RocketChatObservation):
            return RocketChatObservationFailure(
                "reader_transport_failed", "reader returned an unsupported seed observation"
            )
        if (
            seed_observation.reader_context.user_id != tokens["reader_user_id"]
            or seed_observation.reader_context.auth_context_id != tokens["reader_auth_context_id"]
        ):
            return RocketChatObservationFailure(
                "reader_identity_mismatch",
                "reader identity does not match the seed editor's independent reader binding",
            )
        thread_reader = getattr(reader_transport, "thread_history", None)
        if not callable(thread_reader):
            return RocketChatObservationFailure(
                "thread_history_unavailable",
                "Rocket.Chat final-state readback requires a dedicated thread reader",
            )
        rows = thread_reader(room_id=root.room_id, thread_id=root.message_id)
        notification_observation = _notification_from_thread_rows(
            rows,
            conversation,
            notification,
            seed_receipt,
            seed_observation.reader_context,
            action_started_at,
        )
        if isinstance(notification_observation, RocketChatObservationFailure):
            return notification_observation
        observation, persisted_at = notification_observation
        try:
            return RocketChatNotificationFinalStateReadback(
                seed_receipt=seed_receipt,
                observation=observation,
                action_started_at=action_started_at,
                persisted_at=persisted_at,
            )
        except (RocketChatContractError, TypeError) as exc:
            return RocketChatObservationFailure("unverified_readback", str(exc))
    except (RocketChatContractError, RocketChatTransportError, TypeError, ValueError) as exc:
        return RocketChatObservationFailure("reader_transport_failed", str(exc))
    finally:
        _close_notification_transport(reader_transport)


__all__ = [
    "RocketChatNotificationFinalStateReadback",
    "load_rocket_chat_notification_reward_evidence",
    "load_rocket_chat_notification_seed_receipt",
]
