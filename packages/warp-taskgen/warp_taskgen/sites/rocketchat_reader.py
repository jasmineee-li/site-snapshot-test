"""Feature-local exact Rocket.Chat independent-reader REST projection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from warp_taskgen.phase_1.rocket_chat_contracts import (
    RocketChatContractError,
    RocketChatConversation,
    RocketChatMessageIdentity,
    RocketChatObservation,
    RocketChatObservationFailure,
    RocketChatSeedReceipt,
)
from warp_taskgen.sites.rocketchat_transport import (
    RocketChatCredentials,
    RocketChatTransport,
    RocketChatTransportError,
    _context,
    _credentials,
    _declared_reader_user_id,
    _identity,
)

__all__ = ["RocketChatHttpReader"]


def _merge_reader_rows(
    root_rows: Sequence[Mapping[str, Any]],
    thread_rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Mapping[str, Any], ...] | None, str | None]:
    """Join room-root and dedicated-thread rows without hiding conflicts."""

    merged: list[Mapping[str, Any]] = []
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in (*root_rows, *thread_rows):
        if not isinstance(row, Mapping):
            return None, "thread history returned a non-object row"
        message_id = row.get("_id")
        if not isinstance(message_id, str) or not message_id.strip():
            # Ignore unrelated malformed ambient rows only when they do not
            # claim an identity; a seeded row is validated by _identity below.
            merged.append(row)
            continue
        prior = by_id.get(message_id)
        if prior is None:
            by_id[message_id] = row
            merged.append(row)
            continue
        # A deployment may echo a reply in both surfaces.  Identical copies
        # are harmless; conflicting copies are an identity attack and must
        # remain visible to the reader as a failure.
        if dict(prior) != dict(row):
            return None, f"history returned conflicting rows for message {message_id!r}"
    return tuple(merged), None


class RocketChatHttpReader:
    def __init__(
        self,
        instance: Mapping[str, Any],
        *,
        transport: RocketChatTransport,
        credentials: RocketChatCredentials | None = None,
    ) -> None:
        self.instance = dict(instance)
        self.transport = transport
        self.credentials = credentials or _credentials(self.instance, "reader")

    def observe(
        self, conversation: RocketChatConversation, receipt: RocketChatSeedReceipt
    ) -> RocketChatObservation | RocketChatObservationFailure:
        try:
            if not isinstance(conversation, RocketChatConversation):
                return RocketChatObservationFailure(
                    "invalid_conversation", "reader requires a typed conversation"
                )
            if not isinstance(receipt, RocketChatSeedReceipt):
                return RocketChatObservationFailure(
                    "invalid_seed_receipt", "reader requires a typed seed receipt"
                )
            auth = self.transport.login(self.credentials)
            writer = receipt.writer_context
            if auth.user_id == writer.user_id:
                return RocketChatObservationFailure(
                    "writer_context_reused", "reader must use fresh independent credentials"
                )
            reader = _context(auth, role="reader", username=conversation.reader_user)
            if (
                reader.user_id == writer.user_id
                or reader.session_id == writer.session_id
                or reader.auth_context_id == writer.auth_context_id
            ):
                return RocketChatObservationFailure(
                    "writer_context_reused", "reader must use fresh independent credentials"
                )
            try:
                declared_reader_id = _declared_reader_user_id(self.instance)
            except RocketChatContractError as exc:
                return RocketChatObservationFailure("reader_identity_invalid", str(exc))
            if declared_reader_id is not None and reader.user_id != declared_reader_id:
                return RocketChatObservationFailure(
                    "reader_identity_mismatch",
                    "REST reader physical ID does not match the configured browser reader ID",
                )
            root = receipt.messages.get(conversation.thread_key)
            if root is None:
                return RocketChatObservationFailure(
                    "missing_seed_identity", "receipt omits the conversation thread root"
                )
            root_rows = self.transport.history(room_id=root.room_id)
            thread_reader = getattr(self.transport, "thread_history", None)
            if not callable(thread_reader):
                return RocketChatObservationFailure(
                    "thread_history_unavailable",
                    "Rocket.Chat room history does not expose a dedicated thread reader",
                )
            thread_rows = thread_reader(room_id=root.room_id, thread_id=root.message_id)
            rows, merge_error = _merge_reader_rows(root_rows, thread_rows)
            if rows is None:
                return RocketChatObservationFailure(
                    "ambiguous_message_identity",
                    merge_error or "history rows could not be joined",
                )
            by_id: dict[str, Mapping[str, Any]] = {}
            for row in rows:
                message_id = row.get("_id")
                if isinstance(message_id, str):
                    if message_id in by_id:
                        return RocketChatObservationFailure(
                            "ambiguous_message_identity", "history returned duplicate message IDs"
                        )
                    by_id[message_id] = row
            expected_ids = [
                receipt.messages[fact.logical_key].message_id
                for fact in conversation.messages
                if fact.logical_key in receipt.messages
            ]
            if (
                all(message_id in by_id for message_id in expected_ids)
                and [row.get("_id") for row in rows if row.get("_id") in expected_ids]
                != expected_ids
            ):
                return RocketChatObservationFailure(
                    "message_order_mismatch",
                    "history did not expose receipt IDs in conversation order",
                )
            observed: dict[str, RocketChatMessageIdentity] = {}
            for fact in conversation.messages:
                expected = receipt.messages.get(fact.logical_key)
                if expected is None or expected.message_id not in by_id:
                    return RocketChatObservationFailure(
                        "stale_message_identity",
                        f"history omitted {fact.logical_key} message identity",
                    )
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
            if {item.message_id for item in observed.values()} != {
                item.message_id for item in receipt.messages.values()
            }:
                return RocketChatObservationFailure(
                    "stale_message_identity", "history did not expose exactly the receipt IDs"
                )
            return RocketChatObservation(
                conversation.benchmark,
                conversation.site,
                receipt.attempt_id,
                root.room_id,
                receipt.messages[conversation.thread_key].message_id,
                reader,
                observed,
                conversation.current_decision,
            )
        except (RocketChatContractError, RocketChatTransportError) as exc:
            return RocketChatObservationFailure("reader_transport_failed", str(exc))
