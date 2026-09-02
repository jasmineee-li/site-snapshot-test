"""Opt-in Rocket.Chat notification writer and independent readback.

The decision runtime owns conversation seeding.  This module owns only the
dependent notification action and its REST readback so the transport remains
the single production seam while the state contract stays local to the
Rocket.Chat feature.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from warp_taskgen.phase_1.rocket_chat_contracts import (
    RocketChatContractError,
    RocketChatConversation,
    RocketChatMessageIdentity,
    RocketChatObservation,
    RocketChatObservationFailure,
    RocketChatSeedReceipt,
)
from warp_taskgen.phase_1.rocket_chat_notifications import (
    ROCKET_CHAT_NOTIFICATION_LOGICAL_KEY,
    RocketChatNotification,
    RocketChatNotificationObservation,
    RocketChatNotificationReader,
    RocketChatNotificationReceipt,
    RocketChatNotificationWriter,
    derive_rocket_chat_notification,
)
from warp_taskgen.sites.rocketchat_runtime import (
    RocketChatHttpReader,
    RocketChatHttpWriter,
    RocketChatTransport,
    RocketChatTransportError,
    _context,
    _credentials,
)


def _root_for_notification(
    transport: RocketChatTransport,
    conversation: RocketChatConversation,
    receipt: RocketChatSeedReceipt,
) -> RocketChatMessageIdentity:
    """Verify that the receipt's thread root is still the persisted root."""

    root = receipt.messages.get(conversation.thread_key)
    if root is None:
        raise RocketChatContractError(
            "notification receipt is missing the conversation thread root"
        )
    fact = conversation.message(conversation.thread_key)
    rows = transport.history(room_id=root.room_id)
    if any(not isinstance(row, Mapping) for row in rows):
        raise RocketChatTransportError("Rocket.Chat history returned a non-object message")
    matches = [row for row in rows if row.get("_id") == root.message_id]
    if len(matches) != 1:
        raise RocketChatTransportError(
            "notification requires the current persisted seed root; exact root ID was not found"
        )
    row = matches[0]
    user = row.get("u")
    author = user.get("username") if isinstance(user, Mapping) else None
    if (
        row.get("rid") != root.room_id
        or row.get("msg") != fact.body
        or author != fact.author
        or row.get("tmid") not in (None, "")
    ):
        raise RocketChatTransportError(
            "notification requires the current persisted seed root; root identity changed"
        )
    return root


def _mention_usernames(row: Mapping[str, Any]) -> tuple[str, ...]:
    raw = row.get("mentions")
    if not isinstance(raw, list):
        return ()
    names: list[str] = []
    for item in raw:
        if not isinstance(item, Mapping) or not isinstance(item.get("username"), str):
            continue
        names.append(item["username"].strip())
    return tuple(name for name in names if name)


def _notification_identity(
    row: Mapping[str, Any],
    conversation: RocketChatConversation,
    notification: RocketChatNotification,
    attempt_id: str,
    thread_id: str,
    expected_room_id: str,
) -> RocketChatMessageIdentity:
    """Convert one REST message into exact notification identity evidence."""

    message_id = row.get("_id")
    room_id = row.get("rid")
    body = row.get("msg")
    user = row.get("u")
    author = user.get("username") if isinstance(user, Mapping) else None
    actual_thread = row.get("tmid")
    if not all(isinstance(value, str) for value in (message_id, room_id, body, author)):
        raise RocketChatTransportError(
            "Rocket.Chat notification response is missing _id/rid/msg/u.username"
        )
    if room_id != expected_room_id:
        raise RocketChatTransportError("Rocket.Chat notification response has the wrong room")
    if actual_thread != thread_id:
        raise RocketChatTransportError("Rocket.Chat notification response has the wrong thread")
    if author != notification.author:
        raise RocketChatTransportError("Rocket.Chat notification response has the wrong author")
    if body != notification.body:
        raise RocketChatTransportError("Rocket.Chat notification response has the wrong body")
    mentions = _mention_usernames(row)
    if mentions != (notification.recipient,):
        raise RocketChatTransportError(
            "Rocket.Chat notification response does not carry exactly the intended recipient mention"
        )
    return RocketChatMessageIdentity(
        benchmark=conversation.benchmark,
        site=conversation.site,
        attempt_id=attempt_id,
        logical_key=ROCKET_CHAT_NOTIFICATION_LOGICAL_KEY,
        # The typed notification contract uses the generated logical room;
        # the physical REST room was checked against the current seed root
        # immediately above.
        room_id=notification.room_id,
        message_id=message_id,
        thread_id=thread_id,
        author=author,
        body=body,
        recipient=notification.recipient,
    )


class RocketChatHttpNotificationWriter(RocketChatHttpWriter, RocketChatNotificationWriter):
    """Ordinary authenticated notification action over the Rocket.Chat REST seam."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._mutation_possible = False

    def send_notification(
        self,
        conversation: RocketChatConversation,
        seed_receipt: RocketChatSeedReceipt,
        notification: RocketChatNotification,
    ) -> RocketChatNotificationReceipt:
        if not isinstance(conversation, RocketChatConversation):
            raise TypeError("send_notification requires a RocketChatConversation")
        if not isinstance(seed_receipt, RocketChatSeedReceipt):
            raise TypeError("send_notification requires the current seed receipt")
        if not isinstance(notification, RocketChatNotification):
            raise TypeError("send_notification requires a typed notification")
        expected = derive_rocket_chat_notification(conversation)
        if notification != expected:
            raise RocketChatContractError(
                "notification must be derived from the conversation's current decision"
            )
        writer = seed_receipt.writer_context
        auth = self.transport.login(self.credentials)
        context = _context(auth, role="writer", username=conversation.writer_user)
        if (
            context.user_id != writer.user_id
            or context.auth_context_id != writer.auth_context_id
            or context.auth_kind != writer.auth_kind
            or context.role != writer.role
            or self.credentials.username != writer.user_id
        ):
            raise RocketChatContractError(
                "notification writer identity does not match the seeded ordinary participant"
            )
        root = _root_for_notification(self.transport, conversation, seed_receipt)
        # The API may persist before returning a usable response.  Arm the
        # reset-required cleanup state before the mutation call.
        self._mutation_possible = True
        row = self.transport.send_message(
            room_id=root.room_id,
            body=notification.body,
            thread_id=root.message_id,
        )
        message = _notification_identity(
            row,
            conversation,
            notification,
            seed_receipt.attempt_id,
            root.message_id,
            root.room_id,
        )
        if message.message_id in {item.message_id for item in seed_receipt.messages.values()}:
            raise RocketChatTransportError(
                "Rocket.Chat notification response reused a seed message ID"
            )
        return RocketChatNotificationReceipt(
            benchmark=conversation.benchmark,
            site=conversation.site,
            attempt_id=seed_receipt.attempt_id,
            writer_context=writer,
            notification=notification,
            thread_id=root.message_id,
            message=message,
            seed_root=root,
            current_message=message,
        )

    def cleanup(self) -> None:
        if self._mutation_possible:
            raise RuntimeError(
                "Rocket.Chat notification cleanup requires an explicit disposable TAC reset/admin seam"
            )


class RocketChatHttpNotificationReader(RocketChatHttpReader, RocketChatNotificationReader):
    """Fresh authenticated reader for exact persisted notification identity."""

    def observe_notification(
        self,
        conversation: RocketChatConversation,
        seed_receipt: RocketChatSeedReceipt,
        notification_receipt: RocketChatNotificationReceipt,
    ) -> RocketChatNotificationObservation | RocketChatObservationFailure:
        if not isinstance(conversation, RocketChatConversation):
            return RocketChatObservationFailure(
                "invalid_conversation", "notification readback requires typed conversation facts"
            )
        if not isinstance(seed_receipt, RocketChatSeedReceipt):
            return RocketChatObservationFailure(
                "invalid_seed_receipt", "notification readback requires a typed seed receipt"
            )
        if not isinstance(notification_receipt, RocketChatNotificationReceipt):
            return RocketChatObservationFailure(
                "request_only_receipt", "persisted notification readback requires a typed receipt"
            )
        if notification_receipt.writer_context != seed_receipt.writer_context:
            return RocketChatObservationFailure(
                "writer_identity_mismatch",
                "notification receipt writer must bind the current seed writer context",
            )
        if notification_receipt.attempt_id != seed_receipt.attempt_id:
            return RocketChatObservationFailure(
                "stale_message_identity",
                "seed and notification receipts belong to different attempts",
            )
        expected = derive_rocket_chat_notification(conversation)
        if notification_receipt.notification != expected:
            return RocketChatObservationFailure(
                "wrong_target_action", "notification receipt does not bind the current decision"
            )
        seed_observation = self.observe(conversation, seed_receipt)
        if isinstance(seed_observation, RocketChatObservationFailure):
            return seed_observation
        if not isinstance(seed_observation, RocketChatObservation):
            return RocketChatObservationFailure(
                "reader_transport_failed", "seed readback returned an unsupported result"
            )
        root = seed_receipt.messages.get(conversation.thread_key)
        if root is None:
            return RocketChatObservationFailure("missing_seed_identity", "seed root is missing")
        if notification_receipt.thread_id != root.message_id:
            return RocketChatObservationFailure(
                "wrong_thread", "notification is attached to the wrong thread"
            )
        rows = self.transport.history(room_id=root.room_id)
        by_id: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                return RocketChatObservationFailure(
                    "reader_transport_failed", "Rocket.Chat history returned a non-object message"
                )
            message_id = row.get("_id")
            if not isinstance(message_id, str):
                continue
            if message_id in by_id:
                return RocketChatObservationFailure(
                    "ambiguous_message_identity", "history returned duplicate message IDs"
                )
            by_id[message_id] = row
        expected_id = notification_receipt.message_id
        row = by_id.get(expected_id)
        seed_ids = {item.message_id for item in seed_receipt.messages.values()}
        thread_rows = [
            item
            for item in rows
            if item.get("tmid") == root.message_id and item.get("_id") not in seed_ids
        ]
        matching_body = [item for item in thread_rows if item.get("msg") == expected.body]
        if len(matching_body) > 1:
            return RocketChatObservationFailure(
                "duplicate_notification", "exactly one notification is allowed for the attempt"
            )
        extras = [
            item
            for item in thread_rows
            if item.get("_id") != expected_id and item.get("msg") != expected.body
        ]
        if extras:
            return RocketChatObservationFailure(
                "extra_artifact", "the current attempt contains an unauthorized extra message"
            )
        if row is None:
            if matching_body:
                return RocketChatObservationFailure(
                    "stale_message_identity",
                    "a same-text notification exists under a different message ID",
                )
            return RocketChatObservationFailure(
                "stale_message_identity", "the exact notification message ID is missing"
            )
        try:
            message = _notification_identity(
                row,
                conversation,
                expected,
                seed_receipt.attempt_id,
                root.message_id,
                root.room_id,
            )
        except RocketChatTransportError as exc:
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
            elif "recipient mention" in text:
                reason = "wrong_recipient"
            return RocketChatObservationFailure(reason, text)
        if message != notification_receipt.message:
            return RocketChatObservationFailure(
                "message_identity_mismatch",
                "writer-returned notification identity does not match persisted evidence",
            )
        try:
            return RocketChatNotificationObservation(
                benchmark=conversation.benchmark,
                site=conversation.site,
                attempt_id=seed_receipt.attempt_id,
                reader_context=seed_observation.reader_context,
                notification=expected,
                thread_id=root.message_id,
                message=message,
            )
        except RocketChatContractError as exc:
            return RocketChatObservationFailure("message_identity_mismatch", str(exc))


def notification_writer_for(
    instance: Mapping[str, Any],
    transport: RocketChatTransport,
) -> RocketChatHttpNotificationWriter:
    """Build the opt-in writer without changing the default editor registry."""

    return RocketChatHttpNotificationWriter(
        instance,
        transport=transport,
        credentials=_credentials(instance, "writer"),
    )


def notification_reader_for(
    instance: Mapping[str, Any],
    transport: RocketChatTransport,
) -> RocketChatHttpNotificationReader:
    """Build the opt-in reader with explicit independent credentials."""

    return RocketChatHttpNotificationReader(
        instance,
        transport=transport,
        credentials=_credentials(instance, "reader"),
    )


__all__ = [
    "RocketChatHttpNotificationReader",
    "RocketChatHttpNotificationWriter",
    "notification_reader_for",
    "notification_writer_for",
]
