"""Feature-local notification action and readback fakes.

The response fakes remain in ``_rocket_chat_fakes``.  These doubles share its
small persisted store so notification tests exercise a distinct writer and
reader over the same state without introducing a production store.
"""

from __future__ import annotations

from warp_taskgen.phase_1.rocket_chat_contracts import (
    READER_AUTH_KINDS,
    RocketChatContractError,
    RocketChatConversation,
    RocketChatMessageIdentity,
    RocketChatObservationFailure,
    RocketChatSeedReceipt,
)
from warp_taskgen.phase_1.rocket_chat_notifications import (
    ROCKET_CHAT_NOTIFICATION_LOGICAL_KEY,
    RocketChatNotification,
    RocketChatNotificationObservation,
    RocketChatNotificationReceipt,
    derive_rocket_chat_notification,
)

from ._rocket_chat_fakes import (
    FakeAuthenticatedRocketChatReader,
    FakeRocketChatStore,
    FakeRocketChatWriter,
)


def _seed_root(
    store: FakeRocketChatStore,
    conversation: RocketChatConversation,
    receipt: RocketChatSeedReceipt,
) -> RocketChatMessageIdentity:
    """Require the supplied root to be the current persisted root identity."""

    if receipt.attempt_id != store.current_attempt_id:
        raise RocketChatContractError("notification seed receipt belongs to a stale attempt")
    root = receipt.messages.get(conversation.thread_key)
    if root is None:
        raise RocketChatContractError("notification seed receipt is missing the thread root")
    stored = store.get_message(root.message_id)
    current = store.current_identity(receipt.attempt_id, conversation.thread_key)
    fact = conversation.message(conversation.thread_key)
    expected = {
        "benchmark": conversation.benchmark,
        "site": conversation.site,
        "attempt_id": receipt.attempt_id,
        "logical_key": conversation.thread_key,
        "room_id": conversation.room_id,
        "thread_id": None,
        "author": fact.author,
        "body": fact.body,
    }
    actual = (
        None
        if stored is None
        else {
            "benchmark": stored.benchmark,
            "site": stored.site,
            "attempt_id": stored.attempt_id,
            "logical_key": stored.logical_key,
            "room_id": stored.room_id,
            "thread_id": stored.thread_id,
            "author": stored.author,
            "body": stored.body,
        }
    )
    if stored is None or current != root or stored != root or actual != expected:
        raise RocketChatContractError(
            "notification seed root must exist and be the current persisted logical message"
        )
    return root


class FakeRocketChatNotificationWriter(FakeRocketChatWriter):
    """Ordinary writer that binds a notification to the current seed root."""

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
        if seed_receipt.writer_context != self.context:
            raise RocketChatContractError("notification writer context does not match seed receipt")
        root = _seed_root(self.store, conversation, seed_receipt)
        identity = self.store.write_message(
            benchmark=conversation.benchmark,
            site=conversation.site,
            attempt_id=seed_receipt.attempt_id,
            logical_key=ROCKET_CHAT_NOTIFICATION_LOGICAL_KEY,
            room_id=notification.room_id,
            thread_id=root.message_id,
            author=notification.author,
            recipient=notification.recipient,
            body=notification.body,
        )
        return RocketChatNotificationReceipt(
            benchmark=conversation.benchmark,
            site=conversation.site,
            attempt_id=seed_receipt.attempt_id,
            writer_context=self.context,
            notification=notification,
            thread_id=root.message_id,
            message=identity,
            seed_root=root,
            current_message=identity,
        )


class FakeAuthenticatedRocketChatNotificationReader(FakeAuthenticatedRocketChatReader):
    """Independent ordinary reader that verifies shared persisted identities."""

    def observe_notification(
        self,
        conversation: RocketChatConversation,
        seed_receipt: RocketChatSeedReceipt,
        notification_receipt: RocketChatNotificationReceipt,
    ) -> RocketChatNotificationObservation | RocketChatObservationFailure:
        if not isinstance(conversation, RocketChatConversation):
            return RocketChatObservationFailure(
                "invalid_conversation", "reader needs typed conversation facts"
            )
        if not isinstance(seed_receipt, RocketChatSeedReceipt):
            return RocketChatObservationFailure(
                "invalid_seed_receipt", "reader needs a typed current seed receipt"
            )
        if not isinstance(notification_receipt, RocketChatNotificationReceipt):
            return RocketChatObservationFailure(
                "request_only_receipt", "persisted notification readback requires a typed receipt"
            )
        if self.context.auth_kind not in READER_AUTH_KINDS:
            return RocketChatObservationFailure(
                "writer_context_reused",
                "notification reader must use fresh reader credentials, not writer cookies or storage",
            )
        if self.context.role != "ordinary":
            return RocketChatObservationFailure(
                "reader_not_ordinary", "notification reader must be an ordinary participant"
            )
        writer = seed_receipt.writer_context
        expected = derive_rocket_chat_notification(conversation)
        if self.context.user_id == writer.user_id or self.context.user_id == expected.author:
            return RocketChatObservationFailure(
                "writer_context_reused",
                "notification reader must be distinct from the writer and notification author",
            )
        if (
            self.context.session_id == writer.session_id
            or self.context.auth_context_id == writer.auth_context_id
        ):
            return RocketChatObservationFailure(
                "writer_context_reused",
                "notification readback must use a fresh authenticated context distinct from the writer",
            )
        if self.context.user_id != conversation.reader_user:
            return RocketChatObservationFailure(
                "reader_identity_mismatch",
                "notification reader identity does not match the generated reader",
            )
        if seed_receipt.writer_context != notification_receipt.writer_context:
            return RocketChatObservationFailure(
                "writer_identity_mismatch",
                "notification receipt writer must bind the seed writer context",
            )
        if seed_receipt.attempt_id != notification_receipt.attempt_id:
            return RocketChatObservationFailure(
                "stale_message_identity",
                "seed and notification receipts belong to different attempts",
            )
        try:
            root = _seed_root(self.store, conversation, seed_receipt)
        except RocketChatContractError as exc:
            return RocketChatObservationFailure("stale_seed_root", str(exc))
        if self.store.current_attempt_id != seed_receipt.attempt_id:
            return RocketChatObservationFailure(
                "stale_message_identity", "notification receipt belongs to a prior attempt"
            )
        allowed_keys = {message.logical_key for message in conversation.messages}
        allowed_keys.add(ROCKET_CHAT_NOTIFICATION_LOGICAL_KEY)
        extras = tuple(
            item
            for item in self.store.attempt_messages(seed_receipt.attempt_id)
            if item.logical_key not in allowed_keys
        )
        if extras:
            return RocketChatObservationFailure(
                "extra_artifact",
                "the current attempt contains an unauthorized extra persisted message",
            )
        if notification_receipt.notification != expected:
            return RocketChatObservationFailure(
                "wrong_target_action",
                "notification does not target the recipient derived from the current decision",
            )
        if notification_receipt.thread_id != root.message_id:
            return RocketChatObservationFailure(
                "wrong_thread", "notification is not attached to the generated conversation thread"
            )
        persisted = self.store.get_message(notification_receipt.message_id)
        if persisted is None:
            return RocketChatObservationFailure(
                "stale_message_identity",
                "exact notification message ID is missing; stale same-text messages are not accepted",
            )
        matches = tuple(
            item
            for item in self.store.attempt_messages(seed_receipt.attempt_id)
            if item.logical_key == ROCKET_CHAT_NOTIFICATION_LOGICAL_KEY
        )
        if len(matches) != 1:
            return RocketChatObservationFailure(
                "duplicate_notification",
                "exactly one persisted notification is required for the current attempt",
            )
        current = self.store.current_identity(
            seed_receipt.attempt_id, ROCKET_CHAT_NOTIFICATION_LOGICAL_KEY
        )
        if current != notification_receipt.message:
            return RocketChatObservationFailure(
                "stale_message_identity",
                "notification receipt must bind the exact current persisted logical message",
            )
        if matches[0].message_id != notification_receipt.message_id:
            return RocketChatObservationFailure(
                "stale_message_identity",
                "notification receipt does not bind the sole persisted notification message",
            )
        expected_identity = {
            "benchmark": conversation.benchmark,
            "site": conversation.site,
            "attempt_id": seed_receipt.attempt_id,
            "logical_key": ROCKET_CHAT_NOTIFICATION_LOGICAL_KEY,
            "room_id": expected.room_id,
            "thread_id": root.message_id,
            "author": expected.author,
            "recipient": expected.recipient,
            "body": expected.body,
        }
        actual_identity = {
            "benchmark": persisted.benchmark,
            "site": persisted.site,
            "attempt_id": persisted.attempt_id,
            "logical_key": persisted.logical_key,
            "room_id": persisted.room_id,
            "thread_id": persisted.thread_id,
            "author": persisted.author,
            "recipient": persisted.recipient,
            "body": persisted.body,
        }
        if actual_identity != expected_identity:
            field = next(
                key for key, value in expected_identity.items() if actual_identity[key] != value
            )
            reason = {
                "room_id": "wrong_room",
                "thread_id": "wrong_thread",
                "author": "wrong_actor",
                "recipient": "wrong_recipient",
                "body": "message_body_mismatch",
            }.get(field, "message_identity_mismatch")
            return RocketChatObservationFailure(
                reason,
                f"persisted notification {field} does not match the exact generated action contract",
            )
        if persisted != notification_receipt.message:
            return RocketChatObservationFailure(
                "message_identity_mismatch",
                "writer-returned notification identity does not match persisted evidence",
            )
        return RocketChatNotificationObservation(
            benchmark=conversation.benchmark,
            site=conversation.site,
            attempt_id=seed_receipt.attempt_id,
            reader_context=self.context,
            notification=expected,
            thread_id=root.message_id,
            message=persisted,
        )


__all__ = [
    "FakeAuthenticatedRocketChatNotificationReader",
    "FakeRocketChatNotificationWriter",
]
