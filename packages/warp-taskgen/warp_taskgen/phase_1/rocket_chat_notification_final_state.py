"""Typed Rocket.Chat notification final-state evidence.

The final-state seam deliberately carries only the current seed receipt and a
fresh reader observation.  It does not model, retain, or reconstruct a
notification writer receipt.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from warp_taskgen.phase_1.rocket_chat_contracts import (
    RocketChatContractError,
    RocketChatMessageIdentity,
    RocketChatSeedReceipt,
)
from warp_taskgen.phase_1.rocket_chat_notifications import (
    ROCKET_CHAT_NOTIFICATION_LOGICAL_KEY,
    RocketChatNotification,
    RocketChatNotificationObservation,
)


def _seed_root(
    seed_receipt: RocketChatSeedReceipt,
    notification: RocketChatNotification,
) -> RocketChatMessageIdentity:
    root = seed_receipt.messages.get(notification.thread_key)
    if root is None:
        raise RocketChatContractError(
            "final-state readback requires the seed receipt's thread root"
        )
    if root.logical_key != notification.thread_key or root.thread_id is not None:
        raise RocketChatContractError("final-state seed root does not bind the notification thread")
    return root


@dataclass(frozen=True)
class RocketChatNotificationFinalStateReadback:
    """Independent-reader final state bound to the current seed receipt.

    Unlike the legacy writer/reader combination, this evidence intentionally
    carries no notification writer receipt.  The reader discovers the current
    notification message ID from persisted thread state and binds it to the
    seed's physical root, room, and attempt.
    """

    seed_receipt: RocketChatSeedReceipt
    observation: RocketChatNotificationObservation
    action_started_at: datetime
    persisted_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.seed_receipt, RocketChatSeedReceipt):
            raise RocketChatContractError(
                "final-state readback requires a typed current seed receipt"
            )
        if not isinstance(self.observation, RocketChatNotificationObservation):
            raise RocketChatContractError(
                "final-state readback requires a typed independent-reader observation"
            )
        for field, value in (
            ("action_started_at", self.action_started_at),
            ("persisted_at", self.persisted_at),
        ):
            if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
                raise RocketChatContractError(
                    f"final-state {field} must be a timezone-aware timestamp"
                )
            object.__setattr__(self, field, value.astimezone(UTC))
        if self.persisted_at < self.action_started_at:
            raise RocketChatContractError(
                "final-state notification predates the current action attempt"
            )
        if self.observation.attempt_id != self.seed_receipt.attempt_id:
            raise RocketChatContractError("final-state readback belongs to a different attempt")
        if (
            self.observation.benchmark != self.seed_receipt.benchmark
            or self.observation.site != self.seed_receipt.site
        ):
            raise RocketChatContractError(
                "final-state readback Benchmark/Site do not match the seed"
            )

        notification = self.observation.notification
        root = _seed_root(self.seed_receipt, notification)
        if self.observation.thread_id != root.message_id:
            raise RocketChatContractError(
                "final-state notification thread does not bind the seed's current root"
            )
        message = self.observation.message
        seed_message_ids = {identity.message_id for identity in self.seed_receipt.messages.values()}
        if message.message_id in seed_message_ids:
            raise RocketChatContractError(
                "final-state notification message ID aliases a seeded message"
            )
        if message.room_id != root.room_id:
            raise RocketChatContractError(
                "final-state notification does not bind the seed's physical room"
            )
        if message.thread_id != root.message_id:
            raise RocketChatContractError(
                "final-state notification does not bind the seed's current root"
            )
        if message.logical_key != ROCKET_CHAT_NOTIFICATION_LOGICAL_KEY:
            raise RocketChatContractError("final-state notification has an unsupported logical key")
        if message.author != notification.author:
            raise RocketChatContractError(
                "final-state notification author does not match the action"
            )
        if message.recipient != notification.recipient or message.body != notification.body:
            raise RocketChatContractError(
                "final-state notification identity does not match the generated action"
            )

        reader = self.observation.reader_context
        writer = self.seed_receipt.writer_context
        if (writer.username or writer.user_id) != notification.author:
            raise RocketChatContractError(
                "final-state writer identity does not match the notification author"
            )
        if reader.auth_kind != "reader_credentials" or reader.role != "ordinary":
            raise RocketChatContractError(
                "final-state readback requires an independent ordinary reader"
            )
        reader_username = reader.username or reader.user_id
        writer_username = writer.username or writer.user_id
        if reader_username == notification.author or reader_username == writer_username:
            raise RocketChatContractError("final-state reader must be distinct from the writer")
        if (
            reader.user_id == writer.user_id
            or reader.session_id == writer.session_id
            or reader.auth_context_id == writer.auth_context_id
        ):
            raise RocketChatContractError("final-state reader context must be distinct from writer")

    @property
    def notification(self) -> RocketChatNotification:
        return self.observation.notification

    @property
    def message(self) -> RocketChatMessageIdentity:
        return self.observation.message

    @property
    def message_id(self) -> str:
        return self.observation.message_id


__all__ = ["RocketChatNotificationFinalStateReadback"]
