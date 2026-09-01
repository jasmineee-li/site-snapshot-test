"""Test-only Rocket.Chat writer, store, and reader doubles.

Production exposes only the feature protocols.  These deterministic doubles
make provenance failures reproducible without a browser or a live service.
"""

from __future__ import annotations

from warp_taskgen.phase_1.rocket_chat_contracts import (
    READER_AUTH_KINDS,
    WRITER_AUTH_KINDS,
    RocketChatContractError,
    RocketChatConversation,
    RocketChatMessageIdentity,
    RocketChatObservation,
    RocketChatObservationFailure,
    RocketChatParticipantContext,
    RocketChatSeedReceipt,
)


class FakeRocketChatStore:
    """In-memory persisted messages for focused acceptance tests only."""

    def __init__(self) -> None:
        self._messages: dict[str, RocketChatMessageIdentity] = {}
        self._attempt_message_ids: dict[str, dict[str, str]] = {}
        self._current_message_ids: dict[str, dict[str, str]] = {}
        self._next_message = 1
        self._next_attempt = 1
        self._current_attempt_id: str | None = None

    def begin_attempt(self) -> str:
        attempt = f"rc-attempt-{self._next_attempt:04d}"
        self._next_attempt += 1
        self._current_attempt_id = attempt
        self._attempt_message_ids[attempt] = {}
        self._current_message_ids[attempt] = {}
        return attempt

    @property
    def current_attempt_id(self) -> str | None:
        return self._current_attempt_id

    def write_message(
        self,
        *,
        benchmark: str,
        site: str,
        attempt_id: str,
        logical_key: str,
        room_id: str,
        thread_id: str | None,
        author: str,
        body: str,
    ) -> RocketChatMessageIdentity:
        message_id = f"rc-message-{self._next_message:04d}"
        self._next_message += 1
        identity = RocketChatMessageIdentity(
            benchmark=benchmark,
            site=site,
            attempt_id=attempt_id,
            logical_key=logical_key,
            room_id=room_id,
            message_id=message_id,
            thread_id=thread_id,
            author=author,
            body=body,
        )
        self._messages[message_id] = identity
        self._attempt_message_ids.setdefault(attempt_id, {})[logical_key] = message_id
        self._current_message_ids.setdefault(attempt_id, {})[logical_key] = message_id
        return identity

    def get_message(self, message_id: str) -> RocketChatMessageIdentity | None:
        return self._messages.get(message_id)

    def clear_message(self, message_id: str) -> None:
        self._messages.pop(message_id, None)

    def current_message_id(self, attempt_id: str, logical_key: str) -> str | None:
        return self._current_message_ids.get(attempt_id, {}).get(logical_key)

    def replace_message(
        self,
        identity: RocketChatMessageIdentity,
        *,
        message_id: str,
    ) -> str:
        """Replace a message ID while retaining the same attempt and logical key."""

        self._messages.pop(identity.message_id, None)
        replacement = RocketChatMessageIdentity(
            benchmark=identity.benchmark,
            site=identity.site,
            attempt_id=identity.attempt_id,
            logical_key=identity.logical_key,
            room_id=identity.room_id,
            message_id=message_id,
            thread_id=identity.thread_id,
            author=identity.author,
            body=identity.body,
        )
        self._messages[message_id] = replacement
        self._current_message_ids.setdefault(identity.attempt_id, {})[identity.logical_key] = (
            message_id
        )
        return message_id

    def write_stale_copy(
        self,
        identity: RocketChatMessageIdentity,
        *,
        message_id: str,
    ) -> str:
        stale = RocketChatMessageIdentity(
            benchmark=identity.benchmark,
            site=identity.site,
            attempt_id=f"{identity.attempt_id}-stale",
            logical_key=identity.logical_key,
            room_id=identity.room_id,
            message_id=message_id,
            thread_id=identity.thread_id,
            author=identity.author,
            body=identity.body,
        )
        self._messages[message_id] = stale
        return message_id


class FakeRocketChatWriter:
    """Ordinary participant writer for one immutable seed attempt."""

    def __init__(self, store: FakeRocketChatStore, context: RocketChatParticipantContext) -> None:
        if not isinstance(store, FakeRocketChatStore):
            raise TypeError("Rocket.Chat writer requires a FakeRocketChatStore")
        if not isinstance(context, RocketChatParticipantContext):
            raise TypeError("Rocket.Chat writer requires a typed participant context")
        if context.auth_kind not in WRITER_AUTH_KINDS:
            raise RocketChatContractError("Rocket.Chat writer requires writer credentials")
        if context.role != "ordinary":
            raise RocketChatContractError("Rocket.Chat writer must be an ordinary participant")
        self.store = store
        self.context = context

    def seed_conversation(self, conversation: RocketChatConversation) -> RocketChatSeedReceipt:
        if not isinstance(conversation, RocketChatConversation):
            raise TypeError("seed_conversation requires a RocketChatConversation")
        if conversation.writer_user != self.context.user_id:
            raise RocketChatContractError("writer actor does not match generated conversation")
        attempt_id = self.store.begin_attempt()
        identities: dict[str, RocketChatMessageIdentity] = {}
        for fact in conversation.messages:
            if fact.author != self.context.user_id:
                raise RocketChatContractError(
                    f"ordinary writer cannot impersonate message author {fact.author!r}"
                )
            thread_id = None
            if fact.thread_key is not None:
                parent = identities.get(fact.thread_key)
                if parent is None:
                    raise RocketChatContractError(
                        f"message {fact.logical_key!r} references an unseeded thread"
                    )
                thread_id = parent.message_id
            identities[fact.logical_key] = self.store.write_message(
                benchmark=conversation.benchmark,
                site=conversation.site,
                attempt_id=attempt_id,
                logical_key=fact.logical_key,
                room_id=fact.room_id,
                thread_id=thread_id,
                author=fact.author,
                body=fact.body,
            )
        return RocketChatSeedReceipt(
            benchmark=conversation.benchmark,
            site=conversation.site,
            attempt_id=attempt_id,
            writer_context=self.context,
            messages=identities,
        )


class FakeAuthenticatedRocketChatReader:
    """Independent ordinary-reader adapter over the test message store."""

    def __init__(self, store: FakeRocketChatStore, context: RocketChatParticipantContext) -> None:
        if not isinstance(store, FakeRocketChatStore):
            raise TypeError("Rocket.Chat reader requires a FakeRocketChatStore")
        if not isinstance(context, RocketChatParticipantContext):
            raise TypeError("Rocket.Chat reader requires a typed participant context")
        self.store = store
        self.context = context

    def observe(
        self,
        conversation: RocketChatConversation,
        receipt: RocketChatSeedReceipt,
    ) -> RocketChatObservation | RocketChatObservationFailure:
        if not isinstance(conversation, RocketChatConversation):
            return RocketChatObservationFailure(
                "invalid_conversation", "reader needs typed conversation facts"
            )
        if not isinstance(receipt, RocketChatSeedReceipt):
            return RocketChatObservationFailure(
                "invalid_seed_receipt", "reader needs a typed current seed receipt"
            )
        if self.context.auth_kind not in READER_AUTH_KINDS:
            return RocketChatObservationFailure(
                "writer_context_reused",
                "reader must use a fresh independent credential context, not writer cookies or storage",
            )
        if self.context.role != "ordinary":
            return RocketChatObservationFailure(
                "reader_not_ordinary",
                "independent reader must be an ordinary participant",
            )
        if self.context.user_id != conversation.reader_user:
            return RocketChatObservationFailure(
                "reader_identity_mismatch",
                "reader identity does not match the generated authenticated reader",
            )
        writer = receipt.writer_context
        if writer.user_id != conversation.writer_user:
            return RocketChatObservationFailure(
                "writer_identity_mismatch",
                "seed receipt writer identity does not match generated conversation",
            )
        if (
            self.context.user_id == writer.user_id
            or self.context.session_id == writer.session_id
            or self.context.auth_context_id == writer.auth_context_id
        ):
            return RocketChatObservationFailure(
                "writer_context_reused",
                "reader must use a fresh authenticated context distinct from the writer",
            )
        if receipt.benchmark != conversation.benchmark or receipt.site != conversation.site:
            return RocketChatObservationFailure(
                "benchmark_or_site_mismatch",
                "seed receipt Benchmark and Site must match generated conversation",
            )
        if self.store.current_attempt_id != receipt.attempt_id:
            return RocketChatObservationFailure(
                "stale_message_identity",
                "seed receipt belongs to a prior attempt; current message IDs are required",
            )
        expected_by_key = {message.logical_key: message for message in conversation.messages}
        if set(receipt.messages) != set(expected_by_key):
            return RocketChatObservationFailure(
                "message_set_mismatch",
                "reader receipt must contain exactly the generated message keys",
            )
        root_identity = receipt.messages.get(conversation.thread_key)
        if root_identity is None:
            return RocketChatObservationFailure(
                "wrong_thread",
                "reader receipt is missing the generated conversation thread root",
            )
        for key, fact in expected_by_key.items():
            identity = receipt.messages[key]
            stored = self.store.get_message(identity.message_id)
            if stored is not None and stored.attempt_id != receipt.attempt_id:
                return RocketChatObservationFailure(
                    "stale_message_identity",
                    f"message ID for {key!r} points to a stale attempt; current message ID is required",
                )
            current_id = self.store.current_message_id(receipt.attempt_id, key)
            if current_id != identity.message_id:
                return RocketChatObservationFailure(
                    "same_attempt_message_replacement",
                    f"message ID for {key!r} was replaced within attempt {receipt.attempt_id!r}",
                )
            if stored is None:
                return RocketChatObservationFailure(
                    "stale_message_identity",
                    f"exact message ID for {key!r} is missing; stale same-text messages are not accepted",
                )
            if stored.logical_key != key or identity.logical_key != key:
                return RocketChatObservationFailure(
                    "stale_message_identity",
                    f"message ID for {key!r} does not bind to the current logical message",
                )
            if stored.attempt_id != receipt.attempt_id:
                return RocketChatObservationFailure(
                    "stale_message_identity",
                    f"message ID for {key!r} points to a stale attempt; current message ID is required",
                )
            if stored.room_id != fact.room_id or identity.room_id != fact.room_id:
                return RocketChatObservationFailure(
                    "wrong_room",
                    f"message {key!r} is not in the generated room {fact.room_id!r}",
                )
            expected_thread_id = None
            if fact.thread_key is not None:
                parent = receipt.messages.get(fact.thread_key)
                if parent is None:
                    return RocketChatObservationFailure(
                        "wrong_thread",
                        f"message {key!r} references an unknown generated thread",
                    )
                expected_thread_id = parent.message_id
            if stored.thread_id != expected_thread_id or identity.thread_id != expected_thread_id:
                return RocketChatObservationFailure(
                    "wrong_thread",
                    f"message {key!r} is attached to the wrong thread",
                )
            if stored.author != fact.author or identity.author != fact.author:
                return RocketChatObservationFailure(
                    "wrong_actor", f"message {key!r} has the wrong author"
                )
            if stored.body != fact.body or identity.body != fact.body:
                return RocketChatObservationFailure(
                    "message_body_mismatch",
                    f"message {key!r} body does not match the generated exact content",
                )
            if stored != identity:
                return RocketChatObservationFailure(
                    "message_identity_mismatch",
                    f"receipt identity for {key!r} does not match persisted message evidence",
                )
        return RocketChatObservation(
            benchmark=conversation.benchmark,
            site=conversation.site,
            attempt_id=receipt.attempt_id,
            room_id=conversation.room_id,
            thread_id=root_identity.message_id,
            reader_context=self.context,
            messages=receipt.messages,
            current_decision=conversation.current_decision,
        )


__all__ = ["FakeAuthenticatedRocketChatReader", "FakeRocketChatStore", "FakeRocketChatWriter"]
