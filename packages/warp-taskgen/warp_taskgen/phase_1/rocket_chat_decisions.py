"""Static Rocket.Chat conversation generation and task compilation.

The module owns generated conversation facts and the task-shaped contract only.
Storage, authentication, reading, and grading are separate feature seams.
"""

from __future__ import annotations

from collections.abc import Mapping

from warp_taskgen.benchmark_capabilities import resolve_evaluator_authority
from warp_taskgen.phase_1.rocket_chat_contracts import (
    DECISION_FIELDS,
    ROCKET_CHAT_BENCHMARK,
    ROCKET_CHAT_EVALUATOR_AUTHORITY,
    ROCKET_CHAT_EVALUATOR_NAME,
    ROCKET_CHAT_RESOURCE_KIND,
    ROCKET_CHAT_SITE,
    ROCKET_CHAT_TASK_KIND,
    RocketChatContractError,
    RocketChatConversation,
    RocketChatCorrectionFact,
    RocketChatDecision,
    RocketChatMessageFact,
    _identity,
    _text,
    infer_rocket_chat_benchmark,
    validate_rocket_chat_benchmark,
)


def resolve_rocket_chat_evaluator_authority(
    benchmark: object = ROCKET_CHAT_BENCHMARK,
    *,
    task_id: object | None = None,
) -> str:
    """Require TAC's WARP-local evaluator for task-id-less responses."""

    canonical = validate_rocket_chat_benchmark(benchmark)
    if task_id is not None:
        try:
            resolve_evaluator_authority(canonical, task_id=task_id)
        except ValueError as exc:
            raise RocketChatContractError(
                "Rocket.Chat response evaluator is task-id-less; native task IDs are unsupported"
            ) from exc
        raise RocketChatContractError(
            "Rocket.Chat response evaluator is task-id-less; native task IDs are unsupported"
        )
    try:
        authority = resolve_evaluator_authority(canonical, task_id=None)
    except ValueError as exc:
        raise RocketChatContractError(
            "Rocket.Chat Benchmark has no WARP-local evaluator authority"
        ) from exc
    if authority != ROCKET_CHAT_EVALUATOR_AUTHORITY:
        raise RocketChatContractError(
            "Rocket.Chat response requires WARP-local task-id-less evaluator authority"
        )
    return authority


def generate_rocket_chat_conversation(
    *,
    room_id: str = "project-alpha",
    thread_key: str = "plan",
    initial_owner: str = "Alex",
    initial_due_date: str = "2026-09-15",
    corrected_owner: str = "Priya",
    corrected_due_date: str = "2026-09-18",
    writer_user: str = "planner",
    reader_user: str = "reviewer",
    run_marker: str = "RC-DECISION-001",
) -> RocketChatConversation:
    """Generate one deterministic plan/update/confirmed-correction world."""

    # Let the typed records enforce all bounded text and identity invariants.
    room = _identity(room_id, field="conversation room")
    thread = _identity(thread_key, field="conversation thread")
    writer = _identity(writer_user, field="writer identity")
    marker = _text(run_marker, field="conversation marker", max_length=120)
    initial = RocketChatDecision(owner=initial_owner, due_date=initial_due_date)
    corrected_owner_text = _text(corrected_owner, field="corrected owner")
    corrected_due_date_text = _text(corrected_due_date, field="corrected due date")
    messages = (
        RocketChatMessageFact(
            logical_key=thread,
            room_id=room,
            thread_key=None,
            author=writer,
            body=(f"Project plan: owner={initial.owner}; due_date={initial.due_date}. {marker}"),
            kind="plan",
        ),
        RocketChatMessageFact(
            logical_key="update",
            room_id=room,
            thread_key=thread,
            author=writer,
            body=f"Update: implementation is on track; review remains in progress. {marker}",
            kind="update",
        ),
        RocketChatMessageFact(
            logical_key="correction",
            room_id=room,
            thread_key=thread,
            author=writer,
            body=(
                f"Confirmed correction: owner={corrected_owner_text}; "
                f"due_date={corrected_due_date_text}. {marker}"
            ),
            kind="correction",
        ),
    )
    corrections = (
        RocketChatCorrectionFact(
            message_key="correction",
            field="owner",
            previous_value=initial.owner,
            current_value=corrected_owner_text,
            confirmed_by=writer,
        ),
        RocketChatCorrectionFact(
            message_key="correction",
            field="due_date",
            previous_value=initial.due_date,
            current_value=corrected_due_date_text,
            confirmed_by=writer,
        ),
    )
    return RocketChatConversation(
        room_id=room,
        thread_key=thread,
        messages=messages,
        corrections=corrections,
        initial_decision=initial,
        writer_user=writer,
        reader_user=reader_user,
    )


def derive_rocket_chat_decision(conversation: RocketChatConversation) -> RocketChatDecision:
    """Project the current finite decision from one validated conversation."""

    if not isinstance(conversation, RocketChatConversation):
        raise TypeError("derive_rocket_chat_decision requires a RocketChatConversation")
    return conversation.current_decision


_TASK_KEYS = frozenset(
    {
        "benchmark",
        "site",
        "task_kind",
        "task_id",
        "evaluator_authority",
        "start_urls",
        "conversation",
        "response_schema",
        "expected_decision",
        "reward_function",
        "reader_contract",
    }
)
_CONVERSATION_KEYS = frozenset(
    {
        "benchmark",
        "site",
        "room_id",
        "thread_key",
        "writer_user",
        "reader_user",
        "initial_decision",
        "messages",
        "corrections",
        "expected_decision",
    }
)
_MESSAGE_KEYS = frozenset({"logical_key", "room_id", "thread_key", "author", "body", "kind"})
_CORRECTION_KEYS = frozenset(
    {"message_key", "field", "previous_value", "current_value", "confirmed_by", "confirmed"}
)
_SCHEMA_KEYS = frozenset({"type", "required", "additionalProperties", "properties"})
_READER_CONTRACT_KEYS = frozenset(
    {
        "auth",
        "writer_identity",
        "reader_identity",
        "identities_must_be_distinct",
        "resource_kind",
        "message_identity",
    }
)
_REWARD_KEYS = frozenset({"eval"})
_EVAL_KEYS = frozenset({"evaluator", "expected"})


def _require_exact_keys(value: Mapping[str, object], expected: frozenset[str], label: str) -> None:
    actual = set(value)
    missing = expected - actual
    extra = actual - expected
    if missing:
        raise RocketChatContractError(
            f"{label} is missing fields: {', '.join(sorted(str(item) for item in missing))}"
        )
    if extra:
        raise RocketChatContractError(
            f"{label} has extra fields: {', '.join(sorted(str(item) for item in extra))}"
        )


def compile_rocket_chat_task(conversation: RocketChatConversation) -> dict[str, object]:
    """Compile one static response task with explicit TAC provenance."""

    if not isinstance(conversation, RocketChatConversation):
        raise TypeError("compile_rocket_chat_task requires a RocketChatConversation")
    expected = conversation.expected_decision.as_dict()
    task: dict[str, object] = {
        "benchmark": conversation.benchmark,
        "site": conversation.site,
        "task_kind": ROCKET_CHAT_TASK_KIND,
        "task_id": None,
        "evaluator_authority": resolve_rocket_chat_evaluator_authority(conversation.benchmark),
        "start_urls": [f"__ROCKETCHAT__/channel/{conversation.room_id}"],
        "conversation": conversation.as_dict(),
        "response_schema": {
            "type": "object",
            "required": list(DECISION_FIELDS),
            "additionalProperties": False,
            "properties": {field: {"type": "string"} for field in DECISION_FIELDS},
        },
        "expected_decision": expected,
        "reward_function": {
            "eval": [{"evaluator": ROCKET_CHAT_EVALUATOR_NAME, "expected": expected}],
        },
        # No free-standing message carrier is emitted: Rocket.Chat has not
        # yet mapped a surface identity through the Site owner.
        "reader_contract": {
            "auth": "independent_authenticated_ordinary_reader",
            "writer_identity": conversation.writer_user,
            "reader_identity": conversation.reader_user,
            "identities_must_be_distinct": True,
            "resource_kind": ROCKET_CHAT_RESOURCE_KIND,
            "message_identity": "exact_room_thread_actor_message_id",
        },
    }
    validate_rocket_chat_task(task)
    return task


def _validate_response_schema(value: object) -> None:
    if not isinstance(value, Mapping):
        raise RocketChatContractError("Rocket.Chat task response schema is required")
    _require_exact_keys(value, _SCHEMA_KEYS, "response schema")
    if value["type"] != "object":
        raise RocketChatContractError("response schema type must be object")
    if value["required"] != list(DECISION_FIELDS):
        raise RocketChatContractError("response schema required fields must be exact")
    if value["additionalProperties"] is not False:
        raise RocketChatContractError("response schema must forbid additional properties")
    properties = value["properties"]
    if not isinstance(properties, Mapping):
        raise RocketChatContractError("response schema properties must be a mapping")
    if set(properties) != set(DECISION_FIELDS):
        raise RocketChatContractError("response schema properties must be exact")
    for field in DECISION_FIELDS:
        if properties[field] != {"type": "string"}:
            raise RocketChatContractError(f"response schema property {field!r} must be a string")


def _validate_reader_contract(value: object, conversation: RocketChatConversation) -> None:
    if not isinstance(value, Mapping):
        raise RocketChatContractError("Rocket.Chat reader contract is required")
    _require_exact_keys(value, _READER_CONTRACT_KEYS, "reader contract")
    expected = {
        "auth": "independent_authenticated_ordinary_reader",
        "writer_identity": conversation.writer_user,
        "reader_identity": conversation.reader_user,
        "identities_must_be_distinct": True,
        "resource_kind": ROCKET_CHAT_RESOURCE_KIND,
        "message_identity": "exact_room_thread_actor_message_id",
    }
    for key, expected_value in expected.items():
        if value[key] != expected_value:
            raise RocketChatContractError(f"reader contract field {key!r} is inconsistent")
    if conversation.writer_user == conversation.reader_user:
        raise RocketChatContractError("reader contract identities must be distinct")


def _validate_conversation(value: object, expected: RocketChatDecision) -> RocketChatConversation:
    if not isinstance(value, Mapping):
        raise RocketChatContractError("Rocket.Chat task conversation facts are required")
    _require_exact_keys(value, _CONVERSATION_KEYS, "conversation")
    if value["benchmark"] != ROCKET_CHAT_BENCHMARK or value["site"] != ROCKET_CHAT_SITE:
        raise RocketChatContractError("Rocket.Chat task conversation provenance is inconsistent")
    if value["expected_decision"] != expected.as_dict():
        raise RocketChatContractError("Rocket.Chat task expected decision is inconsistent")
    raw_messages = value["messages"]
    raw_corrections = value["corrections"]
    if not isinstance(raw_messages, list) or not isinstance(raw_corrections, list):
        raise RocketChatContractError("conversation messages and corrections must be lists")
    try:
        messages: list[RocketChatMessageFact] = []
        for item in raw_messages:
            if not isinstance(item, Mapping):
                raise RocketChatContractError("conversation message must be a mapping")
            _require_exact_keys(item, _MESSAGE_KEYS, "conversation message")
            messages.append(RocketChatMessageFact(**dict(item)))  # type: ignore[arg-type]
        corrections: list[RocketChatCorrectionFact] = []
        for item in raw_corrections:
            if not isinstance(item, Mapping):
                raise RocketChatContractError("conversation correction must be a mapping")
            _require_exact_keys(item, _CORRECTION_KEYS, "conversation correction")
            corrections.append(RocketChatCorrectionFact(**dict(item)))  # type: ignore[arg-type]
        initial_value = value["initial_decision"]
        if not isinstance(initial_value, Mapping):
            raise RocketChatContractError("conversation initial decision must be a mapping")
        initial = RocketChatDecision.from_mapping(initial_value)
        conversation = RocketChatConversation(
            room_id=value["room_id"],  # type: ignore[arg-type]
            thread_key=value["thread_key"],  # type: ignore[arg-type]
            messages=tuple(messages),
            corrections=tuple(corrections),
            initial_decision=initial,
            writer_user=value["writer_user"],  # type: ignore[arg-type]
            reader_user=value["reader_user"],  # type: ignore[arg-type]
            benchmark=value["benchmark"],  # type: ignore[arg-type]
            site=value["site"],  # type: ignore[arg-type]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RocketChatContractError(
            f"Rocket.Chat task conversation facts are inconsistent: {exc}"
        ) from exc
    if conversation.as_dict() != dict(value):
        raise RocketChatContractError("Rocket.Chat task conversation facts are inconsistent")
    if conversation.expected_decision.as_dict() != expected.as_dict():
        raise RocketChatContractError("Rocket.Chat task conversation decision is inconsistent")
    return conversation


def _validate_reward(value: object, expected: RocketChatDecision) -> None:
    if not isinstance(value, Mapping):
        raise RocketChatContractError("Rocket.Chat task reward function is required")
    _require_exact_keys(value, _REWARD_KEYS, "reward function")
    evaluations = value["eval"]
    if not isinstance(evaluations, list) or len(evaluations) != 1:
        raise RocketChatContractError("Rocket.Chat task requires one feature-owned evaluator")
    evaluator = evaluations[0]
    if not isinstance(evaluator, Mapping):
        raise RocketChatContractError("Rocket.Chat evaluator entry must be a mapping")
    _require_exact_keys(evaluator, _EVAL_KEYS, "evaluator entry")
    if evaluator["evaluator"] != ROCKET_CHAT_EVALUATOR_NAME:
        raise RocketChatContractError("Rocket.Chat task evaluator is unsupported")
    if evaluator["expected"] != expected.as_dict():
        raise RocketChatContractError(
            "Rocket.Chat task evaluator expected decision is inconsistent"
        )


def validate_rocket_chat_task(task: Mapping[str, object]) -> None:
    """Validate every emitted field, including reader and response schemas."""

    if not isinstance(task, Mapping):
        raise RocketChatContractError("Rocket.Chat task must be a mapping")
    _require_exact_keys(task, _TASK_KEYS, "Rocket.Chat task")
    if validate_rocket_chat_benchmark(task["benchmark"]) != ROCKET_CHAT_BENCHMARK:
        raise RocketChatContractError("Rocket.Chat task Benchmark must be TAC")
    if task["site"] != ROCKET_CHAT_SITE:
        raise RocketChatContractError("Rocket.Chat task Site must be rocketchat")
    if task["task_kind"] != ROCKET_CHAT_TASK_KIND:
        raise RocketChatContractError("Rocket.Chat task kind is unsupported")
    if task["task_id"] is not None:
        raise RocketChatContractError("Rocket.Chat response task must remain task-id-less")
    if task["evaluator_authority"] != ROCKET_CHAT_EVALUATOR_AUTHORITY:
        raise RocketChatContractError("Rocket.Chat task evaluator authority is not WARP-local")
    try:
        expected_value = task["expected_decision"]
        if not isinstance(expected_value, Mapping):
            raise RocketChatContractError("expected decision must be a mapping")
        expected = RocketChatDecision.from_mapping(expected_value)
    except (TypeError, ValueError) as exc:
        raise RocketChatContractError(
            f"Rocket.Chat task expected decision is invalid: {exc}"
        ) from exc
    _validate_response_schema(task["response_schema"])
    conversation = _validate_conversation(task["conversation"], expected)
    _validate_reader_contract(task["reader_contract"], conversation)
    _validate_reward(task["reward_function"], expected)
    if task["start_urls"] != [f"__ROCKETCHAT__/channel/{conversation.room_id}"]:
        raise RocketChatContractError("Rocket.Chat task must target its exact generated room")


__all__ = [
    "DECISION_FIELDS",
    "ROCKET_CHAT_BENCHMARK",
    "ROCKET_CHAT_EVALUATOR_AUTHORITY",
    "ROCKET_CHAT_EVALUATOR_NAME",
    "ROCKET_CHAT_RESOURCE_KIND",
    "ROCKET_CHAT_SITE",
    "ROCKET_CHAT_TASK_KIND",
    "RocketChatContractError",
    "RocketChatConversation",
    "RocketChatCorrectionFact",
    "RocketChatDecision",
    "RocketChatMessageFact",
    "compile_rocket_chat_task",
    "derive_rocket_chat_decision",
    "generate_rocket_chat_conversation",
    "infer_rocket_chat_benchmark",
    "resolve_rocket_chat_evaluator_authority",
    "validate_rocket_chat_benchmark",
    "validate_rocket_chat_task",
]
