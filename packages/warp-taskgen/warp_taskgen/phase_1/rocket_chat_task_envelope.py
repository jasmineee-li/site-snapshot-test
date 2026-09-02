"""WARP task envelope for the Rocket.Chat conversation-decision feature.

The strict static compiler remains the authority for generated conversation
facts and finite grading.  This module owns only the adapter into WARP's
ordinary task, seed, Phase 2, and Phase 4 shape.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping

from warp_taskgen.phase_1.rocket_chat_contracts import (
    ROCKET_CHAT_BENCHMARK,
    ROCKET_CHAT_SEED_METHOD,
    ROCKET_CHAT_SITE,
    RocketChatContractError,
    RocketChatConversation,
    _identity,
    _text,
)
from warp_taskgen.phase_1.rocket_chat_decisions import (
    compile_rocket_chat_task,
    validate_rocket_chat_task,
)

ROCKET_CHAT_CONTRACT_FIELD = "rocket_chat_contract"

_STATIC_TASK_KEYS = frozenset(
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
_ENVELOPE_REQUIRED_KEYS = frozenset(
    {
        "id",
        "origin",
        "benchmark",
        "site",
        "sites",
        "instruction",
        "start_urls",
        "data_seed",
        "reward_function",
        ROCKET_CHAT_CONTRACT_FIELD,
    }
)


def project_rocket_chat_static_contract(
    task: Mapping[str, object],
) -> Mapping[str, object]:
    """Return and validate the exact static contract from either task shape."""

    if not isinstance(task, Mapping):
        raise RocketChatContractError("Rocket.Chat task must be a mapping")
    if set(task) == _STATIC_TASK_KEYS:
        validate_rocket_chat_task(task)
        return task
    nested = task.get(ROCKET_CHAT_CONTRACT_FIELD)
    if not isinstance(nested, Mapping):
        raise RocketChatContractError(
            f"Rocket.Chat WARP task requires {ROCKET_CHAT_CONTRACT_FIELD!r}"
        )
    validate_rocket_chat_task(nested)
    return nested


def _expected_seed(static: Mapping[str, object]) -> dict[str, object]:
    conversation = static["conversation"]
    messages = conversation.get("messages") if isinstance(conversation, Mapping) else None
    if not isinstance(messages, list):
        raise RocketChatContractError("Rocket.Chat WARP seed requires conversation messages")
    correction_body = next(
        (
            message.get("body")
            for message in messages
            if isinstance(message, Mapping)
            and message.get("logical_key") == "correction"
            and isinstance(message.get("body"), str)
        ),
        None,
    )
    if not isinstance(correction_body, str) or not correction_body.strip():
        raise RocketChatContractError("Rocket.Chat WARP seed requires a correction signature")
    return {
        "mechanism": "editor",
        # The editor consumes nested generated facts, while the generic render
        # gate needs one explicit visible string tied to this single call.
        "render_signature": correction_body,
        "editor_calls": [
            {
                "benchmark": ROCKET_CHAT_BENCHMARK,
                "site": ROCKET_CHAT_SITE,
                "method": ROCKET_CHAT_SEED_METHOD,
                "args": {"conversation": copy.deepcopy(static["conversation"])},
            }
        ],
    }


def validate_rocket_chat_benign_task(task: Mapping[str, object]) -> None:
    """Validate the feature's cross-phase envelope and its static projection."""

    if not isinstance(task, Mapping):
        raise RocketChatContractError("Rocket.Chat WARP task must be a mapping")
    missing = _ENVELOPE_REQUIRED_KEYS - set(task)
    if missing:
        raise RocketChatContractError(
            "Rocket.Chat WARP task is missing fields: "
            + ", ".join(sorted(str(field) for field in missing))
        )
    _identity(task["id"], field="WARP task id")
    _text(task["instruction"], field="WARP task instruction", max_length=1000)
    if task["origin"] != "new_task":
        raise RocketChatContractError("Rocket.Chat WARP task origin must be new_task")
    if task["benchmark"] != ROCKET_CHAT_BENCHMARK:
        raise RocketChatContractError("Rocket.Chat WARP task Benchmark is inconsistent")
    if task["site"] != ROCKET_CHAT_SITE or task["sites"] != [ROCKET_CHAT_SITE]:
        raise RocketChatContractError("Rocket.Chat WARP task Site fields are inconsistent")
    if task.get("task_id") not in (None, ""):
        raise RocketChatContractError("Rocket.Chat WARP task cannot carry a native task_id")

    static = project_rocket_chat_static_contract(task)
    if task["start_urls"] != static["start_urls"]:
        raise RocketChatContractError("Rocket.Chat WARP task start URLs drifted")
    if task["reward_function"] != static["reward_function"]:
        raise RocketChatContractError("Rocket.Chat WARP task reward drifted")
    if task["data_seed"] != _expected_seed(static):
        raise RocketChatContractError("Rocket.Chat WARP task seed drifted")


def compile_rocket_chat_benign_task(
    conversation: RocketChatConversation,
    *,
    task_id: str,
    instruction: str,
) -> dict[str, object]:
    """Compile one generated conversation into the normal WARP task shape."""

    stable_id = _identity(task_id, field="WARP task id")
    task_instruction = _text(
        instruction,
        field="WARP task instruction",
        max_length=1000,
    )
    static = compile_rocket_chat_task(conversation)
    task: dict[str, object] = {
        "id": stable_id,
        "origin": "new_task",
        "benchmark": ROCKET_CHAT_BENCHMARK,
        "site": ROCKET_CHAT_SITE,
        "sites": [ROCKET_CHAT_SITE],
        "instruction": task_instruction,
        "start_urls": copy.deepcopy(static["start_urls"]),
        "data_seed": _expected_seed(static),
        "reward_function": copy.deepcopy(static["reward_function"]),
        ROCKET_CHAT_CONTRACT_FIELD: copy.deepcopy(static),
    }
    validate_rocket_chat_benign_task(task)
    return task


__all__ = [
    "ROCKET_CHAT_CONTRACT_FIELD",
    "compile_rocket_chat_benign_task",
    "project_rocket_chat_static_contract",
    "validate_rocket_chat_benign_task",
]
