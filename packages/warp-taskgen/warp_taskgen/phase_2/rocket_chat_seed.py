"""Typed Rocket.Chat seed construction and post-fill validation."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from warp_taskgen.phase_1.rocket_chat_contracts import RocketChatDecision
from warp_taskgen.phase_1.rocket_chat_decisions import _validate_conversation
from warp_taskgen.phase_2.rocket_chat_common import (
    ROCKET_CHAT_BENCHMARK,
    ROCKET_CHAT_DELIVERY_METHOD,
    ROCKET_CHAT_SITE,
    composition_supports_rocket_chat,
)
from warp_taskgen.phase_2.text_fill.constants import PAYLOAD_PLACEHOLDER
from warp_taskgen.phase_2.text_fill.seed import (
    materialize_adversarial_seed,
    validate_seed_template_contract,
)
from warp_taskgen.runtime_composition import RuntimeComposition


def materialize_seed_template(
    contract: Mapping[str, Any],
    *,
    runtime_composition: RuntimeComposition | None,
) -> dict[str, Any]:
    """Build one editor call whose typed conversation has one body placeholder."""

    if (
        not isinstance(contract, Mapping)
        or contract.get("eligibility", {}).get("status") != "eligible"
    ):
        raise ValueError("cannot materialize seed_template from ineligible Rocket.Chat contract")
    if not composition_supports_rocket_chat(
        runtime_composition,
        benchmark=ROCKET_CHAT_BENCHMARK,
        site=ROCKET_CHAT_SITE,
    ):
        raise ValueError(
            "Rocket.Chat seed materialization requires an explicit runtime composition"
        )
    args = contract.get("editor_args_template")
    if not isinstance(args, Mapping) or set(args) != {"conversation"}:
        raise ValueError("Rocket.Chat exposure contract must carry one conversation argument")
    conversation = contract.get("conversation")
    seed = {
        "mechanism": "editor",
        "render_signature": _correction_body(conversation),
        "editor_calls": [
            {
                "benchmark": ROCKET_CHAT_BENCHMARK,
                "site": ROCKET_CHAT_SITE,
                "method": ROCKET_CHAT_DELIVERY_METHOD,
                "args": copy.deepcopy(dict(args)),
            }
        ],
    }
    validate_rocket_chat_seed_template(seed, benign_conversation=conversation)
    return seed


def validate_rocket_chat_seed_template(
    seed_template: Mapping[str, Any],
    *,
    benign_conversation: Mapping[str, Any] | None = None,
) -> None:
    """Prove the sole transformation is ``correction.body += placeholder``."""

    if not isinstance(seed_template, Mapping):
        raise ValueError("Rocket.Chat seed_template must be an object")
    validate_seed_template_contract(dict(seed_template))
    if seed_template.get("mechanism") != "editor":
        raise ValueError("Rocket.Chat seed_template mechanism must be editor")
    calls = seed_template.get("editor_calls")
    if not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], Mapping):
        raise ValueError("Rocket.Chat seed_template must contain exactly one editor call")
    call = calls[0]
    if (
        call.get("benchmark") != ROCKET_CHAT_BENCHMARK
        or call.get("site") != ROCKET_CHAT_SITE
        or call.get("method") != ROCKET_CHAT_DELIVERY_METHOD
    ):
        raise ValueError("Rocket.Chat seed_template editor identity is inconsistent")
    args = call.get("args")
    conversation = args.get("conversation") if isinstance(args, Mapping) else None
    if not isinstance(args, Mapping) or set(args) != {"conversation"}:
        raise ValueError("Rocket.Chat seed_template editor args must be conversation-only")
    if not isinstance(conversation, Mapping) or _count_placeholder(conversation) != 1:
        raise ValueError("Rocket.Chat conversation must contain exactly one payload placeholder")
    messages = conversation.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError("Rocket.Chat conversation must contain exactly 3 messages")
    keys = [message.get("logical_key") for message in messages if isinstance(message, Mapping)]
    if len(keys) != 3 or len(set(keys)) != 3:
        raise ValueError("Rocket.Chat conversation message logical_key values must be unique")
    if set(keys) != {"plan", "update", "correction"}:
        raise ValueError(
            "Rocket.Chat conversation must contain plan, update, and correction messages"
        )
    correction = next(
        (
            message
            for message in messages
            if isinstance(message, Mapping) and message.get("logical_key") == "correction"
        ),
        None,
    )
    if not isinstance(correction, Mapping) or not isinstance(correction.get("body"), str):
        raise ValueError("Rocket.Chat correction message is required")
    correction_body = correction["body"]
    if not correction_body.endswith(f" {PAYLOAD_PLACEHOLDER}"):
        raise ValueError("Rocket.Chat payload must append to the correction body")
    if _count_placeholder(correction_body) != 1:
        raise ValueError("Rocket.Chat correction body must contain one payload placeholder")
    if benign_conversation is None:
        return
    expected = copy.deepcopy(dict(benign_conversation))
    expected_messages = expected.get("messages")
    if not isinstance(expected_messages, list) or len(expected_messages) != 3:
        raise ValueError("Rocket.Chat benign conversation must contain exactly 3 messages")
    expected_correction = next(
        (
            message
            for message in expected_messages
            if isinstance(message, Mapping) and message.get("logical_key") == "correction"
        ),
        None,
    )
    if not isinstance(expected_correction, Mapping) or not isinstance(
        expected_correction.get("body"), str
    ):
        raise ValueError("Rocket.Chat benign correction body is required")
    actual = copy.deepcopy(dict(conversation))
    actual_messages = actual["messages"]
    actual_correction = next(
        message
        for message in actual_messages
        if isinstance(message, Mapping) and message.get("logical_key") == "correction"
    )
    actual_correction["body"] = actual_correction["body"][: -len(f" {PAYLOAD_PLACEHOLDER}")]
    if actual != expected:
        raise ValueError(
            "Rocket.Chat seed must preserve the typed conversation except for the appended payload"
        )


def materialize_adversarial_seed_for_runtime(
    seed_template: Mapping[str, Any],
    rendered_payload: str,
    *,
    runtime_composition: RuntimeComposition | None,
) -> dict[str, Any]:
    """Replace the placeholder and ensure the resulting call remains typed."""

    if not isinstance(rendered_payload, str) or not rendered_payload.strip():
        raise ValueError("Rocket.Chat rendered payload must be non-empty text")
    if "\n" in rendered_payload or "\r" in rendered_payload:
        raise ValueError("Rocket.Chat rendered payload must be single-line text")
    if not composition_supports_rocket_chat(
        runtime_composition,
        benchmark=ROCKET_CHAT_BENCHMARK,
        site=ROCKET_CHAT_SITE,
    ):
        raise ValueError(
            "Rocket.Chat seed materialization requires an explicit runtime composition"
        )
    validate_rocket_chat_seed_template(seed_template)
    materialized = materialize_adversarial_seed(dict(seed_template), rendered_payload)
    calls = materialized.get("editor_calls")
    conversation = calls[0].get("args", {}).get("conversation") if calls else None
    if not isinstance(conversation, Mapping) or _count_placeholder(conversation):
        raise ValueError("Rocket.Chat materialized seed retained no typed placeholder")
    # The typed envelope validator catches duplicate keys, wrong hosts, and
    # malformed message fields after generic text replacement.
    expected = conversation.get("expected_decision")
    if not isinstance(expected, Mapping):
        raise ValueError("Rocket.Chat materialized conversation expected_decision is required")
    _validate_conversation(conversation, RocketChatDecision.from_mapping(expected))
    return materialized


def _conversation_with_placeholder(conversation: dict[str, Any]) -> dict[str, Any]:
    messages = conversation.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError("Rocket.Chat conversation must contain exactly 3 messages")
    correction = [
        message
        for message in messages
        if isinstance(message, Mapping) and message.get("logical_key") == "correction"
    ]
    if len(correction) != 1 or not isinstance(correction[0].get("body"), str):
        raise ValueError("Rocket.Chat correction message is required")
    body = correction[0]["body"]
    if body.endswith(PAYLOAD_PLACEHOLDER) or len(body) + len(PAYLOAD_PLACEHOLDER) + 1 > 2000:
        raise ValueError("Rocket.Chat correction body cannot carry one payload placeholder")
    correction[0]["body"] = f"{body} {PAYLOAD_PLACEHOLDER}"
    return conversation


def _correction_body(conversation: object) -> str:
    if not isinstance(conversation, Mapping):
        raise ValueError("Rocket.Chat exposure contract is missing its typed conversation")
    messages = conversation.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Rocket.Chat exposure contract is missing conversation messages")
    for message in messages:
        if isinstance(message, Mapping) and message.get("logical_key") == "correction":
            body = message.get("body")
            if isinstance(body, str) and body.strip():
                return body
    raise ValueError("Rocket.Chat exposure contract is missing correction body")


def _count_placeholder(value: Any) -> int:
    if isinstance(value, str):
        return value.count(PAYLOAD_PLACEHOLDER)
    if isinstance(value, Mapping):
        return sum(_count_placeholder(item) for item in value.values())
    if isinstance(value, list):
        return sum(_count_placeholder(item) for item in value)
    return 0


__all__ = [
    "_conversation_with_placeholder",
    "materialize_adversarial_seed_for_runtime",
    "materialize_seed_template",
    "validate_rocket_chat_seed_template",
]
