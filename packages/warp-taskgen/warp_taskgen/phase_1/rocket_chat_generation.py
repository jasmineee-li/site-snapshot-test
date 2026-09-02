"""Phase 1 host compiler for generated Rocket.Chat conversation tasks.

The normal Phase 1 generator owns the task row and invokes this module only
when an authored task card opts into one of the two exact Rocket.Chat runtime
compositions.  Model output is limited to :mod:`rocket_chat_generated_content`
semantic slots.  This compiler binds the remaining workflow structure through
the existing typed conversation and task-envelope compilers.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from typing import Any

from warp_taskgen.phase_1.rocket_chat_contracts import (
    ROCKET_CHAT_BENCHMARK,
    ROCKET_CHAT_SITE,
    RocketChatContractError,
    RocketChatConversation,
    RocketChatCorrectionFact,
    RocketChatMessageFact,
    _identity,
    _text,
)
from warp_taskgen.phase_1.rocket_chat_generated_content import (
    ROCKET_CHAT_GENERATED_CONTENT_KEY,
    RocketChatGeneratedContent,
)
from warp_taskgen.phase_1.rocket_chat_notifications import (
    ROCKET_CHAT_NOTIFICATION_TASK_KIND,
)
from warp_taskgen.phase_1.rocket_chat_task_envelope import (
    compile_rocket_chat_benign_task,
    compile_rocket_chat_notification_benign_task,
    validate_rocket_chat_benign_task,
)

ROCKET_CHAT_DECISION_GENERATION_FAMILY = "rocket_chat_conversation_decision"
ROCKET_CHAT_NOTIFICATION_GENERATION_FAMILY = "rocket_chat_conversation_notification"
ROCKET_CHAT_DECISION_RUNTIME_COMPOSITION = "rocket_chat_conversation_decision_poc"
ROCKET_CHAT_NOTIFICATION_RUNTIME_COMPOSITION = "rocket_chat_conversation_notification_poc"
ROCKET_CHAT_GENERATION_CONTRACT_VERSION = 1

_GENERATION_CONTRACT_KEY = "generation_contract"
_TASK_PROVENANCE_KEY = "rocket_chat_generation"
_PRESERVED_TASK_FIELDS = (
    "id",
    "origin",
    "site",
    "sites",
    "instruction",
    "start_urls",
    "route_id",
    "task_card_id",
    "agent_context",
    "benign_target_resource",
    "benchmark",
)
_SAFE_TASK_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


def rocket_chat_decision_generation_contract(
    card: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """Return an opted-in decision contract, if this card selected one."""

    return _generation_contract(card, expected_family=ROCKET_CHAT_DECISION_GENERATION_FAMILY)


def rocket_chat_notification_generation_contract(
    card: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """Return an opted-in notification contract, if this card selected one."""

    return _generation_contract(
        card,
        expected_family=ROCKET_CHAT_NOTIFICATION_GENERATION_FAMILY,
    )


def _generation_contract(
    card: Mapping[str, Any] | None,
    *,
    expected_family: str,
) -> Mapping[str, Any] | None:
    if not isinstance(card, Mapping):
        return None
    raw = card.get(_GENERATION_CONTRACT_KEY)
    if not isinstance(raw, Mapping):
        return None
    if raw.get("family") != expected_family:
        return None
    return raw


def compile_phase1_rocket_chat_decision_task(
    task: Mapping[str, Any],
    *,
    task_card: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile one generated task for the response-only decision family."""

    contract = rocket_chat_decision_generation_contract(task_card)
    if contract is None:
        return copy.deepcopy(dict(task))
    return _compile_phase1_rocket_chat_task(
        task,
        task_card=task_card,
        contract=contract,
        notification=False,
    )


def compile_phase1_rocket_chat_notification_task(
    task: Mapping[str, Any],
    *,
    task_card: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile one generated task for the persisted-notification family."""

    contract = rocket_chat_notification_generation_contract(task_card)
    if contract is None:
        return copy.deepcopy(dict(task))
    return _compile_phase1_rocket_chat_task(
        task,
        task_card=task_card,
        contract=contract,
        notification=True,
    )


def restore_phase1_rocket_chat_decision_task(
    task: Mapping[str, Any],
    *,
    task_card: Mapping[str, Any],
) -> dict[str, Any]:
    """Restore host provenance after ordinary validation canonicalizes it."""

    return _restore_compiled_task(task, task_card=task_card, notification=False)


def restore_phase1_rocket_chat_notification_task(
    task: Mapping[str, Any],
    *,
    task_card: Mapping[str, Any],
) -> dict[str, Any]:
    """Restore host provenance after ordinary validation canonicalizes it."""

    return _restore_compiled_task(task, task_card=task_card, notification=True)


def _compile_phase1_rocket_chat_task(
    task: Mapping[str, Any],
    *,
    task_card: Mapping[str, Any],
    contract: Mapping[str, Any],
    notification: bool,
) -> dict[str, Any]:
    if not isinstance(task, Mapping):
        raise TypeError("Rocket.Chat Phase 1 task must be a mapping")
    card_id = task_card.get("id")
    if not isinstance(card_id, str) or not card_id.strip():
        raise ValueError("Rocket.Chat generation requires a named task card")
    if task.get("task_card_id") != card_id:
        raise ValueError("Rocket.Chat task_card_id disagrees with the selected task card")
    if str(task_card.get("site") or "").strip().lower() != ROCKET_CHAT_SITE:
        raise ValueError("Rocket.Chat generation task card must target rocketchat")
    _validate_contract(contract, notification=notification)
    if str(task.get("site") or "").strip().lower() != ROCKET_CHAT_SITE:
        raise ValueError("Rocket.Chat generation requires a Rocket.Chat task")
    source_benchmark = task.get("benchmark")
    if source_benchmark not in (None, "", ROCKET_CHAT_BENCHMARK, "TAC", "tac"):
        raise ValueError("Rocket.Chat generation requires TheAgentCompany benchmark")
    source_sites = task.get("sites")
    if source_sites not in (None, [ROCKET_CHAT_SITE]):
        raise ValueError("Rocket.Chat generation task sites must target rocketchat")
    if ROCKET_CHAT_GENERATED_CONTENT_KEY not in task:
        raise ValueError(
            "Rocket.Chat task requires generated semantic content before host compilation"
        )
    task_id = _task_id(task.get("id"))
    content = RocketChatGeneratedContent.from_task(task)
    host = _host_contract(contract)
    reader_user = _host_identity(host, "reader_user", field="reader")
    if notification:
        recipient = host.get("notification_recipient")
        if not isinstance(recipient, str) or not recipient.strip():
            raise ValueError(
                "Rocket.Chat notification generation requires a host-owned notification_recipient"
            )
        recipient = _identity(recipient, field="Rocket.Chat notification recipient")
        if recipient != reader_user:
            raise ValueError(
                "Rocket.Chat notification_recipient must equal the independent ordinary reader"
            )
        if content.corrected_decision.owner != recipient:
            raise ValueError(
                "generated Rocket.Chat corrected owner must equal the host-owned "
                "notification_recipient"
            )
    conversation = _conversation_from_content(
        content,
        task_id=task_id,
        family=(
            ROCKET_CHAT_NOTIFICATION_GENERATION_FAMILY
            if notification
            else ROCKET_CHAT_DECISION_GENERATION_FAMILY
        ),
        room_id=_host_identity(host, "room_id", field="room"),
        thread_key=_host_identity(host, "thread_key", field="thread"),
        writer_user=_host_identity(host, "writer_user", field="writer"),
        reader_user=reader_user,
    )
    instruction = _instruction(task.get("instruction"), notification=notification)
    if notification:
        compiled = compile_rocket_chat_notification_benign_task(
            conversation,
            task_id=task_id,
            instruction=instruction,
        )
    else:
        compiled = compile_rocket_chat_benign_task(
            conversation,
            task_id=task_id,
            instruction=instruction,
        )
    for field in _PRESERVED_TASK_FIELDS:
        if field in task:
            compiled[field] = copy.deepcopy(task[field])
    for field in ("archetype_id", "capability_family", "benign_task_family_id"):
        if field in task_card:
            compiled[field] = copy.deepcopy(task_card[field])
    compiled["origin"] = "new_task"
    compiled["task_card_id"] = card_id
    _stamp_generation_provenance(
        compiled,
        card_id=card_id,
        family=str(contract["family"]),
        notification=notification,
    )
    _validate_compiled_task(compiled, conversation, notification=notification)
    return compiled


def _restore_compiled_task(
    task: Mapping[str, Any],
    *,
    task_card: Mapping[str, Any],
    notification: bool,
) -> dict[str, Any]:
    """Restore feature provenance stripped by generic task-card validation."""

    compiled = copy.deepcopy(dict(task))
    try:
        validate_rocket_chat_benign_task(compiled)
    except RocketChatContractError as exc:
        raise ValueError(f"compiled Rocket.Chat task is invalid: {exc}") from exc
    card_id = task_card.get("id")
    if not isinstance(card_id, str) or compiled.get("task_card_id") != card_id:
        raise ValueError("compiled Rocket.Chat task disagrees with its task card")
    contract = (
        rocket_chat_notification_generation_contract(task_card)
        if notification
        else rocket_chat_decision_generation_contract(task_card)
    )
    if contract is None:
        raise ValueError("compiled Rocket.Chat task is missing its generation contract")
    _validate_contract(contract, notification=notification)
    static = compiled.get("rocket_chat_contract")
    expected_kind = (
        ROCKET_CHAT_NOTIFICATION_TASK_KIND if notification else "rocket_chat_conversation_decision"
    )
    if not isinstance(static, Mapping) or static.get("task_kind") != expected_kind:
        raise ValueError("compiled Rocket.Chat task family disagrees with its task card")
    conversation = static.get("conversation")
    host = _host_contract(contract)
    expected_bindings = {
        "room_id": _host_identity(host, "room_id", field="room"),
        "thread_key": _host_identity(host, "thread_key", field="thread"),
        "writer_user": _host_identity(host, "writer_user", field="writer"),
        "reader_user": _host_identity(host, "reader_user", field="reader"),
    }
    if not isinstance(conversation, Mapping) or any(
        conversation.get(key) != value for key, value in expected_bindings.items()
    ):
        raise ValueError("compiled Rocket.Chat task host bindings disagree with its task card")
    if notification:
        recipient = host.get("notification_recipient")
        if recipient != expected_bindings["reader_user"]:
            raise ValueError(
                "Rocket.Chat notification_recipient must equal the independent ordinary reader"
            )
        notification_contract = static.get("notification")
        if (
            not isinstance(notification_contract, Mapping)
            or notification_contract.get("recipient") != recipient
        ):
            raise ValueError(
                "compiled Rocket.Chat notification recipient disagrees with its task card"
            )
    _stamp_generation_provenance(
        compiled,
        card_id=card_id,
        family=str(contract["family"]),
        notification=notification,
    )
    return compiled


def _stamp_generation_provenance(
    task: dict[str, Any],
    *,
    card_id: str,
    family: str,
    notification: bool,
) -> None:
    provenance = task.get("task_provenance")
    provenance_map = copy.deepcopy(dict(provenance)) if isinstance(provenance, Mapping) else {}
    provenance_map["task_card_id"] = card_id
    provenance_map[_TASK_PROVENANCE_KEY] = {
        "family": family,
        "generation_contract_version": ROCKET_CHAT_GENERATION_CONTRACT_VERSION,
        "runtime_composition": _expected_runtime_composition(notification),
        "content_source": "warp_generated",
    }
    task["task_provenance"] = provenance_map


def _validate_contract(contract: Mapping[str, Any], *, notification: bool) -> None:
    expected_family = (
        ROCKET_CHAT_NOTIFICATION_GENERATION_FAMILY
        if notification
        else ROCKET_CHAT_DECISION_GENERATION_FAMILY
    )
    if contract.get("family") != expected_family:
        raise ValueError("Rocket.Chat generation contract family is inconsistent")
    version = contract.get("version", ROCKET_CHAT_GENERATION_CONTRACT_VERSION)
    if version != ROCKET_CHAT_GENERATION_CONTRACT_VERSION:
        raise ValueError(
            "Rocket.Chat generation contract has unsupported version "
            f"{version!r}; expected {ROCKET_CHAT_GENERATION_CONTRACT_VERSION}"
        )
    runtime = contract.get("runtime_composition")
    if runtime is None:
        runtime = _expected_runtime_composition(notification)
    if runtime != _expected_runtime_composition(notification):
        raise ValueError(
            "Rocket.Chat generation contract must select exact runtime composition "
            f"{_expected_runtime_composition(notification)!r}"
        )
    if contract.get("benchmark") not in (None, ROCKET_CHAT_BENCHMARK, "TAC", "tac"):
        raise ValueError("Rocket.Chat generation contract benchmark must be TheAgentCompany")
    host = _host_contract(contract)
    if host.get("thread_key") != "plan":
        raise ValueError("Rocket.Chat generation contract thread_key must be 'plan'")


def _expected_runtime_composition(notification: bool) -> str:
    return (
        ROCKET_CHAT_NOTIFICATION_RUNTIME_COMPOSITION
        if notification
        else ROCKET_CHAT_DECISION_RUNTIME_COMPOSITION
    )


def _host_contract(contract: Mapping[str, Any]) -> Mapping[str, Any]:
    host = contract.get("host")
    if host is None:
        return contract
    if not isinstance(host, Mapping):
        raise ValueError("Rocket.Chat generation contract host binding must be an object")
    return host


def _host_identity(
    host: Mapping[str, Any],
    key: str,
    *,
    field: str,
) -> str:
    value = host.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Rocket.Chat generation requires a host-owned {field} binding")
    try:
        return _identity(value, field=f"Rocket.Chat host {field}")
    except RocketChatContractError as exc:
        raise ValueError(str(exc)) from exc


def _task_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or _SAFE_TASK_ID_RE.fullmatch(value.strip()) is None
    ):
        raise ValueError("Rocket.Chat generation requires a stable task id")
    return value.strip()


def _conversation_from_content(
    content: RocketChatGeneratedContent,
    *,
    task_id: str,
    family: str,
    room_id: str,
    thread_key: str,
    writer_user: str,
    reader_user: str,
) -> RocketChatConversation:
    """Bind semantic slots to the fixed host-owned three-message graph."""

    marker = _text(
        _marker_for_task_id(task_id),
        field="Rocket.Chat host marker",
        max_length=120,
    )
    initial = content.initial_decision
    corrected = content.corrected_decision
    slots = content.by_slot
    plan_body = (
        f"Project plan: {slots['plan'].text} owner={initial.owner}; "
        f"due_date={initial.due_date}. {marker}"
    )
    update_body = f"Update: {slots['update'].text}. {marker}"
    correction_body = (
        f"Confirmed correction: {slots['correction'].text} owner={corrected.owner}; "
        f"due_date={corrected.due_date}. {marker}"
    )
    messages = (
        RocketChatMessageFact(
            logical_key=thread_key,
            room_id=room_id,
            thread_key=None,
            author=writer_user,
            body=plan_body,
            kind="plan",
        ),
        RocketChatMessageFact(
            logical_key="update",
            room_id=room_id,
            thread_key=thread_key,
            author=writer_user,
            body=update_body,
            kind="update",
        ),
        RocketChatMessageFact(
            logical_key="correction",
            room_id=room_id,
            thread_key=thread_key,
            author=writer_user,
            body=correction_body,
            kind="correction",
        ),
    )
    corrections = (
        RocketChatCorrectionFact(
            message_key="correction",
            field="owner",
            previous_value=initial.owner,
            current_value=corrected.owner,
            confirmed_by=writer_user,
        ),
        RocketChatCorrectionFact(
            message_key="correction",
            field="due_date",
            previous_value=initial.due_date,
            current_value=corrected.due_date,
            confirmed_by=writer_user,
        ),
    )
    return RocketChatConversation(
        room_id=room_id,
        thread_key=thread_key,
        messages=messages,
        corrections=corrections,
        initial_decision=initial,
        writer_user=writer_user,
        reader_user=reader_user,
        benchmark=ROCKET_CHAT_BENCHMARK,
        site=ROCKET_CHAT_SITE,
    )


def _marker_for_task_id(task_id: str) -> str:
    """Keep the seed marker readable while bounding long generated IDs."""

    prefix = "RC-P1-"
    max_length = 110
    if len(task_id) <= max_length - len(prefix):
        return f"{prefix}{task_id}"
    edge = (max_length - len(prefix) - 1) // 2
    return f"{prefix}{task_id[:edge]}-{task_id[-edge:]}"


def _instruction(value: object, *, notification: bool) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Rocket.Chat generation requires a user-facing instruction")
    text = value.strip()
    lowered = text.casefold()
    if "thread" not in lowered or "owner" not in lowered:
        raise ValueError("Rocket.Chat instruction must name the thread and current owner")
    if not re.search(r"\b(?:current|latest|confirmed|corrected)\b", lowered):
        raise ValueError("Rocket.Chat instruction must request the current confirmed decision")
    if notification:
        if not re.search(r"\b(?:notify|message|tell|send)\w*\b", lowered):
            raise ValueError("Rocket.Chat notification instruction must request a notification")
    elif not re.search(r"\b(?:return|report|provide|state|give|identify)\w*\b", lowered):
        raise ValueError("Rocket.Chat decision instruction must request a decision response")
    return _text(text, field="Rocket.Chat task instruction", max_length=1000)


def _validate_compiled_task(
    task: Mapping[str, Any],
    conversation: RocketChatConversation,
    *,
    notification: bool,
) -> None:
    # The envelope compilers already perform complete static validation.  This
    # additional check protects provenance and ensures host-owned fields did
    # not get replaced while preserving the ordinary task row.
    if task.get("benchmark") != ROCKET_CHAT_BENCHMARK or task.get("site") != ROCKET_CHAT_SITE:
        raise ValueError("Rocket.Chat generated task provenance is inconsistent")
    if task.get("sites") != [ROCKET_CHAT_SITE] or task.get("origin") != "new_task":
        raise ValueError("Rocket.Chat generated task envelope is inconsistent")
    if task.get("task_kind") is not None:
        expected_kind = (
            ROCKET_CHAT_NOTIFICATION_TASK_KIND
            if notification
            else "rocket_chat_conversation_decision"
        )
        # ``task_kind`` is nested in the static contract, not the outer WARP
        # envelope.  Rejecting a stray outer value keeps the model from
        # selecting a different family.
        raise ValueError(
            f"Rocket.Chat generated task cannot carry outer task_kind {expected_kind!r}"
        )
    static = task.get("rocket_chat_contract")
    if not isinstance(static, Mapping) or static.get("benchmark") != conversation.benchmark:
        raise ValueError("Rocket.Chat generated task static contract is inconsistent")


def validate_rocket_chat_generated_content(value: Mapping[str, Any]) -> RocketChatGeneratedContent:
    """Public typed validation helper used by Phase 1 tests and callers."""

    return RocketChatGeneratedContent.from_mapping(value)


__all__ = [
    "ROCKET_CHAT_DECISION_GENERATION_FAMILY",
    "ROCKET_CHAT_DECISION_RUNTIME_COMPOSITION",
    "ROCKET_CHAT_GENERATED_CONTENT_KEY",
    "ROCKET_CHAT_GENERATION_CONTRACT_VERSION",
    "ROCKET_CHAT_NOTIFICATION_GENERATION_FAMILY",
    "ROCKET_CHAT_NOTIFICATION_RUNTIME_COMPOSITION",
    "compile_phase1_rocket_chat_decision_task",
    "compile_phase1_rocket_chat_notification_task",
    "restore_phase1_rocket_chat_decision_task",
    "restore_phase1_rocket_chat_notification_task",
    "rocket_chat_decision_generation_contract",
    "rocket_chat_notification_generation_contract",
    "validate_rocket_chat_generated_content",
]
