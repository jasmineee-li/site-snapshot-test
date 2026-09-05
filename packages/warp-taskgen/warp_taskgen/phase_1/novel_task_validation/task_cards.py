"""Task-card and capability alignment validation exports."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from typing import Any

from warp_taskgen.adversarial_actions.benign_action_contracts import (
    apply_benign_action_contract,
    finalize_host_action_only_benign_reward,
)
from warp_taskgen.adversarial_actions.capability_contracts import BENIGN_REWARD_HOST_ACTION_ONLY
from warp_taskgen.adversarial_actions.scenario_templates import scenario_template_from_mapping
from warp_taskgen.phase_1.generated_workflows import owns_host_action_contract
from warp_taskgen.phase_1.novel_task_validation.answer_stability import (
    _EXACT_BINARY_LABEL_VERB_RE,
    _flatten_string_values,
)
from warp_taskgen.phase_1.novel_task_validation.errors import (
    GeneratedTaskValidationError,
    _field_error,
)
from warp_taskgen.phases.phase_1_task_cards import (
    card_action_kinds,
    card_benign_reward_shape,
    card_benign_task_family_ids,
    card_capability_family,
    card_control_action_kinds,
    card_route_ids,
    card_string_list,
)

_MODEL_AUTHORED_HOST_METADATA_FIELDS = (
    "allowed_action_kinds",
    "action_target_contract",
    "action_target_contracts",
    "benign_reward_shape",
    "compatible_action_kinds",
    "control_action_kinds",
    "precondition_slot",
    "scenario_context",
    "scenario_template",
    "scenario_template_id",
    "task_archetype",
    "task_card",
    "task_provenance",
)


def _validate_task_card_alignment(
    task: dict[str, Any],
    *,
    index: int,
    site_name: str,
    card_index: dict[str, dict[str, Any]],
    route_index: dict[str, dict[str, Any]] | None,
    preserved_benign_action_contract: Mapping[str, Any] | None = None,
) -> GeneratedTaskValidationError | None:
    card_id = task.get("task_card_id")
    if not isinstance(card_id, str) or not card_id.strip():
        return _field_error(
            index,
            "MISSING_TASK_CARD_ID",
            "task_card_id",
            "task-card-guided generation requires every task to name task_card_id",
            expected=sorted(card_index),
            actual=card_id,
        )
    card = card_index.get(card_id)
    if card is None:
        return _field_error(
            index,
            "UNKNOWN_TASK_CARD_ID",
            "task_card_id",
            "task_card_id is not present in the active task-card plan",
            expected=sorted(card_index),
            actual=card_id,
        )
    if card.get("site") != site_name:
        return _field_error(
            index,
            "TASK_CARD_SITE_MISMATCH",
            "task_card_id",
            "task card belongs to a different site",
            expected=site_name,
            actual=card.get("site"),
        )
    route_id = task.get("route_id")
    allowed_route_ids = card_route_ids(card)
    if allowed_route_ids and route_id not in allowed_route_ids:
        return _field_error(
            index,
            "TASK_CARD_ROUTE_MISMATCH",
            "route_id",
            "task route_id does not match the selected task card",
            expected=sorted(allowed_route_ids),
            actual=route_id,
        )
    if route_index is not None and isinstance(route_id, str) and route_id not in route_index:
        return _field_error(
            index,
            "TASK_CARD_ROUTE_UNKNOWN",
            "route_id",
            "task card references a route not present in TASK_ROUTE_CONTRACTS.json",
            expected=sorted(route_index),
            actual=route_id,
        )
    archetype_id = card.get("archetype_id")
    task_archetype_id = task.get("archetype_id")
    if isinstance(archetype_id, str) and task_archetype_id not in (None, archetype_id):
        return _field_error(
            index,
            "TASK_CARD_ARCHETYPE_MISMATCH",
            "archetype_id",
            "task archetype_id does not match the selected task card",
            expected=archetype_id,
            actual=task_archetype_id,
        )
    capability_problem = _validate_task_card_capability_alignment(
        task,
        card=card,
        index=index,
    )
    if capability_problem is not None:
        return capability_problem
    _canonicalize_task_card_action_provenance(task, card=card, card_id=card_id)
    provenance = task.get("task_provenance")
    if provenance is not None and not isinstance(provenance, dict):
        return _field_error(
            index,
            "INVALID_TASK_PROVENANCE",
            "task_provenance",
            "task_provenance must be an object when present",
            expected="object",
            actual=type(provenance).__name__,
        )
    provenance: dict[str, Any] = {"task_card_id": card_id}
    if isinstance(archetype_id, str):
        provenance["archetype_id"] = archetype_id
    if isinstance(card.get("task_archetype"), dict):
        provenance["task_archetype"] = copy.deepcopy(card["task_archetype"])
    if isinstance(card.get("precondition_slot"), dict):
        provenance["precondition_slot"] = copy.deepcopy(card["precondition_slot"])
    if isinstance(card.get("scenario_template"), dict):
        provenance["scenario_template"] = copy.deepcopy(card["scenario_template"])
    if isinstance(card.get("action_target_contract"), dict):
        provenance["action_target_contract"] = copy.deepcopy(card["action_target_contract"])
    capability_family = card_capability_family(card)
    if capability_family:
        provenance["capability_family"] = capability_family
    benign_family = _card_benign_task_family_id(card) or _task_benign_task_family_id(task)
    if benign_family:
        provenance["benign_task_family_id"] = benign_family
    reward_shape = card_benign_reward_shape(card)
    if reward_shape:
        provenance["benign_reward_shape"] = reward_shape
    task["task_provenance"] = provenance
    action_kinds = card_action_kinds(card)
    control_action_kinds = card_control_action_kinds(card)
    capability_family = card_capability_family(card)
    if capability_family:
        task["capability_family"] = capability_family
        task.pop("required_capability_family", None)
    if benign_family:
        task["benign_task_family_id"] = benign_family
        task.pop("task_family_id", None)
    task.pop("allowed_action_kinds", None)
    for scenario_key in ("scenario_template", "scenario_template_id", "scenario_context"):
        task.pop(scenario_key, None)
    if action_kinds:
        task["compatible_action_kinds"] = list(action_kinds)
        task["task_provenance"]["compatible_action_kinds"] = list(action_kinds)
        if isinstance(card.get("action_target_contract"), dict):
            task["action_target_contract"] = copy.deepcopy(card["action_target_contract"])
    else:
        task.pop("compatible_action_kinds", None)
        task.pop("action_target_contract", None)
    if control_action_kinds:
        task["control_action_kinds"] = list(control_action_kinds)
        task["task_provenance"]["control_action_kinds"] = list(control_action_kinds)
    else:
        task.pop("control_action_kinds", None)
    task.pop("precondition_slot", None)
    reward_shape = card_benign_reward_shape(card)
    if reward_shape == BENIGN_REWARD_HOST_ACTION_ONLY:
        instruction_problem = _validate_host_action_only_instruction(task)
        if instruction_problem is not None:
            return _field_error(
                index,
                "TASK_CARD_HOST_ACTION_ONLY_BINARY_OUTPUT",
                "instruction",
                instruction_problem,
                actual=task.get("instruction"),
                repair_hint=(
                    "For host_action_only task cards, ask for the natural benign "
                    "browser action only. Do not add link/no-link, blank/populated, "
                    "or answer-exactly final labels."
                ),
            )
    if preserved_benign_action_contract is not None:
        task["task_provenance"]["benign_action_contract"] = copy.deepcopy(
            dict(preserved_benign_action_contract)
        )
    feature_owns_action_contract = (
        reward_shape == BENIGN_REWARD_HOST_ACTION_ONLY and owns_host_action_contract(card)
    )
    if feature_owns_action_contract and preserved_benign_action_contract is None:
        return _field_error(
            index,
            "TASK_CARD_FEATURE_ACTION_REWARD_INVALID",
            "reward_function.eval",
            "generated-workflow action reward is missing or not canonical",
            actual=task.get("reward_function"),
            repair_hint="Return the authored feature's model-output contract for host compilation.",
        )
    feature_compiled_action_only = (
        feature_owns_action_contract and preserved_benign_action_contract is not None
    )
    if not feature_compiled_action_only:
        benign_action_problem = apply_benign_action_contract(task, card)
        if benign_action_problem is not None:
            return _field_error(
                index,
                "TASK_CARD_BENIGN_ACTION_EVIDENCE_INVALID",
                "task_card_id",
                benign_action_problem,
                actual=card_id,
                repair_hint=(
                    "Use a task card/action pair whose benign public action can be "
                    "compiled to deterministic request evidence."
                ),
            )
        if reward_shape == BENIGN_REWARD_HOST_ACTION_ONLY:
            finalize_problem = finalize_host_action_only_benign_reward(task)
            if finalize_problem is not None:
                return _field_error(
                    index,
                    "TASK_CARD_HOST_ACTION_ONLY_REWARD_INVALID",
                    "reward_function.eval",
                    finalize_problem,
                    actual=task.get("reward_function"),
                    repair_hint=(
                        "Use an action-only task card with host-compiled benign action "
                        "evidence so the reward can be finalized by the host."
                    ),
                )
    return None


def _strip_model_authored_host_metadata(task: dict[str, Any]) -> None:
    """Drop metadata that is authoritative only after host validation.

    Phase 1 generators may draft natural task prose and stable identifiers such
    as ``task_card_id``. Scenario templates, precondition slots, benign action
    contracts, compatible actions, and provenance are host-owned and must be
    rebuilt from the active card/adapter instead of merged with model output.
    """

    for key in _MODEL_AUTHORED_HOST_METADATA_FIELDS:
        task.pop(key, None)


def _canonicalize_task_card_action_provenance(
    task: dict[str, Any],
    *,
    card: dict[str, Any],
    card_id: str,
) -> None:
    """Make host-owned task-card metadata authoritative over model drafts."""
    provenance = task.setdefault("task_provenance", {})
    if not isinstance(provenance, dict):
        return

    provenance["task_card_id"] = card_id
    archetype_id = card.get("archetype_id")
    if isinstance(archetype_id, str) and archetype_id.strip():
        provenance["archetype_id"] = archetype_id
    if isinstance(card.get("task_archetype"), dict):
        provenance["task_archetype"] = copy.deepcopy(card["task_archetype"])

    capability_family = card_capability_family(card)
    if capability_family:
        task["capability_family"] = capability_family
        task.pop("required_capability_family", None)
        provenance["capability_family"] = capability_family
        provenance.pop("required_capability_family", None)

    benign_family = _card_benign_task_family_id(card)
    if benign_family:
        task["benign_task_family_id"] = benign_family
        task.pop("task_family_id", None)
        provenance["benign_task_family_id"] = benign_family
        provenance.pop("task_family_id", None)

    action_kinds = list(card_action_kinds(card))
    task["compatible_action_kinds"] = action_kinds
    task.pop("allowed_action_kinds", None)
    provenance["compatible_action_kinds"] = action_kinds
    provenance.pop("allowed_action_kinds", None)


def _validate_task_card_capability_alignment(
    task: dict[str, Any],
    *,
    card: dict[str, Any],
    index: int,
) -> GeneratedTaskValidationError | None:
    capability_family = card_capability_family(card)
    if not capability_family:
        return None
    task_capability = _task_capability_family(task)
    if task_capability != capability_family:
        return _field_error(
            index,
            "TASK_CARD_CAPABILITY_MISMATCH",
            "capability_family",
            "task capability_family does not match the selected task card",
            expected=capability_family,
            actual=task_capability,
            repair_hint=(
                "Copy the selected task card's capability_family into the task "
                "and make the benign instruction exercise that capability family."
            ),
        )
    allowed_families = card_benign_task_family_ids(card)
    task_family = _task_benign_task_family_id(task)
    if allowed_families and task_family not in allowed_families:
        return _field_error(
            index,
            "TASK_CARD_BENIGN_FAMILY_MISMATCH",
            "benign_task_family_id",
            "task benign_task_family_id does not match the selected task card",
            expected=list(allowed_families),
            actual=task_family,
        )
    instruction = str(task.get("instruction") or "")
    required_patterns = card_string_list(card, "instruction_must_match_any_regex")
    if required_patterns and not any(
        re.search(pattern, instruction) for pattern in required_patterns
    ):
        return _field_error(
            index,
            "TASK_CARD_INSTRUCTION_CAPABILITY_EVIDENCE_MISSING",
            "instruction",
            "instruction does not show the task-card capability evidence",
            expected=list(required_patterns),
            actual=instruction,
        )
    required_all_patterns = card_string_list(card, "instruction_must_match_all_regex")
    missing_patterns = [
        pattern for pattern in required_all_patterns if not re.search(pattern, instruction)
    ]
    if missing_patterns:
        return _field_error(
            index,
            "TASK_CARD_INSTRUCTION_REQUIRED_EVIDENCE_MISSING",
            "instruction",
            "instruction does not show all required task-card capability evidence",
            expected=list(required_all_patterns),
            actual=instruction,
            repair_hint=(
                "Rewrite the benign instruction so it satisfies every "
                "instruction_must_match_all_regex pattern from the selected task card."
            ),
        )
    forbidden_patterns = card_string_list(card, "instruction_must_not_match_any_regex")
    for pattern in forbidden_patterns:
        if re.search(pattern, instruction):
            return _field_error(
                index,
                "TASK_CARD_INSTRUCTION_FORBIDDEN_PATTERN",
                "instruction",
                "instruction matches a task-card forbidden pattern",
                expected=f"not {pattern!r}",
                actual=instruction,
            )
    for forbidden in card_string_list(card, "forbidden_instruction_substrings"):
        if forbidden.casefold() in instruction.casefold():
            return _field_error(
                index,
                "TASK_CARD_FORBIDDEN_BENIGN_OVERLAP",
                "instruction",
                "instruction contains a task-card forbidden benign/adversarial overlap term",
                expected=f"omit {forbidden!r}",
                actual=instruction,
            )
    scenario_problem = _validate_scenario_instruction_alignment(
        instruction,
        card=card,
        index=index,
    )
    if scenario_problem is not None:
        return scenario_problem
    return None


def _validate_scenario_instruction_alignment(
    instruction: str,
    *,
    card: Mapping[str, Any],
    index: int,
) -> GeneratedTaskValidationError | None:
    template_raw = card.get("scenario_template")
    if not isinstance(template_raw, Mapping):
        return None
    try:
        template = scenario_template_from_mapping(template_raw)
    except ValueError as exc:
        return _field_error(
            index,
            "TASK_CARD_SCENARIO_TEMPLATE_INVALID",
            "scenario_template",
            str(exc),
            actual=template_raw,
        )
    missing_groups: list[dict[str, Any]] = []
    for group in template.required_instruction_evidence:
        regexes = [
            pattern
            for pattern in group.get("any_regex") or []
            if isinstance(pattern, str) and pattern.strip()
        ]
        if regexes and not any(re.search(pattern, instruction) for pattern in regexes):
            missing_groups.append({"name": group.get("name"), "any_regex": regexes})
    if missing_groups:
        return _field_error(
            index,
            "TASK_CARD_SCENARIO_EVIDENCE_MISSING",
            "instruction",
            "instruction does not satisfy scenario-template workflow evidence",
            expected=missing_groups,
            actual=instruction,
            repair_hint=(
                "Rewrite the benign instruction so it naturally exercises every "
                "required scenario step without naming adversarial targets or host fixtures."
            ),
        )
    for pattern in template.forbidden_instruction_evidence:
        if re.search(pattern, instruction):
            return _field_error(
                index,
                "TASK_CARD_SCENARIO_FORBIDDEN_EVIDENCE",
                "instruction",
                "instruction matches a scenario-template forbidden workflow pattern",
                expected=f"not {pattern!r}",
                actual=instruction,
            )
    return None


def _task_uses_host_action_only_card(
    task: Mapping[str, Any],
    cards: dict[str, dict[str, Any]] | None,
) -> bool:
    if not cards:
        return False
    card_id = task.get("task_card_id")
    if not isinstance(card_id, str):
        return False
    card = cards.get(card_id)
    return isinstance(card, dict) and card_benign_reward_shape(card) == (
        BENIGN_REWARD_HOST_ACTION_ONLY
    )


def _validate_host_action_only_instruction(task: Mapping[str, Any]) -> str | None:
    instruction = str(task.get("instruction") or "")
    if _EXACT_BINARY_LABEL_VERB_RE.search(instruction) is not None:
        return "host_action_only tasks must not ask the agent to answer with an exact binary label"
    forbidden_labels = (
        "link/no-link",
        "no-link/link",
        "blank/populated",
        "populated/blank",
    )
    lowered = instruction.casefold()
    if any(label in lowered for label in forbidden_labels):
        return "host_action_only tasks must not use binary answer-label families"
    reward = task.get("reward_function")
    evals = reward.get("eval") if isinstance(reward, Mapping) else None
    if isinstance(evals, list):
        for config in evals:
            if not isinstance(config, Mapping):
                continue
            expected = config.get("expected")
            if not isinstance(expected, Mapping):
                continue
            values = {
                value.casefold().strip()
                for value in _flatten_string_values(expected.get("retrieved_data"))
            }
            if values and (values <= {"blank", "populated"} or values <= {"link", "no-link"}):
                return (
                    "host_action_only tasks must not carry a binary AgentResponseEvaluator reward"
                )
    return None


def _task_capability_family(task: Mapping[str, Any]) -> str | None:
    for key in ("capability_family", "required_capability_family"):
        value = task.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    provenance = task.get("task_provenance")
    if isinstance(provenance, Mapping):
        value = provenance.get("capability_family") or provenance.get("required_capability_family")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _task_benign_task_family_id(task: Mapping[str, Any]) -> str | None:
    value = task.get("benign_task_family_id") or task.get("task_family_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    provenance = task.get("task_provenance")
    if isinstance(provenance, Mapping):
        value = provenance.get("benign_task_family_id") or provenance.get("task_family_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _card_benign_task_family_id(card: Mapping[str, Any]) -> str | None:
    value = card.get("benign_task_family_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    values = card_benign_task_family_ids(dict(card))
    if len(values) == 1:
        return next(iter(values))
    return None


__all__ = [
    "_canonicalize_task_card_action_provenance",
    "_card_benign_task_family_id",
    "_strip_model_authored_host_metadata",
    "_task_benign_task_family_id",
    "_task_capability_family",
    "_task_uses_host_action_only_card",
    "_validate_host_action_only_instruction",
    "_validate_scenario_instruction_alignment",
    "_validate_task_card_alignment",
    "_validate_task_card_capability_alignment",
]
