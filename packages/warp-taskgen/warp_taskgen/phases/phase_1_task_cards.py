"""Task-card plan helpers for Phase 1 generate-new-tasks."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

from warp_taskgen.adversarial_actions.action_targets import (
    action_target_contracts_from_card,
    validate_action_target_contract,
    validate_action_target_contracts_field,
)
from warp_taskgen.adversarial_actions.benign_action_contracts import (
    validate_benign_action_evidence_stage,
)
from warp_taskgen.adversarial_actions.capability_contracts import (
    BENIGN_REWARD_HOST_ACTION_ONLY,
    BENIGN_REWARD_SHAPES,
    action_kind_compatible_with_task_card,
    benign_reward_shape_from_task_card,
    capability_family_from_task_card,
    compatibility_reason_for_task_card,
    get_action_capability_contract,
)
from warp_taskgen.adversarial_actions.capability_task_cards import (
    available_capability_task_card_profiles,
    compile_capability_task_card_plan,
)
from warp_taskgen.adversarial_actions.catalog import get_action_spec
from warp_taskgen.adversarial_actions.precondition_slots import validate_precondition_slot
from warp_taskgen.adversarial_actions.scenario_templates import validate_scenario_template


class TaskCardPlanError(ValueError):
    """Raised when a task-card plan is malformed."""


GENERATION_COUNT_FIELD = "generation_count"


def load_or_compile_task_card_plan(
    *,
    path: Path | None = None,
    task_capability_profile: str | None = None,
    sites: set[str] | None = None,
) -> dict[str, Any] | None:
    """Return a validated task-card plan from JSON or a compiled profile."""
    if path is not None and task_capability_profile:
        raise TaskCardPlanError(
            "--task-card-plan and --task-capability-profile are mutually exclusive"
        )
    if task_capability_profile:
        try:
            plan = compile_capability_task_card_plan(
                task_capability_profile,
                sites=sites,
            )
        except ValueError as exc:
            raise TaskCardPlanError(str(exc)) from exc
        validate_task_card_plan(plan)
        return plan
    return load_task_card_plan(path)


def load_task_card_plan(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TaskCardPlanError(f"task-card plan is invalid JSON: {path}: {exc}") from exc
    validate_task_card_plan(data)
    return data


def validate_task_card_plan(plan: Any) -> None:
    if not isinstance(plan, dict):
        raise TaskCardPlanError("task-card plan must be a JSON object")
    cards = plan.get("task_cards")
    if not isinstance(cards, list) or not cards:
        raise TaskCardPlanError("task-card plan must include non-empty task_cards array")
    seen: set[str] = set()
    active_cards_by_site: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, card in enumerate(cards):
        if not isinstance(card, dict):
            raise TaskCardPlanError(f"task_cards[{index}] must be an object")
        card_id = card.get("id")
        if not isinstance(card_id, str) or not card_id.strip():
            raise TaskCardPlanError(f"task_cards[{index}].id must be a non-empty string")
        if card_id in seen:
            raise TaskCardPlanError(f"duplicate task card id {card_id!r}")
        seen.add(card_id)
        site = card.get("site")
        if not isinstance(site, str) or not site.strip():
            raise TaskCardPlanError(f"task_cards[{index}].site must be a non-empty string")
        if str(card.get("status", "active")) == "active":
            active_cards_by_site.setdefault(site, []).append((index, card))
            if GENERATION_COUNT_FIELD in card:
                generation_count = card[GENERATION_COUNT_FIELD]
                if (
                    isinstance(generation_count, bool)
                    or not isinstance(generation_count, int)
                    or generation_count <= 0
                ):
                    raise TaskCardPlanError(
                        f"task_cards[{index}].{GENERATION_COUNT_FIELD} must be a positive integer"
                    )
        route_ids = card.get("route_ids", card.get("route_id"))
        if route_ids is not None:
            values = [route_ids] if isinstance(route_ids, str) else route_ids
            if not isinstance(values, list) or not all(
                isinstance(item, str) and item.strip() for item in values
            ):
                raise TaskCardPlanError(
                    f"task_cards[{index}].route_ids must be a string or array of strings"
                )
        archetype_id = card.get("archetype_id")
        if archetype_id is not None and (
            not isinstance(archetype_id, str) or not archetype_id.strip()
        ):
            raise TaskCardPlanError(f"task_cards[{index}].archetype_id must be a non-empty string")
        task_archetype = card.get("task_archetype")
        if task_archetype is not None and not isinstance(task_archetype, dict):
            raise TaskCardPlanError(f"task_cards[{index}].task_archetype must be an object")
        precondition_slot_problem = validate_precondition_slot(card.get("precondition_slot"))
        if precondition_slot_problem is not None:
            raise TaskCardPlanError(f"task_cards[{index}].{precondition_slot_problem}")
        scenario_problem = validate_scenario_template(card.get("scenario_template"))
        if scenario_problem is not None:
            raise TaskCardPlanError(f"task_cards[{index}].{scenario_problem}")
        target_contract_problem = validate_action_target_contract(
            card.get("action_target_contract")
        )
        if target_contract_problem is not None:
            raise TaskCardPlanError(f"task_cards[{index}].{target_contract_problem}")
        target_contracts_problem = validate_action_target_contracts_field(
            card.get("action_target_contracts")
        )
        if target_contracts_problem is not None:
            raise TaskCardPlanError(f"task_cards[{index}].{target_contracts_problem}")
        target_contracts = action_target_contracts_from_card(card)
        capability_family = capability_family_from_task_card(card)
        for key in ("capability_family", "required_capability_family"):
            if key in card and (not isinstance(card[key], str) or not str(card[key]).strip()):
                raise TaskCardPlanError(f"task_cards[{index}].{key} must be a non-empty string")
        action_kinds = card_action_kinds(card)
        if "allowed_action_kinds" in card:
            raise TaskCardPlanError(
                f"task_cards[{index}].allowed_action_kinds is legacy metadata; "
                "use compatible_action_kinds"
            )
        if action_kinds and not capability_family:
            raise TaskCardPlanError(
                f"task_cards[{index}].compatible_action_kinds requires capability_family"
            )
        for action_kind in action_kinds:
            if not action_kind_compatible_with_task_card(action_kind, card):
                reason = compatibility_reason_for_task_card(action_kind, card)
                raise TaskCardPlanError(
                    f"task_cards[{index}].compatible_action_kinds contains "
                    f"incompatible action {action_kind!r}: {reason}"
                )
            if target_contracts and not any(
                str(contract.get("action_kind") or "").strip() == action_kind
                for contract in target_contracts
            ):
                raise TaskCardPlanError(
                    f"task_cards[{index}].compatible_action_kinds contains "
                    f"{action_kind!r} but no matching action_target_contract"
                )
        control_action_kinds = card_control_action_kinds(card)
        if control_action_kinds and action_kinds:
            raise TaskCardPlanError(
                f"task_cards[{index}].control_action_kinds must not be combined "
                "with compatible_action_kinds"
            )
        for action_kind in control_action_kinds:
            if get_action_spec(action_kind) is None:
                raise TaskCardPlanError(
                    f"task_cards[{index}].control_action_kinds contains "
                    f"unknown action {action_kind!r}"
                )
        reward_shape = benign_reward_shape_from_task_card(card)
        if reward_shape is not None and reward_shape not in BENIGN_REWARD_SHAPES:
            raise TaskCardPlanError(
                f"task_cards[{index}].benign_reward_shape must be one of "
                f"{sorted(BENIGN_REWARD_SHAPES)}"
            )
        expected_reward_shapes = {
            contract.benign_reward_shape
            for action_kind in action_kinds
            if (contract := get_action_capability_contract(action_kind)) is not None
        }
        if action_kinds and not reward_shape:
            raise TaskCardPlanError(
                f"task_cards[{index}].benign_reward_shape is required when "
                "compatible_action_kinds is present"
            )
        if len(expected_reward_shapes) == 1:
            expected_reward_shape = next(iter(expected_reward_shapes))
            stricter_host_action_only = (
                reward_shape == BENIGN_REWARD_HOST_ACTION_ONLY
                and card.get("requires_benign_action_evidence") is True
                and expected_reward_shape != BENIGN_REWARD_HOST_ACTION_ONLY
            )
            if reward_shape != expected_reward_shape and not stricter_host_action_only:
                raise TaskCardPlanError(
                    f"task_cards[{index}].benign_reward_shape {reward_shape!r} "
                    f"does not match compatible action contract "
                    f"{expected_reward_shape!r}"
                )
        elif len(expected_reward_shapes) > 1:
            raise TaskCardPlanError(
                f"task_cards[{index}].compatible_action_kinds map to multiple "
                f"benign_reward_shape values: {sorted(expected_reward_shapes)}"
            )
        _validate_string_or_string_array(
            card,
            index=index,
            key="benign_task_family_ids",
            singular_key="benign_task_family_id",
        )
        for key in (
            "instruction_must_match_any_regex",
            "instruction_must_match_all_regex",
            "instruction_must_not_match_any_regex",
            "forbidden_instruction_substrings",
            "control_action_kinds",
        ):
            _validate_string_array(card, index=index, key=key)
        if "requires_benign_action_evidence" in card and not isinstance(
            card["requires_benign_action_evidence"], bool
        ):
            raise TaskCardPlanError(
                f"task_cards[{index}].requires_benign_action_evidence must be a boolean"
            )
        benign_action_evidence = card.get("benign_action_evidence")
        if benign_action_evidence is not None and not isinstance(benign_action_evidence, dict):
            raise TaskCardPlanError(f"task_cards[{index}].benign_action_evidence must be an object")
        if isinstance(benign_action_evidence, dict):
            required = benign_action_evidence.get("required")
            if required is not None and not isinstance(required, bool):
                raise TaskCardPlanError(
                    f"task_cards[{index}].benign_action_evidence.required must be a boolean"
                )
            stage = benign_action_evidence.get("stage")
            if stage is not None and (
                not isinstance(stage, str) or not validate_benign_action_evidence_stage(stage)
            ):
                raise TaskCardPlanError(
                    f"task_cards[{index}].benign_action_evidence.stage must be a supported stage"
                )
            for key in ("action_kind", "editor_method"):
                value = benign_action_evidence.get(key)
                if value is not None and (not isinstance(value, str) or not value.strip()):
                    raise TaskCardPlanError(
                        f"task_cards[{index}].benign_action_evidence.{key} "
                        "must be a non-empty string"
                    )
        if reward_shape == BENIGN_REWARD_HOST_ACTION_ONLY and not (
            card.get("requires_benign_action_evidence") is True
            or (
                isinstance(benign_action_evidence, dict)
                and benign_action_evidence.get("required") is True
            )
        ):
            raise TaskCardPlanError(
                f"task_cards[{index}].benign_reward_shape host_action_only "
                "requires benign action evidence"
            )
        for key in (
            "instruction_must_match_any_regex",
            "instruction_must_match_all_regex",
            "instruction_must_not_match_any_regex",
        ):
            for pattern in card_string_list(card, key):
                try:
                    re.compile(pattern)
                except re.error as exc:
                    raise TaskCardPlanError(
                        f"task_cards[{index}].{key} contains invalid regex {pattern!r}: {exc}"
                    ) from exc

    for site, active_cards in active_cards_by_site.items():
        declared = [(index, card) for index, card in active_cards if GENERATION_COUNT_FIELD in card]
        if not declared or len(declared) == len(active_cards):
            continue
        missing = [
            f"task_cards[{index}] ({card.get('id')!r})"
            for index, card in active_cards
            if GENERATION_COUNT_FIELD not in card
        ]
        declared_summary = ", ".join(
            f"{card.get('id')!r}={card[GENERATION_COUNT_FIELD]}" for _index, card in declared
        )
        raise TaskCardPlanError(
            f"active task cards for site {site!r} must all declare "
            f"{GENERATION_COUNT_FIELD} when any card uses it; missing on "
            f"{', '.join(missing)} (declared: {declared_summary})"
        )


def task_card_plan_digest(plan: dict[str, Any] | None) -> str | None:
    if plan is None:
        return None
    encoded = json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(encoded.encode("utf-8")).hexdigest()


def task_capability_profile_choices() -> tuple[str, ...]:
    """Return CLI-supported compiled task capability profile names."""
    return available_capability_task_card_profiles()


def task_card_plan_for_site(plan: dict[str, Any] | None, site_name: str) -> dict[str, Any] | None:
    if plan is None:
        return None
    cards = [
        card
        for card in plan.get("task_cards", [])
        if isinstance(card, dict)
        and card.get("site") == site_name
        and str(card.get("status", "active")) == "active"
    ]
    if not cards:
        return None
    site_plan = {key: value for key, value in plan.items() if key != "task_cards"}
    site_plan["task_cards"] = cards
    return site_plan


def task_card_index(plan: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if plan is None:
        return {}
    return {
        str(card["id"]): card
        for card in plan.get("task_cards", [])
        if isinstance(card, dict)
        and isinstance(card.get("id"), str)
        and str(card.get("status", "active")) == "active"
    }


def card_generation_count(card: Mapping[str, Any]) -> int | None:
    """Return an authored positive per-card generation count when present."""
    value = card.get(GENERATION_COUNT_FIELD)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def task_card_generation_counts(
    plan: Mapping[str, Any] | None,
    *,
    site_name: str | None = None,
) -> dict[str, int] | None:
    """Return exact active-card allocations, or ``None`` for legacy plans.

    A plan only opts into exact card allocation when at least one active card
    declares ``generation_count``.  Validation normally guarantees that every
    active card for the selected site declares a positive count; malformed
    direct callers receive ``None`` so the legacy fallback remains intact.
    """
    if not isinstance(plan, Mapping):
        return None
    cards = [
        card
        for card in plan.get("task_cards", [])
        if isinstance(card, Mapping)
        and str(card.get("status", "active")) == "active"
        and (site_name is None or card.get("site") == site_name)
    ]
    if not cards or not any(GENERATION_COUNT_FIELD in card for card in cards):
        return None
    counts: dict[str, int] = {}
    for card in cards:
        card_id = card.get("id")
        count = card_generation_count(card)
        if not isinstance(card_id, str) or not card_id.strip() or count is None:
            return None
        counts[card_id] = count
    return counts


def task_card_generation_count(
    plan: Mapping[str, Any] | None,
    *,
    site_name: str,
) -> int | None:
    """Return the exact active-card total for one site, if opted in."""
    counts = task_card_generation_counts(plan, site_name=site_name)
    return sum(counts.values()) if counts is not None else None


def task_card_generation_prompt_addendum(
    plan: Mapping[str, Any] | None,
    *,
    site_name: str | None = None,
) -> str:
    """Describe an exact per-card allocation to the Phase 1 model."""
    if not isinstance(plan, Mapping):
        return ""
    selected_site = site_name
    active_cards = [
        card
        for card in plan.get("task_cards", [])
        if isinstance(card, Mapping)
        and str(card.get("status", "active")) == "active"
        and (site_name is None or card.get("site") == site_name)
    ]
    counts = task_card_generation_counts(plan, site_name=selected_site)
    if not active_cards or counts is None:
        return ""
    site_name = selected_site or str(active_cards[0].get("site") or "site")
    total = sum(counts.values())
    next_id = 1
    lines: list[str] = []
    for card in active_cards:
        card_id = str(card.get("id") or "")
        count = counts[card_id]
        end_id = next_id + count - 1
        lines.append(
            f"- task_card_id `{card_id}`: exactly {count} task(s) "
            f"(`novel_{site_name}_{next_id}` through `novel_{site_name}_{end_id}`)"
        )
        next_id = end_id + 1
    return "\n".join(
        [
            "<task_card_generation_allocation>",
            f"Generate exactly {total} tasks in this one response; do not split the "
            "allocation across runs or restart the id counter per card.",
            *lines,
            "Use each listed task_card_id exactly as written. IDs must be unique "
            f"within this response and use one global 1-based counter for {site_name}.",
            "</task_card_generation_allocation>",
        ]
    )


def card_route_ids(card: dict[str, Any]) -> set[str]:
    raw = card.get("route_ids", card.get("route_id"))
    if isinstance(raw, str) and raw.strip():
        return {raw}
    if isinstance(raw, list):
        return {item for item in raw if isinstance(item, str) and item.strip()}
    return set()


def card_capability_family(card: dict[str, Any]) -> str | None:
    return capability_family_from_task_card(card)


def card_action_kinds(card: dict[str, Any]) -> tuple[str, ...]:
    return tuple(card_string_list(card, "compatible_action_kinds"))


def card_control_action_kinds(card: dict[str, Any]) -> tuple[str, ...]:
    return tuple(card_string_list(card, "control_action_kinds"))


def card_benign_reward_shape(card: dict[str, Any]) -> str | None:
    return benign_reward_shape_from_task_card(card)


def card_benign_task_family_ids(card: dict[str, Any]) -> tuple[str, ...]:
    values = card_string_list(card, "benign_task_family_ids")
    singular = card.get("benign_task_family_id")
    if isinstance(singular, str) and singular.strip():
        values = (singular.strip(), *values)
    return tuple(dict.fromkeys(values))


def card_string_list(card: dict[str, Any], key: str) -> tuple[str, ...]:
    raw = card.get(key)
    if isinstance(raw, str) and raw.strip():
        return (raw.strip(),)
    if isinstance(raw, list):
        return tuple(item.strip() for item in raw if isinstance(item, str) and item.strip())
    return ()


def _validate_string_array(card: dict[str, Any], *, index: int, key: str) -> None:
    if key not in card:
        return
    values = card.get(key)
    if not isinstance(values, list) or not all(
        isinstance(item, str) and item.strip() for item in values
    ):
        raise TaskCardPlanError(f"task_cards[{index}].{key} must be an array of strings")


def _validate_string_or_string_array(
    card: dict[str, Any],
    *,
    index: int,
    key: str,
    singular_key: str,
) -> None:
    for candidate in (key, singular_key):
        if candidate not in card:
            continue
        values = card.get(candidate)
        if isinstance(values, str) and values.strip():
            continue
        if isinstance(values, list) and all(
            isinstance(item, str) and item.strip() for item in values
        ):
            continue
        raise TaskCardPlanError(
            f"task_cards[{index}].{candidate} must be a string or array of strings"
        )
