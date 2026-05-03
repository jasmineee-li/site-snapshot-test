"""Task-card plan helpers for Phase 1 generate-new-tasks."""

from __future__ import annotations

import json
import re
from hashlib import sha256
from pathlib import Path
from typing import Any

from worldsim.adversarial_actions.capability_contracts import (
    action_kind_compatible_with_task_card,
    capability_family_from_task_card,
    compatibility_reason_for_task_card,
)


class TaskCardPlanError(ValueError):
    """Raised when a task-card plan is malformed."""


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
            raise TaskCardPlanError(
                f"task_cards[{index}].archetype_id must be a non-empty string"
            )
        capability_family = capability_family_from_task_card(card)
        for key in ("capability_family", "required_capability_family"):
            if key in card and (
                not isinstance(card[key], str) or not str(card[key]).strip()
            ):
                raise TaskCardPlanError(
                    f"task_cards[{index}].{key} must be a non-empty string"
                )
        action_kinds = card_action_kinds(card)
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
        ):
            _validate_string_array(card, index=index, key=key)
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


def task_card_plan_digest(plan: dict[str, Any] | None) -> str | None:
    if plan is None:
        return None
    encoded = json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(encoded.encode("utf-8")).hexdigest()


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
        if isinstance(card, dict) and isinstance(card.get("id"), str)
    }


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
