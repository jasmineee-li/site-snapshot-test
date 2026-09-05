"""Batch validation for exact task-card generation allocations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from warp_taskgen.phases.phase_1_task_cards import task_card_generation_counts

from .errors import GeneratedTaskValidationError


def validate_task_card_generation_distribution(
    raw_tasks: list[Any],
    *,
    site_name: str,
    task_card_plan: Mapping[str, Any] | None,
) -> list[GeneratedTaskValidationError]:
    """Check exact per-card quotas after validating the individual rows.

    The task-card plan has already been validated at load time. A missing or
    overfilled card is reported against its original plan index so diagnostics
    remain actionable when inactive cards precede active ones.
    """
    counts = task_card_generation_counts(task_card_plan, site_name=site_name)
    if counts is None:
        return []

    active_cards = [
        (plan_index, card)
        for plan_index, card in enumerate((task_card_plan or {}).get("task_cards", []))
        if isinstance(card, Mapping)
        and str(card.get("status", "active")) == "active"
        and card.get("site") == site_name
        and isinstance(card.get("id"), str)
    ]
    indexes_by_card: dict[str, list[int]] = {card_id: [] for card_id in counts}
    for task_index, task in enumerate(raw_tasks):
        if not isinstance(task, Mapping):
            continue
        card_id = task.get("task_card_id")
        if isinstance(card_id, str) and card_id in indexes_by_card:
            indexes_by_card[card_id].append(task_index)

    errors: list[GeneratedTaskValidationError] = []
    for plan_index, card in active_cards:
        card_id = str(card["id"])
        expected = counts[card_id]
        indexes = indexes_by_card[card_id]
        actual = len(indexes)
        if actual == expected:
            continue
        errors.append(
            GeneratedTaskValidationError(
                code="TASK_CARD_GENERATION_COUNT_MISMATCH",
                path=f"$.task_cards[{plan_index}].generation_count",
                message=(
                    f"task-card allocation for {card_id!r} has {actual} task(s); "
                    f"expected {expected}; task indexes for this card: {indexes!r}"
                ),
                expected=expected,
                actual=actual,
                repair_hint=(
                    "Return one globally unique task id per row and assign exactly "
                    f"{expected} row(s) to task_card_id {card_id!r}."
                ),
            )
        )
    return errors


__all__ = ["validate_task_card_generation_distribution"]
