"""Exact task-card batching for Phase 1 novel-task generation.

This module owns the concrete allocation seam.  The Phase 1 runner supplies a
callback for one isolated card generation; this feature then collects those
results and applies the existing full-plan validators before promotion.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from warp_taskgen.phase_1.generated_workflows import restore_compiled_tasks
from warp_taskgen.phase_1.novel_task_validation import (
    sort_novel_tasks,
    validate_generated_novel_tasks_detailed,
)
from warp_taskgen.phases.phase_1_task_cards import task_card_generation_counts


@dataclass(frozen=True)
class TaskCardGenerationSlice:
    """One authored card and its site-wide task-number offset."""

    task_card_plan: dict[str, Any]
    task_number_start: int


@dataclass(frozen=True)
class CardSliceResult:
    """Result returned by the facade callback for one card slice."""

    benign_tasks: list[dict[str, Any]]
    errors: list[str]


@dataclass(frozen=True)
class CardBatchResult:
    """Aggregated, full-plan-validated output before cache promotion."""

    benign_tasks: list[dict[str, Any]]
    errors: list[str]


CardSliceGenerator = Callable[[TaskCardGenerationSlice, int], Awaitable[CardSliceResult]]


def task_card_generation_slices(
    task_card_plan: Mapping[str, Any] | None,
    *,
    site_name: str,
) -> tuple[TaskCardGenerationSlice, ...]:
    """Return authored active-card slices while retaining root plan metadata."""

    counts = task_card_generation_counts(task_card_plan, site_name=site_name)
    if counts is None or not isinstance(task_card_plan, Mapping):
        return ()
    active_cards = [
        card
        for card in task_card_plan.get("task_cards", [])
        if isinstance(card, Mapping)
        and str(card.get("status", "active")) == "active"
        and str(card.get("site") or "").strip() == site_name
    ]
    if len(active_cards) <= 1:
        return ()

    root_metadata = {key: value for key, value in task_card_plan.items() if key != "task_cards"}
    slices: list[TaskCardGenerationSlice] = []
    task_number_start = 1
    for card in active_cards:
        card_id = str(card.get("id") or "")
        count = counts.get(card_id)
        if count is None:
            # Validated plans cannot reach this branch.  Direct callers retain
            # the legacy path rather than guessing an allocation.
            return ()
        sliced_plan = dict(root_metadata)
        sliced_plan["task_cards"] = [dict(card)]
        slices.append(
            TaskCardGenerationSlice(
                task_card_plan=sliced_plan,
                task_number_start=task_number_start,
            )
        )
        task_number_start += count
    return tuple(slices)


def rekey_sandbox_task_ids(
    tasks: list[Any],
    *,
    site_name: str,
    task_number_start: int,
) -> list[dict[str, Any]]:
    """Validate model IDs, then assign one canonical site-wide sequence."""

    if (
        isinstance(task_number_start, bool)
        or not isinstance(task_number_start, int)
        or task_number_start < 1
    ):
        raise ValueError("task_number_start must be a positive integer")
    pattern = re.compile(rf"^novel_{re.escape(site_name)}_\d+$")
    rekeyed: list[dict[str, Any]] = []
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError(f"sandbox task {index} must be an object before ID rekey")
        raw_id = task.get("id")
        if not isinstance(raw_id, str) or pattern.fullmatch(raw_id) is None:
            raise ValueError(
                f"sandbox task {index} id must match novel_{site_name}_<n> before "
                f"canonical rekey; got {raw_id!r}"
            )
        item = dict(task)
        item["id"] = f"novel_{site_name}_{task_number_start + index}"
        rekeyed.append(item)
    return rekeyed


async def collect_card_slices(
    *,
    card_slices: tuple[TaskCardGenerationSlice, ...],
    generate_slice: CardSliceGenerator,
    expected_task_count: int,
    site_name: str,
    profile: dict[str, Any],
    route_contracts: dict[str, Any],
    task_card_plan: dict[str, Any],
    host_compiled_evaluator_types: frozenset[str],
) -> CardBatchResult:
    """Run card callbacks concurrently and apply unchanged site validation."""

    results = await asyncio.gather(
        *[generate_slice(card_slice, index) for index, card_slice in enumerate(card_slices)],
        return_exceptions=True,
    )
    generated_tasks: list[dict[str, Any]] = []
    failures: list[str] = []
    fatal_error: BaseException | None = None
    for index, result in enumerate(results, start=1):
        if isinstance(result, BaseException):
            if not isinstance(result, Exception) and fatal_error is None:
                fatal_error = result
            elif isinstance(result, Exception):
                failures.append(f"card slice {index}: {type(result).__name__}: {result}")
            continue
        if result.errors:
            failures.extend(f"card slice {index}: {error}" for error in result.errors)
            continue
        generated_tasks.extend(result.benign_tasks)
    if fatal_error is not None:
        raise fatal_error
    if failures:
        return CardBatchResult([], failures)

    validated_tasks, detailed_errors = validate_generated_novel_tasks_detailed(
        generated_tasks,
        site_name=site_name,
        profile=profile,
        expected_task_count=expected_task_count,
        route_contracts=route_contracts,
        task_card_plan=task_card_plan,
        host_compiled_evaluator_types=host_compiled_evaluator_types,
    )
    if detailed_errors:
        return CardBatchResult([], [error.render() for error in detailed_errors])

    try:
        compiled_tasks = restore_compiled_tasks(
            validated_tasks,
            task_card_plan=task_card_plan,
        )
    except ValueError as exc:
        return CardBatchResult([], [str(exc)])
    validated_tasks, detailed_errors = validate_generated_novel_tasks_detailed(
        compiled_tasks,
        site_name=site_name,
        profile=profile,
        expected_task_count=expected_task_count,
        route_contracts=route_contracts,
        task_card_plan=task_card_plan,
        host_compiled_evaluator_types=host_compiled_evaluator_types,
    )
    if detailed_errors:
        return CardBatchResult([], [error.render() for error in detailed_errors])
    return CardBatchResult(sort_novel_tasks(validated_tasks), [])


__all__ = [
    "CardBatchResult",
    "CardSliceGenerator",
    "CardSliceResult",
    "TaskCardGenerationSlice",
    "collect_card_slices",
    "rekey_sandbox_task_ids",
    "task_card_generation_slices",
]
