"""Phase 2b task orchestration over exact text-fill checkpoints."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

from warp_taskgen.phase_2.text_fill.checkpoints import (
    load_text_fill_checkpoint,
    text_fill_checkpoint_path,
    text_fill_task_id,
    write_text_fill_checkpoint,
)
from warp_taskgen.phase_2.text_fill.pause import run_text_fill_units
from warp_taskgen.run_definition_contracts import RunDefinition

FillTextOperation = Callable[..., Awaitable[tuple[list[dict[str, Any]], list[dict[str, Any]]]]]


def validate_unique_text_fill_task_ids(plans: list[dict[str, Any]]) -> list[str]:
    """Fail before admission when two plans share one normalized task ID."""

    task_ids: list[str] = []
    seen: set[str] = set()
    for plan in plans:
        task_id = text_fill_task_id(plan)
        if task_id in seen:
            raise ValueError(f"duplicate normalized text-fill task id: {task_id!r}")
        seen.add(task_id)
        task_ids.append(task_id)
    return task_ids


async def fill_plans_with_checkpoints(
    plans: list[dict[str, Any]],
    *,
    texts_per_plan: int,
    concurrency: int,
    model: str,
    state_dir: Path,
    checkpoint_dir: Path,
    definition: RunDefinition,
    settings: Mapping[str, Any] | None,
    fill_operation: FillTextOperation,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reuse exact task checkpoints and fill only the remaining plans."""

    task_ids = validate_unique_text_fill_task_ids(plans)
    checkpoint_tasks: dict[str, dict[str, Any]] = {}
    checkpoint_diagnostics: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    for plan, task_id in zip(plans, task_ids, strict=True):
        checkpoint = text_fill_checkpoint_path(checkpoint_dir, task_id)
        reusable = load_text_fill_checkpoint(
            checkpoint,
            plan,
            definition=definition,
            text_model=model,
            texts_per_plan=texts_per_plan,
            settings=settings,
        )
        if reusable is None:
            pending.append(plan)
            continue
        task, diagnostics = reusable
        checkpoint_tasks[task_id] = task
        # Preserve the original compact diagnostics byte-for-byte on exact
        # resume.  The checkpoint envelope itself records that this unit was
        # reused; changing the diagnostics would make clean and resumed runs
        # diverge for no result-affecting reason.
        checkpoint_diagnostics[task_id] = diagnostics

    async def _run_one(plan: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        task_id = text_fill_task_id(plan)
        generated, diagnostics = await fill_operation(
            [plan],
            texts_per_plan=texts_per_plan,
            concurrency=1,
            model=model,
        )
        task = generated[0] if generated else None
        unit_diagnostics = (
            diagnostics[0]
            if diagnostics
            else {
                "task_id": task_id,
                "status": "failed",
                "errors": ["text-fill service returned no diagnostics"],
            }
        )
        write_text_fill_checkpoint(
            text_fill_checkpoint_path(checkpoint_dir, task_id),
            plan,
            task,
            unit_diagnostics,
            text_model=model,
            texts_per_plan=texts_per_plan,
            settings=settings,
            definition=definition,
        )
        return task, unit_diagnostics

    results = await run_text_fill_units(
        pending,
        _run_one,
        concurrency=concurrency,
        state_dir=state_dir,
    )
    for plan, result in zip(pending, results, strict=True):
        if isinstance(result, BaseException):
            raise result
        task, diagnostics = result
        task_id = text_fill_task_id(plan)
        checkpoint_diagnostics[task_id] = diagnostics
        if task is not None:
            checkpoint_tasks[task_id] = task

    filled = [checkpoint_tasks[task_id] for task_id in task_ids if task_id in checkpoint_tasks]
    diagnostics = [
        checkpoint_diagnostics[task_id] for task_id in task_ids if task_id in checkpoint_diagnostics
    ]
    return filled, diagnostics


__all__ = [
    "FillTextOperation",
    "fill_plans_with_checkpoints",
    "validate_unique_text_fill_task_ids",
]
