"""Ordering and eligibility exports for novel generated tasks."""

from __future__ import annotations

from typing import Any


def sort_novel_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort novel tasks by site, then id, for deterministic merges."""
    return sorted(
        tasks,
        key=lambda task: (str(task.get("site", "")), str(task.get("id", ""))),
    )


def merge_benign_tasks(
    existing_task_wraps: list[dict[str, Any]],
    novel_tasks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge existing-task wraps and new-task entries deterministically."""
    return list(existing_task_wraps) + sort_novel_tasks(novel_tasks)


def site_is_generate_new_tasks_eligible(profile: dict[str, Any]) -> bool:
    """Legacy coverage-gap helper retained for older callers.

    Phase 1b runtime eligibility now lives in
    ``load_generate_new_tasks_eligible_sites`` and is based on carrier route
    contracts, not this coverage-gap predicate.
    """
    coverage = profile.get("existing_task_coverage", {})
    uncovered = coverage.get("injection_surfaces_without_task_coverage", [])
    return isinstance(uncovered, list) and bool(uncovered)


__all__ = [
    "merge_benign_tasks",
    "site_is_generate_new_tasks_eligible",
    "sort_novel_tasks",
]
