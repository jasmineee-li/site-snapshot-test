"""Pure Phase 2a input filtering and shard-boundary reconstruction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from warp_taskgen.agent_config import cap_tasks_per_site
from warp_taskgen.phases.phase_1_tasks import _parse_sites_filter

TASKS_PER_SHARD = 20


class PlanningShardSpecError(ValueError):
    """A persisted Phase 2 planning input cannot produce shard specs."""

    def __init__(self, message: str, *, reason_code: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True)
class PlanningShardPlan:
    """The filtered Phase 2a inputs and their deterministic shard specs."""

    filtered_tasks: list[dict[str, Any]]
    tasks_by_site: dict[str, list[dict[str, Any]]]
    sites_filter: set[str] | None
    specs: list[dict[str, Any]]
    origin_filtered_count: int
    uncapped_task_count: int


def classify_task_origin(task: Mapping[str, Any]) -> str:
    stamped = task.get("origin")
    if stamped in {"existing_task", "new_task"}:
        return str(stamped)
    task_id = str(task.get("task_id") or task.get("id") or "")
    if task_id.startswith("novel_"):
        return "new_task"
    return "existing_task"


def filter_tasks_by_origin(
    tasks: list[dict[str, Any]],
    task_origin: str | None,
    *,
    phase_label: str,
) -> list[dict[str, Any]]:
    if task_origin in (None, "", "all"):
        return tasks
    if task_origin not in {"existing_task", "new_task"}:
        raise PlanningShardSpecError(
            f"{phase_label}: --task-origin must be one of all, existing_task, new_task; "
            f"got {task_origin!r}",
            reason_code="invalid_task_origin",
        )
    return [task for task in tasks if classify_task_origin(task) == task_origin]


def build_planning_shard_specs(
    tasks: list[dict[str, Any]],
    *,
    task_origin: str | None = "all",
    max_tasks_per_site: int | None = None,
    sites_filter: Any = None,
    phase_label: str = "Phase 2",
) -> PlanningShardPlan:
    """Build the exact Phase 2a site order, caps, and shard boundaries.

    This function performs no I/O and owns the input transformation shared by
    the live runner and read-only checkpoint status.  The returned task objects
    are the caller's existing dictionaries; no task payload is rewritten.
    """

    filtered = filter_tasks_by_origin(tasks, task_origin, phase_label=phase_label)
    origin_filtered_count = len(filtered)
    uncapped_task_count = len(filtered)
    if max_tasks_per_site is not None:
        if type(max_tasks_per_site) is not int or max_tasks_per_site <= 0:
            raise PlanningShardSpecError(
                "max_tasks_per_site must be a positive integer",
                reason_code="invalid_max_tasks_per_site",
            )
        filtered = cap_tasks_per_site(filtered, max_tasks_per_site)

    tasks_by_site: dict[str, list[dict[str, Any]]] = {}
    for task in filtered:
        site = task["site"]
        tasks_by_site.setdefault(site, []).append(task)

    parsed_sites = _parse_sites_filter(sites_filter)
    if parsed_sites is not None:
        unknown = parsed_sites.difference(tasks_by_site)
        if unknown:
            raise PlanningShardSpecError(
                f"{phase_label}: --sites includes unknown site(s): {sorted(unknown)}. "
                f"Known sites: {sorted(tasks_by_site)}",
                reason_code="unknown_site",
            )
        tasks_by_site = {
            site: site_tasks for site, site_tasks in tasks_by_site.items() if site in parsed_sites
        }

    specs: list[dict[str, Any]] = []
    for site, site_tasks in tasks_by_site.items():
        chunks = [
            site_tasks[index : index + TASKS_PER_SHARD]
            for index in range(0, len(site_tasks), TASKS_PER_SHARD)
        ]
        for shard_index, chunk in enumerate(chunks):
            label = f"{site}-shard-{shard_index}" if len(chunks) > 1 else site
            specs.append(
                {
                    "label": label,
                    "site": site,
                    "site_tasks": chunk,
                    "all_site_tasks": site_tasks,
                    "input_task_ids": [str(task.get("id") or "") for task in chunk],
                }
            )

    return PlanningShardPlan(
        filtered_tasks=filtered,
        tasks_by_site=tasks_by_site,
        sites_filter=parsed_sites,
        specs=specs,
        origin_filtered_count=origin_filtered_count,
        uncapped_task_count=uncapped_task_count,
    )
