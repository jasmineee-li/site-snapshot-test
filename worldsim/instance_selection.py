"""Deterministic task-to-instance selection helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from worldsim.config import BenchmarkInstance
from worldsim.placeholders import normalize_site_name


def task_identity(task: dict[str, Any]) -> str:
    """Return a stable identity for deterministic task routing."""
    task_id = task.get("id")
    if task_id not in (None, ""):
        return str(task_id)
    return json.dumps(task, sort_keys=True, separators=(",", ":"))


def ordered_instances(instances: list[BenchmarkInstance]) -> list[BenchmarkInstance]:
    """Return instances in a stable order independent of config file ordering."""
    return sorted(
        instances,
        key=lambda instance: (
            normalize_site_name(instance.site_name),
            instance.replica_index is None,
            instance.replica_index if instance.replica_index is not None else 0,
            instance.replica_name or "",
            instance.site_url,
        ),
    )


def stable_index_for_task(
    task: dict[str, Any],
    modulo: int,
    *,
    salt: str = "",
) -> int:
    """Return a stable bucket index for *task* in ``range(modulo)``."""
    if modulo <= 0:
        raise ValueError("modulo must be positive for stable task routing")
    digest = hashlib.sha256(f"{task_identity(task)}:{salt}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % modulo


def select_task_instance(
    task: dict[str, Any],
    instances: list[BenchmarkInstance],
    *,
    salt: str = "",
) -> BenchmarkInstance:
    """Select one instance deterministically for *task* from *instances*."""
    ordered = ordered_instances(instances)
    if not ordered:
        raise ValueError("no instances available for deterministic selection")
    if len(ordered) == 1:
        return ordered[0]

    index = stable_index_for_task(task, len(ordered), salt=salt)
    return ordered[index]


def select_task_site_instance(
    task: dict[str, Any],
    site_name: str,
    instances: list[BenchmarkInstance],
) -> BenchmarkInstance:
    """Select an instance for one logical site using the shared task hash."""
    normalized_site = normalize_site_name(site_name)
    site_instances = [
        instance
        for instance in instances
        if normalize_site_name(instance.site_name) == normalized_site
    ]
    if not site_instances:
        raise ValueError(f"no instances configured for site {site_name!r}")
    return select_task_instance(task, site_instances, salt=normalized_site)
