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


def _ordered_instance_dicts(instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return raw instance dicts in the same stable order as ``ordered_instances``.

    Mirrors the model-based ordering so dict and pydantic callers produce
    identical hash buckets for the same physical topology.
    """

    def _key(instance: dict[str, Any]) -> tuple[Any, ...]:
        replica_index = instance.get("replica_index")
        has_replica_index = isinstance(replica_index, int)
        return (
            normalize_site_name(str(instance.get("site_name", ""))),
            not has_replica_index,
            replica_index if has_replica_index else 0,
            str(instance.get("replica_name") or ""),
            str(instance.get("site_url") or ""),
        )

    return sorted(instances, key=_key)


def select_task_site_instance_dict(
    task: dict[str, Any],
    site_name: str,
    instances: list[dict[str, Any]],
) -> dict[str, Any]:
    """Dict-valued sibling of :func:`select_task_site_instance`.

    Phase 2c threads raw instance dicts (parsed from
    ``instances.smoke.json`` / ``instances.scale.json``) rather than
    :class:`BenchmarkInstance` models. This helper fans out tasks across
    same-site replicas using the same hash-based routing Phase 4 uses,
    so a task sees the same replica regardless of which code path binds
    it.
    """
    normalized_site = normalize_site_name(site_name)
    site_instances = [
        instance
        for instance in instances
        if normalize_site_name(str(instance.get("site_name", ""))) == normalized_site
    ]
    if not site_instances:
        raise ValueError(f"no instances configured for site {site_name!r}")
    ordered = _ordered_instance_dicts(site_instances)
    if len(ordered) == 1:
        return ordered[0]
    index = stable_index_for_task(task, len(ordered), salt=normalized_site)
    return ordered[index]
