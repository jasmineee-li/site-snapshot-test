"""Async locks for serializing Browser Use reruns on shared benchmark state.

Diagnosis sandboxes (Modal) are stateless and run fully parallel.
Browser Use reruns mutate live benchmark instances, so they must
serialize on the concrete environment footprint they touch.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator, Iterable, Mapping
from contextlib import asynccontextmanager
from typing import Any

_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


@asynccontextmanager
async def site_lock(site_name: str) -> AsyncIterator[None]:
    """Serialize access to a named site.

    Kept for backwards compatibility in tests and older call sites.
    """
    async with lock_keys([f"site:{site_name}"]):
        yield


@asynccontextmanager
async def lock_keys(lock_keys: Iterable[str]) -> AsyncIterator[None]:
    """Acquire multiple locks in deterministic order to avoid deadlocks."""
    normalized = sorted({str(lock_key) for lock_key in lock_keys if str(lock_key)})
    acquired: list[asyncio.Lock] = []
    try:
        for lock_key in normalized:
            lock = _locks[lock_key]
            await lock.acquire()
            acquired.append(lock)
        yield
    finally:
        for lock in reversed(acquired):
            lock.release()


@asynccontextmanager
async def task_lock(task: Mapping[str, Any]) -> AsyncIterator[None]:
    """Serialize access to the concrete reset footprint for a bound task.

    Bound tasks prefer reset-endpoint locks, which preserve parallelism across
    separate instances of the same site while still protecting shared
    secondary-site state for multi-site tasks. Unbound tasks fall back to their
    normalized site set.
    """
    async with lock_keys(task_lock_keys(task)):
        yield


def task_lock_keys(task: Mapping[str, Any]) -> tuple[str, ...]:
    """Return deterministic lock keys for a task's mutation footprint."""
    runtime = task.get("_worldsim_runtime", {})
    reset_endpoints = runtime.get("reset_endpoints") if isinstance(runtime, dict) else None
    if isinstance(reset_endpoints, list) and reset_endpoints:
        keys = [f"reset:{endpoint}" for endpoint in reset_endpoints]
        return tuple(sorted({key for key in keys if key != "reset:"}))

    sites = runtime.get("sites") if isinstance(runtime, dict) else None
    if not isinstance(sites, list) or not sites:
        sites = task.get("sites")
    if not isinstance(sites, list) or not sites:
        site = task.get("site")
        sites = [site] if site else []
    keys = [f"site:{site}" for site in sites]
    return tuple(sorted({key for key in keys if key != "site:"}))


def reset_locks() -> None:
    """Clear all locks. Only for testing."""
    _locks.clear()
