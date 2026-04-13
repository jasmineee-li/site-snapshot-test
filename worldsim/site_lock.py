"""Per-site asyncio locks for serializing Browser Use agent reruns.

Diagnosis sandboxes (Modal) are stateless and run fully parallel.
Agent reruns (Browser Use) mutate shared DB state, so they must
serialize per site. Different sites run in parallel.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


@asynccontextmanager
async def site_lock(site_name: str) -> AsyncIterator[None]:
    """Serialize access to a benchmark site's shared state."""
    async with _locks[site_name]:
        yield


def reset_locks() -> None:
    """Clear all locks. Only for testing."""
    _locks.clear()
