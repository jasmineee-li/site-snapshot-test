"""Bounded semaphore for host-side Anthropic Messages API calls.

Current pipeline fan-out: `_postprocess_one_task` gathers across every task,
each of which may trigger one judge + up to three variant calls. At scale
that's O(100s) of concurrent API calls. Anthropic handles this easily, but
an unbounded fan-out on a runaway resume could still saturate the org's
RPM. `Semaphore(250)` is bounded-but-non-blocking at realistic scale.

The semaphore is lazy-initialized via `get_api_semaphore()` so it binds to
whichever event loop is live when first accessed. A module-level
`asyncio.Semaphore()` created at import time binds to the first loop that
uses it and can raise `RuntimeError: ... bound to a different loop` when
reused across distinct `asyncio.run()` calls in the same process (test
harnesses, composite phase runners).
"""

from __future__ import annotations

import asyncio

_semaphore: asyncio.Semaphore | None = None


def get_api_semaphore() -> asyncio.Semaphore:
    """Return the process-wide Phase 4 API semaphore, creating on first call.

    Call-site pattern:
        async with get_api_semaphore():
            ...
    """
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(250)
    return _semaphore


def reset_api_semaphore_for_tests() -> None:
    """Drop the cached semaphore so the next `get_api_semaphore()` rebinds.

    Tests that create a new event loop between cases need this to avoid
    `RuntimeError: ... bound to a different loop`.
    """
    global _semaphore
    _semaphore = None
