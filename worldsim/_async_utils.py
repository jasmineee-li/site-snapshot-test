"""Internal async helpers shared across the pipeline.

This module is deliberately underscore-prefixed: callers inside ``worldsim/``
are welcome, but the surface is not a public API.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from worldsim.editors import EditorError

logger = logging.getLogger(__name__)

T = TypeVar("T")

_DEFAULT_RETRY_KINDS: tuple[str, ...] = ("request_failed", "unreachable")


async def retrying(
    factory: Callable[[], Awaitable[T]],
    *,
    retries: int = 1,
    backoff_base_seconds: float = 1.0,
    attempts_log: list[dict[str, Any]] | None = None,
    retry_on: tuple[str, ...] = _DEFAULT_RETRY_KINDS,
) -> T:
    """Invoke ``factory()`` up to ``retries + 1`` times.

    Retries only when the awaited call raises ``EditorError`` whose ``kind`` is
    listed in ``retry_on``. 4xx-class ``EditorError`` kinds (e.g.
    ``length_exceeded``, ``field_required``, ``content_policy``,
    ``auth_missing``) are never retried: intentional platform rejection is
    the answer we want to record.

    Backoff is ``backoff_base_seconds * 2 ** attempt``.

    When ``attempts_log`` is supplied, one dict is appended per attempt with
    keys ``attempt`` (0-indexed), ``status`` (``"success"``, ``"retrying"``,
    or the EditorError kind), and ``elapsed_ms``.
    """
    last_exc: EditorError | None = None
    for attempt in range(retries + 1):
        started = time.monotonic()
        try:
            result = await factory()
        except EditorError as exc:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            if attempts_log is not None:
                attempts_log.append(
                    {
                        "attempt": attempt,
                        "status": exc.kind,
                        "elapsed_ms": elapsed_ms,
                    }
                )
            last_exc = exc
            if exc.kind not in retry_on or attempt == retries:
                raise
            delay = backoff_base_seconds * (2**attempt)
            logger.debug(
                "retrying after %s (attempt %d/%d, backoff=%.2fs): %s",
                exc.kind,
                attempt + 1,
                retries,
                delay,
                exc.detail,
            )
            await asyncio.sleep(delay)
            continue
        elapsed_ms = int((time.monotonic() - started) * 1000)
        if attempts_log is not None:
            attempts_log.append(
                {
                    "attempt": attempt,
                    "status": "success",
                    "elapsed_ms": elapsed_ms,
                }
            )
        return result
    # Unreachable: the loop either returns or re-raises.
    assert last_exc is not None
    raise last_exc
