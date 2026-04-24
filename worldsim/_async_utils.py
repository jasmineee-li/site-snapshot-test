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

# Kinds that represent transient network/infrastructure failures worth
# retrying. ``request_failed`` covers generic 4xx/5xx that slipped past
# per-editor classification; ``unreachable`` covers DNS / connect-timeout
# style network errors. ``cleanup_failed`` covers the case where
# ``BaseSiteEditor.cleanup()`` tried to tear down partial state mid-chain
# and the DELETEs themselves hit the same transient 5xx window — that is
# also a retryable situation because the next attempt will re-run the
# whole chain from scratch (editors like GitLab's ``create_group`` and
# ``create_issue`` already implement GET-then-POST for idempotency, and
# ``create_project`` reaps stale ``webagent-task-*`` projects before
# re-creating).
#
# 4xx-class kinds (``length_exceeded``, ``field_required``,
# ``content_policy``, ``auth_missing``, ``project_already_exists``, etc.)
# are never retried: intentional platform rejection is the answer we want
# to record.
_DEFAULT_RETRY_KINDS: tuple[str, ...] = (
    "request_failed",
    "unreachable",
    "cleanup_failed",
)


async def retrying[T](
    factory: Callable[[], Awaitable[T]],
    *,
    retries: int = 1,
    backoff_base_seconds: float = 1.0,
    attempts_log: list[dict[str, Any]] | None = None,
    retry_on: tuple[str, ...] = _DEFAULT_RETRY_KINDS,
    on_retry: Callable[[EditorError], Awaitable[None]] | None = None,
) -> T:
    """Invoke ``factory()`` up to ``retries + 1`` times.

    Retries only when the awaited call raises ``EditorError`` whose ``kind``
    is listed in ``retry_on``. See ``_DEFAULT_RETRY_KINDS`` for the default
    set and the rationale.

    Backoff is ``backoff_base_seconds * 2 ** attempt``.

    When ``attempts_log`` is supplied, one dict is appended per attempt with
    keys ``attempt`` (0-indexed), ``status`` (``"success"``, ``"retrying"``,
    or the EditorError kind), and ``elapsed_ms``.

    When ``on_retry`` is supplied, it is awaited *between attempts* — after
    a retryable failure, before the backoff sleep. Use it to sweep residual
    platform state that the factory's own unwind couldn't clean (e.g. a
    partially-created resource whose DELETE also 5xx'd). The callback must
    not raise; if it does, the original retryable exception takes
    precedence and the hook error is logged at WARNING.
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
            if on_retry is not None:
                try:
                    await on_retry(exc)
                except Exception:  # pragma: no cover - defensive
                    logger.warning(
                        "on_retry hook raised; continuing with backoff",
                        exc_info=True,
                    )
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
