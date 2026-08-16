"""Transport-level retry helper for OpenRouter / OpenAI SDK calls.

Adapted from `worldsim/phase_4/anthropic_client.py::call_with_retry` on
the v5 branch. Differences:

- Uses OpenAI SDK exception types (`openai.APIStatusError`,
  `openai.APIConnectionError`, `openai.APITimeoutError`) since
  data-import goes through OpenRouter via `openai.AsyncOpenAI`, not
  Anthropic SDK.
- Same status-code triage (429, 500, 502, 503, 529 retry; everything
  else bubbles up).
- Same jittered exponential backoff.
- Same `Retry-After` header handling.

The retried function should acquire its concurrency semaphore *inside*
its body so the slot is released during backoff sleeps. Callers that
acquire the semaphore around `call_with_retry` will hold a slot through
every retry, which starves other concurrent calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from openai import APIConnectionError, APIStatusError, APITimeoutError

T = TypeVar("T")

logger = logging.getLogger(__name__)

# 429 = rate limited
# 500/502/503 = server errors (transient)
# 529 = Anthropic "overloaded" passed through OpenRouter
_RETRYABLE_STATUS = {429, 500, 502, 503, 529}


def classify_api_exception(exc: BaseException) -> str:
    """Bucket an exception into a failure-class label.

    Matches v5's classify_api_exception buckets where status codes
    overlap; adds connection_error / timeout / unknown_error for OpenAI
    SDK exception classes that don't subclass APIStatusError.
    """
    if isinstance(exc, APIStatusError):
        if exc.status_code == 401:
            return "auth_invalid"
        if exc.status_code == 402:
            return "insufficient_credits"
        if exc.status_code == 403:
            return "quota_exceeded"
        if exc.status_code == 429:
            return "rate_limited"
        if 500 <= exc.status_code < 600:
            return f"server_error_{exc.status_code}"
        return f"api_error_{exc.status_code}"
    if isinstance(exc, APITimeoutError):
        return "timeout"
    if isinstance(exc, APIConnectionError):
        return "connection_error"
    return "unknown_error"


async def call_with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    retries: int = 3,
    base_delay: float = 1.0,
    label: str = "",
) -> T:
    """Wrap `fn` with jittered exponential backoff for transient API failures.

    Retries on:
        - APIStatusError with status in {429, 500, 502, 503, 529}
        - APIConnectionError (transient network)
        - APITimeoutError (request timed out)
        - JSONDecodeError from malformed provider/transport response bodies

    Bubbles up:
        - APIStatusError with any other status (4xx auth/billing/quota,
          malformed requests, programming errors)
        - All other exception types

    Total attempts = retries + 1 (i.e. retries=3 means 4 attempts total).

    Respects the server `Retry-After` header on rate-limit responses
    when it's longer than our jittered backoff.
    """
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return await fn()
        except APIStatusError as exc:
            last_exc = exc
            if exc.status_code not in _RETRYABLE_STATUS:
                raise
            if attempt >= retries:
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
            try:
                hdr = getattr(exc.response, "headers", None)
                if hdr is not None:
                    raw = hdr.get("Retry-After") or hdr.get("retry-after") or "0"
                    retry_after = float(raw)
                    if retry_after > delay:
                        delay = retry_after
            except (TypeError, ValueError, AttributeError):
                pass
            logger.warning(
                f"[{label}] http {exc.status_code}; "
                f"retry {attempt + 1}/{retries} after {delay:.1f}s"
            )
            await asyncio.sleep(delay)
            continue
        except (APIConnectionError, APITimeoutError) as exc:
            last_exc = exc
            if attempt >= retries:
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
            logger.warning(
                f"[{label}] {type(exc).__name__}; "
                f"retry {attempt + 1}/{retries} after {delay:.1f}s: {exc}"
            )
            await asyncio.sleep(delay)
            continue
        except json.JSONDecodeError as exc:
            last_exc = exc
            if attempt >= retries:
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
            logger.warning(
                f"[{label}] JSONDecodeError; "
                f"retry {attempt + 1}/{retries} after {delay:.1f}s: {exc}"
            )
            await asyncio.sleep(delay)
            continue
    # Unreachable: the loop either returns a value or re-raises.
    assert last_exc is not None
    raise last_exc
