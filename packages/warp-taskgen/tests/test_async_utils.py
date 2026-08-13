"""Unit tests for ``warp_taskgen._async_utils.retrying``.

Covers the transient-error retry policy that Phase 2c relies on, plus
the optional ``on_retry`` hook used to sweep partial platform state
between attempts.
"""

from __future__ import annotations

import asyncio

import pytest

from warp_taskgen._async_utils import _DEFAULT_RETRY_KINDS, retrying
from warp_taskgen.editors import EditorError


def test_default_retry_kinds_include_transient_and_cleanup():
    # These three kinds all represent transient infrastructure failures;
    # 4xx platform rejections must NOT be in the default set.
    assert "request_failed" in _DEFAULT_RETRY_KINDS
    assert "unreachable" in _DEFAULT_RETRY_KINDS
    assert "cleanup_failed" in _DEFAULT_RETRY_KINDS
    # Platform rejections that should never retry:
    for bad in (
        "length_exceeded",
        "field_required",
        "content_policy",
        "auth_missing",
        "project_already_exists",
        "unsupported_method",
    ):
        assert bad not in _DEFAULT_RETRY_KINDS, bad


def test_succeeds_on_first_attempt():
    async def factory():
        return "ok"

    attempts: list = []
    result = asyncio.run(retrying(factory, retries=2, attempts_log=attempts))
    assert result == "ok"
    assert attempts == [
        {"attempt": 0, "status": "success", "elapsed_ms": attempts[0]["elapsed_ms"]}
    ]


def test_retries_request_failed_then_succeeds():
    counter = {"n": 0}

    async def factory():
        counter["n"] += 1
        if counter["n"] == 1:
            raise EditorError("request_failed", "transient 502")
        return "ok"

    attempts: list = []
    result = asyncio.run(
        retrying(factory, retries=2, backoff_base_seconds=0.0, attempts_log=attempts)
    )
    assert result == "ok"
    assert counter["n"] == 2
    assert attempts[0]["status"] == "request_failed"
    assert attempts[1]["status"] == "success"


def test_retries_cleanup_failed_by_default():
    """Bug #5 fix regression: cleanup_failed is now treated as transient."""
    counter = {"n": 0}

    async def factory():
        counter["n"] += 1
        if counter["n"] == 1:
            raise EditorError("cleanup_failed", "delete returned 502")
        return "ok"

    attempts: list = []
    result = asyncio.run(
        retrying(factory, retries=1, backoff_base_seconds=0.0, attempts_log=attempts)
    )
    assert result == "ok"
    assert counter["n"] == 2
    assert attempts[0]["status"] == "cleanup_failed"


def test_does_not_retry_4xx_kinds():
    counter = {"n": 0}

    async def factory():
        counter["n"] += 1
        raise EditorError("length_exceeded", "255 char max")

    with pytest.raises(EditorError) as exc_info:
        asyncio.run(retrying(factory, retries=3, backoff_base_seconds=0.0))
    assert exc_info.value.kind == "length_exceeded"
    # Exactly one attempt — no retry.
    assert counter["n"] == 1


def test_exhausts_retries_and_raises_last():
    counter = {"n": 0}

    async def factory():
        counter["n"] += 1
        raise EditorError("request_failed", f"attempt-{counter['n']}")

    with pytest.raises(EditorError) as exc_info:
        asyncio.run(retrying(factory, retries=2, backoff_base_seconds=0.0))
    assert exc_info.value.kind == "request_failed"
    # retries=2 → 3 total attempts (attempt 0, 1, 2).
    assert counter["n"] == 3
    assert "attempt-3" in exc_info.value.detail


def test_on_retry_hook_fires_between_attempts():
    attempts_to_fail = {"n": 2}
    hook_calls: list[EditorError] = []

    async def factory():
        attempts_to_fail["n"] -= 1
        if attempts_to_fail["n"] >= 0:
            raise EditorError("cleanup_failed", f"still-failing-{attempts_to_fail['n']}")
        return "final_ok"

    async def on_retry(exc: EditorError) -> None:
        hook_calls.append(exc)

    result = asyncio.run(
        retrying(
            factory,
            retries=3,
            backoff_base_seconds=0.0,
            on_retry=on_retry,
        )
    )
    assert result == "final_ok"
    # attempts_to_fail starts at 2: factory decrements before the guard,
    # so attempts 0 and 1 raise (n=1, n=0) and attempt 2 returns (n=-1).
    # Hook fires once per failed-then-retried attempt → 2 firings.
    assert len(hook_calls) == 2
    assert all(exc.kind == "cleanup_failed" for exc in hook_calls)


def test_on_retry_hook_not_fired_on_non_retryable_kind():
    hook_calls: list[EditorError] = []

    async def factory():
        raise EditorError("auth_missing", "401")

    async def on_retry(exc: EditorError) -> None:
        hook_calls.append(exc)

    with pytest.raises(EditorError):
        asyncio.run(
            retrying(
                factory,
                retries=2,
                backoff_base_seconds=0.0,
                on_retry=on_retry,
            )
        )
    # Non-retryable kind raises immediately without firing the hook.
    assert hook_calls == []


def test_on_retry_hook_not_fired_on_final_attempt():
    """Last attempt's failure must raise immediately — the hook only fires
    when a subsequent retry is actually scheduled."""
    hook_calls: list[EditorError] = []

    async def factory():
        raise EditorError("request_failed", "persistent 502")

    async def on_retry(exc: EditorError) -> None:
        hook_calls.append(exc)

    with pytest.raises(EditorError):
        asyncio.run(
            retrying(
                factory,
                retries=2,
                backoff_base_seconds=0.0,
                on_retry=on_retry,
            )
        )
    # retries=2 means 3 attempts total. Hook fires between attempts (after
    # attempt 0 → retry, after attempt 1 → retry), but NOT after attempt 2
    # which simply raises.
    assert len(hook_calls) == 2


def test_on_retry_hook_exception_does_not_mask_original_error():
    async def factory():
        raise EditorError("request_failed", "transient")

    async def bad_hook(exc: EditorError) -> None:
        raise ValueError("hook is broken")

    with pytest.raises(EditorError) as exc_info:
        asyncio.run(
            retrying(
                factory,
                retries=1,
                backoff_base_seconds=0.0,
                on_retry=bad_hook,
            )
        )
    # The final EditorError ("request_failed") surfaces, not the hook's
    # ValueError — the hook's job is best-effort cleanup, not to override
    # the real failure reason.
    assert exc_info.value.kind == "request_failed"


def test_custom_retry_on_overrides_default():
    counter = {"n": 0}

    async def factory():
        counter["n"] += 1
        if counter["n"] == 1:
            raise EditorError("length_exceeded", "just this once")
        return "ok"

    result = asyncio.run(
        retrying(
            factory,
            retries=1,
            backoff_base_seconds=0.0,
            retry_on=("length_exceeded",),
        )
    )
    assert result == "ok"
    assert counter["n"] == 2


def test_attempts_log_records_every_attempt():
    counter = {"n": 0}

    async def factory():
        counter["n"] += 1
        if counter["n"] < 3:
            raise EditorError("request_failed", f"try-{counter['n']}")
        return "ok"

    attempts: list = []
    result = asyncio.run(
        retrying(
            factory,
            retries=3,
            backoff_base_seconds=0.0,
            attempts_log=attempts,
        )
    )
    assert result == "ok"
    assert [a["status"] for a in attempts] == ["request_failed", "request_failed", "success"]
    assert [a["attempt"] for a in attempts] == [0, 1, 2]
    assert all("elapsed_ms" in a for a in attempts)
