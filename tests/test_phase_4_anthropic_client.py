"""Coverage for `worldsim.phase_4.anthropic_client`.

The three auth paths (OpenRouter / OAuth / API key) each construct
`AsyncAnthropic` with a different kwarg shape. These tests verify the
construction rather than the underlying SDK behavior — a kwarg rename in
a future anthropic SDK bump should break these tests loudly instead of
silently misconfiguring the client.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from anthropic import APIStatusError

from worldsim.phase_4 import anthropic_client


@pytest.fixture(autouse=True)
def clear_client(monkeypatch):
    """Every test starts with a fresh cached client and a clean env."""
    for name in (
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    anthropic_client.reset_client_for_tests()
    yield
    anthropic_client.reset_client_for_tests()


def _spy_async_anthropic(monkeypatch):
    """Replace `AsyncAnthropic` with a MagicMock so we can inspect kwargs."""
    spy = MagicMock()
    monkeypatch.setattr(anthropic_client, "AsyncAnthropic", spy)
    return spy


def test_openrouter_path_passes_auth_token_and_base_url(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "sk-or-v1-test")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://openrouter.ai/api/")
    spy = _spy_async_anthropic(monkeypatch)

    anthropic_client.get_client()

    spy.assert_called_once()
    _, kwargs = spy.call_args
    assert kwargs["auth_token"] == "sk-or-v1-test"
    # Trailing slash stripped; /v1/messages appended by the SDK.
    assert kwargs["base_url"] == "https://openrouter.ai/api"
    assert "api_key" not in kwargs
    assert kwargs["max_retries"] == 5


def test_oauth_path_passes_auth_token_only(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "claude-oauth-xxx")
    spy = _spy_async_anthropic(monkeypatch)

    anthropic_client.get_client()

    spy.assert_called_once()
    _, kwargs = spy.call_args
    assert kwargs["auth_token"] == "claude-oauth-xxx"
    assert "base_url" not in kwargs  # direct to api.anthropic.com
    assert "api_key" not in kwargs
    assert kwargs["max_retries"] == 5


def test_api_key_path_passes_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    spy = _spy_async_anthropic(monkeypatch)

    anthropic_client.get_client()

    spy.assert_called_once()
    _, kwargs = spy.call_args
    assert kwargs["api_key"] == "sk-ant-test"
    assert "auth_token" not in kwargs
    assert "base_url" not in kwargs
    assert kwargs["max_retries"] == 5


def test_openrouter_wins_over_oauth_and_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "sk-or-v1-test")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://openrouter.ai/api")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "claude-oauth-xxx")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    spy = _spy_async_anthropic(monkeypatch)

    anthropic_client.get_client()

    _, kwargs = spy.call_args
    assert kwargs["auth_token"] == "sk-or-v1-test"


def test_oauth_wins_over_api_key(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "claude-oauth-xxx")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    spy = _spy_async_anthropic(monkeypatch)

    anthropic_client.get_client()

    _, kwargs = spy.call_args
    assert kwargs["auth_token"] == "claude-oauth-xxx"


def test_empty_string_envs_treated_as_unset(monkeypatch):
    # ANTHROPIC_AUTH_TOKEN set to empty — should not engage the openrouter
    # path; OAuth should win.
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "  ")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "claude-oauth-xxx")
    spy = _spy_async_anthropic(monkeypatch)

    anthropic_client.get_client()

    _, kwargs = spy.call_args
    assert kwargs["auth_token"] == "claude-oauth-xxx"
    assert "base_url" not in kwargs


def test_no_creds_raises_actionable_runtime_error(monkeypatch):
    _spy_async_anthropic(monkeypatch)

    with pytest.raises(RuntimeError) as excinfo:
        anthropic_client.get_client()

    msg = str(excinfo.value)
    assert "No Claude credentials configured" in msg
    assert "CLAUDE_CODE_OAUTH_TOKEN" in msg
    assert "ANTHROPIC_AUTH_TOKEN" in msg
    assert "ANTHROPIC_API_KEY" in msg


def _status_error(status_code: int, message: str = "synthetic") -> APIStatusError:
    import httpx

    request = httpx.Request("POST", "https://example.test/v1/messages")
    response = httpx.Response(status_code, request=request)
    return APIStatusError(
        message=message,
        response=response,
        body={"error": {"code": status_code}},
    )


def test_classify_api_exception_buckets_401_as_auth_invalid():
    assert anthropic_client.classify_api_exception(_status_error(401)) == "auth_invalid"


def test_classify_api_exception_buckets_402_as_insufficient_credits():
    assert anthropic_client.classify_api_exception(_status_error(402)) == "insufficient_credits"


def test_classify_api_exception_buckets_403_as_quota_exceeded():
    assert anthropic_client.classify_api_exception(_status_error(403)) == "quota_exceeded"


def test_classify_api_exception_buckets_other_statuses_as_api_error():
    assert anthropic_client.classify_api_exception(_status_error(500)) == "api_error"
    assert anthropic_client.classify_api_exception(_status_error(429)) == "api_error"


def test_classify_api_exception_buckets_non_status_errors_as_api_error():
    assert anthropic_client.classify_api_exception(RuntimeError("other")) == "api_error"
    assert anthropic_client.classify_api_exception(ValueError("boom")) == "api_error"


@pytest.mark.asyncio
async def test_preflight_check_returns_true_on_ok(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    fake_client = MagicMock()

    async def _ok(**kwargs: Any) -> Any:
        return MagicMock()

    fake_client.messages.create = _ok
    monkeypatch.setattr(anthropic_client, "get_client", lambda: fake_client)

    ok, err = await anthropic_client.preflight_check()
    assert ok is True
    assert err is None


@pytest.mark.asyncio
async def test_preflight_check_returns_false_on_402(monkeypatch):
    import httpx

    fake_client = MagicMock()
    request = httpx.Request("POST", "https://example.test/v1/messages")
    response = httpx.Response(402, request=request)

    async def _402(**kwargs: Any) -> Any:
        raise APIStatusError(
            message="Insufficient credits",
            response=response,
            body={"error": {"code": 402}},
        )

    fake_client.messages.create = _402
    monkeypatch.setattr(anthropic_client, "get_client", lambda: fake_client)

    ok, err = await anthropic_client.preflight_check()
    assert ok is False
    assert err is not None
    assert "insufficient_credits" in err


@pytest.mark.asyncio
async def test_preflight_check_returns_false_on_no_creds(monkeypatch):
    ok, err = await anthropic_client.preflight_check()
    assert ok is False
    assert err is not None
    assert "No Claude credentials configured" in err


@pytest.mark.asyncio
async def test_oauth_token_sent_as_authorization_bearer_on_wire():
    """End-to-end verification that `auth_token=` produces an
    `Authorization: Bearer <token>` header at the transport layer.

    This is what Phase 4's `CLAUDE_CODE_OAUTH_TOKEN` path is
    load-bearing on — a future anthropic SDK bump that changed
    `auth_token` semantics would pass every kwarg-level mock test and
    break production. Use httpx.MockTransport to intercept the real
    request the SDK would send.
    """
    import httpx
    from anthropic import AsyncAnthropic

    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update({k.lower(): v for k, v in request.headers.items()})
        return httpx.Response(
            200,
            json={
                "id": "msg_oauth_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        client = AsyncAnthropic(auth_token="test-oauth-tok", http_client=http_client)
        await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1,
            messages=[{"role": "user", "content": "."}],
        )
    finally:
        await http_client.aclose()

    assert captured.get("authorization") == "Bearer test-oauth-tok", (
        f"expected 'Authorization: Bearer test-oauth-tok'; got {captured.get('authorization')!r}"
    )
    # x-api-key must NOT be set when auth_token is the auth mode.
    assert "x-api-key" not in captured, (
        f"x-api-key leaked into request headers: {captured.get('x-api-key')!r}"
    )


@pytest.mark.asyncio
async def test_call_with_retry_honors_retry_after_header(monkeypatch):
    """A 529 with `Retry-After: N` must wait at least N seconds even if
    the jittered exponential backoff would wait less. Ignoring the hint
    causes immediate re-fire and likely another 529."""
    import httpx

    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(anthropic_client.asyncio, "sleep", fake_sleep)

    request = httpx.Request("POST", "https://example.test/v1/messages")
    response = httpx.Response(529, headers={"Retry-After": "7"}, request=request)

    calls = {"n": 0}

    async def flaky() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise APIStatusError(message="overloaded", response=response, body=None)
        return "ok"

    result = await anthropic_client.call_with_retry(
        flaky, retries=3, base_delay=1.0, label="retry-after-test"
    )
    assert result == "ok"
    assert sleeps, "expected at least one sleep before the retry"
    # First (and only) retry delay must be ≥ Retry-After hint.
    assert sleeps[0] >= 7.0, f"retry slept {sleeps[0]}s, expected ≥7s per Retry-After header"
