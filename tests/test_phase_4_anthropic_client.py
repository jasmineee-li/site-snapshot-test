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


def test_classify_api_exception_buckets_402(monkeypatch):
    import httpx

    request = httpx.Request("POST", "https://example.test/v1/messages")
    response = httpx.Response(402, request=request)

    exc = APIStatusError(
        message="Insufficient credits",
        response=response,
        body={"error": {"code": 402}},
    )
    assert anthropic_client.classify_api_exception(exc) == "insufficient_credits"


def test_classify_api_exception_buckets_other_statuses_as_api_error():
    import httpx

    request = httpx.Request("POST", "https://example.test/v1/messages")
    response = httpx.Response(500, request=request)

    exc = APIStatusError(
        message="boom",
        response=response,
        body={"error": {"code": 500}},
    )
    assert anthropic_client.classify_api_exception(exc) == "api_error"


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
