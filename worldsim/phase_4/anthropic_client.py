"""Shared `anthropic.AsyncAnthropic` client for Phase 4 host-side API calls.

Auth precedence mirrors `worldsim.modal_sandbox._build_claude_secrets` and
`worldsim.phases.phase_2_text_fill._call_anthropic_fallback`:

    1. OpenRouter  — `ANTHROPIC_AUTH_TOKEN` + `ANTHROPIC_BASE_URL`
    2. Anthropic direct + OAuth — `CLAUDE_CODE_OAUTH_TOKEN`
    3. Anthropic direct + API key — `ANTHROPIC_API_KEY`

Empty-string env vars are treated as unset (matches sandbox convention at
`modal_sandbox.py:124`).

The client is module-level lazy so Phase 4's judge + variant calls share one
TCP connection pool. SDK retries are disabled so the host-side retry helper
can release API semaphore slots before backing off.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from anthropic import APIConnectionError, APIStatusError, AsyncAnthropic

logger = logging.getLogger(__name__)

_client: AsyncAnthropic | None = None
_SDK_MAX_RETRIES = 0

T = TypeVar("T")


def _nonempty(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def classify_api_exception(exc: BaseException) -> str:
    """Bucket an API exception into a phase_4 `failure_class` label.

    Auth/billing status codes get their own buckets because they are
    operationally distinct from transient API failures — an operator
    triaging a post-run log needs to know whether to rotate a token,
    top up credits, or just retry.
    """
    if isinstance(exc, APIStatusError):
        if exc.status_code == 401:
            return "auth_invalid"
        if exc.status_code == 402:
            return "insufficient_credits"
        if exc.status_code == 403:
            return "quota_exceeded"
    return "api_error"


def _resolved_auth_path() -> str | None:
    """Return the auth path that `get_client` would pick, without raising.

    `"openrouter"` if OpenRouter envs are set, `"oauth"` if only
    OAuth is set, `"anthropic_api"` for API-key, or None if no
    credentials are available. Used by `normalize_model_for_auth` to
    strip OpenRouter-style `vendor/` prefixes before hitting
    api.anthropic.com.
    """
    auth_token = _nonempty("ANTHROPIC_AUTH_TOKEN")
    base_url = _nonempty("ANTHROPIC_BASE_URL")
    if auth_token and base_url:
        return "openrouter"
    if _nonempty("CLAUDE_CODE_OAUTH_TOKEN"):
        return "oauth"
    if _nonempty("ANTHROPIC_API_KEY"):
        return "anthropic_api"
    return None


def normalize_model_for_auth(model: str) -> str:
    """Strip `vendor/` prefix when the active auth path is Anthropic-direct.

    OpenRouter uses `anthropic/claude-sonnet-4-6` naming. Anthropic-direct
    (`api.anthropic.com`) rejects that with "unknown model" and expects
    the bare `claude-sonnet-4-6`. Mirrors
    `worldsim.phases.phase_2_text_fill._direct_anthropic_model_name` so a
    single `--sandbox-model` flag value works against either endpoint.

    When no credentials are configured (`_resolved_auth_path()` returns
    None), we leave the model string unchanged — the downstream SDK call
    will raise a clearer `RuntimeError` via `get_client()` anyway.
    """
    if not model:
        return model
    auth_path = _resolved_auth_path()
    if auth_path in ("oauth", "anthropic_api") and "/" in model:
        return model.split("/", 1)[1]
    # OpenRouter keeps `vendor/model` naming. No creds resolves to no
    # normalization either — `get_client()` will raise cleanly on the
    # missing-credentials path, and we don't want to mask that by
    # silently mutating the model string.
    return model


def _resolve_auth() -> tuple[str, dict[str, Any]]:
    """Return `(auth_path, kwargs)` for `AsyncAnthropic(**kwargs)`.

    Raises RuntimeError with an actionable message if no credentials are
    configured.
    """
    auth_token = _nonempty("ANTHROPIC_AUTH_TOKEN")
    base_url = _nonempty("ANTHROPIC_BASE_URL")
    if auth_token and base_url:
        # OpenRouter Anthropic-compatible endpoint at `<base>/v1/messages`.
        # AsyncAnthropic appends `/v1/messages` to base_url internally, so
        # pass just the proxy root (e.g. https://openrouter.ai/api — same
        # convention used by `modal_sandbox._build_claude_secrets`).
        return (
            "openrouter",
            {
                "auth_token": auth_token,
                "base_url": base_url.rstrip("/"),
                "max_retries": _SDK_MAX_RETRIES,
            },
        )

    oauth = _nonempty("CLAUDE_CODE_OAUTH_TOKEN")
    if oauth:
        # Anthropic direct with OAuth — Bearer header path.
        return ("oauth", {"auth_token": oauth, "max_retries": _SDK_MAX_RETRIES})

    api_key = _nonempty("ANTHROPIC_API_KEY")
    if api_key:
        return ("anthropic_api", {"api_key": api_key, "max_retries": _SDK_MAX_RETRIES})

    raise RuntimeError(
        "No Claude credentials configured. Phase 4 judge + variant gen run "
        "on the host via the Anthropic Messages API. Set one of:\n"
        "  CLAUDE_CODE_OAUTH_TOKEN=... (Claude Pro/Max)\n"
        "  ANTHROPIC_AUTH_TOKEN=sk-or-v1-... + ANTHROPIC_BASE_URL=https://openrouter.ai/api\n"
        "  ANTHROPIC_API_KEY=sk-ant-...\n"
        "See .env.example."
    )


def get_client() -> AsyncAnthropic:
    """Return the lazy-initialized shared `AsyncAnthropic` client."""
    global _client
    if _client is None:
        _, kwargs = _resolve_auth()
        _client = AsyncAnthropic(**kwargs)
    return _client


def reset_client_for_tests() -> None:
    """Force re-initialization on next `get_client()`.

    Tests that monkeypatch env vars after module import need this to avoid
    a stale client bound to the previous auth config.
    """
    global _client
    _client = None


async def preflight_check(*, sandbox_model: str = "claude-sonnet-4-6") -> tuple[bool, str | None]:
    """Probe the live endpoint with a ~1-token call before burning eval cost.

    Returns `(ok, error_message)`. On success, returns `(True, None)`.
    On failure, returns `(False, "<actionable message>")`. Never raises.

    Rationale: an exhausted or misconfigured token otherwise manifests as
    one 402/401 per task, AFTER Browser-Use has already spent its eval
    cost. Pre-flight cost is ~$0.0001 and the failure mode is bucketed
    distinctly (`insufficient_credits` vs `api_error`) so post-run triage
    can tell them apart.
    """
    try:
        client = get_client()
    except RuntimeError as exc:
        return (False, f"preflight failed: {exc}")

    try:
        await call_with_retry(
            lambda: client.messages.create(
                model=normalize_model_for_auth(sandbox_model),
                max_tokens=1,
                messages=[{"role": "user", "content": "."}],
            ),
            retries=3,
            label="phase4-preflight",
        )
        return (True, None)
    except APIStatusError as exc:
        failure_class = (
            "insufficient_credits" if exc.status_code == 402 else f"api_error({exc.status_code})"
        )
        return (
            False,
            f"preflight failed ({failure_class}): {exc}. Check auth env vars; "
            "see .env.example for the three supported auth paths.",
        )
    except Exception as exc:
        return (False, f"preflight failed (api_error): {exc}")


async def call_with_retry[T](
    fn: Callable[[], Awaitable[T]],
    *,
    retries: int = 3,
    base_delay: float = 1.0,
    label: str = "",
) -> T:
    """Wrap `fn` with jittered exponential backoff for 529 overloaded responses.

    The Anthropic SDK retry layer is disabled (`max_retries=0`) so callers
    can release their API semaphore slot before sleeping. This helper is the
    single transport retry policy for 429/500/502/503/529 plus
    `APIConnectionError`.
    """
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return await fn()
        except APIStatusError as exc:
            last_exc = exc
            if exc.status_code in (429, 500, 502, 503, 529):
                if attempt >= retries:
                    raise
                delay = base_delay * (2**attempt) + random.uniform(0, 0.5)
                # Respect server `Retry-After` hint when it's longer than
                # our jittered backoff. Anthropic returns seconds as an
                # integer for 429/529; tolerate non-numeric values.
                retry_after = 0.0
                try:
                    hdr = getattr(exc.response, "headers", None)
                    if hdr is not None:
                        value = hdr.get("Retry-After") or hdr.get("retry-after") or "0"
                        retry_after = float(value)
                except (TypeError, ValueError):
                    retry_after = 0.0
                if retry_after > delay:
                    delay = retry_after
                logger.warning(
                    "[%s] anthropic %s; retry %d/%d after %.1fs",
                    label,
                    exc.status_code,
                    attempt + 1,
                    retries,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            raise
        except APIConnectionError as exc:
            last_exc = exc
            if attempt >= retries:
                raise
            delay = base_delay * (2**attempt) + random.uniform(0, 0.5)
            logger.warning(
                "[%s] anthropic connection error; retry %d/%d after %.1fs: %s",
                label,
                attempt + 1,
                retries,
                delay,
                exc,
            )
            await asyncio.sleep(delay)
            continue
    assert last_exc is not None
    raise last_exc
