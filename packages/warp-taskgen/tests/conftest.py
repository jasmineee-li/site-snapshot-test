from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_PHASE_4_MODULES_WITH_GET_CLIENT: list[Any] | None = None


def _make_tool_use_response(
    tool_name: str,
    tool_input: dict[str, Any],
    *,
    stop_reason: str = "tool_use",
    input_tokens: int = 1000,
    output_tokens: int = 200,
) -> SimpleNamespace:
    """Shape a fake Anthropic Messages API response.

    Matches the parts Phase 4 code reads: `.content[*].type`, `.content[*].name`,
    `.content[*].input`, `.stop_reason`, `.usage.input_tokens`,
    `.usage.output_tokens`, `.id`, `.model`.
    """
    block = SimpleNamespace(type="tool_use", name=tool_name, input=dict(tool_input))
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    return SimpleNamespace(
        content=[block],
        stop_reason=stop_reason,
        usage=usage,
        id="msg_test_fake",
        model="claude-sonnet-4-6",
    )


@pytest.fixture
def fake_anthropic_response():
    """Factory for building fake AsyncAnthropic.messages.create return values.

    Usage:
        def test_x(fake_anthropic_response):
            resp = fake_anthropic_response("classify_refusal", {"refusal_trigger": "suspicion", "confidence": "high"})
    """
    return _make_tool_use_response


class _FakeStreamContext:
    """Async context manager that yields an object with `get_final_message()`.

    Matches the shape Phase 4 `variant_api._call` expects:
        async with client.messages.stream(...) as stream:
            return await stream.get_final_message()
    """

    def __init__(self, final_message: Any) -> None:
        self._final_message = final_message

    async def __aenter__(self) -> _FakeStreamContext:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def get_final_message(self) -> Any:
        # Re-raise if the "response" is actually an exception — lets
        # side_effect mimic API errors through the streaming path.
        if isinstance(self._final_message, BaseException):
            raise self._final_message
        return self._final_message


def _install_stream_mock(mock_client: MagicMock) -> None:
    """Wire mock_client.messages.stream to a context manager driven by
    mock_client.messages.create's return_value / side_effect.

    This lets tests keep the familiar `mock.messages.create.return_value =
    response` pattern even though production code calls `.stream(...)`.
    `create` remains callable (as an AsyncMock) for anything that still
    uses the non-streaming path.
    """

    # Keep a private iterator for list-style side_effect so we don't
    # mutate the user's configuration between calls.
    state: dict[str, Any] = {"side_effect_iter": None, "side_effect_source": None}

    def stream_factory(**_kwargs: Any) -> _FakeStreamContext:
        create_mock = mock_client.messages.create
        effect = create_mock.side_effect
        if effect is not None:
            if isinstance(effect, BaseException) or (
                isinstance(effect, type) and issubclass(effect, BaseException)
            ):
                # Bare exception (instance or class) set as side_effect:
                # _FakeStreamContext.get_final_message re-raises BaseException
                # so this drives the streaming path through the exception
                # just like AsyncMock would through create().
                return _FakeStreamContext(effect)
            if callable(effect) and not isinstance(effect, (list, tuple)):
                return _FakeStreamContext(effect(**_kwargs))
            # Iterable side_effect: cache an iterator so each call advances.
            # Re-prime if the user swapped in a new iterable.
            if state["side_effect_source"] is not effect:
                state["side_effect_source"] = effect
                state["side_effect_iter"] = iter(effect)
            try:
                return _FakeStreamContext(next(state["side_effect_iter"]))
            except StopIteration:
                raise AssertionError(
                    "mock_client.messages.create.side_effect list exhausted"
                ) from None
        # Fall back to return_value. AsyncMock's default return_value is a
        # MagicMock (not None), so check for an explicitly-set value via
        # the presence of a concrete `.content` attribute on it.
        rv = create_mock.return_value
        if rv is not None and hasattr(rv, "content"):
            return _FakeStreamContext(rv)
        raise AssertionError(
            "mock_client.messages.create not configured — set .return_value "
            "or .side_effect before calling generate_variant_api."
        )

    mock_client.messages.stream = MagicMock(side_effect=stream_factory)


def _max_attempts_from_retry_policy(policy: Any) -> int:
    stop = getattr(policy, "stop", None)
    return int(getattr(stop, "max_attempt_number", 1) or 1)


def _tool_input_from_response(response: Any, tool_name: str) -> dict[str, Any]:
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == tool_name:
            return dict(getattr(block, "input", None) or {})
    raise ValueError(f"Required tool call {tool_name!r} not found in response")


class _FakeInstructorMessages:
    def __init__(self, mock_client: MagicMock) -> None:
        self._mock_client = mock_client

    async def create_with_completion(self, **kwargs: Any) -> tuple[Any, Any]:
        # Instructor is only needed when a test exercises the fake structured
        # completion path. Keep its provider graph out of pytest collection so
        # tests that do not use this fixture can start without importing the
        # heavyweight optional client dependencies.
        from instructor.core.exceptions import FailedAttempt, InstructorRetryException

        response_model = kwargs.pop("response_model")
        context = kwargs.pop("context", None)
        retry_policy = kwargs.pop("max_retries", None)
        hooks = kwargs.pop("hooks", None)
        max_attempts = _max_attempts_from_retry_policy(retry_policy)
        provider_kwargs = dict(kwargs)
        provider_kwargs.setdefault("tools", [{"name": response_model.__name__}])
        provider_kwargs.setdefault(
            "tool_choice",
            {"type": "tool", "name": response_model.__name__},
        )
        failed_attempts: list[FailedAttempt] = []
        last_response: Any = None

        for attempt_number in range(1, max_attempts + 1):
            if hooks is not None:
                hooks.emit_completion_arguments(**provider_kwargs)
            try:
                response = await self._mock_client.messages.create(**provider_kwargs)
            except Exception as exc:
                if hooks is not None:
                    hooks.emit_completion_error(exc)
                raise
            last_response = response
            if hooks is not None:
                hooks.emit_completion_response(response)
            try:
                payload = _tool_input_from_response(response, response_model.__name__)
                parsed = response_model.model_validate(payload, context=context)
                return parsed, response
            except Exception as exc:
                failed_attempts.append(
                    FailedAttempt(
                        attempt_number=attempt_number,
                        exception=exc,
                        completion=response,
                    )
                )
                if hooks is not None:
                    hooks.emit_parse_error(exc)
                if attempt_number >= max_attempts:
                    if hooks is not None:
                        hooks.emit_completion_last_attempt(exc)
                    raise InstructorRetryException(
                        str(exc),
                        last_completion=last_response,
                        messages=provider_kwargs.get("messages"),
                        n_attempts=attempt_number,
                        total_usage=0,
                        create_kwargs=provider_kwargs,
                        failed_attempts=failed_attempts,
                    ) from exc
                provider_kwargs["messages"] = [
                    *provider_kwargs.get("messages", []),
                    {
                        "role": "user",
                        "content": (
                            f"Validation Error found:\n{exc}\n"
                            "Recall the function correctly, fix the errors"
                        ),
                    },
                ]

        raise AssertionError("unreachable instructor fake retry exit")


class _FakeInstructorClient:
    def __init__(self, mock_client: MagicMock) -> None:
        self.messages = _FakeInstructorMessages(mock_client)


@pytest.fixture
def patched_anthropic_client(monkeypatch):
    """Patch `get_client` across every phase_4 module that imports it.

    Returns the MagicMock. Tests set `.messages.create.return_value` or
    `.side_effect` — the fixture also wires `.messages.stream(...)` to a
    context manager driven by the same configuration, so tests written
    against the older `create()` call shape continue to work after the
    variant_api streaming refactor.

    Enumerates `warp_taskgen/phase_4/*.py` dynamically: any module that binds
    `get_client` (typically via `from warp_taskgen.phase_4.anthropic_client
    import get_client`) gets patched. A new module that adds the import
    without being listed here will still be caught automatically — no
    silent live-API calls in tests.
    """
    import importlib
    import pkgutil

    from warp_taskgen import phase_4 as phase_4_pkg
    from warp_taskgen.phase_4 import anthropic_client as ac_module

    mock_client = MagicMock()
    mock_client.messages = MagicMock()
    mock_client.messages.create = AsyncMock()
    _install_stream_mock(mock_client)
    mock_client._worldsim_instructor_client = _FakeInstructorClient(mock_client)

    def _get_client_stub() -> Any:
        return mock_client

    global _PHASE_4_MODULES_WITH_GET_CLIENT
    if _PHASE_4_MODULES_WITH_GET_CLIENT is None:
        modules = []
        for mod_info in pkgutil.iter_modules(phase_4_pkg.__path__):
            mod = importlib.import_module(f"warp_taskgen.phase_4.{mod_info.name}")
            if hasattr(mod, "get_client"):
                modules.append(mod)
        _PHASE_4_MODULES_WITH_GET_CLIENT = modules

    for mod in _PHASE_4_MODULES_WITH_GET_CLIENT:
        monkeypatch.setattr(mod, "get_client", _get_client_stub)

    assert _PHASE_4_MODULES_WITH_GET_CLIENT, (
        "patched_anthropic_client fixture found no phase_4 module binding "
        "get_client; the fixture is stale — did the module layout change?"
    )
    ac_module.reset_client_for_tests()
    # The module-level `_semaphore` in warp_taskgen.phase_4.concurrency binds to
    # whichever event loop first calls `get_api_semaphore()`. Reset it between
    # tests so a semaphore bound to a prior loop can't leak into a new
    # `asyncio.run()` and raise `RuntimeError: ... bound to a different loop`.
    from warp_taskgen.phase_4.concurrency import reset_api_semaphore_for_tests

    reset_api_semaphore_for_tests()
    return mock_client
