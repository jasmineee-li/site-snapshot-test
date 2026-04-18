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


@pytest.fixture
def patched_anthropic_client(monkeypatch):
    """Patch `get_client` across every phase_4 module that imports it.

    Returns the MagicMock. Tests set `.messages.create.return_value` or
    `.side_effect` — the fixture also wires `.messages.stream(...)` to a
    context manager driven by the same configuration, so tests written
    against the older `create()` call shape continue to work after the
    variant_api streaming refactor.

    Enumerates `worldsim/phase_4/*.py` dynamically: any module that binds
    `get_client` (typically via `from worldsim.phase_4.anthropic_client
    import get_client`) gets patched. A new module that adds the import
    without being listed here will still be caught automatically — no
    silent live-API calls in tests.
    """
    import importlib
    import pkgutil

    from worldsim import phase_4 as phase_4_pkg
    from worldsim.phase_4 import anthropic_client as ac_module

    mock_client = MagicMock()
    mock_client.messages = MagicMock()
    mock_client.messages.create = AsyncMock()
    _install_stream_mock(mock_client)

    def _get_client_stub() -> Any:
        return mock_client

    patched_any = False
    for mod_info in pkgutil.iter_modules(phase_4_pkg.__path__):
        mod = importlib.import_module(f"worldsim.phase_4.{mod_info.name}")
        if hasattr(mod, "get_client"):
            monkeypatch.setattr(mod, "get_client", _get_client_stub)
            patched_any = True
    assert patched_any, (
        "patched_anthropic_client fixture found no phase_4 module binding "
        "get_client; the fixture is stale — did the module layout change?"
    )
    ac_module.reset_client_for_tests()
    return mock_client
