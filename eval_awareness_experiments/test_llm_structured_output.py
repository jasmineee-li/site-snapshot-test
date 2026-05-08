from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from eval_awareness_experiments.llm import LLM


@pytest.mark.asyncio
async def test_generate_json_schema_uses_openai_compatible_response_format(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    llm = LLM("anthropic/claude-sonnet-4.6", retries=1)
    llm._client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))]
        )
    )
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }

    result = await llm.generate_json_schema(
        "classify this",
        schema=schema,
        name="test_schema",
    )

    assert result.message.text == '{"ok": true}'
    kwargs = llm._client.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "test_schema",
            "strict": True,
            "schema": schema,
        },
    }
    assert kwargs["extra_body"]["provider"]["require_parameters"] is True


@pytest.mark.asyncio
async def test_generate_json_schema_preserves_thinking_extra_body(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    llm = LLM("anthropic/claude-opus-4.7:thinking", retries=1)
    llm._client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))]
        )
    )

    await llm.generate_json_schema(
        "classify this",
        schema={
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        },
        name="test_schema",
    )

    kwargs = llm._client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "anthropic/claude-opus-4.7"
    assert kwargs["extra_body"] == {
        "reasoning": {"effort": "high"},
        "provider": {"require_parameters": True},
    }
