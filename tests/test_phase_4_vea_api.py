"""P0/P1 tests for the VEA judge."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from worldsim.phase_4.vea_api import run_vea_api


def _tool_use_response(
    tool_input: dict, *, in_tok: int = 100, out_tok: int = 20
) -> SimpleNamespace:
    return SimpleNamespace(
        id="msg_vea",
        content=[
            SimpleNamespace(type="tool_use", name="classify_vea", input=tool_input),
        ],
        usage=SimpleNamespace(
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )


def _text_only_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        id="msg_vea",
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=5,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )


@pytest.mark.asyncio
async def test_happy_path_schema_roundtrip(patched_anthropic_client, tmp_path):
    patched_anthropic_client.messages.create = AsyncMock(
        return_value=_tool_use_response(
            {
                "verbalized_eval_awareness": True,
                "evidence": "This looks like a test of my instruction-following.",
                "confidence": "high",
            }
        )
    )
    result = await run_vea_api({"id": "t"}, tmp_path, "some aer transcript")
    assert result["vea_verbalized"] is True
    assert result["vea_confidence"] == "high"
    assert len(result["vea_evidence"]) <= 200
    assert result["vea_failure"] is None


@pytest.mark.asyncio
async def test_verbalized_false_valid(patched_anthropic_client, tmp_path):
    patched_anthropic_client.messages.create = AsyncMock(
        return_value=_tool_use_response(
            {
                "verbalized_eval_awareness": False,
                "evidence": "",
                "confidence": "medium",
            }
        )
    )
    result = await run_vea_api({"id": "t"}, tmp_path, "aer")
    assert result["vea_verbalized"] is False
    assert result["vea_failure"] is None


@pytest.mark.asyncio
async def test_no_tool_use_block_returns_failure(patched_anthropic_client, tmp_path):
    """Model refused and emitted text only — classify as no_tool_use."""
    patched_anthropic_client.messages.create = AsyncMock(
        return_value=_text_only_response("I cannot help with that.")
    )
    result = await run_vea_api({"id": "t"}, tmp_path, "aer")
    assert result["vea_verbalized"] is None
    assert result["vea_failure"] == "no_tool_use"


@pytest.mark.asyncio
async def test_schema_violation_missing_verbalized_field(patched_anthropic_client, tmp_path):
    patched_anthropic_client.messages.create = AsyncMock(
        return_value=_tool_use_response(
            {"evidence": "x", "confidence": "high"},
        )
    )
    result = await run_vea_api({"id": "t"}, tmp_path, "aer")
    assert result["vea_failure"] == "schema_violation"


@pytest.mark.asyncio
async def test_schema_violation_invalid_confidence_enum(patched_anthropic_client, tmp_path):
    patched_anthropic_client.messages.create = AsyncMock(
        return_value=_tool_use_response(
            {
                "verbalized_eval_awareness": True,
                "evidence": "ok",
                "confidence": "very_high",  # not in enum
            }
        )
    )
    result = await run_vea_api({"id": "t"}, tmp_path, "aer")
    assert result["vea_failure"] == "schema_violation"


@pytest.mark.asyncio
async def test_schema_violation_verbalized_bool_coerced_from_string(
    patched_anthropic_client, tmp_path
):
    """String 'true' for a bool field is not tolerated."""
    patched_anthropic_client.messages.create = AsyncMock(
        return_value=_tool_use_response(
            {
                "verbalized_eval_awareness": "true",
                "evidence": "ok",
                "confidence": "high",
            }
        )
    )
    result = await run_vea_api({"id": "t"}, tmp_path, "aer")
    assert result["vea_failure"] == "schema_violation"


@pytest.mark.asyncio
async def test_evidence_over_200_chars_rejected(patched_anthropic_client, tmp_path):
    long_evidence = "x" * 250
    patched_anthropic_client.messages.create = AsyncMock(
        return_value=_tool_use_response(
            {
                "verbalized_eval_awareness": True,
                "evidence": long_evidence,
                "confidence": "high",
            }
        )
    )
    result = await run_vea_api({"id": "t"}, tmp_path, "aer")
    assert result["vea_failure"] == "schema_violation"


@pytest.mark.asyncio
async def test_forced_tool_use_tool_choice(patched_anthropic_client, tmp_path):
    """tool_choice must force classify_vea selection."""
    patched_anthropic_client.messages.create = AsyncMock(
        return_value=_tool_use_response(
            {
                "verbalized_eval_awareness": True,
                "evidence": "x",
                "confidence": "low",
            }
        )
    )
    await run_vea_api({"id": "t"}, tmp_path, "aer")
    kwargs = patched_anthropic_client.messages.create.call_args.kwargs
    assert kwargs["tool_choice"] == {"type": "tool", "name": "classify_vea"}
    assert len(kwargs["tools"]) == 1
    assert kwargs["tools"][0]["name"] == "classify_vea"


@pytest.mark.asyncio
async def test_api_exception_bucketed(patched_anthropic_client, tmp_path, monkeypatch):
    import asyncio as _asyncio

    monkeypatch.setattr(_asyncio, "sleep", AsyncMock())
    patched_anthropic_client.messages.create = AsyncMock(
        side_effect=RuntimeError("transient api failure")
    )
    result = await run_vea_api({"id": "t"}, tmp_path, "aer")
    assert result["vea_failure"] == "api_error"


@pytest.mark.asyncio
async def test_raw_response_persisted(patched_anthropic_client, tmp_path):
    patched_anthropic_client.messages.create = AsyncMock(
        return_value=_tool_use_response(
            {
                "verbalized_eval_awareness": True,
                "evidence": "quote",
                "confidence": "high",
            }
        )
    )
    await run_vea_api({"id": "t"}, tmp_path, "aer")
    raw = tmp_path / "vea_raw_response.json"
    assert raw.exists()
    payload = json.loads(raw.read_text())
    assert payload["vea_verbalized"] is True
    assert payload["vea_evidence"] == "quote"


@pytest.mark.asyncio
async def test_cost_records_to_phase_4_vea(patched_anthropic_client, tmp_path, monkeypatch):
    from worldsim.cost_tracker import tracker as cost_tracker

    patched_anthropic_client.messages.create = AsyncMock(
        return_value=_tool_use_response(
            {
                "verbalized_eval_awareness": False,
                "evidence": "",
                "confidence": "low",
            }
        )
    )
    recorded = []
    monkeypatch.setattr(
        cost_tracker, "record", lambda phase, summary: recorded.append((phase, summary))
    )
    await run_vea_api({"id": "t"}, tmp_path, "aer")
    phases = [p for p, _ in recorded]
    assert "phase_4:vea" in phases
    assert "phase_4" not in phases


@pytest.mark.asyncio
async def test_cache_control_on_aer_transcript_block(patched_anthropic_client, tmp_path):
    patched_anthropic_client.messages.create = AsyncMock(
        return_value=_tool_use_response(
            {
                "verbalized_eval_awareness": False,
                "evidence": "",
                "confidence": "low",
            }
        )
    )
    await run_vea_api({"id": "t"}, tmp_path, "transcript body")
    kwargs = patched_anthropic_client.messages.create.call_args.kwargs
    content = kwargs["messages"][0]["content"]
    cached = [b for b in content if b.get("cache_control") == {"type": "ephemeral"}]
    assert len(cached) == 1
