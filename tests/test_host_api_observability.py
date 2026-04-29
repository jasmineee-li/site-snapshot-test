from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from tenacity import RetryError

from worldsim.host_api_observability import (
    InstructorCallTrace,
    build_instructor_hooks,
    instructor_semantic_retrying,
    summarize_provider_kwargs,
    summarize_provider_response,
    synthesize_cost_summary,
)


def test_provider_kwargs_summary_hashes_prompt_content():
    summary = summarize_provider_kwargs(
        {
            "model": "claude-sonnet-4-6",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": "secret prompt body"}],
            "tools": [{"name": "emit", "input_schema": {"type": "object"}}],
        }
    )

    assert summary["model"] == "claude-sonnet-4-6"
    assert summary["messages_chars"] > 0
    assert summary["messages_sha256"]
    assert "secret prompt body" not in json.dumps(summary)
    assert summary["tools"][0]["name"] == "emit"
    assert summary["tools_sha256"]


def test_response_summary_captures_stop_reason_tool_blocks_and_usage():
    response = SimpleNamespace(
        id="msg_1",
        model="claude-sonnet-4-6",
        stop_reason="tool_use",
        content=[
            SimpleNamespace(type="tool_use", name="TextPayloadResponse", input={"ok": True})
        ],
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=1,
            cache_read_input_tokens=2,
        ),
    )

    summary = summarize_provider_response(response)

    assert summary == {
        "id": "msg_1",
        "model": "claude-sonnet-4-6",
        "stop_reason": "tool_use",
        "content_blocks": [
            {
                "type": "tool_use",
                "name": "TextPayloadResponse",
                "text_chars": None,
                "text_sha256": None,
                "input_sha256": summary["content_blocks"][0]["input_sha256"],
            }
        ],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_creation_input_tokens": 1,
            "cache_read_input_tokens": 2,
        },
    }


def test_instructor_hooks_record_compact_trace():
    trace = InstructorCallTrace(
        phase="phase_2b",
        label="test",
        task_id="adv_1",
        site="gitlab",
        response_model_name="TextPayloadResponse",
    )
    hooks = build_instructor_hooks(trace)
    response = SimpleNamespace(id="msg_1", model="model", stop_reason="tool_use", content=[])

    hooks.emit_completion_arguments(
        model="model",
        max_tokens=2048,
        messages=[{"role": "user", "content": "prompt"}],
    )
    hooks.emit_completion_response(response)
    hooks.emit_parse_error(ValueError("bad payload"))

    diagnostics = trace.to_diagnostics()
    assert diagnostics["attempts"] == 1
    assert diagnostics["completion_responses"][0]["id"] == "msg_1"
    assert diagnostics["parse_errors"][0]["type"] == "ValueError"


@pytest.mark.asyncio
async def test_instructor_semantic_retrying_does_not_retry_transport_errors():
    attempts = 0

    with pytest.raises(RuntimeError):
        async for attempt in instructor_semantic_retrying(3):
            with attempt:
                attempts += 1
                raise RuntimeError("transport")

    assert attempts == 1


@pytest.mark.asyncio
async def test_instructor_semantic_retrying_retries_parse_errors():
    attempts = 0

    with pytest.raises(RetryError):
        async for attempt in instructor_semantic_retrying(3):
            with attempt:
                attempts += 1
                raise json.JSONDecodeError("bad", "{}", 0)

    assert attempts == 3


def test_synthesize_cost_summary_matches_host_api_shape():
    response = SimpleNamespace(
        id="msg_1",
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=10,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )

    summary = json.loads(
        synthesize_cost_summary(response, model="claude-sonnet-4-6", elapsed_s=1.2)
    )

    assert summary["num_turns"] == 1
    assert summary["duration_ms"] == 1200
    assert summary["session_id"] == "msg_1"
    assert summary["model_usage"]["claude-sonnet-4-6"]["input_tokens"] == 100
