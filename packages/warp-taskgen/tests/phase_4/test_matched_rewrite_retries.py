"""Retry accounting regression for the matched rewrite stream transport."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from warp_taskgen.phase_4 import eval_awareness_rewrite_api as rewrite_api
from warp_taskgen.phase_4.eval_awareness_streaming_tool import (
    StreamingToolResult,
    StreamingToolTruncatedError,
)
from warp_taskgen.phase_4.matched_rewrite_ordinary_api import usage_from_diagnostics


def _diagnostics(response_id: str, *, attempts: int, transport_attempts: int) -> dict[str, object]:
    return {
        "phase": "phase_4",
        "label": "matched-rewrite",
        "attempts": attempts,
        "transport_attempts": transport_attempts,
        "elapsed_s": 0.1,
        "completion_kwargs": [{"max_tokens": 1024}],
        "completion_responses": [
            {
                "id": response_id,
                "stop_reason": "max_tokens" if response_id == "truncated" else "tool_use",
                "usage": {
                    "input_tokens": 10 if response_id == "truncated" else 12,
                    "output_tokens": 7 if response_id == "truncated" else 9,
                },
            }
        ],
        "parse_errors": [],
        "completion_errors": [],
        "retry_feedback": [],
        "retry_fallbacks": [],
    }


def test_truncated_stream_is_merged_into_successful_retry(monkeypatch):
    calls: list[int] = []
    recorded: list[tuple[str, str | None]] = []

    class Parsed:
        def model_dump(self, *, exclude_none: bool) -> dict[str, object]:
            return {"status": "ok", "payload_text": {"rendered_payload": "rewritten"}}

    async def stream(**kwargs: object) -> StreamingToolResult:
        calls.append(int(kwargs["max_tokens"]))
        if len(calls) == 1:
            diagnostics = _diagnostics("truncated", attempts=1, transport_attempts=2)
            raise StreamingToolTruncatedError(
                "truncated",
                diagnostics=diagnostics,
                completion=SimpleNamespace(id="truncated"),
            )
        return StreamingToolResult(
            parsed=Parsed(),
            completion=SimpleNamespace(id="success"),
            diagnostics=_diagnostics("success", attempts=2, transport_attempts=1),
        )

    monkeypatch.setattr(rewrite_api, "stream_pydantic_tool_call", stream)
    monkeypatch.setattr(rewrite_api, "_eval_awareness_rewrite_thinking_kwargs", lambda: {})
    monkeypatch.setattr(rewrite_api, "_initial_max_tokens_for_rewrite", lambda _: 1024)
    monkeypatch.setattr(rewrite_api, "_materialize_rewrite_seed", lambda *_args: {})
    monkeypatch.setattr(rewrite_api, "_merge_rewrite", lambda *_args: {"status": "ok"})
    monkeypatch.setattr(
        rewrite_api.cost_tracker,
        "record",
        lambda phase, summary, **_kwargs: recorded.append((phase, summary)),
    )

    result = asyncio.run(
        rewrite_api.generate_eval_awareness_rewrite_api(
            {"id": "task-1", "site": "gitlab"},
            {},
            iteration=1,
            client=object(),
            sandbox_model="sandbox-model",
            include_api_diagnostics=True,
            cost_phase="phase_4:matched_rewrite_study:proposal",
        )
    )
    diagnostics = result["matched_rewrite_api_diagnostics"]
    assert calls == [1024, 256_000]
    assert diagnostics["attempts"] == 3
    assert diagnostics["transport_attempts"] == 3
    assert [item["id"] for item in diagnostics["completion_responses"]] == [
        "truncated",
        "success",
    ]
    usage = usage_from_diagnostics(
        diagnostics,
        model="sandbox-model",
        fallback_reason="missing",
    )
    assert usage.available
    assert usage.attempts == 3
    assert (usage.input_tokens, usage.output_tokens) == (22, 16)
    assert len(recorded) == 2
    assert all(phase == "phase_4:matched_rewrite_study:proposal" for phase, _ in recorded)
    costs = [json.loads(summary or "{}")["model_usage"] for _, summary in recorded]
    assert sum(item["sandbox-model"]["input_tokens"] for item in costs) == 22
    assert sum(item["sandbox-model"]["output_tokens"] for item in costs) == 16
