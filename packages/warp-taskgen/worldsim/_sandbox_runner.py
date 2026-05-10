"""Sandbox runner -- invoked inside Modal sandbox to run Claude Code via the Agent SDK.

This script is staged into the sandbox image at /workspace/_sdk_runner.py and
executed by :func:`worldsim.modal_sandbox.run_claude_in_sandbox`.  It reads a
prompt from /workspace/_prompt.txt, calls the Claude Agent SDK, streams NDJSON
event lines to stdout, and prints a final summary with cost/token/session data.

The orchestrator parses these lines to provide live tool-call logging and to
capture the summary metadata without changing the file-based output contract.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any


def _jsonable(obj: Any) -> Any:
    """Convert SDK objects to plain JSON-serializable data."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(key): _jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(item) for item in obj]
    if hasattr(obj, "model_dump"):
        return _jsonable(obj.model_dump())
    if hasattr(obj, "__dict__"):
        return _jsonable(vars(obj))
    return str(obj)


def _coerce_unix_timestamp(value: Any) -> float | None:
    """Return a numeric Unix timestamp when one is present."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _rate_limit_event_payload(message: Any, *, now_ts: float | None = None) -> dict[str, Any]:
    """Serialize a Claude Agent SDK rate-limit event for NDJSON logging."""
    now_ts = time.time() if now_ts is None else now_ts
    info = getattr(message, "rate_limit_info", None)
    resets_at = _coerce_unix_timestamp(getattr(info, "resets_at", None))
    retry_after_seconds = None
    if resets_at is not None:
        retry_after_seconds = max(0.0, resets_at - now_ts)
    return {
        "type": "rate_limit",
        "status": getattr(info, "status", None),
        "rate_limit_type": getattr(info, "rate_limit_type", getattr(info, "type", None)),
        "resets_at": resets_at,
        "retry_after_seconds": retry_after_seconds,
        "utilization": _jsonable(getattr(info, "utilization", None)),
    }


async def main() -> None:
    # Late imports -- these packages only exist inside the sandbox image.
    from claude_agent_sdk import ClaudeAgentOptions, query  # type: ignore[import-untyped]
    from claude_agent_sdk.types import (  # type: ignore[import-untyped]
        AssistantMessage,
        RateLimitEvent,
        ResultMessage,
        TextBlock,
        ThinkingBlock,
        ToolUseBlock,
    )

    prompt_path = Path("/workspace/_prompt.txt")
    if not prompt_path.exists():
        print(
            json.dumps({"type": "error", "message": "_prompt.txt not found in /workspace"}),
            flush=True,
        )
        sys.exit(1)

    prompt = prompt_path.read_text()
    model = sys.argv[1] if len(sys.argv) > 1 else "claude-opus-4-6"

    options = ClaudeAgentOptions(
        permission_mode="bypassPermissions",
        system_prompt={
            "type": "preset",
            "preset": "claude_code",
        },
        effort="high",
        cwd="/workspace",
        model=model,
        max_turns=300,
        max_budget_usd=250.0,
    )

    start = time.monotonic()
    tool_calls: list[str] = []
    turn_count = 0
    result_msg = None

    try:
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                turn_count += 1
                for block in message.content:
                    if isinstance(block, ToolUseBlock):
                        event = {
                            "type": "tool_call",
                            "tool": block.name,
                            "id": block.id,
                            "turn": turn_count,
                        }
                        print(json.dumps(event), flush=True)
                        tool_calls.append(block.name)
                    elif isinstance(block, TextBlock):
                        # Log first 200 chars of text blocks for observability
                        event = {"type": "text", "preview": block.text[:200], "turn": turn_count}
                        print(json.dumps(event), flush=True)
                    elif isinstance(block, ThinkingBlock):
                        # Extended thinking is emitted when effort="high"; capture
                        # a preview for observability without dumping the full chain.
                        thinking_text = getattr(block, "thinking", "") or ""
                        event = {
                            "type": "thinking",
                            "preview": thinking_text[:200],
                            "turn": turn_count,
                        }
                        print(json.dumps(event), flush=True)
            elif isinstance(message, RateLimitEvent):
                event = _rate_limit_event_payload(message)
                print(json.dumps(event), flush=True)
            elif isinstance(message, ResultMessage):
                result_msg = message
    except Exception as exc:
        print(
            json.dumps({"type": "error", "message": f"SDK query failed: {exc!r}"}),
            flush=True,
        )
        # Don't sys.exit -- let the summary still be emitted so the
        # orchestrator can see partial data.

    elapsed = time.monotonic() - start

    summary: dict = {
        "type": "summary",
        "elapsed_s": round(elapsed, 1),
        "tool_calls": tool_calls,
        "num_tool_calls": len(tool_calls),
    }
    if result_msg is not None:
        # Convert SDK objects to plain dicts for JSON serialization.
        def _to_dict(obj):
            if obj is None:
                return None
            if isinstance(obj, dict):
                return obj
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            if hasattr(obj, "__dict__"):
                return obj.__dict__
            return obj

        summary.update(
            {
                "session_id": result_msg.session_id,
                "num_turns": result_msg.num_turns,
                "total_cost_usd": result_msg.total_cost_usd,
                "duration_ms": result_msg.duration_ms,
                "duration_api_ms": result_msg.duration_api_ms,
                "is_error": result_msg.is_error,
                "subtype": result_msg.subtype,
                "stop_reason": result_msg.stop_reason,
                "usage": _to_dict(result_msg.usage),
                "model_usage": _to_dict(result_msg.model_usage),
            }
        )
    print(json.dumps(summary, default=str), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
    # Explicit flush so the final NDJSON summary and EOF are delivered to the
    # Modal worker promptly before process teardown. Without this, async
    # subprocess cleanup during asyncio.run() shutdown can delay fd close and
    # leave Modal's stream readers briefly waiting for EOF frames.
    sys.stdout.flush()
    sys.stderr.flush()
