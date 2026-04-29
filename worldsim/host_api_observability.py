"""Observability helpers for structured host-side model API calls.

Instructor is useful for schema validation and validation-aware reasks, but
WorldSim still owns transport retry policy, cost accounting, and run artifacts.
This module keeps those concerns explicit so Instructor-backed calls have the
same diagnostic shape as the hand-written Phase 4 Messages API paths.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from json import JSONDecodeError
from typing import Any

from instructor.core.exceptions import (
    AsyncValidationError,
    IncompleteOutputException,
    InstructorRetryException,
)
from instructor.core.exceptions import (
    ValidationError as InstructorValidationError,
)
from instructor.core.hooks import HookName, Hooks
from pydantic import ValidationError as PydanticValidationError
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt


def instructor_semantic_retrying(max_attempts: int) -> AsyncRetrying:
    """Return Instructor retry policy for semantic/parse failures only.

    WorldSim's ``call_with_retry`` remains responsible for API transport
    failures. Passing a plain integer to Instructor would let Tenacity retry
    broad exceptions, including auth/quota/provider failures, which hides the
    operational buckets the pipeline relies on.
    """

    return AsyncRetrying(
        stop=stop_after_attempt(max(1, max_attempts)),
        retry=retry_if_exception_type(
            (
                PydanticValidationError,
                JSONDecodeError,
                AsyncValidationError,
                InstructorValidationError,
            )
        ),
    )


def build_instructor_hooks(trace: InstructorCallTrace) -> Hooks:
    hooks = Hooks()
    hooks.on(HookName.COMPLETION_KWARGS, trace.record_completion_kwargs)
    hooks.on(HookName.COMPLETION_RESPONSE, trace.record_completion_response)
    hooks.on(HookName.PARSE_ERROR, trace.record_parse_error)
    hooks.on(HookName.COMPLETION_ERROR, trace.record_completion_error)
    hooks.on(HookName.COMPLETION_LAST_ATTEMPT, trace.record_completion_last_attempt)
    return hooks


@dataclass
class InstructorCallTrace:
    """Compact structured trace for one Instructor-backed host API call."""

    phase: str
    label: str
    task_id: str | None = None
    site: str | None = None
    provider: str = "anthropic"
    mode: str = "anthropic_tools"
    response_model_name: str | None = None
    started_at: float = field(default_factory=time.monotonic)
    completion_kwargs: list[dict[str, Any]] = field(default_factory=list)
    completion_responses: list[dict[str, Any]] = field(default_factory=list)
    parse_errors: list[dict[str, Any]] = field(default_factory=list)
    completion_errors: list[dict[str, Any]] = field(default_factory=list)
    last_attempt_errors: list[dict[str, Any]] = field(default_factory=list)

    def record_completion_kwargs(self, *_args: Any, **kwargs: Any) -> None:
        self.completion_kwargs.append(summarize_provider_kwargs(kwargs))

    def record_completion_response(self, response: Any) -> None:
        self.completion_responses.append(summarize_provider_response(response))

    def record_parse_error(self, error: Exception) -> None:
        self.parse_errors.append(summarize_exception(error))

    def record_completion_error(self, error: Exception, **metadata: Any) -> None:
        entry = summarize_exception(error)
        entry["metadata"] = _jsonable(metadata)
        self.completion_errors.append(entry)

    def record_completion_last_attempt(self, error: Exception, **metadata: Any) -> None:
        entry = summarize_exception(error)
        entry["metadata"] = _jsonable(metadata)
        self.last_attempt_errors.append(entry)

    def to_diagnostics(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "label": self.label,
            "task_id": self.task_id,
            "site": self.site,
            "provider": self.provider,
            "mode": self.mode,
            "response_model": self.response_model_name,
            "elapsed_s": round(max(0.0, time.monotonic() - self.started_at), 3),
            "attempts": len(self.completion_kwargs),
            "completion_kwargs": self.completion_kwargs,
            "completion_responses": self.completion_responses,
            "parse_errors": self.parse_errors,
            "completion_errors": self.completion_errors,
            "last_attempt_errors": self.last_attempt_errors,
        }


def instructor_retry_exception_diagnostics(exc: InstructorRetryException) -> dict[str, Any]:
    attempts = []
    for attempt in getattr(exc, "failed_attempts", None) or []:
        attempts.append(
            {
                "attempt_number": getattr(attempt, "attempt_number", None),
                "exception": summarize_exception(getattr(attempt, "exception", None)),
                "completion": summarize_provider_response(getattr(attempt, "completion", None)),
            }
        )
    return {
        "exception": summarize_exception(exc),
        "n_attempts": getattr(exc, "n_attempts", None),
        "total_usage": _jsonable(getattr(exc, "total_usage", None)),
        "last_completion": summarize_provider_response(getattr(exc, "last_completion", None)),
        "create_kwargs": summarize_provider_kwargs(getattr(exc, "create_kwargs", None) or {}),
        "failed_attempts": attempts,
    }


def summarize_provider_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    messages = kwargs.get("messages")
    tools = kwargs.get("tools")
    system = kwargs.get("system")
    return {
        "model": kwargs.get("model"),
        "max_tokens": kwargs.get("max_tokens"),
        "temperature": kwargs.get("temperature"),
        "tool_choice": _jsonable(kwargs.get("tool_choice")),
        "tools": _summarize_tools(tools),
        "tools_sha256": _stable_sha256(tools),
        "messages_count": len(messages) if isinstance(messages, list) else None,
        "messages_chars": _char_count(messages),
        "messages_sha256": _stable_sha256(messages),
        "system_chars": _char_count(system),
        "system_sha256": _stable_sha256(system),
        "metadata": _jsonable(kwargs.get("metadata")),
    }


def summarize_provider_response(response: Any) -> dict[str, Any] | None:
    if response is None:
        return None
    content = getattr(response, "content", None)
    return {
        "id": getattr(response, "id", None),
        "model": getattr(response, "model", None),
        "stop_reason": getattr(response, "stop_reason", None),
        "content_blocks": _summarize_content_blocks(content),
        "usage": usage_dict(response),
    }


def summarize_exception(error: Any) -> dict[str, Any]:
    if error is None:
        return {"type": None, "message": None}
    out = {
        "type": type(error).__name__,
        "message": str(error),
    }
    if isinstance(error, IncompleteOutputException):
        out["last_completion"] = summarize_provider_response(error.last_completion)
    raw_response = getattr(error, "raw_response", None)
    if raw_response is not None:
        out["raw_response"] = summarize_provider_response(raw_response)
    return out


def usage_dict(response: Any) -> dict[str, int] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return {
        "input_tokens": getattr(usage, "input_tokens", 0) or 0,
        "output_tokens": getattr(usage, "output_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
    }


def synthesize_cost_summary(response: Any, *, model: str, elapsed_s: float) -> str:
    usage = usage_dict(response) or {}
    in_tok = int(usage.get("input_tokens", 0) or 0)
    out_tok = int(usage.get("output_tokens", 0) or 0)
    # Sonnet-family indicative pricing used by existing Phase 4 host API paths.
    cost = (in_tok / 1_000_000) * 3.0 + (out_tok / 1_000_000) * 15.0
    return json.dumps(
        {
            "total_cost_usd": cost,
            "num_turns": 1,
            "duration_ms": int(elapsed_s * 1000),
            "session_id": getattr(response, "id", None),
            "model_usage": {
                model: {
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "cache_creation_input_tokens": int(
                        usage.get("cache_creation_input_tokens", 0) or 0
                    ),
                    "cache_read_input_tokens": int(usage.get("cache_read_input_tokens", 0) or 0),
                }
            },
        }
    )


def _summarize_tools(tools: Any) -> list[dict[str, Any]] | None:
    if not isinstance(tools, list):
        return None
    summarized: list[dict[str, Any]] = []
    for tool in tools:
        if isinstance(tool, dict):
            summarized.append(
                {
                    "name": tool.get("name"),
                    "description_chars": _char_count(tool.get("description")),
                    "input_schema_sha256": _stable_sha256(tool.get("input_schema")),
                }
            )
        else:
            summarized.append({"type": type(tool).__name__, "repr_sha256": _stable_sha256(repr(tool))})
    return summarized


def _summarize_content_blocks(content: Any) -> list[dict[str, Any]] | None:
    if not isinstance(content, list):
        return None
    blocks: list[dict[str, Any]] = []
    for block in content:
        block_type = getattr(block, "type", None)
        name = getattr(block, "name", None)
        text = getattr(block, "text", None)
        input_value = getattr(block, "input", None)
        if isinstance(block, dict):
            block_type = block.get("type")
            name = block.get("name")
            text = block.get("text")
            input_value = block.get("input")
        blocks.append(
            {
                "type": block_type,
                "name": name,
                "text_chars": _char_count(text),
                "text_sha256": _stable_sha256(text) if isinstance(text, str) else None,
                "input_sha256": _stable_sha256(input_value) if input_value is not None else None,
            }
        )
    return blocks


def _char_count(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        return len(value)
    try:
        return len(json.dumps(_jsonable(value), sort_keys=True, default=str))
    except TypeError:
        return len(str(value))


def _stable_sha256(value: Any) -> str | None:
    if value is None:
        return None
    try:
        blob = json.dumps(_jsonable(value), sort_keys=True, default=str)
    except TypeError:
        blob = str(value)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _jsonable(model_dump())
        except Exception:
            pass
    return repr(value)
