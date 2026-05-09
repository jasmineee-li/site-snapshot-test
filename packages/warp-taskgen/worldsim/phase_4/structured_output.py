"""Native Anthropic structured-output helpers for Phase 4 host judges."""

from __future__ import annotations

import json
from typing import Any

from anthropic import AsyncAnthropic

from worldsim.phase_4.anthropic_client import call_with_retry, normalize_model_for_auth
from worldsim.phase_4.concurrency import get_api_semaphore

STRUCTURED_OUTPUT_EXTRA_BODY_KEY = "output_config"
STRUCTURED_OUTPUT_MODE = "anthropic_output_config_json_schema_v1"


def structured_output_extra_body(schema: dict[str, Any]) -> dict[str, Any]:
    """Return the Messages API extra body for native JSON-schema output.

    The installed Anthropic SDK version exposes new/experimental request fields
    through ``extra_body``. Keeping this construction centralized lets direct
    Phase 4 judges move away from forced tool-use without duplicating provider
    request shape at each call site.
    """

    return {
        STRUCTURED_OUTPUT_EXTRA_BODY_KEY: {
            "format": {
                "type": "json_schema",
                "schema": schema,
            }
        }
    }


def extract_structured_json_text(response: Any) -> str | None:
    """Extract the first text block from a structured-output response."""

    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "text":
            text = getattr(block, "text", None)
            if isinstance(text, str) and text.strip():
                return text
    return None


def parse_structured_json_payload(response: Any) -> tuple[dict[str, Any] | None, str | None, str]:
    """Return ``(payload, failure_class, raw_text)`` from a Messages response."""

    raw_text = extract_structured_json_text(response) or ""
    if not raw_text:
        return None, "no_structured_output", raw_text
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return None, "json_parse_error", raw_text
    if not isinstance(parsed, dict):
        return None, "schema_violation", raw_text
    return parsed, None, raw_text


async def create_structured_message(
    *,
    client: AsyncAnthropic,
    model: str,
    max_tokens: int,
    messages: list[dict[str, Any]],
    schema: dict[str, Any],
    metadata: dict[str, str] | None,
    retries: int,
    label: str,
) -> Any:
    """Call Anthropic Messages with a JSON-schema structured-output contract."""

    async def _call() -> Any:
        async with get_api_semaphore():
            return await client.messages.create(
                model=normalize_model_for_auth(model),
                max_tokens=max_tokens,
                messages=messages,
                metadata=metadata,
                extra_body=structured_output_extra_body(schema),
            )

    return await call_with_retry(_call, retries=retries, label=label)

