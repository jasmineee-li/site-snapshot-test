from __future__ import annotations

import json
import os
import time
from typing import Any

import instructor
import requests
from instructor.core.exceptions import IncompleteOutputException, InstructorRetryException

from warp_taskgen.cost_tracker import tracker as cost_tracker
from warp_taskgen.host_api_observability import (
    InstructorCallTrace,
    build_instructor_hooks,
    instructor_retry_exception_diagnostics,
    instructor_semantic_retrying,
    synthesize_cost_summary,
)
from warp_taskgen.phase_2.text_fill.api_errors import TextFillAPIError
from warp_taskgen.phase_2.text_fill.constants import (
    _JSON_FENCE_RE,
    _TEXT_FILL_DEFAULT_MAX_TOKENS,
    _TEXT_FILL_EXPANDED_MAX_TOKENS,
    _TEXT_FILL_SHORT_TITLE_MAX_TOKENS,
    REFUSAL_SNIPPETS,
    TEXT_FILL_STRUCTURED_RETRIES,
)
from warp_taskgen.phase_2.text_fill.context import _content_capacity_for_surface
from warp_taskgen.phase_2.text_fill.request_archive import text_fill_request_recorder
from warp_taskgen.phase_2.text_fill.validation import _length_budget_bounds, validate_text_post_hoc
from warp_taskgen.phase_4.anthropic_client import (
    call_with_retry,
    get_client,
    normalize_model_for_auth,
    resolved_messages_provider,
    temperature_kwargs_for_model,
)
from warp_taskgen.phase_4.concurrency import get_api_semaphore
from warp_taskgen.text_payload_contract import TextPayloadResponse, text_payload_validation_context


def is_refusal(text: str) -> bool:
    normalized = text.lower()
    return any(snippet in normalized for snippet in REFUSAL_SNIPPETS)


def _parse_text_fill_response(raw_text: str) -> tuple[dict[str, Any], str | None]:
    candidate = _JSON_FENCE_RE.sub("", raw_text).strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return ({}, f"invalid JSON: {exc}")
    if not isinstance(parsed, dict):
        return ({}, "text fill response must be a JSON object")
    return (parsed, None)


def _text_fill_max_tokens(task: dict[str, Any]) -> int:
    """Choose an output budget from the target field contract.

    The payload schema includes duplicated text fields plus witness arrays, so
    the budget must be larger than the rendered-payload character budget. At
    the same time, using a huge universal cap would make malformed generations
    slower and more expensive without improving validity.
    """

    capacity = _content_capacity_for_surface(task)
    if capacity == "short_title":
        return _TEXT_FILL_SHORT_TITLE_MAX_TOKENS

    budget, _errors = _length_budget_bounds(task)
    max_chars = budget[1] if budget is not None else 1500
    concealment = str(task.get("concealment") or "plaintext")
    if concealment != "plaintext" or max_chars > 2000 or capacity == "code_content":
        return _TEXT_FILL_EXPANDED_MAX_TOKENS
    return _TEXT_FILL_DEFAULT_MAX_TOKENS


async def _call_text_fill_api(
    prompt: str,
    model: str,
    *,
    task: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | str, str] | tuple[dict[str, Any] | str, str, dict[str, Any]]:
    """Call the shared Anthropic-compatible client used by Phase 4 APIs.

    Fresh Phase 2 text fill uses Instructor with a Pydantic response model so
    provider output shape and WorldSim post-hoc constraints are part of the
    generation retry loop. The raw JSON fallback remains for old tests and
    diagnostic callers that intentionally pass no task context.
    """
    client = get_client()
    if task is not None:
        instructor_client = instructor.from_anthropic(client, mode=instructor.Mode.ANTHROPIC_TOOLS)
        normalized_model = normalize_model_for_auth(model)
        max_tokens = _text_fill_max_tokens(task)
        client_provider = resolved_messages_provider()
        trace = InstructorCallTrace(
            phase="phase_2b",
            label="phase2b-text-fill",
            task_id=str(task.get("id") or ""),
            site=str(task.get("site") or ""),
            response_model_name=TextPayloadResponse.__name__,
            resolved_client_provider=client_provider,
            request_recorder=text_fill_request_recorder(
                task_id=str(task.get("id") or ""),
                site=str(task.get("site") or ""),
                configured_model=model,
                client_provider=client_provider,
            ),
        )
        hooks = build_instructor_hooks(trace)

        def _validate_payload(payload: dict[str, Any]) -> list[str]:
            return validate_text_post_hoc(payload, task)

        t0 = time.monotonic()
        try:

            async def _call_structured() -> Any:
                async with get_api_semaphore():
                    return await instructor_client.messages.create_with_completion(
                        model=normalized_model,
                        max_tokens=max_tokens,
                        messages=[{"role": "user", "content": prompt}],
                        response_model=TextPayloadResponse,
                        context=text_payload_validation_context(_validate_payload),
                        max_retries=instructor_semantic_retrying(TEXT_FILL_STRUCTURED_RETRIES),
                        hooks=hooks,
                        **temperature_kwargs_for_model(normalized_model, 0.7),
                    )

            payload, raw_response = await call_with_retry(
                _call_structured,
                retries=3,
                label="phase2b-text-fill",
            )
        except InstructorRetryException as exc:
            diagnostics = trace.to_diagnostics()
            diagnostics["selected_max_tokens"] = max_tokens
            diagnostics["instructor_retry_exception"] = instructor_retry_exception_diagnostics(exc)
            raise TextFillAPIError(
                _format_instructor_retry_exception(exc),
                diagnostics=diagnostics,
            ) from exc
        except IncompleteOutputException as exc:
            diagnostics = trace.to_diagnostics()
            diagnostics["selected_max_tokens"] = max_tokens
            diagnostics["incomplete_output"] = {
                "last_completion": getattr(exc, "last_completion", None) is not None,
            }
            raise TextFillAPIError(
                "structured_text_fill_truncated: output hit max_tokens",
                diagnostics=diagnostics,
            ) from exc
        except Exception as exc:
            diagnostics = trace.to_diagnostics()
            diagnostics["selected_max_tokens"] = max_tokens
            raise TextFillAPIError(str(exc), diagnostics=diagnostics) from exc

        elapsed = time.monotonic() - t0
        cost_tracker.record(
            "phase_2:text_fill",
            synthesize_cost_summary(raw_response, model=normalized_model, elapsed_s=elapsed),
            task_id=str(task.get("id") or ""),
            site=task.get("site") if isinstance(task.get("site"), str) else None,
        )
        diagnostics = trace.to_diagnostics()
        diagnostics["selected_max_tokens"] = max_tokens
        return (payload.model_dump(), "instructor_anthropic", diagnostics)

    async def _call() -> Any:
        async with get_api_semaphore():
            return await client.messages.create(
                model=normalize_model_for_auth(model),
                max_tokens=_TEXT_FILL_DEFAULT_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
                **temperature_kwargs_for_model(model, 0.7),
            )

    response = await call_with_retry(_call, retries=3, label="phase2b-text-fill")
    parts: list[str] = []
    for item in getattr(response, "content", []) or []:
        if getattr(item, "type", None) == "text":
            parts.append(str(getattr(item, "text", "")))
        elif isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text") or ""))
    return ("".join(parts), "shared_api")


def _format_instructor_retry_exception(exc: InstructorRetryException) -> str:
    attempts = getattr(exc, "failed_attempts", None)
    details: list[str] = []
    if isinstance(attempts, list):
        for attempt in attempts[-3:]:
            error = getattr(attempt, "exception", None)
            if error is not None:
                details.append(str(error))
    if details:
        return "structured_text_fill_failed: " + " | ".join(details)
    return f"structured_text_fill_failed: {exc}"


def _openrouter_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_AUTH_TOKEN", "").strip()) and bool(
        os.environ.get("ANTHROPIC_BASE_URL", "").strip()
    )


def _anthropic_available() -> bool:
    return bool(os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "").strip()) or bool(
        os.environ.get("ANTHROPIC_API_KEY", "").strip()
    )


def _call_openrouter(prompt: str, model: str) -> str:
    base_url = os.environ["ANTHROPIC_BASE_URL"].rstrip("/")
    auth_token = os.environ["ANTHROPIC_AUTH_TOKEN"].strip()
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "temperature": 0.7,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "provider": {"require_parameters": True},
        },
        timeout=90,
    )
    response.raise_for_status()
    data = response.json()
    return str(((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "")


def _call_anthropic_fallback(prompt: str, model: str) -> tuple[str, str]:
    oauth_token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "").strip()
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    headers = {
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    auth_path = "anthropic_api"
    if oauth_token:
        headers["Authorization"] = f"Bearer {oauth_token}"
        auth_path = "oauth"
    elif api_key:
        headers["x-api-key"] = api_key
    else:
        raise RuntimeError("no anthropic fallback credentials configured")

    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json={
            "model": _direct_anthropic_model_name(model),
            "max_tokens": 1200,
            "temperature": 0.7,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=90,
    )
    response.raise_for_status()
    data = response.json()
    content = data.get("content") or []
    if isinstance(content, list):
        text_parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return ("".join(text_parts), auth_path)
    return (str(content), auth_path)


def _direct_anthropic_model_name(model: str) -> str:
    return model.split("/", 1)[1] if "/" in model else model
