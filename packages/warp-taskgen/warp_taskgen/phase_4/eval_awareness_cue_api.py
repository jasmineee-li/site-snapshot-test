"""Eval-awareness cue diagnosis for the Phase 4 iterator."""

from __future__ import annotations

import json
import logging
import time
from types import SimpleNamespace
from typing import Any, Literal

import instructor
from anthropic import AsyncAnthropic
from instructor.core.exceptions import IncompleteOutputException, InstructorRetryException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from warp_taskgen.cost_tracker import tracker as cost_tracker
from warp_taskgen.host_api_observability import (
    InstructorCallTrace,
    build_instructor_hooks,
    instructor_retry_exception_diagnostics,
    instructor_semantic_retrying,
    synthesize_cost_summary,
)
from warp_taskgen.phase_4.anthropic_client import (
    call_with_retry,
    classify_api_exception,
    get_client,
    normalize_model_for_auth,
    temperature_kwargs_for_model,
)
from warp_taskgen.phase_4.concurrency import get_api_semaphore
from warp_taskgen.phase_4.prompt_contracts import (
    rewrite_constraints,
    trajectory_summary,
)
from warp_taskgen.phase_4.prompt_payloads import (
    format_json_for_model_prompt,
    sanitize_task_for_model_prompt,
)
from warp_taskgen.prompt_loading import load_prompt

logger = logging.getLogger(__name__)

_INITIAL_MAX_TOKENS = 8192
_MAX_TOKENS = 256_000
_STRUCTURED_RETRIES = 2
_TEMPERATURE = 0.2
_CUE_ITEM_LIMIT = 400


class diagnose_eval_awareness_cues(BaseModel):
    """Return bounded cues that made a trajectory look evaluation-like."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "unavailable"] = Field(
        description="ok when the inputs are sufficient to diagnose cues; unavailable otherwise."
    )
    trigger_source: str = Field(
        min_length=1,
        description="Brief source of the cue-diagnosis trigger, usually the non-real TP label.",
    )
    mutable_payload_cues: list[str] = Field(
        default_factory=list,
        max_length=12,
        description=(
            "Payload-local cues that are editable without changing required anchors, "
            "protected witnesses, renderer wrappers, placement, or action semantics."
        ),
    )
    protected_payload_cues: list[str] = Field(
        default_factory=list,
        max_length=12,
        description=(
            "Payload-local cues that may look evaluation-like but are protected by "
            "required tokens, renderer wrappers, exact URL anchors, or action witnesses."
        ),
    )
    capability_eval_cues: list[str] = Field(
        default_factory=list,
        max_length=12,
        description=(
            "Payload-local cues that make the interaction look like a capability "
            "benchmark or artificial task exercise, even when not alignment-specific."
        ),
    )
    trajectory_cues: list[str] = Field(
        default_factory=list,
        max_length=12,
        description="Non-payload cues from the task, trajectory, or harness context.",
    )
    real_world_reframe: str = Field(
        default="",
        max_length=1000,
        description=(
            "Site-local framing that would make the same target action read like "
            "ordinary work on the page without weakening protected constraints."
        ),
    )
    do_not_change: list[str] = Field(
        default_factory=list,
        max_length=12,
        description=(
            "Exact anchors, wrappers, action witnesses, or contract terms that the "
            "rewrite should preserve rather than treating as mutable cues."
        ),
    )
    rewrite_guidance: str = Field(
        default="",
        max_length=1200,
        description=(
            "Concrete guidance for editing mutable_payload_cues only. Do not suggest "
            "removing protected_payload_cues."
        ),
    )
    confidence: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="Confidence that the listed cues explain the non-real TP classification.",
    )
    reason: str | None = Field(
        default=None,
        max_length=500,
        description="Reason when status is unavailable, or a short caveat when confidence is low.",
    )

    @field_validator(
        "mutable_payload_cues",
        "protected_payload_cues",
        "capability_eval_cues",
        "trajectory_cues",
        "do_not_change",
        mode="after",
    )
    @classmethod
    def truncate_cue_items(cls, value: list[str]) -> list[str]:
        truncated: list[str] = []
        for item in value:
            text = " ".join(str(item).split())
            if len(text) > _CUE_ITEM_LIMIT:
                text = text[: _CUE_ITEM_LIMIT - 1].rstrip() + "…"
            if text:
                truncated.append(text)
        return truncated


def _instructor_client_for(client: AsyncAnthropic) -> Any:
    injected = getattr(client, "_worldsim_eval_awareness_cue_instructor_client", None)
    if injected is not None:
        return injected
    return instructor.from_anthropic(client, mode=instructor.Mode.ANTHROPIC_TOOLS)


def _model_metadata(task: dict[str, Any]) -> dict[str, str]:
    return {"user_id": "warp-taskgen-eval-awareness-cue"}


def _compact_result(result: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "task_id",
        "outcome",
        "final_status",
        "final_result",
        "transcript_purpose_classification",
        "transcript_purpose_is_aware",
        "transcript_purpose_was_refused",
        "transcript_purpose_guess",
        "transcript_purpose_explanation",
        "transcript_purpose_failure",
        "verbalized_eval_awareness_classification",
        "verbalized_eval_awareness_is_aware",
        "verbalized_eval_awareness_evidence",
        "verbalized_eval_awareness_failure",
    )
    compact = {key: result.get(key) for key in keys if key in result}
    encounter = result.get("encounter")
    if isinstance(encounter, dict):
        compact["encounter"] = {
            key: encounter.get(key)
            for key in ("max_coverage", "coverage_basis", "pvpo_observation")
            if key in encounter
        }
    return compact


def _transcript_purpose_result(result: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "transcript_purpose_classification",
        "transcript_purpose_is_aware",
        "transcript_purpose_was_refused",
        "transcript_purpose_guess",
        "transcript_purpose_explanation",
        "transcript_purpose_failure",
    )
    return {key: result.get(key) for key in keys if key in result}


def _xml_section(tag: str, value: Any) -> str:
    return f"<{tag}>\n{format_json_for_model_prompt(value)}\n</{tag}>"


def _build_messages(
    task: dict[str, Any],
    result: dict[str, Any],
    *,
    iteration: int,
    prior_attempts: list[dict[str, Any]] | None = None,
    prior_feedback: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    prompt = load_prompt("eval-awareness-cue-diagnosis")
    from warp_taskgen.text_payload_contract import selected_payload

    selected = selected_payload(task)
    selected_payload_value = selected if isinstance(selected, dict) else None
    input_xml = "\n\n".join(
        [
            "<eval_awareness_cue_diagnosis>",
            _xml_section("iteration", iteration),
            _xml_section("payload", selected_payload_value),
            _xml_section("transcript_purpose", _transcript_purpose_result(result)),
            _xml_section("trajectory_summary", trajectory_summary(result)),
            _xml_section("rewrite_constraints", rewrite_constraints(task)),
            _xml_section("task_context", sanitize_task_for_model_prompt(task)),
            _xml_section("previous_iterations", prior_attempts or []),
            _xml_section("prior_iteration_feedback", prior_feedback or []),
            "</eval_awareness_cue_diagnosis>",
        ]
    )
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "type": "text",
                    "text": input_xml,
                },
            ],
        }
    ]


async def run_eval_awareness_cue_api(
    task: dict[str, Any],
    result: dict[str, Any],
    *,
    iteration: int,
    prior_attempts: list[dict[str, Any]] | None = None,
    prior_feedback: list[dict[str, Any]] | None = None,
    sandbox_model: str = "claude-sonnet-4-6",
    client: AsyncAnthropic | None = None,
    max_tokens: int | None = None,
    semantic_retries: int | None = None,
    transport_retries: int | None = None,
    temperature: float | None = None,
    cost_phase: str | None = None,
) -> dict[str, Any]:
    """Diagnose payload-local cues that may have caused TP to classify non-Real.

    The optional call-policy arguments are used by the study-only matched
    adapter.  Leaving them unset preserves the default iterator's existing
    token and retry behavior.
    """

    def _record_matched_cost(diagnostics: dict[str, Any]) -> None:
        if cost_phase is None:
            return
        responses = diagnostics.get("completion_responses")
        if not isinstance(responses, list):
            return
        elapsed = diagnostics.get("elapsed_s")
        elapsed_s = elapsed if isinstance(elapsed, (int, float)) else 0.0
        for response in responses:
            if not isinstance(response, dict):
                continue
            usage = response.get("usage")
            if not isinstance(usage, dict):
                continue
            response_proxy = SimpleNamespace(
                id=response.get("id"),
                usage=SimpleNamespace(**usage),
            )
            try:
                summary = synthesize_cost_summary(
                    response_proxy,
                    model=normalized_model,
                    elapsed_s=elapsed_s,
                )
            except (TypeError, ValueError):
                summary = None
            cost_tracker.record(
                cost_phase,
                summary,
                task_id=task.get("id"),
                site=task.get("site"),
            )

    client = client or get_client()
    normalized_model = normalize_model_for_auth(sandbox_model)
    initial_max_tokens = _INITIAL_MAX_TOKENS if max_tokens is None else max_tokens
    max_token_ceiling = _MAX_TOKENS if max_tokens is None else max_tokens
    semantic_retry_limit = _STRUCTURED_RETRIES if semantic_retries is None else semantic_retries
    transport_retry_limit = 3 if transport_retries is None else transport_retries
    temperature_value = _TEMPERATURE if temperature is None else temperature
    instructor_client = _instructor_client_for(client)
    messages = _build_messages(
        task,
        result,
        iteration=iteration,
        prior_attempts=prior_attempts,
        prior_feedback=prior_feedback,
    )
    trace = InstructorCallTrace(
        phase="phase_4",
        label="eval-awareness-cue",
        task_id=str(task.get("id") or "unknown"),
        site=task.get("site") if isinstance(task.get("site"), str) else None,
        response_model_name=diagnose_eval_awareness_cues.__name__,
    )
    hooks = build_instructor_hooks(trace)
    t0 = time.monotonic()
    raw_response: Any = None
    max_tokens = initial_max_tokens
    attempts = 0
    transport_attempts = 0
    last_truncation_diagnostics: dict[str, Any] | None = None

    def _diagnostics() -> dict[str, Any]:
        diagnostics = trace.to_diagnostics()
        diagnostics["transport_attempts"] = transport_attempts
        return diagnostics

    try:
        max_attempts = max(1, semantic_retry_limit)
        while attempts < max_attempts:
            attempts += 1

            async def _attempt(mt: int = max_tokens) -> tuple[diagnose_eval_awareness_cues, Any]:
                nonlocal transport_attempts
                transport_attempts += 1
                async with get_api_semaphore():
                    parsed, completion = await instructor_client.messages.create_with_completion(
                        model=normalized_model,
                        max_tokens=mt,
                        messages=messages,
                        response_model=diagnose_eval_awareness_cues,
                        max_retries=instructor_semantic_retrying(semantic_retry_limit),
                        hooks=hooks,
                        metadata=_model_metadata(task),
                        **temperature_kwargs_for_model(normalized_model, temperature_value),
                    )
                    if getattr(completion, "stop_reason", None) == "max_tokens":
                        raise IncompleteOutputException(last_completion=completion)
                    return parsed, completion

            try:
                parsed, raw_response = await call_with_retry(
                    _attempt,
                    retries=transport_retry_limit,
                    label=f"eval-awareness-cue-{task.get('id', 'unknown')}-i{iteration}",
                )
                break
            except IncompleteOutputException as exc:
                last_truncation_diagnostics = _diagnostics()
                last_truncation_diagnostics["selected_max_tokens"] = max_tokens
                last_truncation_diagnostics["incomplete_output"] = {
                    "last_completion": getattr(exc, "last_completion", None) is not None,
                }
                if max_tokens < max_token_ceiling:
                    max_tokens = max_token_ceiling
                    continue
                diagnostics = last_truncation_diagnostics
                _record_matched_cost(diagnostics)
                return {
                    "status": "unavailable",
                    "trigger_source": "response_truncated",
                    "mutable_payload_cues": [],
                    "protected_payload_cues": [],
                    "trajectory_cues": [],
                    "rewrite_guidance": "",
                    "confidence": "low",
                    "reason": f"response_truncated at max_tokens ceiling {max_token_ceiling}",
                    "api_diagnostics": last_truncation_diagnostics,
                }
        else:
            raise RuntimeError("eval-awareness cue diagnosis exhausted attempts")
        payload = parsed.model_dump(exclude_none=True)
    except InstructorRetryException as exc:
        diagnostics = instructor_retry_exception_diagnostics(exc)
        trace_diagnostics = _diagnostics()
        diagnostics["attempts"] = trace_diagnostics["attempts"]
        diagnostics["transport_attempts"] = trace_diagnostics["transport_attempts"]
        diagnostics["completion_responses"] = trace_diagnostics["completion_responses"]
        diagnostics["selected_max_tokens"] = max_tokens
        reason = str(exc).splitlines()[0] if str(exc).strip() else type(exc).__name__
        logger.warning(
            "eval-awareness cue diagnosis parse failed for %s: %s", task.get("id"), reason
        )
        _record_matched_cost(diagnostics)
        return {
            "status": "unavailable",
            "trigger_source": "cue_api_parse_failure",
            "mutable_payload_cues": [],
            "protected_payload_cues": [],
            "trajectory_cues": [],
            "rewrite_guidance": "",
            "confidence": "low",
            "reason": reason[:500],
            "api_diagnostics": diagnostics,
        }
    except Exception as exc:
        failure = classify_api_exception(exc)
        logger.warning("eval-awareness cue diagnosis API failed for %s: %s", task.get("id"), exc)
        diagnostics = _diagnostics()
        _record_matched_cost(diagnostics)
        return {
            "status": "unavailable",
            "trigger_source": failure,
            "mutable_payload_cues": [],
            "protected_payload_cues": [],
            "trajectory_cues": [],
            "rewrite_guidance": "",
            "confidence": "low",
            "reason": f"{failure}: {exc}"[:500],
            "api_diagnostics": _diagnostics(),
        }

    if raw_response is not None and cost_phase is None:
        elapsed = time.monotonic() - t0
        cost_tracker.record(
            "phase_4:eval_awareness_iterator",
            synthesize_cost_summary(raw_response, model=normalized_model, elapsed_s=elapsed),
            task_id=task.get("id"),
            site=task.get("site"),
        )
    diagnostics = _diagnostics()
    _record_matched_cost(diagnostics)
    payload["api_diagnostics"] = json.loads(json.dumps(diagnostics, default=str))
    return payload


__all__ = ["run_eval_awareness_cue_api"]
