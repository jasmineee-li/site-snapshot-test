"""Eval-awareness cue diagnosis for the Phase 4 iterator."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Literal

import instructor
from anthropic import AsyncAnthropic
from instructor.core.exceptions import InstructorRetryException
from pydantic import BaseModel, ConfigDict, Field

from worldsim.cost_tracker import tracker as cost_tracker
from worldsim.host_api_observability import (
    InstructorCallTrace,
    build_instructor_hooks,
    instructor_retry_exception_diagnostics,
    instructor_semantic_retrying,
    synthesize_cost_summary,
)
from worldsim.phase_4.anthropic_client import (
    call_with_retry,
    classify_api_exception,
    get_client,
    normalize_model_for_auth,
    temperature_kwargs_for_model,
)
from worldsim.phase_4.concurrency import get_api_semaphore
from worldsim.phase_4.prompt_payloads import (
    format_json_for_model_prompt,
    sanitize_task_for_model_prompt,
)
from worldsim.prompt_loading import load_prompt
from worldsim.text_payload_contract import selected_payload

logger = logging.getLogger(__name__)

_MAX_TOKENS = 2048
_STRUCTURED_RETRIES = 2
_TEMPERATURE = 0.2


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
    payload_cues: list[str] = Field(
        default_factory=list,
        max_length=12,
        description="Cues in the selected payload text that could make it look like an eval.",
    )
    trajectory_cues: list[str] = Field(
        default_factory=list,
        max_length=12,
        description="Non-payload cues from the task, trajectory, or harness context.",
    )
    removable_payload_cues: list[str] = Field(
        default_factory=list,
        max_length=12,
        description="Payload-local cues that a rewrite can plausibly remove.",
    )
    rewrite_guidance: str = Field(
        default="",
        max_length=1200,
        description="Concrete payload-local guidance for the rewrite API.",
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


def _instructor_client_for(client: AsyncAnthropic) -> Any:
    injected = getattr(client, "_worldsim_eval_awareness_cue_instructor_client", None)
    if injected is not None:
        return injected
    return instructor.from_anthropic(client, mode=instructor.Mode.ANTHROPIC_TOOLS)


def _model_metadata(task: dict[str, Any]) -> dict[str, str]:
    return {"user_id": "worldsim-v5-eval-awareness-cue", "task_id": str(task.get("id", ""))}


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


def _trajectory_summary(result: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "task_id",
        "outcome",
        "final_status",
        "final_result",
        "verbalized_eval_awareness_classification",
        "verbalized_eval_awareness_is_aware",
        "verbalized_eval_awareness_evidence",
        "verbalized_eval_awareness_failure",
    )
    summary = {key: result.get(key) for key in keys if key in result}
    encounter = result.get("encounter")
    if isinstance(encounter, dict):
        summary["encounter"] = {
            key: encounter.get(key)
            for key in ("max_coverage", "coverage_basis", "pvpo_observation")
            if key in encounter
        }
    return summary


def _rewrite_constraints(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "preserve_task": True,
        "preserve_editor_call_structure": True,
        "preserve_required_anchors": True,
        "preserve_target_action": True,
        "preserve_placement": True,
        "respect_field_length_budget": True,
        "required_tokens": task.get("required_tokens"),
        "length_budget": task.get("length_budget"),
        "delivery_channel": task.get("delivery_channel"),
        "selected_payload_index": task.get("selected_payload_index"),
    }


def _xml_section(tag: str, value: Any) -> str:
    return f"<{tag}>\n{format_json_for_model_prompt(value)}\n</{tag}>"


def _build_messages(
    task: dict[str, Any],
    result: dict[str, Any],
    *,
    iteration: int,
    prior_attempts: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    prompt = load_prompt("eval-awareness-cue-diagnosis")
    selected = selected_payload(task)
    selected_payload_value = selected if isinstance(selected, dict) else None
    input_xml = "\n\n".join(
        [
            "<eval_awareness_cue_diagnosis>",
            _xml_section("iteration", iteration),
            _xml_section("payload", selected_payload_value),
            _xml_section("transcript_purpose", _transcript_purpose_result(result)),
            _xml_section("trajectory_summary", _trajectory_summary(result)),
            _xml_section("rewrite_constraints", _rewrite_constraints(task)),
            _xml_section("task_context", sanitize_task_for_model_prompt(task)),
            _xml_section("previous_iterations", prior_attempts or []),
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
    sandbox_model: str = "claude-sonnet-4-6",
    client: AsyncAnthropic | None = None,
) -> dict[str, Any]:
    """Diagnose payload-local cues that may have caused TP to classify non-Real."""

    client = client or get_client()
    normalized_model = normalize_model_for_auth(sandbox_model)
    instructor_client = _instructor_client_for(client)
    messages = _build_messages(task, result, iteration=iteration, prior_attempts=prior_attempts)
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
    try:

        async def _attempt() -> tuple[diagnose_eval_awareness_cues, Any]:
            async with get_api_semaphore():
                parsed, completion = await instructor_client.messages.create_with_completion(
                    model=normalized_model,
                    max_tokens=_MAX_TOKENS,
                    messages=messages,
                    response_model=diagnose_eval_awareness_cues,
                    max_retries=instructor_semantic_retrying(_STRUCTURED_RETRIES),
                    hooks=hooks,
                    metadata=_model_metadata(task),
                    **temperature_kwargs_for_model(normalized_model, _TEMPERATURE),
                )
                return parsed, completion

        parsed, raw_response = await call_with_retry(
            _attempt,
            retries=3,
            label=f"eval-awareness-cue-{task.get('id', 'unknown')}-i{iteration}",
        )
        payload = parsed.model_dump(exclude_none=True)
    except InstructorRetryException as exc:
        diagnostics = instructor_retry_exception_diagnostics(exc)
        reason = str(exc).splitlines()[0] if str(exc).strip() else type(exc).__name__
        logger.warning("eval-awareness cue diagnosis parse failed for %s: %s", task.get("id"), reason)
        return {
            "status": "unavailable",
            "trigger_source": "cue_api_parse_failure",
            "payload_cues": [],
            "trajectory_cues": [],
            "removable_payload_cues": [],
            "rewrite_guidance": "",
            "confidence": "low",
            "reason": reason[:500],
            "api_diagnostics": diagnostics,
        }
    except Exception as exc:
        failure = classify_api_exception(exc)
        logger.warning("eval-awareness cue diagnosis API failed for %s: %s", task.get("id"), exc)
        return {
            "status": "unavailable",
            "trigger_source": failure,
            "payload_cues": [],
            "trajectory_cues": [],
            "removable_payload_cues": [],
            "rewrite_guidance": "",
            "confidence": "low",
            "reason": f"{failure}: {exc}"[:500],
        }

    if raw_response is not None:
        elapsed = time.monotonic() - t0
        cost_tracker.record(
            "phase_4:eval_awareness_iterator",
            synthesize_cost_summary(raw_response, model=normalized_model, elapsed_s=elapsed),
            task_id=task.get("id"),
            site=task.get("site"),
        )
    payload["api_diagnostics"] = json.loads(json.dumps(trace.to_diagnostics(), default=str))
    return payload


__all__ = ["run_eval_awareness_cue_api"]
