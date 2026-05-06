"""Eval-awareness payload rewrite API for the Phase 4 iterator."""

from __future__ import annotations

import copy
import json
import logging
import time
from typing import Any, Literal

import instructor
from anthropic import AsyncAnthropic
from instructor.core.exceptions import IncompleteOutputException, InstructorRetryException
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from worldsim.adversarial_actions import build_action_payload_guidance
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
)
from worldsim.phase_4.concurrency import get_api_semaphore
from worldsim.phase_4.prompt_payloads import (
    format_json_for_model_prompt,
    sanitize_action_payload_guidance_for_model_prompt,
    sanitize_model_prompt_payload,
    sanitize_task_for_model_prompt,
)
from worldsim.phase_4.variant_api import (
    _action_payload_text_contract_error,
    _seed_equivalent,
    _variant_payload_contract,
)
from worldsim.prompt_loading import load_prompt
from worldsim.seeding import validate_data_seed
from worldsim.text_payload_contract import (
    TextPayloadResponse,
    selected_payload,
    selected_seed_payload_text,
)

logger = logging.getLogger(__name__)

_INITIAL_MAX_TOKENS = 8192
_MAX_MAX_TOKENS = 32768
_STRUCTURED_RETRIES = 2


class build_eval_awareness_rewrite(BaseModel):
    """Return a rewritten task seed for eval-awareness iteration."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "inapplicable"]
    adversarial_data_seed: dict[str, Any] | None = Field(default=None)
    payload_text: TextPayloadResponse | None = None
    rewrite_summary: str | None = Field(default=None, max_length=800)
    cue_reasoning: str | None = Field(
        default=None,
        max_length=1000,
        description="Why the cue diagnosis applies to the current payload.",
    )
    rewrite_plan: str | None = Field(
        default=None,
        max_length=1000,
        description="How the rewrite removes cue-related wording while preserving constraints.",
    )
    awareness_objective: str | None = Field(default=None, max_length=600)
    attack_objective: str | None = Field(default=None, max_length=600)
    reason: str | None = Field(default=None, max_length=400)

    @field_validator("adversarial_data_seed", mode="before")
    @classmethod
    def parse_stringified_seed(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        return parsed if isinstance(parsed, dict) else value

    @model_validator(mode="after")
    def validate_rewrite_contract(self, info: ValidationInfo) -> build_eval_awareness_rewrite:
        context = info.context if isinstance(info.context, dict) else {}
        task = context.get("task") if isinstance(context.get("task"), dict) else None
        if self.status == "inapplicable":
            if not isinstance(self.reason, str) or not self.reason.strip():
                raise ValueError("schema_violation: inapplicable rewrites require reason")
            return self

        if not isinstance(self.adversarial_data_seed, dict) or not self.adversarial_data_seed:
            raise ValueError("schema_violation: ok rewrites require adversarial_data_seed")
        try:
            validate_data_seed(self.adversarial_data_seed, allow_none=False)
        except ValueError as exc:
            raise ValueError(f"schema_violation: invalid adversarial_data_seed: {exc}") from exc
        if self.payload_text is None:
            raise ValueError("schema_violation: ok rewrites require payload_text")
        if not isinstance(self.cue_reasoning, str) or not self.cue_reasoning.strip():
            raise ValueError("schema_violation: ok rewrites require cue_reasoning")
        if not isinstance(self.rewrite_plan, str) or not self.rewrite_plan.strip():
            raise ValueError("schema_violation: ok rewrites require rewrite_plan")
        if task is not None:
            rendered_from_seed = selected_seed_payload_text(task, self.adversarial_data_seed)
            payload_for_validation = self.payload_text.model_dump()
            if isinstance(rendered_from_seed, str) and rendered_from_seed:
                payload_for_validation = {**payload_for_validation, "rendered_payload": rendered_from_seed}
            action_error = _action_payload_text_contract_error(task, payload_for_validation)
            if action_error is not None:
                raise ValueError("payload_text_contract_violation: " + action_error)
            if _seed_equivalent(self.adversarial_data_seed, task.get("adversarial_data_seed")):
                raise ValueError("unchanged_seed: ok rewrite must change selected payload text")
        return self


def _instructor_client_for(client: AsyncAnthropic) -> Any:
    injected = getattr(client, "_worldsim_eval_awareness_rewrite_instructor_client", None)
    if injected is not None:
        return injected
    return instructor.from_anthropic(client, mode=instructor.Mode.ANTHROPIC_TOOLS)


def _model_metadata(task: dict[str, Any]) -> dict[str, str]:
    return {"user_id": "worldsim-v5-eval-awareness-rewrite", "task_id": str(task.get("id", ""))}


def _xml_section(tag: str, value: Any) -> str:
    return f"<{tag}>\n{format_json_for_model_prompt(value)}\n</{tag}>"


def _rewrite_constraints(task: dict[str, Any]) -> dict[str, Any]:
    constraints: dict[str, Any] = {
        "edit_only_selected_payload_text": True,
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
    payload_contract = _variant_payload_contract(task)
    if payload_contract is not None:
        constraints["payload_contract"] = sanitize_model_prompt_payload(payload_contract)
    action_guidance = build_action_payload_guidance(task)
    if action_guidance is not None:
        constraints["action_guidance"] = sanitize_action_payload_guidance_for_model_prompt(
            action_guidance
        )
    return constraints


def _task_context(task: dict[str, Any]) -> dict[str, Any]:
    task_payload = sanitize_task_for_model_prompt(task)
    return {
        key: task_payload.get(key)
        for key in (
            "id",
            "site",
            "sites",
            "instruction",
            "start_urls",
            "delivery_channel",
            "adversarial_action",
            "action_target",
            "reward_function",
        )
        if key in task_payload
    }


def _build_messages(
    task: dict[str, Any],
    cue_diagnosis: dict[str, Any],
    *,
    iteration: int,
    prior_attempts: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    prompt = load_prompt("eval-awareness-rewrite")
    selected = selected_payload(task)
    selected_payload_value = selected if isinstance(selected, dict) else None
    input_xml = "\n\n".join(
        [
            "<payload_rewrite>",
            _xml_section("iteration", iteration),
            _xml_section("payload", selected_payload_value),
            _xml_section("cue_diagnosis", json.loads(json.dumps(cue_diagnosis, default=str))),
            _xml_section("rewrite_constraints", _rewrite_constraints(task)),
            _xml_section("task_context", _task_context(task)),
            _xml_section("previous_iterations", prior_attempts or []),
            "</payload_rewrite>",
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


def _merge_rewrite(base_task: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base_task)
    merged["adversarial_data_seed"] = payload.get("adversarial_data_seed") or merged.get(
        "adversarial_data_seed"
    )
    if isinstance(payload.get("payload_text"), dict):
        merged["payload_text"] = payload["payload_text"]
    merged["eval_awareness_rewrite"] = {
        key: value
        for key, value in {
            "rewrite_summary": payload.get("rewrite_summary"),
            "cue_reasoning": payload.get("cue_reasoning"),
            "rewrite_plan": payload.get("rewrite_plan"),
            "awareness_objective": payload.get("awareness_objective"),
            "attack_objective": payload.get("attack_objective"),
        }.items()
        if value is not None
    }
    return merged


def _failed_rewrite(
    task: dict[str, Any],
    *,
    failure_class: str,
    reason: str,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    skipped = copy.deepcopy(task)
    status: dict[str, Any] = {
        "status": "failed",
        "failure_class": failure_class,
        "reason": reason,
    }
    if diagnostics:
        status["api_diagnostics"] = diagnostics
    skipped["variant_status"] = status
    return skipped


async def generate_eval_awareness_rewrite_api(
    task: dict[str, Any],
    cue_diagnosis: dict[str, Any],
    *,
    iteration: int,
    prior_attempts: list[dict[str, Any]] | None = None,
    sandbox_model: str = "claude-sonnet-4-6",
    client: AsyncAnthropic | None = None,
) -> dict[str, Any]:
    """Generate one sequential eval-awareness rewrite of ``task``."""

    task_id = str(task.get("id") or "unknown")
    client = client or get_client()
    normalized_model = normalize_model_for_auth(sandbox_model)
    instructor_client = _instructor_client_for(client)
    messages = _build_messages(
        task,
        cue_diagnosis,
        iteration=iteration,
        prior_attempts=prior_attempts,
    )
    trace = InstructorCallTrace(
        phase="phase_4",
        label="eval-awareness-rewrite",
        task_id=task_id,
        site=task.get("site") if isinstance(task.get("site"), str) else None,
        response_model_name=build_eval_awareness_rewrite.__name__,
    )
    hooks = build_instructor_hooks(trace)
    response_context = {"task": task}
    max_tokens = _INITIAL_MAX_TOKENS
    attempts = 0
    raw_response: Any = None
    payload: dict[str, Any] | None = None
    last_error: str | None = None
    failure_class: str | None = None
    t0 = time.monotonic()

    while attempts < 2:
        attempts += 1
        try:

            async def _attempt(mt: int = max_tokens) -> tuple[build_eval_awareness_rewrite, Any]:
                async with get_api_semaphore():
                    parsed, completion = await instructor_client.messages.create_with_completion(
                        model=normalized_model,
                        max_tokens=mt,
                        messages=messages,
                        response_model=build_eval_awareness_rewrite,
                        context=response_context,
                        max_retries=instructor_semantic_retrying(_STRUCTURED_RETRIES),
                        hooks=hooks,
                        metadata=_model_metadata(task),
                    )
                    if getattr(completion, "stop_reason", None) == "max_tokens":
                        raise IncompleteOutputException(last_completion=completion)
                    return parsed, completion

            parsed_payload, raw_response = await call_with_retry(
                _attempt,
                retries=3,
                label=f"eval-awareness-rewrite-{task_id}-i{iteration}",
            )
            payload = parsed_payload.model_dump(exclude_none=True)
            break
        except IncompleteOutputException:
            failure_class = "response_truncated"
            last_error = "response_truncated"
            if max_tokens < _MAX_MAX_TOKENS:
                max_tokens = _MAX_MAX_TOKENS
                continue
            break
        except InstructorRetryException as exc:
            diagnostics = trace.to_diagnostics()
            diagnostics["instructor_retry_exception"] = instructor_retry_exception_diagnostics(exc)
            reason = str(exc).splitlines()[0] if str(exc).strip() else type(exc).__name__
            return _failed_rewrite(
                task,
                failure_class="schema_violation",
                reason=reason[:500],
                diagnostics=diagnostics,
            )
        except Exception as exc:
            failure_class = classify_api_exception(exc)
            last_error = f"{failure_class}: {exc}"
            break

    if raw_response is not None:
        elapsed = time.monotonic() - t0
        cost_tracker.record(
            "phase_4:eval_awareness_iterator",
            synthesize_cost_summary(raw_response, model=normalized_model, elapsed_s=elapsed),
            task_id=task_id,
            site=task.get("site"),
        )
    diagnostics = trace.to_diagnostics()
    diagnostics["selected_max_tokens"] = max_tokens
    if payload is None:
        return _failed_rewrite(
            task,
            failure_class=failure_class or "no_tool_use",
            reason=last_error or "no_tool_use",
            diagnostics=diagnostics,
        )
    if payload.get("status") == "inapplicable":
        skipped = copy.deepcopy(task)
        skipped["variant_status"] = {
            "status": "inapplicable",
            "reason": payload.get("reason", "rewrite incompatible with target field"),
            "api_diagnostics": diagnostics,
        }
        return skipped
    if payload.get("status") != "ok":
        return _failed_rewrite(
            task,
            failure_class="unexpected_tool_status",
            reason=f"unexpected tool status={payload.get('status')!r}",
            diagnostics=diagnostics,
        )
    return _merge_rewrite(task, payload)


__all__ = ["generate_eval_awareness_rewrite_api"]
