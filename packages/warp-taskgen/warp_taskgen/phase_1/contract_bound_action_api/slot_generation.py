"""Host-API slot generation for contract-bound Phase 1 action tasks."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

from warp_taskgen.cost_tracker import tracker as cost_tracker
from warp_taskgen.host_api_observability import synthesize_cost_summary, usage_dict
from warp_taskgen.phase_1.contract_bound_action_api.contract_selection import (
    SelectedActionTaskContract,
    _filter_contract_to_validated_anchors,
    select_action_task_contracts,
)
from warp_taskgen.phase_1.contract_bound_action_api.instruction_validation import (
    _select_valid_slots,
)
from warp_taskgen.phase_1.contract_bound_action_api.prompt_rendering import (
    _build_messages,
)
from warp_taskgen.phase_1.contract_bound_action_api.slot_compilation import (
    compile_action_task_slot,
)
from warp_taskgen.phase_4.anthropic_client import (
    APIConnectionError,
    APIResponseValidationError,
    APIStatusError,
    call_with_retry,
    classify_api_exception,
    get_client,
    normalize_model_for_auth,
    temperature_kwargs_for_model,
)
from warp_taskgen.phase_4.concurrency import get_api_semaphore

logger = logging.getLogger(__name__)


_EMIT_ACTION_TASK_SLOTS_TOOL_NAME = "emit_action_task_slots"
_OVERGENERATION_MULTIPLIER = 1.5
_MAX_SEMANTIC_RETRIES = 2
_MAX_OUTPUT_TOKENS = 32768
_STREAM_OUTPUT_TOKEN_THRESHOLD = 32768


def contract_bound_tool_schema_digest() -> str:
    """Return a stable digest for cache keys."""

    return sha256(
        json.dumps(build_emit_action_task_slots_tool(), sort_keys=True).encode("utf-8")
    ).hexdigest()


async def generate_contract_bound_action_tasks_api(
    *,
    site_name: str,
    task_card_plan: Mapping[str, Any],
    route_contracts: Mapping[str, Any],
    profile: Mapping[str, Any],
    requested_count: int,
    action_counts: Mapping[str, int] | None = None,
    sandbox_model: str = "claude-sonnet-4-6",
    task_number_start: int = 1,
    cost_report_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Generate and compile host-action-only tasks for one site."""

    if (
        isinstance(task_number_start, bool)
        or not isinstance(task_number_start, int)
        or task_number_start < 1
    ):
        raise ValueError("task_number_start must be a positive integer")

    contracts = select_action_task_contracts(
        site_name=site_name,
        task_card_plan=task_card_plan,
        route_contracts=route_contracts,
        requested_count=requested_count,
        action_counts=action_counts,
    )
    contracts = [
        _filter_contract_to_validated_anchors(contract, profile=profile) for contract in contracts
    ]
    compiled: list[dict[str, Any]] = []
    next_index = task_number_start
    for contract in contracts:
        slot_kwargs: dict[str, Any] = {
            "contract": contract,
            "profile": profile,
            "sandbox_model": sandbox_model,
        }
        if cost_report_path is not None:
            slot_kwargs["cost_report_path"] = cost_report_path
        slots = await _generate_slots_for_contract(**slot_kwargs)
        for offset, slot in enumerate(slots[: contract.count]):
            compiled.append(
                compile_action_task_slot(
                    slot,
                    contract=contract,
                    task_number=next_index + offset,
                    assignment_index=offset,
                )
            )
        next_index += contract.count
    return compiled


def build_emit_action_task_slots_tool() -> dict[str, Any]:
    """Return the forced tool schema for semantic action-task slots."""

    return {
        "name": _EMIT_ACTION_TASK_SLOTS_TOOL_NAME,
        "description": (
            "Emit semantic slots for contract-bound public-action tasks. Do not "
            "emit route_id, task_card_id, start_urls, data_seed, reward_function, "
            "editor methods, evaluator configs, or provenance; host code compiles "
            "those fields from the selected contract."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "slots": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "task_slug": {"type": "string"},
                            "instruction": {"type": "string"},
                            "carrier_title": {"type": "string"},
                            "carrier_body": {"type": "string"},
                            "benign_reference_phrase": {"type": "string"},
                            "abstract_purpose_tag": {"type": "string"},
                            "style_tag": {"type": "string"},
                        },
                        "required": [
                            "task_slug",
                            "instruction",
                            "carrier_title",
                            "carrier_body",
                            "benign_reference_phrase",
                            "abstract_purpose_tag",
                            "style_tag",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["slots"],
            "additionalProperties": False,
        },
    }


async def _generate_slots_for_contract(
    *,
    contract: SelectedActionTaskContract,
    profile: Mapping[str, Any],
    sandbox_model: str,
    cost_report_path: Path | None = None,
) -> list[dict[str, Any]]:
    requested_slots = max(contract.count, int(contract.count * _OVERGENERATION_MULTIPLIER))
    feedback: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    for attempt in range(_MAX_SEMANTIC_RETRIES + 1):
        slot_kwargs: dict[str, Any] = {
            "contract": contract,
            "profile": profile,
            "requested_slots": requested_slots,
            "feedback": feedback,
            "sandbox_model": sandbox_model,
        }
        if cost_report_path is not None:
            slot_kwargs["cost_report_path"] = cost_report_path
        slots = await _call_slots_api(**slot_kwargs)
        accepted, feedback = _select_valid_slots(slots, contract=contract)
        if len(accepted) >= contract.count:
            return accepted[: contract.count]
        feedback.append(
            {
                "code": "UNDERFILLED_VALID_SLOT_COUNT",
                "message": (
                    f"{len(accepted)} valid slots survived host checks; {contract.count} required"
                ),
                "repair_hint": (
                    "Return additional distinct slots. Do not emit structural "
                    "fields; vary wording, slug, reference phrase, purpose tag, "
                    "carrier title, and carrier body."
                ),
            }
        )
        logger.warning(
            "Phase 1 contract-bound API underfilled %s/%s for %s/%s on attempt %d/%d",
            len(accepted),
            contract.count,
            contract.site,
            contract.card_id,
            attempt + 1,
            _MAX_SEMANTIC_RETRIES + 1,
        )
    raise ValueError(
        f"contract-bound API produced only {len(accepted)} valid slots for "
        f"{contract.site}/{contract.card_id}; required {contract.count}: {feedback[:5]}"
    )


async def _call_slots_api(
    *,
    contract: SelectedActionTaskContract,
    profile: Mapping[str, Any],
    requested_slots: int,
    feedback: list[dict[str, Any]],
    sandbox_model: str,
    cost_report_path: Path | None = None,
) -> list[dict[str, Any]]:
    report_path = cost_report_path or _default_cost_report_path()
    cost_tracker.ensure_phase1_paid_dispatch_allowed(report_path)
    client = get_client()
    normalized_model = normalize_model_for_auth(sandbox_model)
    system, messages = _build_messages(
        contract=contract,
        profile=profile,
        requested_slots=requested_slots,
        feedback=feedback,
    )
    tool = build_emit_action_task_slots_tool()

    async def _call() -> Any:
        async with get_api_semaphore():
            kwargs = {
                "model": normalized_model,
                "max_tokens": _MAX_OUTPUT_TOKENS,
                "system": system,
                "messages": messages,
                "tools": [tool],
                "tool_choice": {"type": "tool", "name": _EMIT_ACTION_TASK_SLOTS_TOOL_NAME},
                **temperature_kwargs_for_model(normalized_model, 0.7),
            }
            if _MAX_OUTPUT_TOKENS >= _STREAM_OUTPUT_TOKEN_THRESHOLD:
                async with client.messages.stream(**kwargs) as stream:
                    return await stream.get_final_message()
            return await client.messages.create(**kwargs)

    t0 = time.monotonic()
    try:
        response = await call_with_retry(
            _call,
            retries=3,
            label=f"phase1-contract-bound-{contract.site}-{contract.card_id}",
        )
    except Exception as exc:
        elapsed = time.monotonic() - t0
        error_response = _provider_response_from_error(exc)
        if (
            error_response is not None
            or _provider_response_attached(exc)
            or _is_paid_host_exception(exc)
        ):
            summary = (
                synthesize_cost_summary(error_response, model=normalized_model, elapsed_s=elapsed)
                if error_response is not None
                else None
            )
            cost_tracker.record_and_save(
                "phase_1",
                summary,
                report_path,
                site=contract.site,
            )
        failure_class = classify_api_exception(exc)
        raise RuntimeError(
            f"contract-bound Phase 1 API failed for {contract.site}/{contract.card_id} "
            f"({failure_class}): {exc}"
        ) from exc
    elapsed = time.monotonic() - t0
    # Persist immediately after the paid response returns, before slot
    # extraction can reject malformed tool output.
    error_response = _provider_response_with_usage(response)
    summary = (
        synthesize_cost_summary(response, model=normalized_model, elapsed_s=elapsed)
        if error_response is not None
        else None
    )
    cost_tracker.record_and_save(
        "phase_1",
        summary,
        report_path,
        site=contract.site,
    )
    slots = _extract_slots(response)
    if slots is None:
        stop_reason = getattr(response, "stop_reason", None)
        tool_diagnostic = _extract_slot_tool_diagnostic(response)
        if tool_diagnostic is not None:
            raise ValueError(
                "contract-bound Phase 1 API returned invalid "
                f"{_EMIT_ACTION_TASK_SLOTS_TOOL_NAME} tool_use "
                f"(stop_reason={stop_reason!r}, {tool_diagnostic})"
            )
        raise ValueError(
            f"contract-bound Phase 1 API returned no {_EMIT_ACTION_TASK_SLOTS_TOOL_NAME} "
            f"tool_use (stop_reason={stop_reason!r})"
        )
    return slots


def _provider_response_with_usage(response: Any) -> Any | None:
    """Return a provider completion only when its billable usage is present."""

    if response is None:
        return None
    raw_usage = (
        response.get("usage") if isinstance(response, Mapping) else getattr(response, "usage", None)
    )
    if raw_usage is None:
        return None
    for name in ("input_tokens", "output_tokens"):
        value = (
            raw_usage.get(name)
            if isinstance(raw_usage, Mapping)
            else getattr(raw_usage, name, None)
        )
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None

    # Required fields are now known to be present and valid. Optional cache
    # fields may be absent and are normalized to zero by usage_dict.
    usage = usage_dict(response)
    if usage is None:
        return None
    for _name, value in usage.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
    return response


def _provider_response_from_error(error: BaseException) -> Any | None:
    """Find one provider completion attached to a failed logical call."""

    for attribute in ("raw_response", "last_completion", "completion", "response"):
        candidate = getattr(error, attribute, None)
        response = _provider_response_with_usage(candidate)
        if response is not None:
            return response
    return None


def _provider_response_attached(error: BaseException) -> bool:
    """Return whether an exception carries a provider response boundary."""

    return any(
        getattr(error, attribute, None) is not None
        for attribute in ("raw_response", "last_completion", "completion", "response")
    )


def _is_paid_host_exception(error: BaseException) -> bool:
    """Recognize transport/provider failures without counting local bugs."""

    return isinstance(
        error,
        (
            APIStatusError,
            APIConnectionError,
            APIResponseValidationError,
            ConnectionError,
            TimeoutError,
        ),
    )


def _default_cost_report_path() -> Path:
    from warp_taskgen.state import get_state_dir

    return get_state_dir() / "cost_report.json"


def _extract_slots(response: Any) -> list[dict[str, Any]] | None:
    for block in getattr(response, "content", []) or []:
        if isinstance(block, Mapping):
            block_type = block.get("type")
            block_name = block.get("name")
            block_input = block.get("input")
        else:
            block_type = getattr(block, "type", None)
            block_name = getattr(block, "name", None)
            block_input = getattr(block, "input", None)
        if block_type == "tool_use" and block_name == _EMIT_ACTION_TASK_SLOTS_TOOL_NAME:
            payload = dict(block_input or {})
            slots = payload.get("slots")
            if isinstance(slots, str):
                try:
                    parsed_slots = json.loads(slots)
                except json.JSONDecodeError:
                    parsed_slots = None
                if isinstance(parsed_slots, list):
                    slots = parsed_slots
            if isinstance(slots, list):
                return [slot for slot in slots if isinstance(slot, dict)]
    return None


def _extract_slot_tool_diagnostic(response: Any) -> str | None:
    for block in getattr(response, "content", []) or []:
        if isinstance(block, Mapping):
            block_type = block.get("type")
            block_name = block.get("name")
            block_input = block.get("input")
        else:
            block_type = getattr(block, "type", None)
            block_name = getattr(block, "name", None)
            block_input = getattr(block, "input", None)
        if block_type != "tool_use" or block_name != _EMIT_ACTION_TASK_SLOTS_TOOL_NAME:
            continue
        if not isinstance(block_input, Mapping):
            return f"tool_input_type={type(block_input).__name__}"
        keys = sorted(str(key) for key in block_input)
        slots = block_input.get("slots")
        if isinstance(slots, str):
            try:
                parsed_slots = json.loads(slots)
            except json.JSONDecodeError:
                parsed_slots = None
            if isinstance(parsed_slots, list):
                dict_count = sum(1 for slot in parsed_slots if isinstance(slot, dict))
                return (
                    f"tool_input_keys={keys}, slots_string_parsed={len(parsed_slots)}, "
                    f"dict_slots={dict_count}"
                )
        if isinstance(slots, list):
            dict_count = sum(1 for slot in slots if isinstance(slot, dict))
            return f"tool_input_keys={keys}, slots={len(slots)}, dict_slots={dict_count}"
        return f"tool_input_keys={keys}, slots_type={type(slots).__name__}"
    return None
