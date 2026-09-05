"""Reward-function validation exports for generated tasks."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from warp_taskgen.phase_1.novel_task_validation.task_cards import _task_uses_host_action_only_card

_HOST_ACTION_ONLY_PLACEHOLDER_EVALUATOR = "HostActionOnlyPlaceholder"


def _validate_eval_expected(
    config: dict[str, Any],
    *,
    evaluator: str,
    prefix: str,
    eval_index: int,
) -> str | None:
    """Reject evaluator configs that would degenerate into near-no-op checks."""
    expected = config.get("expected")
    if not isinstance(expected, dict) or not expected:
        return f"{prefix} eval[{eval_index}] must include a non-empty expected object"

    if evaluator == "NetworkEventEvaluator":
        url = expected.get("url")
        urls = url if isinstance(url, list) else [url]
        if not urls or not all(isinstance(item, str) and item.strip() for item in urls):
            return f"{prefix} eval[{eval_index}] NetworkEventEvaluator.expected.url must be a non-empty string or non-empty array of strings"
        http_method = expected.get("http_method")
        if http_method is not None and (
            not isinstance(http_method, str) or not http_method.strip()
        ):
            return f"{prefix} eval[{eval_index}] NetworkEventEvaluator.expected.http_method must be a non-empty string when provided"
        response_status = expected.get("response_status")
        if response_status is not None and not _valid_network_status_expected(response_status):
            return f"{prefix} eval[{eval_index}] NetworkEventEvaluator.expected.response_status must be an integer, status-class string, list, or min/max object"
        post_data = expected.get("post_data")
        if post_data is not None and not isinstance(post_data, dict):
            return f"{prefix} eval[{eval_index}] NetworkEventEvaluator.expected.post_data must be an object when provided"
        post_data_contains = expected.get("post_data_contains")
        if post_data_contains is not None and not (
            isinstance(post_data_contains, str)
            or (
                isinstance(post_data_contains, list)
                and bool(post_data_contains)
                and all(isinstance(item, str) and item for item in post_data_contains)
            )
        ):
            return f"{prefix} eval[{eval_index}] NetworkEventEvaluator.expected.post_data_contains must be a string or non-empty string array when provided"
        method_requirements = expected.get("method_requirements")
        if method_requirements is not None:
            error = _validate_network_method_requirements(
                method_requirements, prefix=prefix, eval_index=eval_index
            )
            if error:
                return error
        return None

    if evaluator == "AgentResponseEvaluator":
        results_schema = config.get("results_schema")
        if not isinstance(results_schema, dict) or not results_schema.get("type"):
            return (
                f"{prefix} eval[{eval_index}] AgentResponseEvaluator must include "
                "a results_schema with a type"
            )
        if not any(key in expected for key in ("task_type", "status", "retrieved_data")):
            return (
                f"{prefix} eval[{eval_index}] AgentResponseEvaluator.expected must include "
                "at least one of task_type, status, or retrieved_data"
            )
        task_type = str(expected.get("task_type") or "").strip().lower()
        status = str(expected.get("status") or "").strip().upper()
        if task_type == "retrieve" and status != "NOT_FOUND_ERROR":
            retrieved = expected.get("retrieved_data")
            if not isinstance(retrieved, list) or not retrieved:
                return (
                    f"{prefix} eval[{eval_index}] AgentResponseEvaluator retrieve "
                    "tasks must include non-empty expected.retrieved_data"
                )
            if results_schema.get("type") != "array":
                return (
                    f"{prefix} eval[{eval_index}] AgentResponseEvaluator retrieve "
                    "tasks must use an array results_schema for retrieved_data"
                )
        return None

    if evaluator == "FinalStateEvaluator":
        action_kind = str(expected.get("action_kind") or "").strip()
        contains = str(expected.get("contains") or "").strip()
        network_event = expected.get("network_event")
        state_probe = expected.get("state_probe")
        if not action_kind:
            return (
                f"{prefix} eval[{eval_index}] FinalStateEvaluator.expected.action_kind "
                "must be a non-empty string"
            )
        if not contains:
            return (
                f"{prefix} eval[{eval_index}] FinalStateEvaluator.expected.contains "
                "must be a non-empty string"
            )
        if not isinstance(network_event, dict) or not network_event:
            return (
                f"{prefix} eval[{eval_index}] FinalStateEvaluator.expected.network_event "
                "must be a non-empty object"
            )
        if not isinstance(state_probe, dict) or not state_probe:
            return (
                f"{prefix} eval[{eval_index}] FinalStateEvaluator.expected.state_probe "
                "must be a non-empty object"
            )
        return None

    return None


def _is_host_compiled_final_state_evaluator(
    task: Mapping[str, Any],
    config: Mapping[str, Any],
) -> bool:
    """Return whether ``config`` is a host-owned benign action readback reward.

    Phase 1 model outputs must not author arbitrary final-state probes. The
    only accepted final-state evaluator at this stage is one compiled by the
    host from a task-card benign action contract and recorded in provenance.
    """
    if config.get("evaluator") != "FinalStateEvaluator":
        return False
    provenance = task.get("task_provenance")
    if not isinstance(provenance, Mapping):
        return False
    contract = provenance.get("benign_action_contract")
    if not isinstance(contract, Mapping):
        return False
    if contract.get("reward_signal") != "final_state_action":
        return False
    if contract.get("schema_version") != "worldsim-benign-action-contract-v1":
        return False
    witness = str(contract.get("witness") or "").strip()
    if not witness:
        return False
    expected = config.get("expected")
    if not isinstance(expected, Mapping):
        return False
    compiled = contract.get("evaluator")
    if not isinstance(compiled, Mapping):
        return False
    return str(expected.get("contains") or "").strip() == witness and json.loads(
        json.dumps(dict(config), sort_keys=True)
    ) == json.loads(json.dumps(dict(compiled), sort_keys=True))


def _is_host_action_only_placeholder_evaluator(
    task: Mapping[str, Any],
    cards: dict[str, dict[str, Any]] | None,
    config: Mapping[str, Any],
) -> bool:
    """Allow action-only generation to omit response-evaluator design.

    The placeholder is accepted only before task-card alignment finalizes the
    host-owned benign action contract. It must never survive validation output:
    ``finalize_host_action_only_benign_reward`` replaces the full reward with
    the compiled request/readback evaluator.
    """
    if config.get("evaluator") != _HOST_ACTION_ONLY_PLACEHOLDER_EVALUATOR:
        return False
    if not _task_uses_host_action_only_card(task, cards):
        return False
    expected = config.get("expected")
    return isinstance(expected, Mapping) and expected.get("host_compiled") is True


def _validate_network_method_requirements(
    value: object,
    *,
    prefix: str,
    eval_index: int,
) -> str | None:
    if not isinstance(value, dict) or not value:
        return f"{prefix} eval[{eval_index}] NetworkEventEvaluator.expected.method_requirements must be a non-empty object when provided"
    for method, requirement in value.items():
        method_text = str(method).strip().upper()
        if not method_text:
            return f"{prefix} eval[{eval_index}] NetworkEventEvaluator.expected.method_requirements keys must be non-empty HTTP methods"
        if not isinstance(requirement, dict) or not requirement:
            return f"{prefix} eval[{eval_index}] NetworkEventEvaluator.expected.method_requirements.{method_text} must be a non-empty object"
        unsupported = set(requirement) - {"post_data", "post_data_contains"}
        if unsupported:
            return (
                f"{prefix} eval[{eval_index}] NetworkEventEvaluator.expected."
                f"method_requirements.{method_text} uses unsupported fields: "
                + ", ".join(sorted(str(key) for key in unsupported))
            )
        post_data = requirement.get("post_data")
        if post_data is not None and not isinstance(post_data, dict):
            return f"{prefix} eval[{eval_index}] NetworkEventEvaluator.expected.method_requirements.{method_text}.post_data must be an object when provided"
        post_data_contains = requirement.get("post_data_contains")
        if post_data_contains is not None and not (
            isinstance(post_data_contains, str)
            or (
                isinstance(post_data_contains, list)
                and bool(post_data_contains)
                and all(isinstance(item, str) and item for item in post_data_contains)
            )
        ):
            return f"{prefix} eval[{eval_index}] NetworkEventEvaluator.expected.method_requirements.{method_text}.post_data_contains must be a string or non-empty string array when provided"
    return None


def _valid_network_status_expected(value: Any) -> bool:
    if isinstance(value, int):
        return True
    if isinstance(value, str):
        stripped = value.strip().lower()
        return stripped.isdigit() or bool(re.fullmatch(r"\dxx", stripped))
    if isinstance(value, list):
        return bool(value) and all(_valid_network_status_expected(item) for item in value)
    if isinstance(value, dict):
        minimum = value.get("min")
        maximum = value.get("max")
        return (
            (minimum is None or isinstance(minimum, int))
            and (maximum is None or isinstance(maximum, int))
            and (minimum is not None or maximum is not None)
        )
    return False


__all__ = [
    "_is_host_action_only_placeholder_evaluator",
    "_is_host_compiled_final_state_evaluator",
    "_valid_network_status_expected",
    "_validate_eval_expected",
    "_validate_network_method_requirements",
]
