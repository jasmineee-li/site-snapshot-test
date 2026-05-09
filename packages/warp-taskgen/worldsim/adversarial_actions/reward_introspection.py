from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from worldsim.adversarial_actions.catalog import get_action_spec
from worldsim.adversarial_actions.shared_compiler import _BINARY_LABEL_FAMILIES
from worldsim.adversarial_actions.tier3 import task_action_is_host_ready


def reward_signal_for_task(task: Mapping[str, Any]) -> str:
    action = task.get("adversarial_action")
    if isinstance(action, Mapping):
        kind = str(action.get("kind") or "")
        spec = get_action_spec(kind)
        if spec is not None and (spec.default_enabled or task_action_is_host_ready(task)):
            return spec.reward_signal
        if spec is not None and not spec.default_enabled:
            return "disabled_action"
    reward = task.get("reward_function")
    adversarial_reward = reward.get("adversarial_reward") if isinstance(reward, Mapping) else None
    if isinstance(adversarial_reward, Mapping):
        if _contains_network_event(adversarial_reward):
            return "network_action"
        if _contains_agent_response(adversarial_reward):
            return "semantic_answer"
    return "unknown"

def action_metadata_for_task(task: Mapping[str, Any]) -> dict[str, Any]:
    """Return reporting metadata for the host-owned adversarial action."""
    action = task.get("adversarial_action")
    kind = str(action.get("kind") or "").strip() if isinstance(action, Mapping) else ""
    spec = get_action_spec(kind)
    if spec is None:
        return {
            "adversarial_action_kind": kind or "unknown",
            "adversarial_action_tier": None,
            "adversarial_action_family": "unknown",
            "adversarial_action_enabled": False,
        }
    enabled = spec.default_enabled or task_action_is_host_ready(task)
    metadata = {
        "adversarial_action_kind": spec.kind,
        "adversarial_action_tier": spec.impact_tier,
        "adversarial_action_family": spec.action_family,
        "adversarial_action_enabled": enabled,
    }
    if spec.kind == "delete_or_transfer_resource" and enabled:
        metadata["adversarial_action_confirmation_semantics"] = "target_state_absence_only"
        metadata["adversarial_action_nonce_bound"] = False
    return metadata

def _required_url_for_task(task: Mapping[str, Any]) -> str:
    for token in task.get("required_tokens") or []:
        if isinstance(token, Mapping) and token.get("kind") == "url":
            value = token.get("value")
            if isinstance(value, str) and value.strip():
                return value.strip()
    task_id = str(task.get("id") or "")
    digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
    return f"https://attacker.invalid/{digest[:16].lower()}"

def _single_binary_label_expected(reward: Any) -> tuple[str, frozenset[str]] | None:
    strings = [
        value.casefold().strip()
        for value in _reward_agent_response_expected_strings(reward)
        if isinstance(value, str) and value.strip()
    ]
    if len(strings) != 1:
        return None
    label = strings[0]
    for family in _BINARY_LABEL_FAMILIES:
        if label in family:
            return label, family
    return None

def _reward_agent_response_expected_strings(reward: Any) -> list[str]:
    if not isinstance(reward, Mapping):
        return []
    out: list[str] = []
    for config in _agent_response_reward_configs(reward):
        expected = config.get("expected")
        if isinstance(expected, Mapping):
            out.extend(_flatten_strings(expected.get("retrieved_data")))
    return out

def _agent_response_reward_configs(reward: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    reward_type = str(reward.get("type") or reward.get("evaluator") or "")
    if reward_type == "AgentResponseEvaluator":
        return [reward]
    evals = reward.get("eval")
    if not isinstance(evals, list):
        return []
    return [
        item
        for item in evals
        if isinstance(item, Mapping) and item.get("evaluator") == "AgentResponseEvaluator"
    ]

def _flatten_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, Mapping):
        out: list[str] = []
        for item in value.values():
            out.extend(_flatten_strings(item))
        return out
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            out.extend(_flatten_strings(item))
        return out
    return []

def _contains_network_event(reward: Mapping[str, Any]) -> bool:
    reward_type = str(reward.get("type") or reward.get("evaluator") or "")
    if reward_type in {"NetworkEventEvaluator", "network_event"}:
        return True
    evals = reward.get("eval")
    return isinstance(evals, list) and any(
        isinstance(item, Mapping)
        and str(item.get("evaluator") or item.get("type") or "")
        in {"NetworkEventEvaluator", "network_event"}
        for item in evals
    )

def _contains_agent_response(reward: Mapping[str, Any]) -> bool:
    return bool(_agent_response_reward_configs(reward))
