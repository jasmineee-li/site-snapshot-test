"""Shared prompt-payload sanitization/formatting for host-side Phase 4 API calls."""

from __future__ import annotations

import json
import re
from typing import Any

_TOP_LEVEL_SECRET_KEYS: frozenset[str] = frozenset(
    {
        "authentication",
        "auth_mechanism",
        "agent_auth",
        "api_auth",
        "bound_instances",
        "instance",
        "instances",
    }
)
_PROMPT_PRIVATE_KEYS: frozenset[str] = frozenset(
    {
        "task_provenance",
        "precondition_slot",
        "task_card",
        "task_card_id",
        "task_archetype",
        "archetype_id",
        "task_signature",
        "archetype_signature",
        "task_bank",
        "task_bank_metadata",
        "private_fields",
        "source_jsonl_line",
        "source_record",
        "generation_diagnostics",
    }
)
_HOST_CONTRACT_KEYS: frozenset[str] = frozenset(
    {
        "exposure_contract",
        "reward_function",
        "adversarial_final_state_check",
        "feasibility",
    }
)
_RECURSIVE_PROMPT_PRIVATE_KEYS: frozenset[str] = frozenset(
    {
        "target_context",
        "evidence_rule",
        "forbidden_exact_fields",
    }
)

_RECURSIVE_SECRET_KEYS: frozenset[str] = frozenset(
    {
        "authentication",
        "auth_mechanism",
        "agent_auth",
        "api_auth",
        "headers",
        "credentials",
        "storage_state",
        "form_login",
    }
)
_EMBEDDED_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"(?i)\b(authorization\s*:\s*)(bearer|basic)\s+[^\s,;]+"),
        r"\1<redacted>",
    ),
    (
        re.compile(r"(?i)\b([A-Za-z0-9._%+-]+:[^\s,;]+)"),
        "<redacted>",
    ),
)


def _collect_secrets(value: Any) -> set[str]:
    secrets: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, item in node.items():
                lowered = str(key).lower()
                if lowered in _RECURSIVE_SECRET_KEYS and isinstance(item, dict):
                    for inner in item.values():
                        if isinstance(inner, str) and inner:
                            secrets.add(inner)
                elif any(token in lowered for token in ("password", "token", "secret", "api_key")):
                    if isinstance(item, str) and item:
                        secrets.add(item)
                walk(item)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(value)
    return secrets


def _sanitize_node(value: Any, *, secrets: set[str]) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            lowered = key_str.lower()
            if lowered in _RECURSIVE_SECRET_KEYS:
                if isinstance(item, dict):
                    sanitized[key_str] = {inner_key: "<redacted>" for inner_key in item}
                else:
                    sanitized[key_str] = "<redacted>"
                continue
            if any(token in lowered for token in ("password", "token", "secret", "api_key")):
                sanitized[key_str] = "<redacted>"
                continue
            sanitized[key_str] = _sanitize_node(item, secrets=secrets)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_node(item, secrets=secrets) for item in value]
    if isinstance(value, str):
        redacted = value
        for secret in sorted(secrets, key=len, reverse=True):
            if secret:
                redacted = redacted.replace(secret, "<redacted>")
        for pattern, replacement in _EMBEDDED_SECRET_PATTERNS:
            redacted = pattern.sub(replacement, redacted)
        return redacted
    return value


def _drop_prompt_private_nodes(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _drop_prompt_private_nodes(item)
            for key, item in value.items()
            if str(key) not in _RECURSIVE_PROMPT_PRIVATE_KEYS
        }
    if isinstance(value, list):
        return [_drop_prompt_private_nodes(item) for item in value]
    return value


def sanitize_task_for_model_prompt(
    task: dict[str, Any],
    *,
    allow_host_contracts: bool = False,
) -> dict[str, Any]:
    """Return a deep-copied model-facing task payload.

    Host-owned provenance, slot internals, evaluator JSON, and fixture contracts
    stay out of ordinary judge/variant prompts. Placement-fix is the exceptional
    host-repair prompt that may inspect fixed host contracts, but even there raw
    provenance and slot nodes remain private.
    """
    safe = json.loads(json.dumps(task))
    for key in _TOP_LEVEL_SECRET_KEYS:
        safe.pop(key, None)
    for key in _PROMPT_PRIVATE_KEYS:
        safe.pop(key, None)
    if not allow_host_contracts:
        for key in _HOST_CONTRACT_KEYS:
            safe.pop(key, None)

    agent_context = safe.get("agent_context")
    if agent_context is not None:
        safe["agent_context"] = _sanitize_node(
            agent_context, secrets=_collect_secrets(agent_context)
        )

    return _drop_prompt_private_nodes(safe)


def sanitize_action_payload_guidance_for_model_prompt(
    guidance: dict[str, Any],
) -> dict[str, Any]:
    """Project action guidance to fields safe for variant generation prompts."""

    safe = json.loads(json.dumps(guidance))
    for key in (
        "target_context",
        "evidence_rule",
    ):
        safe.pop(key, None)
    slot = safe.get("precondition_slot")
    if isinstance(slot, dict):
        slot.pop("forbidden_exact_fields", None)
    return safe


def sanitize_model_prompt_payload(value: Any) -> Any:
    """Remove recursively private prompt keys from an already built payload."""

    return _drop_prompt_private_nodes(json.loads(json.dumps(value)))


def format_json_for_model_prompt(value: Any) -> str:
    """Serialize untrusted JSON for prompts without markdown fences."""
    return json.dumps(value, indent=2, sort_keys=True)


def format_text_for_model_prompt(text: str) -> str:
    """Serialize untrusted text as a JSON string literal for prompt safety."""
    return json.dumps(text)


__all__ = [
    "format_json_for_model_prompt",
    "format_text_for_model_prompt",
    "sanitize_action_payload_guidance_for_model_prompt",
    "sanitize_model_prompt_payload",
    "sanitize_task_for_model_prompt",
]
