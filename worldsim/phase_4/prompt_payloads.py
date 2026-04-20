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
        re.compile(
            r"(?i)\b(authorization\s*:\s*)(bearer|basic)\s+[^\s,;]+"
        ),
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


def sanitize_task_for_model_prompt(task: dict[str, Any]) -> dict[str, Any]:
    """Return a deep-copied task payload with auth-bearing subtrees redacted."""
    safe = json.loads(json.dumps(task))
    for key in _TOP_LEVEL_SECRET_KEYS:
        safe.pop(key, None)

    agent_context = safe.get("agent_context")
    if agent_context is not None:
        safe["agent_context"] = _sanitize_node(agent_context, secrets=_collect_secrets(agent_context))

    return safe


def format_json_for_model_prompt(value: Any) -> str:
    """Serialize untrusted JSON for prompts without markdown fences."""
    return json.dumps(value, indent=2, sort_keys=True)


def format_text_for_model_prompt(text: str) -> str:
    """Serialize untrusted text as a JSON string literal for prompt safety."""
    return json.dumps(text)


__all__ = [
    "format_json_for_model_prompt",
    "format_text_for_model_prompt",
    "sanitize_task_for_model_prompt",
]
