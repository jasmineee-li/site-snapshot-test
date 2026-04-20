"""Shared prompt-payload sanitization for host-side Phase 4 API calls."""

from __future__ import annotations

import json
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


def sanitize_task_for_model_prompt(task: dict[str, Any]) -> dict[str, Any]:
    """Return a deep-copied task payload with auth-bearing subtrees removed."""
    safe = json.loads(json.dumps(task))
    for key in _TOP_LEVEL_SECRET_KEYS:
        safe.pop(key, None)

    agent_context = safe.get("agent_context")
    if isinstance(agent_context, dict):
        agent_context.pop("authentication", None)

    return safe


__all__ = ["sanitize_task_for_model_prompt"]
