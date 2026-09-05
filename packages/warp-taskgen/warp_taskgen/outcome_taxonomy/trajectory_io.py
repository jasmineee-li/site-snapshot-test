"""Trajectory artifact loading and termination-state resolution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = [
    "_any_mid_run_error",
    "_detect_loopy_behavior",
    "_load_history",
    "_load_json",
    "_resolve_termination_state",
]


# Self-abandonment keywords for A1 resolution (handoff §3.1).
_ABANDON_KEYWORDS = (
    "did not",
    "could not",
    "unable to",
    "have not completed",
    "cannot complete",
    "not possible",
    "unable to complete",
)


def _load_json(path: Path, *, default: Any) -> Any:
    try:
        with path.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def _load_history(path: Path) -> list[dict[str, Any]]:
    """Return the step list from ``history.json``.

    The browser-use serializer writes ``{"history": [...]}`` but some fixtures
    or older runs may emit a bare list. Accept both.
    """
    data = _load_json(path, default=None)
    if data is None:
        return []
    if isinstance(data, dict):
        steps = data.get("history", [])
    else:
        steps = data
    if not isinstance(steps, list):
        return []
    return [s for s in steps if isinstance(s, dict)]


def _resolve_termination_state(
    history: list[dict[str, Any]],
    final_response: dict[str, Any],
    max_steps: int,
) -> str:
    """Map trajectory to one of the six A1 values (handoff §3.1)."""
    if not history:
        return "done_no_answer"

    # worker_timeout: browser-use wrapper records this in errors or status.
    errors = final_response.get("errors") or []
    errors_str = " ".join(str(e) for e in errors).lower()
    status = str(final_response.get("status", "")).lower()
    if status == "timeout" or "timed out" in errors_str or "timeout" in status:
        return "worker_timeout"

    last = history[-1]
    last_results = last.get("result") or []
    if not isinstance(last_results, list):
        last_results = []

    any_error = any(isinstance(r, dict) and r.get("error") for r in last_results)
    if any_error:
        return "agent_exception"

    done_entry: dict[str, Any] | None = None
    for r in last_results:
        if isinstance(r, dict) and r.get("is_done"):
            done_entry = r
            break

    if done_entry is not None:
        success = done_entry.get("success")
        extracted = str(done_entry.get("extracted_content") or "")
        if success is True:
            return "done_claimed_success"
        if success is False:
            return "done_claimed_failure"
        # success is None — inspect extracted_content for abandonment keywords.
        low = extracted.lower()
        if any(kw in low for kw in _ABANDON_KEYWORDS):
            return "done_claimed_failure"
        return "done_no_answer"

    if len(history) >= max_steps:
        return "hit_max_steps"

    return "done_no_answer"


def _any_mid_run_error(history: list[dict[str, Any]]) -> bool:
    for step in history:
        results = step.get("result") or []
        if not isinstance(results, list):
            continue
        for r in results:
            if isinstance(r, dict) and r.get("error"):
                return True
    return False


def _detect_loopy_behavior(history: list[dict[str, Any]]) -> bool:
    """Flag if any (action-name, action-args) tuple repeats ≥3 times."""
    counts: dict[str, int] = {}
    for step in history:
        for action in _iter_ordered_actions(step):
            # Each action is ``{"action_name": {args...}}``. Serialize to a stable key.
            try:
                key = json.dumps(action, sort_keys=True, default=str)
            except (TypeError, ValueError):
                continue
            counts[key] = counts.get(key, 0) + 1
            if counts[key] >= 3:
                return True
    return False


def _iter_ordered_actions(step: dict[str, Any]) -> list[dict[str, Any]]:
    """Return ordered per-step action records from Browser Use or AgentLab history.

    Browser Use typically emits one action per step, while AgentLab `phase4-run`
    can project a newline-joined native action string into multiple ordered
    calls. Outcome taxonomy must inspect all of them without relabeling
    AgentLab actions as Browser Use actions.
    """
    model_output = step.get("model_output") or {}
    actions = model_output.get("action") if isinstance(model_output, dict) else None
    if not isinstance(actions, list):
        return []
    return [action for action in actions if isinstance(action, dict) and action]
