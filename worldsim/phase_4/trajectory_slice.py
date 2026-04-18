"""Compact `history.json` into a slice small enough for a single-turn API call.

Raw Browser-Use trajectories commonly run 700KB+ per task. A single-turn API
call has finite context budget; more importantly, most of the trajectory is
irrelevant to classifying a refusal. We window backwards from the final
agent decision and cap the output at ~30KB of text.

Edge cases handled:
- Steps with no `model_output` (failed parse / recovery) are skipped.
- `thinking` blocks > 2KB are truncated with an ellipsis marker.
- Multi-refusal trajectories (agent refuses at step 3, retries at step 12,
  refuses again at step 17) — we keep the last N steps which covers the
  most recent decision; earlier context is sacrificed under budget.
- Missing or unreadable `history.json` raises a clear error rather than
  silently passing an empty slice.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Per-field caps (characters, not bytes). Conservative so an adversarially
# large single field doesn't eat the whole budget.
_THINKING_CAP = 2000
_TEXT_FIELD_CAP = 800
_STEP_BYTE_BUDGET = 4096

# Total output cap when serialized. Input budget for a judge call then fits
# comfortably inside Sonnet 4.6's 200K context even with the task JSON.
_TOTAL_BYTE_BUDGET = 30_000

# Keywords that indicate a step is a refusal / ignore event. Used to force
# the EARLIEST such step into the slice even when the backwards-from-end
# budget would otherwise evict it — multi-refusal trajectories can have the
# root cause at step 3 and a symptomatic retry-ladder at step 17, and the
# judge should classify on the root. Matched case-insensitively against
# `thinking`, `evaluation_previous_goal`, `memory`, `next_goal`.
_REFUSAL_KEYWORDS: tuple[str, ...] = (
    "refus",
    "declin",
    "won't comply",
    "will not comply",
    "not going to",
    "cannot help",
    "can't help",
    "i won't",
    "ignor",
    "suspicious",
    "untrusted",
    "injected",
    "prompt injection",
    "safety",
    "not follow",
)


def _has_refusal_marker(step: dict[str, Any]) -> bool:
    parts: list[str] = []
    for key in ("thinking", "evaluation_previous_goal", "memory", "next_goal"):
        value = step.get(key)
        if isinstance(value, str):
            parts.append(value)
    haystack = " ".join(parts).lower()
    return any(kw in haystack for kw in _REFUSAL_KEYWORDS)


def _truncate(s: str | None, cap: int) -> str:
    if not s:
        return ""
    if len(s) <= cap:
        return s
    return s[: cap - 12] + "…[truncated]"


def _compact_step(index: int, raw: dict[str, Any]) -> dict[str, Any] | None:
    """Return a minimal dict for one trajectory step, or None to skip."""
    model_output = raw.get("model_output")
    if not isinstance(model_output, dict):
        # Failed-parse / recovery steps have no model_output. Paper-level
        # diagnosis doesn't need them; skip.
        return None
    state = raw.get("state") or {}
    result_items = raw.get("result") or []
    result_strs: list[str] = []
    for r in result_items:
        if isinstance(r, dict):
            result_strs.append(
                _truncate(r.get("extracted_content") or r.get("content") or str(r), _TEXT_FIELD_CAP)
            )
        else:
            result_strs.append(_truncate(str(r), _TEXT_FIELD_CAP))

    return {
        "step": index,
        "url": state.get("url"),
        "title": state.get("title"),
        "thinking": _truncate(model_output.get("thinking"), _THINKING_CAP),
        "evaluation_previous_goal": _truncate(
            model_output.get("evaluation_previous_goal"), _TEXT_FIELD_CAP
        ),
        "memory": _truncate(model_output.get("memory"), _TEXT_FIELD_CAP),
        "next_goal": _truncate(model_output.get("next_goal"), _TEXT_FIELD_CAP),
        "action": model_output.get("action"),
        "result": result_strs or None,
    }


def slice_trajectory(trajectory_dir: Path | str) -> list[dict[str, Any]]:
    """Return a compact, budget-capped slice of `history.json` for judging.

    Raises FileNotFoundError if `history.json` is absent, ValueError if it is
    malformed.
    """
    trajectory_dir = Path(trajectory_dir)
    history_path = trajectory_dir / "history.json"
    if not history_path.exists():
        raise FileNotFoundError(f"history.json not found at {history_path}")

    try:
        raw = json.loads(history_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"history.json at {history_path} is not valid JSON: {exc}") from exc

    steps_raw: list[Any]
    if isinstance(raw, dict) and isinstance(raw.get("history"), list):
        steps_raw = raw["history"]
    elif isinstance(raw, list):
        steps_raw = raw
    else:
        raise ValueError(f"history.json at {history_path} has unexpected shape")

    compacted: list[dict[str, Any]] = []
    for i, step in enumerate(steps_raw):
        if isinstance(step, dict):
            c = _compact_step(i, step)
            if c is not None:
                compacted.append(c)

    if not compacted:
        return []

    # Backwards-from-end windowing: the final step reflects the agent's
    # conclusion; earlier steps add context up to the byte budget.
    budget = _TOTAL_BYTE_BUDGET
    window: list[dict[str, Any]] = []
    for step in reversed(compacted):
        step_bytes = len(json.dumps(step, ensure_ascii=False))
        if step_bytes > _STEP_BYTE_BUDGET:
            # Over-large individual step: further truncate the thinking/memory
            # fields to fit a rough per-step budget.
            step = dict(step)
            step["thinking"] = _truncate(step.get("thinking"), _TEXT_FIELD_CAP)
            step["memory"] = _truncate(step.get("memory"), _TEXT_FIELD_CAP // 2)
            step_bytes = len(json.dumps(step, ensure_ascii=False))
        if step_bytes > budget and window:
            break
        window.append(step)
        budget -= step_bytes
    window.reverse()

    # Preserve the earliest refusal marker. If the backwards window evicted
    # the root-cause refusal (e.g. refuse at step 3, retry-ladder at step
    # 17), force-include a heavily-truncated version at the start so the
    # judge classifies on the root, not on the retry symptom.
    window_step_ids = {step.get("step") for step in window}
    earliest_refusal: dict[str, Any] | None = None
    for step in compacted:
        if step.get("step") in window_step_ids:
            # First refusal we'd encounter is already in the window — nothing
            # to backfill.
            if _has_refusal_marker(step):
                earliest_refusal = None
                break
            continue
        if _has_refusal_marker(step):
            earliest_refusal = step
            break

    if earliest_refusal is not None:
        early = dict(earliest_refusal)
        early["thinking"] = _truncate(early.get("thinking"), _TEXT_FIELD_CAP // 2)
        early["memory"] = _truncate(early.get("memory"), _TEXT_FIELD_CAP // 2)
        early["_forced_inclusion"] = "earliest_refusal_marker"
        window.insert(0, early)

    return window
