"""Untruncated trajectory extractor for Phase-4 judges and the AER formatter.

Replaces the legacy ``trajectory_slice`` module. Differences:

- No byte budget, no field truncation. Sonnet 4.6's 1M-context variant has
  two orders of magnitude more headroom than the old 30KB slicer reserved.
- Typed dataclasses (``ExtractedStep``, ``ExtractedTrajectory``) rather
  than raw dicts, so both the refusal judge and the AER formatter see the
  same projection contract.
- ``include_state_message=False`` default. ``state_message`` is
  Browser-Use's serialised browser-state dump and is 72-78% of every
  trajectory's bytes in real logs; including it inflates judge input
  ~27x. Callers that genuinely need page content opt in.
- Partial-envelope detection. When Browser-Use crashes before writing
  history, it emits ``{"history": [], "partial": True, "errors": [...]}``.
  That is structurally distinct from a successfully-empty trajectory and
  the refusal judge routes it to ``failure_class="partial_trajectory"``.
- ``as_judge_view(traj)`` returns the same key-for-key dict shape the old
  slicer produced (minus the ``_forced_inclusion`` marker, which the
  force-include-refusal heuristic is no longer needed to synthesise).
- ``as_aer_view(traj)`` is the formatter-facing projection — includes
  partial steps so per-step enumeration matches Browser-Use's own step
  indices.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Safety ceiling — warn when a trajectory serialises past this size. Set
# generously under Sonnet 4.6's 1M-context limit so any realistic trajectory
# (20 steps typical, ~24KB; 40 steps ~52KB) sits well below.
_SIZE_WARN_BYTES = 800_000 * 4  # ~800K tokens ≈ 3.2MB of text


@dataclass(frozen=True)
class ExtractedStep:
    step: int
    url: str | None = None
    title: str | None = None
    thinking: str | None = None
    evaluation_previous_goal: str | None = None
    memory: str | None = None
    next_goal: str | None = None
    action: list[dict[str, Any]] = field(default_factory=list)
    result: list[str] | None = None
    partial: bool = False
    partial_reason: str | None = None
    state_message: str | None = None
    screenshot_path: str | None = None


@dataclass(frozen=True)
class ExtractedTrajectory:
    steps: list[ExtractedStep]
    partial: bool = False
    agent_errors: list[str] = field(default_factory=list)
    raw_byte_count: int = 0
    decode_warnings: int = 0


def extract_task_intent(task: dict[str, Any]) -> str:
    """Return the task's user-facing instruction string.

    Both Transcript Purpose and VEA need the same task-intent text so their
    classifications see identical context. Tasks normalise through several
    schemas depending on phase and benchmark — Phase 4 tasks carry the
    live agent prompt in ``instruction``, Phase-3 contracts emit
    ``description``, WebArena legacy tasks may use ``intent``, and some
    adapter layers emit ``task_instruction`` / ``task_description`` /
    ``task_intent``. Try them all in a single order so the two
    observational metrics don't silently diverge on the same trajectory.
    """
    for key in (
        "instruction",
        "task_instruction",
        "description",
        "task_description",
        "intent",
        "task_intent",
    ):
        value = task.get(key)
        if value:
            return str(value)
    return ""


def _coerce_text(value: Any) -> str | None:
    """Return a string unchanged, ``None`` unchanged, anything else as ``None``.

    Browser-Use sometimes serialises malformed tool outputs into non-string
    fields (dicts when the model emitted structured JSON inside a ``thinking``
    block, ints on parsing errors). We coerce to None rather than str(value)
    so callers can distinguish "field absent" from "field was present with
    this exact text".
    """
    if value is None or isinstance(value, str):
        return value
    return None


def _coerce_action(value: Any) -> list[dict[str, Any]]:
    """Normalise action to a list of dicts.

    Browser-Use's action field is a list of single-key dicts
    (``[{"click_element_by_index": {"index": 5}}]``) but older logs and
    error-path fallbacks sometimes emit a single dict or ``None``. Callers
    expect ``list[dict]`` so both shapes must normalise.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [entry for entry in value if isinstance(entry, dict)]
    if isinstance(value, dict):
        return [value]
    return []


def _compact_result(result_items: Any) -> list[str] | None:
    """Project Browser-Use's per-step ``result`` array down to a list of strings.

    Returns ``None`` (rather than ``[]``) when the step produced no result
    content. Keeps the old slicer's contract so ``as_judge_view`` stays
    byte-identical for downstream prompt cache stability.
    """
    if not isinstance(result_items, list):
        return None
    out: list[str] = []
    for r in result_items:
        if isinstance(r, dict):
            text = r.get("extracted_content") or r.get("content")
            if isinstance(text, str) and text:
                out.append(text)
            else:
                out.append(str(r))
        elif isinstance(r, str):
            out.append(r)
        else:
            out.append(str(r))
    return out or None


def _extract_step(
    index: int,
    raw: Any,
    *,
    include_state_message: bool,
    include_screenshots: bool,
    decode_counter: list[int],
) -> ExtractedStep:
    """Build an ``ExtractedStep`` from one raw history entry.

    Returns a partial step (``partial=True``) rather than ``None`` when
    ``model_output`` is missing or malformed. Preserving the step keeps the
    enumeration index aligned with Browser-Use's own step numbering, which
    matters for the AER formatter (where per-step blocks are printed in
    order) and for any audit tooling that cross-references step indices
    between ``history.json`` and ``screenshots/step_{i}.png``.
    """
    if not isinstance(raw, dict):
        decode_counter[0] += 1
        return ExtractedStep(step=index, partial=True, partial_reason="non_dict_step")

    model_output = raw.get("model_output")
    if not isinstance(model_output, dict):
        return ExtractedStep(step=index, partial=True, partial_reason="no_model_output")

    state = raw.get("state")
    if not isinstance(state, dict):
        state = {}

    thinking = _coerce_text(model_output.get("thinking"))
    if thinking is None and model_output.get("thinking") is not None:
        decode_counter[0] += 1

    eval_prev = _coerce_text(model_output.get("evaluation_previous_goal"))
    memory = _coerce_text(model_output.get("memory"))
    next_goal = _coerce_text(model_output.get("next_goal"))
    action = _coerce_action(model_output.get("action"))
    result = _compact_result(raw.get("result"))

    state_message = None
    if include_state_message:
        sm = raw.get("state_message")
        if isinstance(sm, str):
            state_message = sm
        elif sm is not None:
            state_message = json.dumps(sm, default=str)

    screenshot_path = None
    if include_screenshots:
        sp = state.get("screenshot_path")
        if isinstance(sp, str) and sp:
            screenshot_path = sp

    return ExtractedStep(
        step=index,
        url=_coerce_text(state.get("url")),
        title=_coerce_text(state.get("title")),
        thinking=thinking,
        evaluation_previous_goal=eval_prev,
        memory=memory,
        next_goal=next_goal,
        action=action,
        result=result,
        state_message=state_message,
        screenshot_path=screenshot_path,
    )


def extract_trajectory(
    trajectory_dir: Path | str,
    *,
    include_state_message: bool = False,
    include_screenshots: bool = False,
) -> ExtractedTrajectory:
    """Load ``history.json`` into typed dataclasses, untruncated.

    Raises:
        FileNotFoundError: ``history.json`` is absent.
        ValueError: file is malformed JSON or has an unexpected shape.

    Never silently returns an empty trajectory when a file exists — callers
    that want to distinguish "empty run" from "crashed before writing"
    inspect ``traj.partial`` and ``traj.agent_errors``.
    """
    trajectory_dir = Path(trajectory_dir)
    history_path = trajectory_dir / "history.json"
    if not history_path.exists():
        raise FileNotFoundError(f"history.json not found at {history_path}")

    # Read with errors="replace" so trajectories containing control characters
    # or non-UTF-8 bytes in model outputs don't crash the extractor; count
    # replacements via a separate stat pass so the counter is accurate.
    raw_bytes = history_path.read_bytes()
    try:
        raw_text = raw_bytes.decode("utf-8")
        decode_warnings = 0
    except UnicodeDecodeError:
        raw_text = raw_bytes.decode("utf-8", errors="replace")
        decode_warnings = raw_text.count("\ufffd")

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"history.json at {history_path} is not valid JSON: {exc}") from exc

    raw_byte_count = len(raw_bytes)
    if raw_byte_count > _SIZE_WARN_BYTES:
        warnings.warn(
            f"history.json at {history_path} is {raw_byte_count:,} bytes "
            f"(warning ceiling {_SIZE_WARN_BYTES:,}). The judge call may approach "
            "Sonnet's 1M-token context limit. Consider trimming the trajectory.",
            stacklevel=2,
        )

    partial = False
    agent_errors: list[str] = []
    steps_raw: list[Any]

    if isinstance(payload, dict):
        if isinstance(payload.get("history"), list):
            steps_raw = payload["history"]
        else:
            raise ValueError(
                f"history.json at {history_path} has unexpected shape (no 'history' list)"
            )
        partial = bool(payload.get("partial", False))
        errs = payload.get("errors")
        if isinstance(errs, list):
            agent_errors = [str(e) for e in errs]
    elif isinstance(payload, list):
        steps_raw = payload
    else:
        raise ValueError(f"history.json at {history_path} has unexpected shape (not dict or list)")

    decode_counter = [decode_warnings]
    steps: list[ExtractedStep] = []
    for i, entry in enumerate(steps_raw):
        steps.append(
            _extract_step(
                i,
                entry,
                include_state_message=include_state_message,
                include_screenshots=include_screenshots,
                decode_counter=decode_counter,
            )
        )

    return ExtractedTrajectory(
        steps=steps,
        partial=partial,
        agent_errors=agent_errors,
        raw_byte_count=raw_byte_count,
        decode_warnings=decode_counter[0],
    )


def as_judge_view(traj: ExtractedTrajectory) -> list[dict[str, Any]]:
    """Project to the legacy slicer's dict shape for the refusal judge.

    Byte-identical to ``slice_trajectory``'s output when truncation is
    disabled, minus the ``_forced_inclusion`` synthetic marker (the
    force-include-refusal heuristic is no longer needed — the full
    trajectory fits comfortably and the judge sees every refusal step
    directly).

    Partial steps are filtered out to preserve the legacy contract: the
    refusal judge's prompt describes each step as having a model_output,
    so passing partial placeholders would surface as prompt noise.
    """
    out: list[dict[str, Any]] = []
    for s in traj.steps:
        if s.partial:
            continue
        out.append(
            {
                "step": s.step,
                "url": s.url,
                "title": s.title,
                "thinking": s.thinking or "",
                "evaluation_previous_goal": s.evaluation_previous_goal or "",
                "memory": s.memory or "",
                "next_goal": s.next_goal or "",
                "action": s.action if s.action else None,
                "result": s.result,
            }
        )
    return out


def as_aer_view(traj: ExtractedTrajectory) -> list[dict[str, Any]]:
    """Project to the AER formatter's consumption shape.

    Includes partial steps so the per-step enumeration printed in the AER
    transcript matches Browser-Use's own step indices. Retains the full
    set of Browser-Use native fields (thinking + evaluation_previous_goal
    + memory + next_goal + action) — the AER formatter joins them verbatim
    per methodological consistency with OS-Harm.
    """
    out: list[dict[str, Any]] = []
    for s in traj.steps:
        entry = {
            "step": s.step,
            "url": s.url,
            "title": s.title,
            "thinking": s.thinking,
            "evaluation_previous_goal": s.evaluation_previous_goal,
            "memory": s.memory,
            "next_goal": s.next_goal,
            "action": s.action,
            "result": s.result,
            "partial": s.partial,
        }
        if s.partial_reason:
            entry["partial_reason"] = s.partial_reason
        if s.state_message is not None:
            entry["state_message"] = s.state_message
        if s.screenshot_path is not None:
            entry["screenshot_path"] = s.screenshot_path
        out.append(entry)
    return out


__all__ = [
    "ExtractedStep",
    "ExtractedTrajectory",
    "as_aer_view",
    "as_judge_view",
    "extract_trajectory",
]
