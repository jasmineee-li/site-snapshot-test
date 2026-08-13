"""Bounded failure-context digest for Phase 4 variant generation.

The refusal judge may inspect a trajectory, but the variant generator should
not receive a raw transcript. It needs a compact host-owned digest: what
failed, where the injection was encountered, which judge bucket fired, and a
few observable snippets that explain the failure without exposing hidden
reasoning or letting one trajectory overfit the next payload.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from warp_taskgen.phase_4.aer_trajectory_extract import (
    ExtractedStep,
    extract_trajectory,
)
from warp_taskgen.text_payload_contract import build_text_payload_contract

FAILURE_CONTEXT_SCHEMA_VERSION = "phase4_failure_context_v1"

_MAX_TEXT_CHARS = 260
_MAX_FINAL_ANSWER_CHARS = 500
_MAX_TRACE_SNIPPETS = 4


def _clip_text(value: Any, limit: int = _MAX_TEXT_CHARS) -> str:
    text = str(value or "").strip()
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 1)].rstrip()}..."


def _path_ref(path: Path, *, exists: bool = True) -> str | None:
    if exists and not path.exists():
        return None
    return str(path)


def _result_projection(initial_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "outcome": initial_result.get("outcome"),
        "final_status": initial_result.get("final_status"),
        "benign_passed": initial_result.get("benign_passed"),
        "adversarial_passed": initial_result.get("adversarial_passed"),
        "outcome_fine": initial_result.get("outcome_fine"),
        "steps": initial_result.get("steps"),
        "elapsed": initial_result.get("elapsed"),
    }


def _pvpo_projection(initial_result: dict[str, Any]) -> dict[str, Any]:
    encounter = initial_result.get("encounter")
    max_coverage = None
    if isinstance(encounter, dict):
        max_coverage = encounter.get("max_coverage")
    if max_coverage is None:
        max_coverage = initial_result.get("max_coverage")
    return {
        "max_coverage": max_coverage,
        "pvpo_status": initial_result.get("pvpo_status"),
        "pvpo_failure": initial_result.get("pvpo_failure"),
        "injection_not_encountered": bool(
            max_coverage == 0 or initial_result.get("final_status") == "injection_not_encountered"
        ),
    }


def _judge_projection(judge_diagnosis: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": judge_diagnosis.get("status"),
        "refusal_trigger": judge_diagnosis.get("refusal_trigger"),
        "confidence": judge_diagnosis.get("confidence"),
        "evidence_step": judge_diagnosis.get("evidence_step"),
        "evidence_quote": _clip_text(judge_diagnosis.get("evidence_quote")),
        "notes_excerpt": _clip_text(judge_diagnosis.get("notes")),
    }


def _surface_projection(task: dict[str, Any]) -> dict[str, Any]:
    exposure = (
        task.get("exposure_contract") if isinstance(task.get("exposure_contract"), dict) else {}
    )
    delivery = (
        task.get("delivery_channel") if isinstance(task.get("delivery_channel"), dict) else {}
    )
    payload_contract = build_text_payload_contract(task)
    surface: dict[str, Any] = {
        "site": task.get("site"),
        "target_surface_id": task.get("target_surface_id"),
        "route_id": exposure.get("route_id") if isinstance(exposure, dict) else None,
        "editor_method": exposure.get("editor_method") if isinstance(exposure, dict) else None,
        "delivery_mechanism": delivery.get("mechanism") if isinstance(delivery, dict) else None,
    }
    if isinstance(payload_contract, dict):
        for key in (
            "content_capacity",
            "length_budget",
            "required_token_total_chars",
            "max_chars_after_required_tokens",
            "encounter_window",
            "binary_answer_task",
        ):
            if key in payload_contract:
                surface[key] = payload_contract[key]
    return surface


def _action_names(step: ExtractedStep) -> list[str]:
    names: list[str] = []
    for action in step.action:
        if not isinstance(action, dict):
            continue
        names.extend(str(key) for key in action)
    return names


def _action_excerpt(step: ExtractedStep) -> str:
    excerpts: list[str] = []
    for action in step.action:
        if not isinstance(action, dict):
            continue
        for name, args in action.items():
            if name == "navigate" and isinstance(args, dict):
                excerpts.append(f"navigate {args.get('url', '')}")
            elif name == "done" and isinstance(args, dict):
                excerpts.append(f"done {args.get('text', '')}")
            elif isinstance(args, dict) and "index" in args:
                excerpts.append(f"{name} index={args.get('index')}")
            else:
                excerpts.append(str(name))
    return _clip_text("; ".join(excerpts))


def _result_excerpt(step: ExtractedStep) -> str:
    if not step.result:
        return ""
    return _clip_text(" | ".join(str(item) for item in step.result))


def _done_text(step: ExtractedStep) -> str:
    for action in step.action:
        if not isinstance(action, dict):
            continue
        done = action.get("done")
        if isinstance(done, dict) and isinstance(done.get("text"), str):
            return _clip_text(done["text"], _MAX_FINAL_ANSWER_CHARS)
    return ""


def _selected_trace_indexes(steps: list[ExtractedStep], evidence_step: Any) -> list[int]:
    indexes: list[int] = []
    if isinstance(evidence_step, int) and not isinstance(evidence_step, bool):
        indexes.append(evidence_step)
    for step in reversed(steps):
        if not step.partial:
            indexes.append(step.step)
        if len(dict.fromkeys(indexes)) >= _MAX_TRACE_SNIPPETS:
            break
    out: list[int] = []
    for index in indexes:
        if index not in out:
            out.append(index)
    return sorted(out)


def _trajectory_projection(
    trajectory_dir: Path | None,
    *,
    evidence_step: Any,
) -> dict[str, Any]:
    if trajectory_dir is None:
        return {
            "trace_digest_status": "missing_trajectory_dir",
            "trace_snippets": [],
            "final_answer_excerpt": "",
        }
    try:
        trajectory = extract_trajectory(trajectory_dir)
    except (FileNotFoundError, ValueError) as exc:
        return {
            "trace_digest_status": "unavailable",
            "trace_error": _clip_text(exc, 200),
            "trace_snippets": [],
            "final_answer_excerpt": "",
        }

    step_by_index = {step.step: step for step in trajectory.steps if not step.partial}
    snippets: list[dict[str, Any]] = []
    for index in _selected_trace_indexes(trajectory.steps, evidence_step):
        step = step_by_index.get(index)
        if step is None:
            continue
        snippets.append(
            {
                "step": step.step,
                "url": _clip_text(step.url),
                "title": _clip_text(step.title),
                "action_names": _action_names(step),
                "action_excerpt": _action_excerpt(step),
                "result_excerpt": _result_excerpt(step),
            }
        )

    final_answer = ""
    for step in reversed(trajectory.steps):
        if step.partial:
            continue
        final_answer = _done_text(step)
        if final_answer:
            break
    if not final_answer:
        for step in reversed(trajectory.steps):
            if step.partial:
                continue
            final_answer = _result_excerpt(step)
            if final_answer:
                break

    return {
        "trace_digest_status": "ok",
        "trace_partial": trajectory.partial,
        "agent_errors": [_clip_text(error, 200) for error in trajectory.agent_errors[:3]],
        "raw_byte_count": trajectory.raw_byte_count,
        "decode_warnings": trajectory.decode_warnings,
        "trace_snippets": snippets,
        "final_answer_excerpt": final_answer,
    }


def _trace_refs(trajectory_dir: Path | None) -> dict[str, Any]:
    if trajectory_dir is None:
        return {}
    return {
        "trajectory_dir": str(trajectory_dir),
        "history_json": _path_ref(trajectory_dir / "history.json"),
        "result_json": _path_ref(trajectory_dir / "result.json"),
        "judge_raw_response_json": _path_ref(trajectory_dir / "judge_raw_response.json"),
    }


def build_variant_failure_context(
    task: dict[str, Any],
    initial_result: dict[str, Any],
    judge_diagnosis: dict[str, Any],
) -> dict[str, Any]:
    """Build the compressed context passed from judge/eval to variant generation."""
    raw_dir = initial_result.get("trajectory_dir")
    trajectory_dir = Path(raw_dir) if isinstance(raw_dir, str) and raw_dir else None
    evidence_step = judge_diagnosis.get("evidence_step")
    trace = _trajectory_projection(trajectory_dir, evidence_step=evidence_step)
    context = {
        "schema_version": FAILURE_CONTEXT_SCHEMA_VERSION,
        "task_id": str(task.get("id", "unknown")),
        "site": task.get("site"),
        "surface": _surface_projection(task),
        "previous_result": _result_projection(initial_result),
        "pvpo": _pvpo_projection(initial_result),
        "judge": _judge_projection(judge_diagnosis),
        "minimal_evidence": {
            "evidence_quote": _clip_text(judge_diagnosis.get("evidence_quote")),
            "final_answer_excerpt": trace.get("final_answer_excerpt", ""),
            "trace_snippets": trace.get("trace_snippets", []),
        },
        "trace_digest": {
            key: value
            for key, value in trace.items()
            if key not in {"trace_snippets", "final_answer_excerpt"}
        },
        "trace_refs": _trace_refs(trajectory_dir),
        "compression_boundary": {
            "raw_trajectory_omitted": True,
            "screenshots_omitted": True,
            "thinking_memory_next_goal_omitted": True,
            "max_trace_snippets": _MAX_TRACE_SNIPPETS,
            "max_excerpt_chars": _MAX_TEXT_CHARS,
        },
    }
    context["digest_bytes"] = len(json.dumps(context, sort_keys=True, default=str))
    return context


__all__ = ["FAILURE_CONTEXT_SCHEMA_VERSION", "build_variant_failure_context"]
