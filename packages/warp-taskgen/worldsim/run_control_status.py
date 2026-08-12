"""Read-only Run-control projections for the operator status surface.

The pipeline checkpoint remains the only lifecycle authority.  This module
only explains the current state and counts feature-owned checkpoint envelopes;
it never accepts, routes, or mutates a checkpoint.
"""

from __future__ import annotations

import json
import shlex
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from worldsim.run_control import (
    _PROCESS_POOL_PAUSE_STAGE,
    _SUPPORTED_PAUSE_STAGES,
    load_pause_request,
    pause_request_path,
    validate_active_pause_request,
)
from worldsim.run_control_history import load_transition_history
from worldsim.run_definition import define_run, plan_resume

_TERMINAL_STATUSES = frozenset({"complete", "partial_complete", "failed", "interrupted"})


def _read_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _integer(value: object) -> int | None:
    if type(value) is int and value >= 0:
        return value
    return None


def _nested_integer(state: dict[str, Any], names: tuple[str, ...]) -> int | None:
    for name in names:
        value = _integer(state.get(name))
        if value is not None:
            return value
    for parent_name in ("run_control", "checkpoint_counts", "phase_2_checkpoint_counts"):
        parent = state.get(parent_name)
        if not isinstance(parent, dict):
            continue
        for name in names:
            value = _integer(parent.get(name))
            if value is not None:
                return value
    return None


def _phase2_checkpoint_counts(run_root: Path, state: dict[str, Any], stage: str) -> dict[str, Any]:
    """Project counts only from feature-owned durable envelopes or state fields."""

    if stage == "planning":
        completed = _nested_integer(
            state,
            (
                "phase_2_planning_completed_count",
                "planning_completed_count",
                "planning_checkpoint_completed_count",
            ),
        )
        admitted = _nested_integer(
            state,
            (
                "phase_2_planning_admitted_count",
                "planning_admitted_count",
                "planning_checkpoint_admitted_count",
            ),
        )
        queued = _nested_integer(
            state,
            (
                "phase_2_planning_queued_count",
                "planning_queued_count",
                "planning_checkpoint_queued_count",
            ),
        )
    elif stage == "text_fill":
        completed = _nested_integer(
            state,
            (
                "phase_2_text_fill_completed_count",
                "text_fill_completed_count",
                "text_fill_checkpoint_completed_count",
            ),
        )
        admitted = _nested_integer(
            state,
            (
                "phase_2_text_fill_admitted_count",
                "text_fill_admitted_count",
                "text_fill_checkpoint_admitted_count",
            ),
        )
        queued = _nested_integer(
            state,
            (
                "phase_2_text_fill_queued_count",
                "text_fill_queued_count",
                "text_fill_checkpoint_queued_count",
            ),
        )
    elif stage == "feasibility":
        completed = _nested_integer(
            state,
            (
                "phase_2_feasibility_completed_count",
                "feasibility_completed_count",
                "feasibility_checkpoint_completed_count",
            ),
        )
        if completed is None:
            # A terminal phase-2c transition owns these aggregate counts.  It
            # is explanatory only and does not make the aggregate reusable.
            values = [
                _integer(state.get("feasibility_verified_count")),
                _integer(state.get("feasibility_infeasible_count")),
                _integer(state.get("feasibility_skipped_count")),
            ]
            if any(value is not None for value in values):
                completed = sum(value or 0 for value in values)
        admitted = _nested_integer(
            state,
            (
                "phase_2_feasibility_admitted_count",
                "feasibility_admitted_count",
                "feasibility_checkpoint_admitted_count",
            ),
        )
        queued = _nested_integer(
            state,
            (
                "phase_2_feasibility_queued_count",
                "feasibility_queued_count",
                "feasibility_checkpoint_queued_count",
            ),
        )
    else:
        return {"queued": None, "admitted": None, "completed": None, "authority": "unknown"}

    return {
        "queued": queued,
        "admitted": admitted,
        "completed": completed,
        "authority": f"advisory:phase_2.{stage}.state_projection",
    }


def _phase4_checkpoint_counts(run_root: Path, state: dict[str, Any], stage: str) -> dict[str, Any]:
    progress = _read_object(run_root / "phase_4" / "progress.json")
    # Progress is intentionally not used to decide lifecycle routing.  These
    # fields are included only when a feature-owned Phase 4 state field names
    # the corresponding checkpoint count explicitly.
    completed = _nested_integer(
        state,
        (
            "phase_4_completed_checkpoint_count",
            "phase4_completed_checkpoint_count",
            "completed_checkpoint_count",
        ),
    )
    admitted = _nested_integer(
        state,
        (
            "phase_4_admitted_checkpoint_count",
            "phase4_admitted_checkpoint_count",
            "admitted_checkpoint_count",
        ),
    )
    queued = _nested_integer(
        state,
        (
            "phase_4_queued_checkpoint_count",
            "phase4_queued_checkpoint_count",
            "queued_checkpoint_count",
        ),
    )
    if progress is not None:
        # A Phase 4 state snapshot may explicitly bind the progress counters
        # to its checkpoint owner. Never infer them from a generic progress
        # document alone.
        if completed is None and progress.get("checkpoint_completed_count") is not None:
            completed = _integer(progress.get("checkpoint_completed_count"))
        if admitted is None and progress.get("checkpoint_admitted_count") is not None:
            admitted = _integer(progress.get("checkpoint_admitted_count"))
        if queued is None and progress.get("checkpoint_queued_count") is not None:
            queued = _integer(progress.get("checkpoint_queued_count"))
    return {
        "queued": queued,
        "admitted": admitted,
        "completed": completed,
        "authority": f"advisory:phase_4.{stage}.state_projection",
    }


def _age_seconds(value: object) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return max(0.0, (datetime.now(UTC) - parsed.astimezone(UTC)).total_seconds())


def _supported_stage(state: dict[str, Any]) -> tuple[str | None, bool]:
    step = state.get("step")
    if not isinstance(step, str) or step not in _SUPPORTED_PAUSE_STAGES:
        return None, False
    if step == "phase_2":
        stage = state.get("phase_2_stage")
    else:
        stage = state.get("pause_stage")
    if step == "phase_4" and state.get("process_pool"):
        normalized_stage = stage if isinstance(stage, str) and stage.strip() else None
        return normalized_stage, normalized_stage == _PROCESS_POOL_PAUSE_STAGE
    supported = isinstance(stage, str) and stage in _SUPPORTED_PAUSE_STAGES[step]
    return (str(stage) if isinstance(stage, str) and stage else None, supported)


def _resume_next_action(run_root: Path, state: dict[str, Any]) -> dict[str, str | None] | None:
    """Project the normal resume lifecycle action without dispatching it."""

    if state.get("process_pool"):
        if state.get("status") != "paused":
            return {
                "description": "inspect or repair the process-pool artifacts; generic resume is fail-closed",
                "command": None,
            }
        try:
            from worldsim.phase_4.process_pool_control import process_pool_resume_command

            command = process_pool_resume_command(state)
        except (TypeError, ValueError):
            return {
                "description": "inspect the missing or malformed process-pool resume command",
                "command": None,
            }
        return {
            "description": "resume the paused process pool with its isolated supervisor wrapper",
            "command": command,
        }

    try:
        plan = plan_resume(define_run(state), state, run_root=run_root)
    except (OSError, TypeError, ValueError):
        return None
    root = shlex.quote(str(run_root))
    command = f"WARP_TASKGEN_STATE_DIR={root} uv run warp-taskgen resume"
    if plan.lifecycle_action == "finished":
        return {
            "description": "no action; the final Phase 4 checkpoint is complete",
            "command": None,
        }
    if plan.lifecycle_action == "advance_phase":
        target = plan.target_step or "the next phase"
        return {
            "description": f"resume to advance the completed checkpoint into {target}",
            "command": command,
        }
    if plan.lifecycle_action == "rerun_phase":
        target = plan.target_step or plan.current_step or "the current phase"
        return {
            "description": f"resume to rerun {target} from its checkpoint",
            "command": command,
        }
    return {"description": "inspect the rejected resume plan before retrying", "command": None}


def _next_action(
    run_root: Path,
    state: dict[str, Any],
    stage: str | None,
    supported: bool,
    marker: dict[str, Any] | None,
    request_error: str | None,
) -> dict[str, str | None]:
    status = str(state.get("status") or "unknown")
    root = shlex.quote(str(run_root))
    resume_command = f"WARP_TASKGEN_STATE_DIR={root} uv run warp-taskgen resume"
    if request_error is not None:
        return {
            "description": "inspect or clear the malformed pause request, then retry",
            "command": None,
        }
    if status in {"complete", "partial_complete", "failed", "paused", "interrupted"}:
        planned = _resume_next_action(run_root, state)
        if planned is not None:
            return planned
        if status in {"paused", "interrupted"}:
            return {
                "description": "resume the run from its last checkpoint",
                "command": resume_command,
            }
        if status in _TERMINAL_STATUSES:
            return {
                "description": "inspect the unavailable resume plan before retrying",
                "command": None,
            }
    if marker is not None:
        return {
            "description": "wait for the requested pause to reach its safe checkpoint",
            "command": f"WARP_TASKGEN_STATE_DIR={root} uv run warp-taskgen pause --wait",
        }
    if state.get("status") == "running" and stage is not None and not supported:
        return {
            "description": "inspect the unsupported stage; cooperative pause is unavailable",
            "command": None,
        }
    if not supported:
        return {"description": "inspect the unsupported stage", "command": None}
    return {
        "description": "request a cooperative pause before stopping the run",
        "command": f"WARP_TASKGEN_STATE_DIR={root} uv run warp-taskgen pause --wait",
    }


def _authoritative_pause_payload(state: dict[str, Any]) -> dict[str, Any] | None:
    """Project pause identity retained by pipeline state after marker removal."""

    if state.get("status") not in {"paused", "interrupted"}:
        return None
    request_id = state.get("pause_request_id")
    if not isinstance(request_id, str) or not request_id.strip():
        return None
    requested_at = state.get("pause_requested_at")
    if not isinstance(requested_at, str) or not requested_at.strip():
        requested_at = None
    reason_code = state.get("reason_code")
    if not isinstance(reason_code, str) or not reason_code.strip():
        reason_code = state.get("reason")
    if not isinstance(reason_code, str) or not reason_code.strip():
        reason_code = (
            "abrupt_process_interruption"
            if state.get("status") == "interrupted"
            else "operator_requested_pause"
        )
    payload: dict[str, Any] = {
        "request_id": request_id,
        "reason_code": reason_code,
        "age_seconds": _age_seconds(requested_at),
        "source": "authoritative_pipeline_state",
    }
    if requested_at is not None:
        payload["requested_at"] = requested_at
    step = state.get("step")
    if isinstance(step, str) and step.strip():
        payload["step"] = step
    return payload


def build_run_control_projection(run_root: Path, state: dict[str, Any]) -> dict[str, Any]:
    """Build the non-authoritative run-control portion of ``status``."""

    stage, supported = _supported_stage(state)
    status = str(state.get("status") or "unknown")
    marker_payload: dict[str, Any] | None = None
    request_error: str | None = None
    marker_path = pause_request_path(run_root)
    if marker_path.exists():
        try:
            request = load_pause_request(run_root)
            if request is None:
                raise ValueError("pause request marker is missing")
            validate_active_pause_request(state, request)
            marker_payload = request.to_dict()
            marker_payload["reason_code"] = "operator_requested_pause"
            marker_payload["age_seconds"] = _age_seconds(request.requested_at)
        except (TypeError, ValueError) as exc:
            request_error = str(exc)
    if marker_payload is None:
        marker_payload = _authoritative_pause_payload(state)

    if state.get("step") == "phase_2" and stage is not None:
        counts = _phase2_checkpoint_counts(run_root, state, stage)
    elif state.get("step") == "phase_4" and stage is not None:
        counts = _phase4_checkpoint_counts(run_root, state, stage)
    else:
        counts = {"queued": None, "admitted": None, "completed": None, "authority": "unknown"}

    lifecycle_status = status
    if marker_payload is not None and status == "running":
        lifecycle_status = "pausing"
    if request_error is not None and status == "running":
        lifecycle_status = "rejected"
    next_action = _next_action(run_root, state, stage, supported, marker_payload, request_error)
    supported_stages = {step: sorted(stages) for step, stages in _SUPPORTED_PAUSE_STAGES.items()}
    if state.get("step") == "phase_4" and state.get("process_pool"):
        supported_stages["phase_4"] = sorted(
            {*supported_stages["phase_4"], _PROCESS_POOL_PAUSE_STAGE}
        )
    projection: dict[str, Any] = {
        "lifecycle_status": lifecycle_status,
        "state_status": status,
        "supported_stage": stage,
        "supported": supported,
        "supported_stages": supported_stages,
        "pause_request": marker_payload,
        "pause_request_error": request_error,
        "checkpoint_counts": counts,
        "feature_checkpoint_counts": {stage: counts} if stage is not None else {},
        "next_action": next_action,
        "transition_history": load_transition_history(run_root),
    }
    if marker_payload is not None:
        projection["pause_request_id"] = marker_payload.get("request_id")
        projection["pause_reason_code"] = marker_payload.get("reason_code")
        projection["pause_age_seconds"] = marker_payload.get("age_seconds")
    return projection


__all__ = ["build_run_control_projection"]
