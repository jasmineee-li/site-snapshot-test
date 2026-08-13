"""Read-only bounded waiting for cooperative pause acknowledgement."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from warp_taskgen.run_control import (
    PauseRequest,
    _load_json_object,
    load_pause_request,
    pause_request_path,
    validate_active_pause_request,
)
from warp_taskgen.state import get_state_dir


@dataclass(frozen=True)
class PauseWaitResult:
    """Read-only result of waiting for one pause request to settle."""

    status: str
    request_id: str
    reason_code: str
    state_status: str | None = None
    elapsed_seconds: float = 0.0
    stage: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"pausing", "paused", "terminal", "rejected", "timed_out"}:
            raise ValueError("pause wait result has an unsupported status")
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("pause wait result request_id must be non-empty")
        if not isinstance(self.reason_code, str) or not self.reason_code.strip():
            raise ValueError("pause wait result reason_code must be non-empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "request_id": self.request_id,
            "reason_code": self.reason_code,
            "state_status": self.state_status,
            "elapsed_seconds": round(max(0.0, self.elapsed_seconds), 3),
            "stage": self.stage,
        }


def _state_stage(state: dict[str, object]) -> str | None:
    step = state.get("step")
    field = "phase_2_stage" if step == "phase_2" else "pause_stage"
    stage = state.get(field)
    return stage if isinstance(stage, str) and stage.strip() else None


def current_pause_stage(state_dir: Path | None = None) -> str | None:
    """Read the authoritative internal stage for bounded progress output."""

    root = (state_dir or get_state_dir()).expanduser().resolve(strict=False)
    try:
        return _state_stage(_load_json_object(root / "pipeline_state.json"))
    except (OSError, ValueError):
        return None


def _read_snapshot(
    root: Path, *, request_id: str, expected_request: PauseRequest | None = None
) -> PauseWaitResult:
    """Read authoritative state and marker once, without creating files."""

    try:
        state = _load_json_object(root / "pipeline_state.json")
    except ValueError:
        return PauseWaitResult("rejected", request_id, "pipeline_state_unreadable")
    stage = _state_stage(state)
    status = state.get("status")
    if not isinstance(status, str) or not status.strip():
        return PauseWaitResult("rejected", request_id, "pipeline_status_malformed")
    status = status.strip()
    if status == "paused":
        observed_request = state.get("pause_request_id")
        # An acknowledgement without the exact request identity is not a
        # readback for this wait. This covers legacy/malformed snapshots and a
        # racing marker from another operator.
        if observed_request != request_id:
            return PauseWaitResult(
                "rejected",
                request_id,
                "pause_acknowledgement_identity_mismatch",
                status,
                0.0,
                stage,
            )
        if expected_request is not None:
            observed_run_id = state.get("pause_request_run_id", state.get("run_id"))
            observed_digest = state.get(
                "pause_request_definition_digest", state.get("definition_digest")
            )
            if (
                observed_run_id != expected_request.run_id
                or observed_digest != expected_request.definition_digest
            ):
                return PauseWaitResult(
                    "rejected",
                    request_id,
                    "pause_acknowledgement_definition_mismatch",
                    status,
                    0.0,
                    stage,
                )
        return PauseWaitResult("paused", request_id, "pause_acknowledged", status, 0.0, stage)
    if status in {"complete", "partial_complete", "failed", "interrupted"}:
        return PauseWaitResult("terminal", request_id, "pipeline_terminal", status, 0.0, stage)
    if status != "running":
        return PauseWaitResult(
            "rejected", request_id, "unsupported_pipeline_status", status, 0.0, stage
        )
    marker = pause_request_path(root)
    if not marker.exists():
        return PauseWaitResult("rejected", request_id, "pause_request_missing", status, 0.0, stage)
    try:
        request = load_pause_request(root)
        if request is None:
            return PauseWaitResult(
                "rejected", request_id, "pause_request_missing", status, 0.0, stage
            )
        validate_active_pause_request(state, request)
    except (TypeError, ValueError):
        return PauseWaitResult("rejected", request_id, "pause_request_invalid", status, 0.0, stage)
    if request.request_id != request_id:
        return PauseWaitResult(
            "rejected", request_id, "pause_request_identity_mismatch", status, 0.0, stage
        )
    return PauseWaitResult("pausing", request_id, "pause_boundary_pending", status, 0.0, stage)


def wait_for_pause(
    state_dir: Path | None,
    request_id: str,
    *,
    timeout: float,
    poll_interval: float = 0.25,
    expected_request: PauseRequest | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> PauseWaitResult:
    """Wait for authoritative pause readback using a monotonic deadline.

    The poller performs reads only: it never acknowledges, clears, repairs,
    or writes a lifecycle state. A final read at the deadline closes the
    timeout/acknowledgement race.
    """

    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("pause wait request_id must be non-empty")
    if expected_request is not None:
        if not isinstance(expected_request, PauseRequest):
            raise ValueError("pause wait expected_request must be a PauseRequest")
        if expected_request.request_id != request_id:
            raise ValueError("pause wait expected_request identity does not match request_id")
    if not isinstance(timeout, (int, float)) or not math.isfinite(float(timeout)):
        raise ValueError("pause wait timeout must be finite")
    if timeout < 0:
        raise ValueError("pause wait timeout must be zero or greater")
    if not isinstance(poll_interval, (int, float)) or not math.isfinite(float(poll_interval)):
        raise ValueError("pause wait poll interval must be finite")
    if poll_interval < 0:
        raise ValueError("pause wait poll interval must be zero or greater")
    if timeout > 0 and poll_interval == 0:
        raise ValueError("pause wait poll interval must be positive for a non-zero timeout")
    root = (state_dir or get_state_dir()).expanduser().resolve(strict=False)
    started = clock()
    deadline = started + float(timeout)
    while True:
        observed = _read_snapshot(root, request_id=request_id, expected_request=expected_request)
        elapsed = max(0.0, clock() - started)
        if observed.status != "pausing":
            return PauseWaitResult(
                observed.status,
                observed.request_id,
                observed.reason_code,
                observed.state_status,
                elapsed,
                observed.stage,
            )
        now = clock()
        if now >= deadline:
            final = _read_snapshot(root, request_id=request_id, expected_request=expected_request)
            final_elapsed = max(0.0, clock() - started)
            if final.status != "pausing":
                return PauseWaitResult(
                    final.status,
                    final.request_id,
                    final.reason_code,
                    final.state_status,
                    final_elapsed,
                    final.stage,
                )
            return PauseWaitResult(
                "timed_out",
                request_id,
                "pause_wait_timeout",
                final.state_status,
                final_elapsed,
                final.stage,
            )
        delay = min(float(poll_interval), max(0.0, deadline - now))
        if delay <= 0:
            continue
        sleeper(delay)


__all__ = ["PauseWaitResult", "current_pause_stage", "wait_for_pause"]
