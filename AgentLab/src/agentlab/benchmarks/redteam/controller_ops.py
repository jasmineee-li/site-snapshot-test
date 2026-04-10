"""State persistence, workspace lifecycle, budget enforcement, phase checkpointing."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agentlab.benchmarks.redteam.app_artifacts import APP_MANIFEST_CONTRACT_VERSION
from agentlab.benchmarks.redteam.controller_state import (
    controller_events_path,
    controller_state_path,
    generation_phase_status_template,
)
from agentlab.benchmarks.redteam.execution import execution_backend_metadata
from agentlab.benchmarks.redteam.git_ops import (
    ControllerWorkspace,
    checkpoint_scope,
    current_head,
    publish_controller_workspace,
    reset_hard,
)
from agentlab.benchmarks.redteam.phase_ids import (
    PHASE_1A,
    PHASE_1B,
    PHASE_1C,
    PHASE_2A,
    PHASE_2B,
    PHASE_3A,
    PHASE_3B,
    PHASE_4A,
    PHASE_4B,
    PHASE_5,
    PHASE_COMPLETED,
    normalize_phase_id,
)
from agentlab.benchmarks.redteam.utils import (
    utc_timestamp as _timestamp,
    write_json as _write_json,
)


# ---------------------------------------------------------------------------
# Forward references — these types live in controller.py but we only need
# them for type annotations.  We import them at runtime from controller.py
# so that the canonical dataclass definitions stay in one place.
# ---------------------------------------------------------------------------

from agentlab.benchmarks.redteam.controller_manifest import _update_manifest_phase


def _get_controller_state_class():
    """Lazy import to break circular dependency with controller.py."""
    from agentlab.benchmarks.redteam.controller import ControllerState
    return ControllerState


def _get_controller_config_class():
    """Lazy import to break circular dependency with controller.py."""
    from agentlab.benchmarks.redteam.controller import ControllerConfig
    return ControllerConfig


# ---------------------------------------------------------------------------
# Phase ordering — needed for _phase_rewind_commit / _mark_remaining_phases_skipped
# ---------------------------------------------------------------------------

_PHASE_ORDER = [
    PHASE_1A,
    PHASE_1B,
    PHASE_1C,
    PHASE_2A,
    PHASE_2B,
    PHASE_3A,
    PHASE_3B,
    PHASE_4A,
    PHASE_4B,
    PHASE_5,
]


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------


def load_controller_state(logs_dir: str | Path) -> dict[str, Any]:
    path = controller_state_path(logs_dir)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def append_controller_event(logs_dir: str | Path, payload: dict[str, Any]) -> None:
    path = controller_events_path(logs_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {"timestamp": _timestamp(), **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def write_controller_state(logs_dir: str | Path, state: Any) -> None:
    ControllerState = _get_controller_state_class()
    assert isinstance(state, ControllerState)
    state.updated_at = _timestamp()
    _write_json(controller_state_path(logs_dir), asdict(state))


# ---------------------------------------------------------------------------
# Workspace lifecycle
# ---------------------------------------------------------------------------


def _publish_workspace(workspace: ControllerWorkspace) -> None:
    if workspace.app_dir.exists():
        publish_controller_workspace(workspace)


_HELPER_PHASES = {PHASE_2B, PHASE_3B, PHASE_4B, PHASE_5}


# ---------------------------------------------------------------------------
# Budget enforcement
# ---------------------------------------------------------------------------


def _enforce_budget(state: Any) -> None:
    budget = int((state.evaluation_config or {}).get("max_total_controller_iterations") or 0)
    used = int(state.total_eval_audit_iterations + state.total_hardening_rounds)
    if budget > 0 and used > budget:
        raise RuntimeError("global_budget_exhausted")


# ---------------------------------------------------------------------------
# Readiness gate helpers
# ---------------------------------------------------------------------------


def _numeric_pass_rate(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _real_readiness_gate(result: dict[str, Any], *, threshold: float) -> tuple[bool, float | None]:
    pass_rate = _numeric_pass_rate(result.get("pass_rate"))
    return (
        bool(result.get("ran"))
        and result.get("error") in (None, "")
        and pass_rate is not None
        and pass_rate >= threshold,
        pass_rate,
    )


# ---------------------------------------------------------------------------
# Phase checkpointing
# ---------------------------------------------------------------------------


def _checkpoint_phase(
    workspace: ControllerWorkspace,
    state: Any,
    phase: str,
    *,
    message_suffix: str,
    allow_empty: bool = False,
    record_as_good: bool = True,
) -> str | None:
    commit = checkpoint_scope(
        workspace.worktree_path,
        scope_path=workspace.app_dir,
        message=f"controller({workspace.app_dir.name}): {message_suffix}",
    )
    if commit and record_as_good:
        state.last_good_commit = commit
        state.phase_checkpoint_commits[phase] = commit
    elif allow_empty:
        commit = current_head(workspace.worktree_path)
        if record_as_good:
            state.last_good_commit = commit
            state.phase_checkpoint_commits[phase] = commit
    return commit


def _effective_repetitions(config: Any) -> int:
    return config.effective_repetitions()


# ---------------------------------------------------------------------------
# Controller state coercion / construction
# ---------------------------------------------------------------------------


def _coerce_controller_state(
    payload: dict[str, Any],
    *,
    workspace: ControllerWorkspace,
    config: Any,
) -> Any:
    ControllerState = _get_controller_state_class()

    if not payload:
        return _build_initial_state(workspace, config)

    if "phase_statuses" in payload:
        allowed_fields = {f.name for f in ControllerState.__dataclass_fields__.values()}
        sanitized_payload = {key: value for key, value in payload.items() if key in allowed_fields}
        sanitized_payload["current_phase"] = normalize_phase_id(sanitized_payload.get("current_phase"))
        sanitized_payload["phase_checkpoint_commits"] = {
            normalize_phase_id(key): str(value)
            for key, value in dict(sanitized_payload.get("phase_checkpoint_commits") or {}).items()
            if str(value)
        }
        return ControllerState(**sanitized_payload)

    raise ValueError(
        "Unsupported legacy controller state artifact. Regenerate or rerun with the current app pipeline."
    )


def _mark_remaining_phases_skipped(
    manifest: dict[str, Any],
    state: Any,
    phase_order: list[str],
    from_phase: str,
) -> None:
    for remaining_phase in phase_order[phase_order.index(from_phase) + 1 :]:
        _update_manifest_phase(manifest, remaining_phase, "skipped")
        state.phase_statuses[remaining_phase] = {"status": "skipped", "updated_at": _timestamp()}


def _update_state_for_phase(
    state: Any,
    *,
    phase: str,
    status: str,
    iteration: int | None = None,
    stop_reason: str = "",
) -> None:
    state.current_phase = phase
    state.phase_statuses.setdefault(phase, {"status": "pending", "updated_at": None})
    state.phase_statuses[phase] = {"status": status, "updated_at": _timestamp()}
    if iteration is not None:
        state.current_iteration = iteration
        state.phase_attempt_counters[phase] = iteration
    if stop_reason:
        state.stop_reason = stop_reason


def _build_initial_state(workspace: ControllerWorkspace, config: Any) -> Any:
    ControllerState = _get_controller_state_class()
    base_commit = current_head(workspace.worktree_path)
    return ControllerState(
        behavior_id=workspace.raw_behavior_id,
        current_phase=PHASE_1A,
        branch=workspace.branch,
        worktree_path=str(workspace.worktree_path),
        owned_paths=[str(workspace.published_app_dir), str(workspace.logs_dir)],
        phase_statuses=generation_phase_status_template(),
        base_commit=base_commit,
        last_good_commit=base_commit,
        evaluation_config={
            "requested_workers": config.workers,
            "requested_repetitions": config.requested_repetitions(),
            "effective_workers": config.workers,
            "effective_repetitions": _effective_repetitions(config),
            "max_eval_iterations": config.max_eval_iterations,
            "hardening_rounds": config.hardening_rounds,
            "tasks_per_hardening_round": config.tasks_per_hardening_round,
            "audit_cadence": config.audit_cadence,
            "max_total_controller_iterations": config.max_total_controller_iterations
            or (2 * config.max_eval_iterations + config.hardening_rounds),
        },
        authoring_backend=execution_backend_metadata(),
    )


def _phase_rewind_commit(state: Any, target_phase: str) -> str:
    target_phase = normalize_phase_id(target_phase)
    if target_phase == PHASE_1A:
        commit = state.base_commit
    else:
        try:
            target_index = _PHASE_ORDER.index(target_phase)
        except ValueError as exc:
            raise RuntimeError(f"Unknown rerun phase: {target_phase}") from exc
        commit = ""
        for phase in reversed(_PHASE_ORDER[:target_index]):
            commit = str((state.phase_checkpoint_commits or {}).get(phase) or "")
            if commit:
                break
    if not commit:
        raise RuntimeError(
            f"Cannot rerun from {target_phase}: missing checkpoint boundary for this controller state"
        )
    return commit


def _prepare_workspace_for_resume(
    workspace: ControllerWorkspace,
    state: Any,
    *,
    rerun_from_phase: str | None = None,
) -> bool:
    last_good_commit = str(state.last_good_commit or "")
    current_phase = normalize_phase_id(state.current_phase)
    phase_status = str((state.phase_statuses.get(current_phase) or {}).get("status") or "")
    preserving_in_progress_helper_state = (
        rerun_from_phase is None
        and current_phase in _HELPER_PHASES
        and phase_status == "in_progress"
    )
    if rerun_from_phase is not None:
        reset_hard(workspace.worktree_path, _phase_rewind_commit(state, rerun_from_phase))
        return True
    if current_phase == PHASE_COMPLETED:
        return False
    if last_good_commit and not preserving_in_progress_helper_state:
        reset_hard(workspace.worktree_path, last_good_commit)
        return True
    return False
