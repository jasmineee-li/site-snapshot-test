"""Pipeline state I/O helpers for the redteam app generation pipeline.

Provides functions to read, write, and query the ``pipeline_state.json``
artifact that tracks progress across generation phases.  Extracted from
``app_pipeline.py`` so that other modules (e.g. the controller, the eval
harness, and CLI tooling) can interact with pipeline state without pulling
in the full pipeline dependency graph.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from agentlab.benchmarks.redteam.phase_ids import (
    PHASE_4A,
    PHASE_4B,
    normalize_phase_id,
)
from agentlab.benchmarks.redteam.pipeline_config import DEFAULT_READINESS_BACKEND
from agentlab.benchmarks.redteam.utils import utc_timestamp, write_json

logger = logging.getLogger(__name__)

_PIPELINE_STATE_FILENAME = "pipeline_state.json"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _logs_side_pipeline_state_path(logs_dir: str | Path) -> Path:
    return Path(logs_dir) / _PIPELINE_STATE_FILENAME


def _pipeline_state_path(app_dir: str | Path, *, logs_dir: str | Path | None = None) -> Path:
    if logs_dir is not None:
        return _logs_side_pipeline_state_path(logs_dir)
    app_dir = Path(app_dir)
    git_root = None
    for candidate in (app_dir, *app_dir.parents):
        if (candidate / ".git").exists():
            git_root = candidate
            break
    if git_root is None:
        return app_dir / "logs" / app_dir.name / _PIPELINE_STATE_FILENAME
    return git_root / "logs" / app_dir.name / _PIPELINE_STATE_FILENAME


# ---------------------------------------------------------------------------
# Public API — load / write
# ---------------------------------------------------------------------------


def load_pipeline_state(
    app_dir: str | Path,
    *,
    logs_dir: str | Path | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Load the pipeline state dict from disk, returning ``{}`` when absent."""
    path = _pipeline_state_path(app_dir, logs_dir=logs_dir)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        if strict:
            raise RuntimeError(
                "Unsupported pipeline_state.json artifact. "
                "Regenerate or rerun with the current app pipeline."
            ) from exc
        logger.warning("Ignoring unreadable pipeline state file: %s", path, exc_info=True)
        return {}
    if isinstance(payload, dict):
        return payload
    if strict:
        raise RuntimeError(
            "Unsupported pipeline_state.json artifact. "
            "Regenerate or rerun with the current app pipeline."
        )
    logger.warning("Ignoring non-object pipeline state file: %s", path)
    return {}


def write_pipeline_state(
    app_dir: str | Path,
    *,
    current_phase: str,
    logs_dir: str | Path | None = None,
    current_iteration: int = 0,
    backend: str | None = None,
    last_results_dirs: dict[str, str] | None = None,
    last_audit_summary_path: str | None = None,
    stop_reason: str | None = None,
    regression_status: str | None = None,
    iteration_status: str | None = None,
    phase_iteration_phase: str | None = None,
    phase_iteration_iteration: int | None = None,
    phase_iteration_status: str | None = None,
    phase_iteration_dir: str | None = None,
    phase_progress_phase: str | None = None,
    phase_progress: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist (merge-write) the pipeline state to disk and return it."""
    app_dir = Path(app_dir)
    existing = load_pipeline_state(app_dir, logs_dir=logs_dir)
    normalized_current_phase = normalize_phase_id(current_phase)
    state = {
        "current_phase": normalized_current_phase,
        "current_iteration": current_iteration,
        "backend": backend or existing.get("backend") or DEFAULT_READINESS_BACKEND,
        "iteration_status": iteration_status if iteration_status is not None else "",
        "last_results_dirs": dict(existing.get("last_results_dirs") or {}),
        "last_audit_summary_path": last_audit_summary_path
        if last_audit_summary_path is not None
        else existing.get("last_audit_summary_path", ""),
        "stop_reason": stop_reason if stop_reason is not None else existing.get("stop_reason", ""),
        "regression_status": regression_status
        if regression_status is not None
        else existing.get("regression_status", ""),
        "phase_iterations": dict(existing.get("phase_iterations") or {}),
        "phase_progress": dict(existing.get("phase_progress") or {}),
        "updated_at": utc_timestamp(),
    }
    if last_results_dirs:
        state["last_results_dirs"].update(
            {normalize_phase_id(key): value for key, value in last_results_dirs.items()}
        )
    target_phase = normalize_phase_id(phase_iteration_phase or normalized_current_phase)
    if any(
        value is not None
        for value in (
            phase_iteration_iteration,
            phase_iteration_status,
            phase_iteration_dir,
        )
    ):
        existing_phase_iteration = dict((state["phase_iterations"] or {}).get(target_phase) or {})
        if phase_iteration_iteration is not None:
            existing_phase_iteration["iteration"] = int(phase_iteration_iteration)
        if phase_iteration_status is not None:
            existing_phase_iteration["status"] = str(phase_iteration_status)
        if phase_iteration_dir is not None:
            existing_phase_iteration["iteration_dir"] = str(phase_iteration_dir)
        existing_phase_iteration["updated_at"] = utc_timestamp()
        state["phase_iterations"][target_phase] = existing_phase_iteration
    if phase_progress is not None:
        progress_phase = normalize_phase_id(phase_progress_phase or normalized_current_phase)
        existing_progress = dict((state["phase_progress"] or {}).get(progress_phase) or {})
        existing_progress.update(dict(phase_progress))
        existing_progress["updated_at"] = utc_timestamp()
        state["phase_progress"][progress_phase] = existing_progress
    path = _pipeline_state_path(app_dir, logs_dir=logs_dir)
    write_json(path, state)
    return state


# ---------------------------------------------------------------------------
# Phase-state query helpers
# ---------------------------------------------------------------------------


def _phase_iteration_state(
    state: dict[str, Any],
    phase_name: str,
) -> dict[str, Any]:
    """Return the iteration sub-dict for *phase_name* (or its alias)."""
    phase_iterations = state.get("phase_iterations") or {}
    for candidate in _phase_state_aliases(phase_name):
        if candidate in phase_iterations:
            return dict(phase_iterations[candidate] or {})
    return {}


def _phase_progress_state(
    state: dict[str, Any],
    phase_name: str,
) -> dict[str, Any]:
    """Return the progress sub-dict for *phase_name* (or its alias)."""
    phase_progress = state.get("phase_progress") or {}
    for candidate in _phase_state_aliases(phase_name):
        if candidate in phase_progress:
            return dict(phase_progress[candidate] or {})
    return {}


def _phase_state_aliases(phase_name: str) -> tuple[str, ...]:
    """Return the canonical phase id plus any fallback aliases."""
    normalized = normalize_phase_id(phase_name)
    if normalized == PHASE_4A:
        return (PHASE_4A, PHASE_4B)
    if normalized == PHASE_4B:
        return (PHASE_4B, PHASE_4A)
    return (normalized,)


# ---------------------------------------------------------------------------
# Path normalisation helpers
# ---------------------------------------------------------------------------


def _normalize_result_dir(path_value: str | Path | None) -> str:
    """Resolve a result directory path to an absolute string (empty when falsy)."""
    if not path_value:
        return ""
    return str(Path(path_value).resolve())


def _expected_eval_iteration_dir(phase_dir: Path, iteration: int) -> Path:
    """Return the conventional ``iter_NN`` subdirectory for an eval iteration."""
    return phase_dir / f"iter_{iteration:02d}"
