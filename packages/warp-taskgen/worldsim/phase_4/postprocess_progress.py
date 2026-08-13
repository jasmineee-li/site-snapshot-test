"""Phase 4 postprocess heartbeat infrastructure.

Observational telemetry for the Phase 4 postprocess + variant-generation loop.
Tracks per-task lifecycle (started, active, completed, failed) and per-variant
counters (generation attempts, evaluations, PVPO validity, compliance) under an
asyncio.Lock and writes the schema_version=1 progress.json that the CLI status
view (worldsim/cli_status.py) and remote-job status scripts
(scripts/remote_job_status.sh) read. Per CLAUDE.md, heartbeats are observational
only; nothing in Phase 4 routing branches on this state.

The lock guards only the dict/set mutations and the JSON write. Callers must
not invoke reward evaluation, PVPO probes, or strategy variation while holding
the lock.

Ported from upstream feat/worldsim-v5 commit 8fcf0602
("feat(phase4): report variant progress heartbeats") which lived as closures
inside worldsim/phases/phase_4_adversarial.run(). The modular layout split run()
across worldsim/phase_4/runner.py + postprocess.py + strategy_variation.py and
dropped the closures; this module restores them as importable helpers that take
an explicit Phase4ProgressState parameter.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from worldsim.atomic_io import write_json_atomic
from worldsim.phase_4.options import (
    DEFAULT_PHASE_4_VARIANT_BUDGET_PRESET as _DEFAULT_PHASE_4_VARIANT_BUDGET_PRESET,
)
from worldsim.phase_4.options import (
    normalize_eval_awareness_max_iterations as _normalize_eval_awareness_max_iterations,
)
from worldsim.phase_4.options import (
    normalize_phase_4_variant_system as _normalize_phase_4_variant_system,
)
from worldsim.phase_4.options import (
    phase_4_variant_budget_shape as _phase_4_variant_budget_shape,
)
from worldsim.phase_4.variant_accounting import semantic_variant_accounting

_PHASE_4_PROGRESS_ACTIVE_TASK_LIMIT = 12

Phase4ProgressCallback = Callable[[str, Mapping[str, Any]], Awaitable[None]]


@dataclass
class Phase4ProgressState:
    """Mutable progress state guarded by an asyncio.Lock.

    All mutation paths must acquire ``lock``. Callers are expected to construct
    one state per Phase 4 ``run()`` invocation; the dataclass itself is not
    thread-safe outside of asyncio cooperative scheduling.
    """

    state_dir: Path
    task_dir_root: Path
    total_tasks: int
    completed_initial_tasks: int
    phase_4_max_workers: int | None = None
    phase_4_variant_budget: str | None = None
    phase_4_variant_system: str | None = None
    phase_4_eval_awareness_max_iterations: int | None = None
    started_task_ids: set[str] = field(default_factory=set)
    active_task_ids: set[str] = field(default_factory=set)
    completed_task_ids: set[str] = field(default_factory=set)
    failed_task_ids: set[str] = field(default_factory=set)
    variant_progress_by_task: dict[str, dict[str, Any]] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def _jsonable_payload(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _phase_4_progress_path(state_dir: Path) -> Path:
    return state_dir / "phase_4" / "progress.json"


def completed_task_ids_from_task_dir_root(task_dir_root: Path) -> set[str]:
    if not task_dir_root.exists():
        return set()
    completed: set[str] = set()
    for result_path in task_dir_root.glob("*/result.json"):
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        task_id = payload.get("task_id")
        if isinstance(task_id, str) and task_id.strip():
            completed.add(task_id.strip())
    return completed


def compute_progress_extra(state: Phase4ProgressState) -> dict[str, Any]:
    active_task_ids = sorted(state.active_task_ids)
    variant_tasks = [
        progress
        for _, progress in sorted(state.variant_progress_by_task.items())
        if isinstance(progress, dict)
    ]
    active_variant_tasks = [
        progress
        for progress in variant_tasks
        if str(progress.get("task_id") or "") in state.active_task_ids
    ]
    variant_system = _normalize_phase_4_variant_system(state.phase_4_variant_system)
    if variant_system == "eval-awareness-iterator":
        max_iterations = _normalize_eval_awareness_max_iterations(
            state.phase_4_eval_awareness_max_iterations
        )
        budget_preset = "eval-awareness-iterator"
        budget_shape = [1] * max_iterations
    else:
        max_iterations = None
        budget_preset = state.phase_4_variant_budget or _DEFAULT_PHASE_4_VARIANT_BUDGET_PRESET
        budget_shape = list(_phase_4_variant_budget_shape(state.phase_4_variant_budget))
    semantic_keys = (
        "rewrite_attempted",
        "variant_evaluated",
        "variant_rejection_records",
        "pre_browser_rejections",
        "post_eval_rejections",
        "tp_regression_rejections",
        "schema_validation_failures",
        "contract_inapplicable_rejections",
    )
    variant_progress = {
        "schema_version": 1,
        "variant_system": variant_system,
        "budget_preset": budget_preset,
        "budget_shape": budget_shape,
        "eval_awareness_max_iterations": max_iterations,
        "entered_tasks": len(variant_tasks),
        "active_tasks": len(active_variant_tasks),
        "generation_attempted": sum(
            int(progress.get("generation_attempted") or 0) for progress in variant_tasks
        ),
        "generation_generated": sum(
            int(progress.get("generation_generated") or 0) for progress in variant_tasks
        ),
        "generation_failed": sum(
            int(progress.get("generation_failed") or 0) for progress in variant_tasks
        ),
        "evaluated": sum(int(progress.get("evaluated") or 0) for progress in variant_tasks),
        "pvpo_valid": sum(int(progress.get("pvpo_valid") or 0) for progress in variant_tasks),
        "complied": sum(int(progress.get("complied") or 0) for progress in variant_tasks),
        **{
            key: sum(int(progress.get(key) or 0) for progress in variant_tasks)
            for key in semantic_keys
        },
        "task_samples": active_variant_tasks[:_PHASE_4_PROGRESS_ACTIVE_TASK_LIMIT],
    }
    return {
        "postprocess_started_tasks": len(state.started_task_ids),
        "active_postprocess_tasks": len(active_task_ids),
        "active_postprocess_task_ids": active_task_ids[:_PHASE_4_PROGRESS_ACTIVE_TASK_LIMIT],
        "variant_progress": variant_progress,
    }


def write_phase_4_progress(
    state_dir: Path,
    *,
    status: str,
    stage: str,
    task_dir_root: Path,
    total_tasks: int,
    completed_initial_tasks: int = 0,
    postprocessed_tasks: int = 0,
    postprocess_attempted_tasks: int = 0,
    postprocess_failed_tasks: int = 0,
    results_path: Path | None = None,
    final_status_counts: dict[str, int] | None = None,
    phase_4_max_workers: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "phase": "phase_4",
        "status": status,
        "stage": stage,
        "updated_at": datetime.now().isoformat(),
        "task_dir_root": str(task_dir_root),
        "total_tasks": total_tasks,
        "completed_initial_tasks": completed_initial_tasks,
        "postprocessed_tasks": postprocessed_tasks,
        "postprocess_attempted_tasks": postprocess_attempted_tasks,
        "postprocess_failed_tasks": postprocess_failed_tasks,
    }
    if phase_4_max_workers is not None:
        payload["phase_4_max_workers"] = phase_4_max_workers
    if results_path is not None:
        payload["results_path"] = str(results_path)
    if final_status_counts is not None:
        payload["final_status_counts"] = dict(sorted(final_status_counts.items()))
    if extra:
        payload.update(json.loads(json.dumps(dict(extra), default=str)))
    path = _phase_4_progress_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, payload, failpoint_base="phase_4.progress")


async def write_postprocess_progress(state: Phase4ProgressState) -> None:
    write_phase_4_progress(
        state.state_dir,
        status="running",
        stage="postprocessing",
        task_dir_root=state.task_dir_root,
        total_tasks=state.total_tasks,
        completed_initial_tasks=state.completed_initial_tasks,
        postprocessed_tasks=len(state.completed_task_ids),
        postprocess_attempted_tasks=len(state.completed_task_ids) + len(state.failed_task_ids),
        postprocess_failed_tasks=len(state.failed_task_ids),
        phase_4_max_workers=state.phase_4_max_workers,
        extra=compute_progress_extra(state),
    )


async def record_postprocess_start(state: Phase4ProgressState, task_id: str) -> None:
    normalized_id = str(task_id or "unknown").strip() or "unknown"
    async with state.lock:
        state.started_task_ids.add(normalized_id)
        state.active_task_ids.add(normalized_id)
        await write_postprocess_progress(state)


async def record_variant_progress(
    state: Phase4ProgressState,
    task_id: str,
    event: str,
    data: Mapping[str, Any],
) -> None:
    normalized_id = str(task_id or "unknown").strip() or "unknown"
    async with state.lock:
        progress = state.variant_progress_by_task.setdefault(
            normalized_id,
            {
                "task_id": normalized_id,
                "event": "entered",
                "generation_attempted": 0,
                "generation_generated": 0,
                "generation_failed": 0,
                "evaluated": 0,
                "pvpo_valid": 0,
                "complied": 0,
                **semantic_variant_accounting(variant_results=[], generation_errors=[]),
            },
        )
        progress["event"] = event
        progress["updated_at"] = datetime.now().isoformat()
        for key, value in data.items():
            if key == "task_id":
                continue
            progress[str(key)] = _jsonable_payload(value)
        await write_postprocess_progress(state)


async def record_postprocess_result(
    state: Phase4ProgressState,
    task_id: str,
    *,
    failed: bool = False,
) -> None:
    normalized_id = str(task_id or "unknown").strip() or "unknown"
    async with state.lock:
        state.active_task_ids.discard(normalized_id)
        if failed:
            state.failed_task_ids.add(normalized_id)
        else:
            state.completed_task_ids.add(normalized_id)
        await write_postprocess_progress(state)
