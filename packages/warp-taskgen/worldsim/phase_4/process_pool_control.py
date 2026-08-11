"""Process-pool supervisor lifecycle and cooperative continuation policy."""

from __future__ import annotations

import fcntl
import os
import shlex
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worldsim.config import BenchmarkInstance
from worldsim.placeholders import normalize_site_name
from worldsim.run_control import pause_control_lock, pause_requested
from worldsim.run_definition import define_run
from worldsim.state import bind_state_paths, load_state_for_current_root, save_state


@dataclass(frozen=True)
class ProcessPoolArgs:
    source_state_dir: Path
    instances: Path
    out_dir: Path
    workers: int
    runner: str
    agent_provider: str | None
    agent_model: str | None
    agent_service_tier: str | None
    agent_llm_timeout: int | None
    agent_step_timeout: int | None
    agent_task_timeout: int | None
    sandbox_model: str | None
    benchmark: Path | None
    sites: str | None
    adversarial_action_kind: str | None
    max_tasks_per_site: int | None
    phase_4_variant_system: str | None
    phase_4_eval_awareness_max_iterations: int | None
    phase_4_variant_budget: str | None
    allow_unknown_auth: bool
    skip_host_bound_storage_state_auth: bool
    task_limit: int | None
    resume: bool = False
    resume_generation: int = 0


@dataclass(frozen=True)
class WorkerAssignment:
    worker_id: int
    task: dict[str, Any]
    instance_index: int
    instance: BenchmarkInstance
    state_dir: Path
    instance_file: Path
    stdout_log: Path
    stderr_log: Path


@dataclass(frozen=True)
class WorkerOutcome:
    assignment: WorkerAssignment
    returncode: int
    timed_out: bool
    started_at: str
    finished_at: str
    results: list[dict[str, Any]]
    error: str | None = None


class ProcessPoolOutputLocked(ValueError):
    """Raised when another supervisor owns the requested output root."""


def source_run_definition(source_state_dir: Path):
    """Load the source identity through the state-owned authority rules."""

    source_state_file = source_state_dir / "pipeline_state.json"
    with bind_state_paths(
        source_state_dir,
        resume_pointer=source_state_dir / "last_run_state.json",
    ):
        source_state = load_state_for_current_root()
    if not source_state:
        if source_state_file.exists():
            raise SystemExit("source process-pool state is unreadable")
        return None
    definition = define_run(source_state)
    return None if definition.legacy else definition


def process_pool_resume_argv(args: ProcessPoolArgs) -> list[str]:
    argv = [
        "uv",
        "run",
        "python",
        "scripts/run_phase4_process_pool.py",
        "--resume",
        "--source-state-dir",
        str(args.source_state_dir.resolve(strict=False)),
        "--instances",
        str(args.instances.resolve(strict=False)),
        "--out-dir",
        str(args.out_dir.resolve(strict=False)),
        "--workers",
        str(args.workers),
        "--runner",
        args.runner,
    ]
    optional_pairs: list[tuple[str, Any]] = [
        ("--agent-provider", args.agent_provider),
        ("--agent-model", args.agent_model),
        ("--agent-service-tier", args.agent_service_tier),
        ("--agent-llm-timeout", args.agent_llm_timeout),
        ("--agent-step-timeout", args.agent_step_timeout),
        ("--agent-task-timeout", args.agent_task_timeout),
        ("--sandbox-model", args.sandbox_model),
        (
            "--benchmark",
            args.benchmark.resolve(strict=False) if args.benchmark is not None else None,
        ),
        ("--sites", args.sites),
        ("--adversarial-action-kind", args.adversarial_action_kind),
        ("--max-tasks-per-site", args.max_tasks_per_site),
        ("--phase-4-variant-system", args.phase_4_variant_system),
        ("--phase-4-eval-awareness-max-iterations", args.phase_4_eval_awareness_max_iterations),
        ("--phase-4-variant-budget", args.phase_4_variant_budget),
        ("--task-limit", args.task_limit),
    ]
    for flag, value in optional_pairs:
        if value is not None:
            argv.extend([flag, str(value)])
    if args.allow_unknown_auth:
        argv.append("--allow-unknown-auth")
    if args.skip_host_bound_storage_state_auth:
        argv.append("--skip-host-bound-storage-state-auth")
    return argv


def process_pool_resume_command(state: dict[str, Any]) -> str:
    argv = state.get("process_pool_resume_argv")
    if not isinstance(argv, list) or not argv:
        raise ValueError("process-pool state is missing its resume command")
    if not all(isinstance(item, str) and item for item in argv):
        raise ValueError("process-pool resume command is malformed")
    return shlex.join(argv)


def outcome_reusable_on_resume(outcome: WorkerOutcome) -> bool:
    if outcome.returncode != 0 or outcome.timed_out or outcome.error is not None:
        return False
    if len(outcome.results) != 1:
        return False
    expected_id = str(outcome.assignment.task.get("id") or "")
    return str(outcome.results[0].get("task_id") or "") == expected_id


def process_pool_state_metadata(
    args: ProcessPoolArgs,
    assignments: Sequence[WorkerAssignment],
    outcomes: Sequence[WorkerOutcome],
    *,
    pause_stage: str,
) -> dict[str, Any]:
    completed_ids = sorted(
        outcome.assignment.worker_id for outcome in outcomes if outcome_reusable_on_resume(outcome)
    )
    assignment_ids = [assignment.worker_id for assignment in assignments]
    completed_set = set(completed_ids)
    return {
        "process_pool": True,
        "pause_stage": pause_stage,
        "source_state_dir": str(args.source_state_dir.resolve(strict=False)),
        "instances_path": str(args.instances.resolve(strict=False)),
        "task_dir_root": str(args.out_dir / "phase_4" / "process_pool_tasks"),
        "process_pool_workers": args.workers,
        "phase_4_max_workers": args.workers,
        "agent_runner": args.runner,
        "agent_provider": args.agent_provider,
        "agent_model": args.agent_model,
        "agent_service_tier": args.agent_service_tier,
        "agent_llm_timeout": args.agent_llm_timeout,
        "agent_step_timeout": args.agent_step_timeout,
        "agent_task_timeout": args.agent_task_timeout,
        "sandbox_model": args.sandbox_model,
        "benchmark_path": str(args.benchmark.resolve(strict=False)) if args.benchmark else None,
        "sites": args.sites,
        "adversarial_action_kind": args.adversarial_action_kind,
        "max_tasks_per_site": args.max_tasks_per_site,
        "phase_4_variant_system": args.phase_4_variant_system,
        "phase_4_eval_awareness_max_iterations": args.phase_4_eval_awareness_max_iterations,
        "phase_4_variant_budget": args.phase_4_variant_budget,
        "allow_unknown_auth": args.allow_unknown_auth,
        "skip_host_bound_storage_state_auth": args.skip_host_bound_storage_state_auth,
        "process_pool_task_limit": args.task_limit,
        "process_pool_resume_generation": args.resume_generation,
        "process_pool_assignment_ids": assignment_ids,
        "process_pool_assignments": assignment_contract(assignments),
        "process_pool_completed_worker_ids": completed_ids,
        "process_pool_attempted_worker_ids": sorted(
            {outcome.assignment.worker_id for outcome in outcomes}
        ),
        "process_pool_pending_worker_ids": [
            worker_id for worker_id in assignment_ids if worker_id not in completed_set
        ],
        "process_pool_resume_argv": process_pool_resume_argv(args),
    }


def assignment_contract(assignments: Sequence[WorkerAssignment]) -> list[dict[str, Any]]:
    return [
        {
            "worker_id": assignment.worker_id,
            "task_id": str(assignment.task.get("id") or ""),
            "site": normalize_site_name(str(assignment.task.get("site") or "")),
            "instance_index": assignment.instance_index,
            "instance_site": normalize_site_name(assignment.instance.site_name),
            "instance_url": assignment.instance.site_url,
        }
        for assignment in assignments
    ]


def save_process_pool_state(
    args: ProcessPoolArgs,
    assignments: Sequence[WorkerAssignment],
    outcomes: Sequence[WorkerOutcome],
    *,
    status: str,
    reason: str,
    pause_stage: str = "process_pool_dispatch",
) -> None:
    metadata = process_pool_state_metadata(
        args,
        assignments,
        outcomes,
        pause_stage=pause_stage,
    )
    with pause_control_lock(args.out_dir):
        save_state("phase_4", status=status, reason=reason, **metadata)


def enter_process_pool_finalizing(
    args: ProcessPoolArgs,
    assignments: Sequence[WorkerAssignment],
    outcomes: Sequence[WorkerOutcome],
) -> bool:
    with pause_control_lock(args.out_dir):
        if pause_requested(args.out_dir):
            return False
        save_state(
            "phase_4",
            status="running",
            reason="process_pool_finalizing",
            **process_pool_state_metadata(
                args,
                assignments,
                outcomes,
                pause_stage="process_pool_finalizing",
            ),
        )
        return True


def validate_paused_resume(
    state: dict[str, Any],
    args: ProcessPoolArgs,
    assignments: Sequence[WorkerAssignment],
) -> list[int]:
    if state.get("status") != "paused" or not state.get("process_pool"):
        raise SystemExit("--resume requires a cooperatively paused process-pool root")
    if state.get("process_pool_resume_argv") != process_pool_resume_argv(args):
        raise SystemExit("process-pool resume arguments do not match the paused supervisor")
    if state.get("process_pool_assignments") != assignment_contract(assignments):
        raise SystemExit("process-pool assignments changed since the supervisor paused")
    raw_completed = state.get("process_pool_completed_worker_ids")
    if not isinstance(raw_completed, list) or any(type(item) is not int for item in raw_completed):
        raise SystemExit("paused process-pool completion metadata is malformed")
    if len(set(raw_completed)) != len(raw_completed):
        raise SystemExit("paused process-pool completion metadata contains duplicates")
    assignment_ids = {assignment.worker_id for assignment in assignments}
    if any(worker_id not in assignment_ids for worker_id in raw_completed):
        raise SystemExit("paused process-pool completion metadata names an unknown worker")
    return raw_completed


@contextmanager
def process_pool_output_lock(out_dir: Path) -> Iterator[None]:
    """Give one supervisor exclusive ownership of an output root."""

    root = out_dir.expanduser().resolve(strict=False)
    root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = root.parent / f".{root.name}.process_pool.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ProcessPoolOutputLocked(f"another process-pool supervisor owns {root}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} cwd={Path.cwd()} cmd={' '.join(sys.argv)}\n")
        handle.flush()
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


__all__ = [
    "ProcessPoolArgs",
    "ProcessPoolOutputLocked",
    "WorkerAssignment",
    "WorkerOutcome",
    "enter_process_pool_finalizing",
    "outcome_reusable_on_resume",
    "process_pool_output_lock",
    "process_pool_resume_argv",
    "process_pool_resume_command",
    "process_pool_state_metadata",
    "save_process_pool_state",
    "source_run_definition",
    "validate_paused_resume",
]
