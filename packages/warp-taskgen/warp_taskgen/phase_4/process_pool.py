"""Process-isolated Phase 4 orchestration.

This module keeps the existing Phase 4 runner as the measurement core.  The
process pool only materializes isolated worker state directories, assigns one
task to one benchmark instance, launches a normal ``warp_taskgen.main phase 4``
subprocess, and merges canonical per-worker results.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import shutil
import sys
from collections import Counter, deque
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from warp_taskgen.atomic_io import write_json_atomic
from warp_taskgen.config import BenchmarkConfig, BenchmarkInstance, load_benchmark_config
from warp_taskgen.phase_4.options import (
    normalize_phase_4_variant_system as _normalize_phase_4_variant_system,
)
from warp_taskgen.phase_4.postprocess_progress import write_phase_4_progress
from warp_taskgen.phase_4.process_pool_control import (
    ProcessPoolArgs,
    ProcessPoolOutputLocked,
    WorkerAssignment,
    WorkerOutcome,
    process_pool_output_lock,
    validate_paused_resume,
)
from warp_taskgen.phase_4.process_pool_control import (
    enter_process_pool_finalizing as _enter_process_pool_finalizing,
)
from warp_taskgen.phase_4.process_pool_control import (
    process_pool_resume_argv as _process_pool_resume_argv,
)
from warp_taskgen.phase_4.process_pool_control import (
    save_process_pool_state as _save_process_pool_state,
)
from warp_taskgen.phase_4.process_pool_control import (
    source_run_definition as _source_run_definition,
)
from warp_taskgen.phase_4.results import _write_phase_4_results
from warp_taskgen.placeholders import normalize_site_name
from warp_taskgen.run_control import (
    acknowledge_pause,
    pause_control_lock,
    pause_requested,
)
from warp_taskgen.state import bind_run_definition, bind_state_paths
from warp_taskgen.task_paths import safe_task_path_component

_REUSABLE_PHASE_INPUT_DIRS = ("phase_0a", "phase_0c", "phase_1", "phase_2", "phase_3")
_WORKER_TIMEOUT_GRACE_S = 45
_VARIANT_WORKER_POSTPROCESS_GRACE_S = 600


@dataclass(frozen=True)
class _RunningWorker:
    assignment: WorkerAssignment
    process: asyncio.subprocess.Process
    slot_id: int
    started_at: str
    timeout: int | None


@dataclass(frozen=True)
class _AssignmentRun:
    outcomes: list[WorkerOutcome]
    paused: bool


def main(argv: list[str] | None = None) -> int:
    from warp_taskgen.cli.phase4_lock import Phase4AlreadyRunning, _phase4_run_lock
    from warp_taskgen.phase_4.sweep_tag import sweep_in_progress

    args = _parse_args(argv)
    pool_args = ProcessPoolArgs(
        source_state_dir=args.source_state_dir,
        instances=args.instances,
        out_dir=args.out_dir
        or Path(os.environ.get("WORLDSIM_STATE_DIR", "") or "logs/phase4_process_pool"),
        workers=args.workers,
        runner=args.runner,
        agent_provider=args.agent_provider,
        agent_model=args.agent_model,
        agent_service_tier=args.agent_service_tier,
        agent_llm_timeout=args.agent_llm_timeout,
        agent_step_timeout=args.agent_step_timeout,
        agent_task_timeout=args.agent_task_timeout,
        sandbox_model=args.sandbox_model,
        benchmark=args.benchmark,
        sites=args.sites,
        adversarial_action_kind=args.adversarial_action_kind,
        max_tasks_per_site=args.max_tasks_per_site,
        phase_4_variant_system=args.phase_4_variant_system,
        phase_4_eval_awareness_max_iterations=args.phase_4_eval_awareness_max_iterations,
        phase_4_variant_budget=args.phase_4_variant_budget,
        allow_unknown_auth=args.allow_unknown_auth,
        skip_host_bound_storage_state_auth=args.skip_host_bound_storage_state_auth,
        task_limit=args.task_limit,
        resume=args.resume,
    )
    try:
        with _phase4_run_lock(pool_args.source_state_dir), sweep_in_progress():
            return asyncio.run(run_process_pool(pool_args))
    except (Phase4AlreadyRunning, ProcessPoolOutputLocked) as exc:
        raise SystemExit(f"process-pool supervisor is already active: {exc}") from exc
    except ValueError as exc:
        raise SystemExit(f"process-pool state rejected: {exc}") from exc


async def run_process_pool(args: ProcessPoolArgs) -> int:
    pointer = args.out_dir / "last_run_state.json"
    source_definition = _source_run_definition(args.source_state_dir)
    with process_pool_output_lock(args.out_dir):
        with bind_state_paths(args.out_dir, resume_pointer=pointer):
            with bind_run_definition(source_definition, state_dir=args.out_dir):
                return await _run_process_pool_bound(args)


async def _run_process_pool_bound(args: ProcessPoolArgs) -> int:
    _validate_source_state(args.source_state_dir)
    if args.source_state_dir.resolve(strict=False) == args.out_dir.resolve(strict=False):
        raise SystemExit("process-pool output must be isolated from its source state root")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if not args.resume and args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit(f"output dir already exists and is not empty: {args.out_dir}")
    if args.resume and not args.out_dir.exists():
        raise SystemExit(f"process-pool resume root does not exist: {args.out_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _materialize_reusable_phase_inputs(args.source_state_dir, args.out_dir)

    config = load_benchmark_config(args.instances)
    tasks = _admitted_tasks_for_pool(args)
    if args.task_limit is not None:
        tasks = tasks[: args.task_limit]
    if not tasks:
        raise SystemExit("no admitted Phase 4 tasks available for process pool")

    if args.resume:
        paused_state = _load_json_dict(args.out_dir / "pipeline_state.json")
        prior_generation = paused_state.get("process_pool_resume_generation", 0)
        if type(prior_generation) is not int or prior_generation < 0:
            raise SystemExit("process-pool resume generation is malformed")
        args = replace(args, resume_generation=prior_generation + 1)
    assignments = _build_assignments(args, config, tasks)
    prior_outcomes = _load_paused_outcomes(args, assignments) if args.resume else []
    completed_ids = {outcome.assignment.worker_id for outcome in prior_outcomes}
    pending = [
        assignment for assignment in assignments if assignment.worker_id not in completed_ids
    ]
    _save_process_pool_state(
        args,
        assignments,
        prior_outcomes,
        status="running",
        reason="process_pool_resumed" if args.resume else "process_pool_started",
    )
    _write_pool_progress(
        args,
        assignments,
        prior_outcomes,
        status="running",
        stage="queued",
    )
    run = await _run_assignments(
        args,
        pending,
        all_assignments=assignments,
        prior_outcomes=prior_outcomes,
    )
    outcomes = [*prior_outcomes, *run.outcomes]
    if run.paused or not _enter_process_pool_finalizing(args, assignments, outcomes):
        _save_process_pool_state(
            args,
            assignments,
            outcomes,
            status="running",
            reason="process_pool_pause_boundary",
        )
        acknowledge_pause(args.out_dir)
        _write_pool_progress(
            args,
            assignments,
            outcomes,
            status="paused",
            stage="process_pool_paused",
        )
        print(
            "Process pool paused after active children finished. Resume with:\n"
            f"  {shlex.join(_process_pool_resume_argv(args))}"
        )
        return 0
    return _merge_outcomes(args, config, tasks, outcomes)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-state-dir", type=Path, required=True)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--runner", default="browser_use")
    parser.add_argument("--agent-provider", default=None)
    parser.add_argument("--agent-model", default=None)
    parser.add_argument("--agent-service-tier", default=None)
    parser.add_argument("--agent-llm-timeout", type=int, default=None)
    parser.add_argument("--agent-step-timeout", type=int, default=None)
    parser.add_argument("--agent-task-timeout", type=int, default=None)
    parser.add_argument("--sandbox-model", default=None)
    parser.add_argument("--benchmark", type=Path, default=None)
    parser.add_argument("--sites", default=None)
    parser.add_argument("--adversarial-action-kind", default=None)
    parser.add_argument("--max-tasks-per-site", type=int, default=None)
    parser.add_argument("--phase-4-variant-system", default=None)
    parser.add_argument("--phase-4-eval-awareness-max-iterations", type=int, default=None)
    parser.add_argument("--phase-4-variant-budget", default=None)
    parser.add_argument("--allow-unknown-auth", action="store_true")
    parser.add_argument("--skip-host-bound-storage-state-auth", action="store_true")
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue a cooperatively paused process-pool root.",
    )
    return parser.parse_args(argv)


def _load_paused_outcomes(
    args: ProcessPoolArgs,
    assignments: list[WorkerAssignment],
) -> list[WorkerOutcome]:
    state = _load_json_dict(args.out_dir / "pipeline_state.json")
    validate_paused_resume(state, args, assignments)
    # A paused pool records prior successes for inspection, but resume does not
    # promote them into a new canonical merge without the child runner's full
    # fingerprint/sidecar validator. The isolated attempt reruns every assignment.
    return []


def _validate_source_state(source: Path) -> None:
    for required in ("phase_2/adversarial_tasks.json", "phase_3/contracts.json"):
        if not (source / required).exists():
            raise SystemExit(f"source state dir is missing required Phase 4 input: {required}")


def _materialize_reusable_phase_inputs(source: Path, dest: Path) -> None:
    for phase_dir in _REUSABLE_PHASE_INPUT_DIRS:
        source_phase = source / phase_dir
        if not source_phase.exists():
            continue
        dest_phase = dest / phase_dir
        if dest_phase.exists():
            continue
        try:
            dest_phase.symlink_to(source_phase.resolve(), target_is_directory=True)
        except OSError:
            shutil.copytree(source_phase, dest_phase)


def _admitted_tasks_for_pool(args: ProcessPoolArgs) -> list[dict[str, Any]]:
    from warp_taskgen.phase_4 import admission as phase_4_admission

    admission = phase_4_admission._load_admitted_phase_4_tasks(
        state_dir=args.source_state_dir,
        sites_filter_raw=args.sites,
        adversarial_action_kind_filter_raw=args.adversarial_action_kind,
        max_tasks_per_site=args.max_tasks_per_site,
        state_metadata={
            "task_dir_root": str(args.out_dir / "phase_4" / "process_pool_tasks"),
            "instances_path": str(args.instances),
            "agent_model": args.agent_model,
            "agent_runner": args.runner,
            "process_pool": True,
        },
    )
    if admission["return_code"] is not None:
        raise SystemExit(int(admission["return_code"]))
    return list(admission["tasks"])


def _build_assignments(
    args: ProcessPoolArgs,
    config: BenchmarkConfig,
    tasks: list[dict[str, Any]],
) -> list[WorkerAssignment]:
    instances_by_site: dict[str, deque[tuple[int, BenchmarkInstance]]] = {}
    for index, instance in enumerate(config.instances):
        instances_by_site.setdefault(normalize_site_name(instance.site_name), deque()).append(
            (index, instance)
        )

    assignments: list[WorkerAssignment] = []
    for worker_id, task in enumerate(tasks):
        site = normalize_site_name(str(task.get("site", "")))
        candidates = instances_by_site.get(site)
        if not candidates:
            raise SystemExit(f"no instances available for task {task.get('id')} site {site!r}")
        instance_index, instance = candidates[0]
        candidates.rotate(-1)
        worker_collection = args.out_dir / "phase_4" / "process_pool_workers"
        if args.resume_generation:
            worker_collection = (
                args.out_dir
                / "phase_4"
                / "process_pool_resume_workers"
                / f"attempt_{args.resume_generation:03d}"
            )
        worker_root = worker_collection / f"worker_{worker_id:03d}"
        assignments.append(
            WorkerAssignment(
                worker_id=worker_id,
                task=task,
                instance_index=instance_index,
                instance=instance,
                state_dir=worker_root / "state",
                instance_file=worker_root / "instances.json",
                stdout_log=worker_root / "stdout.log",
                stderr_log=worker_root / "stderr.log",
            )
        )
    return assignments


async def _run_assignments(
    args: ProcessPoolArgs,
    assignments: list[WorkerAssignment],
    *,
    all_assignments: list[WorkerAssignment] | None = None,
    prior_outcomes: list[WorkerOutcome] | None = None,
) -> _AssignmentRun:
    progress_assignments = all_assignments or assignments
    prior_outcomes = prior_outcomes or []
    pending = deque(assignments)
    outcomes: list[WorkerOutcome] = []
    lock = asyncio.Lock()
    admission_lock = asyncio.Lock()
    busy_instances: set[int] = set()
    instance_available = asyncio.Event()
    instance_available.set()
    active: dict[int, tuple[WorkerAssignment, int]] = {}
    stop_heartbeat = asyncio.Event()
    paused = asyncio.Event()

    async def _slot(slot_id: int) -> None:
        while True:
            assignment: WorkerAssignment | None = None
            running: _RunningWorker | None = None
            launch_error: Exception | None = None
            while assignment is None:
                async with admission_lock:
                    with pause_control_lock(args.out_dir):
                        if pause_requested(args.out_dir):
                            paused.set()
                            return
                        candidate = next(
                            (item for item in pending if item.instance_index not in busy_instances),
                            None,
                        )
                        if candidate is None:
                            if not pending:
                                return
                            instance_available.clear()
                        else:
                            pending.remove(candidate)
                            assignment = candidate
                            busy_instances.add(assignment.instance_index)
                            active[assignment.worker_id] = (assignment, slot_id)
                            _write_pool_progress(
                                args,
                                progress_assignments,
                                [*prior_outcomes, *outcomes],
                                status="running",
                                stage="workers",
                                active_assignments=list(active.values()),
                            )
                            try:
                                running = await _launch_one_worker(
                                    args,
                                    assignment,
                                    slot_id=slot_id,
                                )
                            except Exception as exc:
                                launch_error = exc
                if assignment is None:
                    await instance_available.wait()
            assert assignment is not None
            try:
                if launch_error is not None:
                    outcome = WorkerOutcome(
                        assignment=assignment,
                        returncode=-1,
                        timed_out=False,
                        started_at=datetime.now().isoformat(),
                        finished_at=datetime.now().isoformat(),
                        results=[],
                        error=repr(launch_error),
                    )
                else:
                    assert running is not None
                    try:
                        outcome = await _finish_one_worker(running)
                    except Exception as exc:
                        outcome = WorkerOutcome(
                            assignment=assignment,
                            returncode=-1,
                            timed_out=False,
                            started_at=running.started_at,
                            finished_at=datetime.now().isoformat(),
                            results=[],
                            error=repr(exc),
                        )
                async with lock:
                    active.pop(assignment.worker_id, None)
                    outcomes.append(outcome)
                    _write_pool_progress(
                        args,
                        progress_assignments,
                        [*prior_outcomes, *outcomes],
                        status="running",
                        stage="workers",
                        active_assignments=list(active.values()),
                    )
            finally:
                async with admission_lock:
                    busy_instances.discard(assignment.instance_index)
                    instance_available.set()

    async def _heartbeat() -> None:
        while not stop_heartbeat.is_set():
            await asyncio.sleep(30)
            async with lock:
                _write_pool_progress(
                    args,
                    progress_assignments,
                    [*prior_outcomes, *outcomes],
                    status="running",
                    stage="workers",
                    active_assignments=list(active.values()),
                )

    heartbeat_task = asyncio.create_task(_heartbeat())
    slots = [
        asyncio.create_task(_slot(slot_id))
        for slot_id in range(min(args.workers, len(assignments)))
    ]
    try:
        await asyncio.gather(*slots)
    finally:
        stop_heartbeat.set()
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
    with pause_control_lock(args.out_dir):
        if pause_requested(args.out_dir):
            paused.set()
    return _AssignmentRun(outcomes=outcomes, paused=paused.is_set())


async def _run_one_worker(
    args: ProcessPoolArgs,
    assignment: WorkerAssignment,
    *,
    slot_id: int,
) -> WorkerOutcome:
    running = await _launch_one_worker(args, assignment, slot_id=slot_id)
    return await _finish_one_worker(running)


async def _launch_one_worker(
    args: ProcessPoolArgs,
    assignment: WorkerAssignment,
    *,
    slot_id: int,
) -> _RunningWorker:
    started_at = datetime.now().isoformat()
    _prepare_worker_state(args, assignment)
    command = _worker_command(args, assignment)
    assignment.stdout_log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    worker_state_dir = str(assignment.state_dir)
    env["WARP_TASKGEN_STATE_DIR"] = worker_state_dir
    env["WORLDSIM_STATE_DIR"] = worker_state_dir
    env["WARP_TASKGEN_RESUME_POINTER"] = str(assignment.state_dir / "last_run_state.json")
    env["WORLDSIM_PROCESS_POOL_WORKER_ID"] = str(assignment.worker_id)
    env["WORLDSIM_PROCESS_POOL_SLOT_ID"] = str(slot_id)
    timeout = _worker_timeout_seconds(args)
    with assignment.stdout_log.open("wb") as stdout, assignment.stderr_log.open("wb") as stderr:
        proc = await asyncio.create_subprocess_exec(
            *command,
            cwd=Path.cwd(),
            env=env,
            stdout=stdout,
            stderr=stderr,
        )
    _write_worker_status(
        assignment,
        slot_id=slot_id,
        status="running",
        pid=proc.pid,
        started_at=started_at,
    )
    return _RunningWorker(
        assignment=assignment,
        process=proc,
        slot_id=slot_id,
        started_at=started_at,
        timeout=timeout,
    )


async def _finish_one_worker(running: _RunningWorker) -> WorkerOutcome:
    assignment = running.assignment
    proc = running.process
    timed_out = False
    try:
        returncode = await asyncio.wait_for(proc.wait(), timeout=running.timeout)
    except TimeoutError:
        timed_out = True
        proc.terminate()
        try:
            returncode = await asyncio.wait_for(proc.wait(), timeout=15)
        except TimeoutError:
            proc.kill()
            returncode = await proc.wait()
    results, error = _load_worker_results(assignment)
    status_payload = _write_worker_status(
        assignment,
        slot_id=running.slot_id,
        status="timed_out" if timed_out else "completed",
        returncode=returncode,
        timed_out=timed_out,
        result_count=len(results),
        error=error,
        started_at=running.started_at,
        finished_at=datetime.now().isoformat(),
    )
    return WorkerOutcome(
        assignment=assignment,
        returncode=returncode,
        timed_out=timed_out,
        started_at=running.started_at,
        finished_at=status_payload["finished_at"],
        results=results,
        error=error,
    )


def _prepare_worker_state(args: ProcessPoolArgs, assignment: WorkerAssignment) -> None:
    assignment.state_dir.mkdir(parents=True, exist_ok=True)
    _materialize_reusable_phase_inputs(args.source_state_dir, assignment.state_dir)
    payload = load_benchmark_config(args.instances).model_dump(mode="json")
    payload["instances"] = [assignment.instance.model_dump(mode="json")]
    write_json_atomic(assignment.instance_file, payload)


def _worker_command(args: ProcessPoolArgs, assignment: WorkerAssignment) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "warp_taskgen.main",
        "phase",
        "4",
        "--instances",
        str(assignment.instance_file),
        "--runner",
        args.runner,
        "--phase-4-task-id",
        str(assignment.task.get("id")),
        "--phase-4-max-workers",
        "1",
    ]
    optional_pairs: list[tuple[str, Any]] = [
        ("--agent-provider", args.agent_provider),
        ("--agent-model", args.agent_model),
        ("--agent-service-tier", args.agent_service_tier),
        ("--agent-llm-timeout", args.agent_llm_timeout),
        ("--agent-step-timeout", args.agent_step_timeout),
        ("--agent-task-timeout", args.agent_task_timeout),
        ("--sandbox-model", args.sandbox_model),
        ("--benchmark", args.benchmark),
        ("--sites", args.sites),
        ("--adversarial-action-kind", args.adversarial_action_kind),
        ("--phase-4-variant-system", args.phase_4_variant_system),
        ("--phase-4-eval-awareness-max-iterations", args.phase_4_eval_awareness_max_iterations),
        ("--phase-4-variant-budget", args.phase_4_variant_budget),
    ]
    for flag, value in optional_pairs:
        if value is not None:
            command.extend([flag, str(value)])
    if args.allow_unknown_auth:
        command.append("--allow-unknown-auth")
    if args.skip_host_bound_storage_state_auth:
        command.append("--skip-host-bound-storage-state-auth")
    return command


def _worker_timeout_seconds(args: ProcessPoolArgs) -> int | None:
    """Return the outer wall-clock timeout for a single process-pool task.

    ``--agent-task-timeout`` applies to one browser-agent trajectory. A Phase 4
    task can run the baseline trajectory plus multiple post-resistance variants,
    so the process-pool guard must cover the whole task envelope instead of
    reusing the per-trajectory timeout.
    """

    if not args.agent_task_timeout:
        return None
    trajectory_budget = int(args.agent_task_timeout)
    variant_system = _normalize_phase_4_variant_system(args.phase_4_variant_system)
    variant_runs = 0
    if variant_system != "none":
        if variant_system == "eval-awareness-iterator":
            variant_runs = max(0, int(args.phase_4_eval_awareness_max_iterations or 3))
        else:
            variant_runs = 3
    return (
        trajectory_budget * (1 + variant_runs)
        + _VARIANT_WORKER_POSTPROCESS_GRACE_S
        + _WORKER_TIMEOUT_GRACE_S
    )


def _load_worker_results(assignment: WorkerAssignment) -> tuple[list[dict[str, Any]], str | None]:
    results_path = assignment.state_dir / "phase_4" / "results.json"
    if not results_path.exists():
        salvaged = _salvage_worker_results_from_iterator_checkpoint(assignment)
        if salvaged:
            return (
                salvaged,
                f"salvaged missing worker results from iterator checkpoint: {results_path}",
            )
        return [], f"missing worker results: {results_path}"
    try:
        payload = json.loads(results_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [], f"could not parse worker results {results_path}: {exc}"
    if not isinstance(payload, list):
        return [], f"worker results must be a JSON array: {results_path}"
    return [item for item in payload if isinstance(item, dict)], None


def _salvage_worker_results_from_iterator_checkpoint(
    assignment: WorkerAssignment,
) -> list[dict[str, Any]]:
    """Build an inspectable task row when a worker dies after completed variants.

    This is intentionally narrow: it only handles eval-awareness iterator
    checkpoints that contain at least one completed variant result. It does not
    manufacture successful outcomes from an in-flight browser run.
    """

    phase4_dir = assignment.state_dir / "phase_4"
    if not phase4_dir.exists():
        return []
    task_id = str(assignment.task.get("id") or "")
    task_slug = safe_task_path_component(task_id)
    for run_dir in sorted([item for item in phase4_dir.iterdir() if item.is_dir()], reverse=True):
        trace_dir = run_dir / task_slug
        checkpoint_path = trace_dir / "eval_awareness_iterator_checkpoint.json"
        baseline_result_path = trace_dir / "result.json"
        if not checkpoint_path.exists() or not baseline_result_path.exists():
            continue
        checkpoint = _load_json_dict(checkpoint_path)
        baseline_result = _load_json_dict(baseline_result_path)
        iterations = [item for item in checkpoint.get("iterations", []) if isinstance(item, dict)]
        if not any(isinstance(item.get("result"), dict) for item in iterations):
            continue
        try:
            from warp_taskgen.phase_4.eval_awareness_iterator_budget import (
                build_eval_awareness_iterator_result_from_checkpoint,
            )
        except Exception:
            return []
        variation = build_eval_awareness_iterator_result_from_checkpoint(
            initial_result=baseline_result,
            checkpoint=checkpoint,
            max_iterations=checkpoint.get("max_iterations"),
            stop_reason_override="worker_timeout_salvaged",
        )
        if not variation:
            continue
        variant_results = [
            item for item in variation.get("variant_results", []) if isinstance(item, dict)
        ]
        interpreted_refusals = [
            item
            for item in variant_results
            if _result_is_ecologically_valid(item) and item.get("outcome") == "refused_or_ignored"
        ]
        final_status = (
            "success_on_variant"
            if any(
                _result_is_ecologically_valid(item) and item.get("outcome") == "complied"
                for item in variant_results
            )
            else "resistant"
            if interpreted_refusals
            else "inconclusive"
        )
        result = {
            **baseline_result,
            "task_id": task_id,
            "initial_outcome": baseline_result.get("outcome"),
            "ecologically_valid": _result_is_ecologically_valid(baseline_result),
            "judge_diagnosis": None,
            "strategies_attempted": [
                {
                    "strategy": item.get("strategy"),
                    "outcome": item.get("outcome"),
                    "ecologically_valid": _result_is_ecologically_valid(item),
                }
                for item in variant_results
            ],
            "final_status": final_status,
            "successful_strategy": (
                "eval_awareness_iterator" if final_status == "success_on_variant" else None
            ),
            "eval_awareness_iterator": {
                **variation,
                "salvaged": True,
                "salvage_reason": "process_pool_worker_timeout",
            },
        }
        return [result]
    return []


def _result_is_ecologically_valid(result: dict[str, Any]) -> bool:
    encounter = result.get("encounter")
    if isinstance(encounter, dict):
        try:
            return float(encounter.get("max_coverage") or 0) > 0
        except (TypeError, ValueError):
            return False
    return bool(result.get("ecologically_valid"))


def _merge_outcomes(
    args: ProcessPoolArgs,
    config: BenchmarkConfig,
    tasks: list[dict[str, Any]],
    outcomes: list[WorkerOutcome],
) -> int:
    task_order = {str(task.get("id")): index for index, task in enumerate(tasks)}
    final_results: list[dict[str, Any]] = []
    errors: list[str] = []
    seen: set[str] = set()
    task_root = args.out_dir / "phase_4" / "process_pool_tasks"
    task_root.mkdir(parents=True, exist_ok=True)
    for outcome in outcomes:
        expected_id = str(outcome.assignment.task.get("id"))
        salvaged_timeout = bool(
            outcome.error
            and outcome.error.startswith("salvaged missing worker results from iterator checkpoint")
        )
        if salvaged_timeout:
            errors.append(f"{expected_id}: {outcome.error}")
        if outcome.returncode != 0 and not salvaged_timeout:
            errors.append(f"{expected_id}: worker exited {outcome.returncode}")
        if outcome.timed_out and not salvaged_timeout:
            errors.append(f"{expected_id}: worker timed out")
        if outcome.error and not salvaged_timeout:
            errors.append(f"{expected_id}: {outcome.error}")
        if len(outcome.results) != 1:
            errors.append(f"{expected_id}: expected 1 result, got {len(outcome.results)}")
            continue
        result = outcome.results[0]
        task_id = str(result.get("task_id") or "")
        if task_id != expected_id:
            errors.append(f"{expected_id}: worker result task_id mismatch {task_id!r}")
            continue
        if task_id in seen:
            errors.append(f"{task_id}: duplicate result")
            continue
        seen.add(task_id)
        path_replacements = _copy_worker_task_artifacts(outcome.assignment, task_root)
        final_results.append(
            _rewrite_worker_paths(result, outcome.assignment, task_root, path_replacements)
        )
    missing = sorted(set(task_order) - seen)
    if missing:
        errors.append(f"missing result(s): {', '.join(missing)}")
    _write_process_pool_summary(args, outcomes, errors)
    if errors:
        _write_partial_process_pool_results(
            args=args,
            tasks=tasks,
            final_results=final_results,
            outcomes=outcomes,
            errors=errors,
            task_order=task_order,
        )
        write_phase_4_progress(
            args.out_dir,
            status="failed",
            stage="process_pool_merge_failed",
            task_dir_root=task_root,
            total_tasks=len(tasks),
            completed_initial_tasks=len(final_results),
            postprocessed_tasks=len(final_results),
            postprocess_attempted_tasks=len(outcomes),
            postprocess_failed_tasks=len(errors),
            phase_4_max_workers=args.workers,
            extra={"process_pool_errors": errors[:20]},
        )
        return 1
    final_results.sort(key=lambda result: task_order[str(result.get("task_id"))])
    state_metadata = {
        "task_dir_root": str(task_root),
        "instances_path": str(args.instances),
        "agent_model": args.agent_model,
        "agent_runner": args.runner,
        "sandbox_model": args.sandbox_model,
        "agent_provider": args.agent_provider,
        "agent_service_tier": args.agent_service_tier,
        "agent_llm_timeout": args.agent_llm_timeout,
        "agent_step_timeout": args.agent_step_timeout,
        "agent_task_timeout": args.agent_task_timeout,
        "max_tasks_per_site": args.max_tasks_per_site,
        "allow_unknown_auth": args.allow_unknown_auth,
        "skip_host_bound_storage_state_auth": args.skip_host_bound_storage_state_auth,
        "phase_4_max_workers": args.workers,
        "phase_4_variant_system": args.phase_4_variant_system,
        "phase_4_eval_awareness_max_iterations": args.phase_4_eval_awareness_max_iterations,
        "phase_4_variant_budget": args.phase_4_variant_budget,
        "process_pool": True,
        "process_pool_workers": args.workers,
    }
    _ = config
    return _write_phase_4_results(
        state_dir=args.out_dir,
        state_metadata=state_metadata,
        final_results=final_results,
        tasks=tasks,
    )


def _write_partial_process_pool_results(
    *,
    args: ProcessPoolArgs,
    tasks: list[dict[str, Any]],
    final_results: list[dict[str, Any]],
    outcomes: list[WorkerOutcome],
    errors: list[str],
    task_order: dict[str, int],
) -> None:
    """Persist inspectable partial results while keeping canonical results absent."""

    phase4_dir = args.out_dir / "phase_4"
    phase4_dir.mkdir(parents=True, exist_ok=True)
    final_results.sort(key=lambda result: task_order.get(str(result.get("task_id")), 10**9))
    write_json_atomic(phase4_dir / "results.partial.json", final_results)
    result_ids = {str(result.get("task_id")) for result in final_results}
    expected_ids = [str(task.get("id")) for task in tasks]
    manifest = {
        "schema_version": 1,
        "process_pool": True,
        "paper_eligible": False,
        "canonical_results_written": False,
        "partial_results_path": str(phase4_dir / "results.partial.json"),
        "expected_tasks": len(expected_ids),
        "partial_results": len(final_results),
        "missing_task_ids": [task_id for task_id in expected_ids if task_id not in result_ids],
        "errors": errors,
        "workers": [
            {
                "worker_id": outcome.assignment.worker_id,
                "task_id": outcome.assignment.task.get("id"),
                "returncode": outcome.returncode,
                "timed_out": outcome.timed_out,
                "result_count": len(outcome.results),
                "error": outcome.error,
                "stdout": str(outcome.assignment.stdout_log),
                "stderr": str(outcome.assignment.stderr_log),
                "state_dir": str(outcome.assignment.state_dir),
            }
            for outcome in sorted(outcomes, key=lambda item: item.assignment.worker_id)
        ],
    }
    write_json_atomic(phase4_dir / "partial_manifest.json", manifest)


def _copy_worker_task_artifacts(
    assignment: WorkerAssignment,
    task_root: Path,
) -> list[tuple[str, str]]:
    phase4_dir = assignment.state_dir / "phase_4"
    replacements: list[tuple[str, str]] = []
    for child in phase4_dir.iterdir() if phase4_dir.exists() else []:
        if not child.is_dir():
            continue
        for artifact_dir in child.iterdir():
            if not artifact_dir.is_dir():
                continue
            target = task_root / artifact_dir.name
            if target.exists():
                target = _unique_process_pool_artifact_target(
                    task_root,
                    assignment=assignment,
                    run_dir=child,
                    artifact_dir=artifact_dir,
                )
            shutil.copytree(artifact_dir, target, symlinks=True)
            replacements.extend(
                [
                    (str(artifact_dir), str(target)),
                    (str(artifact_dir.resolve()), str(target.resolve())),
                ]
            )
    return replacements


def _unique_process_pool_artifact_target(
    task_root: Path,
    *,
    assignment: WorkerAssignment,
    run_dir: Path,
    artifact_dir: Path,
) -> Path:
    base = (
        f"worker_{assignment.worker_id:03d}__"
        f"{safe_task_path_component(run_dir.name)}__"
        f"{artifact_dir.name}"
    )
    target = task_root / base
    suffix = 1
    while target.exists():
        target = task_root / f"{base}__{suffix}"
        suffix += 1
    return target


def _rewrite_worker_paths(
    value: Any,
    assignment: WorkerAssignment,
    task_root: Path,
    replacements: list[tuple[str, str]] | None = None,
) -> Any:
    phase4_dir = assignment.state_dir / "phase_4"
    path_replacements: list[tuple[str, str]] = list(replacements or [])
    if phase4_dir.exists():
        for child in phase4_dir.iterdir():
            if not child.is_dir():
                continue
            path_replacements.extend(
                [
                    (str(child), str(task_root)),
                    (str(child.resolve()), str(task_root.resolve())),
                ]
            )
    return _rewrite_string_values(value, path_replacements)


def _rewrite_string_values(value: Any, replacements: list[tuple[str, str]]) -> Any:
    if isinstance(value, str):
        rewritten = value
        for source, target in replacements:
            if rewritten.startswith(source):
                rewritten = target + rewritten[len(source) :]
        return rewritten
    if isinstance(value, list):
        return [_rewrite_string_values(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _rewrite_string_values(item, replacements) for key, item in value.items()}
    return value


def _write_pool_progress(
    args: ProcessPoolArgs,
    assignments: list[WorkerAssignment],
    outcomes: list[WorkerOutcome],
    *,
    status: str,
    stage: str,
    active_assignments: list[tuple[WorkerAssignment, int]] | None = None,
) -> None:
    completed = len(outcomes)
    active_assignments = active_assignments or []
    counts = Counter(
        "timed_out" if outcome.timed_out else "failed" if outcome.returncode != 0 else "complete"
        for outcome in outcomes
    )
    write_phase_4_progress(
        args.out_dir,
        status=status,
        stage=stage,
        task_dir_root=args.out_dir / "phase_4" / "process_pool_tasks",
        total_tasks=len(assignments),
        completed_initial_tasks=completed,
        postprocessed_tasks=completed,
        postprocess_attempted_tasks=completed,
        postprocess_failed_tasks=counts.get("failed", 0) + counts.get("timed_out", 0),
        phase_4_max_workers=args.workers,
        extra={
            "process_pool": True,
            "process_pool_workers": args.workers,
            "process_pool_outcomes": dict(sorted(counts.items())),
            "active_initial_tasks": len(active_assignments),
            "active_initial_task_ids": [
                str(assignment.task.get("id"))
                for assignment, _slot_id in sorted(
                    active_assignments,
                    key=lambda item: item[0].worker_id,
                )[:12]
            ],
            "process_pool_active_workers": [
                _active_worker_progress_payload(assignment, slot_id)
                for assignment, slot_id in sorted(
                    active_assignments,
                    key=lambda item: item[0].worker_id,
                )[:12]
            ],
        },
    )


def _write_worker_status(
    assignment: WorkerAssignment,
    *,
    slot_id: int,
    status: str,
    started_at: str,
    pid: int | None = None,
    returncode: int | None = None,
    timed_out: bool | None = None,
    result_count: int | None = None,
    error: str | None = None,
    finished_at: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "worker_id": assignment.worker_id,
        "slot_id": slot_id,
        "task_id": assignment.task.get("id"),
        "instance_index": assignment.instance_index,
        "pvpo_cdp_url": assignment.instance.pvpo_cdp_url,
        "state_dir": str(assignment.state_dir),
        "stdout": str(assignment.stdout_log),
        "stderr": str(assignment.stderr_log),
        "worker_progress_path": str(_worker_progress_path(assignment)),
        "started_at": started_at,
        "updated_at": datetime.now().isoformat(),
    }
    if pid is not None:
        payload["pid"] = pid
    if returncode is not None:
        payload["returncode"] = returncode
    if timed_out is not None:
        payload["timed_out"] = timed_out
    if result_count is not None:
        payload["result_count"] = result_count
    if error is not None:
        payload["error"] = error
    if finished_at is not None:
        payload["finished_at"] = finished_at
    payload.update(_child_phase4_progress_fields(assignment))
    write_json_atomic(assignment.state_dir / "process_pool_worker_status.json", payload)
    return payload


def _active_worker_progress_payload(
    assignment: WorkerAssignment,
    slot_id: int,
) -> dict[str, Any]:
    status = _load_json_dict(assignment.state_dir / "process_pool_worker_status.json")
    payload: dict[str, Any] = {
        "worker_id": assignment.worker_id,
        "slot_id": slot_id,
        "task_id": assignment.task.get("id"),
        "instance_index": assignment.instance_index,
        "pvpo_cdp_url": assignment.instance.pvpo_cdp_url,
        "state_dir": str(assignment.state_dir),
        "stdout": str(assignment.stdout_log),
        "stderr": str(assignment.stderr_log),
        "worker_progress_path": str(_worker_progress_path(assignment)),
    }
    if status:
        for key in ("status", "pid", "started_at", "updated_at"):
            if key in status:
                payload[key] = status[key]
    payload.update(_child_phase4_progress_fields(assignment))
    return payload


def _child_phase4_progress_fields(assignment: WorkerAssignment) -> dict[str, Any]:
    child_progress = _load_json_dict(_worker_progress_path(assignment))
    task_id = str(assignment.task.get("id") or "unknown")
    fields: dict[str, Any] = {}
    if child_progress:
        task_root = str(child_progress.get("task_dir_root") or "")
        fields["child_phase4_status"] = child_progress.get("status")
        fields["current_step"] = child_progress.get("stage")
        fields["child_progress_updated_at"] = child_progress.get("updated_at")
        fields["child_task_dir_root"] = task_root
        if task_root:
            fields["task_trace_dir"] = str(Path(task_root) / safe_task_path_component(task_id))
        fields["active_postprocess_task_ids"] = child_progress.get("active_postprocess_task_ids")
        fields["active_initial_task_ids"] = child_progress.get("active_initial_task_ids")
    else:
        fields["current_step"] = "worker_subprocess_starting"
    sidecar_status = _load_json_dict(
        _candidate_task_trace_dir(assignment) / "agentlab_sidecar_status.json"
    )
    if sidecar_status:
        for key in (
            "sidecar_status",
            "current_phase",
            "current_step",
            "last_url",
            "last_screenshot",
            "last_network_event_count",
            "timeline_path",
        ):
            source_key = "status" if key == "sidecar_status" else key
            if source_key in sidecar_status:
                fields[key] = sidecar_status[source_key]
    return fields


def _worker_progress_path(assignment: WorkerAssignment) -> Path:
    return assignment.state_dir / "phase_4" / "progress.json"


def _candidate_task_trace_dir(assignment: WorkerAssignment) -> Path:
    progress = _load_json_dict(_worker_progress_path(assignment))
    task_root = progress.get("task_dir_root")
    if isinstance(task_root, str) and task_root.strip():
        return Path(task_root) / safe_task_path_component(assignment.task.get("id"))
    return (
        assignment.state_dir
        / "phase_4"
        / "unknown"
        / safe_task_path_component(assignment.task.get("id"))
    )


def _load_json_dict(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_process_pool_summary(
    args: ProcessPoolArgs,
    outcomes: list[WorkerOutcome],
    errors: list[str],
) -> None:
    payload = {
        "schema_version": 1,
        "process_pool": True,
        "workers": args.workers,
        "total_outcomes": len(outcomes),
        "errors": errors,
        "outcomes": [
            {
                "worker_id": outcome.assignment.worker_id,
                "task_id": outcome.assignment.task.get("id"),
                "instance_index": outcome.assignment.instance_index,
                "pvpo_cdp_url": outcome.assignment.instance.pvpo_cdp_url,
                "returncode": outcome.returncode,
                "timed_out": outcome.timed_out,
                "result_count": len(outcome.results),
                "error": outcome.error,
                "stdout": str(outcome.assignment.stdout_log),
                "stderr": str(outcome.assignment.stderr_log),
                "state_dir": str(outcome.assignment.state_dir),
            }
            for outcome in sorted(outcomes, key=lambda item: item.assignment.worker_id)
        ],
    }
    write_json_atomic(args.out_dir / "phase_4" / "process_pool_summary.json", payload)


if __name__ == "__main__":
    raise SystemExit(main())
