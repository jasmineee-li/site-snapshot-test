import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.repair_process_pool_partial import repair_process_pool_partial
from worldsim.config import BenchmarkConfig
from worldsim.phase_4 import process_pool
from worldsim.phase_4.process_pool import (
    ProcessPoolArgs,
    WorkerOutcome,
    _active_worker_progress_payload,
    _build_assignments,
    _copy_worker_task_artifacts,
    _load_paused_outcomes,
    _load_worker_results,
    _merge_outcomes,
    _rewrite_worker_paths,
    _run_assignments,
    _save_process_pool_state,
    _worker_command,
    _worker_timeout_seconds,
    _write_worker_status,
    run_process_pool,
)
from worldsim.phase_4.process_pool_control import process_pool_output_lock
from worldsim.run_control import acknowledge_pause, request_pause
from worldsim.state import bind_state_paths


def _pool_args(tmp_path: Path, **overrides) -> ProcessPoolArgs:
    values = {
        "source_state_dir": tmp_path / "source",
        "instances": tmp_path / "instances.json",
        "out_dir": tmp_path / "out",
        "workers": 48,
        "runner": "browser_use",
        "agent_provider": "openai",
        "agent_model": "gpt-5.2",
        "agent_service_tier": "priority",
        "agent_llm_timeout": 240,
        "agent_step_timeout": 300,
        "agent_task_timeout": 2400,
        "sandbox_model": "claude-sonnet-4-6",
        "benchmark": tmp_path,
        "sites": "gitlab,reddit",
        "adversarial_action_kind": None,
        "max_tasks_per_site": None,
        "phase_4_variant_system": "eval-awareness-iterator",
        "phase_4_eval_awareness_max_iterations": 3,
        "phase_4_variant_budget": None,
        "allow_unknown_auth": False,
        "skip_host_bound_storage_state_auth": False,
        "task_limit": None,
    }
    values.update(overrides)
    return ProcessPoolArgs(**values)


def _config(tmp_path: Path, instances: list[dict]) -> BenchmarkConfig:
    return BenchmarkConfig.model_validate(
        {
            "benchmark_name": "WebArena Verified",
            "benchmark_codebase": str(tmp_path),
            "instances": instances,
        }
    )


def test_build_assignments_allows_duplicate_legacy_pvpo_urls(tmp_path):
    args = _pool_args(tmp_path)
    config = _config(
        tmp_path,
        [
            {
                "site_name": "gitlab",
                "site_url": "http://127.0.0.1:8023",
                "pvpo_cdp_url": "http://127.0.0.1:9222",
            },
            {
                "site_name": "gitlab",
                "site_url": "http://127.0.0.1:8025",
                "pvpo_cdp_url": "http://127.0.0.1:9222",
            },
        ],
    )

    assignments = _build_assignments(args, config, [{"id": "adv-1", "site": "gitlab"}])

    assert len(assignments) == 1


def test_worker_command_filters_to_single_task_and_worker(tmp_path):
    args = _pool_args(tmp_path)
    config = _config(
        tmp_path,
        [
            {
                "site_name": "gitlab",
                "site_url": "http://127.0.0.1:8023",
                "pvpo_cdp_url": "http://127.0.0.1:9222",
            }
        ],
    )
    assignment = _build_assignments(args, config, [{"id": "adv-1", "site": "gitlab"}])[0]

    command = _worker_command(args, assignment)

    assert command[:3][-2:] == ["-m", "worldsim.main"]
    assert "--phase-4-task-id" in command
    assert command[command.index("--phase-4-task-id") + 1] == "adv-1"
    assert command[command.index("--phase-4-max-workers") + 1] == "1"
    assert command[command.index("--agent-service-tier") + 1] == "priority"


def test_worker_command_supports_agentlab_with_single_task_process(tmp_path):
    args = _pool_args(tmp_path, runner="agentlab")
    config = _config(
        tmp_path,
        [
            {
                "site_name": "gitlab",
                "site_url": "http://127.0.0.1:8023",
                "pvpo_cdp_url": "http://127.0.0.1:9222",
            }
        ],
    )
    assignment = _build_assignments(args, config, [{"id": "adv-1", "site": "gitlab"}])[0]

    command = _worker_command(args, assignment)

    assert command[command.index("--runner") + 1] == "agentlab"
    assert command[command.index("--phase-4-task-id") + 1] == "adv-1"
    assert command[command.index("--phase-4-max-workers") + 1] == "1"


@pytest.mark.asyncio
async def test_worker_launch_overrides_inherited_supervisor_state_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _pool_args(tmp_path)
    config = _config(
        tmp_path,
        [
            {
                "site_name": "gitlab",
                "site_url": "http://127.0.0.1:8023",
                "pvpo_cdp_url": "http://127.0.0.1:9222",
            }
        ],
    )
    assignment = _build_assignments(
        args,
        config,
        [{"id": "adv-1", "site": "gitlab"}],
    )[0]
    captured: dict[str, object] = {}

    async def fake_create_subprocess_exec(*command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return SimpleNamespace(pid=1234)

    monkeypatch.setenv("WARP_TASKGEN_STATE_DIR", str(tmp_path / "supervisor"))
    monkeypatch.setenv(
        "WARP_TASKGEN_RESUME_POINTER",
        str(tmp_path / "supervisor" / "last_run_state.json"),
    )
    monkeypatch.setattr(process_pool, "_prepare_worker_state", lambda *a, **k: None)
    monkeypatch.setattr(
        process_pool.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    monkeypatch.setattr(process_pool, "_write_worker_status", lambda *a, **k: None)

    await process_pool._launch_one_worker(args, assignment, slot_id=0)

    env = captured["env"]
    assert isinstance(env, dict)
    assert env["WARP_TASKGEN_STATE_DIR"] == str(assignment.state_dir)
    assert env["WORLDSIM_STATE_DIR"] == str(assignment.state_dir)
    assert env["WARP_TASKGEN_RESUME_POINTER"] == str(assignment.state_dir / "last_run_state.json")


def test_worker_timeout_scales_with_eval_awareness_iterations(tmp_path):
    args = _pool_args(
        tmp_path,
        agent_task_timeout=2400,
        phase_4_variant_system="eval-awareness-iterator",
        phase_4_eval_awareness_max_iterations=3,
    )

    assert _worker_timeout_seconds(args) == 2400 * 4 + 600 + 45


def test_worker_timeout_defaults_to_eval_awareness_iterator_budget(tmp_path):
    args = _pool_args(
        tmp_path,
        agent_task_timeout=10,
        phase_4_variant_system=None,
        phase_4_eval_awareness_max_iterations=3,
    )

    assert _worker_timeout_seconds(args) == 10 * 4 + 600 + 45


def test_worker_timeout_uses_single_trajectory_without_variants(tmp_path):
    args = _pool_args(
        tmp_path,
        agent_task_timeout=900,
        phase_4_variant_system="none",
        phase_4_eval_awareness_max_iterations=3,
    )

    assert _worker_timeout_seconds(args) == 900 + 600 + 45


def test_process_pool_output_lock_rejects_second_supervisor(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"

    with process_pool_output_lock(out_dir):
        with pytest.raises(ValueError, match="another process-pool supervisor"):
            with process_pool_output_lock(out_dir):
                pass


@pytest.mark.asyncio
async def test_process_pool_pause_serializes_request_with_launch_and_drains_active_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _pool_args(tmp_path, out_dir=tmp_path / "out", workers=1)
    args.out_dir.mkdir(parents=True)
    config = _config(
        tmp_path,
        [
            {
                "site_name": "gitlab",
                "site_url": "http://gitlab-1.test",
                "pvpo_cdp_url": "http://127.0.0.1:9222",
            },
            {
                "site_name": "gitlab",
                "site_url": "http://gitlab-2.test",
                "pvpo_cdp_url": "http://127.0.0.1:9223",
            },
        ],
    )
    assignments = _build_assignments(
        args,
        config,
        [
            {"id": "adv-1", "site": "gitlab"},
            {"id": "adv-2", "site": "gitlab"},
        ],
    )
    launched: list[int] = []
    launch_started = asyncio.Event()
    allow_launch = asyncio.Event()
    request_finished = asyncio.Event()

    async def fake_launch(_args, assignment, *, slot_id):
        launched.append(assignment.worker_id)
        launch_started.set()
        await allow_launch.wait()
        return SimpleNamespace(
            assignment=assignment,
            slot_id=slot_id,
            started_at="2026-08-11T00:00:00",
        )

    async def fake_finish(running):
        await request_finished.wait()
        return WorkerOutcome(
            assignment=running.assignment,
            returncode=0,
            timed_out=False,
            started_at=running.started_at,
            finished_at="2026-08-11T00:00:01",
            results=[{"task_id": running.assignment.task["id"]}],
        )

    monkeypatch.setattr(process_pool, "_launch_one_worker", fake_launch)
    monkeypatch.setattr(process_pool, "_finish_one_worker", fake_finish)
    monkeypatch.setattr(process_pool, "_write_pool_progress", lambda *a, **k: None)

    with bind_state_paths(
        args.out_dir,
        resume_pointer=args.out_dir / "last_run_state.json",
    ):
        _save_process_pool_state(
            args,
            assignments,
            [],
            status="running",
            reason="test",
        )
        scheduler = asyncio.create_task(_run_assignments(args, assignments))
        await launch_started.wait()
        pause_task = asyncio.create_task(asyncio.to_thread(request_pause, args.out_dir))
        await asyncio.sleep(0)
        assert not pause_task.done()
        allow_launch.set()
        await pause_task
        request_finished.set()
        run = await scheduler

    assert launched == [0]
    assert run.paused is True
    assert [outcome.assignment.worker_id for outcome in run.outcomes] == [0]


def test_process_pool_pause_checkpoint_keeps_success_metadata_inspection_only(
    tmp_path: Path,
) -> None:
    args = _pool_args(tmp_path, out_dir=tmp_path / "out", workers=2)
    args.out_dir.mkdir(parents=True)
    config = _config(
        tmp_path,
        [
            {
                "site_name": "gitlab",
                "site_url": "http://gitlab-1.test",
                "pvpo_cdp_url": "http://127.0.0.1:9222",
            },
            {
                "site_name": "gitlab",
                "site_url": "http://gitlab-2.test",
                "pvpo_cdp_url": "http://127.0.0.1:9223",
            },
        ],
    )
    assignments = _build_assignments(
        args,
        config,
        [
            {"id": "adv-1", "site": "gitlab"},
            {"id": "adv-2", "site": "gitlab"},
        ],
    )
    successful = WorkerOutcome(
        assignment=assignments[0],
        returncode=0,
        timed_out=False,
        started_at="2026-08-11T00:00:00",
        finished_at="2026-08-11T00:00:01",
        results=[{"task_id": "adv-1"}],
    )
    failed = WorkerOutcome(
        assignment=assignments[1],
        returncode=1,
        timed_out=False,
        started_at="2026-08-11T00:00:00",
        finished_at="2026-08-11T00:00:01",
        results=[],
        error="missing worker results",
    )
    result_path = assignments[0].state_dir / "phase_4" / "results.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps(successful.results), encoding="utf-8")
    _write_worker_status(
        assignments[0],
        slot_id=0,
        status="completed",
        returncode=0,
        timed_out=False,
        result_count=1,
        started_at=successful.started_at,
        finished_at=successful.finished_at,
    )

    with bind_state_paths(
        args.out_dir,
        resume_pointer=args.out_dir / "last_run_state.json",
    ):
        _save_process_pool_state(
            args,
            assignments,
            [successful, failed],
            status="running",
            reason="test",
        )
        request_pause(args.out_dir)
        acknowledge_pause(args.out_dir)
        resumed = _load_paused_outcomes(args, assignments)

    state = json.loads((args.out_dir / "pipeline_state.json").read_text())
    assert state["status"] == "paused"
    assert state["process_pool_completed_worker_ids"] == [0]
    assert state["process_pool_pending_worker_ids"] == [1]
    assert resumed == []
    assert not (args.out_dir / "phase_4" / "results.json").exists()


@pytest.mark.asyncio
async def test_process_pool_supervisor_persists_paused_root_without_canonical_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _pool_args(tmp_path, out_dir=tmp_path / "out", workers=1)
    config = _config(
        tmp_path,
        [
            {
                "site_name": "gitlab",
                "site_url": "http://gitlab.test",
                "pvpo_cdp_url": "http://127.0.0.1:9222",
            }
        ],
    )
    tasks = [{"id": "adv-1", "site": "gitlab"}]

    monkeypatch.setattr(process_pool, "_validate_source_state", lambda _source: None)
    monkeypatch.setattr(process_pool, "_materialize_reusable_phase_inputs", lambda *a: None)
    monkeypatch.setattr(process_pool, "load_benchmark_config", lambda _path: config)
    monkeypatch.setattr(process_pool, "_admitted_tasks_for_pool", lambda _args: tasks)
    monkeypatch.setattr(process_pool, "_write_pool_progress", lambda *a, **k: None)

    async def pause_before_launch(pool_args, assignments, **_kwargs):
        request_pause(pool_args.out_dir)
        return process_pool._AssignmentRun(outcomes=[], paused=True)

    monkeypatch.setattr(process_pool, "_run_assignments", pause_before_launch)

    rc = await run_process_pool(args)

    state = json.loads((args.out_dir / "pipeline_state.json").read_text())
    pointer = json.loads((args.out_dir / "last_run_state.json").read_text())
    assert rc == 0
    assert state["status"] == "paused"
    assert state["process_pool"] is True
    assert state["process_pool_pending_worker_ids"] == [0]
    assert pointer["status"] == "paused"
    assert not (args.out_dir / "pause_request.json").exists()
    assert not (args.out_dir / "phase_4" / "results.json").exists()


@pytest.mark.asyncio
async def test_process_pool_resume_reruns_every_worker_in_fresh_attempt_and_merges_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial_args = _pool_args(tmp_path, out_dir=tmp_path / "out", workers=2)
    resume_args = _pool_args(tmp_path, out_dir=tmp_path / "out", workers=2, resume=True)
    config = _config(
        tmp_path,
        [
            {
                "site_name": "gitlab",
                "site_url": "http://gitlab-1.test",
                "pvpo_cdp_url": "http://127.0.0.1:9222",
            },
            {
                "site_name": "gitlab",
                "site_url": "http://gitlab-2.test",
                "pvpo_cdp_url": "http://127.0.0.1:9223",
            },
        ],
    )
    tasks = [
        {"id": "adv-1", "site": "gitlab"},
        {"id": "adv-2", "site": "gitlab"},
    ]
    assignments = _build_assignments(initial_args, config, tasks)
    completed = WorkerOutcome(
        assignment=assignments[0],
        returncode=0,
        timed_out=False,
        started_at="2026-08-11T00:00:00",
        finished_at="2026-08-11T00:00:01",
        results=[{"task_id": "adv-1"}],
    )
    completed_path = assignments[0].state_dir / "phase_4" / "results.json"
    completed_path.parent.mkdir(parents=True)
    completed_path.write_text(json.dumps(completed.results), encoding="utf-8")
    _write_worker_status(
        assignments[0],
        slot_id=0,
        status="completed",
        returncode=0,
        timed_out=False,
        result_count=1,
        started_at=completed.started_at,
        finished_at=completed.finished_at,
    )
    with bind_state_paths(
        initial_args.out_dir,
        resume_pointer=initial_args.out_dir / "last_run_state.json",
    ):
        _save_process_pool_state(
            initial_args,
            assignments,
            [completed],
            status="running",
            reason="test",
        )
        request_pause(initial_args.out_dir)
        acknowledge_pause(initial_args.out_dir)

    monkeypatch.setattr(process_pool, "_validate_source_state", lambda _source: None)
    monkeypatch.setattr(process_pool, "_materialize_reusable_phase_inputs", lambda *a: None)
    monkeypatch.setattr(process_pool, "load_benchmark_config", lambda _path: config)
    monkeypatch.setattr(process_pool, "_admitted_tasks_for_pool", lambda _args: tasks)
    monkeypatch.setattr(process_pool, "_write_pool_progress", lambda *a, **k: None)
    scheduled: list[int] = []
    scheduled_paths: list[Path] = []

    async def finish_pending(_args, pending, **_kwargs):
        scheduled.extend(assignment.worker_id for assignment in pending)
        scheduled_paths.extend(assignment.state_dir for assignment in pending)
        return process_pool._AssignmentRun(
            outcomes=[
                WorkerOutcome(
                    assignment=assignment,
                    returncode=0,
                    timed_out=False,
                    started_at="2026-08-11T00:00:02",
                    finished_at="2026-08-11T00:00:03",
                    results=[{"task_id": assignment.task["id"]}],
                )
                for assignment in pending
            ],
            paused=False,
        )

    merged: list[list[int]] = []

    def merge(_args, _config, _tasks, outcomes):
        merged.append([outcome.assignment.worker_id for outcome in outcomes])
        return 0

    monkeypatch.setattr(process_pool, "_run_assignments", finish_pending)
    monkeypatch.setattr(process_pool, "_merge_outcomes", merge)

    rc = await run_process_pool(resume_args)

    assert rc == 0
    assert scheduled == [0, 1]
    assert merged == [[0, 1]]
    assert all(
        "process_pool_resume_workers/attempt_001" in str(state_dir) for state_dir in scheduled_paths
    )


def test_merge_outcomes_fails_closed_on_missing_result(tmp_path):
    args = _pool_args(tmp_path, out_dir=tmp_path / "out")
    config = _config(
        tmp_path,
        [
            {
                "site_name": "gitlab",
                "site_url": "http://127.0.0.1:8023",
                "pvpo_cdp_url": "http://127.0.0.1:9222",
            }
        ],
    )
    task = {"id": "adv-1", "site": "gitlab"}
    assignment = _build_assignments(args, config, [task])[0]
    outcome = WorkerOutcome(
        assignment=assignment,
        returncode=1,
        timed_out=False,
        started_at="2026-05-07T00:00:00",
        finished_at="2026-05-07T00:00:01",
        results=[],
        error="missing worker results",
    )

    rc = _merge_outcomes(args, config, [task], [outcome])

    assert rc == 1
    assert (tmp_path / "out" / "phase_4" / "process_pool_summary.json").exists()
    assert (tmp_path / "out" / "phase_4" / "results.partial.json").exists()
    assert (tmp_path / "out" / "phase_4" / "partial_manifest.json").exists()
    assert not (tmp_path / "out" / "phase_4" / "results.json").exists()


def test_merge_outcomes_writes_partial_results_without_canonical_results(tmp_path):
    args = _pool_args(tmp_path, out_dir=tmp_path / "out")
    config = _config(
        tmp_path,
        [
            {
                "site_name": "gitlab",
                "site_url": "http://127.0.0.1:8023",
                "pvpo_cdp_url": "http://127.0.0.1:9222",
            }
        ],
    )
    tasks = [{"id": "adv-1", "site": "gitlab"}, {"id": "adv-2", "site": "gitlab"}]
    assignments = _build_assignments(args, config, tasks)
    outcomes = [
        WorkerOutcome(
            assignment=assignments[0],
            returncode=0,
            timed_out=False,
            started_at="2026-05-07T00:00:00",
            finished_at="2026-05-07T00:00:01",
            results=[{"task_id": "adv-1", "outcome": "resistant"}],
            error=None,
        ),
        WorkerOutcome(
            assignment=assignments[1],
            returncode=1,
            timed_out=False,
            started_at="2026-05-07T00:00:00",
            finished_at="2026-05-07T00:00:01",
            results=[],
            error="missing worker results",
        ),
    ]

    rc = _merge_outcomes(args, config, tasks, outcomes)

    assert rc == 1
    partial = json.loads((tmp_path / "out" / "phase_4" / "results.partial.json").read_text())
    manifest = json.loads((tmp_path / "out" / "phase_4" / "partial_manifest.json").read_text())
    assert partial == [{"task_id": "adv-1", "outcome": "resistant"}]
    assert manifest["paper_eligible"] is False
    assert manifest["missing_task_ids"] == ["adv-2"]
    assert not (tmp_path / "out" / "phase_4" / "results.json").exists()


def test_repair_process_pool_partial_replaces_failed_retry_result(tmp_path):
    partial_run = tmp_path / "partial"
    retry_run = tmp_path / "retry"
    out_dir = tmp_path / "repaired"
    partial_phase4 = partial_run / "phase_4"
    retry_phase4 = retry_run / "phase_4"
    partial_phase4.mkdir(parents=True)
    retry_trace = retry_phase4 / "20260509_000000" / "adv-2"
    retry_trace.mkdir(parents=True)
    (retry_trace / "result.json").write_text(
        json.dumps({"task_id": "adv-2", "outcome": "complied"}),
        encoding="utf-8",
    )
    (partial_phase4 / "results.partial.json").write_text(
        json.dumps(
            [
                {"task_id": "adv-1", "final_status": "resistant"},
                {
                    "task_id": "adv-2",
                    "final_status": "error",
                    "trajectory_dir": str(partial_phase4 / "old" / "adv-2"),
                },
            ]
        ),
        encoding="utf-8",
    )
    (partial_phase4 / "partial_manifest.json").write_text(
        json.dumps(
            {
                "expected_tasks": 2,
                "errors": ["adv-2: worker exited 1"],
                "paper_eligible": False,
                "canonical_results_written": False,
            }
        ),
        encoding="utf-8",
    )
    (partial_phase4 / "progress.json").write_text(
        json.dumps({"status": "failed", "stage": "process_pool_merge_failed"}),
        encoding="utf-8",
    )
    (retry_phase4 / "results.json").write_text(
        json.dumps(
            [
                {
                    "task_id": "adv-2",
                    "final_status": "complied",
                    "outcome": "complied",
                    "trajectory_dir": str(retry_trace),
                }
            ]
        ),
        encoding="utf-8",
    )

    manifest = repair_process_pool_partial(
        partial_run=partial_run,
        retry_runs=[retry_run],
        out_dir=out_dir,
    )

    repaired = json.loads((out_dir / "phase_4" / "results.json").read_text())
    progress = json.loads((out_dir / "phase_4" / "progress.json").read_text())
    repair_manifest = json.loads(
        (out_dir / "phase_4" / "process_pool_repair_manifest.json").read_text()
    )

    assert manifest["replaced_task_ids"] == ["adv-2"]
    assert [row["task_id"] for row in repaired] == ["adv-1", "adv-2"]
    assert repaired[1]["final_status"] == "complied"
    assert repaired[1]["process_pool_repair"]["repair_reason"]
    assert repaired[1]["trajectory_dir"].startswith(
        str((out_dir / "phase_4" / "process_pool_tasks" / "retry__adv-2").resolve())
    )
    assert progress["status"] == "complete"
    assert progress["stage"] == "complete_repaired"
    assert progress["postprocess_failed_tasks"] == 0
    assert repair_manifest["paper_eligible"] == "operator_review_required"


def test_load_worker_results_salvages_eval_awareness_checkpoint(tmp_path):
    args = _pool_args(tmp_path, out_dir=tmp_path / "out")
    config = _config(
        tmp_path,
        [
            {
                "site_name": "gitlab",
                "site_url": "http://127.0.0.1:8023",
                "pvpo_cdp_url": "http://127.0.0.1:9222",
            }
        ],
    )
    task = {"id": "adv-1", "site": "gitlab"}
    assignment = _build_assignments(args, config, [task])[0]
    trace_dir = assignment.state_dir / "phase_4" / "20260508_000000" / "adv-1"
    trace_dir.mkdir(parents=True)
    baseline = {
        "task_id": "adv-1",
        "outcome": "refused_or_ignored",
        "encounter": {"max_coverage": 1.0},
        "transcript_purpose_classification": "Capability Evaluation",
        "transcript_purpose_is_aware": True,
    }
    variant = {
        "task_id": "adv-1",
        "outcome": "refused_or_ignored",
        "encounter": {"max_coverage": 1.0},
        "strategy": "eval_awareness_iterator",
        "transcript_purpose_classification": "Real",
        "transcript_purpose_is_aware": False,
    }
    (trace_dir / "result.json").write_text(json.dumps(baseline), encoding="utf-8")
    (trace_dir / "eval_awareness_iterator_checkpoint.json").write_text(
        json.dumps(
            {
                "algorithm": "eval-awareness-iterator",
                "version": "test",
                "max_iterations": 3,
                "baseline_attempt": {"iteration": 0, "kind": "baseline"},
                "iterations": [
                    {
                        "iteration": 1,
                        "status": "evaluated",
                        "result": variant,
                        "tp_transition": {"direction": "improved"},
                    },
                    {"iteration": 2, "status": "started"},
                ],
            }
        ),
        encoding="utf-8",
    )

    results, error = _load_worker_results(assignment)

    assert error and error.startswith("salvaged missing worker results")
    assert len(results) == 1
    assert results[0]["final_status"] == "resistant"
    assert results[0]["eval_awareness_iterator"]["salvaged"] is True
    assert results[0]["eval_awareness_iterator"]["variant_results"] == [variant]


def test_load_worker_results_salvages_newest_eval_awareness_checkpoint(tmp_path):
    args = _pool_args(tmp_path, out_dir=tmp_path / "out")
    config = _config(
        tmp_path,
        [
            {
                "site_name": "gitlab",
                "site_url": "http://127.0.0.1:8023",
                "pvpo_cdp_url": "http://127.0.0.1:9222",
            }
        ],
    )
    task = {"id": "adv-1", "site": "gitlab"}
    assignment = _build_assignments(args, config, [task])[0]
    older_trace = assignment.state_dir / "phase_4" / "20260508_000000" / "adv-1"
    newer_trace = assignment.state_dir / "phase_4" / "20260508_000001" / "adv-1"
    older_trace.mkdir(parents=True)
    newer_trace.mkdir(parents=True)
    baseline = {
        "task_id": "adv-1",
        "outcome": "refused_or_ignored",
        "encounter": {"max_coverage": 1.0},
    }
    older_variant = {
        "task_id": "adv-1",
        "outcome": "complied",
        "encounter": {"max_coverage": 1.0},
        "strategy": "eval_awareness_iterator",
    }
    newer_variant = {
        "task_id": "adv-1",
        "outcome": "refused_or_ignored",
        "encounter": {"max_coverage": 1.0},
        "strategy": "eval_awareness_iterator",
    }
    for trace, variant in ((older_trace, older_variant), (newer_trace, newer_variant)):
        (trace / "result.json").write_text(json.dumps(baseline), encoding="utf-8")
        (trace / "eval_awareness_iterator_checkpoint.json").write_text(
            json.dumps(
                {
                    "algorithm": "eval-awareness-iterator",
                    "version": "test",
                    "max_iterations": 3,
                    "baseline_attempt": {"iteration": 0, "kind": "baseline"},
                    "iterations": [
                        {
                            "iteration": 1,
                            "status": "evaluated",
                            "result": variant,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    results, error = _load_worker_results(assignment)

    assert error and error.startswith("salvaged missing worker results")
    assert results[0]["final_status"] == "resistant"
    assert results[0]["eval_awareness_iterator"]["variant_results"] == [newer_variant]


def test_merge_outcomes_fails_closed_for_salvaged_timed_out_worker(tmp_path):
    args = _pool_args(tmp_path, out_dir=tmp_path / "out")
    config = _config(
        tmp_path,
        [
            {
                "site_name": "gitlab",
                "site_url": "http://127.0.0.1:8023",
                "pvpo_cdp_url": "http://127.0.0.1:9222",
            }
        ],
    )
    task = {"id": "adv-1", "site": "gitlab"}
    assignment = _build_assignments(args, config, [task])[0]
    outcome = WorkerOutcome(
        assignment=assignment,
        returncode=-9,
        timed_out=True,
        started_at="2026-05-08T00:00:00",
        finished_at="2026-05-08T00:40:00",
        results=[
            {
                "task_id": "adv-1",
                "outcome": "refused_or_ignored",
                "final_status": "resistant",
                "encounter": {"max_coverage": 1.0},
                "eval_awareness_iterator": {
                    "salvaged": True,
                    "salvage_reason": "process_pool_worker_timeout",
                },
            }
        ],
        error=(
            "salvaged missing worker results from iterator checkpoint: "
            "/tmp/worker/state/phase_4/results.json"
        ),
    )

    rc = _merge_outcomes(args, config, [task], [outcome])

    assert rc == 1
    assert (tmp_path / "out" / "phase_4" / "results.partial.json").exists()
    assert (tmp_path / "out" / "phase_4" / "partial_manifest.json").exists()
    assert not (tmp_path / "out" / "phase_4" / "results.json").exists()


def test_process_pool_copy_and_rewrite_preserves_colliding_trace_dirs(tmp_path):
    args = _pool_args(tmp_path, out_dir=tmp_path / "out")
    config = _config(
        tmp_path,
        [
            {
                "site_name": "gitlab",
                "site_url": "http://127.0.0.1:8023",
                "pvpo_cdp_url": "http://127.0.0.1:9222",
            }
        ],
    )
    assignment = _build_assignments(args, config, [{"id": "adv-1", "site": "gitlab"}])[0]
    first = assignment.state_dir / "phase_4" / "20260508_000000" / "adv-1"
    second = assignment.state_dir / "phase_4" / "20260508_000001" / "adv-1"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "marker.txt").write_text("first", encoding="utf-8")
    (second / "marker.txt").write_text("second", encoding="utf-8")
    task_root = args.out_dir / "phase_4" / "process_pool_tasks"
    task_root.mkdir(parents=True)

    replacements = _copy_worker_task_artifacts(assignment, task_root)
    rewritten = _rewrite_worker_paths(
        {"baseline": str(first), "rerun": str(second)},
        assignment,
        task_root,
        replacements,
    )

    assert (Path(rewritten["baseline"]) / "marker.txt").read_text(encoding="utf-8") == "first"
    assert (Path(rewritten["rerun"]) / "marker.txt").read_text(encoding="utf-8") == "second"
    assert rewritten["baseline"] != rewritten["rerun"]


def test_rewrite_worker_paths_handles_absolute_and_relative_paths(tmp_path):
    args = _pool_args(tmp_path, out_dir=tmp_path / "out")
    config = _config(
        tmp_path,
        [
            {
                "site_name": "gitlab",
                "site_url": "http://127.0.0.1:8023",
                "pvpo_cdp_url": "http://127.0.0.1:9222",
            }
        ],
    )
    assignment = _build_assignments(args, config, [{"id": "adv-1", "site": "gitlab"}])[0]
    worker_root = assignment.state_dir / "phase_4" / "20260507_000000"
    worker_root.mkdir(parents=True)
    task_root = args.out_dir / "phase_4" / "process_pool_tasks"

    payload = {
        "trajectory_dir": str(worker_root.resolve() / "adv-1"),
        "relative_dir": str(worker_root / "adv-1"),
    }

    rewritten = _rewrite_worker_paths(payload, assignment, task_root)

    assert rewritten["trajectory_dir"] == str(task_root.resolve() / "adv-1")
    assert rewritten["relative_dir"] == str(task_root / "adv-1")


def test_process_pool_active_worker_payload_surfaces_live_paths(tmp_path):
    args = _pool_args(tmp_path, out_dir=tmp_path / "out")
    config = _config(
        tmp_path,
        [
            {
                "site_name": "gitlab",
                "site_url": "http://127.0.0.1:8023",
                "pvpo_cdp_url": "http://127.0.0.1:9222",
            }
        ],
    )
    assignment = _build_assignments(args, config, [{"id": "adv-1", "site": "gitlab"}])[0]
    task_root = assignment.state_dir / "phase_4" / "20260507_000000"
    trace_dir = task_root / "adv-1"
    trace_dir.mkdir(parents=True)
    (assignment.state_dir / "phase_4" / "progress.json").write_text(
        json.dumps(
            {
                "status": "running",
                "stage": "initial_evaluation",
                "task_dir_root": str(task_root),
                "updated_at": "2026-05-07T00:00:00",
            }
        ),
        encoding="utf-8",
    )
    (trace_dir / "agentlab_sidecar_status.json").write_text(
        json.dumps(
            {
                "status": "sidecar_running",
                "current_phase": "browser_step",
                "current_step": 3,
                "last_url": "http://gitlab.test/issues/1",
                "last_screenshot": "screenshots/step_3.png",
            }
        ),
        encoding="utf-8",
    )

    _write_worker_status(
        assignment,
        slot_id=2,
        status="running",
        pid=1234,
        started_at="2026-05-07T00:00:00",
    )
    payload = _active_worker_progress_payload(assignment, 2)

    assert payload["worker_id"] == assignment.worker_id
    assert payload["slot_id"] == 2
    assert payload["state_dir"] == str(assignment.state_dir)
    assert payload["task_trace_dir"] == str(trace_dir)
    assert payload["sidecar_status"] == "sidecar_running"
    assert payload["current_phase"] == "browser_step"
    assert payload["current_step"] == 3
    assert payload["last_screenshot"] == "screenshots/step_3.png"
