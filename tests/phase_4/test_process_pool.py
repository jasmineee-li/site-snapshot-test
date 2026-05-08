import json
from pathlib import Path

from worldsim.config import BenchmarkConfig
from worldsim.phase_4.process_pool import (
    ProcessPoolArgs,
    WorkerOutcome,
    _active_worker_progress_payload,
    _build_assignments,
    _copy_worker_task_artifacts,
    _load_worker_results,
    _merge_outcomes,
    _rewrite_worker_paths,
    _worker_command,
    _worker_timeout_seconds,
    _write_worker_status,
)


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
