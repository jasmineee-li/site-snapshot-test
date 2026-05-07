from pathlib import Path

import pytest

from worldsim.config import BenchmarkConfig
from worldsim.phase_4.process_pool import (
    ProcessPoolArgs,
    WorkerOutcome,
    _build_assignments,
    _merge_outcomes,
    _rewrite_worker_paths,
    _worker_command,
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


def test_build_assignments_rejects_duplicate_pvpo(tmp_path):
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

    with pytest.raises(SystemExit, match="duplicate pvpo_cdp_url"):
        _build_assignments(args, config, [{"id": "adv-1", "site": "gitlab"}])


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
    assert not (tmp_path / "out" / "phase_4" / "results.json").exists()


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
