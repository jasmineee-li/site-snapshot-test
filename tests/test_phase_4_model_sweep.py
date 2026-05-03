from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "phase4_model_sweeps" / "phase4_20260501_expanded.json"
NO_TITLE_CONFIG_PATH = (
    REPO_ROOT / "configs" / "phase4_model_sweeps" / "phase4_20260502_no_title.json"
)
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_phase_4_model_sweep.py"


def _load_sweep_module():
    spec = importlib.util.spec_from_file_location("run_phase_4_model_sweep", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_config_parses_completed_run_and_remaining_model_order() -> None:
    sweep = _load_sweep_module()

    config = sweep.load_sweep_config(CONFIG_PATH)

    assert [run.key for run in config.completed_runs] == ["gpt52"]
    assert config.completed_runs[0].service_tier == "priority"
    assert [model.key for model in config.models] == [
        "sonnet46",
        "opus47",
        "glm5",
        "gemini25pro",
        "kimi-k25",
    ]
    assert [model.key for model in sweep.select_models(config, start_at="glm5")] == [
        "glm5",
        "gemini25pro",
        "kimi-k25",
    ]
    assert [model.key for model in sweep.select_models(config, only=["kimi-k25", "glm5"])] == [
        "glm5",
        "kimi-k25",
    ]


def test_no_title_config_uses_fresh_body_description_cohort() -> None:
    sweep = _load_sweep_module()

    config = sweep.load_sweep_config(NO_TITLE_CONFIG_PATH)

    assert config.source_run_dir == "logs/no_title_carrier_regen_20260502T0857Z"
    assert config.benchmark == "/home/ubuntu/vendors/webarena-verified"
    assert config.stale_resume_budget == 2
    assert [run.key for run in config.completed_runs] == ["gpt52"]
    assert [model.key for model in config.models] == [
        "sonnet46",
        "opus47",
        "glm5",
        "gemini25pro",
        "kimi-k25",
    ]


def test_command_generation_preserves_phase4_contract() -> None:
    sweep = _load_sweep_module()
    config = sweep.load_sweep_config(CONFIG_PATH)
    model = sweep.ModelRun(
        key="gpt52-priority",
        provider="openai",
        model="gpt-5.2",
        service_tier="priority",
        retry_budget=5,
    )

    command = sweep.build_phase4_command(
        config,
        model,
        "logs/phase4_deadlines_gpt52_16ps_20260501T000000Z",
    )

    assert "SOURCE=logs/task_bank_live_verify_phase0_path_repair_20260430T1900Z" in command
    assert 'cp -a "$SOURCE/phase_0c" "$RUN/"' in command
    assert 'cp -a "$SOURCE/phase_2" "$RUN/"' in command
    assert 'cp -a "$SOURCE/phase_3" "$RUN/"' in command
    assert "--instances instances.scale.json" in command
    assert "--sites gitlab,reddit" in command
    assert "--task-origin new_task" in command
    assert "--max-tasks-per-site 16" in command
    assert "--agent-provider openai" in command
    assert "--agent-model gpt-5.2" in command
    assert "--agent-service-tier priority" in command
    assert "--agent-llm-timeout 240" in command
    assert "--agent-step-timeout 300" in command
    assert "--agent-task-timeout 900" in command
    assert "--sandbox-model claude-sonnet-4-6" in command
    assert 'tee "$RUN/phase_4/summary.txt"' in command
    assert 'variant_audit.txt" 2>&1 || true' in command


def test_no_title_command_includes_remote_benchmark_root() -> None:
    sweep = _load_sweep_module()
    config = sweep.load_sweep_config(NO_TITLE_CONFIG_PATH)

    command = sweep.build_phase4_command(
        config,
        config.models[0],
        "logs/phase4_no_title_sonnet46_16ps_20260502T000000Z",
    )

    assert "SOURCE=logs/no_title_carrier_regen_20260502T0857Z" in command
    assert "--benchmark /home/ubuntu/vendors/webarena-verified" in command


def test_resume_command_uses_same_phase4_contract_without_recopied_source() -> None:
    sweep = _load_sweep_module()
    config = sweep.load_sweep_config(NO_TITLE_CONFIG_PATH)

    command = sweep.build_phase4_resume_command(
        config,
        config.models[0],
        "logs/phase4_no_title_sonnet46_16ps_20260502T000000Z",
    )

    assert "worldsim.main resume" in command
    assert "--benchmark /home/ubuntu/vendors/webarena-verified" in command
    assert "--instances instances.scale.json" in command
    assert "--sites gitlab,reddit" in command
    assert "--task-origin new_task" in command
    assert "--max-tasks-per-site 16" in command
    assert "--agent-provider anthropic" in command
    assert "--agent-model claude-sonnet-4-6" in command
    assert "--agent-llm-timeout 240" in command
    assert "--agent-step-timeout 300" in command
    assert "--agent-task-timeout 900" in command
    assert 'cp -a "$SOURCE/phase_2" "$RUN/"' not in command
    assert 'tee "$RUN/phase_4/summary.txt"' in command


def test_remote_job_args_use_wrappers_and_expected_output() -> None:
    sweep = _load_sweep_module()
    config = sweep.load_sweep_config(CONFIG_PATH)
    model = config.models[0]
    run_dir = "logs/phase4_deadlines_sonnet46_16ps_20260501T000000Z"
    command = sweep.build_phase4_command(config, model, run_dir)

    args = sweep.build_remote_job_start_args(
        config,
        model,
        run_dir=run_dir,
        command_body=command,
    )

    assert args[:5] == [
        "scripts/remote_job_start.sh",
        "--host-config",
        "configs/benchmark_hosts/r5.yaml",
        "--remote-dir",
        "/home/ubuntu/browser-sim",
    ]
    assert "--expected-output" in args
    assert f"{run_dir}/phase_4/results.json" in args
    assert args[-3:] == ["bash", "-lc", command]


def test_remote_resume_job_args_use_wrappers_and_expected_output() -> None:
    sweep = _load_sweep_module()
    config = sweep.load_sweep_config(NO_TITLE_CONFIG_PATH)
    model = config.models[0]
    run_dir = "logs/phase4_no_title_sonnet46_16ps_20260502T000000Z"
    command = sweep.build_phase4_resume_command(config, model, run_dir)

    args = sweep.build_remote_resume_job_start_args(
        config,
        model,
        run_dir=run_dir,
        command_body=command,
        resume_index=1,
    )

    assert args[:5] == [
        "scripts/remote_job_start.sh",
        "--host-config",
        "configs/benchmark_hosts/r5.yaml",
        "--remote-dir",
        "/home/ubuntu/browser-sim",
    ]
    assert "--expected-output" in args
    assert f"{run_dir}/phase_4/results.json" in args
    assert "phase4-deadlines-sonnet46-16ps-resume1" in args
    assert args[-3:] == ["bash", "-lc", command]


def test_run_dir_slug_sanitization_and_attempt_suffix() -> None:
    sweep = _load_sweep_module()
    config = sweep.load_sweep_config(CONFIG_PATH)
    model = sweep.ModelRun(
        key="Kimi/K 2.5!",
        provider="openrouter",
        model="moonshotai/kimi-k2.5",
        retry_budget=3,
    )

    assert sweep.sanitize_slug(model.key) == "kimi-k-2.5"
    assert sweep.run_dir_for_model(config, model, timestamp="20260501T000000") == (
        "logs/phase4_deadlines_kimi-k-2.5_16ps_20260501T000000Z"
    )
    assert sweep.run_dir_for_model(config, model, timestamp="20260501T000000", attempt=2) == (
        "logs/phase4_deadlines_kimi-k-2.5_16ps_20260501T000000Z_try2"
    )


def test_retry_budget_and_completed_run_serialize_in_dry_run() -> None:
    sweep = _load_sweep_module()
    config = sweep.load_sweep_config(CONFIG_PATH)

    payload = sweep.render_dry_run(config, [config.models[-1]], "20260501T010203")

    assert payload["completed_runs"][0]["key"] == "gpt52"
    assert payload["completed_runs"][0]["service_tier"] == "priority"
    assert payload["runs"][0]["key"] == "kimi-k25"
    assert payload["runs"][0]["retry_budget"] == 3
    assert payload["runs"][0]["run_dir"] == ("logs/phase4_deadlines_kimi-k25_16ps_20260501T010203Z")


def test_dry_run_does_not_call_remote_wrappers(monkeypatch, capsys) -> None:
    sweep = _load_sweep_module()

    def fail_run_checked(_args):
        raise AssertionError("dry run must not call wrapper subprocesses")

    monkeypatch.setattr(sweep, "run_checked", fail_run_checked)

    rc = sweep.main(
        [
            "--config",
            str(CONFIG_PATH),
            "--dry-run",
            "--only",
            "sonnet46",
            "--timestamp",
            "20260501T010203",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert [run["key"] for run in payload["runs"]] == ["sonnet46"]
    assert payload["runs"][0]["remote_job_start_args"][0] == "scripts/remote_job_start.sh"


def test_live_run_refuses_untracked_files_before_sync(monkeypatch, tmp_path, capsys) -> None:
    sweep = _load_sweep_module()

    monkeypatch.setattr(sweep, "tracked_tree_is_dirty", lambda: False)
    monkeypatch.setattr(sweep, "untracked_files", lambda: ["scripts/smoke_local.py"])

    def fail_run_checked(_args):
        raise AssertionError("runner must fail before remote wrappers")

    monkeypatch.setattr(sweep, "run_checked", fail_run_checked)

    rc = sweep.main(
        [
            "--config",
            str(CONFIG_PATH),
            "--only",
            "sonnet46",
            "--state-dir",
            str(tmp_path / "state"),
        ]
    )

    assert rc == 2
    assert "untracked files" in capsys.readouterr().err


def test_phase4_status_line_counts_are_parsed() -> None:
    sweep = _load_sweep_module()

    parsed = sweep.parse_status_output(
        "\n".join(
            [
                "status: exited",
                "returncode: 0",
                (
                    "phase4_results: logs/run/phase_4/results.json total=32 "
                    "sites=gitlab=16,reddit=16 "
                    "final_status=complied=26,inconclusive=1,resistant=2,success_on_variant=3"
                ),
            ]
        )
    )

    assert parsed["total"] == 32
    assert parsed["site_counts"] == {"gitlab": 16, "reddit": 16}
    assert parsed["final_status_counts"] == {
        "complied": 26,
        "inconclusive": 1,
        "resistant": 2,
        "success_on_variant": 3,
    }


def test_phase4_progress_line_is_parsed() -> None:
    sweep = _load_sweep_module()

    parsed = sweep.parse_status_output(
        "\n".join(
            [
                "status: running",
                (
                    "phase4_progress: present logs/run/phase_4/progress.json "
                    "status=running stage=initial_evaluation initial=30/32 "
                    "postprocessed=0/32 age_seconds=901 "
                    "updated_at=2026-05-02T12:00:22.308663"
                ),
                "log_progress: latest write 12s ago",
            ]
        )
    )

    assert parsed["phase4_progress_status"] == "running"
    assert parsed["phase4_progress_stage"] == "initial_evaluation"
    assert parsed["phase4_completed_initial_tasks"] == 30
    assert parsed["phase4_postprocessed_tasks"] == 0
    assert parsed["phase4_total_tasks"] == 32
    assert parsed["phase4_progress_quiet_seconds"] == 901
    assert parsed["log_quiet_seconds"] == 12


def test_monitor_uses_phase4_progress_staleness_before_noisy_logs(monkeypatch) -> None:
    sweep = _load_sweep_module()
    config = sweep.load_sweep_config(NO_TITLE_CONFIG_PATH)

    monkeypatch.setattr(sweep.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        sweep,
        "status_for_job",
        lambda _config, _job_id: {
            "status": "running",
            "phase4_progress": (
                "phase4_progress: present logs/run/phase_4/progress.json "
                "status=running stage=initial_evaluation initial=30/32 "
                "postprocessed=0/32 age_seconds=901 updated_at=2026-05-02T12:00:22"
            ),
            "phase4_progress_quiet_seconds": 901,
            "log_progress": "log_progress: latest write 10s ago",
            "log_quiet_seconds": 10,
        },
    )
    monkeypatch.setattr(sweep, "tail_job", lambda _config, _job_id: "watchdog noise")

    status = sweep.monitor_job(
        config,
        "job-stale-progress",
        initial_poll_seconds=300,
        poll_seconds=600,
        stale_seconds=900,
    )

    assert status["status"] == "attention_required"
    assert status["failure_class"] == "stale_phase4_progress"
    assert status["tail"] == "watchdog noise"


def test_invalid_start_at_and_only_raise_clear_errors() -> None:
    sweep = _load_sweep_module()
    config = sweep.load_sweep_config(CONFIG_PATH)

    with pytest.raises(ValueError, match="--start-at"):
        sweep.select_models(config, start_at="missing")
    with pytest.raises(ValueError, match="--only"):
        sweep.select_models(config, only=["missing"])
