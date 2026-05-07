#!/usr/bin/env python3
"""Run a sequential Phase 4 model sweep through registered r5 jobs."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def log_event(message: str) -> None:
    print(f"[phase4-sweep] {message}", flush=True)


@dataclass(frozen=True)
class CompletedRun:
    key: str
    provider: str
    model: str
    run_dir: str
    job_id: str | None = None
    service_tier: str | None = None


@dataclass(frozen=True)
class ModelRun:
    key: str
    provider: str
    model: str
    retry_budget: int
    service_tier: str | None = None


@dataclass(frozen=True)
class SweepConfig:
    sweep_name: str
    host_config: str
    remote_dir: str
    source_run_dir: str
    run_dir_template: str
    instances: str
    sites: str
    task_origin: str
    max_tasks_per_site: int
    agent_llm_timeout: int
    agent_step_timeout: int
    agent_task_timeout: int
    sandbox_model: str
    inspect_limit: int
    benchmark: str | None = None
    stale_resume_budget: int = 0
    completed_runs: list[CompletedRun] = field(default_factory=list)
    models: list[ModelRun] = field(default_factory=list)


def _require_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _require_int(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _optional_str(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be null or a non-empty string")
    return value


def load_sweep_config(path: Path) -> SweepConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("sweep config must be a JSON object")
    common = data.get("common")
    if not isinstance(common, dict):
        raise ValueError("sweep config must contain a common object")
    completed = data.get("completed_runs", [])
    models = data.get("models", [])
    if not isinstance(completed, list):
        raise ValueError("completed_runs must be a list")
    if not isinstance(models, list) or not models:
        raise ValueError("models must be a non-empty list")
    completed_runs = [
        CompletedRun(
            key=_require_str(item, "key"),
            provider=_require_str(item, "provider"),
            model=_require_str(item, "model"),
            service_tier=_optional_str(item, "service_tier"),
            run_dir=_require_str(item, "run_dir"),
            job_id=_optional_str(item, "job_id"),
        )
        for item in completed
        if isinstance(item, dict)
    ]
    model_runs = [
        ModelRun(
            key=_require_str(item, "key"),
            provider=_require_str(item, "provider"),
            model=_require_str(item, "model"),
            service_tier=_optional_str(item, "service_tier"),
            retry_budget=_require_int(item, "retry_budget"),
        )
        for item in models
        if isinstance(item, dict)
    ]
    if len(model_runs) != len(models):
        raise ValueError("each model entry must be an object")
    keys = [model.key for model in model_runs]
    if len(set(keys)) != len(keys):
        raise ValueError(f"model keys must be unique: {keys}")
    return SweepConfig(
        sweep_name=_require_str(data, "sweep_name"),
        host_config=_require_str(common, "host_config"),
        remote_dir=_require_str(common, "remote_dir"),
        source_run_dir=_require_str(common, "source_run_dir"),
        run_dir_template=_require_str(common, "run_dir_template"),
        instances=_require_str(common, "instances"),
        sites=_require_str(common, "sites"),
        task_origin=_require_str(common, "task_origin"),
        max_tasks_per_site=_require_int(common, "max_tasks_per_site"),
        agent_llm_timeout=_require_int(common, "agent_llm_timeout"),
        agent_step_timeout=_require_int(common, "agent_step_timeout"),
        agent_task_timeout=_require_int(common, "agent_task_timeout"),
        sandbox_model=_require_str(common, "sandbox_model"),
        inspect_limit=_require_int(common, "inspect_limit"),
        benchmark=_optional_str(common, "benchmark"),
        stale_resume_budget=int(common.get("stale_resume_budget") or 0),
        completed_runs=completed_runs,
        models=model_runs,
    )


def utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S")


def sanitize_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-").lower()
    if not slug:
        raise ValueError("slug cannot be empty")
    return slug


def run_dir_for_model(
    config: SweepConfig,
    model: ModelRun,
    *,
    timestamp: str,
    attempt: int = 1,
) -> str:
    suffix = "" if attempt == 1 else f"_try{attempt}"
    return config.run_dir_template.format(
        key=sanitize_slug(model.key),
        timestamp=timestamp,
        attempt=attempt,
        attempt_suffix=suffix,
    )


def build_phase4_command(config: SweepConfig, model: ModelRun, run_dir: str) -> str:
    phase4_args = [
        "uv",
        "run",
        "python",
        "-m",
        "worldsim.main",
        "phase",
        "4",
    ]
    if config.benchmark:
        phase4_args.extend(["--benchmark", config.benchmark])
    phase4_args.extend(
        [
            "--instances",
            config.instances,
            "--sites",
            config.sites,
            "--task-origin",
            config.task_origin,
            "--max-tasks-per-site",
            str(config.max_tasks_per_site),
            "--agent-provider",
            model.provider,
            "--agent-model",
            model.model,
            "--agent-llm-timeout",
            str(config.agent_llm_timeout),
            "--agent-step-timeout",
            str(config.agent_step_timeout),
            "--agent-task-timeout",
            str(config.agent_task_timeout),
            "--sandbox-model",
            config.sandbox_model,
        ]
    )
    if model.service_tier:
        phase4_args.extend(["--agent-service-tier", model.service_tier])
    lines = [
        "set -euo pipefail",
        f"SOURCE={shlex.quote(config.source_run_dir)}",
        f"RUN={shlex.quote(run_dir)}",
        'test -d "$SOURCE/phase_0c"',
        'test -f "$SOURCE/phase_2/adversarial_tasks.json"',
        'test -d "$SOURCE/phase_3"',
        'mkdir -p "$RUN"',
        'cp -a "$SOURCE/phase_0c" "$RUN/"',
        'cp -a "$SOURCE/phase_2" "$RUN/"',
        'cp -a "$SOURCE/phase_3" "$RUN/"',
        'export WORLDSIM_STATE_DIR="$RUN"',
        shlex.join(phase4_args),
        *_post_phase4_report_lines(config),
    ]
    return "\n".join(lines)


def _post_phase4_report_lines(config: SweepConfig) -> list[str]:
    return [
        'mkdir -p "$RUN/phase_4"',
        (
            "uv run python scripts/summarize_phase_4_results.py "
            f'"$RUN/phase_4/results.json" --inspect-limit {config.inspect_limit} '
            '| tee "$RUN/phase_4/summary.txt"'
        ),
        (
            '{ uv run python scripts/audit_phase_4_variants.py "$RUN"; } '
            '> "$RUN/phase_4/variant_audit.txt" 2>&1 || true'
        ),
    ]


def _phase4_resume_args(config: SweepConfig, model: ModelRun) -> list[str]:
    args = [
        "uv",
        "run",
        "python",
        "-m",
        "worldsim.main",
        "resume",
    ]
    if config.benchmark:
        args.extend(["--benchmark", config.benchmark])
    args.extend(
        [
            "--instances",
            config.instances,
            "--sites",
            config.sites,
            "--task-origin",
            config.task_origin,
            "--max-tasks-per-site",
            str(config.max_tasks_per_site),
            "--agent-provider",
            model.provider,
            "--agent-model",
            model.model,
            "--agent-llm-timeout",
            str(config.agent_llm_timeout),
            "--agent-step-timeout",
            str(config.agent_step_timeout),
            "--agent-task-timeout",
            str(config.agent_task_timeout),
            "--sandbox-model",
            config.sandbox_model,
        ]
    )
    if model.service_tier:
        args.extend(["--agent-service-tier", model.service_tier])
    return args


def build_phase4_resume_command(config: SweepConfig, model: ModelRun, run_dir: str) -> str:
    lines = [
        "set -euo pipefail",
        f"RUN={shlex.quote(run_dir)}",
        'test -f "$RUN/phase_2/adversarial_tasks.json"',
        'test -d "$RUN/phase_3"',
        'export WORLDSIM_STATE_DIR="$RUN"',
        shlex.join(_phase4_resume_args(config, model)),
        *_post_phase4_report_lines(config),
    ]
    return "\n".join(lines)


def build_remote_job_start_args(
    config: SweepConfig,
    model: ModelRun,
    *,
    run_dir: str,
    command_body: str,
) -> list[str]:
    return [
        "scripts/remote_job_start.sh",
        "--host-config",
        config.host_config,
        "--remote-dir",
        config.remote_dir,
        "--name",
        f"phase4-deadlines-{sanitize_slug(model.key)}-16ps",
        "--expected-output",
        f"{run_dir}/phase_4/results.json",
        "--",
        "bash",
        "-lc",
        command_body,
    ]


def build_remote_resume_job_start_args(
    config: SweepConfig,
    model: ModelRun,
    *,
    run_dir: str,
    command_body: str,
    resume_index: int,
) -> list[str]:
    return [
        "scripts/remote_job_start.sh",
        "--host-config",
        config.host_config,
        "--remote-dir",
        config.remote_dir,
        "--name",
        f"phase4-deadlines-{sanitize_slug(model.key)}-16ps-resume{resume_index}",
        "--expected-output",
        f"{run_dir}/phase_4/results.json",
        "--",
        "bash",
        "-lc",
        command_body,
    ]


def select_models(
    config: SweepConfig,
    *,
    start_at: str | None = None,
    only: list[str] | None = None,
) -> list[ModelRun]:
    models = config.models
    if start_at:
        keys = [model.key for model in models]
        if start_at not in keys:
            raise ValueError(f"--start-at {start_at!r} not in model keys: {keys}")
        models = models[keys.index(start_at) :]
    if only:
        wanted = set(only)
        missing = wanted - {model.key for model in models}
        if missing:
            raise ValueError(f"--only contains unknown model key(s): {sorted(missing)}")
        models = [model for model in models if model.key in wanted]
    return models


def tracked_tree_is_dirty() -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return bool(completed.stdout.strip())


def untracked_files() -> list[str]:
    completed = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in completed.stdout.splitlines() if line.strip()]


def run_checked(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )


def local_git_stamp(config: SweepConfig) -> dict[str, Any]:
    def git_value(*args: str) -> str | None:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            return None
        return completed.stdout.strip()

    return {
        "recorded_at": datetime.now(UTC).isoformat(),
        "host_config": config.host_config,
        "remote_dir": config.remote_dir,
        "local_git": {
            "sha": git_value("rev-parse", "HEAD"),
            "branch": git_value("rev-parse", "--abbrev-ref", "HEAD"),
            "tracked_dirty": tracked_tree_is_dirty(),
            "untracked_files": untracked_files(),
        },
    }


def running_remote_jobs(config: SweepConfig) -> list[str]:
    completed = run_checked(
        [
            "scripts/remote_job_list.sh",
            "--host-config",
            config.host_config,
            "--remote-dir",
            config.remote_dir,
            "--limit",
            "200",
        ]
    )
    running: list[str] = []
    for line in completed.stdout.splitlines():
        if " running " in f" {line} ":
            running.append(line)
    return running


def require_no_running_remote_jobs(config: SweepConfig) -> list[str]:
    active = running_remote_jobs(config)
    if active:
        raise RuntimeError("remote jobs are already running:\n" + "\n".join(active))
    return active


def parse_job_id(stdout: str) -> str:
    for line in stdout.splitlines():
        if line.startswith("job_id="):
            return line.split("=", 1)[1].strip()
    raise ValueError(f"remote_job_start output did not contain job_id: {stdout}")


def parse_status_output(stdout: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {"raw": stdout}
    for line in stdout.splitlines():
        if line.startswith("status:"):
            parsed["status"] = line.split(":", 1)[1].strip()
        elif line.startswith("returncode:"):
            value = line.split(":", 1)[1].strip()
            parsed["returncode"] = int(value) if value.lstrip("-").isdigit() else value
        elif line.startswith("phase4_results: stale"):
            parsed["phase4_results_stale"] = line.strip()
        elif line.startswith("phase4_results:"):
            parsed["phase4_results"] = line.strip()
            parsed.update(parse_phase4_results_line(line.strip()))
        elif line.startswith("phase4_progress:"):
            parsed["phase4_progress"] = line.strip()
            parsed.update(parse_phase4_progress_line(line.strip()))
        elif line.startswith("log_progress:"):
            parsed["log_progress"] = line.strip()
            match = re.search(r"latest write ([0-9]+)s ago", line)
            if match:
                parsed["log_quiet_seconds"] = int(match.group(1))
    return parsed


def _parse_count_map(raw: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in raw.split(","):
        key, sep, value = item.partition("=")
        if sep and value.isdigit():
            counts[key] = int(value)
    return counts


def parse_phase4_results_line(line: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    total = re.search(r"\btotal=([0-9]+)\b", line)
    if total:
        parsed["total"] = int(total.group(1))
    sites = re.search(r"\bsites=([^ ]+)", line)
    if sites:
        parsed["site_counts"] = _parse_count_map(sites.group(1))
    final_status = re.search(r"\bfinal_status=([^ ]+)", line)
    if final_status:
        parsed["final_status_counts"] = _parse_count_map(final_status.group(1))
    return parsed


def parse_phase4_progress_line(line: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    status = re.search(r"\bstatus=([^ ]+)", line)
    if status:
        parsed["phase4_progress_status"] = status.group(1)
    stage = re.search(r"\bstage=([^ ]+)", line)
    if stage:
        parsed["phase4_progress_stage"] = stage.group(1)
    age = re.search(r"\bage_seconds=([0-9]+)\b", line)
    if age:
        parsed["phase4_progress_quiet_seconds"] = int(age.group(1))
    initial = re.search(r"\binitial=([0-9]+)/([0-9]+)", line)
    if initial:
        parsed["phase4_completed_initial_tasks"] = int(initial.group(1))
        parsed["phase4_total_tasks"] = int(initial.group(2))
    initial_started = re.search(r"\binitial_started=([0-9]+)/([0-9]+)", line)
    if initial_started:
        parsed["phase4_initial_started_tasks"] = int(initial_started.group(1))
        parsed.setdefault("phase4_total_tasks", int(initial_started.group(2)))
    initial_active = re.search(r"\binitial_active=([0-9]+)\b", line)
    if initial_active:
        parsed["phase4_active_initial_tasks"] = int(initial_active.group(1))
    postprocessed = re.search(r"\bpostprocessed=([0-9]+)/([0-9]+)", line)
    if postprocessed:
        parsed["phase4_postprocessed_tasks"] = int(postprocessed.group(1))
    return parsed


def status_for_job(config: SweepConfig, job_id: str) -> dict[str, Any]:
    completed = run_checked(
        [
            "scripts/remote_job_status.sh",
            "--host-config",
            config.host_config,
            "--remote-dir",
            config.remote_dir,
            "--job-id",
            job_id,
        ]
    )
    return parse_status_output(completed.stdout)


def tail_job(config: SweepConfig, job_id: str) -> str:
    args = [
        "scripts/remote_job_tail.sh",
        "--host-config",
        config.host_config,
        "--remote-dir",
        config.remote_dir,
        "--job-id",
        job_id,
        "--lines",
        "120",
        "--both",
    ]
    completed = subprocess.run(
        args,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout + completed.stderr


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_handoff(state_dir: Path, state: dict[str, Any]) -> None:
    lines = [
        f"# Phase 4 Model Sweep Handoff: {state['sweep_name']}",
        "",
        f"- Updated: {state.get('updated_at')}",
        f"- Status: {state.get('status')}",
        f"- Source run: `{state.get('source_run_dir')}`",
        f"- Remote dir: `{state.get('remote_dir')}`",
        "",
        "## Records",
    ]
    for record in state.get("records", []):
        lines.extend(
            [
                "",
                f"### {record.get('key')}",
                f"- model: `{record.get('provider')} / {record.get('model')}`",
                f"- run dir: `{record.get('run_dir')}`",
                f"- job id: `{record.get('job_id')}`",
                f"- status: `{record.get('status')}`",
                f"- retries: `{record.get('retries_consumed', 0)}`",
                f"- phase4: `{record.get('phase4_results')}`",
                f"- progress: `{record.get('phase4_progress')}`",
            ]
        )
    (state_dir / "handoff.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_state(state_dir: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = datetime.now(UTC).isoformat()
    write_json(state_dir / "sweep_state.json", state)
    write_handoff(state_dir, state)


def initial_state(config: SweepConfig, state_dir: Path) -> dict[str, Any]:
    return {
        "sweep_name": config.sweep_name,
        "status": "starting",
        "state_dir": str(state_dir),
        "source_run_dir": config.source_run_dir,
        "remote_dir": config.remote_dir,
        "completed_runs": [run.__dict__ for run in config.completed_runs],
        "records": [],
    }


def sync_to_r5(config: SweepConfig) -> None:
    run_checked(
        [
            "scripts/sync_to_r5.sh",
            "--host-config",
            config.host_config,
            "--remote-dir",
            config.remote_dir,
        ]
    )


def start_remote_job(config: SweepConfig, model: ModelRun, run_dir: str) -> tuple[str, str]:
    command_body = build_phase4_command(config, model, run_dir)
    args = build_remote_job_start_args(
        config,
        model,
        run_dir=run_dir,
        command_body=command_body,
    )
    completed = run_checked(args)
    return parse_job_id(completed.stdout), command_body


def start_remote_resume_job(
    config: SweepConfig,
    model: ModelRun,
    run_dir: str,
    *,
    resume_index: int,
) -> tuple[str, str]:
    command_body = build_phase4_resume_command(config, model, run_dir)
    args = build_remote_resume_job_start_args(
        config,
        model,
        run_dir=run_dir,
        command_body=command_body,
        resume_index=resume_index,
    )
    completed = run_checked(args)
    return parse_job_id(completed.stdout), command_body


def stop_remote_job(config: SweepConfig, job_id: str) -> subprocess.CompletedProcess[str]:
    return run_checked(
        [
            "scripts/remote_job_stop.sh",
            "--host-config",
            config.host_config,
            "--remote-dir",
            config.remote_dir,
            "--job-id",
            job_id,
        ]
    )


def monitor_job(
    config: SweepConfig,
    job_id: str,
    *,
    initial_poll_seconds: int,
    poll_seconds: int,
    stale_seconds: int,
    max_status_checks: int | None = None,
    on_status: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    time.sleep(initial_poll_seconds)
    checks = 0
    while True:
        checks += 1
        status = status_for_job(config, job_id)
        if status.get("status") != "running":
            if on_status:
                on_status(status)
            return status
        quiet = int(status.get("log_quiet_seconds") or 0)
        phase4_quiet = int(status.get("phase4_progress_quiet_seconds") or 0)
        if phase4_quiet >= stale_seconds and not status.get("phase4_results"):
            status["status"] = "attention_required"
            status["failure_class"] = "stale_phase4_progress"
            status["tail"] = tail_job(config, job_id)
            if on_status:
                on_status(status)
            return status
        if quiet >= stale_seconds:
            status["status"] = "attention_required"
            status["failure_class"] = "stale_logs"
            status["tail"] = tail_job(config, job_id)
            if on_status:
                on_status(status)
            return status
        if max_status_checks is not None and checks >= max_status_checks:
            status["status"] = "attention_required"
            status["failure_class"] = "max_status_checks"
            if on_status:
                on_status(status)
            return status
        if on_status:
            on_status(status)
        time.sleep(poll_seconds)


def render_dry_run(config: SweepConfig, models: list[ModelRun], timestamp: str) -> dict[str, Any]:
    runs = []
    for model in models:
        run_dir = run_dir_for_model(config, model, timestamp=timestamp)
        command_body = build_phase4_command(config, model, run_dir)
        runs.append(
            {
                "key": model.key,
                "provider": model.provider,
                "model": model.model,
                "service_tier": model.service_tier,
                "retry_budget": model.retry_budget,
                "run_dir": run_dir,
                "remote_job_start_args": build_remote_job_start_args(
                    config,
                    model,
                    run_dir=run_dir,
                    command_body=command_body,
                ),
                "command_body": command_body,
            }
        )
    return {
        "sweep_name": config.sweep_name,
        "dry_run": True,
        "timestamp": timestamp,
        "completed_runs": [run.__dict__ for run in config.completed_runs],
        "runs": runs,
    }


def run_sweep(args: argparse.Namespace) -> int:
    config = load_sweep_config(args.config)
    timestamp = args.timestamp or utc_timestamp()
    models = select_models(config, start_at=args.start_at, only=args.only)
    if args.dry_run:
        payload = render_dry_run(config, models, timestamp)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if not args.allow_dirty:
        if tracked_tree_is_dirty():
            print(
                "tracked worktree is dirty; commit or pass --allow-dirty "
                "for an intentional dev run",
                file=sys.stderr,
            )
            return 2
        untracked = untracked_files()
        if untracked:
            print(
                "worktree has untracked files; sync_to_r5 rsyncs workspace bytes, "
                "so run from a clean worktree or pass --allow-dirty intentionally",
                file=sys.stderr,
            )
            print("\n".join(f"- {path}" for path in untracked[:20]), file=sys.stderr)
            if len(untracked) > 20:
                print(f"... {len(untracked) - 20} more", file=sys.stderr)
            return 2

    try:
        require_no_running_remote_jobs(config)
    except RuntimeError as exc:
        print("remote jobs are already running; refusing to sync/start:", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 2

    state_dir = args.state_dir or REPO_ROOT / "logs" / f"phase4_model_sweep_{timestamp}Z"
    log_event(f"writing sweep state to {state_dir}")
    state = initial_state(config, state_dir)
    update_state(state_dir, state)

    state["status"] = "syncing"
    update_state(state_dir, state)
    log_event("syncing current checkout to r5")
    sync_to_r5(config)
    state["remote_sync_stamp"] = local_git_stamp(config)
    update_state(state_dir, state)
    log_event("sync complete")

    for model in models:
        record: dict[str, Any] = {
            "key": model.key,
            "provider": model.provider,
            "model": model.model,
            "service_tier": model.service_tier,
            "source_run_dir": config.source_run_dir,
            "retry_budget": model.retry_budget,
            "retries_consumed": 0,
            "status": "pending",
        }
        state["records"].append(record)
        for attempt in range(1, model.retry_budget + 1):
            run_dir = run_dir_for_model(config, model, timestamp=timestamp, attempt=attempt)
            record.update(
                {
                    "status": "starting",
                    "run_dir": run_dir,
                    "result_path": f"{run_dir}/phase_4/results.json",
                    "summary_path": f"{run_dir}/phase_4/summary.txt",
                    "variant_audit_path": f"{run_dir}/phase_4/variant_audit.txt",
                    "attempt": attempt,
                    "retries_consumed": attempt - 1,
                    "remote_sync_stamp": state.get("remote_sync_stamp"),
                }
            )
            state["status"] = f"running:{model.key}"
            update_state(state_dir, state)
            log_event(f"starting {model.key} attempt {attempt}/{model.retry_budget}: {run_dir}")
            try:
                require_no_running_remote_jobs(config)
            except RuntimeError as exc:
                record.update(
                    {
                        "status": "blocked_before_start",
                        "failure_class": "remote_job_already_running",
                        "stderr": str(exc),
                    }
                )
                state["status"] = "failed"
                update_state(state_dir, state)
                return 2
            try:
                job_id, command_body = start_remote_job(config, model, run_dir)
            except subprocess.CalledProcessError as exc:
                record.update(
                    {
                        "status": "failed_to_start",
                        "failure_class": "remote_job_start_failed",
                        "stdout": exc.stdout,
                        "stderr": exc.stderr,
                    }
                )
                state["status"] = "failed"
                update_state(state_dir, state)
                return exc.returncode or 1
            record.update(
                {
                    "status": "running",
                    "job_id": job_id,
                    "command_body": command_body,
                    "stale_resumes_consumed": 0,
                    "resume_job_ids": [],
                    "stale_resume_events": [],
                    "remote_job_start_args": build_remote_job_start_args(
                        config,
                        model,
                        run_dir=run_dir,
                        command_body=command_body,
                    ),
                }
            )
            update_state(state_dir, state)
            log_event(f"{model.key} launched as job {job_id}")

            def record_status(
                status: dict[str, Any],
                *,
                current_record: dict[str, Any] = record,
                model_key: str = model.key,
            ) -> None:
                current_record.update(
                    {
                        "status": status.get("status"),
                        "returncode": status.get("returncode"),
                        "phase4_results": status.get("phase4_results"),
                        "phase4_progress": status.get("phase4_progress"),
                        "phase4_progress_quiet_seconds": status.get(
                            "phase4_progress_quiet_seconds"
                        ),
                        "phase4_progress_stage": status.get("phase4_progress_stage"),
                        "log_progress": status.get("log_progress"),
                        "failure_class": status.get("failure_class"),
                        "total": status.get("total"),
                        "site_counts": status.get("site_counts"),
                        "final_status_counts": status.get("final_status_counts"),
                    }
                )
                if status.get("tail"):
                    current_record["tail"] = status["tail"]
                update_state(state_dir, state)
                summary = (
                    status.get("phase4_results")
                    or status.get("phase4_progress")
                    or status.get("log_progress")
                    or ""
                )
                log_event(f"{model_key} status={status.get('status')} {summary}".rstrip())

            stale_resumes_consumed = 0
            while True:
                status = monitor_job(
                    config,
                    job_id,
                    initial_poll_seconds=args.initial_poll_seconds,
                    poll_seconds=args.poll_seconds,
                    stale_seconds=args.stale_seconds,
                    max_status_checks=args.max_status_checks,
                    on_status=record_status,
                )
                record.update(
                    {
                        "status": status.get("status"),
                        "returncode": status.get("returncode"),
                        "phase4_results": status.get("phase4_results"),
                        "phase4_progress": status.get("phase4_progress"),
                        "phase4_progress_quiet_seconds": status.get(
                            "phase4_progress_quiet_seconds"
                        ),
                        "phase4_progress_stage": status.get("phase4_progress_stage"),
                        "log_progress": status.get("log_progress"),
                        "failure_class": status.get("failure_class"),
                        "total": status.get("total"),
                        "site_counts": status.get("site_counts"),
                        "final_status_counts": status.get("final_status_counts"),
                    }
                )
                if status.get("tail"):
                    record["tail"] = status["tail"]
                update_state(state_dir, state)
                if status.get("status") == "exited" and status.get("returncode") == 0:
                    record["status"] = "completed"
                    update_state(state_dir, state)
                    log_event(f"{model.key} completed")
                    break
                if (
                    status.get("status") == "attention_required"
                    and status.get("failure_class")
                    in {"stale_logs", "stale_phase4_progress"}
                    and stale_resumes_consumed < config.stale_resume_budget
                ):
                    stale_resumes_consumed += 1
                    log_event(
                        f"{model.key} stale; stopping job {job_id} and resuming "
                        f"checkpoint ({stale_resumes_consumed}/{config.stale_resume_budget})"
                    )
                    event = {
                        "stale_job_id": job_id,
                        "resume_index": stale_resumes_consumed,
                        "status": status,
                    }
                    try:
                        stop_remote_job(config, job_id)
                    except subprocess.CalledProcessError as exc:
                        event.update(
                            {
                                "stop_failed": True,
                                "stop_stdout": exc.stdout,
                                "stop_stderr": exc.stderr,
                            }
                        )
                        record.setdefault("stale_resume_events", []).append(event)
                        record["status"] = "failed"
                        state["status"] = "failed"
                        update_state(state_dir, state)
                        return exc.returncode or 1
                    event["stopped"] = True
                    record.setdefault("stale_resume_events", []).append(event)
                    try:
                        require_no_running_remote_jobs(config)
                        job_id, command_body = start_remote_resume_job(
                            config,
                            model,
                            run_dir,
                            resume_index=stale_resumes_consumed,
                        )
                    except (subprocess.CalledProcessError, RuntimeError) as exc:
                        record.update(
                            {
                                "status": "failed_to_resume",
                                "failure_class": "remote_resume_start_failed",
                                "stderr": str(exc),
                            }
                        )
                        state["status"] = "failed"
                        update_state(state_dir, state)
                        return getattr(exc, "returncode", 1) or 1
                    record.update(
                        {
                            "status": "running",
                            "job_id": job_id,
                            "command_body": command_body,
                            "stale_resumes_consumed": stale_resumes_consumed,
                            "last_resume_job_id": job_id,
                        }
                    )
                    record.setdefault("resume_job_ids", []).append(job_id)
                    update_state(state_dir, state)
                    log_event(f"{model.key} resume launched as job {job_id}")
                    continue
                record["status"] = "failed"
                update_state(state_dir, state)
                # Preserve failed artifacts and stop. The retry budget is recorded
                # but not consumed automatically because reruns require diagnosis.
                state["status"] = "failed"
                update_state(state_dir, state)
                return 1
            if record.get("status") == "completed":
                break
        else:
            state["status"] = "failed"
            update_state(state_dir, state)
            return 1

    state["status"] = "completed"
    update_state(state_dir, state)
    log_event("sweep completed")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Sweep config JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Render commands only.")
    parser.add_argument("--timestamp", default=None, help="UTC timestamp override.")
    parser.add_argument("--start-at", default=None, help="First model key to run.")
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        help="Run only this model key. Repeatable.",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=None,
        help="Local sweep state directory. Defaults under logs/.",
    )
    parser.add_argument("--initial-poll-seconds", type=int, default=300)
    parser.add_argument("--poll-seconds", type=int, default=600)
    parser.add_argument("--stale-seconds", type=int, default=900)
    parser.add_argument(
        "--max-status-checks",
        type=int,
        default=None,
        help="Testing escape hatch; stop monitoring after N checks.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow tracked local changes for an intentional dev run.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run_sweep(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
