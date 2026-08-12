from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def _base_env(repo_root: Path, tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = os.environ.get("PATH", "")
    env["HOME"] = str(tmp_path / "home")
    env["HOME"] and Path(env["HOME"]).mkdir(parents=True, exist_ok=True)
    return env


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(0o755)


def _write_fake_ssh(fakebin: Path) -> None:
    _write_executable(
        fakebin / "ssh",
        """#!/usr/bin/env python3
import subprocess
import sys

args = sys.argv[1:]
remote_cmd = args[-1]
raise SystemExit(subprocess.run(["bash", "-lc", remote_cmd], stdin=sys.stdin).returncode)
""",
    )


def _start_control_test_job(
    tmp_path: Path,
    *,
    fakebin: Path,
    pause_mode: str = "paused",
) -> tuple[Path, str, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    host_config = _host_config(tmp_path)
    remote_dir = tmp_path / "remote" / "browser-sim"
    remote_dir.mkdir(parents=True)
    _install_remote_process_group_helper(remote_dir)
    _write_fake_ssh(fakebin)
    state_root = remote_dir / "logs" / "run"
    state_root.mkdir(parents=True)
    (state_root / "pipeline_state.json").write_text(
        json.dumps(
            {
                "step": "phase_2",
                "status": "running",
                "phase_2_stage": "planning",
                "logs_dir": str(state_root),
                "run_id": "run-control-test",
                "definition_digest": "a" * 64,
                "timestamp": "2026-08-11T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    _write_executable(
        fakebin / "uv",
        f"""#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
root = Path({str(state_root)!r})
mode = os.environ.get("FAKE_PAUSE_MODE", {pause_mode!r})
if mode == "paused":
    state_path = root / "pipeline_state.json"
    state = json.loads(state_path.read_text())
    state.update(status="paused", pause_request_id="pause-" + "1" * 32)
    state_path.write_text(json.dumps(state))
    print("Pause acknowledged (pause-" + "1" * 32 + ").")
    raise SystemExit(0)
if mode == "timeout":
    print("Pause still pending after 1.0s (pause-test; reason=pause_wait_timeout).")
    raise SystemExit(1)
if mode == "terminal":
    state_path = root / "pipeline_state.json"
    state = json.loads(state_path.read_text())
    state["status"] = "complete"
    state_path.write_text(json.dumps(state))
    print("Pause ended because the pipeline is terminal (complete).")
    raise SystemExit(0)
if mode == "swap":
    state_path = root / "pipeline_state.json"
    state = json.loads(state_path.read_text())
    state.update(
        status="paused",
        pause_request_id="pause-" + "1" * 32,
        run_id="different-run",
        definition_digest="b" * 64,
    )
    state_path.write_text(json.dumps(state))
    print("Pause acknowledged (pause-" + "1" * 32 + ").")
    raise SystemExit(0)
print("pause wait rejected: unsupported stage", file=sys.stderr)
raise SystemExit(2)
""",
    )
    env = _base_env(repo_root, tmp_path)
    env["PATH"] = f"{fakebin}:{env['PATH']}"
    env["FAKE_PAUSE_MODE"] = pause_mode
    started = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "remote_job_start.sh"),
            "--host-config",
            str(host_config),
            "--remote-dir",
            str(remote_dir),
            "--name",
            "graceful-control-test",
            "--state-dir",
            "logs/run",
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(60)",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert started.returncode == 0, started.stderr
    job_id = next(
        line.split("=", 1)[1] for line in started.stdout.splitlines() if line.startswith("job_id=")
    )
    metadata_path = remote_dir / "logs" / "remote_jobs" / job_id / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(run_id="run-control-test", definition_digest="a" * 64)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return host_config, job_id, remote_dir


def _control_job_paths(remote_dir: Path, job_id: str) -> tuple[Path, Path]:
    job_dir = remote_dir / "logs" / "remote_jobs" / job_id
    return job_dir, job_dir / "metadata.json"


def _cleanup_control_job(remote_dir: Path, job_id: str) -> None:
    _, metadata_path = _control_job_paths(remote_dir, job_id)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        pgid = int(metadata.get("pgid") or 0)
        if pgid > 1:
            os.killpg(pgid, signal.SIGKILL)
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        pass


def _prepare_process_identity_fixture(remote_dir: Path, job_id: str) -> Path | None:
    """Provide proc-shaped identity files for fake SSH on non-Linux hosts."""

    if sys.platform.startswith("linux"):
        return None
    job_dir, metadata_path = _control_job_paths(remote_dir, job_id)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    pid = int(metadata["pid"])
    fingerprint = str(metadata["command_fingerprint"])
    proc_root = remote_dir / ".fake_proc"
    proc_dir = proc_root / str(pid)
    proc_dir.mkdir(parents=True, exist_ok=True)
    # Keep a stable synthetic start tick so stale metadata still fails the
    # exact comparison while valid metadata can exercise the full gate.
    if "process_start_ticks" in metadata and metadata.get("process_start_ticks") is None:
        metadata["process_start_ticks"] = "12345"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    stat_fields = ["0"] * 18 + ["12345"]
    (proc_dir / "stat").write_text(
        f"{pid} (run_job.py) R " + " ".join(stat_fields), encoding="utf-8"
    )
    (proc_dir / "cmdline").write_bytes(
        f"python3\0{job_dir / 'run_job.py'}\0{fingerprint}\0".encode()
    )
    cwd_path = proc_dir / "cwd"
    try:
        cwd_path.unlink()
    except FileNotFoundError:
        pass
    cwd_path.symlink_to(remote_dir, target_is_directory=True)
    return proc_root


def _run_graceful_stop(
    repo_root: Path,
    host_config: Path,
    remote_dir: Path,
    job_id: str,
    fakebin: Path,
    *extra: str,
) -> subprocess.CompletedProcess[str]:
    env = _base_env(repo_root, host_config.parent)
    env["PATH"] = f"{fakebin}:{env['PATH']}"
    proc_root = _prepare_process_identity_fixture(remote_dir, job_id)
    if proc_root is not None:
        env["WARP_REMOTE_JOB_PROC_ROOT"] = str(proc_root)
    return subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "remote_job_stop.sh"),
            "--host-config",
            str(host_config),
            "--remote-dir",
            str(remote_dir),
            "--job-id",
            job_id,
            "--graceful",
            "--pause-timeout",
            "1",
            "--pause-poll-interval",
            "0.1",
            "--timeout",
            "2",
            *extra,
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )


def _assert_control_process_alive(remote_dir: Path, job_id: str) -> bool:
    _, metadata_path = _control_job_paths(remote_dir, job_id)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    pid = int(metadata["pid"])
    if sys.platform.startswith("linux"):
        return Path("/proc", str(pid)).exists()
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _install_remote_process_group_helper(remote_dir: Path) -> None:
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    target_dir = remote_dir / "scripts"
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in ("remote_job_process_group.py", "remote_job_decisions.py"):
        source = scripts_dir / name
        target = target_dir / name
        shutil.copy2(source, target)
        target.chmod(source.stat().st_mode & 0o777)


def _host_config(tmp_path: Path) -> Path:
    path = tmp_path / "host.yaml"
    path.write_text(
        "\n".join(
            [
                "name: fake",
                "advertise_host: 203.0.113.10",
                "ssh_user: ubuntu",
                f"compose_dir_remote: {tmp_path}",
                "",
            ]
        )
    )
    return path


def _remote_direct_host_config_with_orchestrator(tmp_path: Path) -> Path:
    path = tmp_path / "host.remote.yaml"
    path.write_text(
        "\n".join(
            [
                "name: fake-r5",
                "access_mode: remote_direct_restricted",
                "advertise_host: 203.0.113.10",
                "orchestrator_host: 172.17.0.1",
                "bind_host: 0.0.0.0",
                "db_bind_host: 0.0.0.0",
                "allow_public_web_bind: true",
                "allow_public_db_bind: true",
                "ssh_user: ubuntu",
                f"compose_dir_remote: {tmp_path}",
                "",
            ]
        )
    )
    return path


def test_sync_dry_run_expands_ssh_key_and_excludes_sensitive_paths(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    host_config = _host_config(tmp_path)
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    args_file = tmp_path / "rsync_args.json"
    _write_executable(
        fakebin / "rsync",
        f"""#!/usr/bin/env python3
import json, sys
open({str(args_file)!r}, "w", encoding="utf-8").write(json.dumps(sys.argv[1:]))
""",
    )

    env = _base_env(repo_root, tmp_path)
    env["PATH"] = f"{fakebin}:{env['PATH']}"
    env["SSH_KEY"] = "$HOME/.ssh/webarena-key.pem"

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "sync_to_host.sh"),
            "--host-config",
            str(host_config),
            "--dry-run",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    args = json.loads(args_file.read_text())
    joined = "\n".join(args)
    assert "--dry-run" in args
    assert "--delete" in args
    exclude_values = {
        args[index + 1] for index, value in enumerate(args[:-1]) if value == "--exclude"
    }
    assert ".env" in joined
    assert ".git" in joined
    assert ".git/" in joined
    assert ".venv/" in joined
    assert ".venv.*" in joined
    assert ".codex-worktrees/" in joined
    assert "docs/handoffs/codex-handoff-*.md" in joined
    assert "logs/" in joined
    assert "pipeline_outputs/" in joined
    assert "data/" in joined
    assert "CODEX.local.md" in joined
    assert ".cache/" in joined
    assert "dist/" in joined
    assert ".DS_Store" in joined
    assert ".worldsim_sync_stamp.json" in joined
    assert {
        "data/",
        "CODEX.local.md",
        ".cache/",
        "dist/",
        ".DS_Store",
        ".worldsim_sync_stamp.json",
        "vendors/",
        ".cursor/",
        ".claude/local.md",
        ".claude/settings.local.json",
        "instances.smoke.local.json",
        "instances.json",
        "configs/benchmark_hosts/*.local.yaml",
        "configs/benchmark_hosts/r5.yaml",
    } <= exclude_values
    assert "*.sqlite" in joined
    assert "*.sqlite3" in joined
    assert ".modal/" in joined
    assert "scripts/smoke_phase_*.py" in joined
    assert "instances.scale.json" in joined
    assert "instances.scale.json.fragment" in joined
    assert "compose.scale.yml" in joined
    assert "compose.smoke.yml" in joined
    assert "scripts/docker-compose.scale.yml" in joined
    assert "scripts/proxy_ports.conf" in joined
    assert "agent-tools/" in joined
    assert "AgentLab/" in joined
    assert ".claude/worktrees/" in joined
    assert "vendors/" in joined
    assert "$HOME" not in joined
    assert str(Path(env["HOME"]) / ".ssh" / "webarena-key.pem") in joined


def test_sync_dry_run_excludes_linked_worktree_git_file(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    worktree_root = tmp_path / "linked-worktree"
    worktree_root.mkdir()
    (worktree_root / ".git").write_text("gitdir: /local/only/path/.git/worktrees/demo\n")
    scripts_dir = worktree_root / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "sync_to_host.sh").write_text(
        (repo_root / "scripts" / "sync_to_host.sh").read_text()
    )
    (scripts_dir / "lib").mkdir()
    (scripts_dir / "lib" / "remote_jobs.sh").write_text(
        (repo_root / "scripts" / "lib" / "remote_jobs.sh").read_text()
    )
    host_config = _host_config(tmp_path)
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    args_file = tmp_path / "rsync_args.json"
    _write_executable(
        fakebin / "rsync",
        f"""#!/usr/bin/env python3
import json, sys
open({str(args_file)!r}, "w", encoding="utf-8").write(json.dumps(sys.argv[1:]))
""",
    )

    env = _base_env(repo_root, tmp_path)
    env["PATH"] = f"{fakebin}:{env['PATH']}"

    completed = subprocess.run(
        [
            "bash",
            str(worktree_root / "scripts" / "sync_to_host.sh"),
            "--host-config",
            str(host_config),
            "--dry-run",
        ],
        cwd=worktree_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    args = json.loads(args_file.read_text())
    git_exclude_index = args.index(".git")
    assert args[git_exclude_index - 1] == "--exclude"
    assert "linked-worktree .git file" in completed.stderr


def test_remote_env_push_sends_selected_secret_over_stdin(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    host_config = _host_config(tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=or-test-secret\nOTHER=value\n", encoding="utf-8")
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    args_file = tmp_path / "ssh_args.json"
    payload_file = tmp_path / "payload.json"
    _write_executable(
        fakebin / "ssh",
        f"""#!/usr/bin/env python3
import json, sys
open({str(args_file)!r}, "w", encoding="utf-8").write(json.dumps(sys.argv[1:]))
open({str(payload_file)!r}, "w", encoding="utf-8").write(sys.stdin.read())
print("updated remote .env keys: OPENROUTER_API_KEY")
""",
    )

    env = _base_env(repo_root, tmp_path)
    env.pop("OPENROUTER_API_KEY", None)
    env["REMOTE_JOBS_SSH_BIN"] = str(fakebin / "ssh")

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "remote_env_push.sh"),
            "--host-config",
            str(host_config),
            "--remote-dir",
            "/remote/browser-sim",
            "--env-file",
            str(env_file),
            "--key",
            "OPENROUTER_API_KEY",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "updated remote .env keys: OPENROUTER_API_KEY"
    assert "or-test-secret" not in completed.stdout
    assert "or-test-secret" not in completed.stderr
    payload_stdin = payload_file.read_text()
    assert "PAYLOAD_JSON" in payload_stdin
    assert "or-test-secret" in payload_stdin
    assert "or-test-secret" not in "\n".join(json.loads(args_file.read_text()))


def test_sync_blocks_when_remote_job_registry_has_active_job(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    host_config = _host_config(tmp_path)
    remote_dir = tmp_path / "remote" / "browser-sim"
    job_dir = remote_dir / "logs" / "remote_jobs" / "job-active"
    job_dir.mkdir(parents=True)
    (job_dir / "metadata.json").write_text(
        json.dumps({"name": "phase-chain", "pid": 1}),
        encoding="utf-8",
    )
    (job_dir / "heartbeat.json").write_text(
        json.dumps({"status": "running"}),
        encoding="utf-8",
    )
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    rsync_called = tmp_path / "rsync_called"
    _write_executable(
        fakebin / "ssh",
        """#!/usr/bin/env python3
import subprocess
import sys

args = sys.argv[1:]
remote_cmd = args[-1]
raise SystemExit(subprocess.run(["bash", "-lc", remote_cmd], stdin=sys.stdin).returncode)
""",
    )
    _write_executable(
        fakebin / "rsync",
        f"""#!/usr/bin/env python3
from pathlib import Path
Path({str(rsync_called)!r}).write_text("called", encoding="utf-8")
""",
    )

    env = _base_env(repo_root, tmp_path)
    env["PATH"] = f"{fakebin}:{env['PATH']}"

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "sync_to_host.sh"),
            "--host-config",
            str(host_config),
            "--remote-dir",
            str(remote_dir),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "sync guard blocked" in completed.stderr
    assert "phase-chain" in completed.stderr
    assert "mix code versions" in completed.stderr
    assert not rsync_called.exists()


def test_start_rejects_missing_name_and_missing_command(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    host_config = _host_config(tmp_path)
    env = _base_env(repo_root, tmp_path)

    missing_name = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "remote_job_start.sh"),
            "--host-config",
            str(host_config),
            "--",
            "echo",
            "ok",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert missing_name.returncode == 2
    assert "--name required" in missing_name.stderr

    missing_command = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "remote_job_start.sh"),
            "--host-config",
            str(host_config),
            "--name",
            "demo",
            "--",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert missing_command.returncode == 2
    assert "missing command after --" in missing_command.stderr


def test_start_rejects_phase2c_smoke_instances_on_remote_orchestrator_host(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    host_config = _remote_direct_host_config_with_orchestrator(tmp_path)
    env = _base_env(repo_root, tmp_path)

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "remote_job_start.sh"),
            "--host-config",
            str(host_config),
            "--name",
            "bad phase2c",
            "--",
            "bash",
            "-lc",
            (
                "uv run python -m worldsim.main phase 2c "
                "--feasibility-instances instances.smoke.json"
            ),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "instance-topology guard blocked" in completed.stderr
    assert "orchestrator_host=172.17.0.1" in completed.stderr
    assert "--feasibility-instances instances.smoke.json" in completed.stderr
    assert "instances.scale.json" in completed.stderr


def test_start_rejects_shell_wrapped_resume_with_saved_smoke_instances_on_remote_host(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    host_config = _remote_direct_host_config_with_orchestrator(tmp_path)
    remote_dir = tmp_path / "remote" / "browser-sim"
    (remote_dir / "logs").mkdir(parents=True)
    _install_remote_process_group_helper(remote_dir)
    (remote_dir / "logs" / "pipeline_state.json").write_text(
        json.dumps({"step": "phase_4", "instances_path": "instances.smoke.json"}),
        encoding="utf-8",
    )
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    _write_executable(
        fakebin / "ssh",
        """#!/usr/bin/env python3
import subprocess
import sys

args = sys.argv[1:]
remote_cmd = args[-1]
raise SystemExit(subprocess.run(["bash", "-lc", remote_cmd], stdin=sys.stdin).returncode)
""",
    )

    env = _base_env(repo_root, tmp_path)
    env["PATH"] = f"{fakebin}:{env['PATH']}"

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "remote_job_start.sh"),
            "--host-config",
            str(host_config),
            "--remote-dir",
            str(remote_dir),
            "--name",
            "bad shell resume",
            "--",
            "bash",
            "-lc",
            "uv run python -m worldsim.main resume --agent-task-timeout 900",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "resume would fall back to pipeline_state instances.smoke.json" in completed.stderr
    assert "instances.scale.json" in completed.stderr


def test_start_rejects_shell_phase4_with_inline_state_dir_without_expected_output(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    host_config = _remote_direct_host_config_with_orchestrator(tmp_path)
    remote_dir = tmp_path / "remote" / "browser-sim"
    remote_dir.mkdir(parents=True)
    _install_remote_process_group_helper(remote_dir)
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    _write_executable(
        fakebin / "ssh",
        """#!/usr/bin/env python3
import subprocess
import sys

args = sys.argv[1:]
remote_cmd = args[-1]
raise SystemExit(subprocess.run(["bash", "-lc", remote_cmd], stdin=sys.stdin).returncode)
""",
    )

    env = _base_env(repo_root, tmp_path)
    env["PATH"] = f"{fakebin}:{env['PATH']}"
    env["WORLDSIM_ALLOW_REMOTE_INSTANCE_TOPOLOGY_MISMATCH"] = "1"

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "remote_job_start.sh"),
            "--host-config",
            str(host_config),
            "--remote-dir",
            str(remote_dir),
            "--name",
            "unobservable phase4",
            "--",
            "bash",
            "-lc",
            (
                "export WORLDSIM_STATE_DIR=logs/custom_run; "
                "uv run python -m worldsim.main phase 4 --instances instances.scale.json"
            ),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "phase4 observability guard blocked" in completed.stderr
    assert "--expected-output <run>/phase_4/results.json" in completed.stderr


def test_start_allows_chained_phase0_smoke_and_phase2_scale_topology(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    host_config = _remote_direct_host_config_with_orchestrator(tmp_path)
    remote_dir = tmp_path / "remote" / "browser-sim"
    remote_dir.mkdir(parents=True)
    _install_remote_process_group_helper(remote_dir)
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    _write_executable(
        fakebin / "ssh",
        """#!/usr/bin/env python3
import subprocess
import sys

args = sys.argv[1:]
remote_cmd = args[-1]
raise SystemExit(subprocess.run(["bash", "-lc", remote_cmd], stdin=sys.stdin).returncode)
""",
    )

    env = _base_env(repo_root, tmp_path)
    env["PATH"] = f"{fakebin}:{env['PATH']}"

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "remote_job_start.sh"),
            "--host-config",
            str(host_config),
            "--remote-dir",
            str(remote_dir),
            "--name",
            "correct mixed topology",
            "--",
            "bash",
            "-lc",
            (
                ": worldsim.main phase 0 --instances instances.smoke.json && "
                ": worldsim.main phase 2 --feasibility-instances instances.scale.json"
            ),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "instance-topology guard blocked" not in completed.stderr


def test_start_exports_orchestrator_host_to_remote_job_env(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    host_config = _remote_direct_host_config_with_orchestrator(tmp_path)
    remote_dir = tmp_path / "remote" / "browser-sim"
    remote_dir.mkdir(parents=True)
    _install_remote_process_group_helper(remote_dir)
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    _write_executable(
        fakebin / "ssh",
        """#!/usr/bin/env python3
import subprocess
import sys

args = sys.argv[1:]
remote_cmd = args[-1]
raise SystemExit(subprocess.run(["bash", "-lc", remote_cmd], stdin=sys.stdin).returncode)
""",
    )

    env = _base_env(repo_root, tmp_path)
    env["PATH"] = f"{fakebin}:{env['PATH']}"

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "remote_job_start.sh"),
            "--host-config",
            str(host_config),
            "--remote-dir",
            str(remote_dir),
            "--name",
            "env export",
            "--",
            "python3",
            "-c",
            (
                "import os, pathlib; "
                "pathlib.Path('orchestrator_env.txt').write_text("
                "os.environ.get('WORLDSIM_ORCHESTRATOR_HOST', ''))"
            ),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    output = remote_dir / "orchestrator_env.txt"
    for _ in range(50):
        if output.exists():
            break
        time.sleep(0.05)
    assert output.read_text() == "172.17.0.1"


def test_start_allows_chained_phase1_novel_with_explicit_benchmark(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    host_config = _remote_direct_host_config_with_orchestrator(tmp_path)
    remote_dir = tmp_path / "remote" / "browser-sim"
    remote_dir.mkdir(parents=True)
    _install_remote_process_group_helper(remote_dir)
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    _write_executable(
        fakebin / "ssh",
        """#!/usr/bin/env python3
import subprocess
import sys

args = sys.argv[1:]
remote_cmd = args[-1]
raise SystemExit(subprocess.run(["bash", "-lc", remote_cmd], stdin=sys.stdin).returncode)
""",
    )

    env = _base_env(repo_root, tmp_path)
    env["PATH"] = f"{fakebin}:{env['PATH']}"

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "remote_job_start.sh"),
            "--host-config",
            str(host_config),
            "--remote-dir",
            str(remote_dir),
            "--name",
            "phase1 benchmark ok",
            "--",
            "bash",
            "-lc",
            (
                ": worldsim.main phase 0 --benchmark /home/ubuntu/vendors/webarena-verified "
                "--instances instances.smoke.json "
                "--host-inventory-instances instances.scale.json && "
                ": worldsim.main phase 1 --benchmark /home/ubuntu/vendors/webarena-verified "
                "--generate-novel --sites gitlab,reddit && "
                ": worldsim.main phase 2 --feasibility-instances instances.scale.json"
            ),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Chained Phase 0 -> Phase 1 novel generation" not in completed.stderr


def test_start_allows_chained_phase2_and_phase3_with_matching_origin(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    host_config = _remote_direct_host_config_with_orchestrator(tmp_path)
    remote_dir = tmp_path / "remote" / "browser-sim"
    remote_dir.mkdir(parents=True)
    _install_remote_process_group_helper(remote_dir)
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    _write_executable(
        fakebin / "ssh",
        """#!/usr/bin/env python3
import subprocess
import sys

args = sys.argv[1:]
remote_cmd = args[-1]
raise SystemExit(subprocess.run(["bash", "-lc", remote_cmd], stdin=sys.stdin).returncode)
""",
    )

    env = _base_env(repo_root, tmp_path)
    env["PATH"] = f"{fakebin}:{env['PATH']}"

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "remote_job_start.sh"),
            "--host-config",
            str(host_config),
            "--remote-dir",
            str(remote_dir),
            "--name",
            "origin matched",
            "--",
            "bash",
            "-lc",
            (
                ": worldsim.main phase 2 --task-origin new_task "
                "--feasibility-instances instances.scale.json && "
                ": worldsim.main phase 3 --task-origin new_task --sites gitlab,reddit"
            ),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "same --task-origin to Phase 3" not in completed.stderr


def test_start_allows_phase4_with_scale_instances_and_task_timeout(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    host_config = _remote_direct_host_config_with_orchestrator(tmp_path)
    remote_dir = tmp_path / "remote" / "browser-sim"
    remote_dir.mkdir(parents=True)
    _install_remote_process_group_helper(remote_dir)
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    _write_executable(
        fakebin / "ssh",
        """#!/usr/bin/env python3
import subprocess
import sys

args = sys.argv[1:]
remote_cmd = args[-1]
raise SystemExit(subprocess.run(["bash", "-lc", remote_cmd], stdin=sys.stdin).returncode)
""",
    )

    env = _base_env(repo_root, tmp_path)
    env["PATH"] = f"{fakebin}:{env['PATH']}"

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "remote_job_start.sh"),
            "--host-config",
            str(host_config),
            "--remote-dir",
            str(remote_dir),
            "--name",
            "phase4 bounded",
            "--expected-output",
            "logs/phase4_bounded/phase_4/results.json",
            "--",
            "bash",
            "-lc",
            (
                ": worldsim.main phase 4 --instances instances.scale.json "
                "--agent-llm-timeout 240 --agent-step-timeout 300 "
                "--agent-task-timeout 900"
            ),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "instance-topology guard blocked" not in completed.stderr


def test_start_allows_scale_instances_on_remote_orchestrator_host(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    host_config = _remote_direct_host_config_with_orchestrator(tmp_path)
    remote_dir = tmp_path / "remote" / "browser-sim"
    remote_dir.mkdir(parents=True)
    _install_remote_process_group_helper(remote_dir)
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    _write_executable(
        fakebin / "ssh",
        """#!/usr/bin/env python3
import subprocess
import sys

args = sys.argv[1:]
remote_cmd = args[-1]
raise SystemExit(subprocess.run(["bash", "-lc", remote_cmd], stdin=sys.stdin).returncode)
""",
    )

    env = _base_env(repo_root, tmp_path)
    env["PATH"] = f"{fakebin}:{env['PATH']}"

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "remote_job_start.sh"),
            "--host-config",
            str(host_config),
            "--remote-dir",
            str(remote_dir),
            "--name",
            "scale phase2",
            "--",
            "bash",
            "-lc",
            (
                ": worldsim.main phase 2 --sites gitlab,reddit "
                "--feasibility-instances instances.scale.json"
            ),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "instance-topology guard blocked" not in completed.stderr


def test_start_writes_remote_metadata_with_fake_ssh(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    host_config = _host_config(tmp_path)
    remote_dir = tmp_path / "remote" / "browser-sim"
    remote_dir.mkdir(parents=True)
    _install_remote_process_group_helper(remote_dir)
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    _write_executable(
        fakebin / "ssh",
        """#!/usr/bin/env python3
import subprocess
import sys

args = sys.argv[1:]
remote_cmd = args[-1]
raise SystemExit(subprocess.run(["bash", "-lc", remote_cmd], stdin=sys.stdin).returncode)
""",
    )

    env = _base_env(repo_root, tmp_path)
    env["PATH"] = f"{fakebin}:{env['PATH']}"

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "remote_job_start.sh"),
            "--host-config",
            str(host_config),
            "--remote-dir",
            str(remote_dir),
            "--name",
            "phase1 route diversity",
            "--expected-output",
            "logs/phase_1/benign_tasks.json",
            "--state-dir",
            "auto",
            "--",
            sys.executable,
            "-c",
            "pass",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    job_id_line = next(line for line in completed.stdout.splitlines() if line.startswith("job_id="))
    job_id = job_id_line.split("=", 1)[1]
    job_dir = remote_dir / "logs" / "remote_jobs" / job_id
    metadata = json.loads((job_dir / "metadata.json").read_text())
    argv = json.loads((job_dir / "command.argv.json").read_text())

    assert metadata["job_id"] == job_id
    assert metadata["name"] == "phase1 route diversity"
    assert metadata["state_dir"] == f"logs/remote_jobs/{job_id}/state"
    assert metadata["expected_outputs"] == ["logs/phase_1/benign_tasks.json"]
    assert argv[:2] == [sys.executable, "-c"]
    assert (job_dir / "stdout.log").exists()
    assert (job_dir / "stderr.log").exists()
    assert (job_dir / "pid").exists()
    assert (job_dir / "pgid").exists()


def test_start_auto_wraps_uv_command_in_login_shell(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    host_config = _host_config(tmp_path)
    remote_dir = tmp_path / "remote" / "browser-sim"
    remote_dir.mkdir(parents=True)
    _install_remote_process_group_helper(remote_dir)
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    _write_executable(
        fakebin / "ssh",
        """#!/usr/bin/env python3
import subprocess
import sys

args = sys.argv[1:]
remote_cmd = args[-1]
raise SystemExit(subprocess.run(["bash", "-lc", remote_cmd], stdin=sys.stdin).returncode)
""",
    )

    env = _base_env(repo_root, tmp_path)
    env["PATH"] = f"{fakebin}:{env['PATH']}"

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "remote_job_start.sh"),
            "--host-config",
            str(host_config),
            "--remote-dir",
            str(remote_dir),
            "--name",
            "direct uv command",
            "--",
            "uv",
            "--version",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    job_id_line = next(line for line in completed.stdout.splitlines() if line.startswith("job_id="))
    job_id = job_id_line.split("=", 1)[1]
    job_dir = remote_dir / "logs" / "remote_jobs" / job_id
    metadata = json.loads((job_dir / "metadata.json").read_text())
    argv = json.loads((job_dir / "command.argv.json").read_text())

    assert argv == ["bash", "-lc", "uv --version"]
    assert metadata["command"] == ["bash", "-lc", "uv --version"]
    assert metadata["original_command"] == ["uv", "--version"]
    assert metadata["command_execution"] == {
        "mode": "auto",
        "normalized": True,
        "reason": "auto_login_shell_for_uv",
        "original_command": ["uv", "--version"],
    }


def test_start_records_child_launch_failure_with_fake_ssh(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    host_config = _host_config(tmp_path)
    remote_dir = tmp_path / "remote" / "browser-sim"
    remote_dir.mkdir(parents=True)
    _install_remote_process_group_helper(remote_dir)
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    _write_executable(
        fakebin / "ssh",
        """#!/usr/bin/env python3
import subprocess
import sys

args = sys.argv[1:]
remote_cmd = args[-1]
raise SystemExit(subprocess.run(["bash", "-lc", remote_cmd], stdin=sys.stdin).returncode)
""",
    )

    env = _base_env(repo_root, tmp_path)
    env["PATH"] = f"{fakebin}:{env['PATH']}"

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "remote_job_start.sh"),
            "--host-config",
            str(host_config),
            "--remote-dir",
            str(remote_dir),
            "--name",
            "missing child command",
            "--",
            "worldsim-command-that-does-not-exist",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    job_id_line = next(line for line in completed.stdout.splitlines() if line.startswith("job_id="))
    job_id = job_id_line.split("=", 1)[1]
    job_dir = remote_dir / "logs" / "remote_jobs" / job_id
    exit_path = job_dir / "exit.json"
    for _ in range(40):
        if exit_path.exists():
            break
        time.sleep(0.05)

    exit_data = json.loads(exit_path.read_text())
    stderr = (job_dir / "stderr.log").read_text()
    assert exit_data["status"] == "launch_failed"
    assert exit_data["returncode"] == 127
    assert "remote job child launch failed" in stderr
    assert "worldsim-command-that-does-not-exist" in stderr


def test_status_surfaces_remote_launch_failed_state(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    host_config = _host_config(tmp_path)
    remote_dir = tmp_path / "remote" / "browser-sim"
    job_id = "20260429t000000z-launch-failed-abcdef"
    job_dir = remote_dir / "logs" / "remote_jobs" / job_id
    job_dir.mkdir(parents=True)
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    _write_executable(
        fakebin / "ssh",
        """#!/usr/bin/env python3
import subprocess
import sys

args = sys.argv[1:]
remote_cmd = args[-1]
raise SystemExit(subprocess.run(["bash", "-lc", remote_cmd], stdin=sys.stdin).returncode)
""",
    )
    (job_dir / "metadata.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "name": "launch failed",
                "created_at": "2026-04-29T00:00:00+00:00",
                "pid": 999999,
                "pgid": 999999,
                "remote_dir": str(remote_dir),
                "command": ["worldsim-command-that-does-not-exist"],
                "expected_outputs": [],
            }
        )
    )
    (job_dir / "exit.json").write_text(json.dumps({"status": "launch_failed", "returncode": 127}))
    (job_dir / "stdout.log").write_text("")
    (job_dir / "stderr.log").write_text("remote job child launch failed\n")

    env = _base_env(repo_root, tmp_path)
    env["PATH"] = f"{fakebin}:{env['PATH']}"

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "remote_job_status.sh"),
            "--host-config",
            str(host_config),
            "--remote-dir",
            str(remote_dir),
            "--job-id",
            job_id,
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "status: launch_failed" in completed.stdout
    assert "returncode: 127" in completed.stdout
    assert "remote job child launch failed" in completed.stdout


def test_tail_resolves_completed_process_pool_task_from_summary(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    host_config = _host_config(tmp_path)
    remote_dir = tmp_path / "remote" / "browser-sim"
    job_id = "20260507t000000z-phase4-abcdef"
    job_dir = remote_dir / "logs" / "remote_jobs" / job_id
    state_dir = remote_dir / "logs" / "phase4-run"
    worker_dir = state_dir / "phase_4" / "process_pool_workers" / "worker_007"
    worker_dir.mkdir(parents=True)
    job_dir.mkdir(parents=True)
    (job_dir / "metadata.json").write_text(
        json.dumps({"job_id": job_id, "state_dir": str(state_dir)}),
        encoding="utf-8",
    )
    (job_dir / "stdout.log").write_text("job stdout should not be tailed\n", encoding="utf-8")
    (worker_dir / "stdout.log").write_text("worker stdout\n", encoding="utf-8")
    (worker_dir / "stderr.log").write_text("worker stderr\n", encoding="utf-8")
    (state_dir / "phase_4" / "progress.json").write_text(
        json.dumps({"process_pool_active_workers": []}),
        encoding="utf-8",
    )
    (state_dir / "phase_4" / "process_pool_summary.json").write_text(
        json.dumps(
            {
                "outcomes": [
                    {
                        "worker_id": 7,
                        "task_id": "adv-7",
                        "stdout": str(worker_dir / "stdout.log"),
                        "stderr": str(worker_dir / "stderr.log"),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    _write_executable(
        fakebin / "ssh",
        """#!/usr/bin/env python3
import subprocess
import sys

remote_cmd = sys.argv[-1]
raise SystemExit(subprocess.run(["bash", "-lc", remote_cmd], stdin=sys.stdin).returncode)
""",
    )

    env = _base_env(repo_root, tmp_path)
    env["PATH"] = f"{fakebin}:{env['PATH']}"

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "remote_job_tail.sh"),
            "--host-config",
            str(host_config),
            "--remote-dir",
            str(remote_dir),
            "--job-id",
            job_id,
            "--task-id",
            "adv-7",
            "--stderr",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "worker stderr\n"
    assert "job stdout" not in completed.stdout


def test_tail_task_id_miss_does_not_fall_back_to_job_stdout(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    host_config = _host_config(tmp_path)
    remote_dir = tmp_path / "remote" / "browser-sim"
    job_id = "20260507t000000z-phase4-abcdef"
    job_dir = remote_dir / "logs" / "remote_jobs" / job_id
    state_dir = remote_dir / "logs" / "phase4-run"
    job_dir.mkdir(parents=True)
    (state_dir / "phase_4").mkdir(parents=True)
    (job_dir / "metadata.json").write_text(
        json.dumps({"job_id": job_id, "state_dir": str(state_dir)}),
        encoding="utf-8",
    )
    (job_dir / "stdout.log").write_text("job stdout should not be tailed\n", encoding="utf-8")
    (state_dir / "phase_4" / "progress.json").write_text(
        json.dumps(
            {
                "process_pool_active_workers": [
                    {"worker_id": 1, "task_id": "other-task", "stdout": "x", "stderr": "y"}
                ]
            }
        ),
        encoding="utf-8",
    )
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    _write_executable(
        fakebin / "ssh",
        """#!/usr/bin/env python3
import subprocess
import sys

remote_cmd = sys.argv[-1]
raise SystemExit(subprocess.run(["bash", "-lc", remote_cmd], stdin=sys.stdin).returncode)
""",
    )

    env = _base_env(repo_root, tmp_path)
    env["PATH"] = f"{fakebin}:{env['PATH']}"

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "remote_job_tail.sh"),
            "--host-config",
            str(host_config),
            "--remote-dir",
            str(remote_dir),
            "--job-id",
            job_id,
            "--task-id",
            "missing-task",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "task id not found in process-pool worker logs: missing-task" in completed.stderr
    assert "other-task (progress)" in completed.stderr
    assert "job stdout" not in completed.stdout


def test_status_surfaces_recent_health_warnings(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    host_config = _host_config(tmp_path)
    remote_dir = tmp_path / "remote" / "browser-sim"
    job_id = "20260429t000000z-health-warning-abcdef"
    job_dir = remote_dir / "logs" / "remote_jobs" / job_id
    job_dir.mkdir(parents=True)
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    _write_executable(
        fakebin / "ssh",
        """#!/usr/bin/env python3
import subprocess
import sys

args = sys.argv[1:]
remote_cmd = args[-1]
raise SystemExit(subprocess.run(["bash", "-lc", remote_cmd], stdin=sys.stdin).returncode)
""",
    )
    (job_dir / "metadata.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "name": "phase0 warning",
                "created_at": "2026-04-29T00:00:00+00:00",
                "pid": 999999,
                "pgid": 999999,
                "remote_dir": str(remote_dir),
                "command": ["bash", "-lc", "uv run python -m worldsim.main phase 0"],
                "expected_outputs": [],
            }
        )
    )
    (job_dir / "exit.json").write_text(json.dumps({"status": "exited", "returncode": 0}))
    (job_dir / "stdout.log").write_text(
        "The instance is confirmed unreachable (connection timeout).\n"
    )
    (job_dir / "stderr.log").write_text("")

    env = _base_env(repo_root, tmp_path)
    env["PATH"] = f"{fakebin}:{env['PATH']}"

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "remote_job_status.sh"),
            "--host-config",
            str(host_config),
            "--remote-dir",
            str(remote_dir),
            "--job-id",
            job_id,
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "health_warnings:" in completed.stdout
    assert "host_unreachable:" in completed.stdout
    assert "phase0_unreachable:" in completed.stdout


def test_status_surfaces_phase4_variant_quality_flags(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    host_config = _host_config(tmp_path)
    remote_dir = tmp_path / "remote" / "browser-sim"
    job_id = "20260429t000000z-phase4-quality-abcdef"
    job_dir = remote_dir / "logs" / "remote_jobs" / job_id
    job_dir.mkdir(parents=True)
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    _write_executable(
        fakebin / "ssh",
        """#!/usr/bin/env python3
import subprocess
import sys

args = sys.argv[1:]
remote_cmd = args[-1]
raise SystemExit(subprocess.run(["bash", "-lc", remote_cmd], stdin=sys.stdin).returncode)
""",
    )
    (job_dir / "metadata.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "name": "phase4 quality",
                "created_at": "2026-04-29T00:00:00+00:00",
                "pid": 999999,
                "pgid": 999999,
                "remote_dir": str(remote_dir),
                "command": ["bash", "-lc", "uv run python -m worldsim.main phase 4"],
                "expected_outputs": [],
            }
        )
    )
    (job_dir / "exit.json").write_text(json.dumps({"status": "exited", "returncode": 0}))
    (job_dir / "stdout.log").write_text("Quality flags: generated_contract_qa_failed=3.\n")
    (job_dir / "stderr.log").write_text("")

    env = _base_env(repo_root, tmp_path)
    env["PATH"] = f"{fakebin}:{env['PATH']}"

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "remote_job_status.sh"),
            "--host-config",
            str(host_config),
            "--remote-dir",
            str(remote_dir),
            "--job-id",
            job_id,
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "health_warnings:" in completed.stdout
    assert "phase4_variant_quality:" in completed.stdout
    assert "generated_contract_qa_failed=3" in completed.stdout


def test_status_prints_phase4_summary_and_followup_command(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    host_config = _host_config(tmp_path)
    remote_dir = tmp_path / "remote" / "browser-sim"
    job_id = "20260429t000000z-phase4-demo-abcdef"
    job_dir = remote_dir / "logs" / "remote_jobs" / job_id
    phase4_dir = remote_dir / "logs" / "phase_4"
    phase2_dir = remote_dir / "logs" / "phase_2"
    job_dir.mkdir(parents=True)
    phase4_dir.mkdir(parents=True)
    phase2_dir.mkdir(parents=True)
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    _write_executable(
        fakebin / "ssh",
        """#!/usr/bin/env python3
import subprocess
import sys

args = sys.argv[1:]
remote_cmd = args[-1]
raise SystemExit(subprocess.run(["bash", "-lc", remote_cmd], stdin=sys.stdin).returncode)
""",
    )
    (job_dir / "metadata.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "name": "phase4 demo",
                "created_at": "2026-04-29T00:00:00+00:00",
                "pid": 999999,
                "pgid": 999999,
                "remote_dir": str(remote_dir),
                "command": ["uv", "run", "python", "-m", "worldsim.main", "phase", "4"],
                "expected_outputs": ["logs/phase_4/results.json"],
            }
        )
    )
    (job_dir / "exit.json").write_text(json.dumps({"status": "exited", "returncode": 0}))
    (job_dir / "stdout.log").write_text("stdout line\n")
    (job_dir / "stderr.log").write_text("stderr line\n")
    (phase4_dir / "results.json").write_text(
        json.dumps(
            [
                {
                    "task_id": "adv_gitlab",
                    "final_status": "success_on_variant",
                    "primary_inspection_trace": str(phase4_dir / "20260429" / "adv_gitlab_variant"),
                },
                {
                    "task_id": "adv_reddit",
                    "final_status": "resistant",
                    "trajectory_dir": str(phase4_dir / "20260429" / "adv_reddit"),
                },
            ]
        )
    )
    (phase2_dir / "adversarial_tasks.json").write_text(
        json.dumps(
            [
                {"id": "adv_gitlab", "site": "gitlab"},
                {"id": "adv_reddit", "site": "reddit"},
            ]
        )
    )
    (remote_dir / "logs" / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "artifacts_source": "s3://bucket/source-run",
                "artifacts": [{"path": "phase_0c"}, {"path": "phase_2"}, {"path": "phase_3"}],
            }
        )
    )

    env = _base_env(repo_root, tmp_path)
    env["PATH"] = f"{fakebin}:{env['PATH']}"

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "remote_job_status.sh"),
            "--host-config",
            str(host_config),
            "--remote-dir",
            str(remote_dir),
            "--job-id",
            job_id,
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "phase4_results: present logs/phase_4/results.json total=2" in completed.stdout
    assert "final_status=resistant=1,success_on_variant=1" in completed.stdout
    assert "sites=gitlab=1,reddit=1" in completed.stdout
    assert f"phase4_trace_root: {phase4_dir / '20260429'}" in completed.stdout
    assert (
        "artifact_manifest: logs/artifact_manifest.json source=s3://bucket/source-run artifacts=3"
    ) in completed.stdout
    assert (
        f"phase4_summary_command: cd {remote_dir} && uv run python "
        "scripts/summarize_phase_4_results.py logs/phase_4/results.json --inspect-limit 8"
    ) in completed.stdout


def test_status_marks_preexisting_phase4_results_stale(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    host_config = _host_config(tmp_path)
    remote_dir = tmp_path / "remote" / "browser-sim"
    job_id = "20260429t010000z-phase4-stale-abcdef"
    job_dir = remote_dir / "logs" / "remote_jobs" / job_id
    phase4_dir = remote_dir / "logs" / "phase_4"
    job_dir.mkdir(parents=True)
    phase4_dir.mkdir(parents=True)
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    _write_executable(
        fakebin / "ssh",
        """#!/usr/bin/env python3
import subprocess
import sys

args = sys.argv[1:]
remote_cmd = args[-1]
raise SystemExit(subprocess.run(["bash", "-lc", remote_cmd], stdin=sys.stdin).returncode)
""",
    )
    results_path = phase4_dir / "results.json"
    results_path.write_text(json.dumps([{"task_id": "adv_gitlab", "final_status": "resistant"}]))
    old_time = datetime(2026, 4, 29, tzinfo=UTC).timestamp()
    os.utime(results_path, (old_time, old_time))
    (job_dir / "metadata.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "name": "phase4 stale",
                "created_at": "2026-04-29T06:00:00+00:00",
                "pid": 999999,
                "pgid": 999999,
                "remote_dir": str(remote_dir),
                "command": ["uv", "run", "python", "-m", "worldsim.main", "phase", "4"],
                "expected_outputs": ["logs/phase_4/results.json"],
            }
        )
    )
    (job_dir / "exit.json").write_text(json.dumps({"status": "running"}))
    (job_dir / "stdout.log").write_text("")
    (job_dir / "stderr.log").write_text("")

    env = _base_env(repo_root, tmp_path)
    env["PATH"] = f"{fakebin}:{env['PATH']}"

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "remote_job_status.sh"),
            "--host-config",
            str(host_config),
            "--remote-dir",
            str(remote_dir),
            "--job-id",
            job_id,
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "stale logs/phase_4/results.json" in completed.stdout
    assert "phase4_results: stale logs/phase_4/results.json" in completed.stdout
    assert "phase4_results: present logs/phase_4/results.json" not in completed.stdout


def test_status_prints_phase4_progress_age_from_expected_run_dir(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    host_config = _host_config(tmp_path)
    remote_dir = tmp_path / "remote" / "browser-sim"
    job_id = "20260429t000000z-phase4-progress-abcdef"
    job_dir = remote_dir / "logs" / "remote_jobs" / job_id
    phase4_dir = remote_dir / "logs" / "custom_run" / "phase_4"
    job_dir.mkdir(parents=True)
    phase4_dir.mkdir(parents=True)
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    _write_executable(
        fakebin / "ssh",
        """#!/usr/bin/env python3
import subprocess
import sys

args = sys.argv[1:]
remote_cmd = args[-1]
raise SystemExit(subprocess.run(["bash", "-lc", remote_cmd], stdin=sys.stdin).returncode)
""",
    )
    (job_dir / "metadata.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "name": "phase4 progress",
                "created_at": "2026-04-29T00:00:00+00:00",
                "pid": 999999,
                "pgid": 999999,
                "remote_dir": str(remote_dir),
                "command": ["uv", "run", "python", "-m", "worldsim.main", "phase", "4"],
                "expected_outputs": ["logs/custom_run/phase_4/results.json"],
            }
        )
    )
    (job_dir / "stdout.log").write_text("watchdog line\n")
    (job_dir / "stderr.log").write_text("")
    (phase4_dir / "progress.json").write_text(
        json.dumps(
            {
                "status": "running",
                "stage": "initial_evaluation",
                "updated_at": "2026-05-02T12:00:22.308663",
                "total_tasks": 32,
                "completed_initial_tasks": 30,
                "postprocess_started_tasks": 4,
                "active_postprocess_tasks": 2,
                "postprocessed_tasks": 0,
                "postprocess_attempted_tasks": 3,
                "postprocess_failed_tasks": 1,
                "variant_progress": {
                    "variant_system": "eval-awareness-iterator",
                    "budget_preset": "smoke-3-probe",
                    "eval_awareness_max_iterations": 3,
                    "entered_tasks": 2,
                    "active_tasks": 2,
                    "generation_attempted": 6,
                    "generation_generated": 5,
                    "generation_failed": 1,
                    "evaluated": 3,
                    "pvpo_valid": 3,
                    "complied": 0,
                    "task_samples": [
                        {
                            "task_id": "adv-1",
                            "event": "variant_evaluation_started",
                            "round_index": 1,
                        }
                    ],
                },
            }
        )
    )

    env = _base_env(repo_root, tmp_path)
    env["PATH"] = f"{fakebin}:{env['PATH']}"

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "remote_job_status.sh"),
            "--host-config",
            str(host_config),
            "--remote-dir",
            str(remote_dir),
            "--job-id",
            job_id,
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "phase4_progress: present logs/custom_run/phase_4/progress.json" in completed.stdout
    assert "status=running stage=initial_evaluation" in completed.stdout
    assert (
        "initial=30/32 initial_started=0/32 initial_active=0 "
        "started=4/32 active=2 postprocessed=0/32"
    ) in completed.stdout
    assert "postprocess_attempted=3/32 postprocess_failed=1" in completed.stdout
    assert (
        "phase4_variant_progress: system=eval-awareness-iterator "
        "budget=smoke-3-probe eval_awareness_max_iterations=3 entered=2 active=2"
    ) in completed.stdout
    assert "rewrite_attempted=6 variant_evaluated=3 rejection_records=1" in completed.stdout
    assert "legacy_generated=5/6 evaluated=3 pvpo_valid=3 complied=0" in completed.stdout
    assert "phase4_variant_active: adv-1:variant_evaluation_started:round1" in completed.stdout
    assert "age_seconds=" in completed.stdout
    assert "updated_at=2026-05-02T12:00:22.308663" in completed.stdout


def test_status_does_not_report_stale_default_phase4_when_custom_expected_missing(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    host_config = _host_config(tmp_path)
    remote_dir = tmp_path / "remote" / "browser-sim"
    job_id = "20260429t000000z-custom-phase4-abcdef"
    job_dir = remote_dir / "logs" / "remote_jobs" / job_id
    stale_phase4_dir = remote_dir / "logs" / "phase_4"
    job_dir.mkdir(parents=True)
    stale_phase4_dir.mkdir(parents=True)
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    _write_executable(
        fakebin / "ssh",
        """#!/usr/bin/env python3
import subprocess
import sys

args = sys.argv[1:]
remote_cmd = args[-1]
raise SystemExit(subprocess.run(["bash", "-lc", remote_cmd], stdin=sys.stdin).returncode)
""",
    )
    (job_dir / "metadata.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "name": "custom phase4",
                "created_at": "2026-04-29T00:00:00+00:00",
                "pid": 999999,
                "pgid": 999999,
                "remote_dir": str(remote_dir),
                "command": ["uv", "run", "python", "-m", "worldsim.main", "phase", "4"],
                "expected_outputs": ["logs/custom_run/phase_4/results.json"],
            }
        )
    )
    (job_dir / "exit.json").write_text(json.dumps({"status": "exited", "returncode": 1}))
    (job_dir / "stdout.log").write_text("")
    (job_dir / "stderr.log").write_text("custom state dir failed\n")
    (stale_phase4_dir / "results.json").write_text(
        json.dumps([{"task_id": "stale", "final_status": "task_broke"}])
    )

    env = _base_env(repo_root, tmp_path)
    env["PATH"] = f"{fakebin}:{env['PATH']}"

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "remote_job_status.sh"),
            "--host-config",
            str(host_config),
            "--remote-dir",
            str(remote_dir),
            "--job-id",
            job_id,
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "missing logs/custom_run/phase_4/results.json" in completed.stdout
    assert "phase4_results: present logs/phase_4/results.json" not in completed.stdout
    assert "task_broke=1" not in completed.stdout


def test_status_does_not_report_default_phase4_for_non_phase4_job(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    host_config = _host_config(tmp_path)
    remote_dir = tmp_path / "remote" / "browser-sim"
    job_id = "20260429t000000z-audit-job-abcdef"
    job_dir = remote_dir / "logs" / "remote_jobs" / job_id
    stale_phase4_dir = remote_dir / "logs" / "phase_4"
    job_dir.mkdir(parents=True)
    stale_phase4_dir.mkdir(parents=True)
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    _write_executable(
        fakebin / "ssh",
        """#!/usr/bin/env python3
import subprocess
import sys

args = sys.argv[1:]
remote_cmd = args[-1]
raise SystemExit(subprocess.run(["bash", "-lc", remote_cmd], stdin=sys.stdin).returncode)
""",
    )
    (job_dir / "metadata.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "name": "audit job",
                "created_at": "2026-04-29T00:00:00+00:00",
                "pid": 999999,
                "pgid": 999999,
                "remote_dir": str(remote_dir),
                "command": [
                    "bash",
                    "-lc",
                    "uv run python scripts/audit_phase_4_variants.py logs/run/phase_4/results.json",
                ],
                "original_command": [
                    "uv",
                    "run",
                    "python",
                    "scripts/audit_phase_4_variants.py",
                    "logs/run/phase_4/results.json",
                ],
                "expected_outputs": [],
            }
        )
    )
    (job_dir / "exit.json").write_text(json.dumps({"status": "exited", "returncode": 0}))
    (job_dir / "stdout.log").write_text("audit output\n")
    (job_dir / "stderr.log").write_text("")
    (stale_phase4_dir / "results.json").write_text(
        json.dumps([{"task_id": "stale", "final_status": "task_broke"}])
    )

    env = _base_env(repo_root, tmp_path)
    env["PATH"] = f"{fakebin}:{env['PATH']}"

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "remote_job_status.sh"),
            "--host-config",
            str(host_config),
            "--remote-dir",
            str(remote_dir),
            "--job-id",
            job_id,
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "phase4_results:" not in completed.stdout
    assert "task_broke=1" not in completed.stdout
    assert "audit output" in completed.stdout


def test_graceful_stop_waits_for_authoritative_pause_before_term(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    host_config, job_id, remote_dir = _start_control_test_job(
        tmp_path, fakebin=fakebin, pause_mode="paused"
    )
    try:
        completed = _run_graceful_stop(repo_root, host_config, remote_dir, job_id, fakebin)
        assert completed.returncode == 0, completed.stderr
        assert "Pause acknowledged" in completed.stdout
        assert f"stopped job {job_id} with TERM" in completed.stdout
        job_dir, _ = _control_job_paths(remote_dir, job_id)
        stop = json.loads((job_dir / "stop.json").read_text(encoding="utf-8"))
        exit_data = json.loads((job_dir / "exit.json").read_text(encoding="utf-8"))
        assert stop["status"] == "term_sent"
        assert stop["control"] == "graceful"
        assert stop["pause_request_id"] == "pause-" + "1" * 32
        assert stop["run_id"] == "run-control-test"
        assert stop["definition_digest"] == "a" * 64
        assert stop["observed_status"] == "paused"
        assert exit_data["signal"] == "TERM"
    finally:
        _cleanup_control_job(remote_dir, job_id)


def test_graceful_pause_outcomes_never_send_term(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for mode in ("timeout", "terminal", "unsupported", "swap"):
        case_dir = tmp_path / mode
        case_dir.mkdir()
        fakebin = case_dir / "bin"
        fakebin.mkdir()
        host_config, job_id, remote_dir = _start_control_test_job(
            case_dir, fakebin=fakebin, pause_mode=mode
        )
        try:
            completed = _run_graceful_stop(repo_root, host_config, remote_dir, job_id, fakebin)
            assert completed.returncode != 0, (mode, completed.stdout, completed.stderr)
            job_dir, _ = _control_job_paths(remote_dir, job_id)
            stop = json.loads((job_dir / "stop.json").read_text(encoding="utf-8"))
            assert stop["status"] == "pause_rejected", (mode, stop)
            assert "term_sent" not in stop["status"]
            assert not (job_dir / "exit.json").exists(), mode
            assert _assert_control_process_alive(remote_dir, job_id), mode
        finally:
            _cleanup_control_job(remote_dir, job_id)


def test_graceful_stop_rejects_missing_state_or_identity_metadata(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for mutate in ("state_dir", "identity"):
        case_dir = tmp_path / mutate
        case_dir.mkdir()
        fakebin = case_dir / "bin"
        fakebin.mkdir()
        host_config, job_id, remote_dir = _start_control_test_job(
            case_dir, fakebin=fakebin, pause_mode="paused"
        )
        try:
            job_dir, metadata_path = _control_job_paths(remote_dir, job_id)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if mutate == "state_dir":
                metadata.pop("state_dir", None)
            else:
                metadata.pop("process_start_ticks", None)
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            completed = _run_graceful_stop(repo_root, host_config, remote_dir, job_id, fakebin)
            assert completed.returncode != 0
            stop = json.loads((job_dir / "stop.json").read_text(encoding="utf-8"))
            assert stop["status"] == "pause_rejected"
            assert not (job_dir / "exit.json").exists()
            assert _assert_control_process_alive(remote_dir, job_id)
        finally:
            _cleanup_control_job(remote_dir, job_id)


def test_graceful_stop_records_stale_live_process_rejection(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    host_config, job_id, remote_dir = _start_control_test_job(
        tmp_path, fakebin=fakebin, pause_mode="paused"
    )
    try:
        job_dir, metadata_path = _control_job_paths(remote_dir, job_id)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["process_start_ticks"] = "stale-start-ticks"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        completed = _run_graceful_stop(repo_root, host_config, remote_dir, job_id, fakebin)
        assert completed.returncode != 0
        stop = json.loads((job_dir / "stop.json").read_text(encoding="utf-8"))
        assert stop["status"] == "pause_rejected"
        assert stop["control"] == "graceful"
        assert "start time" in stop["reason"]
        assert not (job_dir / "exit.json").exists()
        assert _assert_control_process_alive(remote_dir, job_id)
    finally:
        _cleanup_control_job(remote_dir, job_id)


def test_graceful_stop_ssh_failure_cannot_signal_local_process(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    host_config, job_id, remote_dir = _start_control_test_job(
        tmp_path, fakebin=fakebin, pause_mode="paused"
    )
    try:
        _write_executable(fakebin / "ssh", "#!/bin/sh\nexit 42\n")
        completed = _run_graceful_stop(repo_root, host_config, remote_dir, job_id, fakebin)
        assert completed.returncode == 42
        assert _assert_control_process_alive(remote_dir, job_id)
        job_dir, _ = _control_job_paths(remote_dir, job_id)
        assert not (job_dir / "stop.json").exists()
    finally:
        _cleanup_control_job(remote_dir, job_id)


def test_status_json_keeps_job_and_run_identity_distinct(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    host_config, job_id, remote_dir = _start_control_test_job(
        tmp_path, fakebin=fakebin, pause_mode="paused"
    )
    try:
        env = _base_env(repo_root, tmp_path)
        env["PATH"] = f"{fakebin}:{env['PATH']}"
        completed = subprocess.run(
            [
                "bash",
                str(repo_root / "scripts" / "remote_job_status.sh"),
                "--host-config",
                str(host_config),
                "--remote-dir",
                str(remote_dir),
                "--job-id",
                job_id,
                "--json",
            ],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        payload = json.loads(completed.stdout)
        assert payload["remote_job_id"] == job_id
        assert payload["run_id"] == "run-control-test"
        assert payload["definition_digest"] == "a" * 64
        assert payload["run_state_dir"].endswith("logs/run")
        assert payload["remote_job_id"] != payload["run_id"]
    finally:
        _cleanup_control_job(remote_dir, job_id)


def test_stop_requires_job_id_and_rejects_patterns(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    host_config = _host_config(tmp_path)
    env = _base_env(repo_root, tmp_path)

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "remote_job_stop.sh"),
            "--host-config",
            str(host_config),
            "uv run python",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "unknown arg" in completed.stderr
    assert "pkill" not in completed.stderr

    missing = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "remote_job_stop.sh"),
            "--host-config",
            str(host_config),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert missing.returncode == 2
    assert "--job-id required" in missing.stderr
