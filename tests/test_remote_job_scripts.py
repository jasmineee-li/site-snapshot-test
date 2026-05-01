from __future__ import annotations

import json
import os
import subprocess
import sys
import time
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
            str(repo_root / "scripts" / "sync_to_r5.sh"),
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
    assert ".env" in joined
    assert ".git" in joined
    assert ".git/" in joined
    assert ".venv/" in joined
    assert ".venv.*" in joined
    assert ".codex-worktrees/" in joined
    assert "logs/" in joined
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
    (scripts_dir / "sync_to_r5.sh").write_text(
        (repo_root / "scripts" / "sync_to_r5.sh").read_text()
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
            str(worktree_root / "scripts" / "sync_to_r5.sh"),
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
            str(repo_root / "scripts" / "sync_to_r5.sh"),
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

    assert completed.returncode == 2
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

    assert completed.returncode == 2
    assert "instance-topology guard blocked" in completed.stderr
    assert "orchestrator_host=172.17.0.1" in completed.stderr
    assert "--feasibility-instances instances.smoke.json" in completed.stderr
    assert "instances.scale.json" in completed.stderr


def test_start_rejects_phase0_scale_instances_on_remote_orchestrator_host(
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
            "bad phase0 topology",
            "--",
            "bash",
            "-lc",
            "uv run python -m worldsim.main phase 0 --instances instances.scale.json",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "Phase 0c runs inside Modal sandboxes" in completed.stderr
    assert "instances.smoke.json" in completed.stderr
    assert "instances.scale.json" in completed.stderr


def test_start_allows_chained_phase0_smoke_and_phase2_scale_topology(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    host_config = _remote_direct_host_config_with_orchestrator(tmp_path)
    remote_dir = tmp_path / "remote" / "browser-sim"
    remote_dir.mkdir(parents=True)
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


def test_start_rejects_chained_phase1_novel_without_benchmark(
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
            "missing phase1 benchmark",
            "--",
            "bash",
            "-lc",
            (
                "uv run python -m worldsim.main phase 0 "
                "--benchmark vendors/webarena-verified "
                "--instances instances.smoke.json && "
                "uv run python -m worldsim.main phase 1 "
                "--generate-novel --sites gitlab,reddit && "
                "uv run python -m worldsim.main phase 2 "
                "--feasibility-instances instances.scale.json"
            ),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "Chained Phase 0 -> Phase 1 novel generation" in completed.stderr
    assert "--benchmark or --config on the Phase 1 command" in completed.stderr


def test_start_allows_chained_phase1_novel_with_explicit_benchmark(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    host_config = _remote_direct_host_config_with_orchestrator(tmp_path)
    remote_dir = tmp_path / "remote" / "browser-sim"
    remote_dir.mkdir(parents=True)
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
                ": worldsim.main phase 0 --benchmark vendors/webarena-verified "
                "--instances instances.smoke.json && "
                ": worldsim.main phase 1 --benchmark vendors/webarena-verified "
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


def test_start_rejects_phase2_default_smoke_instances_on_remote_orchestrator_host(
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
            "bad phase2 default",
            "--",
            "uv",
            "run",
            "python",
            "-m",
            "worldsim.main",
            "phase",
            "2",
            "--sites",
            "gitlab,reddit",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "CLI default is instances.smoke.json" in completed.stderr
    assert "--feasibility-instances instances.scale.json" in completed.stderr


def test_start_rejects_phase4_smoke_instances_on_remote_orchestrator_host(
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
            "bad phase4",
            "--",
            "uv",
            "run",
            "python",
            "-m",
            "worldsim.main",
            "phase",
            "4",
            "--instances",
            "instances.smoke.json",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "Phase 4 uses --instances instances.smoke.json" in completed.stderr
    assert "host-bound storage_state mismatches" in completed.stderr


def test_start_allows_scale_instances_on_remote_orchestrator_host(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    host_config = _remote_direct_host_config_with_orchestrator(tmp_path)
    remote_dir = tmp_path / "remote" / "browser-sim"
    remote_dir.mkdir(parents=True)
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
            "import time; time.sleep(1)",
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
    (job_dir / "exit.json").write_text(
        json.dumps({"status": "launch_failed", "returncode": 127})
    )
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
