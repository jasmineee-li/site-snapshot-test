from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _base_env(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = os.environ.get("PATH", "")
    env["HOME"] = str(repo_root)
    return env


def _write_fake_executable(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(0o755)


def _write_host_config(path: Path, instance_id: str | None) -> None:
    lines = [
        "name: fakehost",
        "access_mode: remote_direct_restricted",
        "advertise_host: 203.0.113.10",
        "bind_host: 0.0.0.0",
        "allow_public_web_bind: true",
        "allow_public_db_bind: true",
        "region: us-east-2",
        "ssh_user: runner",
    ]
    if instance_id is not None:
        lines.append(f"instance_id: {instance_id}")
    path.write_text("\n".join(lines) + "\n")


_AWS_PREAMBLE = """#!/bin/sh
set -e
cmd="$1"
shift
sub="$1"
shift 2>/dev/null || true
"""


def test_host_park_requires_host_config() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        ["bash", str(repo_root / "scripts" / "host_park.sh")],
        cwd=repo_root,
        env=_base_env(repo_root),
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "--host-config is required" in completed.stderr


def test_host_park_refuses_missing_instance_id(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = tmp_path / "host.yaml"
    _write_host_config(cfg, instance_id=None)

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "host_park.sh"),
            "--host-config",
            str(cfg),
        ],
        cwd=repo_root,
        env=_base_env(repo_root),
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "no instance_id" in completed.stderr


def test_host_park_refuses_when_sweep_tag_set(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = tmp_path / "host.yaml"
    _write_host_config(cfg, instance_id="i-0123456789abcdef0")

    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    fake_aws = (
        _AWS_PREAMBLE
        + """
if [ "$cmd" = "ec2" ] && [ "$sub" = "describe-instances" ]; then
  printf '%s\\n' '{"State":"running","Root":"ebs","InstanceType":"r8a.24xlarge","Tags":[{"Key":"worldsim:sweep-in-progress","Value":"true"}]}'
  exit 0
fi
echo "fake-aws: unhandled $cmd $sub $*" >&2
exit 99
"""
    )
    _write_fake_executable(fakebin / "aws", fake_aws)

    env = _base_env(repo_root)
    env["PATH"] = f"{fakebin}:{env['PATH']}"

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "host_park.sh"),
            "--host-config",
            str(cfg),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1, f"stdout={completed.stdout} stderr={completed.stderr}"
    assert "worldsim:sweep-in-progress=true" in completed.stderr
    assert "refusing to stop" in completed.stderr


def test_host_park_refuses_instance_store_root(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = tmp_path / "host.yaml"
    _write_host_config(cfg, instance_id="i-0123456789abcdef0")

    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    fake_aws = (
        _AWS_PREAMBLE
        + """
if [ "$cmd" = "ec2" ] && [ "$sub" = "describe-instances" ]; then
  printf '%s\\n' '{"State":"running","Root":"instance-store","InstanceType":"i3.large","Tags":[]}'
  exit 0
fi
echo "fake-aws: unhandled $cmd $sub $*" >&2
exit 99
"""
    )
    _write_fake_executable(fakebin / "aws", fake_aws)

    env = _base_env(repo_root)
    env["PATH"] = f"{fakebin}:{env['PATH']}"

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "host_park.sh"),
            "--host-config",
            str(cfg),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2, f"stdout={completed.stdout} stderr={completed.stderr}"
    assert "instance-store" in completed.stderr
    assert "EBS-backed" in completed.stderr


def test_host_park_dry_run_does_not_call_stop(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = tmp_path / "host.yaml"
    _write_host_config(cfg, instance_id="i-0123456789abcdef0")

    stop_marker = tmp_path / "stop_called"
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    fake_aws = (
        _AWS_PREAMBLE
        + f"""
if [ "$cmd" = "ec2" ] && [ "$sub" = "describe-instances" ]; then
  printf '%s\\n' '{{"State":"running","Root":"ebs","InstanceType":"r8a.24xlarge","Tags":[]}}'
  exit 0
fi
if [ "$cmd" = "ec2" ] && [ "$sub" = "stop-instances" ]; then
  : > "{stop_marker}"
  exit 0
fi
echo "fake-aws: unhandled $cmd $sub $*" >&2
exit 99
"""
    )
    _write_fake_executable(fakebin / "aws", fake_aws)

    env = _base_env(repo_root)
    env["PATH"] = f"{fakebin}:{env['PATH']}"

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "host_park.sh"),
            "--host-config",
            str(cfg),
            "--dry-run",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, f"stdout={completed.stdout} stderr={completed.stderr}"
    assert not stop_marker.exists(), "stop-instances was called during --dry-run"
    assert "DRY RUN" in completed.stdout


def test_host_park_idempotent_when_already_stopped(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = tmp_path / "host.yaml"
    _write_host_config(cfg, instance_id="i-0123456789abcdef0")

    stop_marker = tmp_path / "stop_called"
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    fake_aws = (
        _AWS_PREAMBLE
        + f"""
if [ "$cmd" = "ec2" ] && [ "$sub" = "describe-instances" ]; then
  printf '%s\\n' '{{"State":"stopped","Root":"ebs","InstanceType":"r8a.24xlarge","Tags":[]}}'
  exit 0
fi
if [ "$cmd" = "ec2" ] && [ "$sub" = "stop-instances" ]; then
  : > "{stop_marker}"
  exit 0
fi
echo "fake-aws: unhandled $cmd $sub $*" >&2
exit 99
"""
    )
    _write_fake_executable(fakebin / "aws", fake_aws)

    env = _base_env(repo_root)
    env["PATH"] = f"{fakebin}:{env['PATH']}"

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "host_park.sh"),
            "--host-config",
            str(cfg),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, f"stdout={completed.stdout} stderr={completed.stderr}"
    assert not stop_marker.exists(), "stop-instances called on an already-stopped host"
    assert "already stopped" in completed.stdout


def test_host_resume_requires_host_config() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        ["bash", str(repo_root / "scripts" / "host_resume.sh")],
        cwd=repo_root,
        env=_base_env(repo_root),
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "--host-config is required" in completed.stderr


def test_host_resume_sets_sweep_tag_before_start(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = tmp_path / "host.yaml"
    _write_host_config(cfg, instance_id="i-0123456789abcdef0")

    call_log = tmp_path / "aws_calls.log"
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    # describe-instances returns "stopped" so resume actually calls start-instances.
    fake_aws = (
        _AWS_PREAMBLE
        + f"""
printf '%s %s\\n' "$cmd" "$sub" >> "{call_log}"
if [ "$cmd" = "ec2" ] && [ "$sub" = "describe-instances" ]; then
  echo stopped
  exit 0
fi
if [ "$cmd" = "ec2" ] && [ "$sub" = "create-tags" ]; then
  exit 0
fi
if [ "$cmd" = "ec2" ] && [ "$sub" = "start-instances" ]; then
  exit 0
fi
if [ "$cmd" = "ec2" ] && [ "$sub" = "wait" ]; then
  exit 0
fi
if [ "$cmd" = "ec2" ] && [ "$sub" = "describe-instance-status" ]; then
  printf '%s\\n' '{{"System":"ok","Instance":"ok"}}'
  exit 0
fi
echo "fake-aws: unhandled $cmd $sub $*" >&2
exit 99
"""
    )
    _write_fake_executable(fakebin / "aws", fake_aws)

    env = _base_env(repo_root)
    env["PATH"] = f"{fakebin}:{env['PATH']}"

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "host_resume.sh"),
            "--host-config",
            str(cfg),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, f"stdout={completed.stdout} stderr={completed.stderr}"
    calls = call_log.read_text().splitlines()
    create_tags_idx = next((i for i, c in enumerate(calls) if c == "ec2 create-tags"), -1)
    start_idx = next((i for i, c in enumerate(calls) if c == "ec2 start-instances"), -1)
    assert create_tags_idx >= 0, f"create-tags never called; calls={calls}"
    assert start_idx >= 0, f"start-instances never called; calls={calls}"
    assert create_tags_idx < start_idx, f"create-tags must precede start-instances; calls={calls}"


def test_host_resume_no_tag_flag_skips_tag(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = tmp_path / "host.yaml"
    _write_host_config(cfg, instance_id="i-0123456789abcdef0")

    call_log = tmp_path / "aws_calls.log"
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    fake_aws = (
        _AWS_PREAMBLE
        + f"""
printf '%s %s\\n' "$cmd" "$sub" >> "{call_log}"
if [ "$cmd" = "ec2" ] && [ "$sub" = "describe-instances" ]; then
  echo running
  exit 0
fi
if [ "$cmd" = "ec2" ] && [ "$sub" = "describe-instance-status" ]; then
  printf '%s\\n' '{{"System":"ok","Instance":"ok"}}'
  exit 0
fi
echo "fake-aws: unhandled $cmd $sub $*" >&2
exit 99
"""
    )
    _write_fake_executable(fakebin / "aws", fake_aws)

    env = _base_env(repo_root)
    env["PATH"] = f"{fakebin}:{env['PATH']}"

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "host_resume.sh"),
            "--host-config",
            str(cfg),
            "--no-tag",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, f"stdout={completed.stdout} stderr={completed.stderr}"
    assert "ec2 create-tags" not in call_log.read_text()


def test_host_resume_refuses_missing_instance_id(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = tmp_path / "host.yaml"
    _write_host_config(cfg, instance_id=None)

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "host_resume.sh"),
            "--host-config",
            str(cfg),
        ],
        cwd=repo_root,
        env=_base_env(repo_root),
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "no instance_id" in completed.stderr
