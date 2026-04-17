from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _base_env(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = os.environ.get("PATH", "")
    env["HOME"] = str(repo_root)
    return env


def test_bootstrap_ec2_requires_explicit_host_contract() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        ["bash", str(repo_root / "scripts" / "bootstrap_ec2.sh")],
        cwd=repo_root,
        env=_base_env(repo_root),
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "missing benchmark host" in completed.stderr


def test_bootstrap_ec2_rejects_raw_public_binds_without_explicit_opt_in() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = _base_env(repo_root)
    env["HOST_IP"] = "203.0.113.10"
    env["WORLDSIM_BIND_HOST"] = "0.0.0.0"
    env["WORLDSIM_DB_BIND_HOST"] = "0.0.0.0"

    completed = subprocess.run(
        ["bash", str(repo_root / "scripts" / "bootstrap_ec2.sh")],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "allow_public_web_bind=true" in completed.stderr


def test_configure_db_access_requires_explicit_host_contract() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        ["bash", str(repo_root / "scripts" / "configure_db_access.sh")],
        cwd=repo_root,
        env=_base_env(repo_root),
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "missing benchmark host" in completed.stderr


def test_deploy_proxy_requires_explicit_host_contract() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        ["bash", str(repo_root / "scripts" / "deploy_benchmark_proxy.sh")],
        cwd=repo_root,
        env=_base_env(repo_root),
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "missing benchmark host" in completed.stderr
