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


def test_setup_phase4_on_host_fails_when_playwright_install_deps_fails(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    fakebin = tmp_path / "bin"
    fakebin.mkdir()

    _write_fake_executable(
        fakebin / "uv",
        """#!/bin/sh
if [ "$1" = "lock" ] && [ "$2" = "--check" ]; then
  exit 0
fi
if [ "$1" = "sync" ] && [ "$2" = "--locked" ]; then
  exit 0
fi
if [ "$1" = "run" ] && [ "$2" = "python" ] && [ "$3" = "-m" ] && [ "$4" = "playwright" ] && [ "$5" = "install" ] && [ "$6" = "chromium" ]; then
  exit 0
fi
if [ "$1" = "run" ] && [ "$2" = "python" ] && [ "$3" = "-m" ] && [ "$4" = "playwright" ] && [ "$5" = "install-deps" ] && [ "$6" = "chromium" ]; then
  exit 1
fi
exit 0
""",
    )
    _write_fake_executable(
        fakebin / "sudo",
        """#!/bin/sh
"$@"
""",
    )

    env = _base_env(repo_root)
    env["PATH"] = f"{fakebin}:{env['PATH']}"

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "setup_phase4_on_host.sh"),
            "--skip-pvpo-container",
            "--skip-magento-sync",
            "--skip-gitlab-mint",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert completed.stderr


# test_sync_magento_base_urls_fails_when_repair_not_applied_without_verify_after
# removed 2026-04-21 with the WASP-aligned scoping decision (see
# docs/handoffs/wasp-aligned-scoping-decision.md).
