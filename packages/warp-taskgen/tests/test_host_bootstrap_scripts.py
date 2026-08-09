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


def test_r8a_wrappers_use_r8a_host_config_and_control_plane_audit() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    bootstrap = (repo_root / "scripts" / "bootstrap_r8a.sh").read_text()
    proxy = (repo_root / "scripts" / "deploy_proxy_r8a.sh").read_text()

    assert "configs/benchmark_hosts/r8a.yaml" in bootstrap
    assert "r8a_host_config.sh" in bootstrap
    assert "FORWARD_ARGS" in bootstrap
    assert "scale_config.r8a-24x24.yml" in bootstrap
    assert "audit_r8a_control_plane.sh" in bootstrap
    assert "configs/benchmark_hosts/r5.yaml" not in bootstrap
    assert "configs/benchmark_hosts/r8a.yaml" in proxy
    assert "r8a_host_config.sh" in proxy
    assert "FORWARD_ARGS" in proxy
    assert "audit_r8a_control_plane.sh" in proxy
    assert "configs/benchmark_hosts/r5.yaml" not in proxy


def test_r8a_wrappers_refuse_public_template() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = _base_env(repo_root)

    for script in ("bootstrap_r8a.sh", "deploy_proxy_r8a.sh"):
        completed = subprocess.run(
            [
                "bash",
                str(repo_root / "scripts" / script),
                "--host-config",
                "configs/benchmark_hosts/r8a.yaml",
            ],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 2
        assert "refuses the tracked public template" in completed.stderr


def test_r8a_wrappers_help_is_available_without_host_config() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = _base_env(repo_root)

    for script in ("bootstrap_r8a.sh", "deploy_proxy_r8a.sh"):
        completed = subprocess.run(
            ["bash", str(repo_root / "scripts" / script), "--help"],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 0
        assert "gitignored *.local.yaml overlay" in completed.stdout


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


def test_setup_phase4_on_host_rejects_r8a_with_r5_scale_config() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "setup_phase4_on_host.sh"),
            "--host-config",
            "configs/benchmark_hosts/r8a.yaml",
            "--scale-config",
            "scripts/scale_config.yml",
            "--skip-gitlab-mint",
        ],
        cwd=repo_root,
        env=_base_env(repo_root),
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "r8a setup requires scripts/scale_config.r8a-24x24.yml" in completed.stderr


def test_setup_phase4_on_host_audits_r8a_before_regenerating_topology(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    audit = tmp_path / "audit-r8a"

    _write_fake_executable(
        fakebin / "uv",
        """#!/bin/sh
if [ "$1" = "run" ] && [ "$2" = "python" ] && [ "$3" = "-c" ]; then
  printf 'r8a\\n'
  exit 0
fi
if [ "$1" = "lock" ] && [ "$2" = "--check" ]; then
  exit 0
fi
if [ "$1" = "sync" ] && [ "$2" = "--locked" ]; then
  exit 0
fi
exit 0
""",
    )
    _write_fake_executable(
        audit,
        """#!/bin/sh
printf 'r8a audit failed\\n' >&2
exit 42
""",
    )

    env = _base_env(repo_root)
    env["PATH"] = f"{fakebin}:{env['PATH']}"
    env["WORLDSIM_R8A_CONTROL_PLANE_AUDIT"] = str(audit)

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "setup_phase4_on_host.sh"),
            "--host-config",
            "configs/benchmark_hosts/r8a.yaml",
            "--skip-gitlab-mint",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 42
    assert "r8a audit failed" in completed.stderr
    assert "step 1b: regen" not in completed.stderr


def test_run_integration_tests_fails_fast_when_playwright_browser_missing(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    instances = tmp_path / "instances.smoke.json"
    instances.write_text("{}")

    _write_fake_executable(
        fakebin / "uv",
        """#!/bin/sh
if [ "$1" = "run" ] && [ "$2" = "python" ] && [ "$3" = "-" ] && [ -n "${INSTANCES_FILE_VALUE:-}" ]; then
  echo "export LIVE_INSTANCES_FILE='$INSTANCES_FILE_VALUE'"
  echo "export LIVE_PHASE2_ARTIFACT='$PWD/logs/phase_2/adversarial_tasks.json'"
  echo "export LIVE_PHASE2C_ARTIFACT='$PWD/logs/phase_2/feasibility_report.json'"
  echo "export LIVE_HOST_IP=''"
  exit 0
fi
if [ "$1" = "run" ] && [ "$2" = "python" ] && [ "$3" = "-" ]; then
  echo "ERROR: Playwright Chromium is not installed for this environment." >&2
  echo "Run: uv run python -m playwright install chromium" >&2
  exit 2
fi
if [ "$1" = "run" ] && [ "$2" = "pytest" ]; then
  echo "pytest should not run before Playwright preflight" >&2
  exit 99
fi
exit 0
""",
    )

    env = _base_env(repo_root)
    env["PATH"] = f"{fakebin}:{env['PATH']}"

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "run_integration_tests.sh"),
            "--instances",
            str(instances),
            "--quiet",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "Playwright Chromium is not installed" in completed.stderr
    assert "uv run python -m playwright install chromium" in completed.stderr
    assert "pytest should not run" not in completed.stderr


def test_run_integration_tests_bootstraps_home_local_uv_path(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    home = tmp_path / "home"
    fakebin = home / ".local" / "bin"
    fakebin.mkdir(parents=True)
    instances = tmp_path / "instances.smoke.json"
    instances.write_text("{}")

    _write_fake_executable(
        fakebin / "uv",
        """#!/bin/sh
if [ "$1" = "run" ] && [ "$2" = "python" ] && [ "$3" = "-" ] && [ -n "${INSTANCES_FILE_VALUE:-}" ]; then
  echo "export LIVE_INSTANCES_FILE='$INSTANCES_FILE_VALUE'"
  echo "export LIVE_PHASE2_ARTIFACT='$PWD/logs/phase_2/adversarial_tasks.json'"
  echo "export LIVE_PHASE2C_ARTIFACT='$PWD/logs/phase_2/feasibility_report.json'"
  echo "export LIVE_HOST_IP=''"
  exit 0
fi
if [ "$1" = "run" ] && [ "$2" = "python" ] && [ "$3" = "-" ]; then
  exit 0
fi
if [ "$1" = "run" ] && [ "$2" = "pytest" ]; then
  echo "======================== 1 passed in 0.01s ========================"
  exit 0
fi
exit 99
""",
    )

    env = _base_env(repo_root)
    env["HOME"] = str(home)
    env["PATH"] = "/usr/bin:/bin"

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "run_integration_tests.sh"),
            "--instances",
            str(instances),
            "--quiet",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "1 passed" in completed.stdout


# test_sync_magento_base_urls_fails_when_repair_not_applied_without_verify_after
# removed 2026-04-21 with the WASP-aligned scoping decision (see
# docs/handoffs/wasp-aligned-scoping-decision.md).
