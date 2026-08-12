from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from worldsim import main as worldsim_main


def test_standalone_preflight_runs_from_package_root(monkeypatch, tmp_path: Path) -> None:
    package_root = Path(__file__).resolve().parents[1]
    host_config = tmp_path / "host.yaml"
    instances = tmp_path / "instances.json"
    host_config.write_text("host: test\n", encoding="utf-8")
    instances.write_text("{}\n", encoding="utf-8")
    observed: dict[str, object] = {}

    def fake_run(
        command: list[str], *, cwd: Path, env: dict[str, str]
    ) -> subprocess.CompletedProcess[str]:
        observed.update(command=command, cwd=cwd, env=env)
        assert cwd == package_root
        assert (cwd / "tests" / "preflight").is_dir()
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = worldsim_main.main(
        [
            "preflight",
            "--host-config",
            str(host_config),
            "--instances",
            str(instances),
            "--",
            "-q",
        ]
    )

    assert result == 0
    assert observed["command"] == [
        sys.executable,
        "-m",
        "pytest",
        "-m",
        "preflight",
        "tests/preflight",
        "-q",
    ]


def test_standalone_preflight_resolves_relative_configs_from_package_root(
    monkeypatch,
) -> None:
    package_root = Path(__file__).resolve().parents[1]
    observed: dict[str, object] = {}

    def fake_run(
        command: list[str], *, cwd: Path, env: dict[str, str]
    ) -> subprocess.CompletedProcess[str]:
        observed.update(command=command, cwd=cwd, env=env)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = worldsim_main.main(
        [
            "preflight",
            "--host-config",
            "configs/benchmark_hosts/r8a.yaml",
            "--instances",
            "instances.example.json",
        ]
    )

    assert result == 0
    env = observed["env"]
    assert isinstance(env, dict)
    assert env["WORLDSIM_PREFLIGHT_HOST_CONFIG"] == str(
        package_root / "configs/benchmark_hosts/r8a.yaml"
    )
    assert env["WORLDSIM_PREFLIGHT_INSTANCES"] == str(package_root / "instances.example.json")
