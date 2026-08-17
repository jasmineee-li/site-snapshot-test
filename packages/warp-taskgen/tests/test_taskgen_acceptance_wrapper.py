from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
ACCEPTANCE = REPO_ROOT / "scripts" / "accept_taskgen.sh"
LANES = ("package-proof", "core-tests", "remote-job-tests")


def _run_wrapper(
    *args: str,
    changed_files: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("TASKGEN_ACCEPTANCE_FORCE", None)
    if changed_files is not None:
        env["TASKGEN_ACCEPTANCE_CHANGED_FILES"] = changed_files
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(ACCEPTANCE), *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_help_describes_full_and_named_lanes() -> None:
    result = _run_wrapper("--help")

    assert result.returncode == 0
    assert "full" in result.stdout
    for lane in LANES:
        assert lane in result.stdout


def test_unknown_lane_is_rejected() -> None:
    result = _run_wrapper("--lane", "not-a-lane")

    assert result.returncode == 2
    assert "not-a-lane" in result.stderr


@pytest.mark.parametrize(
    ("changed_files", "expected"),
    [("README.md\n", "skip"), ("packages/warp-taskgen/README.md\n", "run")],
)
def test_route_only_reports_machine_readable_decision(changed_files: str, expected: str) -> None:
    result = _run_wrapper("--route-only", changed_files=changed_files)

    assert result.returncode == 0
    assert result.stdout == f"{expected}\n"
    assert result.stderr == ""


def test_route_only_unresolved_base_fails_open_with_clean_stdout() -> None:
    missing_ref = "origin/does-not-exist"

    result = _run_wrapper(
        "--route-only",
        extra_env={
            "GITHUB_BASE_REF": "does-not-exist",
            "GITHUB_BASE_SHA": missing_ref,
        },
    )

    assert result.returncode == 0
    assert result.stdout == "run\n"
    assert missing_ref in result.stderr


@pytest.mark.parametrize("args", [(), ("--lane", "full"), *(("--lane", lane) for lane in LANES)])
def test_full_and_named_lanes_preserve_unrelated_change_noop(args: tuple[str, ...]) -> None:
    result = _run_wrapper(*args, changed_files="README.md\n")

    assert result.returncode == 0
    assert "skip" in result.stdout
    assert "uv sync" not in result.stdout


@pytest.mark.parametrize(
    ("lane", "required", "forbidden"),
    [
        (
            "core-tests",
            "--ignore tests/test_remote_job_scripts.py",
            "--dist load tests/test_remote_job_scripts.py",
        ),
        (
            "remote-job-tests",
            "--dist load tests/test_remote_job_scripts.py",
            "--ignore tests/test_remote_job_scripts.py",
        ),
    ],
)
def test_test_lanes_partition_remote_job_file(
    tmp_path: Path, lane: str, required: str, forbidden: str
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    command_log = tmp_path / "uv-commands.log"
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        '#!/usr/bin/env bash\nprintf \'%s\\n\' "$*" >> "$TASKGEN_TEST_UV_LOG"\n',
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    result = _run_wrapper(
        "--lane",
        lane,
        extra_env={
            "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
            "TASKGEN_ACCEPTANCE_FORCE": "1",
            "TASKGEN_TEST_UV_LOG": str(command_log),
        },
    )

    assert result.returncode == 0, result.stderr
    commands = command_log.read_text(encoding="utf-8")
    assert required in commands
    assert forbidden not in commands


def test_package_proof_uses_ordinary_versioned_pip_upgrade() -> None:
    source = ACCEPTANCE.read_text(encoding="utf-8")

    assert "tests/fixtures/namespace_compatibility/adapter_wheel_0_1_0" in source
    assert 'python" -m pip install --upgrade --no-deps' in source
    assert "pip install --force-reinstall" not in source
    assert "find_spec('worldsim') is None" in source


def test_package_proof_smokes_site_composition_check_from_wheel_and_sdist() -> None:
    source = ACCEPTANCE.read_text(encoding="utf-8")

    assert "site composition check gitlab --benchmark webarena_verified" in source
    assert "site composition check reddit --benchmark webarena_verified" in source
    assert "site composition check classifieds --benchmark visualwebarena" in source
    assert "--use-case public_reply --carrier listing_reply.body" in source
    assert "--action-kind answer_opposite_binary_label --json" in source
    assert 'sdists=("$build_dir"/*.tar.gz)' in source
    assert "from warp_taskgen.site_composition import default_site_compositions" in source
    assert "joinpath('site_compositions', 'classifieds.py').is_file()" in source
    assert "missing sdist Site Composition resources" in source
    assert "omitted sdist Site Composition resource was accepted" in source
    assert "report['error'] == 'ModuleNotFoundError'" in source
    assert "source_package_version='0.0.0'" in source
    assert "source package version is incompatible" in source
    assert 'cd "$temporary_root"' in source
