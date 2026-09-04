from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPO_ROOT / "packages" / "warp-taskgen"
ACCEPTANCE = REPO_ROOT / "scripts" / "accept_taskgen.sh"
RUN_SILENT = REPO_ROOT / "packages" / "warp-taskgen" / "scripts" / "lib" / "run_silent.sh"
LANES = (
    "package-proof",
    "core-context-boundaries",
    "core-feature-tests",
    "remote-job-tests",
)
CORE_CONTEXT_BOUNDARY_FILES = (
    "tests/phase_2/phase_2c/test_stage_context_boundary.py",
    "tests/phase_2/test_eligibility_context_boundary.py",
    "tests/phase_2/test_generation_context_boundary.py",
    "tests/phase_2/test_option_a_context_boundary.py",
    "tests/phase_2/test_plan_validation_context_boundary.py",
    "tests/phase_2/test_shard_context_boundary.py",
    "tests/phase_2/test_target_context_boundary.py",
    "tests/phase_4/test_leaf_context_boundary.py",
)
CORE_REMOTE_IGNORES = (
    "--ignore",
    "tests/test_remote_job_scripts.py",
    "--ignore",
    "tests/test_remote_job_decisions.py",
)


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


def _run_silent(
    description: str,
    command: str,
    *,
    show_success_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if show_success_output:
        env["RUN_SILENT_SHOW_SUCCESS_OUTPUT"] = "1"
    else:
        env.pop("RUN_SILENT_SHOW_SUCCESS_OUTPUT", None)
    return subprocess.run(
        ["bash", str(RUN_SILENT), description, command],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _collect_nodes(*args: str) -> set[str]:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *args],
        cwd=PACKAGE_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return {
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip().startswith("tests/") and "::" in line
    }


def test_help_describes_full_and_named_lanes() -> None:
    result = _run_wrapper("--help")

    assert result.returncode == 0
    assert "full" in result.stdout
    for lane in LANES:
        assert lane in result.stdout


def test_core_selections_are_disjoint_and_cover_former_core_collection() -> None:
    discovered_context_files = tuple(
        sorted(
            path.relative_to(PACKAGE_ROOT).as_posix()
            for path in (PACKAGE_ROOT / "tests").rglob("*context_boundary.py")
        )
    )
    assert discovered_context_files == tuple(sorted(CORE_CONTEXT_BOUNDARY_FILES))

    baseline = _collect_nodes(*CORE_REMOTE_IGNORES)
    context = _collect_nodes(*CORE_CONTEXT_BOUNDARY_FILES, *CORE_REMOTE_IGNORES)
    feature = _collect_nodes(
        "--ignore-glob=*context_boundary.py",
        *CORE_REMOTE_IGNORES,
    )

    assert context
    assert context.isdisjoint(feature)
    assert context | feature == baseline
    assert {node.split("::", 1)[0] for node in context} == set(CORE_CONTEXT_BOUNDARY_FILES)
    assert not any(
        node.startswith("tests/test_remote_job_scripts.py::")
        or node.startswith("tests/test_remote_job_decisions.py::")
        for node in baseline
    )


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


def test_run_silent_keeps_success_quiet_by_default() -> None:
    result = _run_silent("quiet success", "printf 'pytest summary\\n'")

    assert result.returncode == 0
    assert result.stdout == "  ✓ quiet success\n"


def test_run_silent_can_expose_success_output() -> None:
    result = _run_silent(
        "visible success",
        "printf '2 passed, 1 skipped in 0.10s\\n'",
        show_success_output=True,
    )

    assert result.returncode == 0
    assert "2 passed, 1 skipped in 0.10s" in result.stdout
    assert result.stdout.endswith("  ✓ visible success\n")


def test_run_silent_preserves_failure_output_with_success_output_enabled() -> None:
    result = _run_silent(
        "failed command",
        "sh -c \"printf 'pytest failure details\\n'; exit 7\"",
        show_success_output=True,
    )

    assert result.returncode == 7
    assert "pytest failure details" in result.stdout
    assert "  ✗ failed command\n" in result.stdout


@pytest.mark.parametrize("lane", ("core-context-boundaries", "core-feature-tests"))
def test_core_lane_exposes_pytest_summary_and_slowest_nodes(tmp_path: Path, lane: str) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    command_log = tmp_path / "uv-commands.log"
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "$*" >> "$TASKGEN_TEST_UV_LOG"\n'
        "printf '3 passed, 1 skipped in 0.10s\\n'\n"
        "printf '%s\\n' 'slowest 20 durations'\n",
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
    assert "3 passed, 1 skipped in 0.10s" in result.stdout
    assert "slowest 20 durations" in result.stdout
    command = command_log.read_text(encoding="utf-8")
    assert "-q -n 4 --dist worksteal --durations=20" in command
    assert "--ignore tests/test_remote_job_scripts.py" in command
    assert "--ignore tests/test_remote_job_decisions.py" in command
    if lane == "core-context-boundaries":
        assert "--ignore-glob=*context_boundary.py" not in command
        for path in CORE_CONTEXT_BOUNDARY_FILES:
            assert path in command
    else:
        assert "--ignore-glob=*context_boundary.py" in command
        for path in CORE_CONTEXT_BOUNDARY_FILES:
            assert path not in command


@pytest.mark.parametrize(
    ("lane", "required", "forbidden"),
    [
        (
            "core-context-boundaries",
            "--ignore tests/test_remote_job_scripts.py",
            "--dist load tests/test_remote_job_scripts.py",
        ),
        (
            "core-feature-tests",
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
