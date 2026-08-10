from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
ACCEPTANCE = REPO_ROOT / "scripts" / "accept_taskgen.sh"
LANES = ("package-proof", "core-tests", "remote-job-tests")


def _run_wrapper(*args: str, changed_files: str | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("TASKGEN_ACCEPTANCE_FORCE", None)
    if changed_files is not None:
        env["TASKGEN_ACCEPTANCE_CHANGED_FILES"] = changed_files
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


@pytest.mark.parametrize("args", [(), ("--lane", "full"), *(("--lane", lane) for lane in LANES)])
def test_full_and_named_lanes_preserve_unrelated_change_noop(args: tuple[str, ...]) -> None:
    result = _run_wrapper(*args, changed_files="README.md\n")

    assert result.returncode == 0
    assert "skip" in result.stdout
    assert "uv sync" not in result.stdout
