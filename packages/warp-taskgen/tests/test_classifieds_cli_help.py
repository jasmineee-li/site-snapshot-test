from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PACKAGE_ROOT / "scripts"


PYTHON_HELP_CASES = (
    (
        "bootstrap_classifieds_writer.py",
        ("--site-url", "--output-path"),
    ),
    (
        "preflight_classifieds_canary.py",
        ("--run-dir", "--instances", "--expected-task-id", "--output"),
    ),
    (
        "prepare_classifieds_canary.py",
        ("--site-url", "--listing-id", "--run-dir", "--overlay-path"),
    ),
    (
        "record_classifieds_canary_images.py",
        ("--web-image-ref", "--db-image-ref", "--output"),
    ),
    (
        "run_classifieds_canary.py",
        ("--host-config", "--run-dir", "--timeout-seconds"),
    ),
    (
        "verify_classifieds_canary_completion.py",
        ("--run-dir", "--expected-task-id"),
    ),
)


def _help(command: list[str]) -> str:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(PACKAGE_ROOT), env.get("PYTHONPATH")) if item
    )
    result = subprocess.run(
        command,
        cwd=PACKAGE_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout + result.stderr


@pytest.mark.parametrize(("script", "options"), PYTHON_HELP_CASES)
def test_classifieds_python_help_describes_inputs_outputs_and_safety(
    script: str, options: tuple[str, ...]
) -> None:
    output = _help([sys.executable, str(SCRIPT_ROOT / script), "--help"])

    assert "Inputs:" in output
    assert "Output:" in output
    assert "Safety:" in output
    for option in options:
        assert option in output


@pytest.mark.parametrize("mode", ("precondition", "write-read", "absence"))
def test_classifieds_probe_help_names_each_mode_and_its_contract(mode: str) -> None:
    output = _help(
        [sys.executable, str(SCRIPT_ROOT / "classifieds_canary_probe.py"), mode, "--help"]
    )

    assert mode in output
    assert "--site-url" in output
    assert "--listing-id" in output
    assert "--evidence" in output
    assert "Safety:" in output or "anonymous" in output


def test_classifieds_remote_help_lists_sanitized_inputs_and_completion_artifact(
    tmp_path: Path,
) -> None:
    env = os.environ.copy()
    env["HOME"] = str(tmp_path)
    result = subprocess.run(
        ["bash", str(SCRIPT_ROOT / "run_classifieds_canary_remote.sh"), "--help"],
        cwd=PACKAGE_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr

    for option in (
        "--run-dir",
        "--site-url",
        "--listing-id",
        "--overlay-path",
        "--project-name",
        "--network",
        "--web-port",
        "--instances",
        "--writer-storage-state",
        "--app-env-file",
        "--web-image-ref",
        "--db-image-ref",
        "--source-commit",
    ):
        assert option in output
    assert "completion.json" in output
    assert "sanitized" in output
    assert "host YAML" in output
