from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from root_gate_router import requires_root_gate

REPO_ROOT = Path(__file__).resolve().parents[1]
ROUTER = REPO_ROOT / "scripts" / "root_gate_router.py"


def test_taskgen_tree_and_acceptance_entrypoints_skip() -> None:
    assert not requires_root_gate(
        [
            "packages/warp-taskgen/warp_taskgen/main.py",
            "./packages/warp-taskgen/tests/test_main.py",
            ".github/workflows/taskgen-acceptance.yml",
            "scripts/accept_taskgen.sh",
        ]
    )


@pytest.mark.parametrize(
    "path",
    [
        "README.md",
        "eval_awareness_experiments/run.py",
        "pyproject.toml",
        "uv.lock",
        ".github/workflows/check-root.yml",
        "scripts/root_gate_router.py",
    ],
)
def test_root_owned_paths_run(path: str) -> None:
    assert requires_root_gate([path])


def test_mixed_changes_run() -> None:
    assert requires_root_gate(["packages/warp-taskgen/README.md", "README.md"])


@pytest.mark.parametrize("paths", [[], ["", "  ", "./"]])
def test_empty_changes_fail_safe_to_run(paths: list[str]) -> None:
    assert requires_root_gate(paths)


def test_untrusted_path_spelling_fails_safe() -> None:
    assert requires_root_gate(["../packages/warp-taskgen/README.md"])
    assert requires_root_gate(["packages/warp-taskgen/../README.md"])


@pytest.mark.parametrize(
    ("stdin", "expected"),
    [
        ("packages/warp-taskgen/README.md\n", "skip\n"),
        ("packages/warp-taskgen/README.md\nREADME.md\n", "run\n"),
        ("", "run\n"),
    ],
)
def test_cli_reads_paths_from_stdin(stdin: str, expected: str) -> None:
    result = subprocess.run(
        [sys.executable, str(ROUTER), "--paths-from-stdin"],
        cwd=REPO_ROOT,
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout == expected
    assert result.stderr == ""
