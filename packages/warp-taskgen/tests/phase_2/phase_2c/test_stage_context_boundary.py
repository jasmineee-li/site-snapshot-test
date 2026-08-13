"""Structural guards for the Phase 2c stage ownership boundary."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parents[3]
PHASE_2_ROOT = PACKAGE_ROOT / "worldsim" / "phase_2"
STAGE_ROOT = PHASE_2_ROOT / "phase_2c"


def _source(name: str) -> str:
    return (STAGE_ROOT / name).read_text()


def _import_smoke(statement: str) -> None:
    package_root = str(PACKAGE_ROOT)
    subprocess.run(
        [sys.executable, "-c", statement],
        check=True,
        cwd=package_root,
        env={"PYTHONPATH": package_root},
    )


def test_phase_2c_stage_has_explicit_owner_dependencies() -> None:
    """The stage must not inherit runner globals through linked context."""
    source = _source("stage.py")
    assert "install_context" not in source
    assert "ruff: noqa: F821" not in source
    assert "from worldsim.phase_2.phase_2c import artifacts as _phase_2c_artifacts" in source
    assert "from worldsim.phase_2.phase_2c.runner import verify_feasibility" in source
    assert "from worldsim.phase_2.phase_2c.types import FeasibilityReport" in source


def test_phase_2c_stage_and_runner_import_in_either_order() -> None:
    """Import order must not decide whether stage dependencies exist."""
    _import_smoke(
        "from worldsim.phase_2.phase_2c import stage, runner; "
        "assert stage._run_feasibility_stage; assert runner.verify_feasibility"
    )
    _import_smoke(
        "from worldsim.phase_2.phase_2c import runner, stage; "
        "assert stage._run_feasibility_stage; assert runner.verify_feasibility"
    )
