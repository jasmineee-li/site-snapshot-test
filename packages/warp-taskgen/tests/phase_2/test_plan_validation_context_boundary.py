"""Structural guards for the Phase 2 plan-validation ownership boundary."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parents[2]
PHASE_2_ROOT = PACKAGE_ROOT / "warp_taskgen" / "phase_2"


def _source(name: str) -> str:
    return (PHASE_2_ROOT / name).read_text()


def _import_smoke(statement: str) -> None:
    subprocess.run(
        [sys.executable, "-c", statement],
        check=True,
        cwd=PACKAGE_ROOT,
        env={"PYTHONPATH": str(PACKAGE_ROOT)},
    )


def test_plan_validation_has_explicit_owner_dependencies() -> None:
    source = _source("plan_validation.py")
    assert "install_context" not in source
    assert "ruff: noqa: F821" not in source
    assert "_REQUIRED_PLAN_FIELDS" in source
    assert "_FINAL_STAGE_ONLY_FIELDS" in source
    assert "from warp_taskgen.seed_contracts.delivery import" in source
    assert "from warp_taskgen.seed_contracts.surface import" in source
    assert "from warp_taskgen.seed_contracts.validation import" in source


def test_generation_calls_plan_validation_owner_explicitly() -> None:
    source = _source("generation.py")
    assert "from warp_taskgen.phase_2 import plan_validation as _plan_validation" in source
    assert "_plan_validation._validate_generated_adversarial_tasks" in source


def test_plan_validation_and_runner_import_in_either_order() -> None:
    _import_smoke(
        "from warp_taskgen.phase_2 import plan_validation, runner; "
        "assert plan_validation._validate_generated_adversarial_task; "
        "assert runner.run"
    )
    _import_smoke(
        "from warp_taskgen.phase_2 import runner, plan_validation; "
        "assert plan_validation._validate_generated_adversarial_task; "
        "assert runner.run"
    )
