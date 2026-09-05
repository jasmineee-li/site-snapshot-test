"""Import-isolation guards for the Phase 1 novel-task validation siblings."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).parents[2]
PACKAGE = "warp_taskgen.phase_1.novel_task_validation"
SIBLINGS = (
    "answer_stability",
    "batch",
    "errors",
    "ordering",
    "placement",
    "rewards",
    "route_alignment",
    "single_task",
    "task_card_generation",
    "task_cards",
)


@pytest.mark.parametrize("module", SIBLINGS)
def test_sibling_imports_in_a_fresh_interpreter(module: str) -> None:
    """Each sibling must import first, without the package root or another sibling.

    The drained modules form a DAG (errors <- answer_stability <- route_alignment,
    placement, task_cards <- rewards <- single_task <- batch). A cycle would surface
    here as a partially-initialized-module ImportError.
    """
    subprocess.run(
        [sys.executable, "-c", f"import {PACKAGE}.{module}"],
        check=True,
        cwd=PACKAGE_ROOT,
        env={"PYTHONPATH": str(PACKAGE_ROOT)},
    )


def test_parity_module_is_gone() -> None:
    assert not (
        PACKAGE_ROOT / "warp_taskgen" / "phase_1" / "novel_task_validation" / "_impl.py"
    ).exists()
    with pytest.raises(ModuleNotFoundError):
        __import__(f"{PACKAGE}._impl")
