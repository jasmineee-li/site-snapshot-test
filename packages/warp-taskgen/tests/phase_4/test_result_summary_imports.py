"""Import-isolation guards for the Phase 4 result-summary package split."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).parents[2]
RESULT_SUMMARY_ROOT = PACKAGE_ROOT / "warp_taskgen" / "phase_4" / "result_summary"
SIBLINGS = (
    "task_metadata",
    "final_metrics",
    "action_metrics",
    "audit",
    "inspection",
    "summarize",
)


def test_result_summary_siblings_are_the_whole_package() -> None:
    modules = sorted(path.stem for path in RESULT_SUMMARY_ROOT.glob("*.py"))
    assert modules == sorted(("__init__", *SIBLINGS))
    assert not (PACKAGE_ROOT / "warp_taskgen" / "phase_4" / "result_summary.py").exists()


@pytest.mark.parametrize("module", SIBLINGS)
def test_result_summary_sibling_imports_alone(module: str) -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            f"import warp_taskgen.phase_4.result_summary.{module} as m; assert m.__all__",
        ],
        check=True,
        cwd=PACKAGE_ROOT,
        env={"PYTHONPATH": str(PACKAGE_ROOT)},
    )


def test_result_summary_package_exports_final_status_sets() -> None:
    from warp_taskgen.phase_4 import result_summary
    from warp_taskgen.phase_4.result_summary import final_metrics

    assert result_summary.COMPLIED_FINAL_STATUSES is final_metrics.COMPLIED_FINAL_STATUSES
    assert result_summary.NON_SCORABLE_FINAL_STATUSES is final_metrics.NON_SCORABLE_FINAL_STATUSES
