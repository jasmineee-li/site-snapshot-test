"""Import-isolation guards for the ``warp_taskgen.outcome_taxonomy`` siblings."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).parents[1]
SIBLINGS = (
    "trajectory_io",
    "read_surface",
    "engagement",
    "signals",
    "classification",
    "summary",
    "io",
    "serialization",
)


@pytest.mark.parametrize("module", SIBLINGS)
def test_outcome_taxonomy_sibling_imports_in_fresh_interpreter(module: str) -> None:
    subprocess.run(
        [sys.executable, "-c", f"import warp_taskgen.outcome_taxonomy.{module}"],
        check=True,
        cwd=PACKAGE_ROOT,
        env={"PYTHONPATH": str(PACKAGE_ROOT)},
    )


def test_outcome_taxonomy_legacy_impl_module_is_retired() -> None:
    assert importlib.util.find_spec("warp_taskgen.outcome_taxonomy._impl") is None
