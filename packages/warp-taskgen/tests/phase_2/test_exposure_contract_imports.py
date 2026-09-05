"""Each exposure_contract sibling must import alone in a fresh interpreter.

The ``_impl.py`` parity module used to load every sibling at once, which hid
import cycles between them. Importing each module by itself proves the
behavior-owned split has no cycle and no dependency on package-level side
effects.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).parents[2]
EXPOSURE_CONTRACT_ROOT = PACKAGE_ROOT / "warp_taskgen" / "phase_2" / "exposure_contract"

SIBLING_MODULES = sorted(
    path.stem for path in EXPOSURE_CONTRACT_ROOT.glob("*.py") if path.stem != "__init__"
)


def test_parity_module_is_gone() -> None:
    assert not (EXPOSURE_CONTRACT_ROOT / "_impl.py").exists()
    assert "_impl" not in SIBLING_MODULES


@pytest.mark.parametrize("module", SIBLING_MODULES)
def test_exposure_contract_sibling_imports_alone(module: str) -> None:
    subprocess.run(
        [sys.executable, "-c", f"import warp_taskgen.phase_2.exposure_contract.{module}"],
        check=True,
        cwd=PACKAGE_ROOT,
        env={"PYTHONPATH": str(PACKAGE_ROOT)},
    )
