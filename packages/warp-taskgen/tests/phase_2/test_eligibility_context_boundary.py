"""Structural guards for the Phase 2 eligibility ownership boundary."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parents[2]
PHASE_2_ROOT = PACKAGE_ROOT / "worldsim" / "phase_2"


def _source(name: str) -> str:
    return (PHASE_2_ROOT / name).read_text()


def _linked_module_names() -> set[str]:
    tree = ast.parse(_source("runner.py"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "_link_modules" or not node.args:
            continue
        modules = node.args[0]
        if not isinstance(modules, ast.List):
            continue
        names.update(item.id for item in modules.elts if isinstance(item, ast.Name))
    return names


def _import_smoke(statement: str) -> None:
    package_root = str(PACKAGE_ROOT)
    subprocess.run(
        [sys.executable, "-c", statement],
        check=True,
        cwd=package_root,
        env={"PYTHONPATH": package_root},
    )


def test_eligibility_has_explicit_owner_dependencies() -> None:
    """Eligibility must not inherit runner globals through linked context."""
    source = _source("eligibility.py")
    assert "install_context" not in source
    assert "ruff: noqa: F821" not in source

    assert "_eligibility" not in _linked_module_names()


def test_generation_calls_the_eligibility_owner_explicitly() -> None:
    """Generation should name the owner for every eligibility helper call."""
    generation = _source("generation.py")
    assert "from worldsim.phase_2 import eligibility as _eligibility" in generation
    for helper in (
        "_build_cell_targets",
        "_build_exposure_contracts_for_shard",
        "_persist_exposure_contracts",
        "_phase_2a_eligible_tasks_for_benchmark",
        "_select_balanced_subset",
        "_seed_delivery_mechanism",
        "_surface_visibility_by_id",
        "_write_eligibility_drops",
    ):
        assert f"_eligibility.{helper}" in generation


def test_eligibility_and_runner_import_in_either_order() -> None:
    """Import order must not decide whether eligibility helpers exist."""
    _import_smoke(
        "from worldsim.phase_2 import eligibility, runner; "
        "assert eligibility._phase_2a_eligible_tasks; "
        "assert runner.run"
    )
    _import_smoke(
        "from worldsim.phase_2 import runner, eligibility; "
        "assert eligibility._phase_2a_eligible_tasks; "
        "assert runner.run"
    )
