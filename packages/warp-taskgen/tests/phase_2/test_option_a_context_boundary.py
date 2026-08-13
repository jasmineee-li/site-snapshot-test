"""Structural guards for the Phase 2 Option A ownership boundary."""

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
        for item in modules.elts:
            if isinstance(item, ast.Name):
                names.add(item.id)
    return names


def _import_smoke(statement: str) -> None:
    package_root = str(PACKAGE_ROOT)
    subprocess.run(
        [sys.executable, "-c", statement],
        check=True,
        cwd=package_root,
        env={"PYTHONPATH": package_root},
    )


def test_option_a_has_explicit_owner_dependencies() -> None:
    """Option A must not inherit runner globals through linked context."""
    source = _source("option_a.py")
    assert "install_context" not in source
    assert "ruff: noqa: F821" not in source

    linked = _linked_module_names()
    assert "_option_a" not in linked


def test_option_a_callers_use_the_feature_owner() -> None:
    """Generation and validation should name the Option A owner explicitly."""
    generation = _source("generation.py")
    plan_validation = _source("plan_validation.py")
    assert "from worldsim.phase_2 import option_a as _option_a" in generation
    assert "_option_a._benchmark_for_option_a_plan" in generation
    assert "from worldsim.phase_2 import option_a as _option_a" in plan_validation
    assert "_option_a._is_option_a_site" in plan_validation
    assert "_option_a._validate_option_a_placement" in plan_validation


def test_option_a_and_runner_import_in_either_order() -> None:
    """Import order must not decide whether Option A helpers exist."""
    _import_smoke(
        "from worldsim.phase_2 import option_a, runner; "
        "assert option_a._is_option_a_site({'site': 'gitlab'}); "
        "assert runner.run"
    )
    _import_smoke(
        "from worldsim.phase_2 import runner, option_a; "
        "assert option_a._is_option_a_site({'site': 'reddit'}); "
        "assert runner.run"
    )
