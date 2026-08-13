"""Structural guards for the Phase 2 generation ownership boundary."""

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
        if not isinstance(node, ast.Call) or not node.args:
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "_link_modules":
            continue
        modules = node.args[0]
        if isinstance(modules, ast.List):
            names.update(item.id for item in modules.elts if isinstance(item, ast.Name))
    return names


def _import_smoke(statement: str) -> None:
    subprocess.run(
        [sys.executable, "-c", statement],
        check=True,
        cwd=PACKAGE_ROOT,
        env={"PYTHONPATH": str(PACKAGE_ROOT)},
    )


def test_generation_has_explicit_owner_dependencies() -> None:
    source = _source("generation.py")
    assert "install_context" not in source
    assert "ruff: noqa: F821" not in source
    assert "sys.modules" not in source
    assert "from worldsim.phase_2 import runner_api as _runner_api" in source
    assert "from worldsim.phase_2.output import" in source
    assert "from worldsim.phase_2.text_fill.voice import" in source
    assert "from worldsim.seed_contracts.validation import" in source
    assert "from worldsim.profile_validation import" in source
    assert "from worldsim.prompt_loading import" in source
    assert "from worldsim.state import" in source


def test_runner_qualifies_generation_and_does_not_link_it() -> None:
    runner_source = _source("runner.py")
    assert "_generation._collect_site_profiles" in runner_source
    assert "_generation" not in _linked_module_names()


def test_generation_and_runner_import_in_either_order() -> None:
    _import_smoke(
        "from worldsim.phase_2 import generation, runner; "
        "assert generation._generate_injections_for_site; assert runner.run"
    )
    _import_smoke(
        "from worldsim.phase_2 import runner, generation; "
        "assert generation._generate_injections_for_site; assert runner.run"
    )
