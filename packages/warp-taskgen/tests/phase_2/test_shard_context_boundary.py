"""Structural guards for the Phase 2 shard ownership boundary."""

from __future__ import annotations

import ast
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


def test_site_injection_result_has_one_feature_local_owner() -> None:
    """The shared shard result type is imported, not inherited from runner globals."""
    from worldsim.phase_2 import generation, runner, shards
    from worldsim.phase_2.planning_types import SiteInjectionResult

    assert runner.SiteInjectionResult is SiteInjectionResult
    assert generation.SiteInjectionResult is SiteInjectionResult
    assert shards.SiteInjectionResult is SiteInjectionResult

    for name in ("generation.py", "shards.py"):
        tree = ast.parse(_source(name))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        assert "SiteInjectionResult" in imported


def test_shards_are_explicitly_imported_and_not_linked() -> None:
    """Shard behavior must run with its own imports while other links remain."""
    shards_source = _source("shards.py")
    runner_source = _source("runner.py")

    assert "install_context" not in shards_source
    assert "ruff: noqa: F821" not in shards_source
    assert "from worldsim.phase_2 import shards as _shards" in runner_source
    assert "_shards._" in runner_source
    assert "_shards" not in _linked_module_names()
