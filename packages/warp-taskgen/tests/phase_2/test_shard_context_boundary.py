"""Structural guards for the Phase 2 shard ownership boundary."""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parents[2]
PHASE_2_ROOT = PACKAGE_ROOT / "worldsim" / "phase_2"


def _source(name: str) -> str:
    return (PHASE_2_ROOT / name).read_text()


def test_site_injection_result_has_one_feature_local_owner() -> None:
    """The shared shard result type is imported, not inherited from runner globals."""
    from worldsim.phase_2 import generation, shards
    from worldsim.phase_2.planning_types import SiteInjectionResult

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
    assert "link_modules" not in runner_source


def test_phase_2_linked_context_is_deleted() -> None:
    """The Phase 2 package must not retain the linked-context aggregator."""
    assert not (PHASE_2_ROOT / "_context.py").exists()
    runner_source = _source("runner.py")
    assert "install_context" not in runner_source
    assert "link_modules" not in runner_source
    assert "sys.modules" not in runner_source
    assert "ruff: noqa: F821" not in runner_source
    assert "ruff: noqa: E402" not in runner_source


def test_runner_and_feature_import_in_either_order() -> None:
    """Importing runner first must not manufacture feature globals."""
    import subprocess
    import sys

    package_root = str(PACKAGE_ROOT)
    for statement in (
        "from worldsim.phase_2 import runner, generation; assert runner.run; assert generation._generate_injections_for_site",
        "from worldsim.phase_2 import generation, runner; assert runner.run; assert generation._generate_injections_for_site",
    ):
        subprocess.run(
            [sys.executable, "-c", statement],
            check=True,
            cwd=PACKAGE_ROOT,
            env={"PYTHONPATH": package_root},
        )
