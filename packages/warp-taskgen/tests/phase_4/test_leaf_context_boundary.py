"""Structural guards for the Phase 4 leaf-module context cutover."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parents[2]
PHASE_4_ROOT = PACKAGE_ROOT / "worldsim" / "phase_4"
LEAF_MODULES = ("payload_text", "metrics", "admission", "preflight", "execution_helpers")
EXECUTION_MODULE = "execution"


def _source(name: str) -> str:
    return (PHASE_4_ROOT / f"{name}.py").read_text()


def test_phase4_leaf_modules_have_explicit_imports_and_local_constants() -> None:
    expected_local_constants = {
        "payload_text": ("_PAYLOAD_BODY_FIELD_HINTS", "_MARKDOWN_SYSTEM_BLOCK_RE"),
        "metrics": ("_ACTION_REWARD_SIGNALS", "_PVPO_METRIC_KEYS", "LAYOUT_SCROLL_BUCKETS"),
        "admission": ("STRICT_FEASIBILITY_ADMISSION",),
        "execution_helpers": ("_RESET_TIMEOUT", "_RESET_MAX_RETRIES", "_RESET_RETRY_DELAY"),
        "preflight": (),
    }
    for name in LEAF_MODULES:
        source = _source(name)
        assert "install_context" not in source
        assert "ruff: noqa: F821" not in source
        tree = ast.parse(source)
        assigned: set[str] = set()
        for node in tree.body:
            targets = (
                node.targets
                if isinstance(node, ast.Assign)
                else ([node.target] if isinstance(node, ast.AnnAssign) else [])
            )
            assigned.update(target.id for target in targets if isinstance(target, ast.Name))
        for constant in expected_local_constants[name]:
            assert constant in assigned, f"{name} must own {constant}"


def test_runner_does_not_link_explicit_leaf_modules() -> None:
    source = _source("runner")
    assert "link_modules" in source
    for name in LEAF_MODULES:
        assert f"phase_4 import {name} as _{name}" not in source


def test_results_module_has_explicit_dependencies() -> None:
    source = _source("results")
    assert "install_context" not in source
    assert "ruff: noqa: F821" not in source
    assert "from worldsim.phase_4._context" not in source


def test_runner_calls_results_owner_directly() -> None:
    source = _source("runner")
    assert "from worldsim.phase_4.results import _write_phase_4_results" in source
    assert "from worldsim.phase_4 import results as _results" not in source
    assert "        _results,\n" not in source


def test_execution_owns_only_explicit_execution_dependencies() -> None:
    source = _source(EXECUTION_MODULE)
    assert "install_context" not in source
    assert "ruff: noqa: F821" not in source
    assert "from worldsim.phase_4._context" not in source
    assert "from worldsim.phase_4.placement_loop import _run_pvpo_gate" in source
    assert "from worldsim.phase_4.resume import (" in source
    for helper in ("_seed_has_actions", "_seed_requires_reset", "_seed_target_benchmark"):
        assert helper in source


def test_runner_and_variant_evaluation_call_execution_owner_directly() -> None:
    runner = _source("runner")
    variant_eval = _source("variant_eval")
    assert "from worldsim.phase_4 import execution as _execution" in runner
    assert "_execution.run_adversarial_task(" in runner
    assert "_execution.run_adversarial_task(" in variant_eval
    assert "        _execution," not in runner
    assert "from worldsim.phase_4.execution import run_adversarial_task" not in variant_eval


def test_execution_tests_do_not_patch_runner_execution_exports() -> None:
    test_root = PACKAGE_ROOT / "tests"
    for path in (*test_root.glob("phase_4/test_*.py"), test_root / "crash_resume_scenarios.py"):
        if path.name == "test_leaf_context_boundary.py":
            continue
        source = path.read_text()
        assert "phase_4_adversarial.run_adversarial_task" not in source
        assert 'setattr(phase_4_adversarial, "run_adversarial_task"' not in source


def test_leaf_modules_import_in_either_order() -> None:
    package_root = str(PACKAGE_ROOT)
    for name in LEAF_MODULES:
        for statement in (
            f"from worldsim.phase_4 import {name}, runner; "
            f"assert {name}.__name__ == 'worldsim.phase_4.{name}'; assert runner.run",
            f"from worldsim.phase_4 import runner, {name}; "
            f"assert {name}.__name__ == 'worldsim.phase_4.{name}'; assert runner.run",
        ):
            subprocess.run(
                [sys.executable, "-c", statement],
                check=True,
                cwd=PACKAGE_ROOT,
                env={"PYTHONPATH": package_root},
            )


def test_results_module_imports_in_either_order() -> None:
    package_root = str(PACKAGE_ROOT)
    for statement in (
        "from worldsim.phase_4 import results, runner; "
        "assert results._write_phase_4_results; assert runner.run",
        "from worldsim.phase_4 import runner, results; "
        "assert results._write_phase_4_results; assert runner.run",
    ):
        subprocess.run(
            [sys.executable, "-c", statement],
            check=True,
            cwd=PACKAGE_ROOT,
            env={"PYTHONPATH": package_root},
        )


def test_execution_imports_in_either_order() -> None:
    package_root = str(PACKAGE_ROOT)
    for statement in (
        "from worldsim.phase_4 import execution, runner; "
        "assert execution.run_adversarial_task; assert runner.run",
        "from worldsim.phase_4 import runner, execution; "
        "assert execution.run_adversarial_task; assert runner.run",
    ):
        subprocess.run(
            [sys.executable, "-c", statement],
            check=True,
            cwd=PACKAGE_ROOT,
            env={"PYTHONPATH": package_root},
        )
