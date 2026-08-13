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
RESUME_MODULE = "resume"
VARIANT_EVAL_MODULE = "variant_eval"
PLACEMENT_LOOP_MODULE = "placement_loop"
EVAL_AWARENESS_ITERATOR_MODULE = "eval_awareness_iterator"
STRATEGY_VARIATION_MODULE = "strategy_variation"
POSTPROCESS_MODULE = "postprocess"


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
    assert "link_modules" not in source
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


def test_resume_owns_explicit_dependencies_and_runner_does_not_link_it() -> None:
    source = _source(RESUME_MODULE)
    runner = _source("runner")
    assert "install_context" not in source
    assert "ruff: noqa: F821" not in source
    assert "from worldsim.phase_4._context" not in source
    assert "from worldsim.phase_4 import resume as _resume" not in runner
    assert "        _resume,\n" not in runner
    for constant in (
        "_CHECKPOINT_FINGERPRINT_KEY",
        "_PHASE_4_RESUME_VERSION",
        "_VARIANT_RESULT_METADATA",
        "PLACEMENT_FIX_MAX_ITERATIONS",
    ):
        assert source.count(constant) >= 1


def test_resume_consumers_import_resume_owner_directly() -> None:
    expectations = {
        "runner": ("_phase_4_state_metadata", "_phase_4_result_fingerprint"),
        "preflight": ("_fingerprint_payload", "_seed_target_benchmark"),
        "admission": ("_task_reachable_sites",),
        "execution": ("_seed_has_actions", "_seed_requires_reset"),
        "postprocess": ("_phase_4_postprocess_fingerprint", "_write_json_atomic"),
        "placement_loop": (
            "_load_saved_placement_iteration_result",
            "_write_placement_fix_checkpoint",
        ),
        "strategy_variation": ("_strategy_variation_checkpoint_path", "_VARIANT_ROUNDS_KEY"),
        "eval_awareness_iterator": ("_phase_4_postprocess_fingerprint", "_variant_changes_seed"),
        "variant_eval": ("_load_saved_variant_result", "_phase_4_variant_fingerprint"),
    }
    for module, names in expectations.items():
        source = _source(module)
        assert "from worldsim.phase_4.resume import" in source
        for name in names:
            assert name in source


def test_resume_tests_do_not_patch_runner_resume_exports() -> None:
    test_root = PACKAGE_ROOT / "tests"
    resume_names = (
        "_phase_4_result_fingerprint",
        "_phase_4_postprocess_fingerprint",
        "_phase_4_variant_fingerprint",
        "_CHECKPOINT_FINGERPRINT_KEY",
        "_VARIANT_RESULT_METADATA",
        "_seed_target_benchmark",
        "_normalize_saved_adversarial_result",
        "_sweep_orphan_inflight_sentinels",
    )
    for path in (*test_root.glob("phase_4/test_*.py"), test_root / "crash_resume_scenarios.py"):
        if path.name == "test_leaf_context_boundary.py":
            continue
        source = path.read_text()
        for name in resume_names:
            assert f"phase_4_adversarial.{name}" not in source


def test_variant_eval_owns_explicit_dependencies_and_runner_does_not_link_it() -> None:
    source = _source(VARIANT_EVAL_MODULE)
    runner = _source("runner")
    assert "install_context" not in source
    assert "ruff: noqa: F821" not in source
    assert "from worldsim.phase_4._context" not in source
    assert "from worldsim.phase_4 import variant_eval as _variant_eval" not in runner
    assert "        _variant_eval,\n" not in runner
    for module in (
        "execution_helpers",
        "payload_text",
        "resume",
        "seeding",
        "task_paths",
        "config",
    ):
        assert f"worldsim.{module}" in source or f"worldsim.phase_4.{module}" in source
    tree = ast.parse(source)
    assert not any(
        isinstance(node, ast.ImportFrom)
        and node.module == "worldsim.phase_4"
        and any(alias.name == "execution" for alias in node.names)
        for node in tree.body
    )
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "worldsim.phase_4"
        and any(alias.name == "execution" for alias in node.names)
        for node in ast.walk(tree)
    )
    assert "from worldsim.phase_4.execution import run_adversarial_task" not in source


def test_variant_consumers_import_variant_owner_directly() -> None:
    expectations = {
        "placement_loop": ("_merge_variant_task", "_rerun_adversarial_task"),
        "strategy_variation": ("_evaluate_variant",),
        "eval_awareness_iterator": ("_evaluate_variant", "_merge_variant_task"),
        "admission": ("_rebase_adversarial_task",),
    }
    for module, names in expectations.items():
        source = _source(module)
        for name in names:
            assert name in source
            assert f"phase_4_adversarial.{name}" not in source


def test_variant_eval_imports_in_either_order() -> None:
    package_root = str(PACKAGE_ROOT)
    for statement in (
        "from worldsim.phase_4 import variant_eval, runner; "
        "assert variant_eval._evaluate_variant; assert runner.run",
        "from worldsim.phase_4 import runner, variant_eval; "
        "assert variant_eval._evaluate_variant; assert runner.run",
    ):
        subprocess.run(
            [sys.executable, "-c", statement],
            check=True,
            cwd=PACKAGE_ROOT,
            env={"PYTHONPATH": package_root},
        )


def test_eval_awareness_iterator_owns_explicit_dependencies_and_runner_does_not_link_it() -> None:
    source = _source(EVAL_AWARENESS_ITERATOR_MODULE)
    runner = _source("runner")
    assert "install_context" not in source
    assert "ruff: noqa: F821" not in source
    assert "from worldsim.phase_4._context" not in source
    assert (
        "from worldsim.phase_4 import eval_awareness_iterator as _eval_awareness_iterator"
        not in runner
    )
    assert "        _eval_awareness_iterator,\n" not in runner
    for dependency in (
        "import asyncio",
        "import json",
        "import logging",
        "from collections.abc import Callable, Mapping",
        "from pathlib import Path",
        "from typing import Any",
        "from worldsim.config import BenchmarkInstance",
        "from worldsim.agent_runtime import AgentRunner",
        "from worldsim.phase_4 import result_summary as phase4_result_summary",
        "from worldsim.phase_4.options import",
        "from worldsim.task_paths import safe_task_path_component",
    ):
        assert dependency in source
    assert "logger = logging.getLogger(__name__)" in source


def test_eval_awareness_iterator_imports_in_either_order() -> None:
    package_root = str(PACKAGE_ROOT)
    for statement in (
        "from worldsim.phase_4 import eval_awareness_iterator, runner; "
        "assert eval_awareness_iterator.run_eval_awareness_iterator; assert runner.run",
        "from worldsim.phase_4 import runner, eval_awareness_iterator; "
        "assert eval_awareness_iterator.run_eval_awareness_iterator; assert runner.run",
    ):
        subprocess.run(
            [sys.executable, "-c", statement],
            check=True,
            cwd=PACKAGE_ROOT,
            env={"PYTHONPATH": package_root},
        )


def test_eval_awareness_iterator_tests_do_not_patch_runner_iterator_exports() -> None:
    test_root = PACKAGE_ROOT / "tests"
    iterator_names = ("_prior_iteration_feedback",)
    for path in (
        *test_root.glob("test_*.py"),
        *test_root.glob("phase_4/test_*.py"),
        test_root / "crash_resume_scenarios.py",
    ):
        if path.name == "test_leaf_context_boundary.py":
            continue
        source = path.read_text()
        for name in iterator_names:
            assert f"phase_4_adversarial.{name}" not in source
            assert f'setattr(phase_4_adversarial, "{name}"' not in source


def test_placement_loop_owns_explicit_dependencies_and_runner_does_not_link_it() -> None:
    source = _source(PLACEMENT_LOOP_MODULE)
    runner = _source("runner")
    assert "install_context" not in source
    assert "ruff: noqa: F821" not in source
    assert "from worldsim.phase_4._context" not in source
    assert "from worldsim.phase_4 import placement_loop as _placement_loop" not in runner
    assert "        _placement_loop,\n" not in runner
    for module in (
        "aer_trajectory_extract",
        "encounter_detection",
        "needham_trace",
        "placement_api",
        "transcript_purpose_api",
        "verbalized_eval_awareness_api",
        "agent_runtime",
        "config",
        "site_lock",
        "task_paths",
    ):
        assert f"worldsim.{module}" in source or f"worldsim.phase_4.{module}" in source
    assert "from worldsim.phase_4.resume import (" in source
    assert "from worldsim.phase_4.variant_eval import" in source


def test_placement_consumers_import_placement_owner_directly() -> None:
    execution = _source("execution")
    postprocess = _source("postprocess")
    assert "from worldsim.phase_4.placement_loop import _run_pvpo_gate" in execution
    assert "from worldsim.phase_4.placement_loop import _run_placement_fix_loop" in postprocess
    for source in (execution, postprocess):
        assert "phase_4_adversarial._run_pvpo_gate" not in source
        assert "phase_4_adversarial._run_placement_fix_loop" not in source


def test_placement_loop_imports_in_either_order() -> None:
    package_root = str(PACKAGE_ROOT)
    for statement in (
        "from worldsim.phase_4 import placement_loop, runner; "
        "assert placement_loop._run_pvpo_gate; assert runner.run",
        "from worldsim.phase_4 import runner, placement_loop; "
        "assert placement_loop._run_pvpo_gate; assert runner.run",
        "from worldsim.phase_4 import placement_loop, execution; "
        "assert placement_loop._run_pvpo_gate; assert execution.run_adversarial_task",
        "from worldsim.phase_4 import execution, placement_loop; "
        "assert placement_loop._run_pvpo_gate; assert execution.run_adversarial_task",
    ):
        subprocess.run(
            [sys.executable, "-c", statement],
            check=True,
            cwd=PACKAGE_ROOT,
            env={"PYTHONPATH": package_root},
        )


def test_placement_tests_do_not_patch_runner_exports() -> None:
    test_root = PACKAGE_ROOT / "tests"
    placement_names = (
        "_run_pvpo_gate",
        "_run_placement_fix_loop",
        "determine_encounter",
        "run_transcript_purpose_api",
        "run_verbalized_eval_awareness_api",
        "run_placement_api",
    )
    for path in (
        *test_root.glob("test_*.py"),
        *test_root.glob("phase_4/test_*.py"),
        test_root / "crash_resume_scenarios.py",
    ):
        if path.name == "test_leaf_context_boundary.py":
            continue
        source = path.read_text()
        for name in placement_names:
            assert f"phase_4_adversarial.{name}" not in source
            assert f'setattr(phase_4_adversarial, "{name}"' not in source


def test_resume_imports_in_either_order() -> None:
    package_root = str(PACKAGE_ROOT)
    for statement in (
        "from worldsim.phase_4 import resume, runner; "
        "assert resume._phase_4_result_fingerprint; assert runner.run",
        "from worldsim.phase_4 import runner, resume; "
        "assert resume._phase_4_result_fingerprint; assert runner.run",
    ):
        subprocess.run(
            [sys.executable, "-c", statement],
            check=True,
            cwd=PACKAGE_ROOT,
            env={"PYTHONPATH": package_root},
        )


def test_strategy_variation_owns_explicit_dependencies_and_runner_does_not_link_it() -> None:
    source = _source(STRATEGY_VARIATION_MODULE)
    runner = _source("runner")
    postprocess = _source("postprocess")
    assert "install_context" not in source
    assert "ruff: noqa: F821" not in source
    assert "from worldsim.phase_4._context" not in source
    assert "from worldsim.phase_4 import strategy_variation as _strategy_variation" not in runner
    assert "        _strategy_variation,\n" not in runner
    for dependency in (
        "import asyncio",
        "import logging",
        "from collections.abc import Callable, Mapping",
        "from pathlib import Path",
        "from typing import Any",
        "from worldsim.agent_runtime import AgentRunner",
        "from worldsim.config import BenchmarkInstance",
        "from worldsim.failpoints import crash_if_enabled",
        "from worldsim.resume_metadata import instances_identity",
        "from worldsim.task_reset_cache import callable_accepts_keyword",
        "from worldsim.phase_4.options import",
        "from worldsim.phase_4.strategy_catalog import ALLOWED_STRATEGIES as _ALLOWED_STRATEGIES",
    ):
        assert dependency in source
    assert "logger = logging.getLogger(__name__)" in source
    assert source.count("async def run_judge(") == 1
    assert "from worldsim.phase_4.judge_api import run_judge_api" in source
    assert "async def run_judge(" not in postprocess
    assert "from worldsim.phase_4.strategy_variation import" in postprocess


def test_strategy_variation_imports_in_either_order() -> None:
    package_root = str(PACKAGE_ROOT)
    for statement in (
        "from worldsim.phase_4 import strategy_variation, runner; "
        "assert strategy_variation.run_strategy_variation; assert runner.run",
        "from worldsim.phase_4 import runner, strategy_variation; "
        "assert strategy_variation.run_strategy_variation; assert runner.run",
        "from worldsim.phase_4 import strategy_variation, postprocess; "
        "assert strategy_variation.run_judge is postprocess.run_judge",
        "from worldsim.phase_4 import postprocess, strategy_variation; "
        "assert strategy_variation.run_judge is postprocess.run_judge",
    ):
        subprocess.run(
            [sys.executable, "-c", statement],
            check=True,
            cwd=PACKAGE_ROOT,
            env={"PYTHONPATH": package_root},
        )


def test_strategy_tests_do_not_patch_runner_strategy_exports() -> None:
    test_root = PACKAGE_ROOT / "tests"
    strategy_names = ("run_judge", "generate_variant", "_evaluate_variant")
    for path in (
        *test_root.glob("test_*.py"),
        *test_root.glob("phase_4/test_*.py"),
        test_root / "crash_resume_scenarios.py",
    ):
        if path.name == "test_leaf_context_boundary.py":
            continue
        source = path.read_text()
        for name in strategy_names:
            assert f"phase_4_adversarial.{name}" not in source
            assert f'setattr(phase_4_adversarial, "{name}"' not in source


def test_postprocess_owns_explicit_dependencies_and_runner_keeps_only_named_seam() -> None:
    source = _source(POSTPROCESS_MODULE)
    runner = _source("runner")
    assert "install_context" not in source
    assert "ruff: noqa: F821" not in source
    assert "from worldsim.phase_4._context" not in source
    for dependency in (
        "import asyncio",
        "import json",
        "import logging",
        "from collections.abc import Callable",
        "from pathlib import Path",
        "from typing import Any",
        "from worldsim.agent_runtime import AgentRunner",
        "from worldsim.config import BenchmarkConfig, BenchmarkInstance",
        "from worldsim.phase_4.options import",
        "from worldsim.task_paths import safe_task_path_component",
        "from worldsim.agent_config import instances_for_site",
    ):
        assert dependency in source
    assert "logger = logging.getLogger(__name__)" in source
    assert "from worldsim.phase_4.strategy_variation import" in source
    assert "run_judge = _strategy_run_judge" in source
    assert "from worldsim.phase_4 import postprocess as _postprocess" in runner
    assert "_postprocess._postprocess_one_task(" in runner
    assert "link_modules" not in runner
    assert "import sys as _sys" not in runner


def test_postprocess_imports_in_either_order() -> None:
    package_root = str(PACKAGE_ROOT)
    for statement in (
        "from worldsim.phase_4 import postprocess, runner; "
        "assert postprocess._postprocess_one_task; assert runner.run",
        "from worldsim.phase_4 import runner, postprocess; "
        "assert postprocess._postprocess_one_task; assert runner.run",
        "from worldsim.phase_4 import postprocess, strategy_variation; "
        "assert strategy_variation.run_judge is postprocess.run_judge",
        "from worldsim.phase_4 import strategy_variation, postprocess; "
        "assert strategy_variation.run_judge is postprocess.run_judge",
    ):
        subprocess.run(
            [sys.executable, "-c", statement],
            check=True,
            cwd=PACKAGE_ROOT,
            env={"PYTHONPATH": package_root},
        )


def test_postprocess_owner_tests_do_not_call_runner_owner_exports() -> None:
    expected_absences = {
        "test_postprocess_1.py": ("_process_adversarial_result",),
        "test_eval_awareness_iterator.py": ("_process_adversarial_result",),
        "test_resume_1.py": ("_postprocess_one_task",),
    }
    for filename, names in expected_absences.items():
        source = (PACKAGE_ROOT / "tests" / "phase_4" / filename).read_text()
        for name in names:
            assert f"phase_4_adversarial.{name}" not in source


def test_postprocess_runner_seam_is_module_qualified_repo_wide() -> None:
    roots = (PACKAGE_ROOT / "worldsim", PACKAGE_ROOT / "tests")
    forbidden = (
        "from worldsim.phase_4.postprocess import _postprocess_one_task",
        "phase_4_adversarial._postprocess_one_task",
        'setattr(phase_4_adversarial, "_postprocess_one_task"',
    )
    for root in roots:
        for path in root.rglob("*.py"):
            if path == Path(__file__):
                continue
            source = path.read_text()
            for text in forbidden:
                assert text not in source, f"stale runner postprocess seam in {path}: {text}"
    runner = _source("runner")
    assert "from worldsim.phase_4 import postprocess as _postprocess" in runner
    assert "_postprocess._postprocess_one_task(" in runner
