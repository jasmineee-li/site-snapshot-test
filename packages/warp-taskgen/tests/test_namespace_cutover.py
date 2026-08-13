"""Structural and compatibility contracts for the WARP Taskgen namespace cutover."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parents[1]


def test_canonical_package_owns_the_implementation_tree() -> None:
    canonical = PACKAGE_ROOT / "warp_taskgen"
    legacy = PACKAGE_ROOT / "worldsim"

    assert canonical.is_dir()
    assert (canonical / "main.py").is_file()
    assert legacy.is_dir()
    implementation_files = [path for path in legacy.rglob("*.py") if path.name != "__init__.py"]
    assert implementation_files == []


def test_legacy_root_is_canonical_and_does_not_eagerly_load_children() -> None:
    env = {**os.environ, "PYTHONPATH": str(PACKAGE_ROOT)}
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, warp_taskgen, worldsim; "
            "assert worldsim is warp_taskgen; "
            "assert not any(name.startswith('warp_taskgen.') for name in sys.modules if name != 'warp_taskgen')",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_legacy_and_canonical_submodules_are_the_same_module() -> None:
    canonical = importlib.import_module("warp_taskgen.phase_4.pvpo_cdp")
    legacy = importlib.import_module("worldsim.phase_4.pvpo_cdp")

    assert legacy is canonical
    assert sys.modules["worldsim.phase_4.pvpo_cdp"] is sys.modules["warp_taskgen.phase_4.pvpo_cdp"]


def test_legacy_first_import_still_reuses_the_canonical_module() -> None:
    for name in ("worldsim.phase_2", "worldsim.phase_2.phase_2c", "warp_taskgen.phase_2"):
        sys.modules.pop(name, None)

    legacy = importlib.import_module("worldsim.phase_2.phase_2c")
    canonical = importlib.import_module("warp_taskgen.phase_2.phase_2c")

    assert legacy is canonical


def test_representative_imports_share_identity_and_canonical_metadata() -> None:
    names = (
        "phase_1.novel_task_validation",
        "phase_2.phase_2c",
        "phase_2.target_resolution.types",
        "phase_4.pvpo_cdp",
        "seeding.tokens",
        "editors.base",
        "rewards.final_state",
        "sites.catalog",
    )
    env = {**os.environ, "PYTHONPATH": str(PACKAGE_ROOT)}
    for suffix in names:
        for first, second in (("warp_taskgen", "worldsim"), ("worldsim", "warp_taskgen")):
            script = (
                "import importlib; "
                f"first=importlib.import_module('{first}.{suffix}'); "
                f"second=importlib.import_module('{second}.{suffix}'); "
                f"canonical=importlib.import_module('warp_taskgen.{suffix}'); "
                "assert first is second is canonical; "
                f"assert canonical.__name__ == 'warp_taskgen.{suffix}'; "
                f"assert canonical.__spec__.name == 'warp_taskgen.{suffix}'; "
                "assert importlib.reload(first) is canonical"
            )
            completed = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )
            assert completed.returncode == 0, f"{first}->{second} {suffix}: {completed.stderr}"


def test_legacy_patch_reaches_canonical_module() -> None:
    canonical = importlib.import_module("warp_taskgen.atomic_io")
    legacy = importlib.import_module("worldsim.atomic_io")
    sentinel = object()
    legacy._namespace_cutover_sentinel = sentinel
    assert canonical._namespace_cutover_sentinel is sentinel


def test_reload_preserves_canonical_metadata() -> None:
    module = importlib.import_module("worldsim.atomic_io")
    reloaded = importlib.reload(module)
    assert reloaded is importlib.import_module("warp_taskgen.atomic_io")
    assert reloaded.__name__ == "warp_taskgen.atomic_io"
    assert reloaded.__spec__ is not None
    assert reloaded.__spec__.name == "warp_taskgen.atomic_io"


def test_both_cli_module_names_show_help() -> None:
    env = {**os.environ, "PYTHONPATH": str(PACKAGE_ROOT)}
    for module in ("warp_taskgen.main", "worldsim.main"):
        completed = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert "WARP Taskgen" in completed.stdout


def test_retired_wave_d_paths_stay_absent() -> None:
    retired = (
        "phase_2_injections_api",
        "phase_1_generate_new_tasks_validation",
        "phase_2_exposure_contract",
        "phase_2_feasibility",
        "phase_2_text_fill",
    )
    for module in retired:
        for root in ("warp_taskgen.phases", "worldsim.phases"):
            try:
                importlib.import_module(f"{root}.{module}")
            except ModuleNotFoundError:
                continue
            raise AssertionError(
                f"retired compatibility path unexpectedly importable: {root}.{module}"
            )
