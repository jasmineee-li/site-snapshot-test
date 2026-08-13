"""Structural contracts for the completed WARP Taskgen namespace cutover."""

from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).parents[1]


def test_canonical_package_is_the_only_implementation_tree() -> None:
    canonical = PACKAGE_ROOT / "warp_taskgen"
    legacy = PACKAGE_ROOT / "worldsim"

    assert canonical.is_dir()
    assert (canonical / "main.py").is_file()
    assert not legacy.exists()


def test_build_metadata_packages_only_the_canonical_namespace() -> None:
    metadata = tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert metadata["project"]["scripts"] == {"warp-taskgen": "warp_taskgen.main:main"}
    assert metadata["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["warp_taskgen"]
    assert "/worldsim" not in metadata["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]


def test_distribution_and_package_versions_match_upgrade_release() -> None:
    from warp_taskgen import __version__

    metadata = tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert metadata["project"]["version"] == "0.1.1"
    assert __version__ == "0.1.1"


def test_retired_core_namespace_fails_closed() -> None:
    importlib.invalidate_caches()

    assert importlib.util.find_spec("worldsim") is None
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("worldsim")


def test_canonical_cli_module_shows_help() -> None:
    env = {**os.environ, "PYTHONPATH": str(PACKAGE_ROOT)}
    completed = subprocess.run(
        [sys.executable, "-m", "warp_taskgen.main", "--help"],
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
        assert importlib.util.find_spec(f"warp_taskgen.phases.{module}") is None
