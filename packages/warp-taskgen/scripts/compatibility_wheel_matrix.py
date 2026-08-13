#!/usr/bin/env python3
"""Run the installed-wheel compatibility matrix for the namespace cutover.

This script deliberately runs import probes with a working directory outside
the checkout.  It is called by the package-proof lane after installing the
wheel into a fresh Python environment, so a passing result proves that the
adapter and resources are present in the distribution rather than only in a
source checkout.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_MODULES = (
    "phase_1.novel_task_validation",
    "phase_2.phase_2c",
    "phase_2.target_resolution.types",
    "phase_4.pvpo_cdp",
    "phase_4.result_summary",
    "seeding.tokens",
    "editors.base",
    "rewards.final_state",
    "sites.catalog",
    "cli_status",
)

_RESOURCE_PATHS = (
    "prompts/profile-site.md",
    "prompts/generate-benign-tasks.md",
    "voice_exemplars/registry.json",
    "phase_4/pvpo_query.js",
)


def _run(python: Path, code: str, *, cwd: Path) -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHON_DOTENV_DISABLED"] = "1"
    completed = subprocess.run(
        [str(python), "-c", code],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"installed-wheel probe failed: {detail}")


def _import_probe(first: str, second: str) -> str:
    imports = "; ".join(
        f"first=importlib.import_module('{first}.{suffix}'); "
        f"second=importlib.import_module('{second}.{suffix}'); "
        f"canonical=importlib.import_module('warp_taskgen.{suffix}'); "
        "assert first is second is canonical; "
        f"assert canonical.__name__ == 'warp_taskgen.{suffix}'; "
        f"assert canonical.__spec__ is not None and canonical.__spec__.name == 'warp_taskgen.{suffix}'; "
        "assert importlib.reload(first) is canonical"
        for suffix in _MODULES
    )
    return (
        "import importlib\n"
        f"{imports}\n"
        "sentinel = object(); first._compatibility_sentinel = sentinel; "
        "assert second._compatibility_sentinel is sentinel; "
        "assert canonical._compatibility_sentinel is sentinel\n"
    )


def _root_probe(first: str, second: str) -> str:
    return (
        "import importlib, sys\n"
        f"first=importlib.import_module('{first}'); "
        f"second=importlib.import_module('{second}'); "
        "canonical=importlib.import_module('warp_taskgen'); "
        "assert first is second is canonical; "
        "assert canonical.__name__ == 'warp_taskgen'; "
        "assert canonical.__spec__ is not None and canonical.__spec__.name == 'warp_taskgen'; "
        "assert not any(name.startswith('warp_taskgen.') for name in sys.modules if name != 'warp_taskgen')\n"
    )


def _resource_probe() -> str:
    paths = ", ".join(repr(path) for path in _RESOURCE_PATHS)
    return (
        "from importlib import resources\n"
        "from warp_taskgen.phase_2.text_fill.voice import load_voice_registry\n"
        "from warp_taskgen.prompt_loading import load_prompt\n"
        f"root=resources.files('warp_taskgen'); paths=({paths},)\n"
        "for relative in paths:\n"
        "    resource=root.joinpath(*relative.split('/')); "
        "    assert resource.is_file(), relative; "
        "    assert resource.read_bytes(), relative\n"
        "assert load_prompt('profile-site').strip()\n"
        "registry=load_voice_registry(); assert registry and registry['_registry_path']\n"
    )


def _sidecar_probe(package_root: Path) -> None:
    sidecar_root = package_root / "packages" / "worldsim-agentlab-runner"
    pyproject = sidecar_root / "pyproject.toml"
    if not pyproject.is_file():
        raise RuntimeError(f"sidecar metadata is missing: {pyproject}")
    text = pyproject.read_text(encoding="utf-8")
    required = (
        'name = "warp-taskgen-agentlab-runner"',
        'warp-taskgen-agentlab-runner = "worldsim_agentlab_runner.cli:main"',
        'worldsim-agentlab-runner = "worldsim_agentlab_runner.cli:main"',
    )
    for marker in required:
        if marker not in text:
            raise RuntimeError(f"sidecar metadata missing {marker!r}")

    source_root = sidecar_root / "src" / "worldsim_agentlab_runner"
    source = "\n".join(path.read_text(encoding="utf-8") for path in source_root.glob("*.py"))
    if "from worldsim." in source or "import worldsim." in source:
        raise RuntimeError("AgentLab sidecar still imports the retired core namespace")
    if "warp_taskgen.phase_4" not in (source_root / "sync_pvpo.py").read_text(encoding="utf-8"):
        raise RuntimeError("AgentLab sidecar PVPO bridge does not resolve canonical core imports")


def run(python: Path, *, package_root: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="warp-taskgen-wheel-probe-") as temporary:
        cwd = Path(temporary)
        _run(python, _root_probe("warp_taskgen", "worldsim"), cwd=cwd)
        _run(python, _root_probe("worldsim", "warp_taskgen"), cwd=cwd)
        for first, second in (("warp_taskgen", "worldsim"), ("worldsim", "warp_taskgen")):
            _run(python, _import_probe(first, second), cwd=cwd)
        _run(python, _resource_probe(), cwd=cwd)
        for command in (
            ("-m", "warp_taskgen.main", "--help"),
            ("-m", "worldsim.main", "--help"),
            ("warp-taskgen", "--help"),
            ("worldsim", "--help"),
        ):
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            env["PYTHON_DOTENV_DISABLED"] = "1"
            if command[0] == "-m":
                argv = [str(python), *command]
            else:
                # Use the installed venv's console scripts while retaining a
                # subprocess cwd outside the checkout.
                env["PATH"] = str(python.parent) + os.pathsep + env.get("PATH", "")
                argv = [command[0], *command[1:]]
            completed = subprocess.run(
                argv,
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode or "WARP Taskgen" not in completed.stdout:
                detail = completed.stderr.strip() or completed.stdout.strip()
                raise RuntimeError(f"CLI probe failed for {' '.join(command)}: {detail}")
    _sidecar_probe(package_root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python executable containing the installed wheel",
    )
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Taskgen package checkout used for sidecar metadata verification",
    )
    args = parser.parse_args(argv)
    # Preserve the venv executable path. Resolving its symlink would select the
    # base uv-managed interpreter and lose the wheel installed in this venv.
    run(args.python.absolute(), package_root=args.package_root.resolve())
    print("installed-wheel compatibility matrix: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
