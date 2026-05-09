"""Repo-root discovery helper.

Finding the WARP Taskgen repo root from a module needs to survive module
relocation and installation into site-packages. ``Path(__file__).parent.parent``
breaks silently under those conditions. The sentinel-walk + env-var override
pattern below is stable regardless of internal reorganization.
"""

from __future__ import annotations

import os
from pathlib import Path


def find_repo_root() -> Path:
    """Return the WARP Taskgen repo root directory.

    Precedence:
    1. ``WARP_TASKGEN_REPO_ROOT`` env var (honored verbatim).
    2. Legacy ``WORLDSIM_REPO_ROOT`` env var.
    3. Walk up from this module's location looking for a directory that
       contains both ``pyproject.toml`` and a ``worldsim`` package dir.
    """
    override = (
        os.environ.get("WARP_TASKGEN_REPO_ROOT", "").strip()
        or os.environ.get("WORLDSIM_REPO_ROOT", "").strip()
    )
    if override:
        return Path(override)
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists() and (parent / "worldsim").is_dir():
            return parent
    raise RuntimeError(
        "Cannot locate WARP Taskgen repo root; set WARP_TASKGEN_REPO_ROOT "
        "or legacy WORLDSIM_REPO_ROOT"
    )
