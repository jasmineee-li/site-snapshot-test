"""Compatibility facade for Phase 1 generate-new-tasks validation."""

from __future__ import annotations

# ruff: noqa: F403
from worldsim.phase_1.novel_task_validation import *
from worldsim.phase_1.novel_task_validation import _impl as _legacy_impl

globals().update(
    {
        name: value
        for name, value in vars(_legacy_impl).items()
        if not name.startswith("__")
    }
)
