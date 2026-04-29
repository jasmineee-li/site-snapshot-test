"""Compatibility wrapper for Phase 2c artifact helpers."""

from __future__ import annotations

from worldsim.phase_2.phase_2c import artifacts as _artifacts

globals().update(
    {name: value for name, value in vars(_artifacts).items() if not name.startswith("__")}
)

__all__ = [name for name in globals() if not name.startswith("__")]
