"""Compatibility wrapper for Phase 2 output helpers."""

from __future__ import annotations

from worldsim.phase_2 import output as _output

globals().update({name: value for name, value in vars(_output).items() if not name.startswith("__")})

__all__ = [name for name in globals() if not name.startswith("__")]
