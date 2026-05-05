"""Compatibility wrapper for Phase 2c config helpers."""

from __future__ import annotations

from worldsim.phase_2.phase_2c import config as _config

globals().update({name: value for name, value in vars(_config).items() if not name.startswith("__")})

__all__ = [name for name in globals() if not name.startswith("__")]
