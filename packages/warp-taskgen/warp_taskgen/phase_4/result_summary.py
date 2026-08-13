"""Compatibility facade for Phase 4 result summaries."""

from __future__ import annotations

# ruff: noqa: F403
from warp_taskgen.phase_4.result_summary import *
from warp_taskgen.phase_4.result_summary import _impl as _legacy_impl

globals().update(
    {name: value for name, value in vars(_legacy_impl).items() if not name.startswith("__")}
)
