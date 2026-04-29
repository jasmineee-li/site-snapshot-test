"""Compatibility wrapper for Phase 2 target resolution.

Canonical implementation lives under :mod:`worldsim.phase_2.target_resolution`.
"""

from __future__ import annotations

import sys

from worldsim.phase_2.target_resolution import runner as _runner

sys.modules[__name__] = _runner
