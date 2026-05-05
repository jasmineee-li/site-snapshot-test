"""Compatibility wrapper for Phase 2 injection generation.

Canonical implementation lives under :mod:`worldsim.phase_2`. This module stays
for one migration cycle so hidden consumers of ``worldsim.phases`` imports keep
working while in-repo imports move to feature-owned modules.
"""

from __future__ import annotations

import sys

from worldsim.phase_2 import runner as _runner

sys.modules[__name__] = _runner
