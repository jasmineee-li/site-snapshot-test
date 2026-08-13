"""Compatibility alias for the feature-owned Phase 2a runner API.

The implementation lives at :mod:`worldsim.phase_2.runner_api`.  Keep this
historical module path as a one-cycle alias for out-of-tree consumers while
callers migrate to the feature-owned package.
"""

from __future__ import annotations

import sys

from worldsim.phase_2 import runner_api

sys.modules[__name__] = runner_api
