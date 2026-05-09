"""Named crash failpoints for crash/resume testing.

Failpoints are disabled by default. When the ``WORLDSIM_FAILPOINTS``
environment variable is set to a comma-separated list of names, calling
``crash_if_enabled(name)`` hard-exits the current process if *name* is
active. This is intentionally coarse: the parent test process runs the
pipeline in a child subprocess and then verifies resume behavior.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

FAILPOINTS_ENV = "WORLDSIM_FAILPOINTS"
FAILPOINT_EXIT_CODE = 91


def active_failpoints() -> set[str]:
    raw = os.environ.get(FAILPOINTS_ENV, "")
    return {item.strip() for item in raw.split(",") if item.strip()}


def is_enabled(name: str) -> bool:
    active = active_failpoints()
    return bool(active and (name in active or "*" in active))


def crash_if_enabled(name: str) -> None:
    if not is_enabled(name):
        return
    logger.warning("Crash failpoint triggered: %s", name)
    os._exit(FAILPOINT_EXIT_CODE)

