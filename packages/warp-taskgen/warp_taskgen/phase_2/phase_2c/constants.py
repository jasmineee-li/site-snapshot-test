"""Phase 2c constants shared by the runner, artifacts, and tests."""

from __future__ import annotations

from warp_taskgen.phase_2.phase_2c.probes import (
    _BROWSER_PROBE_CAP,
    _PROBE_LAUNCH_ARGS,
    _RENDER_UNVERIFIED_KIND,
    _RENDER_UNVERIFIED_RETRY_DELAY_S,
    _SKIP_RENDER_CHECK_ENV,
)

# Failpoint bases fired by ``write_json_atomic``. Callers wire these up so the
# crash-resume tests can interrupt each write.
FAILPOINT_DATASET = "phase_2.output.feasibility_dataset"
FAILPOINT_QUARANTINE = "phase_2.output.feasibility_quarantine"
FAILPOINT_REPORT = "phase_2.output.feasibility_report"
FAILPOINT_DROPPED_SOURCE_DATA = "phase_2.output.feasibility_dropped_source_data"

__all__ = [
    "FAILPOINT_DATASET",
    "FAILPOINT_DROPPED_SOURCE_DATA",
    "FAILPOINT_QUARANTINE",
    "FAILPOINT_REPORT",
    "_BROWSER_PROBE_CAP",
    "_PROBE_LAUNCH_ARGS",
    "_RENDER_UNVERIFIED_KIND",
    "_RENDER_UNVERIFIED_RETRY_DELAY_S",
    "_SKIP_RENDER_CHECK_ENV",
]
