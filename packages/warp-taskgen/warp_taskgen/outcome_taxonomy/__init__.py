"""Public package surface for the Phase 4 outcome taxonomy classifier."""

from __future__ import annotations

# ruff: noqa: F401
from warp_taskgen.outcome_taxonomy import _impl as _legacy_impl
from warp_taskgen.outcome_taxonomy.classification import classify
from warp_taskgen.outcome_taxonomy.io import classify_from_dir
from warp_taskgen.outcome_taxonomy.serialization import as_result_fields
from warp_taskgen.outcome_taxonomy.signals import (
    ClassifiedOutcome,
    TrajectorySignals,
    extract_signals,
)
from warp_taskgen.outcome_taxonomy.summary import (
    format_stratified_summary_log,
    stratified_summary,
)

globals().update(
    {name: value for name, value in vars(_legacy_impl).items() if not name.startswith("__")}
)
