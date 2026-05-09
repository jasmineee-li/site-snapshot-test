"""Public package surface for the Phase 4 outcome taxonomy classifier."""

from __future__ import annotations

# ruff: noqa: F401
from worldsim.outcome_taxonomy import _impl as _legacy_impl
from worldsim.outcome_taxonomy.classification import classify
from worldsim.outcome_taxonomy.io import classify_from_dir
from worldsim.outcome_taxonomy.serialization import as_result_fields
from worldsim.outcome_taxonomy.signals import (
    ClassifiedOutcome,
    TrajectorySignals,
    extract_signals,
)
from worldsim.outcome_taxonomy.summary import (
    format_stratified_summary_log,
    stratified_summary,
)

globals().update(
    {
        name: value
        for name, value in vars(_legacy_impl).items()
        if not name.startswith("__")
    }
)
