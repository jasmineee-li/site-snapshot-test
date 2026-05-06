"""Trajectory signal dataclasses and extraction entry point."""

from __future__ import annotations

from worldsim.outcome_taxonomy._impl import (
    ClassifiedOutcome,
    TrajectorySignals,
    extract_signals,
)

__all__ = ["ClassifiedOutcome", "TrajectorySignals", "extract_signals"]
