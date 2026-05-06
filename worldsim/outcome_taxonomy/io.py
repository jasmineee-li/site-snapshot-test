"""Disk-backed outcome classification entry points."""

from __future__ import annotations

from worldsim.outcome_taxonomy._impl import classify_from_dir

__all__ = ["classify_from_dir"]
