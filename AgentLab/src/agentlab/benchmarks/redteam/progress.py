"""Progress reporting for the redteam generation pipeline.

Provides structured progress events emitted during generation phases,
replacing silent periods with actionable status updates.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from agentlab.benchmarks.redteam.utils import utc_timestamp

logger = logging.getLogger(__name__)


@dataclass
class PhaseProgress:
    """A single progress event from the generation pipeline."""

    phase_id: str
    status: str  # starting | running | waiting_for_llm | validating | complete | failed | skipped
    elapsed_seconds: float = 0.0
    detail: str = ""
    substep: str = ""
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = utc_timestamp()


class ProgressReporter(Protocol):
    """Protocol for progress reporting backends."""

    def report(self, progress: PhaseProgress) -> None: ...


class LoggingProgressReporter:
    """Reports progress via structured log messages to stderr."""

    def __init__(self, behavior_id: str = "") -> None:
        self.behavior_id = behavior_id
        self._phase_starts: dict[str, float] = {}

    def report(self, progress: PhaseProgress) -> None:
        phase = progress.phase_id
        if progress.status == "starting":
            self._phase_starts[phase] = time.monotonic()

        elapsed = progress.elapsed_seconds
        if not elapsed and phase in self._phase_starts:
            elapsed = time.monotonic() - self._phase_starts[phase]

        prefix = f"[{self.behavior_id}] " if self.behavior_id else ""
        detail = f" — {progress.detail}" if progress.detail else ""
        substep = f" [{progress.substep}]" if progress.substep else ""

        logger.info(
            "%s%s: %s%s%s (%.1fs)",
            prefix,
            phase,
            progress.status,
            substep,
            detail,
            elapsed,
        )


class NullProgressReporter:
    """No-op reporter for tests and batch runs."""

    def report(self, progress: PhaseProgress) -> None:
        pass


class GenerationProgressTracker:
    """Tracks progress across multiple app generations.

    Used by ``generate_apps.py`` to show elapsed time, success/fail
    tallies, and ETA.
    """

    def __init__(self, total: int) -> None:
        self.total = total
        self.completed = 0
        self.succeeded = 0
        self.failed = 0
        self.start_time = time.monotonic()

    def record(self, *, success: bool) -> str:
        """Record a completed app and return a formatted status line."""
        self.completed += 1
        if success:
            self.succeeded += 1
        else:
            self.failed += 1

        elapsed = time.monotonic() - self.start_time
        avg = elapsed / self.completed if self.completed else 0
        remaining = (self.total - self.completed) * avg

        return (
            f"[{self.completed}/{self.total}] "
            f"{self.succeeded} ok, {self.failed} failed | "
            f"elapsed {elapsed:.0f}s, ETA {remaining:.0f}s"
        )
