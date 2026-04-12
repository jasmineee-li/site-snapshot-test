"""Aggregate cost reporting across sandbox calls within a pipeline run.

Accumulates per-sandbox cost data from ``run_claude_in_sandbox`` ``_summary``
payloads and produces phase-level and pipeline-level reports written to
``logs/cost_report.json``.

Usage: import the module-level ``tracker`` singleton and call ``tracker.record``
after each sandbox call. At phase end, call ``tracker.log_phase_summary`` and
``tracker.save``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SandboxCostEntry:
    """One record per ``run_claude_in_sandbox`` call."""

    phase: str
    task_id: str | None
    site: str | None
    total_cost_usd: float | None
    num_turns: int | None
    duration_ms: int | None
    session_id: str | None
    model_usage: dict | None
    timestamp: str


class CostTracker:
    """Accumulates cost data across sandbox calls within a pipeline run."""

    def __init__(self) -> None:
        self.entries: list[SandboxCostEntry] = []

    # -- Recording ------------------------------------------------------------

    def record(
        self,
        phase: str,
        summary_json: str | None,
        *,
        task_id: str | None = None,
        site: str | None = None,
    ) -> None:
        """Parse a ``_summary`` JSON string and append an entry.

        Gracefully handles ``None`` or malformed JSON by recording an entry
        with cost fields set to ``None``.
        """
        parsed: dict[str, Any] = {}
        if summary_json:
            try:
                parsed = json.loads(summary_json)
            except (json.JSONDecodeError, TypeError):
                logger.debug("Could not parse _summary JSON; recording entry with nulls")

        entry = SandboxCostEntry(
            phase=phase,
            task_id=task_id,
            site=site,
            total_cost_usd=parsed.get("total_cost_usd"),
            num_turns=parsed.get("num_turns"),
            duration_ms=parsed.get("duration_ms"),
            session_id=parsed.get("session_id"),
            model_usage=parsed.get("model_usage"),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.entries.append(entry)

    # -- Aggregation ----------------------------------------------------------

    def total_cost(self) -> float:
        """Sum of ``total_cost_usd`` across all entries (skipping None)."""
        return sum(e.total_cost_usd for e in self.entries if e.total_cost_usd is not None)

    def cost_by_phase(self) -> dict[str, float]:
        """Cost summed per phase label."""
        by_phase: dict[str, float] = {}
        for e in self.entries:
            if e.total_cost_usd is not None:
                by_phase[e.phase] = by_phase.get(e.phase, 0.0) + e.total_cost_usd
        return by_phase

    def _entries_for_phase(self, phase: str) -> list[SandboxCostEntry]:
        return [e for e in self.entries if e.phase == phase]

    def phase_sandbox_count(self, phase: str) -> int:
        return len(self._entries_for_phase(phase))

    def phase_total_turns(self, phase: str) -> int:
        return sum(
            e.num_turns for e in self._entries_for_phase(phase) if e.num_turns is not None
        )

    def phase_total_duration_s(self, phase: str) -> float:
        """Total sandbox wall-clock time for a phase, in seconds."""
        return sum(
            e.duration_ms for e in self._entries_for_phase(phase) if e.duration_ms is not None
        ) / 1000.0

    # -- Reporting ------------------------------------------------------------

    def summary_report(self) -> str:
        """Human-readable multi-line cost summary."""
        lines: list[str] = []
        for phase, cost in sorted(self.cost_by_phase().items()):
            count = self.phase_sandbox_count(phase)
            turns = self.phase_total_turns(phase)
            dur = self.phase_total_duration_s(phase)
            lines.append(
                f"{phase} cost: ${cost:.2f} "
                f"({count} sandbox{'es' if count != 1 else ''}, "
                f"{turns} turns, {dur:.0f}s)"
            )
        total = self.total_cost()
        lines.append(f"Pipeline total: ${total:.2f}")
        return "\n".join(lines)

    def log_phase_summary(self, phase: str) -> None:
        """Log a one-line cost summary for a single phase."""
        entries = self._entries_for_phase(phase)
        if not entries:
            return
        cost = sum(e.total_cost_usd for e in entries if e.total_cost_usd is not None)
        count = len(entries)
        turns = sum(e.num_turns for e in entries if e.num_turns is not None)
        dur = sum(e.duration_ms for e in entries if e.duration_ms is not None) / 1000.0
        logger.info(
            "%s cost: $%.2f (%d sandbox%s, %d turns, %.0fs)",
            phase, cost, count, "es" if count != 1 else "", turns, dur,
        )

    # -- Persistence ----------------------------------------------------------

    def save(self, path: Path) -> None:
        """Write the full cost report to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "total_cost_usd": round(self.total_cost(), 4),
            "by_phase": {k: round(v, 4) for k, v in sorted(self.cost_by_phase().items())},
            "entries": [asdict(e) for e in self.entries],
        }
        path.write_text(json.dumps(report, indent=2))
        logger.info("Cost report written to %s", path)

    def load(self, path: Path) -> None:
        """Load previously saved entries (for --resume across phases)."""
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            for raw in data.get("entries", []):
                self.entries.append(SandboxCostEntry(**raw))
            logger.info("Loaded %d cost entries from %s", len(data.get("entries", [])), path)
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.warning("Could not load cost report from %s: %s", path, exc)


# Module-level singleton. Since each phase runs in its own ``asyncio.run()``
# call from ``main.py``, all sandbox calls within a phase share this tracker.
# The tracker persists across phases within the same process via ``save``/``load``.
tracker = CostTracker()
