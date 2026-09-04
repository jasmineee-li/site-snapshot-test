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
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import Any

from warp_taskgen.atomic_io import write_json_atomic

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


class CostReportMalformedError(ValueError):
    """Raised when a Phase 1 paid dispatch would overwrite bad cost evidence."""


@dataclass(frozen=True)
class CostReportInspection:
    """Read-only accounting state used by dispatch gates and operator status."""

    status: str
    known_total_cost_usd: float | None
    known_entry_count: int
    unknown_entry_count: int
    recorded_entry_count: int
    completeness: str = "lower_bound"
    reason_code: str | None = None
    report: dict[str, Any] | None = None
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the stable, non-secret status projection."""

        return {
            "path": self.path,
            "status": self.status,
            "known_total_cost_usd": self.known_total_cost_usd,
            "known_entry_count": self.known_entry_count,
            "unknown_entry_count": self.unknown_entry_count,
            "recorded_entry_count": self.recorded_entry_count,
            "completeness": self.completeness,
            "reason_code": self.reason_code,
        }


_ENTRY_FIELDS = frozenset(
    {
        "phase",
        "task_id",
        "site",
        "total_cost_usd",
        "num_turns",
        "duration_ms",
        "session_id",
        "model_usage",
        "timestamp",
    }
)


def _nonnegative_number(value: Any, *, name: str, allow_none: bool = True) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a non-negative number or null")
    try:
        normalized = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative number or null") from exc
    if not isfinite(normalized) or normalized < 0:
        raise ValueError(f"{name} must be a non-negative number or null")
    return normalized


def _nonnegative_integer(value: Any, *, name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer or null")


def _parse_report(data: Any) -> tuple[dict[str, Any], list[SandboxCostEntry]]:
    if not isinstance(data, dict):
        raise ValueError("report must be a JSON object")
    raw_entries = data.get("entries")
    if not isinstance(raw_entries, list):
        raise ValueError("report.entries must be a JSON array")

    top_level_cost = data.get("total_cost_usd")
    if top_level_cost is not None:
        _nonnegative_number(top_level_cost, name="report.total_cost_usd")
    by_phase = data.get("by_phase")
    if by_phase is not None:
        if not isinstance(by_phase, dict):
            raise ValueError("report.by_phase must be a JSON object")
        for phase, value in by_phase.items():
            if not isinstance(phase, str) or not phase:
                raise ValueError("report.by_phase keys must be non-empty strings")
            _nonnegative_number(value, name=f"report.by_phase[{phase!r}]")

    entries: list[SandboxCostEntry] = []
    for index, raw in enumerate(raw_entries):
        if not isinstance(raw, dict):
            raise ValueError(f"report.entries[{index}] must be a JSON object")
        if set(raw) != _ENTRY_FIELDS:
            raise ValueError(
                f"report.entries[{index}] has invalid fields; expected {sorted(_ENTRY_FIELDS)}"
            )
        try:
            entry = SandboxCostEntry(**raw)
        except TypeError as exc:
            raise ValueError(f"report.entries[{index}] has invalid shape") from exc
        if not isinstance(entry.phase, str) or not entry.phase:
            raise ValueError(f"report.entries[{index}].phase must be a non-empty string")
        for name, value in (
            ("task_id", entry.task_id),
            ("site", entry.site),
            ("session_id", entry.session_id),
        ):
            if value is not None and not isinstance(value, str):
                raise ValueError(f"report.entries[{index}].{name} must be a string or null")
        _nonnegative_number(
            entry.total_cost_usd,
            name=f"report.entries[{index}].total_cost_usd",
        )
        _nonnegative_integer(entry.num_turns, name=f"report.entries[{index}].num_turns")
        _nonnegative_integer(entry.duration_ms, name=f"report.entries[{index}].duration_ms")
        if entry.model_usage is not None and not isinstance(entry.model_usage, dict):
            raise ValueError(f"report.entries[{index}].model_usage must be an object or null")
        if not isinstance(entry.timestamp, str) or not entry.timestamp:
            raise ValueError(f"report.entries[{index}].timestamp must be a non-empty string")
        entries.append(entry)
    return data, entries


class CostTracker:
    """Accumulates cost data across sandbox calls within a pipeline run."""

    def __init__(self) -> None:
        self.entries: list[SandboxCostEntry] = []
        self._loaded_path: Path | None = None

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
                decoded = json.loads(summary_json)
                if isinstance(decoded, dict):
                    parsed = decoded
            except (json.JSONDecodeError, TypeError):
                logger.debug("Could not parse _summary JSON; recording entry with nulls")

        total_cost_usd = parsed.get("total_cost_usd")
        if isinstance(total_cost_usd, bool) or not isinstance(total_cost_usd, (int, float)):
            total_cost_usd = None
        else:
            try:
                if not isfinite(float(total_cost_usd)) or float(total_cost_usd) < 0:
                    total_cost_usd = None
            except (OverflowError, ValueError):
                total_cost_usd = None

        entry = SandboxCostEntry(
            phase=phase,
            task_id=task_id,
            site=site,
            total_cost_usd=total_cost_usd,
            num_turns=(
                parsed.get("num_turns")
                if isinstance(parsed.get("num_turns"), int)
                and not isinstance(parsed.get("num_turns"), bool)
                else None
            ),
            duration_ms=(
                parsed.get("duration_ms")
                if isinstance(parsed.get("duration_ms"), int)
                and not isinstance(parsed.get("duration_ms"), bool)
                else None
            ),
            session_id=parsed.get("session_id")
            if isinstance(parsed.get("session_id"), str)
            else None,
            model_usage=parsed.get("model_usage")
            if isinstance(parsed.get("model_usage"), dict)
            else None,
            timestamp=datetime.now(UTC).isoformat(),
        )
        self.entries.append(entry)

    def record_and_save(
        self,
        phase: str,
        summary_json: str | None,
        path: Path,
        *,
        task_id: str | None = None,
        site: str | None = None,
    ) -> None:
        """Record one observation and persist it before the caller continues.

        This deliberately remains a small operation on the existing report;
        it is not a provider-attempt ledger. If persistence fails, the
        exception reaches the paid-call seam and the caller must fail closed.
        """

        # Treat the requested report as authoritative for every immediate
        # append. This also notices a same-path deletion between calls and
        # prevents stale in-memory rows from crossing a new run boundary.
        self.load(path)
        self.record(phase, summary_json, task_id=task_id, site=site)
        self.save(path)

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
        return sum(e.num_turns for e in self._entries_for_phase(phase) if e.num_turns is not None)

    def phase_total_duration_s(self, phase: str) -> float:
        """Total sandbox wall-clock time for a phase, in seconds."""
        return (
            sum(e.duration_ms for e in self._entries_for_phase(phase) if e.duration_ms is not None)
            / 1000.0
        )

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
            phase,
            cost,
            count,
            "es" if count != 1 else "",
            turns,
            dur,
        )

    # -- Persistence ----------------------------------------------------------

    def save(self, path: Path) -> None:
        """Write the full cost report to a JSON file.

        Atomic write via ``warp_taskgen.atomic_io.write_json_atomic`` — a
        crash mid-write would leave partial JSON on disk, which
        ``load`` would then reject, silently dropping prior-phase cost
        entries from the cross-phase total.
        """
        report = {
            "total_cost_usd": round(self.total_cost(), 4),
            "by_phase": {k: round(v, 4) for k, v in sorted(self.cost_by_phase().items())},
            "entries": [asdict(e) for e in self.entries],
        }
        write_json_atomic(path, report)
        self._loaded_path = path
        logger.info("Cost report written to %s", path)

    # -- Validation ----------------------------------------------------------

    def inspect_report(self, path: Path) -> CostReportInspection:
        """Inspect an existing report without mutating it or the tracker."""

        if not path.exists():
            return CostReportInspection(
                path=str(path),
                status="missing",
                known_total_cost_usd=None,
                known_entry_count=0,
                unknown_entry_count=0,
                recorded_entry_count=0,
            )
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            report, entries = _parse_report(data)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("Could not inspect cost report from %s: %s", path, exc)
            return CostReportInspection(
                path=str(path),
                status="malformed",
                known_total_cost_usd=None,
                known_entry_count=0,
                unknown_entry_count=0,
                recorded_entry_count=0,
                reason_code="cost_report_malformed",
            )

        known = [entry for entry in entries if entry.total_cost_usd is not None]
        unknown_count = len(entries) - len(known)
        return CostReportInspection(
            path=str(path),
            status="valid",
            known_total_cost_usd=round(
                sum(float(entry.total_cost_usd) for entry in known),
                4,
            ),
            known_entry_count=len(known),
            unknown_entry_count=unknown_count,
            recorded_entry_count=len(entries),
            report=report,
        )

    def ensure_phase1_paid_dispatch_allowed(self, path: Path) -> CostReportInspection:
        """Reject a malformed prior report before another Phase 1 paid call."""

        inspection = self.inspect_report(path)
        if inspection.status == "malformed":
            raise CostReportMalformedError(
                "Phase 1 paid dispatch refused: cost report is malformed at "
                f"{path}. Repair or move that report before retrying."
            )
        return inspection

    def load(self, path: Path) -> None:
        """Load previously saved entries (for --resume across phases).

        Replaces in-memory entries with the file contents to prevent
        double-counting when a phase is re-run.
        """
        inspection = self.inspect_report(path)
        if inspection.status == "missing":
            self.entries = []
            self._loaded_path = path
            return
        if inspection.status == "malformed":
            # Preserve the historical cross-phase behavior: callers may
            # decide whether their phase should overwrite an invalid report.
            # Phase 1's finalizer explicitly preserves it, while the paid
            # dispatch gate refuses to proceed against it.
            logger.warning("Could not load malformed cost report from %s", path)
            return
        assert inspection.report is not None
        _report, loaded = _parse_report(inspection.report)
        self.entries = loaded
        self._loaded_path = path
        logger.info("Loaded %d cost entries from %s", len(loaded), path)


# Module-level singleton. Since each phase runs in its own ``asyncio.run()``
# call from ``main.py``, all sandbox calls within a phase share this tracker.
# The tracker persists across phases within the same process via ``save``/``load``.
tracker = CostTracker()
