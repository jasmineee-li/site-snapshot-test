"""Phase 2c report and observability types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class _ReplicaStats:
    """Per-replica observability counters for a Phase 2c run.

    Lives entirely in-process; logged as a single line per replica at
    end of run. Cheap enough to leave always-on — this is the data a
    future AIMD wrapper (or manual cap tuning) needs; shipping it here
    avoids guessing from dmesg and nginx error logs next time.
    """

    site_name: str
    replica_name: str
    requests: int = 0
    errors: int = 0
    in_flight_peak: int = 0
    latencies_ms: list[float] = field(default_factory=list)

    def record(self, *, elapsed_ms: float, ok: bool) -> None:
        self.requests += 1
        if not ok:
            self.errors += 1
        # Cap the sample list so long runs do not balloon memory; 2048
        # samples is plenty for p50/p99 estimation within ±1 %.
        if len(self.latencies_ms) < 2048:
            self.latencies_ms.append(elapsed_ms)

    def summary(self) -> str:
        if not self.latencies_ms:
            return (
                f"replica={self.replica_name} site={self.site_name} "
                f"requests={self.requests} errors={self.errors} "
                f"in_flight_peak={self.in_flight_peak} latency_ms=<none>"
            )
        ordered = sorted(self.latencies_ms)
        n = len(ordered)
        p50 = ordered[n // 2]
        p99 = ordered[min(n - 1, (n * 99) // 100)]
        return (
            f"replica={self.replica_name} site={self.site_name} "
            f"requests={self.requests} errors={self.errors} "
            f"in_flight_peak={self.in_flight_peak} "
            f"p50_ms={p50:.0f} p99_ms={p99:.0f}"
        )


@dataclass(frozen=True)
class FeasibilityReport:
    """Aggregated outcome of a Phase 2c run.

    The caller in ``phase_2_injections`` is responsible for persisting the
    three artifacts; this dataclass is a pure value type.
    """

    verified: list[dict[str, Any]]
    infeasible: list[dict[str, Any]]
    skipped_already_verified: list[dict[str, Any]]
    cleanup_warnings: list[str]
    host_fingerprint: dict[str, str]
    elapsed_seconds: float
    per_site_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    phase_2_status: str | None = None
    # Bug I: tasks whose benign_target_resource preflight deterministically
    # failed (login_redirect, 404, 403, 410, 401). Separate from
    # ``infeasible`` because they are dataset-quality issues, not probe
    # failures; re-running Phase 2c will not rehabilitate them.
    dropped_source_data: list[dict[str, Any]] = field(default_factory=list)
    # Number of complete task units reconstructed from durable checkpoints.
    # This is observational and does not change the verified/infeasible
    # meanings consumed by Phase 3/4.
    reused_checkpoints: int = 0


__all__ = ["FeasibilityReport", "_ReplicaStats"]
