"""Benchmark/site-local policy for Phase 2c source-data preflight."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from worldsim.benchmark_capabilities import normalize_benchmark_name


@dataclass(frozen=True)
class PreflightClassification:
    kind: str
    quarantine: bool
    http_status: int | None
    detail: str


@dataclass(frozen=True)
class ProbeTarget:
    url: str
    source: str


@dataclass(frozen=True)
class SourceDataDecision:
    action: Literal["keep", "drop", "bailout"]
    classification: PreflightClassification | None = None
    target: ProbeTarget | None = None
    evidence: dict[str, Any] = field(default_factory=dict)


class FeasibilityPolicy(Protocol):
    benchmark: str
    site: str

    def auth_self_test_path(self) -> str | None: ...

    def requires_authenticated_preflight(self) -> bool: ...

    def probe_targets(self, task: dict[str, Any], instance_site_url: str) -> list[ProbeTarget]: ...

    def classify_probe(
        self,
        *,
        status: int | None,
        headers: dict[str, str] | None,
        body_snippet: str,
        exception_name: str | None,
    ) -> PreflightClassification: ...

    def decide_source_data(
        self,
        *,
        task: dict[str, Any],
        classifications_by_target: dict[int, list[PreflightClassification]],
        target_audit: dict[int, ProbeTarget],
        candidate_replica_count: int,
        login_redirect_count: int,
        probed_count: int,
        bailout_ratio: float,
    ) -> SourceDataDecision: ...

    def counts_toward_run_bailout(self, classification: PreflightClassification) -> bool: ...

    def should_bailout_source_data_run(
        self,
        *,
        bailout_count: int,
        probed_count: int,
        bailout_ratio: float,
    ) -> bool: ...

    def restore_drop_on_run_bailout(self, issue: dict[str, Any]) -> bool: ...


_POLICIES: dict[tuple[str, str], FeasibilityPolicy] = {}


def register_feasibility_policy(policy: FeasibilityPolicy) -> None:
    key = (normalize_benchmark_name(policy.benchmark), policy.site.strip().lower())
    _POLICIES[key] = policy


def get_feasibility_policy(benchmark: str, site: str) -> FeasibilityPolicy | None:
    return _POLICIES.get((normalize_benchmark_name(benchmark), site.strip().lower()))


def clear_feasibility_policy_registry() -> None:
    _POLICIES.clear()


__all__ = [
    "FeasibilityPolicy",
    "PreflightClassification",
    "ProbeTarget",
    "SourceDataDecision",
    "clear_feasibility_policy_registry",
    "get_feasibility_policy",
    "register_feasibility_policy",
]
