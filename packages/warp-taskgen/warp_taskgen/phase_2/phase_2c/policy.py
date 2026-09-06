"""Benchmark/Site feasibility policy contracts for Phase 2c preflight."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Protocol

from warp_taskgen.benchmark_capabilities import normalize_benchmark_name


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


PolicyKey = tuple[str, str]


def _normalize_key(benchmark: object, site: object) -> PolicyKey:
    if not isinstance(benchmark, str) or not benchmark.strip():
        raise ValueError("feasibility policy requires a non-empty benchmark")
    if not isinstance(site, str) or not site.strip():
        raise ValueError("feasibility policy requires a non-empty site")
    return normalize_benchmark_name(benchmark), site.strip().lower()


def _policy_key(policy: FeasibilityPolicy) -> PolicyKey:
    try:
        benchmark = policy.benchmark
        site = policy.site
    except AttributeError as exc:
        raise TypeError("feasibility policy must expose benchmark and site") from exc
    key = _normalize_key(benchmark, site)
    required_methods = (
        "auth_self_test_path",
        "requires_authenticated_preflight",
        "probe_targets",
        "classify_probe",
        "decide_source_data",
        "counts_toward_run_bailout",
        "should_bailout_source_data_run",
        "restore_drop_on_run_bailout",
    )
    missing = [name for name in required_methods if not callable(getattr(policy, name, None))]
    if missing:
        raise TypeError(
            "feasibility policy is missing callable methods: " + ", ".join(sorted(missing))
        )
    return key


@dataclass(frozen=True)
class FeasibilityPolicyCatalog:
    """Immutable benchmark/Site policy bindings for one preflight run."""

    policies: Mapping[PolicyKey, FeasibilityPolicy]

    def __post_init__(self) -> None:
        if not isinstance(self.policies, Mapping):
            raise TypeError("feasibility policy catalog policies must be a mapping")
        normalized: dict[PolicyKey, FeasibilityPolicy] = {}
        for raw_key, policy in self.policies.items():
            if not isinstance(raw_key, tuple) or len(raw_key) != 2:
                raise ValueError(f"invalid feasibility policy key: {raw_key!r}")
            raw_benchmark, raw_site = raw_key
            requested_key = _normalize_key(raw_benchmark, raw_site)
            policy_key = _policy_key(policy)
            if requested_key != policy_key:
                raise ValueError(
                    "feasibility policy key does not match its policy: "
                    f"{requested_key!r} != {policy_key!r}"
                )
            if policy_key in normalized:
                raise ValueError(f"duplicate feasibility policy: {policy_key!r}")
            normalized[policy_key] = policy
        object.__setattr__(self, "policies", MappingProxyType(normalized))

    @classmethod
    def from_policies(cls, policies: Iterable[FeasibilityPolicy]) -> FeasibilityPolicyCatalog:
        """Build a normalized catalog and reject aliases that collide."""
        by_key: dict[PolicyKey, FeasibilityPolicy] = {}
        for policy in policies:
            key = _policy_key(policy)
            if key in by_key:
                raise ValueError(f"duplicate feasibility policy: {key!r}")
            by_key[key] = policy
        return cls(by_key)

    def get(self, benchmark: object, site: object) -> FeasibilityPolicy | None:
        """Return a policy for normalized benchmark/Site names, if present."""
        if not isinstance(site, str):
            return None
        key = (normalize_benchmark_name(benchmark), site.strip().lower())
        return self.policies.get(key)


def default_feasibility_policy_catalog() -> FeasibilityPolicyCatalog:
    """Assemble the explicit built-in WebArena GitLab/Reddit policies."""
    from warp_taskgen.phase_2.phase_2c.webarena_policy import WebArenaFeasibilityPolicy

    return FeasibilityPolicyCatalog.from_policies(
        (
            WebArenaFeasibilityPolicy(site="gitlab", auth_path="/-/profile"),
            WebArenaFeasibilityPolicy(site="reddit"),
        )
    )


def resolve_feasibility_policy(
    benchmark: object,
    site: object,
    *,
    feasibility_policy_catalog: FeasibilityPolicyCatalog,
) -> FeasibilityPolicy | None:
    """Resolve through the Run's policy catalog."""
    return feasibility_policy_catalog.get(benchmark, site)


def task_probe_targets(
    task: dict[str, Any],
    instance_site_url: str,
    *,
    feasibility_policy_catalog: FeasibilityPolicyCatalog,
    benchmark: object | None = None,
) -> list[ProbeTarget]:
    """Resolve probe targets through the Run's policy catalog."""
    site = str(task.get("site") or "").strip().lower()
    if benchmark is None:
        benchmark = next(
            (
                task.get(key)
                for key in ("benchmark", "benchmark_name", "benchmark_adapter")
                if task.get(key)
            ),
            "webarena_verified",
        )
    policy = resolve_feasibility_policy(
        benchmark,
        site,
        feasibility_policy_catalog=feasibility_policy_catalog,
    )
    return policy.probe_targets(task, instance_site_url) if policy is not None else []


__all__ = [
    "FeasibilityPolicy",
    "FeasibilityPolicyCatalog",
    "PolicyKey",
    "PreflightClassification",
    "ProbeTarget",
    "SourceDataDecision",
    "default_feasibility_policy_catalog",
    "resolve_feasibility_policy",
    "task_probe_targets",
]
