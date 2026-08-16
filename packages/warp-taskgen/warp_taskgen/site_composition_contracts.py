"""Immutable contracts for static Site composition diagnostics."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from warp_taskgen.benchmark_capabilities import DEFAULT_BENCHMARK_CATALOG

if TYPE_CHECKING:
    from warp_taskgen.adversarial_actions.capability_adapters import CapabilityTaskAdapter
    from warp_taskgen.editors._registry import EditorMethodSpec

CapabilityState = Literal["supported", "not_applicable", "unsupported", "missing"]
FindingOutcome = Literal["pass", "failure", "blocked"]
StaticStatus = Literal["complete", "incomplete", "invalid"]
OperationalState = Literal["supported", "unsupported", "missing"]

_CAPABILITY_STATES = frozenset({"supported", "not_applicable", "unsupported", "missing"})
_OPERATIONAL_STATES = frozenset({"supported", "unsupported", "missing"})
_FINDING_OUTCOMES = frozenset({"pass", "failure", "blocked"})
_STATIC_STATUSES = frozenset({"complete", "incomplete", "invalid"})
_REPORT_STATUSES = frozenset({"ready", "blocked", "invalid"})
EDGE_ORDER = (
    "targeting",
    "profile",
    "editor_specs",
    "seed",
    "feasibility",
    "read_surface",
    "readback",
    "final_state",
    "action_cards",
)

_PROVENANCE_PATTERN = re.compile(r"[A-Za-z0-9_.:-]+")
_SENSITIVE_PROVENANCE_TERMS = (
    "authorization",
    "cookie",
    "credential",
    "header",
    "password",
    "payload",
    "secret",
    "token",
)


def _normalize_label(value: object) -> str:
    text = str(value or "").strip().lower()
    normalized = "_".join(part for part in text.replace("-", " ").split() if part)
    return normalized if re.fullmatch(r"[a-z0-9_]+", normalized) else ""


def _freeze_texts(values: Iterable[object]) -> tuple[str, ...]:
    normalized = {str(value).strip() for value in values if str(value).strip()}
    if any(
        len(value) > 200
        or _PROVENANCE_PATTERN.fullmatch(value) is None
        or any(term in value.casefold() for term in _SENSITIVE_PROVENANCE_TERMS)
        for value in normalized
    ):
        raise ValueError("diagnostic provenance must contain symbolic, single-line identities")
    return tuple(sorted(normalized))


@dataclass(frozen=True)
class CapabilityReference[T]:
    """One explicit capability state and, when supported, its existing owner."""

    state: CapabilityState
    owner: T | None
    provenance: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.state not in _CAPABILITY_STATES:
            raise ValueError(f"unknown capability state {self.state!r}")
        if (self.state == "supported") != (self.owner is not None):
            raise ValueError("supported capabilities require an owner; other states forbid one")
        object.__setattr__(self, "provenance", _freeze_texts(self.provenance))


@dataclass(frozen=True)
class SiteBenchmarkBinding:
    """Existing capability owners for one canonical Benchmark/Site pair."""

    benchmark: str
    targeting: CapabilityReference[Any]
    profile: CapabilityReference[Any]
    editor_specs: CapabilityReference[tuple[EditorMethodSpec, ...]]
    seed: CapabilityReference[Any]
    feasibility: CapabilityReference[Any]
    read_surface: CapabilityReference[Any]
    readback: CapabilityReference[Any]
    final_state: CapabilityReference[Any]
    action_cards: CapabilityReference[tuple[CapabilityTaskAdapter, ...]]

    def __post_init__(self) -> None:
        benchmark = DEFAULT_BENCHMARK_CATALOG.normalize(self.benchmark)
        if not benchmark or _normalize_label(benchmark) != benchmark:
            raise ValueError("Site Benchmark binding requires a Benchmark")
        object.__setattr__(self, "benchmark", benchmark)
        for name in EDGE_ORDER:
            if not isinstance(getattr(self, name), CapabilityReference):
                raise TypeError(f"{name} must be a CapabilityReference")


@dataclass(frozen=True)
class SiteDefinition:
    """Immutable diagnostic definition for one Site."""

    site: str
    bindings: tuple[SiteBenchmarkBinding, ...]
    provenance: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        site = _normalize_label(self.site)
        if not site:
            raise ValueError("Site definition requires a Site name")
        bindings = tuple(self.bindings)
        if not bindings:
            raise ValueError("Site definition requires at least one Benchmark binding")
        if any(not isinstance(binding, SiteBenchmarkBinding) for binding in bindings):
            raise TypeError("Site definition bindings must be typed")
        benchmarks = [binding.benchmark for binding in bindings]
        if len(set(benchmarks)) != len(benchmarks):
            raise ValueError(f"duplicate Benchmark binding for Site {site!r}")
        object.__setattr__(self, "site", site)
        object.__setattr__(
            self, "bindings", tuple(sorted(bindings, key=lambda item: item.benchmark))
        )
        object.__setattr__(self, "provenance", _freeze_texts(self.provenance))


@dataclass(frozen=True)
class SiteDoctorRequest:
    site: str
    benchmark: str
    use_case: str
    carrier: str | None = None
    action_kind: str | None = None

    def __post_init__(self) -> None:
        site = _normalize_label(self.site)
        benchmark = DEFAULT_BENCHMARK_CATALOG.normalize(self.benchmark)
        use_case = _normalize_label(self.use_case)
        if not site or not benchmark or _normalize_label(benchmark) != benchmark or not use_case:
            raise ValueError("site doctor requires Site, Benchmark, and use case")
        object.__setattr__(self, "site", site)
        object.__setattr__(self, "benchmark", benchmark)
        object.__setattr__(self, "use_case", use_case)
        for name in ("carrier", "action_kind"):
            value = getattr(self, name)
            object.__setattr__(self, name, str(value).strip() if value else None)


@dataclass(frozen=True)
class ActiveSitePolicy:
    """Explicit diagnostic-only allow-set; definitions cannot authorize themselves."""

    authorized_keys: frozenset[tuple[str, str, str]] = frozenset()

    def __post_init__(self) -> None:
        normalized: set[tuple[str, str, str]] = set()
        for raw in self.authorized_keys:
            if not isinstance(raw, tuple) or len(raw) != 3:
                raise ValueError("active Site policy keys must be (benchmark, site, use_case)")
            benchmark, site, use_case = raw
            key = (
                DEFAULT_BENCHMARK_CATALOG.normalize(benchmark),
                _normalize_label(site),
                _normalize_label(use_case),
            )
            if not all(key) or _normalize_label(key[0]) != key[0]:
                raise ValueError("active Site policy keys require non-empty identities")
            normalized.add(key)
        object.__setattr__(self, "authorized_keys", frozenset(normalized))

    def authorizes(self, request: SiteDoctorRequest) -> bool:
        return (request.benchmark, request.site, request.use_case) in self.authorized_keys


@dataclass(frozen=True)
class OperationalEvidence:
    """Safe outcome states supplied by external host/runtime proof owners."""

    configured_host: OperationalState = "missing"
    admission: OperationalState = "missing"
    execution: OperationalState = "missing"
    scoring: OperationalState = "missing"

    def __post_init__(self) -> None:
        for name in ("configured_host", "admission", "execution", "scoring"):
            if getattr(self, name) not in _OPERATIONAL_STATES:
                raise ValueError(f"unknown operational evidence state for {name}")


@dataclass(frozen=True)
class CapabilityFinding:
    capability: str
    state: CapabilityState
    outcome: FindingOutcome
    code: str
    detail: str
    provenance: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _normalize_label(self.capability):
            raise ValueError("finding capability requires a semantic identity")
        if self.state not in _CAPABILITY_STATES:
            raise ValueError("finding state is invalid")
        if self.outcome not in _FINDING_OUTCOMES:
            raise ValueError("finding outcome is invalid")
        expected_pass = self.state in {"supported", "not_applicable"}
        if (self.outcome == "pass") != expected_pass:
            raise ValueError("finding state and outcome contradict")
        if not isinstance(self.code, str) or _PROVENANCE_PATTERN.fullmatch(self.code) is None:
            raise ValueError("finding code requires a symbolic identity")
        if (
            not isinstance(self.detail, str)
            or not self.detail.strip()
            or "\n" in self.detail
            or "\r" in self.detail
            or "://" in self.detail
            or any(term in self.detail.casefold() for term in _SENSITIVE_PROVENANCE_TERMS)
        ):
            raise ValueError("finding detail requires safe single-line text")
        object.__setattr__(self, "provenance", _freeze_texts(self.provenance))
        object.__setattr__(self, "dependencies", _freeze_texts(self.dependencies))

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "state": self.state,
            "outcome": self.outcome,
            "code": self.code,
            "detail": self.detail,
            "provenance": list(self.provenance),
            "dependencies": list(self.dependencies),
        }


@dataclass(frozen=True)
class SiteDoctorReport:
    site: str
    benchmark: str
    use_case: str
    static_status: StaticStatus
    status: Literal["ready", "blocked", "invalid"]
    definition_digest: str
    findings: tuple[CapabilityFinding, ...]

    def __post_init__(self) -> None:
        if self.static_status not in _STATIC_STATUSES or self.status not in _REPORT_STATUSES:
            raise ValueError("Site doctor report has an invalid status")
        if (self.static_status == "invalid") != (self.status == "invalid"):
            raise ValueError("Site doctor static and overall statuses contradict")
        if self.status == "ready" and self.static_status != "complete":
            raise ValueError("only a complete static report can be ready")
        findings = tuple(self.findings)
        if any(not isinstance(finding, CapabilityFinding) for finding in findings):
            raise TypeError("Site doctor findings must be typed")
        capabilities = [finding.capability for finding in findings]
        if len(set(capabilities)) != len(capabilities):
            raise ValueError("Site doctor findings require unique capability identities")
        object.__setattr__(self, "findings", findings)

    def finding(self, capability: str) -> CapabilityFinding:
        for finding in self.findings:
            if finding.capability == capability:
                return finding
        raise KeyError(capability)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "warp-site-doctor-experimental-v1",
            "contract_kit_version": "warp-site-composition-v1",
            "site": self.site,
            "benchmark": self.benchmark,
            "use_case": self.use_case,
            "static_status": self.static_status,
            "status": self.status,
            "definition_digest": self.definition_digest,
            "findings": [finding.to_dict() for finding in self.findings],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


__all__ = [
    "EDGE_ORDER",
    "ActiveSitePolicy",
    "CapabilityFinding",
    "CapabilityReference",
    "CapabilityState",
    "FindingOutcome",
    "OperationalEvidence",
    "OperationalState",
    "SiteBenchmarkBinding",
    "SiteDefinition",
    "SiteDoctorReport",
    "SiteDoctorRequest",
    "StaticStatus",
]
