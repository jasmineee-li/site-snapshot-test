"""Immutable declarations and reports for static Site Composition checks.

This module deliberately contains no executable Site, editor, policy, browser,
or evaluator objects.  A declaration identifies an existing host-owned seam;
the check compiler only reads those declarations.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from warp_taskgen import __version__ as WARP_TASKGEN_VERSION
from warp_taskgen.benchmark_capabilities import DEFAULT_BENCHMARK_CATALOG

type StaticCapabilityState = Literal["supported", "not_applicable", "unsupported", "missing"]
type SiteCompositionStatus = Literal["complete", "incomplete", "invalid"]
type SiteCompositionFindingOutcome = Literal["pass", "failure"]
type SiteCompositionDigest = str
type SiteCompositionScope = Literal[
    "static_diagnostic", "warp_generation", "warp_execution", "warp_evaluation"
]
type SiteOwnerRole = Literal[
    "site_targeting",
    "profile",
    "editor_specification",
    "regular_participant_writer",
    "feasibility",
    "read_surface",
    "readback",
    "final_state_evaluation",
    "action_cards",
]

SITE_OWNER_ROLE_ORDER: tuple[SiteOwnerRole, ...] = (
    "site_targeting",
    "profile",
    "editor_specification",
    "regular_participant_writer",
    "feasibility",
    "read_surface",
    "readback",
    "final_state_evaluation",
    "action_cards",
)
SITE_COMPOSITION_CONTRACT_VERSION = "warp-site-composition-v1"
SITE_COMPOSITION_SOURCE_PACKAGE = "warp-taskgen"
SITE_OWNER_CONTRACT_VERSION = "v1"

_CAPABILITY_STATES = frozenset({"supported", "not_applicable", "unsupported", "missing"})
_DECLARATION_STATES = frozenset({"supported", "unsupported", "missing"})
_FINDING_OUTCOMES = frozenset({"pass", "failure"})
_STATUSES = frozenset({"complete", "incomplete", "invalid"})
_IDENTITY = re.compile(r"[a-z0-9][a-z0-9_.:-]*\Z")
_PROVENANCE = re.compile(r"[A-Za-z0-9_.:-]+\Z")
_SENSITIVE_TERMS = (
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


def _normalize_symbol(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not text or _IDENTITY.fullmatch(text.lower()) is None:
        raise ValueError(f"{field} requires a stable semantic identity")
    return text.lower()


def _freeze_provenance(values: Iterable[object]) -> tuple[str, ...]:
    normalized = {str(value).strip() for value in values if str(value).strip()}
    if any(
        len(value) > 200
        or _PROVENANCE.fullmatch(value) is None
        or any(term in value.casefold() for term in _SENSITIVE_TERMS)
        for value in normalized
    ):
        raise ValueError("diagnostic provenance must contain symbolic, single-line identities")
    return tuple(sorted(normalized))


def _freeze_symbols(values: Iterable[object], *, field: str) -> tuple[str, ...]:
    normalized = tuple(sorted({_normalize_symbol(value, field=field) for value in values}))
    return normalized


@dataclass(frozen=True)
class SiteOwnerDeclaration:
    """Data-only declaration for one host-owned Site capability seam."""

    state: StaticCapabilityState
    owner_id: str | None = None
    contract_version: str = "v1"
    provenance: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.state == "not_applicable":
            raise ValueError("not_applicable is compiler-derived and forbidden in declarations")
        if self.state not in _DECLARATION_STATES:
            raise ValueError("unknown static capability declaration state")
        owner_id = None
        if self.owner_id is not None:
            owner_id = _normalize_symbol(self.owner_id, field="owner_id")
        if self.state == "supported" and owner_id is None:
            raise ValueError("supported owner declarations require owner_id")
        if self.state != "supported" and owner_id is not None:
            raise ValueError("non-supported owner declarations forbid owner_id")
        version = _normalize_symbol(self.contract_version, field="contract_version")
        if version != SITE_OWNER_CONTRACT_VERSION:
            raise ValueError(
                "unsupported Site owner contract version: "
                f"{version!r}; expected {SITE_OWNER_CONTRACT_VERSION!r}"
            )
        object.__setattr__(self, "owner_id", owner_id)
        object.__setattr__(self, "contract_version", version)
        object.__setattr__(self, "provenance", _freeze_provenance(self.provenance))


@dataclass(frozen=True)
class SiteBenchmarkComposition:
    """Immutable owner declarations for one Benchmark projection of a Site."""

    benchmark: str
    site_targeting: SiteOwnerDeclaration
    profile: SiteOwnerDeclaration
    editor_specification: SiteOwnerDeclaration
    regular_participant_writer: SiteOwnerDeclaration
    feasibility: SiteOwnerDeclaration
    read_surface: SiteOwnerDeclaration
    readback: SiteOwnerDeclaration
    final_state_evaluation: SiteOwnerDeclaration
    action_cards: SiteOwnerDeclaration
    supported_carriers: tuple[str, ...] = ()
    supported_action_kinds: tuple[str, ...] = ()
    provenance: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        benchmark = DEFAULT_BENCHMARK_CATALOG.resolve(self.benchmark).canonical_name
        object.__setattr__(self, "benchmark", benchmark)
        for role in SITE_OWNER_ROLE_ORDER:
            declaration = getattr(self, role)
            if not isinstance(declaration, SiteOwnerDeclaration):
                raise TypeError(f"{role} must be a SiteOwnerDeclaration")
        object.__setattr__(
            self,
            "supported_carriers",
            _freeze_symbols(self.supported_carriers, field="carrier"),
        )
        object.__setattr__(
            self,
            "supported_action_kinds",
            _freeze_symbols(self.supported_action_kinds, field="action_kind"),
        )
        object.__setattr__(self, "provenance", _freeze_provenance(self.provenance))

    def owner(self, role: SiteOwnerRole) -> SiteOwnerDeclaration:
        if role not in SITE_OWNER_ROLE_ORDER:
            raise KeyError(role)
        return getattr(self, role)


@dataclass(frozen=True)
class SiteComposition:
    """Immutable static projection for one Site across explicit Benchmarks."""

    site: str
    benchmark_compositions: tuple[SiteBenchmarkComposition, ...]
    provenance: tuple[str, ...] = ()
    source_package: str = SITE_COMPOSITION_SOURCE_PACKAGE
    source_package_version: str = WARP_TASKGEN_VERSION

    def __post_init__(self) -> None:
        site = _normalize_label(self.site)
        if not site:
            raise ValueError("Site Composition requires a Site name")
        compositions = tuple(self.benchmark_compositions)
        if not compositions:
            raise ValueError("Site Composition requires a Benchmark projection")
        if any(not isinstance(item, SiteBenchmarkComposition) for item in compositions):
            raise TypeError("Site Composition projections must be typed")
        benchmarks = [item.benchmark for item in compositions]
        if len(set(benchmarks)) != len(benchmarks):
            raise ValueError(f"duplicate Benchmark projection for Site {site!r}")
        object.__setattr__(self, "site", site)
        object.__setattr__(
            self,
            "benchmark_compositions",
            tuple(sorted(compositions, key=lambda item: item.benchmark)),
        )
        object.__setattr__(self, "provenance", _freeze_provenance(self.provenance))
        object.__setattr__(
            self,
            "source_package",
            _normalize_symbol(self.source_package, field="source_package"),
        )
        object.__setattr__(
            self,
            "source_package_version",
            _normalize_symbol(self.source_package_version, field="source_package_version"),
        )

    def benchmark(self, name: str) -> SiteBenchmarkComposition | None:
        canonical = DEFAULT_BENCHMARK_CATALOG.normalize(name)
        return next(
            (item for item in self.benchmark_compositions if item.benchmark == canonical),
            None,
        )


@dataclass(frozen=True)
class SiteCompositionUseCase:
    """Host-owned immutable requirements for one static check use case."""

    id: str
    scope: SiteCompositionScope
    required_owner_roles: tuple[SiteOwnerRole, ...]
    requires_carrier: bool = False
    requires_action_kind: bool = False

    def __post_init__(self) -> None:
        use_case = _normalize_label(self.id)
        if not use_case:
            raise ValueError("use case requires a semantic identity")
        scope = _normalize_symbol(self.scope, field="scope")
        if scope not in {
            "static_diagnostic",
            "warp_generation",
            "warp_execution",
            "warp_evaluation",
        }:
            raise ValueError("use case scope is invalid")
        required = tuple(self.required_owner_roles)
        if any(role not in SITE_OWNER_ROLE_ORDER for role in required):
            raise ValueError("use case references an unknown Site owner role")
        if len(set(required)) != len(required):
            raise ValueError("use case owner roles must be unique")
        object.__setattr__(self, "id", use_case)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "required_owner_roles", required)


@dataclass(frozen=True)
class SiteCompositionUseCaseCatalog:
    """Host-Owned immutable catalog of static Site Composition use cases."""

    entries: tuple[SiteCompositionUseCase, ...]

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        if any(not isinstance(entry, SiteCompositionUseCase) for entry in entries):
            raise TypeError("use-case catalog entries must be typed")
        ids = [entry.id for entry in entries]
        if len(ids) != len(set(ids)):
            raise ValueError("use-case catalog entries must have unique IDs")
        object.__setattr__(self, "entries", tuple(sorted(entries, key=lambda entry: entry.id)))

    @classmethod
    def default(cls) -> SiteCompositionUseCaseCatalog:
        return cls(
            entries=(
                SiteCompositionUseCase(
                    id="phase_1_generation",
                    scope="warp_generation",
                    required_owner_roles=(
                        "site_targeting",
                        "profile",
                        "editor_specification",
                        "action_cards",
                    ),
                ),
                SiteCompositionUseCase(
                    id="phase_2_generation",
                    scope="warp_generation",
                    required_owner_roles=(
                        "site_targeting",
                        "profile",
                        "editor_specification",
                        "regular_participant_writer",
                        "read_surface",
                        "readback",
                    ),
                ),
                SiteCompositionUseCase(
                    id="phase_2_feasibility",
                    scope="warp_generation",
                    required_owner_roles=(
                        "site_targeting",
                        "profile",
                        "editor_specification",
                        "regular_participant_writer",
                        "feasibility",
                        "read_surface",
                        "readback",
                    ),
                ),
                SiteCompositionUseCase(
                    id="phase_4_execution",
                    scope="warp_execution",
                    required_owner_roles=SITE_OWNER_ROLE_ORDER,
                ),
                SiteCompositionUseCase(
                    id="warp_evaluation",
                    scope="warp_evaluation",
                    required_owner_roles=(
                        "site_targeting",
                        "profile",
                        "final_state_evaluation",
                    ),
                ),
                SiteCompositionUseCase(
                    id="public_reply",
                    scope="static_diagnostic",
                    requires_carrier=True,
                    requires_action_kind=True,
                    required_owner_roles=(
                        "site_targeting",
                        "profile",
                        "editor_specification",
                        "regular_participant_writer",
                        "feasibility",
                        "read_surface",
                        "readback",
                        "action_cards",
                    ),
                ),
            )
        )

    def resolve(self, use_case: object) -> SiteCompositionUseCase | None:
        canonical = _normalize_label(use_case)
        return next((entry for entry in self.entries if entry.id == canonical), None)


@dataclass(frozen=True)
class SiteCompositionCheckRequest:
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
            raise ValueError("Site Composition check requires Site, Benchmark, and use case")
        object.__setattr__(self, "site", site)
        object.__setattr__(self, "benchmark", benchmark)
        object.__setattr__(self, "use_case", use_case)
        for name in ("carrier", "action_kind"):
            value = getattr(self, name)
            if value is None or not str(value).strip():
                object.__setattr__(self, name, None)
            else:
                object.__setattr__(
                    self,
                    name,
                    _normalize_symbol(value, field=name),
                )


@dataclass(frozen=True)
class SiteCompositionFinding:
    capability: str
    state: StaticCapabilityState
    outcome: SiteCompositionFindingOutcome
    code: str
    detail: str
    provenance: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        capability = _normalize_symbol(self.capability, field="capability")
        if self.state not in _CAPABILITY_STATES:
            raise ValueError("finding state is invalid")
        if self.outcome not in _FINDING_OUTCOMES:
            raise ValueError("finding outcome is invalid")
        expected_pass = self.state in {"supported", "not_applicable"}
        if (self.outcome == "pass") != expected_pass:
            raise ValueError("finding state and outcome contradict")
        code = _normalize_symbol(self.code, field="finding code")
        if (
            not isinstance(self.detail, str)
            or not self.detail.strip()
            or "\n" in self.detail
            or "\r" in self.detail
            or "://" in self.detail
            or any(term in self.detail.casefold() for term in _SENSITIVE_TERMS)
        ):
            raise ValueError("finding detail requires safe single-line text")
        object.__setattr__(self, "capability", capability)
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "provenance", _freeze_provenance(self.provenance))
        object.__setattr__(self, "dependencies", _freeze_provenance(self.dependencies))

    def to_dict(self) -> dict[str, object]:
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
class SiteCompositionCheckReport:
    site: str
    benchmark: str
    use_case: str
    static_status: SiteCompositionStatus
    site_composition_digest: SiteCompositionDigest | None
    findings: tuple[SiteCompositionFinding, ...]
    carrier: str | None = None
    action_kind: str | None = None
    source_package: str | None = None
    source_package_version: str | None = None
    source_provenance: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        site = _normalize_label(self.site)
        benchmark = _normalize_label(self.benchmark)
        use_case = _normalize_label(self.use_case)
        if (
            not site
            or not benchmark
            or not use_case
            or any(
                term in value.casefold()
                for value in (site, benchmark, use_case)
                for term in _SENSITIVE_TERMS
            )
        ):
            raise ValueError("Site Composition report identities must be semantic labels")
        object.__setattr__(self, "site", site)
        object.__setattr__(self, "benchmark", benchmark)
        object.__setattr__(self, "use_case", use_case)
        for name in ("carrier", "action_kind"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _normalize_symbol(value, field=name))
        for name in ("source_package", "source_package_version"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _normalize_symbol(value, field=name))
        object.__setattr__(self, "source_provenance", _freeze_provenance(self.source_provenance))
        if self.static_status not in _STATUSES:
            raise ValueError("Site Composition report has an invalid status")
        if self.static_status == "invalid":
            if self.site_composition_digest is not None:
                raise ValueError("invalid Site Composition reports cannot carry a digest")
        elif not isinstance(self.site_composition_digest, str) or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", self.site_composition_digest
        ):
            raise ValueError("complete or incomplete reports require sha256:<hex> digest")
        findings = tuple(self.findings)
        if any(not isinstance(item, SiteCompositionFinding) for item in findings):
            raise TypeError("Site Composition findings must be typed")
        capabilities = [item.capability for item in findings]
        if len(capabilities) != len(set(capabilities)):
            raise ValueError("Site Composition findings require unique capabilities")
        by_capability = {item.capability: item for item in findings}
        if "registration" not in by_capability or "static_closure" not in by_capability:
            raise ValueError(
                "Site Composition reports require registration and static_closure findings"
            )
        registration = by_capability["registration"]
        static_closure = by_capability["static_closure"]
        if self.static_status == "invalid":
            if registration.state != "unsupported" or registration.outcome != "failure":
                raise ValueError("invalid reports require a failed registration finding")
            if static_closure.state != "unsupported" or static_closure.outcome != "failure":
                raise ValueError("invalid reports require a failed static_closure finding")
        elif self.static_status == "complete":
            if registration.state != "supported" or registration.outcome != "pass":
                raise ValueError("complete reports require a passing registration finding")
            if static_closure.state != "supported" or static_closure.outcome != "pass":
                raise ValueError("complete reports require a passing static_closure finding")
            if any(item.outcome == "failure" for item in findings):
                raise ValueError("complete reports cannot contain failed findings")
        else:
            if registration.state != "supported" or registration.outcome != "pass":
                raise ValueError("incomplete reports require a passing registration finding")
            if static_closure.state not in {"missing", "unsupported"}:
                raise ValueError("incomplete reports require a failed static_closure finding")
        object.__setattr__(self, "findings", findings)

    def finding(self, capability: str) -> SiteCompositionFinding:
        for finding in self.findings:
            if finding.capability == capability:
                return finding
        raise KeyError(capability)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "warp-site-composition-check-v1",
            "scope": "static_site_composition_only",
            "site": self.site,
            "benchmark": self.benchmark,
            "use_case": self.use_case,
            "carrier": self.carrier,
            "action_kind": self.action_kind,
            "source_package": self.source_package,
            "source_package_version": self.source_package_version,
            "source_provenance": list(self.source_provenance),
            "static_status": self.static_status,
            "site_composition_digest": self.site_composition_digest,
            "readiness_status": "blocked",
            "readiness_blockers": ["active_policy_not_checked", "live_evidence_not_checked"],
            "active_policy_checked": False,
            "live_evidence_checked": False,
            "findings": [item.to_dict() for item in self.findings],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


__all__ = [
    "SITE_COMPOSITION_CONTRACT_VERSION",
    "SITE_COMPOSITION_SOURCE_PACKAGE",
    "SITE_OWNER_CONTRACT_VERSION",
    "SITE_OWNER_ROLE_ORDER",
    "SiteBenchmarkComposition",
    "SiteComposition",
    "SiteCompositionCheckReport",
    "SiteCompositionCheckRequest",
    "SiteCompositionDigest",
    "SiteCompositionFinding",
    "SiteCompositionFindingOutcome",
    "SiteCompositionScope",
    "SiteCompositionStatus",
    "SiteCompositionUseCase",
    "SiteCompositionUseCaseCatalog",
    "SiteOwnerDeclaration",
    "SiteOwnerRole",
    "StaticCapabilityState",
]
