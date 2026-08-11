"""Immutable Benchmark Contract types used by WARP Taskgen.

This module owns the contract value objects and their validation.  The public
``benchmark_capabilities`` module remains the compatibility facade that
assembles the default catalog and exposes metadata helpers.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

Phase4Mode = Literal["worldsim_v5", "comparison_runner", "unsupported"]
ComparisonOutcomeMode = Literal["attack_success", "resistance", "capability", "unknown"]
BenchmarkCapability = Literal[
    "phase_1_generation",
    "phase_2_generation",
    "phase_2_feasibility",
    "phase_4_execution",
    "warp_evaluation",
    "comparison_ingestion",
]
EvaluatorAuthority = Literal[
    "canonical_vendor_task_id",
    "warp_local_task_idless",
    "comparison_runner",
]

_CAPABILITY_NAMES = frozenset(
    {
        "phase_1_generation",
        "phase_2_generation",
        "phase_2_feasibility",
        "phase_4_execution",
        "warp_evaluation",
        "comparison_ingestion",
    }
)
_EVALUATOR_AUTHORITIES = frozenset(
    {
        "canonical_vendor_task_id",
        "warp_local_task_idless",
        "comparison_runner",
    }
)
_UNSET = object()


def _normalize_label(value: object) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    return "_".join(part for part in text.replace("-", " ").split() if part)


@dataclass(frozen=True, init=False)
class BenchmarkCapabilities:
    """Immutable capabilities and evaluator authority for one Benchmark.

    The old phase flags below are compatibility properties derived from the
    immutable capability set.  New callers should use :meth:`supports` or
    :meth:`require` with one of the exact :class:`BenchmarkCapability` values.
    """

    canonical_name: str
    default_runner: str
    supported_runners: tuple[str, ...]
    capabilities: frozenset[BenchmarkCapability] = frozenset()
    phase_4_mode: Phase4Mode = "unsupported"
    comparison_outcome_mode: ComparisonOutcomeMode | None = None
    requires_host_api_preflight: bool = False
    evaluator_authorities: tuple[EvaluatorAuthority, ...] = ()

    def __init__(
        self,
        canonical_name: str,
        default_runner: str,
        supported_runners: tuple[str, ...],
        phase_1_supported: bool | object = _UNSET,
        phase_2_supported: bool | object = _UNSET,
        phase_2_feasibility_supported: bool | object = _UNSET,
        phase_4_mode: Phase4Mode = "unsupported",
        comparison_outcome_mode: ComparisonOutcomeMode | None = None,
        requires_host_api_preflight: bool = False,
        *,
        capabilities: Iterable[BenchmarkCapability] | object = _UNSET,
        evaluator_authorities: Iterable[EvaluatorAuthority] | object = _UNSET,
    ) -> None:
        """Build a contract, translating the previous constructor for one cycle.

        The positional parameters through ``requires_host_api_preflight`` are
        the historical public signature. New registrations declare exact
        ``capabilities`` and ``evaluator_authorities`` instead. Explicit legacy
        flags must agree with an exact declaration, except that the former
        comparison Phase 1 default is accepted and discarded. Legacy flags
        never grant additional admission implicitly.
        """

        legacy_flags = {
            "phase_1_supported": phase_1_supported,
            "phase_2_supported": phase_2_supported,
            "phase_2_feasibility_supported": phase_2_feasibility_supported,
        }
        if capabilities is _UNSET:
            capabilities, default_authorities = _capabilities_from_legacy_fields(
                phase_1_supported=phase_1_supported,
                phase_2_supported=phase_2_supported,
                phase_2_feasibility_supported=phase_2_feasibility_supported,
                phase_4_mode=phase_4_mode,
            )
            if evaluator_authorities is _UNSET:
                evaluator_authorities = default_authorities
        elif evaluator_authorities is _UNSET:
            evaluator_authorities = ()

        object.__setattr__(self, "canonical_name", canonical_name)
        object.__setattr__(self, "default_runner", default_runner)
        object.__setattr__(self, "supported_runners", supported_runners)
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "phase_4_mode", phase_4_mode)
        object.__setattr__(self, "comparison_outcome_mode", comparison_outcome_mode)
        object.__setattr__(self, "requires_host_api_preflight", requires_host_api_preflight)
        object.__setattr__(self, "evaluator_authorities", evaluator_authorities)
        self.__post_init__()

        derived_flags = {
            "phase_1_supported": self.phase_1_supported,
            "phase_2_supported": self.phase_2_supported,
            "phase_2_feasibility_supported": self.phase_2_feasibility_supported,
        }
        for name, value in legacy_flags.items():
            if value is _UNSET:
                continue
            if not isinstance(value, bool):
                raise ValueError(f"legacy benchmark flag {name} must be boolean")
            if phase_4_mode == "comparison_runner" and name == "phase_1_supported":
                # Historical comparison entries declared Phase 1 by default.
                # Accept that old constructor shape without retaining the
                # unsafe WARP-generation grant in the authoritative ledger.
                continue
            if value != derived_flags[name]:
                raise ValueError(f"legacy benchmark flag {name} conflicts with capabilities")

    def __post_init__(self) -> None:
        if not isinstance(self.canonical_name, str):
            raise ValueError("benchmark capability requires a string canonical name")
        canonical_name = _normalize_label(self.canonical_name)
        if not canonical_name:
            raise ValueError("benchmark capability requires a canonical name")
        try:
            supported_runners = tuple(self.supported_runners)
        except TypeError as exc:
            raise ValueError("benchmark capability requires iterable supported runners") from exc
        if not supported_runners or any(
            not isinstance(runner, str) or not runner.strip() for runner in supported_runners
        ):
            raise ValueError("benchmark capability requires supported runners")
        if len(set(supported_runners)) != len(supported_runners):
            raise ValueError("benchmark capability supported runners must be unique")
        if not isinstance(self.default_runner, str) or not self.default_runner.strip():
            raise ValueError("benchmark capability requires a string default runner")
        if self.default_runner not in supported_runners:
            raise ValueError("benchmark default runner must be listed in supported runners")
        try:
            capabilities = frozenset(self.capabilities)
        except TypeError as exc:
            raise ValueError("benchmark capabilities must be iterable") from exc
        if not capabilities.issubset(_CAPABILITY_NAMES):
            unknown = sorted(capabilities - _CAPABILITY_NAMES)
            raise ValueError(f"unknown benchmark capabilities: {unknown}")
        try:
            authorities = tuple(self.evaluator_authorities)
        except TypeError as exc:
            raise ValueError("benchmark evaluator authorities must be iterable") from exc
        if any(not isinstance(authority, str) for authority in authorities):
            raise ValueError(f"unknown evaluator authorities: {authorities!r}")
        if not set(authorities).issubset(_EVALUATOR_AUTHORITIES):
            raise ValueError(f"unknown evaluator authorities: {authorities!r}")
        if len(set(authorities)) != len(authorities):
            raise ValueError("benchmark evaluator authorities must be unique")
        if self.phase_4_mode not in {"worldsim_v5", "comparison_runner", "unsupported"}:
            raise ValueError(f"unknown Phase 4 mode {self.phase_4_mode!r}")
        if self.comparison_outcome_mode not in {
            None,
            "attack_success",
            "resistance",
            "capability",
            "unknown",
        }:
            raise ValueError(f"unknown comparison outcome mode {self.comparison_outcome_mode!r}")
        if not isinstance(self.requires_host_api_preflight, bool):
            raise ValueError("benchmark host API preflight declaration must be boolean")
        if "phase_2_feasibility" in capabilities and "phase_2_generation" not in capabilities:
            raise ValueError("Phase 2 feasibility requires phase 2 generation")
        phase_4_execution = "phase_4_execution" in capabilities
        if phase_4_execution != (self.phase_4_mode == "worldsim_v5"):
            raise ValueError("phase_4_execution capability must match phase_4_mode='worldsim_v5'")
        if phase_4_execution and "warp_evaluation" not in capabilities:
            raise ValueError("Phase 4 execution requires WARP evaluation")
        has_warp_capability = bool(capabilities & (_CAPABILITY_NAMES - {"comparison_ingestion"}))
        has_comparison_ingestion = "comparison_ingestion" in capabilities
        if has_warp_capability and has_comparison_ingestion:
            raise ValueError("WARP capabilities cannot coexist with comparison_ingestion")
        if has_comparison_ingestion:
            if authorities != ("comparison_runner",):
                raise ValueError(
                    "comparison_ingestion entries must use only comparison_runner authority"
                )
        if "warp_evaluation" in capabilities:
            if "comparison_runner" in authorities:
                raise ValueError("WARP evaluation entries cannot use comparison_runner authority")
            if not set(authorities) & {
                "canonical_vendor_task_id",
                "warp_local_task_idless",
            }:
                raise ValueError("WARP evaluation entries require evaluator authority")
        if self.comparison_outcome_mode is not None and not has_comparison_ingestion:
            raise ValueError("comparison_outcome_mode requires the comparison_ingestion capability")
        object.__setattr__(self, "canonical_name", canonical_name)
        object.__setattr__(self, "supported_runners", supported_runners)
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "evaluator_authorities", authorities)

    @property
    def phase_1_supported(self) -> bool:
        return self.supports("phase_1_generation")

    @property
    def phase_2_supported(self) -> bool:
        return self.supports("phase_2_generation")

    @property
    def phase_2_feasibility_supported(self) -> bool:
        return self.supports("phase_2_feasibility")

    @property
    def phase_4_supported(self) -> bool:
        return self.supports("phase_4_execution")

    @property
    def warp_phase_admission(self) -> tuple[str, ...]:
        return tuple(
            capability
            for capability in (
                "phase_1_generation",
                "phase_2_generation",
                "phase_2_feasibility",
                "phase_4_execution",
            )
            if capability in self.capabilities
        )

    @property
    def is_comparison_only(self) -> bool:
        return self.capabilities == frozenset({"comparison_ingestion"})

    @property
    def comparison_only_ingestion_supported(self) -> bool:
        return self.supports("comparison_ingestion")

    def supports(self, capability: str) -> bool:
        """Return whether an exact named capability is explicitly admitted."""

        if capability not in _CAPABILITY_NAMES:
            raise ValueError(f"unknown benchmark capability {capability!r}")
        return capability in self.capabilities

    def require(self, capability: str) -> BenchmarkCapabilities:
        """Require an exact capability and return this immutable contract."""

        if not self.supports(capability):
            raise ValueError(
                f"benchmark {self.canonical_name!r} does not support capability {capability!r}"
            )
        return self

    def supports_runner(self, runner: object) -> bool:
        return _normalize_label(runner) in {
            _normalize_label(value) for value in self.supported_runners
        }

    def evaluator_authority_for_task(self, *, task_id: object | None) -> EvaluatorAuthority:
        """Resolve vendor versus local authority from task-ID presence."""

        authority: EvaluatorAuthority = (
            "canonical_vendor_task_id" if task_id is not None else "warp_local_task_idless"
        )
        if authority not in self.evaluator_authorities:
            raise ValueError(
                f"benchmark {self.canonical_name!r} has no evaluator authority for {authority}"
            )
        return authority


def _capabilities_from_legacy_fields(
    *,
    phase_1_supported: bool | object,
    phase_2_supported: bool | object,
    phase_2_feasibility_supported: bool | object,
    phase_4_mode: Phase4Mode,
) -> tuple[frozenset[BenchmarkCapability], tuple[EvaluatorAuthority, ...]]:
    """Translate the pre-ledger constructor without preserving unsafe grants."""

    values = {
        "phase_1_generation": True if phase_1_supported is _UNSET else phase_1_supported,
        "phase_2_generation": False if phase_2_supported is _UNSET else phase_2_supported,
        "phase_2_feasibility": False
        if phase_2_feasibility_supported is _UNSET
        else phase_2_feasibility_supported,
    }
    if any(not isinstance(value, bool) for value in values.values()):
        raise ValueError("legacy benchmark phase flags must be boolean")
    capabilities: set[BenchmarkCapability] = {
        capability
        for capability, enabled in values.items()
        if enabled  # type: ignore[misc]
    }
    authorities: tuple[EvaluatorAuthority, ...] = ()
    if phase_4_mode == "worldsim_v5":
        capabilities.update({"phase_4_execution", "warp_evaluation"})
        authorities = ("canonical_vendor_task_id", "warp_local_task_idless")
    elif phase_4_mode == "comparison_runner":
        capabilities = {"comparison_ingestion"}
        authorities = ("comparison_runner",)
    return frozenset(capabilities), authorities


@dataclass(frozen=True)
class BenchmarkCatalog:
    """Immutable benchmark identity and capability catalog."""

    entries: Mapping[str, BenchmarkCapabilities]
    aliases: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.entries, Mapping):
            raise ValueError("benchmark catalog entries must be a mapping")
        if not isinstance(self.aliases, Mapping):
            raise ValueError("benchmark catalog aliases must be a mapping")
        normalized_entries: dict[str, BenchmarkCapabilities] = {}
        for raw_name, capabilities in self.entries.items():
            if not isinstance(raw_name, str):
                raise ValueError("benchmark catalog entry names must be strings")
            if not isinstance(capabilities, BenchmarkCapabilities):
                raise ValueError("benchmark catalog entries must be BenchmarkCapabilities")
            name = _normalize_label(raw_name)
            if not name or name != capabilities.canonical_name:
                raise ValueError("benchmark catalog key must match canonical capability name")
            if name in normalized_entries:
                raise ValueError(f"duplicate benchmark catalog entry {name!r}")
            normalized_entries[name] = capabilities
        normalized_aliases: dict[str, str] = {}
        for raw_alias, raw_target in self.aliases.items():
            if not isinstance(raw_alias, str) or not isinstance(raw_target, str):
                raise ValueError("benchmark catalog aliases must map strings to strings")
            alias = _normalize_label(raw_alias)
            target = _normalize_label(raw_target)
            if not alias or target not in normalized_entries:
                raise ValueError(f"benchmark alias {raw_alias!r} points to unknown benchmark")
            existing = normalized_aliases.get(alias)
            if existing is not None and existing != target:
                raise ValueError(f"conflicting benchmark alias {raw_alias!r}")
            if alias in normalized_entries and alias != target:
                raise ValueError(f"benchmark alias {raw_alias!r} conflicts with a canonical entry")
            normalized_aliases[alias] = target
        object.__setattr__(self, "entries", MappingProxyType(normalized_entries))
        object.__setattr__(self, "aliases", MappingProxyType(normalized_aliases))

    def normalize(self, value: object) -> str:
        normalized = _normalize_label(value)
        return self.aliases.get(normalized, normalized)

    def resolve(self, value: object) -> BenchmarkCapabilities:
        normalized = self.normalize(value)
        capabilities = self.entries.get(normalized)
        if capabilities is None:
            raise ValueError(f"unknown benchmark {value!r}; available={sorted(self.entries)}")
        return capabilities

    def require(self, benchmark: object, capability: str) -> BenchmarkCapabilities:
        """Resolve ``benchmark`` and require one exact capability."""

        return self.resolve(benchmark).require(capability)

    def available(self) -> tuple[str, ...]:
        return tuple(sorted(self.entries))

    def infer(self, values: Iterable[object]) -> str | None:
        inferred: set[str] = set()
        normalized_seen: set[str] = set()
        for value in values:
            normalized = self.normalize(value)
            if not normalized:
                continue
            normalized_seen.add(normalized)
            inferred.add(self.resolve(normalized).canonical_name)
        if not inferred:
            return None
        if len(inferred) == 1:
            return next(iter(inferred))
        raise ValueError(f"mixed benchmark metadata: {sorted(normalized_seen)}")


__all__ = [
    "BenchmarkCapabilities",
    "BenchmarkCapability",
    "BenchmarkCatalog",
    "ComparisonOutcomeMode",
    "EvaluatorAuthority",
    "Phase4Mode",
]
