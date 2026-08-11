"""Immutable composition for local final-state reward evaluators."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

from worldsim.benchmark_capabilities import (
    get_benchmark_capabilities,
    normalize_benchmark_name,
)
from worldsim.rewards.evidence import EvidencePolicy

_LOCAL_FINAL_STATE_BENCHMARK = "webarena_verified"


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return value


def thaw_final_state_value(value: Any) -> Any:
    """Restore immutable request evidence to ordinary evaluator containers."""

    if isinstance(value, Mapping):
        return {key: thaw_final_state_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_final_state_value(item) for item in value]
    if isinstance(value, frozenset):
        return {thaw_final_state_value(item) for item in value}
    return value


@dataclass(frozen=True)
class FinalStateEvaluationRequest:
    """Validated, immutable inputs for one local final-state evaluation."""

    benchmark: str
    site: str
    action_kind: str
    witness: str
    network_expected: Mapping[str, Any]
    state_probe: Mapping[str, Any] | None
    evidence_policy: EvidencePolicy
    network_trace: tuple[Mapping[str, Any], ...]
    instance: Mapping[str, Any]
    initial_events: tuple[Mapping[str, Any], ...]
    initial_message: str

    def __post_init__(self) -> None:
        if not isinstance(self.benchmark, str) or not self.benchmark.strip():
            raise ValueError("final-state request requires a Benchmark")
        benchmark = get_benchmark_capabilities(self.benchmark).canonical_name
        if benchmark != _LOCAL_FINAL_STATE_BENCHMARK:
            raise ValueError(
                f"local final-state evaluation supports only WebArena Verified; got {benchmark!r}"
            )
        if not isinstance(self.site, str) or not self.site.strip():
            raise ValueError("final-state request requires a Site")
        site = self.site.strip().lower()
        if not isinstance(self.action_kind, str) or not self.action_kind.strip():
            raise ValueError("final-state request requires an action kind")
        if not isinstance(self.witness, str) or not self.witness.strip():
            raise ValueError("final-state request requires a witness")
        if not isinstance(self.network_expected, Mapping):
            raise ValueError("final-state request requires network-event evidence")
        if self.state_probe is not None and not isinstance(self.state_probe, Mapping):
            raise ValueError("final-state request state probe must be an object")
        if not isinstance(self.instance, Mapping):
            raise ValueError("final-state request instance must be an object")
        if not isinstance(self.initial_message, str):
            raise ValueError("final-state request message must be text")
        trace = tuple(_freeze(event) for event in self.network_trace)
        events = tuple(_freeze(event) for event in self.initial_events)
        object.__setattr__(self, "benchmark", benchmark)
        object.__setattr__(self, "site", site)
        object.__setattr__(self, "action_kind", self.action_kind.strip())
        object.__setattr__(self, "witness", self.witness.strip())
        object.__setattr__(self, "network_expected", _freeze(self.network_expected))
        object.__setattr__(
            self,
            "state_probe",
            _freeze(self.state_probe) if self.state_probe is not None else None,
        )
        object.__setattr__(self, "network_trace", trace)
        object.__setattr__(self, "instance", _freeze(self.instance))
        object.__setattr__(self, "initial_events", events)


class FinalStateEvaluator(Protocol):
    benchmark: str
    site: str

    def evaluate(self, request: FinalStateEvaluationRequest) -> tuple[bool, str]: ...


@dataclass(frozen=True)
class FinalStateEvaluatorCatalog:
    """Explicit immutable catalog of WARP-local final-state evaluators."""

    _evaluators: Mapping[tuple[str, str], FinalStateEvaluator]

    def __post_init__(self) -> None:
        if not isinstance(self._evaluators, Mapping):
            raise ValueError("final-state evaluator catalog requires a mapping")
        normalized: dict[tuple[str, str], FinalStateEvaluator] = {}
        for raw_key, evaluator in self._evaluators.items():
            if not isinstance(raw_key, tuple) or len(raw_key) != 2:
                raise ValueError("final-state evaluator catalog keys must be (benchmark, site)")
            if any(not isinstance(value, str) or not value.strip() for value in raw_key):
                raise ValueError("final-state evaluator catalog keys must contain text")
            benchmark = get_benchmark_capabilities(raw_key[0]).canonical_name
            if benchmark != _LOCAL_FINAL_STATE_BENCHMARK:
                raise ValueError(
                    "local final-state evaluator bindings support only WebArena Verified; "
                    f"got {benchmark!r}"
                )
            site = raw_key[1].strip().lower()
            declared_benchmark_raw = getattr(evaluator, "benchmark", "")
            declared_site_raw = getattr(evaluator, "site", "")
            if (
                not isinstance(declared_benchmark_raw, str)
                or not declared_benchmark_raw.strip()
                or not isinstance(declared_site_raw, str)
                or not declared_site_raw.strip()
            ):
                raise ValueError("final-state evaluator identity must contain text")
            declared_benchmark = get_benchmark_capabilities(declared_benchmark_raw).canonical_name
            declared_site = declared_site_raw.strip().lower()
            if (benchmark, site) != (declared_benchmark, declared_site):
                raise ValueError("final-state evaluator key does not match evaluator identity")
            key = (benchmark, site)
            if key in normalized:
                raise ValueError(f"duplicate final-state evaluator binding {key!r}")
            if not callable(getattr(evaluator, "evaluate", None)):
                raise ValueError(f"final-state evaluator {key!r} has no evaluate method")
            normalized[key] = evaluator
        object.__setattr__(self, "_evaluators", MappingProxyType(normalized))

    @classmethod
    def from_evaluators(
        cls,
        evaluators: Iterable[FinalStateEvaluator],
    ) -> FinalStateEvaluatorCatalog:
        bindings: dict[tuple[str, str], FinalStateEvaluator] = {}
        for evaluator in evaluators:
            benchmark_raw = getattr(evaluator, "benchmark", "")
            site_raw = getattr(evaluator, "site", "")
            if (
                not isinstance(benchmark_raw, str)
                or not benchmark_raw.strip()
                or not isinstance(site_raw, str)
                or not site_raw.strip()
            ):
                raise ValueError("final-state evaluator identity must contain text")
            benchmark = get_benchmark_capabilities(benchmark_raw).canonical_name
            if benchmark != _LOCAL_FINAL_STATE_BENCHMARK:
                raise ValueError(
                    "local final-state evaluator bindings support only WebArena Verified; "
                    f"got {benchmark!r}"
                )
            site = site_raw.strip().lower()
            key = (benchmark, site)
            if key in bindings:
                raise ValueError(f"duplicate final-state evaluator binding {key!r}")
            bindings[key] = evaluator
        return cls(bindings)

    @property
    def bindings(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(self._evaluators))

    def evaluate(self, request: FinalStateEvaluationRequest) -> tuple[bool, str]:
        if not isinstance(request, FinalStateEvaluationRequest):
            return False, "FinalStateEvaluator received an invalid request"
        key = (normalize_benchmark_name(request.benchmark), request.site)
        evaluator = self._evaluators.get(key)
        if evaluator is None:
            return False, (
                f"FinalStateEvaluator unsupported site {request.site!r} "
                f"for benchmark {request.benchmark!r}"
            )
        try:
            result = evaluator.evaluate(request)
        except Exception as exc:
            return False, (
                f"FinalStateEvaluator adapter failed for {key!r}: {exc.__class__.__name__}: {exc}"
            )
        if (
            not isinstance(result, tuple)
            or len(result) != 2
            or not isinstance(result[0], bool)
            or not isinstance(result[1], str)
        ):
            return False, f"FinalStateEvaluator adapter returned an invalid result for {key!r}"
        return result


def default_final_state_evaluator_catalog() -> FinalStateEvaluatorCatalog:
    """Build the explicit default local evaluator composition."""

    from worldsim.rewards.final_state_gitlab_adapter import GitLabFinalStateEvaluator
    from worldsim.rewards.final_state_reddit_adapter import RedditFinalStateEvaluator

    return FinalStateEvaluatorCatalog.from_evaluators(
        (GitLabFinalStateEvaluator(), RedditFinalStateEvaluator())
    )


__all__ = [
    "FinalStateEvaluationRequest",
    "FinalStateEvaluator",
    "FinalStateEvaluatorCatalog",
    "default_final_state_evaluator_catalog",
    "thaw_final_state_value",
]
