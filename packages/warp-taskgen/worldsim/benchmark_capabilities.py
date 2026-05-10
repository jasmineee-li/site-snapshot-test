"""Benchmark capability registry.

Keeps benchmark normalization and phase-support decisions explicit so generic
pipeline code does not silently assume WebArena semantics.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

Phase4Mode = Literal["worldsim_v5", "comparison_runner", "unsupported"]
ComparisonOutcomeMode = Literal["attack_success", "resistance", "capability", "unknown"]


@dataclass(frozen=True)
class BenchmarkCapabilities:
    canonical_name: str
    default_runner: str
    supported_runners: tuple[str, ...]
    phase_1_supported: bool = True
    phase_2_supported: bool = False
    phase_2_feasibility_supported: bool = False
    phase_4_mode: Phase4Mode = "unsupported"
    comparison_outcome_mode: ComparisonOutcomeMode | None = None
    requires_host_api_preflight: bool = False


_ALIASES: dict[str, str] = {
    "webarena_verified": "webarena_verified",
    "webarena-verified": "webarena_verified",
    "webarena verified": "webarena_verified",
    "web arena verified": "webarena_verified",
    "wasp": "wasp",
    "stwebagentbench": "stwebagentbench",
    "st-webagentbench": "stwebagentbench",
    "st webagentbench": "stwebagentbench",
    "doomarena": "doomarena",
    "doom arena": "doomarena",
}

_CAPABILITIES: dict[str, BenchmarkCapabilities] = {
    "webarena_verified": BenchmarkCapabilities(
        canonical_name="webarena_verified",
        default_runner="browser_use",
        supported_runners=("browser_use", "agentlab"),
        phase_1_supported=True,
        phase_2_supported=True,
        phase_2_feasibility_supported=True,
        phase_4_mode="worldsim_v5",
        requires_host_api_preflight=True,
    ),
    "wasp": BenchmarkCapabilities(
        canonical_name="wasp",
        default_runner="agentlab",
        supported_runners=("agentlab",),
        phase_1_supported=True,
        phase_2_supported=False,
        phase_2_feasibility_supported=False,
        phase_4_mode="comparison_runner",
        comparison_outcome_mode="resistance",
    ),
    "stwebagentbench": BenchmarkCapabilities(
        canonical_name="stwebagentbench",
        default_runner="agentlab",
        supported_runners=("agentlab",),
        phase_1_supported=True,
        phase_2_supported=False,
        phase_2_feasibility_supported=False,
        phase_4_mode="comparison_runner",
        comparison_outcome_mode="capability",
    ),
    "doomarena": BenchmarkCapabilities(
        canonical_name="doomarena",
        default_runner="agentlab",
        supported_runners=("agentlab",),
        phase_1_supported=True,
        phase_2_supported=False,
        phase_2_feasibility_supported=False,
        phase_4_mode="comparison_runner",
        comparison_outcome_mode="attack_success",
    ),
}


def normalize_benchmark_name(value: object) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    return _ALIASES.get(text, text.replace("-", "_").replace(" ", "_"))


def get_benchmark_capabilities(value: object) -> BenchmarkCapabilities:
    normalized = normalize_benchmark_name(value)
    if normalized in _CAPABILITIES:
        return _CAPABILITIES[normalized]
    raise ValueError(f"unknown benchmark {value!r}; available={sorted(_CAPABILITIES)}")


def available_benchmark_capabilities() -> list[str]:
    return sorted(_CAPABILITIES)


def infer_benchmark_name(values: Iterable[object]) -> str | None:
    inferred: set[str] = set()
    raw_seen: set[str] = set()
    for value in values:
        normalized = normalize_benchmark_name(value)
        if not normalized:
            continue
        raw_seen.add(normalized)
        inferred.add(get_benchmark_capabilities(normalized).canonical_name)
    if not inferred:
        return None
    if len(inferred) == 1:
        return next(iter(inferred))
    raise ValueError(f"mixed benchmark metadata: {sorted(raw_seen)}")


def infer_instances_config_benchmark(raw: dict[str, object]) -> str | None:
    values: list[object] = [
        raw.get("benchmark"),
        raw.get("benchmark_name"),
        raw.get("benchmark_adapter"),
    ]
    raw_instances = raw.get("instances")
    if isinstance(raw_instances, list):
        for instance in raw_instances:
            if not isinstance(instance, dict):
                continue
            values.extend(
                (
                    instance.get("benchmark"),
                    instance.get("benchmark_name"),
                    instance.get("benchmark_adapter"),
                )
            )
    return infer_benchmark_name(values)


__all__ = [
    "BenchmarkCapabilities",
    "available_benchmark_capabilities",
    "get_benchmark_capabilities",
    "infer_benchmark_name",
    "infer_instances_config_benchmark",
    "normalize_benchmark_name",
]
