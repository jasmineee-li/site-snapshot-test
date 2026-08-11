"""Public Benchmark Contract compatibility facade and default catalog.

Immutable contract value objects live in :mod:`worldsim.benchmark_contracts`;
this module retains the historical import path and assembles WARP's default
Benchmark catalog.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from worldsim.benchmark_contracts import (
    BenchmarkCapabilities,
    BenchmarkCapability,
    BenchmarkCatalog,
    ComparisonOutcomeMode,
    EvaluatorAuthority,
    Phase4Mode,
)

_BENCHMARK_METADATA_KEYS = ("benchmark", "benchmark_name", "benchmark_adapter")

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

_WARP_CAPABILITIES = frozenset(
    {
        "phase_1_generation",
        "phase_2_generation",
        "phase_2_feasibility",
        "phase_4_execution",
        "warp_evaluation",
    }
)
_COMPARISON_CAPABILITIES = frozenset({"comparison_ingestion"})

_CAPABILITIES: dict[str, BenchmarkCapabilities] = {
    "webarena_verified": BenchmarkCapabilities(
        canonical_name="webarena_verified",
        default_runner="browser_use",
        supported_runners=("browser_use", "agentlab"),
        capabilities=_WARP_CAPABILITIES,
        phase_4_mode="worldsim_v5",
        requires_host_api_preflight=True,
        evaluator_authorities=("canonical_vendor_task_id", "warp_local_task_idless"),
    ),
    "wasp": BenchmarkCapabilities(
        canonical_name="wasp",
        default_runner="agentlab",
        supported_runners=("agentlab",),
        capabilities=_COMPARISON_CAPABILITIES,
        phase_4_mode="comparison_runner",
        comparison_outcome_mode="resistance",
        evaluator_authorities=("comparison_runner",),
    ),
    "stwebagentbench": BenchmarkCapabilities(
        canonical_name="stwebagentbench",
        default_runner="agentlab",
        supported_runners=("agentlab",),
        capabilities=_COMPARISON_CAPABILITIES,
        phase_4_mode="comparison_runner",
        comparison_outcome_mode="capability",
        evaluator_authorities=("comparison_runner",),
    ),
    "doomarena": BenchmarkCapabilities(
        canonical_name="doomarena",
        default_runner="agentlab",
        supported_runners=("agentlab",),
        capabilities=_COMPARISON_CAPABILITIES,
        phase_4_mode="comparison_runner",
        comparison_outcome_mode="attack_success",
        evaluator_authorities=("comparison_runner",),
    ),
}

DEFAULT_BENCHMARK_CATALOG = BenchmarkCatalog(_CAPABILITIES, _ALIASES)


def normalize_benchmark_name(value: object) -> str:
    """Normalize a benchmark alias without accepting unknown metadata."""

    return DEFAULT_BENCHMARK_CATALOG.normalize(value)


def get_benchmark_capabilities(value: object) -> BenchmarkCapabilities:
    return DEFAULT_BENCHMARK_CATALOG.resolve(value)


def available_benchmark_capabilities() -> list[str]:
    return list(DEFAULT_BENCHMARK_CATALOG.available())


def infer_benchmark_name(values: Iterable[object]) -> str | None:
    return DEFAULT_BENCHMARK_CATALOG.infer(values)


def resolve_evaluator_authority(
    benchmark: object,
    *,
    task_id: object | None,
) -> EvaluatorAuthority:
    return get_benchmark_capabilities(benchmark).evaluator_authority_for_task(task_id=task_id)


def infer_benchmark_from_metadata(
    sources: Iterable[Mapping[str, object]],
) -> str | None:
    """Infer explicit Benchmark identity from metadata-bearing mappings.

    ``None`` means no source declared Benchmark metadata. Once a key is
    present, even a blank value is an explicit malformed declaration and must
    fail closed rather than silently selecting the historical default.
    """

    values: list[object] = []
    metadata_declared = False
    for source in sources:
        if not isinstance(source, Mapping):
            raise ValueError("benchmark metadata sources must be mappings")
        for key in _BENCHMARK_METADATA_KEYS:
            if key not in source or source[key] is None:
                continue
            metadata_declared = True
            values.append(source[key])
    if not metadata_declared:
        return None
    benchmark = DEFAULT_BENCHMARK_CATALOG.infer(values)
    if benchmark is None:
        raise ValueError("benchmark metadata is empty")
    return benchmark


def resolve_evaluator_authority_from_metadata(
    sources: Iterable[Mapping[str, object]],
    *,
    task_id: object | None,
) -> EvaluatorAuthority | None:
    """Resolve explicit evaluation admission without guessing legacy metadata."""

    benchmark = infer_benchmark_from_metadata(sources)
    if benchmark is None:
        return None
    capabilities = DEFAULT_BENCHMARK_CATALOG.require(benchmark, "warp_evaluation")
    return capabilities.evaluator_authority_for_task(task_id=task_id)


def infer_instances_config_benchmark(raw: dict[str, object]) -> str | None:
    values: list[object] = [raw.get(key) for key in _BENCHMARK_METADATA_KEYS]
    raw_instances = raw.get("instances")
    if isinstance(raw_instances, list):
        for instance in raw_instances:
            if not isinstance(instance, dict):
                continue
            values.extend(instance.get(key) for key in _BENCHMARK_METADATA_KEYS)
    return infer_benchmark_name(values)


__all__ = [
    "DEFAULT_BENCHMARK_CATALOG",
    "BenchmarkCapabilities",
    "BenchmarkCapability",
    "BenchmarkCatalog",
    "ComparisonOutcomeMode",
    "EvaluatorAuthority",
    "Phase4Mode",
    "available_benchmark_capabilities",
    "get_benchmark_capabilities",
    "infer_benchmark_from_metadata",
    "infer_benchmark_name",
    "infer_instances_config_benchmark",
    "normalize_benchmark_name",
    "resolve_evaluator_authority",
    "resolve_evaluator_authority_from_metadata",
]
