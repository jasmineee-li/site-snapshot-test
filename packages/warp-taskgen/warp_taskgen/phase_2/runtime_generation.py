"""Narrow opt-in seam for composition-owned Phase 2 generation behavior."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from warp_taskgen.runtime_composition import RuntimeComposition


@dataclass(frozen=True)
class PreparedPhase2Shard:
    """Feature-owned planning inputs in the ordinary Phase 2 record shapes."""

    tasks: list[dict[str, Any]]
    benign_target_resources: dict[str, dict[str, Any]]
    exposure_contracts: dict[str, dict[str, Any]]
    eligibility_drops: list[dict[str, Any]]
    plans: list[dict[str, Any]]


@runtime_checkable
class Phase2Generation(Protocol):
    """Behavior a named runtime composition must supply to own Phase 2 rows."""

    def applies_to(self, *, benchmark: object, site: object) -> bool: ...

    def prepare_shard(
        self,
        tasks: Sequence[Mapping[str, Any]],
        runtime_composition: RuntimeComposition,
    ) -> PreparedPhase2Shard: ...

    def validate_and_enrich_plans(
        self,
        plans: Sequence[dict[str, Any]],
        benign_tasks: Sequence[dict[str, Any]],
        *,
        exposure_contracts: Mapping[str, Mapping[str, Any]],
        runtime_composition: RuntimeComposition,
    ) -> tuple[list[dict[str, Any]], list[str]]: ...

    def validate_plan(
        self,
        plan: object,
        *,
        index: int,
        benign_by_id: Mapping[str, Mapping[str, Any]],
        exposure_contracts: Mapping[str, Mapping[str, Any]],
        runtime_composition: RuntimeComposition,
    ) -> str | None: ...

    def validate_materialized_task(
        self,
        task: Mapping[str, Any],
        *,
        benign_task: Mapping[str, Any],
        runtime_composition: RuntimeComposition,
    ) -> str | None: ...


def generation_for_runtime(
    runtime_composition: RuntimeComposition | None,
    *,
    benchmark: object,
    site: object,
) -> Phase2Generation | None:
    """Return the scoped feature implementation for this exact shard."""

    if runtime_composition is None:
        return None
    feature = runtime_composition.phase_2_generation
    if feature is None:
        return None
    if not isinstance(feature, Phase2Generation):
        raise TypeError("runtime composition phase_2_generation is not a valid implementation")
    return feature if feature.applies_to(benchmark=benchmark, site=site) else None


__all__ = ["Phase2Generation", "PreparedPhase2Shard", "generation_for_runtime"]
