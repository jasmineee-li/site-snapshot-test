"""Compatibility facade for canonical Phase 2c feasibility policies."""

from __future__ import annotations

from warp_taskgen.benchmark_capabilities import normalize_benchmark_name
from warp_taskgen.phase_2.phase_2c.policy import (
    FeasibilityPolicy,
    FeasibilityPolicyCatalog,
    PolicyKey,
    PreflightClassification,
    ProbeTarget,
    SourceDataDecision,
    default_feasibility_policy_catalog,
    resolve_feasibility_policy,
    task_probe_targets,
)

# The historical policy registry remains available only to callers that have
# not migrated. It is initialized when this compatibility module is imported,
# preserving the old get/register/clear lifecycle; active Phase 2c callers use
# the canonical immutable catalog and never consult this mapping.
_LEGACY_POLICIES: dict[PolicyKey, FeasibilityPolicy] = {}


def register_feasibility_policy(policy: FeasibilityPolicy) -> None:
    from warp_taskgen.phase_2.phase_2c.policy import _policy_key

    _LEGACY_POLICIES[_policy_key(policy)] = policy


def get_feasibility_policy(benchmark: str, site: str) -> FeasibilityPolicy | None:
    if not isinstance(site, str):
        return None
    return _LEGACY_POLICIES.get((normalize_benchmark_name(benchmark), site.strip().lower()))


def clear_feasibility_policy_registry() -> None:
    _LEGACY_POLICIES.clear()


def _register_initial_defaults() -> None:
    for policy in default_feasibility_policy_catalog().policies.values():
        register_feasibility_policy(policy)


_register_initial_defaults()

__all__ = [
    "FeasibilityPolicy",
    "FeasibilityPolicyCatalog",
    "PolicyKey",
    "PreflightClassification",
    "ProbeTarget",
    "SourceDataDecision",
    "clear_feasibility_policy_registry",
    "default_feasibility_policy_catalog",
    "get_feasibility_policy",
    "register_feasibility_policy",
    "resolve_feasibility_policy",
    "task_probe_targets",
]
