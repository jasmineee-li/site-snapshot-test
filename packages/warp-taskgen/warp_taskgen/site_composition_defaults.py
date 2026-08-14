"""Built-in GitLab and Reddit projections for static Site diagnostics."""

from __future__ import annotations

from warp_taskgen.adversarial_actions.capability_adapters import (
    capability_adapters_for_profile,
)
from warp_taskgen.editors import EDITOR_REGISTRY
from warp_taskgen.editors._registry import iter_specs
from warp_taskgen.phase_2.phase_2c.webarena_policy import WebArenaFeasibilityPolicy
from warp_taskgen.rewards.final_state_gitlab_adapter import GitLabFinalStateEvaluator
from warp_taskgen.rewards.final_state_reddit_adapter import RedditFinalStateEvaluator
from warp_taskgen.seeding.site_contracts import SeedSiteRegistration
from warp_taskgen.site_composition_contracts import (
    CapabilityReference,
    SiteBenchmarkBinding,
    SiteDefinition,
)
from warp_taskgen.sites import GitLabSite, RedditSite


def default_site_definitions() -> tuple[SiteDefinition, ...]:
    """Project current GitLab/Reddit owners into diagnostic definitions."""

    definitions: list[SiteDefinition] = []
    for site, adapter, policy, evaluator in (
        (
            "gitlab",
            GitLabSite(),
            WebArenaFeasibilityPolicy(site="gitlab", auth_path="/-/profile"),
            GitLabFinalStateEvaluator(),
        ),
        (
            "reddit",
            RedditSite(),
            WebArenaFeasibilityPolicy(site="reddit"),
            RedditFinalStateEvaluator(),
        ),
    ):
        benchmark = "webarena_verified"
        editor_specs = tuple(iter_specs(site=site, benchmark=benchmark))
        editor_factory = EDITOR_REGISTRY.get((benchmark, site))
        seed = (
            CapabilityReference(
                "supported",
                SeedSiteRegistration(benchmark, site, editor_factory),
                ("seeding.site_contracts",),
            )
            if callable(editor_factory)
            else CapabilityReference("missing", None, ("editors.EDITOR_REGISTRY",))
        )
        cards = capability_adapters_for_profile(
            "semantic_minval",
            benchmark_family=benchmark,
            sites=(site,),
        )
        binding = SiteBenchmarkBinding(
            benchmark=benchmark,
            targeting=CapabilityReference("supported", adapter, (f"sites.{site}",)),
            profile=CapabilityReference("supported", adapter, (f"sites.{site}_profile",)),
            editor_specs=CapabilityReference("supported", editor_specs, ("editors._registry",)),
            seed=seed,
            feasibility=CapabilityReference(
                "supported", policy, ("phase_2.phase_2c.webarena_policy",)
            ),
            read_surface=CapabilityReference("supported", adapter, (f"sites.{site}",)),
            readback=CapabilityReference("supported", adapter, (f"sites.{site}",)),
            final_state=CapabilityReference(
                "supported", evaluator, ("rewards.final_state_catalog",)
            ),
            action_cards=CapabilityReference(
                "supported", cards, ("adversarial_actions.capability_adapters",)
            ),
        )
        definitions.append(
            SiteDefinition(site=site, bindings=(binding,), provenance=(f"sites.{site}",))
        )
    return tuple(definitions)


__all__ = ["default_site_definitions"]
