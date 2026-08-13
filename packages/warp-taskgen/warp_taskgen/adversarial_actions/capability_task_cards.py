"""Host-compiled Phase 1 task-card profiles for adversarial action pilots.

These profiles keep workflow diversity in deterministic host code. The model
still writes natural task prose, but the host chooses which browser capability
family, route family, and action family a pilot is allowed to exercise.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from warp_taskgen.adversarial_actions.action_targets import ActionTargetContract
from warp_taskgen.adversarial_actions.capability_adapters import (
    CapabilityTaskAdapter,
    available_capability_adapter_profiles,
    capability_adapters_for_profile,
)
from warp_taskgen.adversarial_actions.capability_contracts import (
    get_action_capability_contract,
)
from warp_taskgen.adversarial_actions.precondition_slots import PreconditionSlot
from warp_taskgen.adversarial_actions.scenario_templates import ScenarioTemplate

CAPABILITY_TASK_CARD_SCHEMA_VERSION = "worldsim-task-card-plan-v1"


@dataclass(frozen=True)
class CapabilityTaskCardBlueprint:
    """A deterministic, host-owned blueprint for one Phase 1 task card."""

    id: str
    site: str
    action_kind: str
    route_ids: tuple[str, ...]
    archetype_id: str
    benign_task_family_id: str
    instruction_must_match_any_regex: tuple[str, ...] = ()
    instruction_must_match_all_regex: tuple[str, ...] = ()
    instruction_must_not_match_any_regex: tuple[str, ...] = ()
    forbidden_instruction_substrings: tuple[str, ...] = (
        "title",
        "attacker.invalid",
        "worldsim",
        "fixture",
    )
    requires_benign_action_evidence: bool = False
    benign_action_evidence: dict[str, Any] | None = None
    benign_reward_shape: str | None = None
    task_archetype: dict[str, Any] | None = None
    precondition_slot: PreconditionSlot | dict[str, Any] | None = None
    scenario_template: ScenarioTemplate | dict[str, Any] | None = None
    action_target_contract: ActionTargetContract | dict[str, Any] | None = None

    @classmethod
    def from_adapter(cls, adapter: CapabilityTaskAdapter) -> CapabilityTaskCardBlueprint:
        """Build a task-card blueprint from a benchmark capability adapter."""
        return cls(
            id=adapter.id,
            site=adapter.site,
            action_kind=adapter.action_kind,
            route_ids=adapter.route_ids,
            archetype_id=adapter.archetype_id,
            benign_task_family_id=adapter.benign_task_family_id,
            instruction_must_match_any_regex=adapter.instruction_must_match_any_regex,
            instruction_must_match_all_regex=adapter.instruction_must_match_all_regex,
            instruction_must_not_match_any_regex=adapter.instruction_must_not_match_any_regex,
            forbidden_instruction_substrings=adapter.forbidden_instruction_substrings,
            requires_benign_action_evidence=adapter.requires_benign_action_evidence,
            benign_action_evidence=adapter.benign_action_evidence,
            benign_reward_shape=adapter.benign_reward_shape,
            task_archetype=adapter.task_archetype,
            precondition_slot=adapter.precondition_slot,
            scenario_template=adapter.scenario_template,
            action_target_contract=adapter.action_target_contract,
        )

    def to_task_card(self) -> dict[str, Any]:
        """Compile the blueprint into a Phase 1 task-card plan entry."""
        contract = get_action_capability_contract(self.action_kind)
        if contract is None:
            raise ValueError(f"unknown action kind for capability task card: {self.action_kind}")
        if self.site not in contract.compatible_sites:
            raise ValueError(
                f"task card {self.id!r} site {self.site!r} is incompatible with "
                f"action {self.action_kind!r}"
            )
        if self.benign_task_family_id not in contract.benign_task_family_ids:
            raise ValueError(
                f"task card {self.id!r} benign task family {self.benign_task_family_id!r} "
                f"is incompatible with action {self.action_kind!r}"
            )
        card: dict[str, Any] = {
            "id": self.id,
            "site": self.site,
            "route_ids": list(self.route_ids),
            "archetype_id": self.archetype_id,
            "capability_family": contract.capability_family,
            "compatible_action_kinds": [self.action_kind],
            "benign_task_family_id": self.benign_task_family_id,
        }
        card["benign_reward_shape"] = self.benign_reward_shape or contract.benign_reward_shape
        if self.instruction_must_match_any_regex:
            card["instruction_must_match_any_regex"] = list(self.instruction_must_match_any_regex)
        if self.instruction_must_match_all_regex:
            card["instruction_must_match_all_regex"] = list(self.instruction_must_match_all_regex)
        if self.instruction_must_not_match_any_regex:
            card["instruction_must_not_match_any_regex"] = list(
                self.instruction_must_not_match_any_regex
            )
        if self.forbidden_instruction_substrings:
            card["forbidden_instruction_substrings"] = list(self.forbidden_instruction_substrings)
        if self.requires_benign_action_evidence:
            card["requires_benign_action_evidence"] = True
        if self.benign_action_evidence is not None:
            card["benign_action_evidence"] = dict(self.benign_action_evidence)
        if self.task_archetype is not None:
            card["task_archetype"] = copy.deepcopy(self.task_archetype)
        if self.precondition_slot is not None:
            card["precondition_slot"] = (
                self.precondition_slot.to_dict()
                if isinstance(self.precondition_slot, PreconditionSlot)
                else copy.deepcopy(self.precondition_slot)
            )
        if self.scenario_template is not None:
            card["scenario_template"] = (
                self.scenario_template.to_dict()
                if isinstance(self.scenario_template, ScenarioTemplate)
                else copy.deepcopy(self.scenario_template)
            )
        if self.action_target_contract is not None:
            card["action_target_contract"] = (
                self.action_target_contract.to_dict()
                if isinstance(self.action_target_contract, ActionTargetContract)
                else copy.deepcopy(self.action_target_contract)
            )
        return card


_PROFILE_DESCRIPTIONS: dict[str, str] = {
    "semantic_minval": (
        "Minimum validation cards for host-compiled semantic actions on no-title "
        "GitLab/Reddit WASP body carriers. Use with --phase-2a-action-policy semantic_only."
    ),
    "tier2_mutation_pilot": (
        "Pilot cards for bounded public mutation actions on no-title GitLab/Reddit "
        "WASP body carriers. Use with --phase-2a-action-policy tier2_pilot so "
        "semantic/navigation fallbacks fail closed."
    ),
    "tier2_gitlab_public_comment_pilot": (
        "Pilot card for GitLab issue-description carriers that target a same-issue "
        "public comment through a host-owned action target contract. Use with "
        "--phase-2a-action-policy tier2_pilot."
    ),
    "tier2_pure_action_paper": (
        "Paper-facing Tier 2 cards for IPI-bearing public-action tasks. The "
        "profile uses host_action_only benign utility; Reddit/Postmill "
        "submit_comment rows remain fail-closed unless Phase 2c proves "
        "seed-specific painted comment visibility. Use with "
        "--phase-2a-action-policy tier2_pilot."
    ),
    "tier3_repository_pilot": (
        "Tier 3 repository/content maintenance pilot cards. Use only with "
        "--phase-2a-action-policy tier3_pilot and verified disposable fixture configuration."
    ),
    "tier3_delete_resource_pilot": (
        "WASP-style Tier 3 destructive-resource pilot cards. Benign tasks must "
        "exercise harmless UGC acknowledgement while host-owned payload actions "
        "target a separately verified disposable project."
    ),
}


def available_capability_task_card_profiles() -> tuple[str, ...]:
    """Return supported compiled Phase 1 task-card profile names."""
    return available_capability_adapter_profiles()


def compile_capability_task_card_plan(
    profile: str,
    *,
    sites: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Compile a named capability profile into a task-card plan.

    The returned object has the same schema as a JSON task-card plan and should
    be validated by Phase 1 before use.
    """
    profile_name = profile.strip()
    if profile_name not in available_capability_task_card_profiles():
        allowed = ", ".join(available_capability_task_card_profiles())
        raise ValueError(
            f"unknown task capability profile {profile_name!r}; choose one of: {allowed}"
        )
    site_filter = _normalize_sites(sites)
    all_blueprints = tuple(
        CapabilityTaskCardBlueprint.from_adapter(adapter)
        for adapter in capability_adapters_for_profile(profile_name)
    )
    supported_sites = {blueprint.site for blueprint in all_blueprints}
    if site_filter is not None:
        missing_sites = sorted(site_filter - supported_sites)
        if missing_sites:
            raise ValueError(
                f"task capability profile {profile_name!r} has no cards for requested "
                f"site(s): {', '.join(missing_sites)}"
            )
    cards = [
        blueprint.to_task_card()
        for blueprint in all_blueprints
        if site_filter is None or blueprint.site in site_filter
    ]
    if not cards:
        requested = ", ".join(sorted(site_filter or ()))
        raise ValueError(
            f"task capability profile {profile_name!r} has no cards for requested sites: "
            f"{requested or '<none>'}"
        )
    return {
        "schema_version": CAPABILITY_TASK_CARD_SCHEMA_VERSION,
        "source": "compiled_action_capability_profile",
        "task_capability_profile": profile_name,
        "description": _PROFILE_DESCRIPTIONS[profile_name],
        "task_cards": cards,
    }


def _normalize_sites(sites: Iterable[str] | None) -> set[str] | None:
    if sites is None:
        return None
    normalized = {site.strip() for site in sites if isinstance(site, str) and site.strip()}
    return normalized or None
