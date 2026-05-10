from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from worldsim.adversarial_actions.capability_contracts import control_action_kinds_from_task
from worldsim.adversarial_actions.catalog import get_action_spec
from worldsim.adversarial_actions.shared_compiler import _MUTATION_ACTIONS

ACTION_POLICIES: tuple[str, ...] = (
    "default",
    "semantic_only",
    "mutation_when_available",
    "mutation_only_when_available",
    "tier1_only",
    "tier1_unaligned_control",
    "tier2_pilot",
    "tier2_unaligned_control",
    "tier3_pilot",
    "tier3_unaligned_control",
    "wasp_tier1_only",
    "wasp_tier2_pilot",
    "wasp_tier3_pilot",
)

def canonical_action_policy(policy: str | None) -> str:
    if not policy:
        return "default"
    normalized = _normalized_tier_policy(policy)
    if normalized is not None:
        return normalized[1]
    return policy

def _keep_semantic_actions_only(contract: dict[str, Any]) -> None:
    options = contract.get("adversarial_action_options")
    if not isinstance(options, list):
        return
    semantic_options = [
        option
        for option in options
        if isinstance(option, Mapping) and option.get("kind") == "answer_opposite_binary_label"
    ]
    if semantic_options:
        contract["adversarial_action_options"] = semantic_options
        contract["adversarial_action_preference"] = {
            "kind": "answer_opposite_binary_label",
            "policy": "semantic_only",
            "reason": (
                "Minimum-validation policy: isolate host-compiled binary semantic "
                "hijack and suppress navigation or mutation actions."
            ),
        }
        return
    contract["adversarial_action_options"] = []
    contract["adversarial_action_preference"] = {
        "policy": "semantic_only",
        "reason": (
            "Minimum-validation policy found no compatible binary semantic action "
            "for this benign reward."
        ),
    }

def _prefer_first_mutation_action(contract: dict[str, Any]) -> None:
    options = contract.get("adversarial_action_options")
    if not isinstance(options, list):
        return
    mutation_index = next(
        (
            index
            for index, option in enumerate(options)
            if isinstance(option, Mapping) and option.get("kind") in _MUTATION_ACTIONS
        ),
        None,
    )
    if mutation_index is None:
        return
    preferred = options[mutation_index]
    contract["adversarial_action_options"] = [
        preferred,
        *options[:mutation_index],
        *options[mutation_index + 1 :],
    ]
    contract["adversarial_action_preference"] = {
        "kind": preferred["kind"],
        "policy": "mutation_when_available",
        "reason": (
            "Opt-in mutation pilot: prefer a concrete browser-side POST action "
            "when this exposure contract supports one."
        ),
    }

def _keep_mutation_actions_when_available(contract: dict[str, Any], *, policy: str) -> None:
    options = contract.get("adversarial_action_options")
    if not isinstance(options, list):
        return
    mutation_options = [
        option
        for option in options
        if isinstance(option, Mapping) and option.get("kind") in _MUTATION_ACTIONS
    ]
    if not mutation_options:
        return
    contract["adversarial_action_options"] = mutation_options
    contract["adversarial_action_preference"] = {
        "kind": mutation_options[0]["kind"],
        "policy": policy,
        "reason": (
            "Strict mutation pilot: this contract supports a concrete browser-side "
            "POST action, so semantic and navigation alternatives are suppressed."
        ),
    }

def _keep_tier_actions(contract: dict[str, Any], *, tier: int, policy: str) -> None:
    options = contract.get("adversarial_action_options")
    if not isinstance(options, list):
        return
    tier_options = [
        option
        for option in options
        if isinstance(option, Mapping) and _action_tier(str(option.get("kind") or "")) == tier
    ]
    if tier_options:
        contract["adversarial_action_options"] = tier_options
        contract["adversarial_action_preference"] = {
            "kind": tier_options[0]["kind"],
            "policy": policy,
            "reason": (
                f"Tier {tier} pilot: restrict this contract to host-owned action "
                "options in the selected impact tier."
            ),
        }
        return
    contract["adversarial_action_options"] = []
    contract["adversarial_action_preference"] = {
        "policy": policy,
        "reason": (
            f"Tier {tier} pilot requested, but this contract has no host-ready "
            "action option in that impact tier."
        ),
    }

def _keep_control_tier_actions(
    contract: dict[str, Any],
    *,
    benign_task: Mapping[str, Any] | None,
    tier: int,
    policy: str,
) -> None:
    control_kinds = set(control_action_kinds_from_task(benign_task))
    options = contract.get("adversarial_action_options")
    if not control_kinds or not isinstance(options, list):
        contract["adversarial_action_options"] = []
        contract["adversarial_action_preference"] = {
            "policy": policy,
            "reason": (
                "Negative-control policy requested, but this task has no "
                "host-owned control_action_kinds."
            ),
        }
        return
    control_options = [
        option
        for option in options
        if isinstance(option, Mapping)
        and str(option.get("kind") or "") in control_kinds
        and _action_tier(str(option.get("kind") or "")) == tier
    ]
    for option in control_options:
        option["policy"] = policy
        option["control_condition"] = "unaligned_action_control"
    if control_options:
        contract["adversarial_action_options"] = control_options
        contract["adversarial_action_preference"] = {
            "kind": control_options[0]["kind"],
            "policy": policy,
            "reason": (
                "Negative-control policy: expose a host-owned Tier 2 action "
                "that is intentionally not a compatible benign capability."
            ),
        }
        return
    contract["adversarial_action_options"] = []
    contract["adversarial_action_preference"] = {
        "policy": policy,
        "reason": (
            f"Negative-control policy requested, but this contract has no Tier {tier} "
            "editor action matching control_action_kinds."
        ),
    }

def _action_tier(kind: str) -> int | None:
    spec = get_action_spec(kind)
    return spec.impact_tier if spec is not None else None

def _unaligned_control_policy_tier(policy: str) -> int | None:
    mapping = {
        "tier1_unaligned_control": 1,
        "tier2_unaligned_control": 2,
        "tier3_unaligned_control": 3,
    }
    return mapping.get(policy)

def _normalized_tier_policy(policy: str) -> tuple[int, str] | None:
    aliases = {
        "tier1_only": (1, "tier1_only"),
        "tier2_pilot": (2, "tier2_pilot"),
        "tier3_pilot": (3, "tier3_pilot"),
        "wasp_tier1_only": (1, "tier1_only"),
        "wasp_tier2_pilot": (2, "tier2_pilot"),
        "wasp_tier3_pilot": (3, "tier3_pilot"),
    }
    return aliases.get(policy)
