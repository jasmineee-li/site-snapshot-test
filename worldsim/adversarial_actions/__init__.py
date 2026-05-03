"""Host-owned adversarial action contracts and reward compilation."""

from worldsim.adversarial_actions.capability_contracts import (
    ACTION_CAPABILITY_CONTRACTS,
    ActionCapabilityContract,
    action_kind_compatible_with_task_card,
    action_kinds_for_capability_family,
    capability_family_from_task_card,
    compatibility_reason_for_task_card,
    describe_action_capability,
    get_action_capability_contract,
)
from worldsim.adversarial_actions.catalog import (
    ACTION_SPECS,
    ActionSpec,
    describe_action_kind,
    get_action_spec,
)
from worldsim.adversarial_actions.compiler import (
    ACTION_KINDS,
    ACTION_POLICIES,
    ACTION_SIGNAL_BY_KIND,
    action_metadata_for_task,
    allowed_action_options,
    annotate_exposure_contracts_with_action_policy,
    annotate_exposure_contracts_with_actions,
    canonical_action_policy,
    compile_adversarial_final_state_check,
    compile_adversarial_reward,
    reward_signal_for_task,
)
from worldsim.adversarial_actions.readiness import build_action_readiness_artifacts
from worldsim.adversarial_actions.tier3 import (
    TIER3_ADAPTER_SPECS,
    TIER3_MATURITY_LEVELS,
    action_kinds_for_exposure_contracts,
)

__all__ = [
    "ACTION_CAPABILITY_CONTRACTS",
    "ACTION_KINDS",
    "ACTION_POLICIES",
    "ACTION_SIGNAL_BY_KIND",
    "ACTION_SPECS",
    "TIER3_ADAPTER_SPECS",
    "TIER3_MATURITY_LEVELS",
    "ActionCapabilityContract",
    "ActionSpec",
    "action_kind_compatible_with_task_card",
    "action_kinds_for_capability_family",
    "action_kinds_for_exposure_contracts",
    "action_metadata_for_task",
    "allowed_action_options",
    "annotate_exposure_contracts_with_action_policy",
    "annotate_exposure_contracts_with_actions",
    "build_action_readiness_artifacts",
    "canonical_action_policy",
    "capability_family_from_task_card",
    "compatibility_reason_for_task_card",
    "compile_adversarial_final_state_check",
    "compile_adversarial_reward",
    "describe_action_capability",
    "describe_action_kind",
    "get_action_capability_contract",
    "get_action_spec",
    "reward_signal_for_task",
]
