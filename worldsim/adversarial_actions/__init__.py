"""Host-owned adversarial action contracts and reward compilation."""

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
    allowed_action_options,
    annotate_exposure_contracts_with_action_policy,
    annotate_exposure_contracts_with_actions,
    compile_adversarial_final_state_check,
    compile_adversarial_reward,
    reward_signal_for_task,
)

__all__ = [
    "ACTION_KINDS",
    "ACTION_POLICIES",
    "ACTION_SIGNAL_BY_KIND",
    "ACTION_SPECS",
    "ActionSpec",
    "allowed_action_options",
    "annotate_exposure_contracts_with_action_policy",
    "annotate_exposure_contracts_with_actions",
    "compile_adversarial_final_state_check",
    "compile_adversarial_reward",
    "describe_action_kind",
    "get_action_spec",
    "reward_signal_for_task",
]
