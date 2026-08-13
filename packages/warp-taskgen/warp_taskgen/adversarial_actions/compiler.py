from __future__ import annotations

from warp_taskgen.adversarial_actions.options import (
    allowed_action_options,
    annotate_exposure_contracts_with_action_policy,
    annotate_exposure_contracts_with_actions,
)
from warp_taskgen.adversarial_actions.policies import ACTION_POLICIES, canonical_action_policy
from warp_taskgen.adversarial_actions.public_mutation_compiler import (
    compile_public_benign_mutation_evaluator,
    compile_public_mutation_network_reward,
)
from warp_taskgen.adversarial_actions.reward_compiler import (
    compile_adversarial_final_state_check,
    compile_adversarial_reward,
)
from warp_taskgen.adversarial_actions.reward_introspection import (
    action_metadata_for_task,
    reward_signal_for_task,
)
from warp_taskgen.adversarial_actions.shared_compiler import ACTION_KINDS, ACTION_SIGNAL_BY_KIND

__all__ = [
    "ACTION_KINDS",
    "ACTION_POLICIES",
    "ACTION_SIGNAL_BY_KIND",
    "action_metadata_for_task",
    "allowed_action_options",
    "annotate_exposure_contracts_with_action_policy",
    "annotate_exposure_contracts_with_actions",
    "canonical_action_policy",
    "compile_adversarial_final_state_check",
    "compile_adversarial_reward",
    "compile_public_benign_mutation_evaluator",
    "compile_public_mutation_network_reward",
    "reward_signal_for_task",
]
