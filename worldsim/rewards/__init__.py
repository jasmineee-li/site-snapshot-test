"""Reward function dispatcher facade.

Behavior lives in feature-local modules under :mod:`worldsim.rewards`; this
package preserves the historic ``worldsim.rewards`` public import path.
"""

from __future__ import annotations

from worldsim.rewards.action_attempt import extract_network_action_attempt
from worldsim.rewards.dispatcher import run_reward_function
from worldsim.rewards.vendor_webarena import (
    WEBARENA_EVAL_MODULE,
    WEBARENA_EVAL_PYTHON_ENV,
    _apply_webarena_vendor_shims,
    _build_webarena_environment_payload,
    _default_eval_python,
    _run_webarena_verified_eval,
    _run_webarena_verified_subprocess,
    subprocess,
)

__all__ = [
    "WEBARENA_EVAL_MODULE",
    "WEBARENA_EVAL_PYTHON_ENV",
    "_apply_webarena_vendor_shims",
    "_build_webarena_environment_payload",
    "_default_eval_python",
    "_run_webarena_verified_eval",
    "_run_webarena_verified_subprocess",
    "extract_network_action_attempt",
    "run_reward_function",
    "subprocess",
]
