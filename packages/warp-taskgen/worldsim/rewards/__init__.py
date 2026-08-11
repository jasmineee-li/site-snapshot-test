"""Stable public facade for reward evaluation."""

from __future__ import annotations

from worldsim.rewards.action_attempt import extract_network_action_attempt
from worldsim.rewards.dispatcher import run_reward_function
from worldsim.rewards.final_state_catalog import (
    FinalStateEvaluationRequest,
    FinalStateEvaluator,
    FinalStateEvaluatorCatalog,
    default_final_state_evaluator_catalog,
)

__all__ = [
    "FinalStateEvaluationRequest",
    "FinalStateEvaluator",
    "FinalStateEvaluatorCatalog",
    "default_final_state_evaluator_catalog",
    "extract_network_action_attempt",
    "run_reward_function",
]
