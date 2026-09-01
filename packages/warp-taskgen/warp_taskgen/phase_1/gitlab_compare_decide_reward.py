"""Reward envelope and scoring for the GitLab comparison feature."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from warp_taskgen.phase_1.gitlab_compare_decide import (
    GitLabAttemptBinding,
    GitLabComparisonWorld,
    _world_from_task_or_world,
    expected_gitlab_compare_decide_response,
)


def build_gitlab_compare_decide_reward(
    task_or_world: GitLabComparisonWorld | Mapping[str, Any],
    *,
    binding: GitLabAttemptBinding | None = None,
) -> dict[str, Any]:
    """Build the feature-local exact structured reward envelope."""
    world = _world_from_task_or_world(task_or_world)
    return {
        "benchmark": world.benchmark,
        "eval": [
            {
                "evaluator": "AgentResponseEvaluator",
                "ordered": True,
                "expected": expected_gitlab_compare_decide_response(world, binding=binding),
            }
        ],
    }


def grade_gitlab_compare_decide(
    task_or_world: GitLabComparisonWorld | Mapping[str, Any],
    agent_response: Any,
    *,
    binding: GitLabAttemptBinding | None = None,
) -> tuple[bool, str]:
    """Grade through the existing reward dispatcher and exact evaluator."""
    from warp_taskgen.rewards.dispatcher import run_reward_function

    world = _world_from_task_or_world(task_or_world)
    response = agent_response
    if isinstance(response, Mapping):

        class _Result:
            def __init__(self, final_result: Mapping[str, Any]) -> None:
                self.final_result = dict(final_result)
                self.is_done = True

        response = _Result(response)
    return run_reward_function(
        build_gitlab_compare_decide_reward(world, binding=binding),
        {"benchmark": world.benchmark, "site_name": "gitlab"},
        response,
    )


def materialize_gitlab_compare_decide_reward(
    task_or_world: GitLabComparisonWorld | Mapping[str, Any],
    binding: GitLabAttemptBinding,
) -> dict[str, Any]:
    """Build the exact reward after a fresh attempt supplies physical IDs."""
    return build_gitlab_compare_decide_reward(task_or_world, binding=binding)


__all__ = [
    "build_gitlab_compare_decide_reward",
    "grade_gitlab_compare_decide",
    "materialize_gitlab_compare_decide_reward",
]
