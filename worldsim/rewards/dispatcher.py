from __future__ import annotations

from typing import Any

from worldsim.rewards.agent_response import _build_agent_response, _eval_agent_response
from worldsim.rewards.final_state import _eval_final_state
from worldsim.rewards.homebrew import _run_homebrew_eval
from worldsim.rewards.network_event import _eval_network_event
from worldsim.rewards.vendor_webarena import _run_webarena_verified_eval

_NETWORK_EVENT_REWARD_TYPES = frozenset({"NetworkEventEvaluator", "network_event"})
_FINAL_STATE_REWARD_TYPES = frozenset({"FinalStateEvaluator", "final_state"})
_REMOVED_REWARD_TYPES = frozenset({"db_query_match"})


def run_reward_function(
    reward: dict[str, Any],
    instance: dict[str, Any],
    agent_result: Any | None = None,
    network_trace: list[dict] | None = None,
) -> tuple[bool, str]:
    """Run one reward spec against a benchmark instance.

    Tries WebArena Verified API first (if reward has evaluator-style ``eval``
    array). Falls back to homebrew evaluators for WorldSim reward specs.

    Args:
        reward: Reward spec. For WebArena Verified tasks, contains ``eval``
            (array of evaluator configs) and ``task_id``. For custom checks,
            contains ``type`` + type-specific fields.
        instance: Running benchmark instance dict with ``site_url`` etc.
        agent_result: The agent's ``AgentResult`` from Browser Use. Provides
            ``final_result`` for retrieve tasks.
        network_trace: HAR-format network events captured during the agent run.
            Required for tasks with ``NetworkEventEvaluator`` configs.

    Returns:
        ``(passed, message)`` tuple.
    """
    # WebArena Verified evaluation path (canonical tasks with task_id)
    if "eval" in reward and isinstance(reward["eval"], list):
        if reward.get("task_id") is not None:
            return _run_webarena_verified_eval(reward, instance, agent_result, network_trace)
        # new_task novel benigns omit task_id, use homebrew evaluator directly
        return _run_homebrew_eval(reward, instance, agent_result, network_trace)

    # Custom checker path (legacy / extension)
    eval_type = reward.get("type") or reward.get("evaluator")
    if eval_type is None:
        return False, "Reward spec has neither 'eval' array nor 'type'/'evaluator' field"

    if eval_type in _NETWORK_EVENT_REWARD_TYPES:
        return _eval_network_event(reward, network_trace, instance)
    if eval_type in _FINAL_STATE_REWARD_TYPES:
        return _eval_final_state(reward, network_trace, instance)
    if eval_type == "AgentResponseEvaluator":
        agent_response = _build_agent_response([reward], agent_result)
        return _eval_agent_response(reward, agent_response)
    if eval_type in _REMOVED_REWARD_TYPES:
        raise NotImplementedError(
            f"Reward type {eval_type!r} is a removed legacy evaluator. "
            "Regenerate the task with NetworkEventEvaluator, FinalStateEvaluator, "
            "or AgentResponseEvaluator evidence."
        )
    raise NotImplementedError(f"Reward type {eval_type!r} not supported in worldsim.rewards.")
