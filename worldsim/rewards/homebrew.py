from __future__ import annotations

from typing import Any

from worldsim.rewards.agent_response import _build_agent_response, _eval_agent_response
from worldsim.rewards.final_state import _eval_final_state
from worldsim.rewards.network_event import _eval_network_event

_FINAL_STATE_REWARD_TYPES = frozenset({"FinalStateEvaluator", "final_state"})


def _is_network_event_evaluator_name(name: Any) -> bool:
    return isinstance(name, str) and name in {"NetworkEventEvaluator", "network_event"}


def _run_homebrew_eval(
    reward: dict[str, Any],
    instance: dict[str, Any],
    agent_result: Any | None,
    network_trace: list[dict] | None,
) -> tuple[bool, str]:
    """Evaluate novel, non-canonical rewards emitted by WorldSim compilers.

    Canonical WebArena tasks fail closed through the vendor adapter. This path
    exists for generated tasks whose reward specs are already in WorldSim's
    supported deterministic subset.
    """
    eval_configs = reward["eval"]

    # Build once so composite rewards evaluate a single normalized final answer.
    agent_response = _build_agent_response(eval_configs, agent_result)

    all_passed = True
    messages: list[str] = []

    for config in eval_configs:
        evaluator_type = config.get("evaluator") or config.get("type") or ""

        if evaluator_type == "AgentResponseEvaluator":
            passed, msg = _eval_agent_response(config, agent_response)
        elif _is_network_event_evaluator_name(evaluator_type):
            passed, msg = _eval_network_event(config, network_trace, instance)
        elif evaluator_type in _FINAL_STATE_REWARD_TYPES:
            passed, msg = _eval_final_state(config, network_trace, instance)
        else:
            passed, msg = False, f"Unknown evaluator type: {evaluator_type}"

        if not passed:
            all_passed = False
        messages.append(f"[{evaluator_type}] {'PASS' if passed else 'FAIL'}: {msg}")

    combined = "; ".join(messages)
    return all_passed, combined
