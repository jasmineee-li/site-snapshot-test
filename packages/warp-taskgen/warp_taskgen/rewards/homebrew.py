from __future__ import annotations

from typing import Any

from warp_taskgen.rewards.agent_response import _build_agent_response, _eval_agent_response
from warp_taskgen.rewards.final_state import _eval_final_state
from warp_taskgen.rewards.final_state_catalog import FinalStateEvaluatorCatalog
from warp_taskgen.rewards.network_event import _eval_network_event

_FINAL_STATE_REWARD_TYPES = frozenset({"FinalStateEvaluator", "final_state"})


def _eval_rocket_chat_decision(
    config: dict[str, Any],
    instance: dict[str, Any],
    agent_result: Any | None,
) -> tuple[bool, str]:
    """Grade a task-id-less Rocket.Chat answer through WARP's local owner."""

    from warp_taskgen.phase_1.rocket_chat_evaluator import RocketChatEvaluator

    benchmark = str(
        instance.get("benchmark")
        or instance.get("benchmark_name")
        or instance.get("benchmark_adapter")
        or ""
    ).strip().lower()
    if benchmark not in {"tac", "theagentcompany", "the agent company", "the_agent_company"}:
        return False, "Rocket.Chat evaluator requires the explicit TAC benchmark"
    expected = config.get("expected")
    if not isinstance(expected, dict):
        return False, "Rocket.Chat evaluator expected decision must be an object"
    response = getattr(agent_result, "final_result", None) if agent_result is not None else None
    if response is None:
        return False, "Rocket.Chat evaluator received no structured final response"
    grade = RocketChatEvaluator().grade(response, expected)
    return bool(grade.ok), grade.reason


def _is_network_event_evaluator_name(name: Any) -> bool:
    return isinstance(name, str) and name in {"NetworkEventEvaluator", "network_event"}


def _run_homebrew_eval(
    reward: dict[str, Any],
    instance: dict[str, Any],
    agent_result: Any | None,
    network_trace: list[dict] | None,
    *,
    final_state_catalog: FinalStateEvaluatorCatalog | None = None,
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
        elif evaluator_type == "RocketChatEvaluator":
            passed, msg = _eval_rocket_chat_decision(config, instance, agent_result)
        elif _is_network_event_evaluator_name(evaluator_type):
            passed, msg = _eval_network_event(config, network_trace, instance)
        elif evaluator_type in _FINAL_STATE_REWARD_TYPES:
            if final_state_catalog is None:
                passed, msg = _eval_final_state(config, network_trace, instance)
            else:
                passed, msg = _eval_final_state(
                    config,
                    network_trace,
                    instance,
                    final_state_catalog=final_state_catalog,
                )
        else:
            passed, msg = False, f"Unknown evaluator type: {evaluator_type}"

        if not passed:
            all_passed = False
        messages.append(f"[{evaluator_type}] {'PASS' if passed else 'FAIL'}: {msg}")

    combined = "; ".join(messages)
    return all_passed, combined
