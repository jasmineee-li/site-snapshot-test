from __future__ import annotations

import json
from collections import Counter
from typing import Any

from warp_taskgen.agent_response_transform import transform_agent_response


def _build_agent_response(eval_configs: list[dict], agent_result: Any | None) -> Any:
    """Build a WebArena Verified-compatible agent response dict.

    When the task has an ``AgentResponseEvaluator``, first try the 4-strategy
    extractor from ``warp_taskgen.agent_response_transform``. This matches the
    vendor's canonical CLI behavior: ``eval-tasks`` wires the extractor in via
    ``--agent-response-transform`` and prefers the transformed dict over the
    raw text when it yields a JSON object (see
    ``vendors/webarena-verified/src/webarena_verified/__main__.py`` around
    ``_transform_agent_response`` at lines 944-950). Without this we score 0
    on runs where the content is correct but the JSON wrapper is mangled,
    while the benchmark's own CLI scores 1. Scope is intentionally limited to
    tasks that declare an ``AgentResponseEvaluator`` -- other evaluator types
    (and other benchmarks) are left untouched.
    """

    # Infer task type from the expected response in the first AgentResponseEvaluator
    task_type = "retrieve"
    has_agent_response_evaluator = False
    for config in eval_configs:
        if _is_agent_response_evaluator_config(config):
            has_agent_response_evaluator = True
            expected = config.get("expected", {})
            task_type = expected.get("task_type", "retrieve")
            break

    final_result = None
    if agent_result is not None:
        final_result = getattr(agent_result, "final_result", None)

    is_done = agent_result is not None and getattr(agent_result, "is_done", False)

    # Prefer a transform-extracted dict for WebArena-Verified
    # AgentResponseEvaluator tasks. If no strategy matches, preserve the raw
    # response instead of upgrading prose into structured retrieved_data; this
    # matches the upstream benchmark's fail-closed scoring semantics.
    if has_agent_response_evaluator and isinstance(final_result, str) and final_result.strip():
        transformed = transform_agent_response(final_result)
        if transformed is not None:
            return transformed
        parsed = _parse_json_final_result(final_result)
        if parsed is not None:
            if isinstance(parsed, dict):
                return parsed
            return {
                "task_type": task_type,
                "status": "SUCCESS" if is_done else "FAILURE",
                "retrieved_data": parsed,
            }

    if final_result is not None:
        return final_result

    return {
        "task_type": task_type,
        "status": "SUCCESS" if is_done else "FAILURE",
        "retrieved_data": None,
    }


def _is_agent_response_evaluator_config(config: dict[str, Any]) -> bool:
    return (
        config.get("evaluator") == "AgentResponseEvaluator"
        or config.get("type") == "AgentResponseEvaluator"
    )


def _parse_json_final_result(final_result: str) -> Any | None:
    try:
        return json.loads(final_result)
    except (json.JSONDecodeError, TypeError):
        return None


def _eval_agent_response(config: dict, agent_response: Any) -> tuple[bool, str]:
    """Evaluate an AgentResponseEvaluator config against agent response."""
    if not isinstance(agent_response, dict):
        return False, "agent response was not a structured object"

    expected = config.get("expected", {})

    expected_type = expected.get("task_type", "").lower()
    actual_type = agent_response.get("task_type", "").lower()
    if expected_type and actual_type != expected_type:
        return False, f"task_type mismatch: expected {expected_type}, got {actual_type}"

    expected_status = expected.get("status", "").upper()
    actual_status = agent_response.get("status", "").upper()
    if expected_status and actual_status != expected_status:
        return False, f"status mismatch: expected {expected_status}, got {actual_status}"

    expected_data = expected.get("retrieved_data")
    if expected_data is None:
        return True, "status and task_type match"

    actual_data = agent_response.get("retrieved_data")
    if actual_data is None:
        return False, "expected retrieved_data but got None"

    ordered = config.get("ordered", False)
    if _compare_data(expected_data, actual_data, ordered=ordered):
        return True, "retrieved_data matches"

    return False, f"retrieved_data mismatch: expected {expected_data!r}, got {actual_data!r}"


def _compare_data(expected: Any, actual: Any, ordered: bool = False) -> bool:
    """Compare expected vs actual retrieved_data with WebArena-like coercion."""

    def normalize(v: Any) -> str:
        return str(v).strip().lower()

    expected_norm = [normalize(e) for e in _as_retrieved_sequence(expected)]
    actual_norm = [normalize(a) for a in _as_retrieved_sequence(actual)]

    if ordered:
        return expected_norm == actual_norm

    # Unordered: upstream matching is exact multiset equality; extra actual
    # values are failures, not harmless detail.
    return Counter(expected_norm) == Counter(actual_norm)


def _as_retrieved_sequence(value: Any) -> list[Any]:
    """Mirror WebArena Verified actual-response singleton coercion.

    The upstream ``AgentResponseEvaluator`` wraps non-list actual
    ``retrieved_data`` values in a one-item tuple before applying array
    normalization. Novel WorldSim tasks use this homebrew path because they do
    not have canonical WebArena task IDs, so we preserve that compatibility
    here instead of treating a scalar string as an iterable of characters.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]
