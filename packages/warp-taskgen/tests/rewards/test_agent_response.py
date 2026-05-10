from __future__ import annotations

from worldsim.rewards import run_reward_function


class _AgentResult:
    def __init__(self, final_result: str, *, is_done: bool = True) -> None:
        self.final_result = final_result
        self.is_done = is_done

def test_agent_response_fallback_rejects_plaintext_final_answer() -> None:
    passed, message = run_reward_function(
        {
            "eval": [
                {
                    "evaluator": "AgentResponseEvaluator",
                    "results_schema": {"type": "array", "items": {"type": "string"}},
                    "expected": {
                        "task_type": "retrieve",
                        "status": "SUCCESS",
                        "retrieved_data": ["red widget"],
                    },
                }
            ]
        },
        {"site_name": "reddit", "site_url": "http://reddit.test"},
        agent_result=_AgentResult("red widget"),
    )

    assert passed is False
    assert "not a structured object" in message


def test_agent_response_fallback_rejects_extra_retrieved_data() -> None:
    passed, message = run_reward_function(
        {
            "eval": [
                {
                    "evaluator": "AgentResponseEvaluator",
                    "results_schema": {"type": "array", "items": {"type": "string"}},
                    "expected": {
                        "task_type": "retrieve",
                        "status": "SUCCESS",
                        "retrieved_data": ["red widget"],
                    },
                }
            ]
        },
        {"site_name": "reddit", "site_url": "http://reddit.test"},
        agent_result=_AgentResult(
            '{"task_type":"retrieve","status":"SUCCESS","retrieved_data":["red widget","blue widget"]}'
        ),
    )

    assert passed is False
    assert "retrieved_data mismatch" in message


def test_top_level_agent_response_reward_accepts_evaluator_alias() -> None:
    passed, message = run_reward_function(
        {
            "evaluator": "AgentResponseEvaluator",
            "results_schema": {"type": "array", "items": {"type": "string"}},
            "expected": {
                "task_type": "retrieve",
                "status": "SUCCESS",
                "retrieved_data": ["blank"],
            },
        },
        {"site_name": "reddit", "site_url": "http://reddit.test"},
        agent_result=_AgentResult(
            '{"task_type":"retrieve","status":"SUCCESS","retrieved_data":["blank"]}'
        ),
    )

    assert passed is True
    assert "retrieved_data matches" in message
