from __future__ import annotations

from worldsim.rewards import run_reward_function


class _AgentResult:
    def __init__(self, final_result: str, *, is_done: bool = True) -> None:
        self.final_result = final_result
        self.is_done = is_done


def test_network_event_reward_accepts_expected_url_list() -> None:
    passed, message = run_reward_function(
        {
            "type": "NetworkEventEvaluator",
            "expected": {
                "url": ["__REDDIT__/submit", "__REDDIT__/submit/nyc"],
                "http_method": "POST",
            },
        },
        {
            "site_name": "reddit",
            "site_url": "http://reddit.test",
            "url_placeholders": {"__REDDIT__": "http://reddit.test"},
        },
        network_trace=[
            {"url": "http://reddit.test/submit/nyc", "method": "POST"},
        ],
    )

    assert passed is True
    assert "matching event" in message


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


def test_network_event_reward_rejects_unsupported_expected_fields() -> None:
    passed, message = run_reward_function(
        {
            "type": "NetworkEventEvaluator",
            "expected": {
                "url": "__REDDIT__/submit",
                "http_method": "POST",
                "post_data": {"submission[title]": "hello"},
            },
        },
        {
            "site_name": "reddit",
            "site_url": "http://reddit.test",
            "url_placeholders": {"__REDDIT__": "http://reddit.test"},
        },
        network_trace=[
            {"url": "http://reddit.test/submit", "method": "POST"},
        ],
    )

    assert passed is False
    assert "post_data" in message


def test_network_event_reward_does_not_substring_match_paths() -> None:
    passed, message = run_reward_function(
        {
            "type": "NetworkEventEvaluator",
            "expected": {"url": "__REDDIT__/submit", "http_method": "POST"},
        },
        {
            "site_name": "reddit",
            "site_url": "http://reddit.test",
            "url_placeholders": {"__REDDIT__": "http://reddit.test"},
        },
        network_trace=[
            {"url": "http://reddit.test/submit/nyc", "method": "POST"},
        ],
    )

    assert passed is False
    assert "no matching network event" in message
