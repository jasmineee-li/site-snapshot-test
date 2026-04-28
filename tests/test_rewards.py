from __future__ import annotations

from worldsim.rewards import run_reward_function


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
