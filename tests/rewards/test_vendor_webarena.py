from __future__ import annotations

import builtins
import json
import types

import worldsim.rewards as rewards
from worldsim.rewards import run_reward_function, vendor_webarena


def test_webarena_eval_fails_closed_without_vendor_package(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("webarena_verified"):
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delenv(rewards.WEBARENA_EVAL_PYTHON_ENV, raising=False)
    monkeypatch.setattr(vendor_webarena, "_default_eval_python", lambda: "")

    passed, message = run_reward_function(
        reward={"task_id": "123", "eval": [{"evaluator": "AgentResponseEvaluator"}]},
        instance={},
    )

    assert passed is False
    assert "canonical WebArena Verified evaluation unavailable" in message


def test_webarena_eval_uses_separate_python_when_configured(monkeypatch):
    captured = {}

    def fake_run(cmd, input, capture_output, text, timeout, check):
        captured["cmd"] = cmd
        captured["payload"] = json.loads(input)
        return types.SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"passed": True, "message": "[AgentResponseEvaluator] PASS"}),
            stderr="",
        )

    monkeypatch.setenv(rewards.WEBARENA_EVAL_PYTHON_ENV, "/tmp/webarena-python")
    monkeypatch.setattr(vendor_webarena.subprocess, "run", fake_run)

    passed, message = run_reward_function(
        reward={"task_id": "123", "eval": [{"evaluator": "AgentResponseEvaluator"}]},
        instance={
            "site_name": "gitlab",
            "site_url": "http://gitlab.test",
        },
    )

    assert passed is True
    assert "PASS" in message
    assert captured["cmd"] == [
        "/tmp/webarena-python",
        "-m",
        rewards.WEBARENA_EVAL_MODULE,
    ]
    assert captured["payload"]["task_id"] == "123"
    assert captured["payload"]["environments"]["gitlab"] == ["http://gitlab.test"]


def test_webarena_environment_payload_ignores_unsupported_sites():
    payload = rewards._build_webarena_environment_payload(
        {
            "site_name": "shopping",
            "site_url": "http://shopping.test",
            "url_placeholders": {
                "__SHOPPING__": "http://shopping.test",
                "__WIKIPEDIA__": "http://wiki.test",
                "__GITLAB__": "http://gitlab.test",
            },
        }
    )

    assert payload == {"gitlab": ["http://gitlab.test"]}


def test_webarena_environment_payload_maps_postmill_to_reddit():
    payload = rewards._build_webarena_environment_payload(
        {
            "site_name": "postmill",
            "site_url": "http://reddit.test",
        }
    )

    assert payload == {"reddit": ["http://reddit.test"]}


def test_webarena_vendor_shims_normalize_network_event_alias():
    configs = rewards._apply_webarena_vendor_shims(
        [
            {
                "evaluator": "network_event",
                "expected": {"url": "http://reddit.test/submit", "http_method": "POST"},
            }
        ]
    )

    assert configs == [
        {
            "evaluator": "NetworkEventEvaluator",
            "expected": {"url": "http://reddit.test/submit", "http_method": "POST"},
        }
    ]
