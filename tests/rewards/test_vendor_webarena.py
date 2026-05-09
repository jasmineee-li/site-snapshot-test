from __future__ import annotations

import builtins
import json
import types

import worldsim.rewards.vendor_webarena as vendor_webarena
from worldsim.rewards import run_reward_function
from worldsim.rewards.vendor_webarena import (
    LEGACY_WEBARENA_EVAL_PYTHON_ENV,
    WEBARENA_EVAL_MODULE,
    WEBARENA_EVAL_PYTHON_ENV,
    _apply_webarena_vendor_shims,
    webarena_eval_python_override,
)


def test_webarena_eval_fails_closed_without_vendor_package(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("webarena_verified"):
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delenv(WEBARENA_EVAL_PYTHON_ENV, raising=False)
    monkeypatch.delenv(LEGACY_WEBARENA_EVAL_PYTHON_ENV, raising=False)
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

    monkeypatch.setenv(WEBARENA_EVAL_PYTHON_ENV, "/tmp/webarena-python")
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
        WEBARENA_EVAL_MODULE,
    ]
    assert captured["payload"]["task_id"] == "123"
    assert captured["payload"]["environments"]["gitlab"] == ["http://gitlab.test"]


def test_webarena_eval_python_override_prefers_warp_taskgen_env(monkeypatch):
    monkeypatch.setenv(WEBARENA_EVAL_PYTHON_ENV, "/tmp/canonical-python")
    monkeypatch.setenv(LEGACY_WEBARENA_EVAL_PYTHON_ENV, "/tmp/legacy-python")

    assert webarena_eval_python_override() == "/tmp/canonical-python"


def test_webarena_vendor_shims_normalize_network_event_alias():
    configs = _apply_webarena_vendor_shims(
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


def _install_fake_subprocess(monkeypatch) -> dict:
    captured: dict = {}

    def fake_run(cmd, input, capture_output, text, timeout, check):
        captured["cmd"] = cmd
        captured["payload"] = json.loads(input)
        return types.SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"passed": True, "message": "[AgentResponseEvaluator] PASS"}),
            stderr="",
        )

    monkeypatch.setenv(WEBARENA_EVAL_PYTHON_ENV, "/tmp/webarena-python")
    monkeypatch.setattr(vendor_webarena.subprocess, "run", fake_run)
    return captured


def test_reward_subprocess_receives_placeholder_for_empty_network_trace(monkeypatch):
    captured = _install_fake_subprocess(monkeypatch)

    passed, _ = run_reward_function(
        reward={"task_id": 11, "eval": [{"evaluator": "AgentResponseEvaluator"}]},
        instance={"site_name": "gitlab", "site_url": "http://gitlab.test"},
        network_trace=[],
    )

    assert passed is True
    trace = captured["payload"]["network_trace"]
    assert len(trace) == 1
    assert trace[0]["request"]["url"] == "about:blank"


def test_reward_subprocess_receives_placeholder_for_none_network_trace(monkeypatch):
    captured = _install_fake_subprocess(monkeypatch)

    run_reward_function(
        reward={"task_id": 11, "eval": [{"evaluator": "AgentResponseEvaluator"}]},
        instance={"site_name": "gitlab", "site_url": "http://gitlab.test"},
        network_trace=None,
    )

    trace = captured["payload"]["network_trace"]
    assert len(trace) == 1
    assert trace[0]["request"]["url"] == "about:blank"


def test_reward_network_evaluator_fails_closed_on_empty_network_trace(monkeypatch):
    monkeypatch.setenv(WEBARENA_EVAL_PYTHON_ENV, "/tmp/webarena-python")

    passed, message = run_reward_function(
        reward={"task_id": 11, "eval": [{"evaluator": "NetworkEventEvaluator"}]},
        instance={"site_name": "gitlab", "site_url": "http://gitlab.test"},
        network_trace=[],
    )

    assert passed is False
    assert message == "network_trace_unavailable: trace is missing or empty"


def test_reward_subprocess_receives_converted_har_for_flat_input(monkeypatch):
    captured = _install_fake_subprocess(monkeypatch)

    flat_trace = [
        {
            "url": "http://gitlab.test/api/v4/projects/1/issues",
            "method": "GET",
            "response_status": 200,
            "headers": {"Accept": "application/json"},
        },
        {
            "url": "http://gitlab.test/api/v4/projects/1/issues",
            "method": "POST",
            "response_status": 201,
            "headers": {"Content-Type": "application/json"},
        },
    ]
    run_reward_function(
        reward={"task_id": 11, "eval": [{"evaluator": "AgentResponseEvaluator"}]},
        instance={"site_name": "gitlab", "site_url": "http://gitlab.test"},
        network_trace=flat_trace,
    )

    trace = captured["payload"]["network_trace"]
    assert len(trace) == 2
    for entry in trace:
        assert "request" in entry
        assert "response" in entry
    assert trace[0]["request"]["method"] == "GET"
    assert trace[1]["request"]["method"] == "POST"


def test_reward_subprocess_passes_through_already_har_input(monkeypatch):
    captured = _install_fake_subprocess(monkeypatch)

    har_input = [
        {
            "request": {
                "method": "GET",
                "url": "http://gitlab.test/api/v4/projects/1/issues",
                "headers": [{"name": "Accept", "value": "application/json"}],
            },
            "response": {
                "status": 200,
                "headers": [],
                "cookies": [],
            },
        }
    ]
    run_reward_function(
        reward={"task_id": 11, "eval": [{"evaluator": "AgentResponseEvaluator"}]},
        instance={"site_name": "gitlab", "site_url": "http://gitlab.test"},
        network_trace=har_input,
    )

    trace = captured["payload"]["network_trace"]
    assert len(trace) == 1
    assert trace[0]["request"]["method"] == "GET"
    assert trace[0]["request"]["url"] == "http://gitlab.test/api/v4/projects/1/issues"
    assert "request" not in trace[0]["request"]
