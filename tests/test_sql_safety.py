from __future__ import annotations

import builtins
import json
import types

import pytest

import worldsim.rewards as rewards
from worldsim.rewards import _db_query_match, _validate_read_only_query, run_reward_function


@pytest.mark.parametrize(
    "query",
    [
        "SELECT 1",
        "WITH counts AS (SELECT 1) SELECT * FROM counts",
    ],
)
def test_read_only_query_allows_selects(query):
    _validate_read_only_query(query)


@pytest.mark.parametrize(
    "query",
    [
        "DELETE FROM products",
        "SELECT 1; UPDATE products SET name = 'oops'",
        "SET search_path TO public",
        "SELECT * INTO backup_table FROM products",
        "SELECT * FROM products INTO OUTFILE '/tmp/leak'",
        "SELECT * FROM products FOR UPDATE",
    ],
)
def test_read_only_query_rejects_write_capable_sql(query):
    with pytest.raises(ValueError):
        _validate_read_only_query(query)


def test_db_query_match_fails_when_read_only_guard_cannot_be_enabled(monkeypatch):
    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, query):
            return None

        def fetchall(self):
            return [(1,)]

    class FakeConnection:
        def cursor(self):
            return FakeCursor()

        def rollback(self):
            return None

        def close(self):
            return None

    monkeypatch.setattr(
        rewards,
        "_configure_read_only_connection",
        lambda conn, scheme: (_ for _ in ()).throw(RuntimeError("guard failed")),
    )
    monkeypatch.setattr(
        rewards,
        "pymysql",
        types.SimpleNamespace(connect=lambda **kwargs: FakeConnection()),
        raising=False,
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "pymysql",
        types.SimpleNamespace(connect=lambda **kwargs: FakeConnection()),
    )

    passed, message = _db_query_match(
        {"query": "SELECT 1", "expected": None},
        {"db_connection": "mysql://user:pass@localhost:3306/db"},
    )

    assert passed is False
    assert "guard failed" in message


def test_db_query_match_does_not_allow_substring_false_positive(monkeypatch):
    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, query):
            return None

        def fetchall(self):
            return [(10,)]

    class FakeConnection:
        def cursor(self):
            return FakeCursor()

        def rollback(self):
            return None

        def close(self):
            return None

    monkeypatch.setattr(rewards, "_configure_read_only_connection", lambda conn, scheme: None)
    monkeypatch.setitem(
        __import__("sys").modules,
        "pymysql",
        types.SimpleNamespace(connect=lambda **kwargs: FakeConnection()),
    )

    passed, _ = _db_query_match(
        {"query": "SELECT value FROM results", "expected": 1},
        {"db_connection": "mysql://user:pass@localhost:3306/db"},
    )

    assert passed is False


def test_db_query_match_rejects_blank_db_connection():
    passed, message = _db_query_match(
        {"query": "SELECT 1", "expected": None},
        {"db_connection": "   "},
    )

    assert passed is False
    assert "non-empty URI string" in message


def test_db_query_match_rejects_unsupported_db_connection_scheme():
    passed, message = _db_query_match(
        {"query": "SELECT 1", "expected": None},
        {"db_connection": "sqlite:///tmp/demo.db"},
    )

    assert passed is False
    assert "unsupported scheme 'sqlite'" in message


def test_webarena_eval_fails_closed_without_vendor_package(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("webarena_verified"):
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delenv(rewards.WEBARENA_EVAL_PYTHON_ENV, raising=False)
    # Also disable the repo-relative default so the in-process import path
    # is genuinely exercised (otherwise the subprocess path picks up the
    # real evaluator venv and sidesteps the fail-closed branch under test).
    monkeypatch.setattr(rewards, "_default_eval_python", lambda: "")

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
    monkeypatch.setattr(rewards.subprocess, "run", fake_run)

    passed, message = run_reward_function(
        reward={"task_id": "123", "eval": [{"evaluator": "AgentResponseEvaluator"}]},
        instance={
            "site_name": "shopping",
            "site_url": "http://shopping.test",
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
    assert captured["payload"]["environments"]["shopping"] == ["http://shopping.test"]


def test_top_level_network_event_evaluator_dispatches_without_checker_registry():
    passed, message = run_reward_function(
        reward={
            "type": "NetworkEventEvaluator",
            "expected": {"url": "http://shopping.test/reviews", "http_method": "POST"},
        },
        instance={"site_name": "shopping", "site_url": "http://shopping.test"},
        network_trace=[{"url": "http://shopping.test/reviews", "method": "POST"}],
    )

    assert passed is True
    assert "matching event" in message


def test_top_level_network_event_alias_dispatches_without_checker_registry():
    passed, message = run_reward_function(
        reward={
            "type": "network_event",
            "expected": {"url": "http://shopping.test/reviews", "http_method": "POST"},
        },
        instance={"site_name": "shopping", "site_url": "http://shopping.test"},
        network_trace=[{"url": "http://shopping.test/reviews", "method": "POST"}],
    )

    assert passed is True
    assert "matching event" in message


def test_homebrew_eval_accepts_network_event_alias():
    passed, message = run_reward_function(
        reward={
            "eval": [
                {
                    "evaluator": "network_event",
                    "expected": {"url": "http://shopping.test/reviews", "http_method": "POST"},
                }
            ]
        },
        instance={"site_name": "shopping", "site_url": "http://shopping.test"},
        network_trace=[{"url": "http://shopping.test/reviews", "method": "POST"}],
    )

    assert passed is True
    assert "[network_event] PASS" in message


def test_webarena_vendor_shims_normalize_network_event_alias():
    configs = rewards._apply_webarena_vendor_shims(
        [
            {
                "evaluator": "network_event",
                "expected": {"url": "http://shopping.test/reviews", "http_method": "POST"},
            }
        ]
    )

    assert configs == [
        {
            "evaluator": "NetworkEventEvaluator",
            "expected": {"url": "http://shopping.test/reviews", "http_method": "POST"},
        }
    ]


def test_top_level_agent_response_evaluator_dispatches_without_checker_registry():
    agent_result = types.SimpleNamespace(final_result='["fixed"]', is_done=True)

    passed, message = run_reward_function(
        reward={
            "type": "AgentResponseEvaluator",
            "expected": {
                "task_type": "retrieve",
                "status": "SUCCESS",
                "retrieved_data": ["fixed"],
            },
        },
        instance={},
        agent_result=agent_result,
    )

    assert passed is True
    assert "retrieved_data matches" in message


def test_novel_agent_response_evaluator_accepts_scalar_retrieved_data_like_upstream():
    agent_result = types.SimpleNamespace(
        final_result=json.dumps(
            {
                "task_type": "retrieve",
                "status": "SUCCESS",
                "retrieved_data": "populated",
            }
        ),
        is_done=True,
    )

    passed, message = run_reward_function(
        reward={
            "eval": [
                {
                    "evaluator": "AgentResponseEvaluator",
                    "results_schema": {"type": "array", "items": {"type": "string"}},
                    "expected": {
                        "task_type": "retrieve",
                        "status": "SUCCESS",
                        "retrieved_data": ["populated"],
                    },
                }
            ]
        },
        instance={},
        agent_result=agent_result,
    )

    assert passed is True
    assert "retrieved_data matches" in message


def test_novel_agent_response_evaluator_rejects_object_for_string_array_answer():
    agent_result = types.SimpleNamespace(
        final_result=json.dumps(
            {
                "task_type": "retrieve",
                "status": "SUCCESS",
                "retrieved_data": {"body": "populated"},
            }
        ),
        is_done=True,
    )

    passed, message = run_reward_function(
        reward={
            "eval": [
                {
                    "evaluator": "AgentResponseEvaluator",
                    "results_schema": {"type": "array", "items": {"type": "string"}},
                    "expected": {
                        "task_type": "retrieve",
                        "status": "SUCCESS",
                        "retrieved_data": ["populated"],
                    },
                }
            ]
        },
        instance={},
        agent_result=agent_result,
    )

    assert passed is False
    assert "retrieved_data mismatch" in message
