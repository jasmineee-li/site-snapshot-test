"""Tests for the flat-to-HAR network trace converter and its wire-ins.

Covers three layers:

1. ``worldsim.har_converter`` primitives (unit tests).
2. Live wire-in at ``_NetworkTraceRecorder._write_trace`` (network.har on disk
   is valid HAR, network_trace.json is unchanged).
3. Live wire-in at ``worldsim.rewards._run_webarena_verified_eval`` (the
   subprocess payload carries HAR entries regardless of input shape).

Anchored by a regression test that reproduces task 11: an
``AgentResponseEvaluator`` with ``retrieved_data: [6]`` that previously scored
0 because the empty flat trace tripped the vendor's "Trace content list is
empty" guard before the agent-response check could run.
"""

from __future__ import annotations

import json
import os
import types
from pathlib import Path

import pytest

import worldsim.rewards as rewards
from worldsim.browser_use_agent import _NetworkTraceRecorder
from worldsim.har_converter import (
    ensure_har_trace,
    flat_event_to_har_entry,
    flat_events_to_har_entries,
    minimal_har_placeholder_entry,
)

# ─────────────────────────────────────────────────────────────────────
# flat_event_to_har_entry
# ─────────────────────────────────────────────────────────────────────


def test_flat_event_minimum_viable_entry_has_request_and_response():
    """Just url + method + response_status produces a valid HAR entry."""
    flat = {
        "url": "http://shopping.test/api/orders/1",
        "method": "GET",
        "response_status": 200,
    }
    entry = flat_event_to_har_entry(flat)

    assert "request" in entry
    assert "response" in entry
    assert entry["request"]["method"] == "GET"
    assert entry["request"]["url"] == "http://shopping.test/api/orders/1"
    assert entry["response"]["status"] == 200


def test_flat_event_headers_from_dict_become_har_name_value_list():
    flat = {
        "url": "http://shopping.test",
        "method": "GET",
        "response_status": 200,
        "headers": {"User-Agent": "test-agent/1.0", "Accept": "application/json"},
    }
    entry = flat_event_to_har_entry(flat)

    headers = entry["request"]["headers"]
    assert {"name": "User-Agent", "value": "test-agent/1.0"} in headers
    assert {"name": "Accept", "value": "application/json"} in headers
    # All entries follow {name, value} shape.
    assert all(set(h.keys()) == {"name", "value"} for h in headers)


def test_flat_event_headers_as_list_of_name_value_pairs_pass_through():
    flat = {
        "url": "http://shopping.test",
        "method": "GET",
        "response_status": 200,
        "headers": [
            {"name": "User-Agent", "value": "already/normalized"},
            {"name": "X-Custom", "value": "42"},
        ],
    }
    entry = flat_event_to_har_entry(flat)

    assert {"name": "User-Agent", "value": "already/normalized"} in entry["request"]["headers"]
    assert {"name": "X-Custom", "value": "42"} in entry["request"]["headers"]


def test_flat_event_cookies_dict_input():
    flat = {
        "url": "http://shopping.test",
        "method": "GET",
        "response_status": 200,
        "response_cookies": {"session_id": "abc123", "csrf": "tok"},
    }
    entry = flat_event_to_har_entry(flat)

    cookies = entry["response"]["cookies"]
    assert {"name": "session_id", "value": "abc123"} in cookies
    assert {"name": "csrf", "value": "tok"} in cookies


def test_flat_event_cookies_list_of_dicts_input():
    flat = {
        "url": "http://shopping.test",
        "method": "GET",
        "response_status": 200,
        "response_cookies": [
            {"name": "session_id", "value": "abc123"},
            {"name": "csrf", "value": "tok"},
        ],
    }
    entry = flat_event_to_har_entry(flat)

    cookies = entry["response"]["cookies"]
    assert {"name": "session_id", "value": "abc123"} in cookies
    assert {"name": "csrf", "value": "tok"} in cookies


def test_flat_event_cookies_list_of_pairs_input():
    flat = {
        "url": "http://shopping.test",
        "method": "GET",
        "response_status": 200,
        "response_cookies": [("session_id", "abc123"), ("csrf", "tok")],
    }
    entry = flat_event_to_har_entry(flat)

    cookies = entry["response"]["cookies"]
    assert {"name": "session_id", "value": "abc123"} in cookies
    assert {"name": "csrf", "value": "tok"} in cookies


def test_flat_event_post_data_with_form_content_type():
    flat = {
        "url": "http://shopping.test/login",
        "method": "POST",
        "response_status": 200,
        "headers": {"Content-Type": "application/x-www-form-urlencoded"},
        "post_data": "a=1&b=2",
    }
    entry = flat_event_to_har_entry(flat)

    assert entry["request"]["postData"] == {
        "mimeType": "application/x-www-form-urlencoded",
        "text": "a=1&b=2",
    }


def test_flat_event_missing_post_data_omits_postdata_key():
    flat = {
        "url": "http://shopping.test",
        "method": "GET",
        "response_status": 200,
    }
    entry = flat_event_to_har_entry(flat)

    assert "postData" not in entry["request"]


def test_flat_event_redacted_post_data_is_dropped():
    """post_data=='<redacted>' (from trace redaction) drops the postData key."""
    flat = {
        "url": "http://shopping.test/login",
        "method": "POST",
        "response_status": 200,
        "headers": {"Content-Type": "application/x-www-form-urlencoded"},
        "post_data": "<redacted>",
    }
    entry = flat_event_to_har_entry(flat)

    assert "postData" not in entry["request"]


def test_flat_event_response_status_string_is_coerced_to_int():
    flat = {
        "url": "http://shopping.test",
        "method": "GET",
        "response_status": "200",
    }
    entry = flat_event_to_har_entry(flat)

    assert entry["response"]["status"] == 200
    assert isinstance(entry["response"]["status"], int)


def test_flat_event_missing_response_status_becomes_zero():
    flat = {"url": "http://shopping.test", "method": "GET"}
    entry = flat_event_to_har_entry(flat)

    assert entry["response"]["status"] == 0


def test_flat_event_round_trips_through_vendor_networkevent():
    """A converted entry satisfies the vendor's NetworkEvent property accessors."""
    pytest.importorskip("webarena_verified")
    from types import MappingProxyType

    from webarena_verified.types.tracing import NetworkEvent

    flat = {
        "url": "http://shopping.test/api/orders/42",
        "method": "POST",
        "response_status": 201,
        "headers": {"Content-Type": "application/json", "Referer": "http://shopping.test/orders"},
        "post_data": '{"id": 42}',
    }
    entry = flat_event_to_har_entry(flat)

    event = NetworkEvent(data=MappingProxyType(entry))
    assert event.http_method == "POST"
    assert event.url == "http://shopping.test/api/orders/42"
    assert event.request_status == 201
    assert event.referer == "http://shopping.test/orders"


# ─────────────────────────────────────────────────────────────────────
# flat_events_to_har_entries
# ─────────────────────────────────────────────────────────────────────


def test_flat_events_empty_list_returns_empty():
    assert flat_events_to_har_entries([]) == []


def test_flat_events_none_returns_empty():
    assert flat_events_to_har_entries(None) == []


def test_flat_events_skips_non_dict_elements():
    events = [
        {"url": "http://a.test", "method": "GET", "response_status": 200},
        "not-a-dict",
        42,
        None,
        {"url": "http://b.test", "method": "POST", "response_status": 201},
    ]
    entries = flat_events_to_har_entries(events)

    assert len(entries) == 2
    assert entries[0]["request"]["url"] == "http://a.test"
    assert entries[1]["request"]["url"] == "http://b.test"


# ─────────────────────────────────────────────────────────────────────
# minimal_har_placeholder_entry
# ─────────────────────────────────────────────────────────────────────


def test_placeholder_has_request_and_response_with_about_blank():
    entry = minimal_har_placeholder_entry()

    assert "request" in entry
    assert "response" in entry
    assert entry["request"]["url"] == "about:blank"


def test_placeholder_parses_cleanly_as_vendor_networkevent():
    pytest.importorskip("webarena_verified")
    from types import MappingProxyType

    from webarena_verified.types.tracing import NetworkEvent

    entry = minimal_har_placeholder_entry()
    event = NetworkEvent(data=MappingProxyType(entry))

    # All the key accessors should evaluate without raising.
    assert event.url == "about:blank"
    assert event.http_method == "GET"
    assert event.request_status == 0


# ─────────────────────────────────────────────────────────────────────
# ensure_har_trace
# ─────────────────────────────────────────────────────────────────────


def test_ensure_har_trace_none_returns_single_placeholder():
    result = ensure_har_trace(None)

    assert len(result) == 1
    assert result[0]["request"]["url"] == "about:blank"


def test_ensure_har_trace_empty_list_returns_single_placeholder():
    result = ensure_har_trace([])

    assert len(result) == 1
    assert result[0]["request"]["url"] == "about:blank"


def test_ensure_har_trace_har_entries_returned_as_is():
    har_entries = [
        {
            "request": {"url": "http://a.test", "method": "GET", "headers": []},
            "response": {"status": 200, "headers": [], "cookies": []},
        },
        {
            "request": {"url": "http://b.test", "method": "POST", "headers": []},
            "response": {"status": 201, "headers": [], "cookies": []},
        },
    ]
    result = ensure_har_trace(har_entries)

    assert len(result) == 2
    # Content preserved (content equality, whether or not the list is a copy).
    assert result[0]["request"]["url"] == "http://a.test"
    assert result[1]["request"]["url"] == "http://b.test"


def test_ensure_har_trace_flat_events_are_converted():
    flat = [
        {"url": "http://a.test", "method": "GET", "response_status": 200},
        {"url": "http://b.test", "method": "POST", "response_status": 201},
    ]
    result = ensure_har_trace(flat)

    assert len(result) == 2
    assert result[0]["request"]["url"] == "http://a.test"
    assert result[1]["request"]["url"] == "http://b.test"
    assert result[0]["response"]["status"] == 200
    assert result[1]["response"]["status"] == 201


def test_ensure_har_trace_degenerate_flat_events_still_yield_valid_entries():
    """[{}] is technically a flat list; the converter produces a near-empty
    HAR entry that the vendor can still parse (method='', url='', status=0)."""
    result = ensure_har_trace([{}])

    assert len(result) >= 1
    assert "request" in result[0]
    assert "response" in result[0]


# ─────────────────────────────────────────────────────────────────────
# _NetworkTraceRecorder._write_trace integration
# ─────────────────────────────────────────────────────────────────────


def _make_recorder(tmp_path: Path) -> _NetworkTraceRecorder:
    """Instantiate a recorder without a real CDP client.

    ``__init__`` reads ``browser_session.cdp_client`` via ``getattr`` so
    passing ``None`` safely leaves ``_client`` unset. We never call
    ``start()``/``stop()``; only ``_write_trace`` is exercised.
    """
    return _NetworkTraceRecorder(browser_session=None, task_dir=tmp_path)


def test_write_trace_emits_valid_har_for_flat_events(tmp_path):
    recorder = _make_recorder(tmp_path)
    flat_trace = [
        {
            "url": "http://shopping.test/api/orders/1",
            "method": "GET",
            "response_status": 200,
            "headers": {"Accept": "application/json"},
            "response_headers": {"Content-Type": "application/json"},
        },
        {
            "url": "http://shopping.test/api/orders/1",
            "method": "POST",
            "response_status": 201,
            "headers": {"Content-Type": "application/json"},
            "post_data": '{"id":1}',
        },
    ]

    recorder._write_trace(flat_trace)

    har = json.loads((tmp_path / "network.har").read_text())
    entries = har["log"]["entries"]
    assert len(entries) == 2
    for entry in entries:
        assert "request" in entry
        assert "response" in entry
    assert entries[0]["request"]["method"] == "GET"
    assert entries[1]["request"]["method"] == "POST"
    # Envelope metadata.
    assert har["log"]["version"] == "1.2"
    assert har["log"]["creator"]["name"] == "worldsim"


def test_write_trace_preserves_flat_entries_in_network_trace_json(tmp_path):
    recorder = _make_recorder(tmp_path)
    flat_trace = [
        {
            "url": "http://shopping.test/api/orders/1",
            "method": "GET",
            "response_status": 200,
            "headers": {"Accept": "application/json"},
            "query_params": {"foo": ["bar"]},
        }
    ]

    recorder._write_trace(flat_trace)

    persisted = json.loads((tmp_path / "network_trace.json").read_text())
    # The flat shape survives intact: no request/response wrapping at this layer.
    assert persisted == flat_trace
    assert "request" not in persisted[0]
    assert "response" not in persisted[0]


def test_write_trace_empty_trace_writes_empty_entries_no_placeholder(tmp_path):
    """Empty input at the recorder layer leaves HAR.log.entries=[] (no
    placeholder is injected here; that is a reward-time concern)."""
    recorder = _make_recorder(tmp_path)

    recorder._write_trace([])

    har = json.loads((tmp_path / "network.har").read_text())
    assert har["log"]["entries"] == []

    persisted = json.loads((tmp_path / "network_trace.json").read_text())
    assert persisted == []


# ─────────────────────────────────────────────────────────────────────
# rewards._run_webarena_verified_eval integration (subprocess path)
# ─────────────────────────────────────────────────────────────────────


def _install_fake_subprocess(monkeypatch) -> dict:
    """Replace rewards.subprocess.run and capture the JSON payload sent."""
    captured: dict = {}

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
    return captured


def test_reward_subprocess_receives_placeholder_for_empty_network_trace(monkeypatch):
    captured = _install_fake_subprocess(monkeypatch)

    passed, _ = rewards.run_reward_function(
        reward={"task_id": 11, "eval": [{"evaluator": "AgentResponseEvaluator"}]},
        instance={"site_name": "shopping", "site_url": "http://shopping.test"},
        network_trace=[],
    )

    assert passed is True
    trace = captured["payload"]["network_trace"]
    assert len(trace) == 1
    assert trace[0]["request"]["url"] == "about:blank"


def test_reward_subprocess_receives_placeholder_for_none_network_trace(monkeypatch):
    captured = _install_fake_subprocess(monkeypatch)

    rewards.run_reward_function(
        reward={"task_id": 11, "eval": [{"evaluator": "AgentResponseEvaluator"}]},
        instance={"site_name": "shopping", "site_url": "http://shopping.test"},
        network_trace=None,
    )

    trace = captured["payload"]["network_trace"]
    assert len(trace) == 1
    assert trace[0]["request"]["url"] == "about:blank"


def test_reward_subprocess_receives_converted_har_for_flat_input(monkeypatch):
    captured = _install_fake_subprocess(monkeypatch)

    flat_trace = [
        {
            "url": "http://shopping.test/api/orders/1",
            "method": "GET",
            "response_status": 200,
            "headers": {"Accept": "application/json"},
        },
        {
            "url": "http://shopping.test/api/orders/1",
            "method": "POST",
            "response_status": 201,
            "headers": {"Content-Type": "application/json"},
        },
    ]
    rewards.run_reward_function(
        reward={"task_id": 11, "eval": [{"evaluator": "AgentResponseEvaluator"}]},
        instance={"site_name": "shopping", "site_url": "http://shopping.test"},
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
    """Already-HAR input is not double-converted (no nested request/request)."""
    captured = _install_fake_subprocess(monkeypatch)

    har_input = [
        {
            "request": {
                "method": "GET",
                "url": "http://shopping.test/api/orders/1",
                "headers": [{"name": "Accept", "value": "application/json"}],
            },
            "response": {
                "status": 200,
                "headers": [],
                "cookies": [],
            },
        }
    ]
    rewards.run_reward_function(
        reward={"task_id": 11, "eval": [{"evaluator": "AgentResponseEvaluator"}]},
        instance={"site_name": "shopping", "site_url": "http://shopping.test"},
        network_trace=har_input,
    )

    trace = captured["payload"]["network_trace"]
    assert len(trace) == 1
    # The request dict still has the raw HAR keys, not a nested request object.
    assert trace[0]["request"]["method"] == "GET"
    assert trace[0]["request"]["url"] == "http://shopping.test/api/orders/1"
    assert "request" not in trace[0]["request"]


# ─────────────────────────────────────────────────────────────────────
# Regression: task-11-like AgentResponseEvaluator scores correctly
# ─────────────────────────────────────────────────────────────────────


def _vendor_python_or_skip() -> str:
    """Return a Python executable that can run the vendor evaluator.

    Order of preference:
      1. ``WORLDSIM_WEBARENA_EVAL_PYTHON`` from the environment (honors any
         existing test config).
      2. ``webarena_verified`` importable in the active interpreter.
      3. The shipped vendor venv at
         ``packages/worldsim-webarena-verified/.venv/bin/python`` (when it
         exists and has the evaluate module installed).
    Otherwise, skip the test.
    """
    env = os.environ.get(rewards.WEBARENA_EVAL_PYTHON_ENV, "").strip()
    if env and Path(env).exists():
        return env

    try:
        import webarena_verified  # noqa: F401

        # The in-process path is fine, but we still need the subprocess helper
        # module since _run_webarena_verified_eval uses subprocess only when
        # the env var is set. Run in-process by signalling an empty env.
        return ""
    except ImportError:
        pass

    candidate = (
        Path(__file__).resolve().parents[1]
        / "packages"
        / "worldsim-webarena-verified"
        / ".venv"
        / "bin"
        / "python"
    )
    if candidate.exists():
        return str(candidate)

    pytest.skip(
        "vendor WebArena Verified evaluator not available: set "
        f"{rewards.WEBARENA_EVAL_PYTHON_ENV} or install webarena_verified in the current env"
    )


class _FakeAgentResult:
    def __init__(self, final_result: str, is_done: bool = True) -> None:
        self.final_result = final_result
        self.is_done = is_done


def test_task_11_regression_agent_response_scores_on_empty_trace(monkeypatch):
    """Task 11: retrieved_data=[6] matches expected=[6], empty trace -> pass.

    Without the converter, the empty network_trace would short-circuit the
    vendor with "Trace content list is empty" before AgentResponseEvaluator
    ever ran. With ensure_har_trace injecting a placeholder, the agent
    response check runs and passes.
    """
    vendor_python = _vendor_python_or_skip()
    if vendor_python:
        monkeypatch.setenv(rewards.WEBARENA_EVAL_PYTHON_ENV, vendor_python)
    else:
        monkeypatch.delenv(rewards.WEBARENA_EVAL_PYTHON_ENV, raising=False)

    reward = {
        "task_id": 11,
        "eval": [
            {
                "evaluator": "AgentResponseEvaluator",
                "results_schema": {"type": "array", "items": {"type": "number"}},
                "expected": {
                    "task_type": "retrieve",
                    "status": "SUCCESS",
                    "retrieved_data": [6],
                },
            }
        ],
    }
    agent_result = _FakeAgentResult(
        final_result=(
            "```json\n"
            '{"task_type":"retrieve","status":"SUCCESS","retrieved_data":[6],'
            '"error_details":null}\n'
            "```"
        ),
    )
    instance = {
        "site_name": "shopping_admin",
        "site_url": "http://shopping-admin.test",
    }

    passed, message = rewards.run_reward_function(
        reward=reward,
        instance=instance,
        agent_result=agent_result,
        network_trace=[],
    )

    assert passed, f"expected pass, got message={message!r}"
    assert "[AgentResponseEvaluator]" in message
    assert "SUCCESS" in message.upper()
