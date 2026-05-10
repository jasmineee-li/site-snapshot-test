"""Tests for the flat-to-HAR network trace converter and its wire-ins.

Covers three layers:

1. ``worldsim.har_converter`` primitives (unit tests).
2. Live wire-in at ``_NetworkTraceRecorder._write_trace`` (network.har on disk
   is valid HAR, network_trace.json is unchanged).
3. Live wire-in at ``_NetworkTraceRecorder._write_trace`` preserves the flat
   network_trace.json shape while writing valid HAR.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import worldsim.browser_use_agent as browser_use_agent
from worldsim.browser_use_agent import _NetworkTraceRecorder
from worldsim.har_converter import (
    NetworkTraceUnavailableError,
    ensure_har_trace,
    flat_event_to_har_entry,
    flat_events_to_har_entries,
    minimal_har_placeholder_entry,
    strict_runtime_har_trace,
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


def test_flat_event_response_content_is_preserved_for_network_evaluators():
    flat = {
        "url": "http://shopping.test/api/orders/1",
        "method": "GET",
        "response_status": 200,
        "response_headers": {"Content-Type": "application/json"},
        "response_content": '{"status":"ok"}',
    }
    entry = flat_event_to_har_entry(flat)

    assert entry["response"]["content"] == {
        "size": len('{"status":"ok"}'),
        "mimeType": "application/json",
        "text": '{"status":"ok"}',
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
# strict_runtime_har_trace
# ─────────────────────────────────────────────────────────────────────


def test_strict_runtime_har_trace_none_raises():
    with pytest.raises(NetworkTraceUnavailableError, match="network_trace_unavailable"):
        strict_runtime_har_trace(None)


def test_strict_runtime_har_trace_empty_list_raises():
    with pytest.raises(NetworkTraceUnavailableError, match="network_trace_unavailable"):
        strict_runtime_har_trace([])


def test_strict_runtime_har_trace_har_entries_returned_as_is():
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
    result = strict_runtime_har_trace(har_entries)

    assert len(result) == 2
    assert result[0]["request"]["url"] == "http://a.test"
    assert result[1]["request"]["url"] == "http://b.test"


def test_strict_runtime_har_trace_flat_events_are_converted():
    flat = [
        {
            "url": "http://a.test?ticket=123&ticket=456",
            "method": "GET",
            "response_status": 200,
        },
        {"url": "http://b.test", "method": "POST", "response_status": 201},
    ]
    result = strict_runtime_har_trace(flat)

    assert len(result) == 2
    assert result[0]["request"]["url"] == "http://a.test?ticket=123&ticket=456"
    assert result[1]["request"]["url"] == "http://b.test"
    assert result[0]["response"]["status"] == 200
    assert result[1]["response"]["status"] == 201
    assert result[0]["request"]["queryString"] == [
        {"name": "ticket", "value": "123"},
        {"name": "ticket", "value": "456"},
    ]


def test_strict_runtime_har_trace_degenerate_flat_events_raise():
    with pytest.raises(NetworkTraceUnavailableError, match="real HTTP evidence"):
        strict_runtime_har_trace([{}])


# ─────────────────────────────────────────────────────────────────────
# ensure_har_trace (rescore-only placeholder helper)
# ─────────────────────────────────────────────────────────────────────


def test_ensure_har_trace_none_returns_single_placeholder():
    result = ensure_har_trace(None)

    assert len(result) == 1
    assert result[0]["request"]["url"] == "about:blank"


def test_ensure_har_trace_empty_list_returns_single_placeholder():
    result = ensure_har_trace([])

    assert len(result) == 1
    assert result[0]["request"]["url"] == "about:blank"


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


def test_write_trace_preserves_browser_use_agent_writer_patch(monkeypatch, tmp_path):
    recorder = _make_recorder(tmp_path)
    writes = []

    def fake_write_json_atomic(path, payload):
        writes.append((Path(path).name, payload))

    monkeypatch.setattr(browser_use_agent, "write_json_atomic", fake_write_json_atomic)

    recorder._write_trace([{"url": "http://shopping.test", "method": "GET"}])

    assert [name for name, _payload in writes] == [
        "network_trace.json",
        "navigation_trace.json",
        "network.har",
    ]


def test_write_trace_empty_trace_writes_empty_entries_no_placeholder(tmp_path):
    """Empty input at the recorder layer leaves HAR.log.entries=[] (no
    placeholder is injected here; that is a reward-time concern)."""
    recorder = _make_recorder(tmp_path)

    recorder._write_trace([])

    har = json.loads((tmp_path / "network.har").read_text())
    assert har["log"]["entries"] == []

    persisted = json.loads((tmp_path / "network_trace.json").read_text())
    assert persisted == []
