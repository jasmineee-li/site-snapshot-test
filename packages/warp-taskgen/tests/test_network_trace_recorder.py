"""Unit tests for Commit 1 instrumentation additions to ``_NetworkTraceRecorder``.

These exercise the sync CDP handlers directly (no real browser). The
classifier (C1b) relies on the fields verified here: ``is_document_load``,
``resource_type``, ``redirect_chain``, and the ``navigation_trace.json``
artifact populated from ``Page.frameNavigated`` / ``Page.navigatedWithinDocument``.
"""

from __future__ import annotations

import json
from pathlib import Path

from warp_taskgen.browser_use_agent import _NetworkTraceRecorder


def _recorder(tmp_path: Path) -> _NetworkTraceRecorder:
    rec = _NetworkTraceRecorder(browser_session=None, task_dir=tmp_path)
    rec._recording = True
    return rec


def test_flatten_entry_marks_document_loads(tmp_path):
    rec = _recorder(tmp_path)
    rec._on_request_will_be_sent(
        {
            "requestId": "req-1",
            "timestamp": 1.0,
            "wallTime": 1700000000.0,
            "type": "Document",
            "request": {"url": "http://site.test/issues/42", "method": "GET"},
        }
    )
    rec._on_response_received(
        {
            "requestId": "req-1",
            "response": {
                "status": 200,
                "mimeType": "text/html",
                "headers": {"content-type": "text/html"},
            },
        }
    )
    trace = rec._finalize_trace()
    assert len(trace) == 1
    assert trace[0]["is_document_load"] is True
    assert trace[0]["resource_type"] == "Document"
    assert trace[0]["response_mime_type"] == "text/html"


def test_flatten_entry_non_document_loads(tmp_path):
    rec = _recorder(tmp_path)
    rec._on_request_will_be_sent(
        {
            "requestId": "xhr-1",
            "timestamp": 1.0,
            "wallTime": 1700000000.0,
            "type": "XHR",
            "request": {"url": "http://site.test/api/data.json", "method": "GET"},
        }
    )
    trace = rec._finalize_trace()
    assert trace[0]["is_document_load"] is False
    assert trace[0]["resource_type"] == "XHR"


def test_redirect_chain_preserves_hops(tmp_path):
    rec = _recorder(tmp_path)
    # First request: /short
    rec._on_request_will_be_sent(
        {
            "requestId": "req-1",
            "timestamp": 1.0,
            "wallTime": 1700000000.0,
            "type": "Document",
            "request": {"url": "http://site.test/short", "method": "GET"},
        }
    )
    # 302 redirect to /medium
    rec._on_request_will_be_sent(
        {
            "requestId": "req-1",
            "timestamp": 1.05,
            "wallTime": 1700000000.05,
            "type": "Document",
            "redirectResponse": {"status": 302, "url": "http://site.test/short"},
            "request": {"url": "http://site.test/medium", "method": "GET"},
        }
    )
    # Another 301 to final
    rec._on_request_will_be_sent(
        {
            "requestId": "req-1",
            "timestamp": 1.10,
            "wallTime": 1700000000.10,
            "type": "Document",
            "redirectResponse": {"status": 301, "url": "http://site.test/medium"},
            "request": {
                "url": "http://site.test/owner/repo/-/issues/42",
                "method": "GET",
            },
        }
    )
    trace = rec._finalize_trace()
    assert len(trace) == 1
    entry = trace[0]
    assert entry["url"] == "http://site.test/owner/repo/-/issues/42"
    assert entry["redirect_chain"] == [
        {"url": "http://site.test/short", "status": 302},
        {"url": "http://site.test/medium", "status": 301},
    ]


def test_frame_navigated_records_pageref(tmp_path):
    rec = _recorder(tmp_path)
    rec._on_frame_navigated(
        {
            "frame": {"url": "http://site.test/login", "parentId": None},
            "type": "Navigation",
        }
    )
    rec._on_frame_navigated(
        {
            "frame": {"url": "http://site.test/dashboard", "parentId": None},
            "type": "Navigation",
        }
    )
    # Sub-frame nav must be ignored.
    rec._on_frame_navigated(
        {
            "frame": {"url": "http://tracker.test/pixel", "parentId": "frame-1"},
            "type": "Navigation",
        }
    )
    docs = [n for n in rec._nav_events if n.get("kind") == "document"]
    assert len(docs) == 2
    assert docs[0]["pageref"] == "page_1"
    assert docs[1]["pageref"] == "page_2"
    assert docs[0]["url"] == "http://site.test/login"


def test_navigated_within_document_records_spa_hop(tmp_path):
    rec = _recorder(tmp_path)
    rec._on_navigated_within_document(
        {
            "url": "http://site.test/app#/users",
            "navigationType": "fragment",
        }
    )
    rec._on_navigated_within_document(
        {
            "url": "http://site.test/app/settings",
            "navigationType": "historyApi",
        }
    )
    within = [n for n in rec._nav_events if n.get("kind") == "within_document"]
    assert len(within) == 2
    assert within[0]["url"] == "http://site.test/app#/users"
    assert within[1]["navigation_type"] == "historyApi"
    # Within-doc events do NOT get pagerefs — only full document loads do.
    assert "pageref" not in within[0]


def test_write_trace_emits_redacted_navigation_trace_json(tmp_path):
    rec = _recorder(tmp_path)
    rec._on_frame_navigated(
        {
            "frame": {
                "url": "http://site.test/issues/42?token=secret#frag",
                "parentId": None,
            },
            "type": "Navigation",
        }
    )
    rec._on_navigated_within_document(
        {
            "url": "http://site.test/issues/42?sid=secret#note_1",
            "navigationType": "fragment",
        }
    )
    rec._write_trace([])
    persisted = json.loads((tmp_path / "navigation_trace.json").read_text())
    assert len(persisted) == 2
    kinds = [entry["kind"] for entry in persisted]
    assert kinds == ["document", "within_document"]
    assert persisted[0]["url"] == "http://site.test/issues/42?token=%3Credacted%3E"
    assert persisted[1]["url"] == "http://site.test/issues/42?sid=%3Credacted%3E"

    har = json.loads((tmp_path / "network.har").read_text())
    assert har["log"]["pages"][0]["title"] == ("http://site.test/issues/42?token=%3Credacted%3E")


def test_finalize_trace_assigns_pageref_to_entries(tmp_path):
    rec = _recorder(tmp_path)
    # Inject nav events with deterministic timestamps so pageref assignment
    # doesn't race with the test's wall clock.
    rec._nav_seq = 2
    rec._nav_events.extend(
        [
            {
                "url": "http://site.test/home",
                "navigation_type": "Navigation",
                "timestamp": 1000.0,
                "kind": "document",
                "pageref": "page_1",
            },
            {
                "url": "http://site.test/issues/42",
                "navigation_type": "Navigation",
                "timestamp": 2000.0,
                "kind": "document",
                "pageref": "page_2",
            },
        ]
    )
    rec._on_request_will_be_sent(
        {
            "requestId": "req-1",
            "timestamp": 1.0,
            "wallTime": 1500.0,  # between page_1 and page_2
            "type": "XHR",
            "request": {"url": "http://site.test/api/home", "method": "GET"},
        }
    )
    rec._on_request_will_be_sent(
        {
            "requestId": "req-2",
            "timestamp": 2.0,
            "wallTime": 2500.0,  # after page_2
            "type": "Document",
            "request": {"url": "http://site.test/issues/42", "method": "GET"},
        }
    )
    trace = rec._finalize_trace()
    assert len(trace) == 2
    assert trace[0]["pageref"] == "page_1"
    assert trace[1]["pageref"] == "page_2"
    assert trace[1]["is_document_load"] is True


def test_write_trace_populates_har_pages_and_pageref(tmp_path):
    rec = _recorder(tmp_path)
    rec._nav_seq = 1
    rec._nav_events.append(
        {
            "url": "http://site.test/issues/42",
            "navigation_type": "Navigation",
            "timestamp": 1000.0,
            "kind": "document",
            "pageref": "page_1",
        }
    )
    rec._on_request_will_be_sent(
        {
            "requestId": "req-1",
            "timestamp": 1.0,
            "wallTime": 1000.5,
            "type": "Document",
            "request": {"url": "http://site.test/issues/42", "method": "GET"},
        }
    )
    rec._on_response_received(
        {
            "requestId": "req-1",
            "response": {"status": 200, "mimeType": "text/html", "headers": {}},
        }
    )
    trace = rec._finalize_trace()
    rec._write_trace(trace)

    har = json.loads((tmp_path / "network.har").read_text())
    pages = har["log"]["pages"]
    assert len(pages) == 1
    assert pages[0]["id"] == "page_1"
    assert pages[0]["title"] == "http://site.test/issues/42"
    entries = har["log"]["entries"]
    assert len(entries) == 1
    assert entries[0]["pageref"] == "page_1"


def test_sec_fetch_headers_survive_redaction():
    headers = {
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Authorization": "Bearer secret",
    }
    redacted = _NetworkTraceRecorder._redact_headers(headers)
    assert redacted["Sec-Fetch-Dest"] == "document"
    assert redacted["Sec-Fetch-Mode"] == "navigate"
    assert redacted["Sec-Fetch-Site"] == "same-origin"
    assert redacted["Authorization"] == "<redacted>"


def test_is_document_load_on_empty_type(tmp_path):
    rec = _recorder(tmp_path)
    rec._on_request_will_be_sent(
        {
            "requestId": "req-1",
            "timestamp": 1.0,
            "wallTime": 1700000000.0,
            "request": {"url": "http://site.test/", "method": "GET"},
            # No `type` — some CDP events omit it.
        }
    )
    trace = rec._finalize_trace()
    assert trace[0]["is_document_load"] is False
    assert trace[0]["resource_type"] is None
