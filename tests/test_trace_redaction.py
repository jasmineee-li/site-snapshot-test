from __future__ import annotations

from worldsim.browser_use_agent import _build_initial_actions, _NetworkTraceRecorder


def test_trace_redaction_removes_sensitive_wire_data():
    entry = {
        "url": "http://example.test",
        "headers": {
            "Authorization": "Bearer secret",
            "Content-Type": "application/json",
        },
        "response_headers": {
            "Set-Cookie": "session=abc",
            "Cache-Control": "no-cache",
        },
        "post_data": '{"password": "secret"}',
        "response_cookies": {"session": "abc"},
    }

    redacted = _NetworkTraceRecorder._redact_trace_entry(entry)

    assert redacted["headers"]["Authorization"] == "<redacted>"
    assert redacted["headers"]["Content-Type"] == "application/json"
    assert redacted["response_headers"]["Set-Cookie"] == "<redacted>"
    assert redacted["response_headers"]["Cache-Control"] == "no-cache"
    assert redacted["post_data"] == "<redacted>"
    assert redacted["response_cookies"] == {"session": "<redacted>"}


def test_build_initial_actions_opens_start_urls_in_order():
    actions = _build_initial_actions(
        ["http://shopping.test", "http://gitlab.test", "http://shopping.test", ""]
    )

    assert actions == [
        {"navigate": {"url": "http://shopping.test", "new_tab": False}},
        {"navigate": {"url": "http://gitlab.test", "new_tab": True}},
    ]
