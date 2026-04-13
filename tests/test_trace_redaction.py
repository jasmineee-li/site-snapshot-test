from __future__ import annotations

from worldsim.browser_use_agent import _build_initial_actions, _NetworkTraceRecorder


def test_trace_redaction_removes_sensitive_wire_data():
    entry = {
        "url": "http://user:pass@example.test/reset/secret-token?token=abc&user=demo#frag",
        "headers": {
            "Authorization": "Bearer secret",
            "Content-Type": "application/json",
            "Referer": "https://source.test/private/path?code=123",
        },
        "query_params": {"token": ["abc"], "user": ["demo"]},
        "response_headers": {
            "Set-Cookie": "session=abc",
            "Cache-Control": "no-cache",
            "Location": "https://example.test/account/123?next=/dashboard",
        },
        "post_data": '{"password": "secret"}',
        "response_cookies": {"session": "abc"},
    }

    redacted = _NetworkTraceRecorder._redact_trace_entry(entry)

    assert redacted["url"] == (
        "http://example.test/<redacted>/<redacted>?token=%3Credacted%3E&user=%3Credacted%3E"
    )
    assert redacted["query_params"] == {
        "token": ["<redacted>"],
        "user": ["<redacted>"],
    }
    assert redacted["headers"]["Authorization"] == "<redacted>"
    assert redacted["headers"]["Content-Type"] == "application/json"
    assert redacted["headers"]["Referer"] == (
        "https://source.test/<redacted>/<redacted>?code=%3Credacted%3E"
    )
    assert redacted["response_headers"]["Set-Cookie"] == "<redacted>"
    assert redacted["response_headers"]["Cache-Control"] == "no-cache"
    assert redacted["response_headers"]["Location"] == (
        "https://example.test/<redacted>/<redacted>?next=%3Credacted%3E"
    )
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
