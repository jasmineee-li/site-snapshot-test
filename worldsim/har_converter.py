"""Convert flat CDP-derived trace entries to HAR entries the vendor can parse.

``worldsim._NetworkTraceRecorder`` captures network events in a compact
flat shape (``url``, ``method``, ``response_status``, etc. at the top
level). WebArena Verified's ``NetworkEvent`` reads HAR entries with
nested ``request``/``response`` objects and header/cookie lists of
``{name, value}`` pairs. Feeding flat events straight into the vendor
raises ``"Unknown trace format"``; feeding ``[]`` raises ``"Trace
content list is empty"``. Both cause the evaluator to short-circuit
before any evaluator runs, silently zeroing every score.

This module centralizes the conversion + placeholder insertion so the
live scoring path, the rescore CLI, and the on-disk ``network.har``
artifact all emit the same HAR shape.
"""

from __future__ import annotations

from typing import Any


def _headers_to_har(headers: Any) -> list[dict[str, str]]:
    """Flat ``{k: v}`` mapping -> HAR ``[{"name": k, "value": v}, ...]``."""
    if not headers:
        return []
    if isinstance(headers, list):
        return [
            {"name": str(h.get("name", "")), "value": str(h.get("value", ""))}
            for h in headers
            if isinstance(h, dict)
        ]
    if isinstance(headers, dict):
        return [{"name": str(k), "value": str(v)} for k, v in headers.items()]
    return []


def _cookies_to_har(cookies: Any) -> list[dict[str, str]]:
    """Accept dict / list of pairs / list of dicts; emit HAR cookie list."""
    if not cookies:
        return []
    if isinstance(cookies, list):
        out: list[dict[str, str]] = []
        for c in cookies:
            if isinstance(c, dict):
                out.append({"name": str(c.get("name", "")), "value": str(c.get("value", ""))})
            elif isinstance(c, (list, tuple)) and len(c) == 2:
                out.append({"name": str(c[0]), "value": str(c[1])})
        return out
    if isinstance(cookies, dict):
        return [{"name": str(k), "value": str(v)} for k, v in cookies.items()]
    return []


def _post_data_to_har(post_data: Any, request_headers: Any) -> dict[str, str] | None:
    """Wrap raw post-data text into HAR ``{mimeType, text}`` shape."""
    if post_data is None:
        return None
    if isinstance(post_data, str):
        text = post_data
    else:
        text = str(post_data)
    if not text or text == "<redacted>":
        return None
    mime = ""
    if isinstance(request_headers, dict):
        for key, value in request_headers.items():
            if str(key).lower() == "content-type":
                mime = str(value)
                break
    elif isinstance(request_headers, list):
        for h in request_headers:
            if isinstance(h, dict) and str(h.get("name", "")).lower() == "content-type":
                mime = str(h.get("value", ""))
                break
    return {"mimeType": mime, "text": text}


def flat_event_to_har_entry(flat: dict[str, Any]) -> dict[str, Any]:
    """Translate one flat ``_NetworkTraceRecorder`` entry to a HAR entry."""
    request = {
        "method": str(flat.get("method", "")),
        "url": str(flat.get("url", "")),
        "httpVersion": "HTTP/1.1",
        "headers": _headers_to_har(flat.get("headers")),
        "cookies": [],
        "queryString": [],
        "headersSize": -1,
        "bodySize": -1,
    }
    post = _post_data_to_har(flat.get("post_data"), flat.get("headers"))
    if post is not None:
        request["postData"] = post
    response = {
        "status": int(flat.get("response_status") or 0),
        "statusText": "",
        "httpVersion": "HTTP/1.1",
        "headers": _headers_to_har(flat.get("response_headers")),
        "cookies": _cookies_to_har(flat.get("response_cookies")),
        "content": {"size": 0, "mimeType": "", "text": ""},
        "redirectURL": "",
        "headersSize": -1,
        "bodySize": -1,
    }
    return {
        "startedDateTime": "1970-01-01T00:00:00Z",
        "time": 0,
        "request": request,
        "response": response,
        "cache": {},
        "timings": {"send": 0, "wait": 0, "receive": 0},
    }


def flat_events_to_har_entries(events: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Map a list of flat events to HAR entries, skipping non-dicts."""
    if not events:
        return []
    return [flat_event_to_har_entry(e) for e in events if isinstance(e, dict)]


def minimal_har_placeholder_entry() -> dict[str, Any]:
    """Single HAR entry that satisfies the vendor parser but matches nothing.

    Used when an ``AgentResponseEvaluator``-only task has no trace to
    submit. ``NetworkEventEvaluator`` would simply not match anything
    against this, which is the correct behavior for tasks that do not
    need a trace.
    """
    return {
        "startedDateTime": "1970-01-01T00:00:00Z",
        "time": 0,
        "request": {
            "method": "GET",
            "url": "about:blank",
            "httpVersion": "HTTP/1.1",
            "headers": [],
            "cookies": [],
            "queryString": [],
            "headersSize": -1,
            "bodySize": -1,
        },
        "response": {
            "status": 0,
            "statusText": "",
            "httpVersion": "HTTP/1.1",
            "headers": [],
            "cookies": [],
            "content": {"size": 0, "mimeType": "", "text": ""},
            "redirectURL": "",
            "headersSize": -1,
            "bodySize": -1,
        },
        "cache": {},
        "timings": {"send": 0, "wait": 0, "receive": 0},
    }


def ensure_har_trace(
    network_trace: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Normalize an input trace into a non-empty HAR entry list.

    Accepts either HAR entries (already contain ``request`` and
    ``response``) or flat ``_NetworkTraceRecorder`` events. Returns a
    list guaranteed to be non-empty so the vendor parser does not
    reject it.
    """
    if not network_trace:
        return [minimal_har_placeholder_entry()]
    first = network_trace[0]
    if isinstance(first, dict) and "request" in first and "response" in first:
        return list(network_trace)
    converted = flat_events_to_har_entries(network_trace)
    return converted if converted else [minimal_har_placeholder_entry()]
