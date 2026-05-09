"""Convert flat CDP-derived trace entries to HAR entries the vendor can parse.

``worldsim._NetworkTraceRecorder`` captures network events in a compact
flat shape (``url``, ``method``, ``response_status``, etc. at the top
level). WebArena Verified's ``NetworkEvent`` reads HAR entries with
nested ``request``/``response`` objects and header/cookie lists of
``{name, value}`` pairs. Feeding flat events straight into the vendor
raises ``"Unknown trace format"``; feeding ``[]`` raises ``"Trace
content list is empty"``. Both cause the evaluator to short-circuit
before any evaluator runs, silently zeroing every score.

This module centralizes the conversion logic for runtime scoring and the
placeholder insertion used only by the rescore CLI's AgentResponse-only
compatibility path.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import parse_qs, urlsplit


class NetworkTraceUnavailableError(RuntimeError):
    """Raised when runtime evaluation has no usable network trace evidence."""


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


def _response_content_to_har(flat: dict[str, Any]) -> dict[str, Any]:
    text = flat.get("response_content")
    if not isinstance(text, str):
        text = ""
    mime = str(flat.get("response_mime_type") or "")
    headers = flat.get("response_headers")
    if not mime:
        if isinstance(headers, dict):
            for key, value in headers.items():
                if str(key).lower() == "content-type":
                    mime = str(value)
                    break
        elif isinstance(headers, list):
            for header in headers:
                if (
                    isinstance(header, dict)
                    and str(header.get("name", "")).lower() == "content-type"
                ):
                    mime = str(header.get("value", ""))
                    break
    return {"size": len(text), "mimeType": mime, "text": text}


def flat_event_to_har_entry(flat: dict[str, Any]) -> dict[str, Any]:
    """Translate one flat ``_NetworkTraceRecorder`` entry to a HAR entry."""
    url = str(flat.get("url", ""))
    request = {
        "method": str(flat.get("method", "")),
        "url": url,
        "httpVersion": "HTTP/1.1",
        "headers": _headers_to_har(flat.get("headers")),
        "cookies": [],
        "queryString": _query_string_to_har(flat.get("query_params"), url),
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
        "content": _response_content_to_har(flat),
        "redirectURL": "",
        "headersSize": -1,
        "bodySize": -1,
    }
    entry: dict[str, Any] = {
        "startedDateTime": "1970-01-01T00:00:00Z",
        "time": 0,
        "request": request,
        "response": response,
        "cache": {},
        "timings": {"send": 0, "wait": 0, "receive": 0},
    }
    pageref = flat.get("pageref")
    if pageref:
        entry["pageref"] = str(pageref)
    return entry


def _query_string_to_har(query_params: Any, url: str) -> list[dict[str, str]]:
    if isinstance(query_params, dict):
        out: list[dict[str, str]] = []
        for key, values in query_params.items():
            if isinstance(values, list):
                out.extend({"name": str(key), "value": str(value)} for value in values)
            else:
                out.append({"name": str(key), "value": str(values)})
        return out
    try:
        parsed = parse_qs(urlsplit(url).query, keep_blank_values=True)
    except Exception:
        return []
    return [
        {"name": str(key), "value": str(value)}
        for key, values in parsed.items()
        for value in values
    ]


def flat_events_to_har_entries(events: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Map a list of flat events to HAR entries, skipping non-dicts."""
    if not events:
        return []
    return [flat_event_to_har_entry(e) for e in events if isinstance(e, dict)]


def nav_events_to_har_pages(
    nav_events: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Build a HAR ``pages[]`` list from top-frame document navigations.

    ``kind=document`` entries emitted by ``_NetworkTraceRecorder._on_frame_navigated``
    carry a ``pageref`` and a wall-clock ``timestamp``. Within-document (SPA)
    nav events are preserved in ``navigation_trace.json`` but do not enter
    HAR pages[] — the HAR spec keys pages by full document loads.
    """
    if not nav_events:
        return []
    from datetime import UTC, datetime

    pages: list[dict[str, Any]] = []
    for nav in nav_events:
        if not isinstance(nav, dict):
            continue
        if nav.get("kind") != "document":
            continue
        pageref = nav.get("pageref")
        if not pageref:
            continue
        ts = nav.get("timestamp")
        try:
            started = (
                datetime.fromtimestamp(float(ts), tz=UTC).isoformat()
                if ts is not None
                else "1970-01-01T00:00:00+00:00"
            )
        except (TypeError, ValueError, OverflowError):
            started = "1970-01-01T00:00:00+00:00"
        pages.append(
            {
                "id": str(pageref),
                "startedDateTime": started,
                "title": str(nav.get("url") or ""),
                "pageTimings": {"onContentLoad": -1, "onLoad": -1},
            }
        )
    return pages


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


def strict_runtime_har_trace(
    network_trace: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Normalize runtime trace evidence into HAR entries or fail closed.

    Accepts either HAR entries (already contain ``request`` and
    ``response``) or flat ``_NetworkTraceRecorder`` events. Raises when
    runtime evaluation has no usable network evidence.
    """
    if not network_trace:
        raise NetworkTraceUnavailableError("network_trace_unavailable: trace is missing or empty")
    first = network_trace[0]
    if isinstance(first, dict) and "request" in first and "response" in first:
        entries = list(network_trace)
        if not entries:
            raise NetworkTraceUnavailableError("network_trace_unavailable: HAR trace is empty")
        return entries
    converted = flat_events_to_har_entries(network_trace)
    if not converted:
        raise NetworkTraceUnavailableError(
            "network_trace_unavailable: trace did not contain any valid network events"
        )
    if not _contains_real_http_evidence(converted):
        raise NetworkTraceUnavailableError(
            "network_trace_unavailable: trace did not contain real HTTP evidence"
        )
    return converted


def _contains_real_http_evidence(entries: list[dict[str, Any]]) -> bool:
    for entry in entries:
        request = entry.get("request") if isinstance(entry, dict) else None
        response = entry.get("response") if isinstance(entry, dict) else None
        if not isinstance(request, dict) or not isinstance(response, dict):
            continue
        url = request.get("url")
        method = request.get("method")
        status = response.get("status")
        if (
            isinstance(url, str)
            and url.startswith(("http://", "https://"))
            and isinstance(method, str)
            and bool(method.strip())
            and isinstance(status, int)
            and status != 0
        ):
            return True
    return False


def ensure_har_trace(
    network_trace: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Rescore-only helper that guarantees a non-empty HAR trace."""
    if not network_trace:
        return [minimal_har_placeholder_entry()]
    first = network_trace[0]
    if isinstance(first, dict) and "request" in first and "response" in first:
        return list(network_trace)
    converted = flat_events_to_har_entries(network_trace)
    return converted if converted else [minimal_har_placeholder_entry()]
