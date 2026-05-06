from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

_SENSITIVE_HEADERS = {
    "authorization",
    "cookie",
    "set-cookie",
    "proxy-authorization",
    "x-api-key",
    "x-auth-token",
    "x-csrf-token",
    "x-csrftoken",
}
_SENSITIVE_HEADER_SUBSTRINGS = ("token", "secret", "session", "auth", "login", "cookie", "csrf", "key")


class NetworkTraceRecorder:
    """Small Playwright-sync network recorder for WorldSim reward evaluators."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self._events_by_request: dict[int, dict[str, Any]] = {}
        self._ordered_ids: list[int] = []
        self._started_at = time.time()
        self._attached_context_ids: set[int] = set()

    def attach(self, context: Any) -> None:
        context_id = id(context)
        if context_id in self._attached_context_ids:
            return
        self._attached_context_ids.add(context_id)
        context.on("request", self._on_request)
        context.on("response", self._on_response)
        context.on("requestfailed", self._on_request_failed)

    @property
    def events(self) -> list[dict[str, Any]]:
        return [self._events_by_request[key] for key in self._ordered_ids]

    def persist(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        events = self.events
        (self.output_dir / "network_trace.json").write_text(
            json.dumps(events, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (self.output_dir / "network.har").write_text(
            json.dumps(_as_har(events, started_at=self._started_at), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        navigation = [
            {
                "url": event.get("url"),
                "method": event.get("method"),
                "response_status": event.get("response_status"),
                "timestamp": event.get("timestamp"),
            }
            for event in events
            if _is_navigation_like(event)
        ]
        (self.output_dir / "navigation_trace.json").write_text(
            json.dumps(navigation, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _on_request(self, request: Any) -> None:
        key = id(request)
        headers = _redact_headers(_safe_call(lambda: request.headers) or {})
        post_data = _safe_call(lambda: request.post_data)
        event = {
            "url": _safe_call(lambda: request.url) or "",
            "method": str(_safe_call(lambda: request.method) or "GET").upper(),
            "request_headers": headers,
            "headers": headers,
            "post_data": post_data or "",
            "query_params": _query_params(_safe_call(lambda: request.url) or ""),
            "resource_type": _safe_call(lambda: request.resource_type) or "",
            "is_navigation_request": bool(_safe_call(lambda: request.is_navigation_request())),
            "timestamp": time.time(),
        }
        self._events_by_request[key] = event
        self._ordered_ids.append(key)

    def _on_response(self, response: Any) -> None:
        request = _safe_call(lambda: response.request)
        if request is None:
            return
        event = self._events_by_request.get(id(request))
        if event is None:
            return
        response_headers = _redact_headers(_safe_call(lambda: response.headers) or {})
        status = _safe_call(lambda: response.status)
        event["response_status"] = status
        event["response_headers"] = response_headers
        event["response_cookies"] = _cookies_from_headers(response_headers)

    def _on_request_failed(self, request: Any) -> None:
        event = self._events_by_request.get(id(request))
        if event is None:
            return
        failure = _safe_call(lambda: request.failure) or {}
        event["failure"] = failure


def _safe_call(fn: Any) -> Any:
    try:
        value = fn()
        return value() if callable(value) and not isinstance(value, (str, bytes, dict, list)) else value
    except Exception:
        return None


def _redact_headers(headers: Any) -> dict[str, str]:
    if not isinstance(headers, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in headers.items():
        name = str(key).lower()
        out[name] = (
            "<redacted>"
            if name in _SENSITIVE_HEADERS or any(marker in name for marker in _SENSITIVE_HEADER_SUBSTRINGS)
            else str(value)
        )
    return out


def _is_navigation_like(event: dict[str, Any]) -> bool:
    return bool(event.get("is_navigation_request")) or str(event.get("resource_type")) == "document"


def _query_params(url: str) -> dict[str, list[str]]:
    try:
        return {str(k): [str(item) for item in v] for k, v in parse_qs(urlsplit(url).query).items()}
    except Exception:
        return {}


def _cookies_from_headers(headers: dict[str, str]) -> list[dict[str, str]]:
    raw = headers.get("set-cookie")
    if not raw or raw == "<redacted>":
        return []
    cookies = []
    for chunk in raw.split(","):
        first = chunk.split(";", 1)[0]
        if "=" in first:
            name, _value = first.split("=", 1)
            cookies.append({"name": name.strip(), "value": "<redacted>"})
    return cookies


def _as_har(events: list[dict[str, Any]], *, started_at: float) -> dict[str, Any]:
    entries = []
    for event in events:
        entries.append(
            {
                "startedDateTime": _har_datetime(event.get("timestamp", started_at)),
                "time": 0,
                "request": {
                    "method": event.get("method", "GET"),
                    "url": event.get("url", ""),
                    "httpVersion": "HTTP/1.1",
                    "headers": _har_headers(event.get("request_headers")),
                    "cookies": [],
                    "queryString": [
                        {"name": key, "value": value}
                        for key, values in (event.get("query_params") or {}).items()
                        for value in values
                    ],
                    "headersSize": -1,
                    "bodySize": -1,
                    **_har_post_data(event),
                },
                "response": {
                    "status": event.get("response_status") or 0,
                    "statusText": "",
                    "httpVersion": "HTTP/1.1",
                    "headers": _har_headers(event.get("response_headers")),
                    "cookies": event.get("response_cookies") or [],
                    "content": {"size": 0, "mimeType": "", "text": ""},
                    "redirectURL": "",
                    "headersSize": -1,
                    "bodySize": -1,
                },
                "cache": {},
                "timings": {"send": 0, "wait": 0, "receive": 0},
            }
        )
    return {"log": {"version": "1.2", "creator": {"name": "worldsim-agentlab"}, "entries": entries}}


def _har_headers(headers: Any) -> list[dict[str, str]]:
    if not isinstance(headers, dict):
        return []
    return [{"name": str(key), "value": str(value)} for key, value in headers.items()]


def _har_post_data(event: dict[str, Any]) -> dict[str, Any]:
    text = event.get("post_data")
    if not isinstance(text, str) or not text or text == "<redacted>":
        return {}
    mime = ""
    headers = event.get("request_headers")
    if isinstance(headers, dict):
        for key, value in headers.items():
            if str(key).lower() == "content-type":
                mime = str(value)
                break
    return {"postData": {"mimeType": mime, "text": text}}


def _har_datetime(value: Any) -> str:
    try:
        return datetime.fromtimestamp(float(value), tz=UTC).isoformat()
    except (TypeError, ValueError, OverflowError):
        return "1970-01-01T00:00:00+00:00"
