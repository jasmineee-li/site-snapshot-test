from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

_SENSITIVE_HEADERS = {"authorization", "cookie", "proxy-authorization", "x-csrf-token"}


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
            "request": {
                "url": _safe_call(lambda: request.url) or "",
                "method": str(_safe_call(lambda: request.method) or "GET").upper(),
                "headers": headers,
                "postData": post_data or "",
            },
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
        event["response"] = {
            "status": status,
            "headers": response_headers,
            "cookies": event["response_cookies"],
        }

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
        out[name] = "<redacted>" if name in _SENSITIVE_HEADERS else str(value)
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
            name, value = first.split("=", 1)
            cookies.append({"name": name.strip(), "value": value.strip()})
    return cookies


def _as_har(events: list[dict[str, Any]], *, started_at: float) -> dict[str, Any]:
    entries = []
    for event in events:
        entries.append(
            {
                "startedDateTime": event.get("timestamp", started_at),
                "time": 0,
                "request": {
                    "method": event.get("method", "GET"),
                    "url": event.get("url", ""),
                    "headers": _har_headers(event.get("request_headers")),
                    "queryString": [
                        {"name": key, "value": value}
                        for key, values in (event.get("query_params") or {}).items()
                        for value in values
                    ],
                    "postData": {"text": event.get("post_data", "")},
                },
                "response": {
                    "status": event.get("response_status") or 0,
                    "headers": _har_headers(event.get("response_headers")),
                    "cookies": event.get("response_cookies") or [],
                },
            }
        )
    return {"log": {"version": "1.2", "creator": {"name": "worldsim-agentlab"}, "entries": entries}}


def _har_headers(headers: Any) -> list[dict[str, str]]:
    if not isinstance(headers, dict):
        return []
    return [{"name": str(key), "value": str(value)} for key, value in headers.items()]
