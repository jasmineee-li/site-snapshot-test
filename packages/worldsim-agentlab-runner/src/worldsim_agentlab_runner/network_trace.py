from __future__ import annotations

import json
import re
import time
from datetime import UTC, datetime
from http.cookies import SimpleCookie
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, parse_qsl, urlencode, urlsplit, urlunsplit

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
_SENSITIVE_HEADER_SUBSTRINGS = (
    "token",
    "secret",
    "session",
    "auth",
    "login",
    "cookie",
    "csrf",
    "key",
)
_SENSITIVE_FIELD_NAMES = {
    "password",
    "passwd",
    "secret",
    "csrf",
    "csrf_token",
    "authenticity_token",
    "access_token",
    "refresh_token",
    "id_token",
    "api_key",
    "apikey",
    "session",
    "session_id",
    "_session",
    "cookie",
}
_SENSITIVE_FIELD_SUBSTRINGS = (
    "password",
    "passwd",
    "secret",
    "csrf",
    "authenticity_token",
    "access_token",
    "refresh_token",
    "id_token",
    "api_key",
    "session",
)
_BODY_CAPTURE_CAP = 120_000


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

    @property
    def redacted_events(self) -> list[dict[str, Any]]:
        return [_redact_event(event) for event in self.events]

    def persist(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        raw_events = self.events
        events = self.redacted_events
        har = _as_har(events, started_at=self._started_at)
        validate_har_1_2_shape(har, require_real_entry=bool(events))
        (self.output_dir / "network_trace.json").write_text(
            json.dumps(events, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (self.output_dir / "network.har").write_text(
            json.dumps(har, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        private_dir = self.output_dir / "reward_private"
        private_dir.mkdir(parents=True, exist_ok=True)
        private_events_path = private_dir / "network_trace.json"
        private_har_path = private_dir / "network.har"
        raw_har = _as_har(raw_events, started_at=self._started_at)
        validate_har_1_2_shape(raw_har, require_real_entry=bool(raw_events))
        private_events_path.write_text(
            json.dumps(raw_events, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        private_har_path.write_text(
            json.dumps(raw_har, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        private_events_path.chmod(0o600)
        private_har_path.chmod(0o600)
        (self.output_dir / "network_evidence.json").write_text(
            json.dumps(
                {
                    "public_trace": "redacted",
                    "private_reward_trace": "available",
                    "private_reward_trace_dir": "reward_private",
                    "private_reward_trace_mode": "0600",
                },
                indent=2,
                sort_keys=True,
            ),
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
        headers = _string_map(_safe_call(lambda: request.headers) or {})
        url = _safe_call(lambda: request.url) or ""
        event = {
            "url": url,
            "method": str(_safe_call(lambda: request.method) or "GET").upper(),
            "request_headers": headers,
            "headers": headers,
            "post_data": _safe_call(lambda: request.post_data) or "",
            "query_params": _query_params(url),
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
        raw_response_headers = _safe_call(lambda: response.headers) or {}
        response_cookies = _cookies_from_response(response, raw_response_headers)
        status = _safe_call(lambda: response.status)
        event["response_status"] = status
        event["response_headers"] = _string_map(raw_response_headers)
        event["response_cookies"] = response_cookies
        event["response_content"] = _safe_call(lambda: response.text) or ""

    def _on_request_failed(self, request: Any) -> None:
        event = self._events_by_request.get(id(request))
        if event is None:
            return
        failure = _safe_call(lambda: request.failure) or {}
        event["failure"] = failure


def _safe_call(fn: Any) -> Any:
    try:
        value = fn()
        return (
            value()
            if callable(value) and not isinstance(value, (str, bytes, dict, list))
            else value
        )
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
            if name in _SENSITIVE_HEADERS
            or any(marker in name for marker in _SENSITIVE_HEADER_SUBSTRINGS)
            else str(value)
        )
    return out


def _string_map(headers: Any) -> dict[str, str]:
    if not isinstance(headers, dict):
        return {}
    return {str(key): str(value) for key, value in headers.items()}


def _redact_event(event: dict[str, Any]) -> dict[str, Any]:
    out = dict(event)
    url = str(out.get("url") or "")
    redacted_url, query_params = _redact_url_and_query(url)
    out["url"] = redacted_url
    out["query_params"] = query_params
    out["request_headers"] = _redact_headers(out.get("request_headers"))
    out["headers"] = _redact_headers(out.get("headers"))
    out["post_data"] = _redact_post_data(out.get("post_data"))
    out["response_headers"] = _redact_headers(out.get("response_headers"))
    out["response_cookies"] = _redact_cookie_values(out.get("response_cookies"))
    out["response_content"] = _redact_response_text(out.get("response_content"))
    return out


def _redact_post_data(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return ""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = None
    if parsed is not None:
        redacted = _redact_sensitive_fields(parsed)
        if redacted != parsed:
            return json.dumps(redacted, sort_keys=True, separators=(",", ":"))
        return value
    form_pairs = parse_qsl(value, keep_blank_values=True)
    if form_pairs and urlencode(form_pairs) == value.replace(" ", "+"):
        return urlencode(
            [
                (key, "<redacted>" if _is_sensitive_field_name(key) else item)
                for key, item in form_pairs
            ]
        )
    if re.search(
        r"(?i)(password|passwd|secret|csrf|authenticity_token|access_token|refresh_token|id_token|api[_-]?key|session)\s*[:=]",
        value,
    ):
        return "<redacted>"
    return value


def _is_navigation_like(event: dict[str, Any]) -> bool:
    return bool(event.get("is_navigation_request")) or str(event.get("resource_type")) == "document"


def _query_params(url: str) -> dict[str, list[str]]:
    try:
        return {str(k): [str(item) for item in v] for k, v in parse_qs(urlsplit(url).query).items()}
    except Exception:
        return {}


def _redact_url_and_query(url: str) -> tuple[str, dict[str, list[str]]]:
    try:
        parts = urlsplit(url)
        pairs = parse_qsl(parts.query, keep_blank_values=True)
    except Exception:
        return url, {}
    redacted_pairs = [
        (key, "<redacted>" if _is_sensitive_field_name(key) else value) for key, value in pairs
    ]
    query: dict[str, list[str]] = {}
    for key, value in redacted_pairs:
        query.setdefault(str(key), []).append(str(value))
    redacted_url = urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path,
            urlencode(redacted_pairs, doseq=True),
            parts.fragment,
        )
    )
    return redacted_url, query


def _redact_response_text(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return ""
    return _redact_post_data(value[:_BODY_CAPTURE_CAP])


def _redact_sensitive_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): "<redacted>"
            if _is_sensitive_field_name(str(key))
            else _redact_sensitive_fields(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_sensitive_fields(item) for item in value]
    return value


def _is_sensitive_field_name(name: str) -> bool:
    normalized = name.strip().lower().replace("-", "_")
    if normalized in _SENSITIVE_FIELD_NAMES:
        return True
    # Do not redact generic benchmark evidence like "ticket", "issue_key", or
    # task-local "token" unless the field name clearly denotes credentials.
    return any(marker in normalized for marker in _SENSITIVE_FIELD_SUBSTRINGS)


def _cookies_from_response(response: Any, fallback_headers: dict[str, str]) -> list[dict[str, str]]:
    raw_headers = _safe_call(lambda: response.headers_array)
    cookie_headers: list[str] = []
    if isinstance(raw_headers, list):
        for header in raw_headers:
            if not isinstance(header, dict):
                continue
            if str(header.get("name") or "").lower() == "set-cookie":
                value = header.get("value")
                if isinstance(value, str) and value:
                    cookie_headers.append(value)
    if not cookie_headers:
        cookie_headers = _set_cookie_header_values(fallback_headers)
    cookies: list[dict[str, str]] = []
    for raw in cookie_headers:
        cookies.extend(_cookies_from_set_cookie_value(raw))
    return cookies


def _set_cookie_header_values(headers: dict[str, str]) -> list[str]:
    if isinstance(headers, dict):
        for key, value in headers.items():
            if str(key).lower() == "set-cookie":
                raw = str(value)
                if raw and raw != "<redacted>":
                    return _split_combined_set_cookie_header(raw)
    return []


def _split_combined_set_cookie_header(raw: str) -> list[str]:
    parts: list[str] = []
    start = 0
    index = 0
    in_expires = False
    while index < len(raw):
        lower_tail = raw[index : index + 8].lower()
        if lower_tail == "expires=":
            in_expires = True
            index += 8
            continue
        char = raw[index]
        if in_expires and char == ";":
            in_expires = False
        if char == "," and not in_expires:
            parts.append(raw[start:index].strip())
            start = index + 1
        index += 1
    parts.append(raw[start:].strip())
    return [part for part in parts if part]


def _cookies_from_set_cookie_value(raw: str) -> list[dict[str, str]]:
    if not raw or raw == "<redacted>":
        return []
    cookies: list[dict[str, str]] = []
    parsed = SimpleCookie()
    try:
        parsed.load(raw)
    except Exception:
        parsed = SimpleCookie()
    for name, morsel in parsed.items():
        cookies.append({"name": str(name), "value": str(morsel.value)})
    if cookies:
        return cookies
    for match in re.finditer(r"(?:^|,\s*)([^=;,\s]+)=", raw):
        cookies.append({"name": match.group(1).strip(), "value": ""})
    return cookies


def _redact_cookie_values(cookies: Any) -> list[dict[str, str]]:
    if not isinstance(cookies, list):
        return []
    out: list[dict[str, str]] = []
    for cookie in cookies:
        if isinstance(cookie, dict):
            name = str(cookie.get("name") or "")
            if name:
                out.append({"name": name, "value": "<redacted>"})
    return out


def validate_har_1_2_shape(har: dict[str, Any], *, require_real_entry: bool = False) -> None:
    log = har.get("log") if isinstance(har, dict) else None
    if not isinstance(log, dict):
        raise ValueError("HAR missing log object")
    if log.get("version") != "1.2":
        raise ValueError("HAR log.version must be 1.2")
    creator = log.get("creator")
    if not isinstance(creator, dict) or not isinstance(creator.get("name"), str):
        raise ValueError("HAR missing creator.name")
    entries = log.get("entries")
    if not isinstance(entries, list):
        raise ValueError("HAR log.entries must be a list")
    real_entries = 0
    nonzero_statuses = 0
    for entry in entries:
        _validate_har_entry(entry)
        request = entry["request"]
        response = entry["response"]
        url = request["url"]
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            real_entries += 1
        if isinstance(response.get("status"), int) and response["status"] != 0:
            nonzero_statuses += 1
    if require_real_entry and (real_entries == 0 or nonzero_statuses == 0):
        raise ValueError("HAR completed trace must include real HTTP evidence")


def _validate_har_entry(entry: Any) -> None:
    if not isinstance(entry, dict):
        raise ValueError("HAR entry must be an object")
    request = entry.get("request")
    response = entry.get("response")
    if not isinstance(request, dict) or not isinstance(response, dict):
        raise ValueError("HAR entry must contain request and response objects")
    for key in ("url", "method", "httpVersion"):
        if not isinstance(request.get(key), str) or not request[key]:
            raise ValueError(f"HAR request.{key} must be a non-empty string")
    for key in ("headers", "cookies", "queryString"):
        _validate_name_value_list(request.get(key), f"HAR request.{key}")
    if not isinstance(response.get("status"), int):
        raise ValueError("HAR response.status must be an int")
    for key in ("statusText", "httpVersion", "redirectURL"):
        if not isinstance(response.get(key), str):
            raise ValueError(f"HAR response.{key} must be a string")
    for key in ("headers", "cookies"):
        _validate_name_value_list(response.get(key), f"HAR response.{key}")
    content = response.get("content")
    if not isinstance(content, dict):
        raise ValueError("HAR response.content must be an object")
    post_data = request.get("postData")
    if post_data is not None:
        if not isinstance(post_data, dict):
            raise ValueError("HAR request.postData must be an object")
        if not isinstance(post_data.get("mimeType"), str) or not isinstance(
            post_data.get("text"), str
        ):
            raise ValueError("HAR request.postData must contain string mimeType and text")


def _validate_name_value_list(value: Any, label: str) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"{label} entries must be objects")
        if not isinstance(item.get("name"), str) or not isinstance(item.get("value"), str):
            raise ValueError(f"{label} entries must contain string name and value")


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
                    "content": _har_response_content(event),
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
    if not isinstance(text, str) or not text:
        return {}
    mime = ""
    headers = event.get("request_headers")
    if isinstance(headers, dict):
        for key, value in headers.items():
            if str(key).lower() == "content-type":
                mime = str(value)
                break
    return {"postData": {"mimeType": mime, "text": text}}


def _har_response_content(event: dict[str, Any]) -> dict[str, Any]:
    text = event.get("response_content")
    if not isinstance(text, str):
        text = ""
    mime = ""
    headers = event.get("response_headers")
    if isinstance(headers, dict):
        for key, value in headers.items():
            if str(key).lower() == "content-type":
                mime = str(value)
                break
    return {"size": len(text), "mimeType": mime, "text": text}


def _har_datetime(value: Any) -> str:
    try:
        return datetime.fromtimestamp(float(value), tz=UTC).isoformat()
    except (TypeError, ValueError, OverflowError):
        return "1970-01-01T00:00:00+00:00"
