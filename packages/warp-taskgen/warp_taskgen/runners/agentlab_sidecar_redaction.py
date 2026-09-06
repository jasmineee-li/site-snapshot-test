"""Redaction of AgentLab sidecar payloads, logs, status files, and traces.

Every value the runner writes beside a trajectory passes through this module so
request auth material, cookies, and echoed secrets never reach a sidecar log,
status file, or persisted result.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


def _redact_sidecar_payload(value: Any, *, secret_values: set[str] | None = None) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            lower = str(key).lower()
            if lower == "storage_state":
                redacted[key] = {"present": bool(item), "runtime_only": True}
            elif lower == "storage_state_runtime_dir":
                redacted[key] = "<runtime-only>"
            elif lower == "network_trace" and isinstance(item, list):
                redacted[key] = [
                    _redact_network_event(event, secret_values=secret_values) for event in item
                ]
            elif lower in {"authorization", "cookie", "set-cookie", "proxy-authorization"} or any(
                marker in lower
                for marker in ("token", "secret", "password", "auth", "cookie", "csrf", "key")
            ):
                redacted[key] = "<redacted>"
            elif lower == "headers" and isinstance(item, dict):
                redacted[key] = _redact_sidecar_headers(item)
            else:
                redacted[key] = _redact_sidecar_payload(item, secret_values=secret_values)
        return redacted
    if isinstance(value, list):
        return [_redact_sidecar_payload(item, secret_values=secret_values) for item in value]
    if isinstance(value, str) and secret_values:
        return _redact_text_values(value, secret_values)
    return value


def _redact_network_event(
    value: Any,
    *,
    secret_values: set[str] | None = None,
) -> Any:
    if not isinstance(value, dict):
        return _redact_sidecar_payload(value, secret_values=secret_values)
    event: dict[str, Any] = {}
    for key, item in value.items():
        lower = str(key).lower()
        if lower in {"url"} and isinstance(item, str):
            # AgentLab network traces intentionally keep benchmark request
            # payloads visible. These are controlled local benchmark actions,
            # and ASR debugging needs the exact source-action URL/query/body
            # to distinguish missing writes, duplicate transports, and wrong
            # anchors. Auth headers/cookies and explicit secret echoes are
            # still redacted below.
            event[key] = item
        elif lower == "query_params" and isinstance(item, dict):
            event[key] = item
        elif lower == "post_data":
            event[key] = item
        elif lower == "response_content":
            event[key] = _redact_network_body(item, secret_values=secret_values)
        elif lower in {"request_headers", "headers", "response_headers"} and isinstance(item, dict):
            event[key] = _redact_sidecar_headers(item)
        elif lower == "response_cookies" and isinstance(item, list):
            event[key] = [
                {"name": str(cookie.get("name") or ""), "value": "<redacted>"}
                for cookie in item
                if isinstance(cookie, dict)
            ]
        else:
            event[key] = _redact_sidecar_payload(item, secret_values=secret_values)
    return event


def _redact_url_value(value: str, *, secret_values: set[str] | None = None) -> str:
    parsed = urlparse(value)
    pairs = parse_qsl(parsed.query, keep_blank_values=True)
    if not pairs:
        return _redact_text_values(value, secret_values or set())
    redacted_pairs = [
        (
            key,
            "<redacted>"
            if _is_sensitive_network_field(key)
            else _redact_text_values(val, secret_values or set()),
        )
        for key, val in pairs
    ]
    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            urlencode(redacted_pairs, doseq=True),
            parsed.fragment,
        )
    )


def _redact_network_body(value: Any, *, secret_values: set[str] | None = None) -> Any:
    if not isinstance(value, str) or not value:
        return value
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = None
    if parsed is not None:
        redacted = _redact_network_json(parsed, secret_values=secret_values)
        return json.dumps(redacted, sort_keys=True, separators=(",", ":"))
    pairs = parse_qsl(value, keep_blank_values=True)
    if pairs and urlencode(pairs) == value.replace(" ", "+"):
        return urlencode(
            [
                (
                    key,
                    _redact_network_scalar(key, item, secret_values=secret_values),
                )
                for key, item in pairs
            ]
        )
    return _redact_text_values(value, secret_values or set())


def _redact_network_json(value: Any, *, secret_values: set[str] | None = None) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _redact_network_scalar(str(key), item, secret_values=secret_values)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_network_json(item, secret_values=secret_values) for item in value]
    if isinstance(value, str):
        return _redact_text_values(value, secret_values or set())
    return value


def _redact_network_scalar(
    field_name: str, value: Any, *, secret_values: set[str] | None = None
) -> Any:
    if _is_sensitive_network_field(field_name):
        return "<redacted>"
    if isinstance(value, str):
        return _redact_text_values(value, secret_values or set())
    return _redact_network_json(value, secret_values=secret_values)


def _is_sensitive_network_field(name: str) -> bool:
    normalized = name.strip().lower().replace("-", "_")
    return normalized in {
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
    } or any(
        marker in normalized
        for marker in (
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
    )


def _secret_strings_from_payload(value: Any) -> set[str]:
    secrets: set[str] = set()

    def visit(item: Any, *, sensitive: bool = False) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                lower = str(key).lower()
                child_sensitive = (
                    sensitive
                    or lower
                    in {
                        "authorization",
                        "cookie",
                        "set-cookie",
                        "proxy-authorization",
                    }
                    or any(
                        marker in lower
                        for marker in (
                            "token",
                            "secret",
                            "password",
                            "auth",
                            "cookie",
                            "csrf",
                            "key",
                        )
                    )
                )
                visit(child, sensitive=child_sensitive)
            return
        if isinstance(item, list):
            for child in item:
                visit(child, sensitive=sensitive)
            return
        if sensitive and isinstance(item, str) and item:
            secrets.add(item)

    visit(value)
    return secrets


def _redact_sidecar_text(text: str, request: dict[str, Any]) -> str:
    return _redact_text_values(text, _secret_strings_from_payload(request))


def _redact_text_values(text: str, secret_values: set[str]) -> str:
    redacted = text
    for secret in sorted(secret_values, key=len, reverse=True):
        if len(secret) >= 4:
            redacted = redacted.replace(secret, "<redacted>")
    return redacted


def _redact_sidecar_headers(headers: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, item in headers.items():
        lower = str(key).lower()
        if lower in {"authorization", "cookie", "set-cookie", "proxy-authorization"} or any(
            marker in lower
            for marker in ("token", "secret", "session", "auth", "cookie", "csrf", "key")
        ):
            out[str(key)] = "<redacted>"
        else:
            out[str(key)] = item
    return out
