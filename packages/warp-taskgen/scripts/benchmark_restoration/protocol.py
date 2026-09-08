from __future__ import annotations

import json
import socket
from typing import Any

from .errors import RestorationError

MAX_REQUEST_BYTES = 64 * 1024


def read_request(conn: socket.socket) -> dict[str, Any]:
    """Read exactly one bounded newline-delimited JSON request."""

    conn.settimeout(5.0)
    data = bytearray()
    while len(data) <= MAX_REQUEST_BYTES:
        try:
            chunk = conn.recv(min(4096, MAX_REQUEST_BYTES + 1 - len(data)))
        except TimeoutError as exc:
            raise RestorationError("request_timeout") from exc
        if not chunk:
            break
        data.extend(chunk)
        if b"\n" in chunk:
            break
    if len(data) > MAX_REQUEST_BYTES:
        raise RestorationError("request_too_large")
    if b"\n" not in data:
        raise RestorationError("invalid_request")
    line, remainder = bytes(data).split(b"\n", 1)
    if remainder.strip():
        raise RestorationError("invalid_request")
    try:
        payload = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RestorationError("invalid_request") from exc
    if not isinstance(payload, dict):
        raise RestorationError("invalid_request")
    return payload
