#!/usr/bin/env python3
"""Bounded stdlib HTTP verifier staged into Phase 0c sandboxes."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from typing import Any


def verify_http(
    *,
    url: str,
    method: str = "GET",
    auth_header: str | None = None,
    headers: Iterable[str] = (),
    data: str | None = None,
    timeout: float = 8.0,
    max_bytes: int = 4096,
) -> dict[str, Any]:
    """Make one bounded HTTP request and return a JSON-serializable record."""
    request_headers = _parse_headers(headers)
    if auth_header:
        name, value = _split_header(auth_header)
        request_headers[name] = value
    body = data.encode("utf-8") if data is not None else None
    request = urllib.request.Request(
        url,
        data=body,
        headers=request_headers,
        method=method.upper(),
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body_bytes = response.read(max(0, max_bytes))
            return _record(
                url=url,
                method=method,
                status=response.status,
                outcome="ok",
                elapsed=time.monotonic() - started,
                body=body_bytes,
                final_url=response.geturl(),
                error=None,
            )
    except urllib.error.HTTPError as exc:
        body_bytes = exc.read(max(0, max_bytes))
        return _record(
            url=url,
            method=method,
            status=exc.code,
            outcome="http_error",
            elapsed=time.monotonic() - started,
            body=body_bytes,
            final_url=exc.geturl(),
            error=str(exc),
        )
    except urllib.error.URLError as exc:
        return _record(
            url=url,
            method=method,
            status=None,
            outcome="network_error",
            elapsed=time.monotonic() - started,
            body=b"",
            final_url=None,
            error=str(exc.reason),
        )
    except Exception as exc:
        return _record(
            url=url,
            method=method,
            status=None,
            outcome="error",
            elapsed=time.monotonic() - started,
            body=b"",
            final_url=None,
            error=repr(exc),
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--method", default="GET")
    parser.add_argument("--auth-header", default=None)
    parser.add_argument("--header", action="append", default=[])
    parser.add_argument("--data", default=None)
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--max-bytes", type=int, default=4096)
    args = parser.parse_args(argv)
    result = verify_http(
        url=args.url,
        method=args.method,
        auth_header=args.auth_header,
        headers=args.header,
        data=args.data,
        timeout=args.timeout,
        max_bytes=args.max_bytes,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


def _parse_headers(headers: Iterable[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for header in headers:
        name, value = _split_header(header)
        parsed[name] = value
    return parsed


def _split_header(header: str) -> tuple[str, str]:
    if ":" not in header:
        raise ValueError(f"invalid header {header!r}; expected 'Name: value'")
    name, _, value = header.partition(":")
    name = name.strip()
    if not name:
        raise ValueError(f"invalid header {header!r}; header name is empty")
    return name, value.strip()


def _record(
    *,
    url: str,
    method: str,
    status: int | None,
    outcome: str,
    elapsed: float,
    body: bytes,
    final_url: str | None,
    error: str | None,
) -> dict[str, Any]:
    return {
        "url": url,
        "method": method.upper(),
        "status": status,
        "outcome": outcome,
        "elapsed_ms": int(elapsed * 1000),
        "bytes_read": len(body),
        "body_preview": body.decode("utf-8", errors="replace")[:1000],
        "final_url": final_url,
        "error": error,
    }


if __name__ == "__main__":
    raise SystemExit(main())
