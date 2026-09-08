"""Network trace recording for Browser Use runs."""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from contextlib import suppress
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse, urlsplit, urlunsplit

from warp_taskgen.atomic_io import write_json_atomic

logger = logging.getLogger(__name__)
_DEFAULT_WRITE_JSON_ATOMIC = write_json_atomic

_SENSITIVE_HEADER_NAMES = {
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
_URL_VALUE_HEADER_NAMES = {
    "referer",
    "referrer",
    "location",
    "content-location",
}


def _write_json_atomic_compat(path: Path, payload: object) -> None:
    """Preserve the old browser_use_agent monkeypatch surface for trace writes."""
    agent_module = sys.modules.get("warp_taskgen.browser_use_agent")
    agent_writer = getattr(agent_module, "write_json_atomic", _DEFAULT_WRITE_JSON_ATOMIC)
    writer = agent_writer if agent_writer is not _DEFAULT_WRITE_JSON_ATOMIC else write_json_atomic
    writer(path, payload)


class _NetworkTraceRecorder:
    """Task-scoped CDP network recorder.

    Captures Network.* events via CDP from all active page sessions and
    produces a flat trace matching the format WebArena Verified's
    ``NetworkEventEvaluator`` expects::

        {
            "url": "...",
            "method": "GET",
            "headers": {"key": "value"},
            "query_params": {"key": ["value"]},
            "post_data": "...",
            "response_status": 200,
            "response_headers": {"key": "value"},
            "response_cookies": {"name": "value"},
        }

    Browser Use wraps ``cdp_use.CDPClient`` which dispatches all events
    (including from auto-attached child sessions) through a single root
    WebSocket.  We register handlers on that root client and enable the
    Network domain per-page-session so Chrome emits the events.
    """

    def __init__(
        self,
        browser_session: Any,
        task_dir: Path,
        *,
        target_filter: set[str] | None = None,
        sensitive_header_names: set[str] | None = None,
    ) -> None:
        self._browser_session = browser_session
        self._task_dir = Path(task_dir)
        self._client = getattr(browser_session, "cdp_client", None)
        self._recording = False
        self._poll_task: asyncio.Task | None = None
        self._enabled_targets: set[str] = set()
        self._target_filter = target_filter
        self._sensitive_header_names = {
            str(name).lower() for name in (sensitive_header_names or set()) if str(name).strip()
        }
        # Raw CDP entries for the currently active hop, keyed by requestId.
        self._requests: dict[str, dict[str, Any]] = {}
        # Completed redirect hops are retained separately because Chrome reuses
        # one requestId for every hop in a redirect chain.
        self._completed_request_hops: list[dict[str, Any]] = []
        self._request_sequence = 0
        # Top-frame navigation events for C1b URL matching + HAR pages[].
        self._nav_events: list[dict[str, Any]] = []
        self._nav_seq: int = 0

    async def start(self) -> None:
        """Register CDP event handlers and enable the Network domain."""
        if self._client is None:
            logger.warning("CDP client not available, network trace capture disabled")
            return
        self._client.register.Network.requestWillBeSent(self._on_request_will_be_sent)
        self._client.register.Network.requestWillBeSentExtraInfo(
            self._on_request_will_be_sent_extra_info
        )
        self._client.register.Network.responseReceived(self._on_response_received)
        self._client.register.Network.responseReceivedExtraInfo(
            self._on_response_received_extra_info
        )
        self._client.register.Network.loadingFinished(self._on_loading_finished)
        self._client.register.Network.loadingFailed(self._on_loading_failed)
        # Page-domain navigation events — top-frame document nav + SPA within-doc.
        # Both require ``Page.enable`` on the session (sent alongside ``Network.enable``
        # in ``_enable_current_page_sessions``) before Chrome dispatches events.
        self._client.register.Page.frameNavigated(self._on_frame_navigated)
        self._client.register.Page.navigatedWithinDocument(self._on_navigated_within_document)

        await self._enable_current_page_sessions()
        self._recording = True
        # Poll for newly-opened tabs/popups so Network.enable is sent promptly.
        self._poll_task = asyncio.create_task(self._poll_sessions(), name="network-trace-poller")

    async def stop(self) -> list[dict[str, Any]]:
        """Stop recording, finalize trace, write to disk, return entries."""
        self._recording = False
        if self._poll_task is not None:
            self._poll_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._poll_task
            self._poll_task = None

        trace = self._finalize_trace()
        persisted_trace = [
            self._redact_trace_entry(
                entry,
                sensitive_header_names=self._sensitive_header_names,
                redact_payloads=False,
            )
            for entry in trace
        ]
        evaluator_trace = [
            self._redact_trace_entry(
                entry,
                sensitive_header_names=self._sensitive_header_names,
                redact_payloads=False,
            )
            for entry in trace
        ]
        # WorldSim intentionally preserves benchmark request payloads in local
        # Browser Use traces, matching AgentLab. These controlled WebArena-style
        # runs need URL/query/body evidence for host-owned validators,
        # especially Reddit/Postmill submit-comment attribution. Request-body
        # evidence alone is not accepted as final state success; Reddit comment
        # rewards still require attributed readback that excludes seeded carrier
        # comments. Credential-bearing headers and cookies remain redacted.
        self._write_trace(persisted_trace)
        return evaluator_trace

    async def _poll_sessions(self) -> None:
        """Periodically discover new page targets and enable Network on them."""
        try:
            while True:
                await self._enable_current_page_sessions()
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            raise

    async def _enable_current_page_sessions(self) -> None:
        """Send ``Network.enable`` to every known page target (idempotent)."""
        session_manager = getattr(self._browser_session, "session_manager", None)
        if session_manager is None:
            return

        for target in session_manager.get_all_page_targets():
            target_id = getattr(target, "target_id", None)
            if not target_id or target_id in self._enabled_targets:
                continue
            if self._target_filter is not None and target_id not in self._target_filter:
                # Each worker gets a dedicated external PVPO browser endpoint.
                # Once the task reset is complete, any newly discovered page
                # target on that endpoint belongs to the current task (popup,
                # window.open, etc.), so adopt it for tracing and cleanup.
                self._target_filter.add(target_id)

            try:
                session = await self._browser_session.get_or_create_cdp_session(
                    target_id, focus=False
                )
                await session.cdp_client.send.Network.enable(session_id=session.session_id)
                # Page.enable must be sent before the frame-navigation handlers
                # registered in ``start()`` will receive events.
                await session.cdp_client.send.Page.enable(session_id=session.session_id)
                self._enabled_targets.add(target_id)
            except Exception as e:
                logger.debug("Network trace enable failed for target %s: %s", target_id, e)

    def _new_entry(self, request_id: str, session_id: str | None = None) -> dict[str, Any]:
        """Create a raw CDP entry with a stable event-order sequence."""
        self._request_sequence += 1
        return {
            "request_id": request_id,
            "session_id": session_id,
            "url": "",
            "method": "GET",
            "_sequence": self._request_sequence,
        }

    def _entry(self, request_id: str, session_id: str | None = None) -> dict[str, Any]:
        """Get or create the active raw CDP entry for *request_id*."""
        entry = self._requests.get(request_id)
        if entry is None:
            entry = self._new_entry(request_id, session_id)
            self._requests[request_id] = entry
        if session_id is not None:
            entry["session_id"] = session_id
        return entry

    @staticmethod
    def _bind_redirect_response(entry: dict[str, Any], response: dict[str, Any]) -> None:
        """Bind the response carried by a redirect event to its request hop."""
        if "status" in response:
            entry["response_status"] = response.get("status")
        if "statusText" in response:
            entry["response_status_text"] = response.get("statusText")
        if "mimeType" in response:
            entry["response_mime_type"] = response.get("mimeType")
        if "headers" in response:
            headers = dict(entry.get("response_headers") or {})
            headers.update(_NetworkTraceRecorder._headers(response.get("headers")))
            entry["response_headers"] = headers
        if "fromDiskCache" in response:
            entry["response_from_cache"] = response.get("fromDiskCache")

    @staticmethod
    def _headers(headers: Any) -> dict[str, str]:
        """Normalize CDP headers (always a dict) to ``{str: str}``."""
        if not headers:
            return {}
        if isinstance(headers, dict):
            return {str(k): str(v) for k, v in headers.items()}
        return {}

    def _on_request_will_be_sent(
        self, event: dict[str, Any], session_id: str | None = None
    ) -> None:
        if not self._recording:
            return

        request = event.get("request", {})
        request_id = event.get("requestId")
        if not request_id:
            return

        entry = self._entry(request_id, session_id)

        # Preserve redirect hops. CDP fires one requestWillBeSent per hop with
        # the same requestId; the new event carries ``redirectResponse`` (the
        # prior hop's response). Close the previous request before starting a
        # fresh active entry so its method/body and response remain bound.
        redirect_response = event.get("redirectResponse")
        if redirect_response and entry.get("url"):
            redirect_chain = list(entry.get("redirect_chain") or [])
            redirect_chain.append(
                {
                    "url": entry["url"],
                    "status": redirect_response.get("status"),
                }
            )
            completed_hop = deepcopy(entry)
            self._bind_redirect_response(completed_hop, redirect_response)
            self._completed_request_hops.append(completed_hop)
            entry = self._new_entry(request_id, session_id)
            entry["redirect_chain"] = redirect_chain
            self._requests[request_id] = entry

        entry.update(
            {
                "timestamp": event.get("timestamp"),
                "wall_time": event.get("wallTime"),
                "document_url": event.get("documentURL"),
                "type": event.get("type"),
                "url": request.get("url", ""),
                "method": str(request.get("method", "GET")).upper(),
                "request_headers": self._headers(request.get("headers")),
                "post_data": request.get("postData"),
            }
        )

    def _on_request_will_be_sent_extra_info(
        self, event: dict[str, Any], session_id: str | None = None
    ) -> None:
        if not self._recording:
            return

        request_id = event.get("requestId")
        if not request_id:
            return

        entry = self._entry(request_id, session_id)
        # Extra-info headers are the *actual* wire headers (after cookie
        # injection, etc.) so they take precedence over the request headers.
        extra_headers = self._headers(event.get("headers"))
        if extra_headers:
            entry.setdefault("request_headers_extra", {}).update(extra_headers)
        # Cookies associated with this request (sent by browser).
        entry["associated_cookies"] = event.get("associatedCookies", [])

    def _on_response_received(self, event: dict[str, Any], session_id: str | None = None) -> None:
        if not self._recording:
            return

        request_id = event.get("requestId")
        response = event.get("response", {})
        if not request_id:
            return

        entry = self._entry(request_id, session_id)
        entry["response_status"] = response.get("status")
        entry["response_status_text"] = response.get("statusText")
        entry["response_mime_type"] = response.get("mimeType")
        entry["response_headers"] = self._headers(response.get("headers"))
        entry["response_from_cache"] = response.get("fromDiskCache", False)

    def _on_response_received_extra_info(
        self, event: dict[str, Any], session_id: str | None = None
    ) -> None:
        if not self._recording:
            return

        request_id = event.get("requestId")
        if not request_id:
            return

        entry = self._entry(request_id, session_id)
        # Wire-level response headers (may differ from response_headers above
        # due to CORS filtering, etc.).
        extra_resp_headers = self._headers(event.get("headers"))
        if extra_resp_headers:
            entry.setdefault("response_headers_extra", {}).update(extra_resp_headers)
        # Response cookies the browser blocked or exempted.
        entry["blocked_cookies"] = event.get("blockedCookies", [])
        entry["exempted_cookies"] = event.get("exemptedCookies", [])

    def _on_loading_finished(self, event: dict[str, Any], session_id: str | None = None) -> None:
        if not self._recording:
            return

        request_id = event.get("requestId")
        if not request_id:
            return

        entry = self._entry(request_id, session_id)
        entry["loading_finished"] = True
        entry["encoded_data_length"] = event.get("encodedDataLength")

    def _on_loading_failed(self, event: dict[str, Any], session_id: str | None = None) -> None:
        if not self._recording:
            return

        request_id = event.get("requestId")
        if not request_id:
            return

        entry = self._entry(request_id, session_id)
        entry["loading_failed"] = True
        entry["error_text"] = event.get("errorText")
        entry["canceled"] = event.get("canceled")

    def _on_frame_navigated(self, event: dict[str, Any], session_id: str | None = None) -> None:
        if not self._recording:
            return
        frame = event.get("frame") or {}
        # Top frame only — sub-frame navs don't carry the read surface the
        # C1b classifier cares about.
        if frame.get("parentId"):
            return
        self._nav_seq += 1
        self._nav_events.append(
            {
                "url": frame.get("url") or "",
                "navigation_type": event.get("type"),
                "timestamp": time.time(),
                "kind": "document",
                "pageref": f"page_{self._nav_seq}",
            }
        )

    def _on_navigated_within_document(
        self, event: dict[str, Any], session_id: str | None = None
    ) -> None:
        if not self._recording:
            return
        self._nav_events.append(
            {
                "url": event.get("url") or "",
                "navigation_type": event.get("navigationType"),
                "timestamp": time.time(),
                "kind": "within_document",
            }
        )

    @staticmethod
    def _parse_cookies_from_headers(headers: dict[str, str]) -> dict[str, str]:
        """Extract ``{name: value}`` from ``set-cookie`` response headers.

        CDP collapses multiple Set-Cookie headers into a single newline-
        separated value.  We parse them all.
        """
        cookies: dict[str, str] = {}
        raw = headers.get("set-cookie") or headers.get("Set-Cookie") or ""
        if not raw:
            return cookies
        for line in raw.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Each Set-Cookie line: "name=value; Path=/; ..."
            kv = line.split(";", 1)[0]
            if "=" in kv:
                name, _, value = kv.partition("=")
                cookies[name.strip()] = value.strip()
        return cookies

    def _flatten_entry(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Convert a raw CDP-collected entry to the evaluator-expected format.

        Merges request/response/extra-info fields into the flat schema that
        ``NetworkEventEvaluator`` consumes.
        """
        url = raw.get("url", "")

        # Best-effort query-param parsing from the URL.
        query_params: dict[str, list[str]] = {}
        try:
            parsed = urlparse(url)
            query_params = parse_qs(parsed.query, keep_blank_values=True)
        except Exception:
            pass

        # Merge request headers: prefer extra-info (wire-level) when available.
        headers = dict(raw.get("request_headers", {}))
        headers.update(raw.get("request_headers_extra", {}))

        # Merge response headers similarly.
        resp_headers = dict(raw.get("response_headers", {}))
        resp_headers.update(raw.get("response_headers_extra", {}))

        # Extract cookies from Set-Cookie response headers.
        resp_cookies = self._parse_cookies_from_headers(resp_headers)

        resource_type = raw.get("type")
        flat: dict[str, Any] = {
            "url": url,
            "method": raw.get("method", "GET"),
            "headers": headers,
            "query_params": query_params,
            "post_data": raw.get("post_data"),
            "response_status": raw.get("response_status"),
            "response_mime_type": raw.get("response_mime_type"),
            "response_headers": resp_headers,
            "response_cookies": resp_cookies,
            "is_document_load": resource_type == "Document",
            "resource_type": resource_type,
        }
        if raw.get("request_id") is not None:
            flat["request_id"] = raw["request_id"]
        if raw.get("session_id") is not None:
            flat["session_id"] = raw["session_id"]
        redirect_chain = raw.get("redirect_chain")
        if redirect_chain:
            flat["redirect_chain"] = list(redirect_chain)
        return flat

    def _finalize_trace(self) -> list[dict[str, Any]]:
        """Return flat, evaluator-ready entries sorted by CDP timestamp."""
        raw_entries = [*self._completed_request_hops, *self._requests.values()]
        raw_entries.sort(
            key=lambda e: (
                e.get("timestamp") is None,
                e.get("timestamp", 0),
                e.get("_sequence", 0),
            )
        )
        flat_entries = [self._flatten_entry(e) for e in raw_entries]

        # Assign HAR ``pageref`` to each entry based on the most recent
        # top-frame document navigation. Nav-event timestamps use
        # ``time.time()`` and CDP entries carry ``wallTime`` (both wall-clock
        # epoch seconds) so they are directly comparable.
        doc_navs = sorted(
            (
                (nav.get("timestamp") or 0.0, nav.get("pageref") or "")
                for nav in self._nav_events
                if nav.get("kind") == "document" and nav.get("pageref")
            ),
            key=lambda item: item[0],
        )
        if doc_navs:
            for raw, flat_entry in zip(raw_entries, flat_entries, strict=False):
                wall = raw.get("wall_time")
                if wall is None:
                    continue
                pageref = ""
                for nav_time, nav_pageref in doc_navs:
                    if nav_time <= wall:
                        pageref = nav_pageref
                    else:
                        break
                if pageref:
                    flat_entry["pageref"] = pageref
        return flat_entries

    def _write_trace(self, trace: list[dict[str, Any]]) -> None:
        """Write the flat internal trace and a valid HAR file.

        ``network_trace.json`` keeps the flat shape that older diagnostic
        tooling expects. ``network.har`` is converted to the HAR entry
        shape the vendor's NetworkEvent parser actually accepts; the
        previous version wrote flat entries under a HAR envelope, which
        the vendor rejected with ``"Unknown trace format"``.
        """
        # Import lazily to avoid a circular dep at module load time.
        from warp_taskgen.har_converter import (
            flat_events_to_har_entries,
            nav_events_to_har_pages,
        )

        redacted_nav_events = [self._redact_navigation_event(event) for event in self._nav_events]

        try:
            _write_json_atomic_compat(
                self._task_dir / "network_trace.json",
                trace,
            )
        except Exception as e:
            logger.warning("Failed to write network_trace.json: %s", e)

        try:
            _write_json_atomic_compat(
                self._task_dir / "navigation_trace.json",
                redacted_nav_events,
            )
        except Exception as e:
            logger.warning("Failed to write navigation_trace.json: %s", e)

        har_entries = flat_events_to_har_entries(trace)
        har_pages = nav_events_to_har_pages(redacted_nav_events)
        payload = {
            "log": {
                "version": "1.2",
                "creator": {
                    "name": "worldsim",
                    "version": "phase-3-network-trace",
                },
                "pages": har_pages,
                "entries": har_entries,
            }
        }
        try:
            _write_json_atomic_compat(
                self._task_dir / "network.har",
                payload,
            )
        except Exception as e:
            logger.warning("Failed to write network.har: %s", e)

    @classmethod
    def _redact_trace_entry(
        cls,
        entry: dict[str, Any],
        *,
        sensitive_header_names: set[str] | None = None,
        redact_payloads: bool = True,
    ) -> dict[str, Any]:
        """Redact sensitive wire data before persisting trajectory artifacts."""
        redacted = dict(entry)
        if redact_payloads:
            redacted["url"] = cls._redact_url(redacted.get("url", ""))
            redacted["query_params"] = cls._redact_query_params(redacted.get("query_params", {}))
            if redacted.get("redirect_chain"):
                redacted["redirect_chain"] = cls._redact_redirect_chain(redacted["redirect_chain"])
        redacted["headers"] = cls._redact_headers(
            redacted.get("headers", {}),
            sensitive_header_names=sensitive_header_names,
            redact_url_values=redact_payloads,
        )
        redacted["response_headers"] = cls._redact_headers(
            redacted.get("response_headers", {}),
            sensitive_header_names=sensitive_header_names,
            redact_url_values=redact_payloads,
        )
        if redact_payloads and redacted.get("post_data") is not None:
            redacted["post_data"] = "<redacted>"
        if redacted.get("response_cookies"):
            redacted["response_cookies"] = cls._redact_cookies(redacted["response_cookies"])
        return redacted

    @classmethod
    def _redact_navigation_event(cls, event: dict[str, Any]) -> dict[str, Any]:
        """Redact top-frame navigation URLs before persisting artifacts."""
        redacted = dict(event)
        if redacted.get("url") is not None:
            redacted["url"] = cls._redact_url(str(redacted.get("url") or ""))
        return redacted

    @classmethod
    def _redact_redirect_chain(cls, redirect_chain: Any) -> list[Any]:
        """Redact URL-bearing redirect hops while preserving hop metadata."""
        if not isinstance(redirect_chain, list):
            return []
        redacted_chain: list[Any] = []
        for hop in redirect_chain:
            if not isinstance(hop, dict):
                redacted_chain.append(hop)
                continue
            redacted_hop = dict(hop)
            if redacted_hop.get("url") is not None:
                redacted_hop["url"] = cls._redact_url(str(redacted_hop.get("url") or ""))
            redacted_chain.append(redacted_hop)
        return redacted_chain

    @classmethod
    def _redact_cookies(cls, cookies: Any) -> Any:
        """Preserve cookie names while removing all cookie values."""
        if isinstance(cookies, dict):
            return {str(key): "<redacted>" for key in cookies}
        if isinstance(cookies, list):
            redacted: list[Any] = []
            for cookie in cookies:
                if isinstance(cookie, dict):
                    cookie_copy = dict(cookie)
                    if "value" in cookie_copy:
                        cookie_copy["value"] = "<redacted>"
                    redacted.append(cookie_copy)
                elif isinstance(cookie, (list, tuple)) and len(cookie) == 2:
                    redacted.append([cookie[0], "<redacted>"])
                else:
                    redacted.append("<redacted>")
            return redacted
        return "<redacted>"

    @classmethod
    def _redact_headers(
        cls,
        headers: dict[str, Any],
        *,
        sensitive_header_names: set[str] | None = None,
        redact_url_values: bool = False,
    ) -> dict[str, Any]:
        """Redact sensitive header values while preserving non-secret metadata."""
        redacted: dict[str, Any] = {}
        configured_sensitive = {
            str(name).lower() for name in (sensitive_header_names or set()) if str(name).strip()
        }
        for key, value in headers.items():
            lower = str(key).lower()
            if (
                lower in configured_sensitive
                or lower in _SENSITIVE_HEADER_NAMES
                or any(marker in lower for marker in _SENSITIVE_HEADER_SUBSTRINGS)
            ):
                redacted[str(key)] = "<redacted>"
            elif redact_url_values:
                redacted[str(key)] = cls._redact_header_url_value(lower, value)
            else:
                redacted[str(key)] = value
        return redacted

    @classmethod
    def _redact_header_url_value(cls, lower_key: str, value: Any) -> Any:
        """Redact query/fragment data from URL-bearing header values."""
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if not stripped:
            return value
        if lower_key in _URL_VALUE_HEADER_NAMES or stripped.lower().startswith(
            ("http://", "https://")
        ):
            return cls._redact_url(stripped)
        return value

    @classmethod
    def _redact_query_params(cls, query_params: dict[str, Any]) -> dict[str, list[str]]:
        """Preserve query keys while stripping all values."""
        redacted: dict[str, list[str]] = {}
        for key, value in query_params.items():
            if isinstance(value, list):
                redacted[str(key)] = ["<redacted>"] * len(value)
            else:
                redacted[str(key)] = ["<redacted>"]
        return redacted

    @classmethod
    def _redact_url(cls, url: str) -> str:
        """Strip fragments and redact query string values."""
        if not url:
            return ""
        try:
            parsed = urlsplit(url)
        except Exception:
            return url

        raw_query = parse_qs(parsed.query, keep_blank_values=True)
        redacted_query = urlencode(cls._redact_query_params(raw_query), doseq=True)
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, redacted_query, ""))
