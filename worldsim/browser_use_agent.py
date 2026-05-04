"""Browser Use agent runner.

Canonical source: ``docs/worldsim-v5-technical-specifcation.md`` "Browser Use Integration".

We use Browser Use as an async Python library (not a subprocess) for running
browser agents against pre-running benchmark environments. Each worker owns
the runner object, and each task gets a fresh ``BrowserSession`` so trajectory
artifacts such as network traces stay isolated per task directory.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import shutil
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import parse_qs, urlencode, urlparse, urlsplit, urlunsplit

from worldsim.agent_auth import (
    _resolve_declared_storage_state_path,
    playwright_storage_state,
    read_storage_state_payload,
    resolve_agent_auth_headers,
    storage_state_preflight_error_for_payload,
)
from worldsim.atomic_io import write_json_atomic
from worldsim.config import has_configured_agent_auth, has_effective_agent_auth
from worldsim.pvpo_endpoint import validate_pvpo_cdp_url

logger = logging.getLogger(__name__)


def _ensure_browser_use_runtime_env() -> None:
    """Default Browser Use to local-only runtime behavior.

    WorldSim runs Browser Use as an embedded evaluator component. Cloud sync
    and anonymous telemetry are not part of the benchmark contract, and in live
    r5 runs the telemetry client can leave a non-daemon worker thread behind
    after Phase 4 has already written complete artifacts. Keep these defaults
    local-only unless a caller explicitly opts back in through the environment.
    """
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
    os.environ.setdefault("BROWSER_USE_CLOUD_SYNC", "false")
    os.environ.setdefault("POSTHOG_DISABLED", "true")


_ensure_browser_use_runtime_env()

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
_PHASE_0D_SITE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_CDP_VIEWPORT_JS = """
(() => ({
  w: Math.max(0, Number(window.innerWidth || 0)),
  h: Math.max(0, Number(window.innerHeight || 0))
}))()
"""
_PVPO_CDP_TIMEOUT_ENV = "WORLDSIM_PVPO_CDP_TIMEOUT_S"
_PVPO_CDP_TIMEOUT_DEFAULT_S = 10.0
_PVPO_SCROLL_ACTION_TIMEOUT_ENV = "WORLDSIM_PVPO_SCROLL_ACTION_TIMEOUT_S"
_PVPO_SCROLL_ACTION_TIMEOUT_DEFAULT_S = 1.0
_PVPO_SCROLL_EPSILON_PX = 1.0
_PVPO_NAVIGATION_TICK_INTERVAL_ENV = "WORLDSIM_PVPO_NAVIGATION_TICK_MS"
_PVPO_NAVIGATION_TICK_DEFAULT_MS = 50.0
_PVPO_NAVIGATION_TICK_STOP_GRACE_MIN_S = 2.0
_PVPO_SCREENSHOT_PATCHED = False
_PVPO_SCROLL_PATCHED = False
_PVPO_NAVIGATION_TICK_PATCHED = False
_TRANSPARENT_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


@dataclass(frozen=True)
class _PvpoScrollState:
    x: float
    y: float
    max_y: float
_CLEAR_PAGE_STORAGE_JS = """
(() => {
  try { window.localStorage.clear(); } catch (_) {}
  try { window.sessionStorage.clear(); } catch (_) {}
  return true;
})()
"""


@dataclass
class AgentResult:
    """Summary of one agent run, extracted from the Browser Use history."""

    elapsed: float
    steps: int
    is_done: bool
    final_result: str | None
    status: str = "success"
    errors: list[str] = field(default_factory=list)
    network_trace: list[dict[str, Any]] = field(default_factory=list)


class AgentRunner(Protocol):
    """Protocol every agent implementation in the worker pool must satisfy."""

    async def setup(self, server_url: str) -> None: ...

    async def run(
        self,
        task: str,
        server_url: str,
        task_dir: Path,
        *,
        start_urls: list[str] | None = None,
        site_prompt: str | None = None,
        auth_mechanism: dict[str, Any] | None = None,
        benchmark_root: Path | None = None,
        task_site: str | None = None,
        payload_text: str | None = None,
        payload_witnesses: list[str | dict[str, Any]] | None = None,
        pvpo_cdp_url: str | None = None,
        instance_id: str | None = None,
        url_origin_rewrites: dict[str, str] | None = None,
    ) -> AgentResult: ...

    async def teardown(self) -> None: ...


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
        # Raw CDP entries keyed by requestId.
        self._requests: dict[str, dict[str, Any]] = {}
        # Top-frame navigation events for C1b URL matching + HAR pages[].
        self._nav_events: list[dict[str, Any]] = []
        self._nav_seq: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

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
        redacted_trace = [
            self._redact_trace_entry(
                entry,
                sensitive_header_names=self._sensitive_header_names,
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
        # Persist only redacted wire artifacts; downstream sandboxes may stage
        # network.har wholesale from the trajectory directory.
        self._write_trace(redacted_trace)
        return evaluator_trace

    # ------------------------------------------------------------------
    # Session discovery
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _entry(self, request_id: str, session_id: str | None = None) -> dict[str, Any]:
        """Get or create a raw CDP entry for *request_id*."""
        entry = self._requests.setdefault(
            request_id,
            {
                "request_id": request_id,
                "session_id": session_id,
                "url": "",
                "method": "GET",
            },
        )
        if session_id is not None:
            entry["session_id"] = session_id
        return entry

    @staticmethod
    def _headers(headers: Any) -> dict[str, str]:
        """Normalize CDP headers (always a dict) to ``{str: str}``."""
        if not headers:
            return {}
        if isinstance(headers, dict):
            return {str(k): str(v) for k, v in headers.items()}
        return {}

    # ------------------------------------------------------------------
    # CDP event handlers (sync — cdp_use dispatches synchronously)
    # ------------------------------------------------------------------

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
        # prior hop's response). Record the URL we had + that response's status
        # before overwriting.
        redirect_response = event.get("redirectResponse")
        if redirect_response and entry.get("url"):
            entry.setdefault("redirect_chain", []).append(
                {
                    "url": entry["url"],
                    "status": redirect_response.get("status"),
                }
            )

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

    # ------------------------------------------------------------------
    # Trace finalization
    # ------------------------------------------------------------------

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
        redirect_chain = raw.get("redirect_chain")
        if redirect_chain:
            flat["redirect_chain"] = list(redirect_chain)
        return flat

    def _finalize_trace(self) -> list[dict[str, Any]]:
        """Return flat, evaluator-ready entries sorted by CDP timestamp."""
        raw_entries = list(self._requests.values())
        raw_entries.sort(key=lambda e: (e.get("timestamp") is None, e.get("timestamp", 0)))
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
        from worldsim.har_converter import (
            flat_events_to_har_entries,
            nav_events_to_har_pages,
        )

        redacted_nav_events = [self._redact_navigation_event(event) for event in self._nav_events]

        try:
            write_json_atomic(
                self._task_dir / "network_trace.json",
                trace,
            )
        except Exception as e:
            logger.warning("Failed to write network_trace.json: %s", e)

        try:
            write_json_atomic(
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
            write_json_atomic(
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


class AuthArtifactMissingError(RuntimeError):
    """Raised when a declared storage_state artifact (or its generator) is missing."""


def _phase_0d_fallback_path(
    task: dict[str, Any] | None,
    *,
    instance_id: str | None = None,
) -> Path | None:
    """Return the Phase 0d-bootstrapped storage_state.json path for ``task``'s site.

    Phase 0d writes artifacts to ``<state_dir>/phase_0d/<site>/storage_state.json``
    plus a per-instance copy under ``instances/<instance_id>/storage_state.json``
    when multiple replicas of a site are configured. When ``instance_id`` is
    supplied, the per-instance file is preferred; missing per-instance file falls
    back to the shared one with a warning so multi-replica deploys that have not
    yet re-run Phase 0d still execute (with the cookie likely rejected).

    Returns ``None`` when the task has no ``site`` field or phase 0d was never
    run. The import is local to avoid a circular module dependency between
    ``browser_use_agent`` and ``phases.phase_0d_auth_bootstrap``.
    """
    if not isinstance(task, dict):
        return None
    site = task.get("site")
    if not isinstance(site, str) or not site.strip():
        return None
    if _PHASE_0D_SITE_RE.fullmatch(site.strip()) is None:
        return None
    try:
        from worldsim.phases.phase_0d_auth_bootstrap import (
            phase_0d_artifact_path,
            phase_0d_completion_path,
            phase_0d_instance_artifact_path_by_id,
        )
    except ImportError:  # pragma: no cover — only triggers on misinstalled env.
        return None
    completion_path = phase_0d_completion_path(site.strip())
    if instance_id:
        per_instance = phase_0d_instance_artifact_path_by_id(site.strip(), instance_id)
        if per_instance.exists():
            return per_instance
        logger.warning(
            "agent auth: per-instance Phase 0d artifact %s missing; falling back to shared "
            "storage_state for site %r (re-run Phase 0d to populate the per-instance file)",
            per_instance,
            site.strip(),
        )
    artifact_path = phase_0d_artifact_path(site.strip())
    if not artifact_path.exists() or not completion_path.exists():
        return None
    return artifact_path


def _resolve_pvpo_cdp_url(raw_url: str) -> str:
    """Validate the optional external CDP endpoint for PVPO."""
    resolved = validate_pvpo_cdp_url(
        raw_url,
        field_name="WORLDSIM_PVPO_CDP_URL",
        allow_empty=True,
    )
    return resolved or ""


def _origin_from_url(raw_url: str) -> str | None:
    """Return ``scheme://host[:port]`` for http(s) URLs."""
    if not isinstance(raw_url, str) or not raw_url.strip():
        return None
    parsed = urlparse(raw_url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None

    default_port = 80 if parsed.scheme == "http" else 443
    if parsed.port and parsed.port != default_port:
        return f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"
    return f"{parsed.scheme}://{parsed.hostname}"


def _origins_from_network_trace(trace: list[dict[str, Any]]) -> set[str]:
    """Collect http(s) origins observed in the network trace."""
    origins: set[str] = set()
    for entry in trace:
        origin = _origin_from_url(str(entry.get("url") or ""))
        if origin:
            origins.add(origin)
    return origins


class _ScopedHeaderAuthInjector:
    """CDP Fetch-based same-origin request mutator for BrowserUse runtime auth.

    It injects auth headers for configured origins and rewrites known
    benchmark-origin aliases back to the replica selected for the current
    task. Keeping both in one Fetch handler avoids double-continuing the same
    request when a task needs both behaviours.
    """

    def __init__(
        self,
        *,
        origin: str = "",
        headers: dict[str, str] | None = None,
        url_origin_rewrites: dict[str, str] | None = None,
    ) -> None:
        self.headers_by_origin: dict[str, dict[str, str]] = (
            {origin: dict(headers or {})} if origin and headers else {}
        )
        self.url_origin_rewrites = _normalize_origin_rewrites(url_origin_rewrites)
        self._browser_session: Any = None
        self._enabled_target_patterns: dict[str, tuple[str, ...]] = {}
        self._enabled_sessions: dict[str, Any] = {}
        self._poll_task: asyncio.Task | None = None
        self._continue_tasks: set[asyncio.Task] = set()
        self._running = False

    def add_headers(self, origin: str, headers: dict[str, str]) -> None:
        if not origin or not headers:
            return
        existing = self.headers_by_origin.setdefault(origin, {})
        existing.update({str(key): str(value) for key, value in headers.items()})

    def add_url_origin_rewrites(self, rewrites: dict[str, str] | None) -> None:
        self.url_origin_rewrites.update(_normalize_origin_rewrites(rewrites))

    async def start(self, browser_session: Any) -> None:
        self._browser_session = browser_session
        client = getattr(browser_session, "cdp_client", None)
        if client is None:
            raise AuthArtifactMissingError("CDP client unavailable for scoped http_headers auth")
        client.register.Fetch.requestPaused(self._on_request_paused)
        self._running = True
        await self._enable_current_page_sessions(require_enabled=True)
        self._poll_task = asyncio.create_task(
            self._poll_sessions(),
            name="scoped-header-auth-injector",
        )

    async def _poll_sessions(self) -> None:
        try:
            while self._running:
                try:
                    await self._enable_current_page_sessions()
                except Exception:
                    logger.debug("scoped http_headers poll iteration failed", exc_info=True)
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            raise

    async def _enable_current_page_sessions(self, *, require_enabled: bool = False) -> None:
        session_manager = getattr(self._browser_session, "session_manager", None)
        if session_manager is None:
            if require_enabled:
                raise AuthArtifactMissingError(
                    "scoped http_headers auth could not attach: session_manager unavailable"
                )
            return
        errors: list[Exception] = []
        for target in session_manager.get_all_page_targets():
            target_id = getattr(target, "target_id", None)
            if not target_id:
                continue
            origins = sorted(set(self.headers_by_origin) | set(self.url_origin_rewrites))
            if not origins:
                continue
            patterns = tuple(f"{origin}/*" for origin in origins)
            if self._enabled_target_patterns.get(target_id) == patterns:
                continue
            try:
                session = await asyncio.wait_for(
                    self._browser_session.get_or_create_cdp_session(target_id, focus=False),
                    timeout=2,
                )
                await asyncio.wait_for(
                    session.cdp_client.send.Fetch.enable(
                        {
                            "patterns": [
                                {
                                    "urlPattern": pattern,
                                    "requestStage": "Request",
                                }
                                for pattern in patterns
                            ]
                        },
                        session_id=session.session_id,
                    ),
                    timeout=2,
                )
            except Exception as exc:
                logger.debug(
                    "scoped http_headers Fetch.enable failed for target %s",
                    target_id,
                    exc_info=True,
                )
                errors.append(exc)
                continue
            self._enabled_target_patterns[target_id] = patterns
            self._enabled_sessions[target_id] = session
        if require_enabled and not self._enabled_sessions:
            message = "scoped http_headers auth could not attach to any page target"
            if errors:
                raise AuthArtifactMissingError(message) from errors[0]
            raise AuthArtifactMissingError(message)

    async def stop(self) -> None:
        self._running = False
        if self._poll_task is not None:
            self._poll_task.cancel()
            try:
                await asyncio.wait_for(self._poll_task, timeout=1)
            except asyncio.CancelledError:
                pass
            except TimeoutError:
                logger.debug("scoped http_headers poll task did not stop before timeout")
            except Exception:
                logger.debug("scoped http_headers poll task failed during shutdown", exc_info=True)
            self._poll_task = None
        if self._continue_tasks:
            done, pending = await asyncio.wait(self._continue_tasks, timeout=1)
            for task in done:
                with suppress(asyncio.CancelledError):
                    exc = task.exception()
                    if exc is not None:
                        logger.debug(
                            "scoped http_headers continueRequest failed during shutdown",
                            exc_info=(type(exc), exc, exc.__traceback__),
                        )
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            self._continue_tasks.clear()
        for session in list(self._enabled_sessions.values()):
            fetch = getattr(getattr(session.cdp_client, "send", None), "Fetch", None)
            disable = getattr(fetch, "disable", None)
            if callable(disable):
                try:
                    await asyncio.wait_for(disable(session_id=session.session_id), timeout=1)
                except TimeoutError:
                    logger.debug("scoped http_headers Fetch.disable timed out")
                except Exception:
                    logger.debug("scoped http_headers Fetch.disable failed", exc_info=True)
        self._enabled_sessions.clear()
        self._enabled_target_patterns.clear()

    def _on_request_paused(
        self,
        event: dict[str, Any],
        session_id: str | None = None,
    ) -> None:
        if not self._running:
            return
        task = asyncio.create_task(self._continue_request(event, session_id))
        self._continue_tasks.add(task)
        task.add_done_callback(self._continue_tasks.discard)

    async def _continue_request(
        self,
        event: dict[str, Any],
        session_id: str | None,
    ) -> None:
        request_id = event.get("requestId")
        if not request_id:
            return
        request = event.get("request")
        request_url = request.get("url") if isinstance(request, dict) else ""
        params: dict[str, Any] = {"requestId": request_id}
        rewritten_url = _rewrite_url_origin(str(request_url or ""), self.url_origin_rewrites)
        if rewritten_url and rewritten_url != request_url:
            params["url"] = rewritten_url
        effective_url = rewritten_url or str(request_url or "")
        headers_for_origin = self.headers_by_origin.get(_origin_from_url(effective_url))
        existing = request.get("headers") if isinstance(request, dict) else {}
        existing_headers = (
            {str(k): str(v) for k, v in existing.items()} if isinstance(existing, dict) else {}
        )
        headers, headers_changed = _headers_for_rewritten_request(
            existing_headers,
            original_url=str(request_url or ""),
            rewritten_url=effective_url,
        )
        if headers_for_origin:
            headers.update(headers_for_origin)
            headers_changed = True
        if headers_changed:
            params["headers"] = [
                {"name": name, "value": value}
                for name, value in headers.items()
            ]
        try:
            await self._browser_session.cdp_client.send.Fetch.continueRequest(
                params,
                session_id=session_id,
            )
        except Exception:
            logger.debug("scoped http_headers continueRequest failed", exc_info=True)


def _scoped_header_auth_action(origin: str, headers: dict[str, str]):
    async def _action(browser_session: Any) -> None:
        injector = await _ensure_scoped_request_mutator(
            browser_session,
            origin=origin,
            headers=headers,
        )
        browser_session._worldsim_scoped_header_auth = injector

    return _action


async def _ensure_scoped_request_mutator(
    browser_session: Any,
    *,
    origin: str = "",
    headers: dict[str, str] | None = None,
    url_origin_rewrites: dict[str, str] | None = None,
) -> _ScopedHeaderAuthInjector:
    injector = getattr(browser_session, "_worldsim_scoped_header_auth", None)
    if not isinstance(injector, _ScopedHeaderAuthInjector):
        injector = _ScopedHeaderAuthInjector(
            origin=origin,
            headers=headers,
            url_origin_rewrites=url_origin_rewrites,
        )
        await injector.start(browser_session)
        browser_session._worldsim_scoped_header_auth = injector
        return injector

    if origin and headers:
        injector.add_headers(origin, headers)
    injector.add_url_origin_rewrites(url_origin_rewrites)
    await injector._enable_current_page_sessions()
    return injector


def _normalize_origin_rewrites(rewrites: dict[str, str] | None) -> dict[str, str]:
    normalized: dict[str, str] = {}
    if not isinstance(rewrites, dict):
        return normalized
    for raw_origin, raw_target in rewrites.items():
        origin = _origin_from_url(str(raw_origin or ""))
        target = _origin_from_url(str(raw_target or ""))
        if origin and target and origin != target:
            normalized[origin] = target
    return normalized


def _rewrite_url_origin(url: str, origin_rewrites: dict[str, str]) -> str:
    try:
        parsed = urlsplit(str(url or ""))
    except ValueError:
        return str(url or "")
    if not parsed.scheme or not parsed.netloc:
        return str(url or "")
    origin = f"{parsed.scheme}://{parsed.netloc}"
    replacement = origin_rewrites.get(origin)
    if not replacement:
        return str(url or "")
    replacement_parts = urlsplit(replacement)
    return urlunsplit(
        (
            replacement_parts.scheme,
            replacement_parts.netloc,
            parsed.path,
            parsed.query,
            parsed.fragment,
        )
    )


def _headers_for_rewritten_request(
    headers: dict[str, str],
    *,
    original_url: str,
    rewritten_url: str,
) -> tuple[dict[str, str], bool]:
    """Return request headers consistent with a CDP URL-origin rewrite."""

    if not headers:
        return {}, False
    old_origin = _origin_from_url(original_url)
    new_origin = _origin_from_url(rewritten_url)
    if not old_origin or not new_origin or old_origin == new_origin:
        return dict(headers), False

    rewritten_headers = dict(headers)
    changed = False
    rewritten_parts = urlsplit(rewritten_url)
    for name, value in list(rewritten_headers.items()):
        lower = name.lower()
        if lower == "origin" and _origin_from_url(value) == old_origin:
            rewritten_headers[name] = new_origin
            changed = True
        elif lower in _URL_VALUE_HEADER_NAMES and _origin_from_url(value) == old_origin:
            rewritten_value = _rewrite_url_origin(value, {old_origin: new_origin})
            if rewritten_value != value:
                rewritten_headers[name] = rewritten_value
                changed = True
        elif lower == "host" and rewritten_parts.netloc and value != rewritten_parts.netloc:
            rewritten_headers[name] = rewritten_parts.netloc
            changed = True
    return rewritten_headers, changed


def _auth_sensitive_header_names(auth_mechanism: dict[str, Any] | None) -> set[str]:
    if not isinstance(auth_mechanism, dict):
        return set()
    mech_type = str(auth_mechanism.get("type") or "").strip()
    if mech_type == "http_basic":
        return {"Authorization"}
    if mech_type == "http_headers":
        try:
            return set(resolve_agent_auth_headers(auth_mechanism))
        except RuntimeError:
            return set()
    return set()


def _resolve_storage_state_path(
    raw_path: str,
    benchmark_root: Path | None,
    *,
    site_name: str,
    instance_id: str | None = None,
) -> Path:
    """Resolve a storage-state artifact path and enforce benchmark-root containment."""
    if not site_name:
        path = Path(raw_path)
        if path.is_absolute():
            if benchmark_root is None:
                raise AuthArtifactMissingError(
                    "absolute auth_mechanism.storage_state.path requires a benchmark root; "
                    "pass --benchmark so the runtime can validate containment"
                )
            resolved_root = Path(benchmark_root).resolve()
            try:
                path.resolve().relative_to(resolved_root)
            except ValueError as exc:
                raise AuthArtifactMissingError(
                    f"storage_state path {raw_path!r} resolves outside benchmark root {resolved_root}"
                ) from exc
            return path
        if benchmark_root is None:
            raise AuthArtifactMissingError(
                "relative auth_mechanism.storage_state.path requires a benchmark root; "
                "pass --benchmark so the runtime can resolve the artifact safely"
            )
        resolved_root = Path(benchmark_root).resolve()
        resolved = (resolved_root / path).resolve()
        try:
            resolved.relative_to(resolved_root)
        except ValueError as exc:
            raise AuthArtifactMissingError(
                f"storage_state path {raw_path!r} resolves outside benchmark root {resolved_root}"
            ) from exc
        return resolved

    path, error = _resolve_declared_storage_state_path(
        raw_path,
        benchmark_root=benchmark_root,
        site_name=site_name,
        instance_id=instance_id,
    )
    if error is not None or path is None:
        raise AuthArtifactMissingError(error or "storage_state path could not be resolved")
    return path


def _storage_state_site_error(path: Path, site_url: str | None) -> str | None:
    if not site_url:
        return None
    payload, error = read_storage_state_payload(path)
    if error is not None:
        return error
    return storage_state_preflight_error_for_payload(path, payload, site_url)


def _validate_storage_state_for_site(path: Path, site_url: str | None) -> None:
    error = _storage_state_site_error(path, site_url)
    if error is not None:
        raise AuthArtifactMissingError(error)


def _storage_state_context_value(path: Path, *, runtime_dir: Path | None = None) -> str:
    """Return a Browser-Use storage_state path after validating/normalizing JSON.

    Browser-Use accepts inline dicts for initial auth, but its
    StorageStateWatchdog needs a real file path for save-back. Phase 4 also
    runs many workers concurrently, so when a task directory is available we
    materialize a per-task copy instead of letting workers race on the shared
    Phase 0d artifact.
    """
    storage_state, error = playwright_storage_state(path)
    if error is not None:
        raise AuthArtifactMissingError(error)
    if not isinstance(storage_state, dict):
        return str(storage_state)
    if runtime_dir is None:
        target = path
    else:
        target = runtime_dir / "storage_state.json"
    write_json_atomic(target, storage_state)
    return str(target.resolve())


def _augment_storage_state_origin_aliases(
    path: str | Path,
    url_origin_rewrites: dict[str, str] | None,
) -> dict[str, Any]:
    """Copy target-origin auth state to same-site aliases used by browser links.

    GitLab replicas can emit absolute links for their baked-in canonical origin
    (for example ``localhost:8023``) even when WorldSim binds a task to
    ``172.17.0.1:<replica-port>``. CDP request rewriting keeps the network on
    the bound replica, but Chromium chooses cookies/localStorage before that
    rewrite. Mirroring the already validated storage state to alias origins
    keeps browser state consistent without changing the task or reward.
    """

    rewrites = _normalize_origin_rewrites(url_origin_rewrites)
    if not rewrites:
        return {"aliases": [], "cookies_added": 0, "origins_added": 0}

    payload, error = read_storage_state_payload(Path(path))
    if error is not None:
        raise AuthArtifactMissingError(error)
    if not isinstance(payload, dict):
        return {"aliases": [], "cookies_added": 0, "origins_added": 0}

    cookies = payload.get("cookies")
    if not isinstance(cookies, list):
        cookies = []
        payload["cookies"] = cookies
    origins = payload.get("origins")
    if not isinstance(origins, list):
        origins = []
        payload["origins"] = origins

    cookie_keys = {
        (
            str(cookie.get("name") or ""),
            str(cookie.get("domain") or ""),
            str(cookie.get("path") or "/"),
        )
        for cookie in cookies
        if isinstance(cookie, dict)
    }
    origin_values = {
        str(origin.get("origin") or "")
        for origin in origins
        if isinstance(origin, dict) and origin.get("origin")
    }

    aliases: list[dict[str, str]] = []
    cookies_added = 0
    origins_added = 0

    for alias_origin, target_origin in sorted(rewrites.items()):
        alias = urlsplit(alias_origin)
        target = urlsplit(target_origin)
        if not alias.hostname or not target.hostname:
            continue
        aliases.append({"alias": alias_origin, "target": target_origin})

        target_host = target.hostname
        alias_host = alias.hostname
        for cookie in list(cookies):
            if not isinstance(cookie, dict):
                continue
            domain = str(cookie.get("domain") or "")
            if not _storage_cookie_domain_matches_host(domain, target_host):
                continue
            cloned = dict(cookie)
            cloned["domain"] = _rewrite_storage_cookie_domain(domain, alias_host)
            key = (
                str(cloned.get("name") or ""),
                str(cloned.get("domain") or ""),
                str(cloned.get("path") or "/"),
            )
            if key in cookie_keys:
                continue
            cookies.append(cloned)
            cookie_keys.add(key)
            cookies_added += 1

        for origin in list(origins):
            if not isinstance(origin, dict):
                continue
            if str(origin.get("origin") or "") != target_origin:
                continue
            if alias_origin in origin_values:
                continue
            cloned_origin = dict(origin)
            cloned_origin["origin"] = alias_origin
            origins.append(cloned_origin)
            origin_values.add(alias_origin)
            origins_added += 1

    if cookies_added or origins_added:
        write_json_atomic(Path(path), payload)
    return {
        "aliases": aliases,
        "cookies_added": cookies_added,
        "origins_added": origins_added,
    }


def _storage_cookie_domain_matches_host(domain: str, host: str) -> bool:
    normalized_domain = domain.lstrip(".").lower()
    normalized_host = host.lower()
    return bool(
        normalized_domain
        and (
            normalized_domain == normalized_host
            or normalized_host.endswith(f".{normalized_domain}")
        )
    )


def _rewrite_storage_cookie_domain(domain: str, alias_host: str) -> str:
    return f".{alias_host}" if domain.startswith(".") else alias_host


def _storage_state_cdp_cookie_params(path: str | Path) -> list[dict[str, Any]]:
    payload, error = read_storage_state_payload(Path(path))
    if error is not None:
        raise AuthArtifactMissingError(error)
    cookies = payload.get("cookies")
    if not isinstance(cookies, list):
        return []

    out: list[dict[str, Any]] = []
    for cookie in cookies:
        if not isinstance(cookie, dict):
            continue
        name = cookie.get("name")
        value = cookie.get("value")
        if not isinstance(name, str) or not isinstance(value, str):
            continue
        item: dict[str, Any] = {"name": name, "value": value}
        for key in ("domain", "path", "sameSite"):
            raw = cookie.get(key)
            if isinstance(raw, str) and raw:
                item[key] = raw
        for key in ("secure", "httpOnly"):
            raw = cookie.get(key)
            if isinstance(raw, bool):
                item[key] = raw
        expires = cookie.get("expires")
        if isinstance(expires, int | float) and expires not in (0, 0.0, -1, -1.0):
            item["expires"] = expires
        out.append(item)
    return out


async def _restore_external_cdp_storage_state(session: Any, storage_state_path: str | Path) -> None:
    """Apply storage_state cookies after a CDP task target exists.

    Browser Use's StorageStateWatchdog restores cookies on BrowserConnectedEvent.
    In external-CDP PVPO mode that event can fire before an agent-focus target
    exists, so BrowserSession._cdp_set_cookies returns early. Re-applying after
    WorldSim creates the task-owned target makes auth deterministic.
    """

    cookies = _storage_state_cdp_cookie_params(storage_state_path)
    if not cookies:
        return
    cdp_client = getattr(session, "cdp_client", None)
    storage_sender = getattr(getattr(cdp_client, "send", None), "Storage", None)
    set_cookies = getattr(storage_sender, "setCookies", None)
    if not callable(set_cookies):
        raise AuthArtifactMissingError(
            "external CDP session does not expose Storage.setCookies for storage_state auth"
        )
    await set_cookies(params={"cookies": cookies})


def _pvpo_cdp_timeout_s() -> float:
    raw = os.environ.get(_PVPO_CDP_TIMEOUT_ENV, "").strip()
    if not raw:
        return _PVPO_CDP_TIMEOUT_DEFAULT_S
    try:
        timeout_s = float(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not a number; using %.1fs",
            _PVPO_CDP_TIMEOUT_ENV,
            raw,
            _PVPO_CDP_TIMEOUT_DEFAULT_S,
        )
        return _PVPO_CDP_TIMEOUT_DEFAULT_S
    if timeout_s <= 0:
        logger.warning(
            "%s=%r is not positive; using %.1fs",
            _PVPO_CDP_TIMEOUT_ENV,
            raw,
            _PVPO_CDP_TIMEOUT_DEFAULT_S,
        )
        return _PVPO_CDP_TIMEOUT_DEFAULT_S
    return timeout_s


def _pvpo_scroll_action_timeout_s() -> float:
    raw = os.environ.get(_PVPO_SCROLL_ACTION_TIMEOUT_ENV, "").strip()
    if not raw:
        return _PVPO_SCROLL_ACTION_TIMEOUT_DEFAULT_S
    try:
        timeout_s = float(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not a number; using %.1fs",
            _PVPO_SCROLL_ACTION_TIMEOUT_ENV,
            raw,
            _PVPO_SCROLL_ACTION_TIMEOUT_DEFAULT_S,
        )
        return _PVPO_SCROLL_ACTION_TIMEOUT_DEFAULT_S
    if timeout_s <= 0:
        logger.warning(
            "%s=%r is not positive; using %.1fs",
            _PVPO_SCROLL_ACTION_TIMEOUT_ENV,
            raw,
            _PVPO_SCROLL_ACTION_TIMEOUT_DEFAULT_S,
        )
        return _PVPO_SCROLL_ACTION_TIMEOUT_DEFAULT_S
    return timeout_s


def _pvpo_navigation_tick_interval_s() -> float:
    raw = os.environ.get(_PVPO_NAVIGATION_TICK_INTERVAL_ENV, "").strip()
    env_name = _PVPO_NAVIGATION_TICK_INTERVAL_ENV
    if not raw:
        # Keep the original emergency kill switch meaningful during the
        # transition away from the always-on sidecar pump.
        raw = os.environ.get("WORLDSIM_PVPO_FRAME_PUMP_MS", "").strip()
        env_name = "WORLDSIM_PVPO_FRAME_PUMP_MS"
    if not raw:
        return _PVPO_NAVIGATION_TICK_DEFAULT_MS / 1000.0
    try:
        return float(raw) / 1000.0
    except ValueError:
        logger.warning(
            "%s=%r is not a number; using default %.0fms",
            env_name,
            raw,
            _PVPO_NAVIGATION_TICK_DEFAULT_MS,
        )
        return _PVPO_NAVIGATION_TICK_DEFAULT_MS / 1000.0


def _pvpo_navigation_tick_stop_grace_s() -> float:
    """Allow an in-flight navigation tick to finish on its CDP timeout budget."""
    return max(_PVPO_NAVIGATION_TICK_STOP_GRACE_MIN_S, _pvpo_cdp_timeout_s() + 0.25)


def _pvpo_navigation_tick_beginframe_params() -> dict[str, Any]:
    """Return low-impact beginFrame params for navigation progress ticks."""
    return {"noDisplayUpdates": True}


def _install_pvpo_beginframe_screenshot_patch() -> None:
    """Patch Browser Use screenshots for begin-frame-controlled PVPO sessions.

    Browser Use's DOM watchdog dispatches ``ScreenshotEvent`` even when the
    agent runs with vision disabled. On ``chrome-headless-shell`` launched with
    ``--enable-begin-frame-control``, its default ``Page.captureScreenshot``
    path can hang indefinitely. PVPO sessions must capture via
    ``HeadlessExperimental.beginFrame({screenshot: ...})`` instead.
    """
    global _PVPO_SCREENSHOT_PATCHED
    if _PVPO_SCREENSHOT_PATCHED:
        return

    from browser_use.browser.watchdogs.screenshot_watchdog import ScreenshotWatchdog

    original = ScreenshotWatchdog.on_ScreenshotEvent

    async def _worldsim_on_screenshot_event(self: Any, event: Any) -> str:
        browser_session = getattr(self, "browser_session", None)
        if not getattr(browser_session, "cdp_url", None):
            return await original(self, event)
        if getattr(browser_session, "_worldsim_pvpo_disable_browser_use_screenshots", False):
            return _pvpo_browser_use_screenshot_fallback(browser_session)
        return await _capture_pvpo_beginframe_screenshot(self, event)

    _worldsim_on_screenshot_event.__name__ = "on_ScreenshotEvent"
    ScreenshotWatchdog.on_ScreenshotEvent = _worldsim_on_screenshot_event
    _PVPO_SCREENSHOT_PATCHED = True


def _install_pvpo_scroll_patch() -> None:
    """Patch Browser Use scroll gestures for begin-frame-controlled PVPO sessions.

    Browser Use target-level scrolling uses ``Input.synthesizeScrollGesture``.
    In ``chrome-headless-shell --enable-begin-frame-control`` that CDP call can
    hang until Browser Use's 8s ``ScrollEvent`` timeout. Browser Use's newer
    mouse actor already prefers ``Input.dispatchMouseEvent(type="mouseWheel")``
    before falling back; mirror that safer actuator only for PVPO external-CDP
    sessions. Non-PVPO Browser Use sessions keep the upstream implementation.
    """
    global _PVPO_SCROLL_PATCHED
    if _PVPO_SCROLL_PATCHED:
        return

    from browser_use.browser.watchdogs.default_action_watchdog import DefaultActionWatchdog

    original = DefaultActionWatchdog._scroll_with_cdp_gesture

    async def _worldsim_scroll_with_pvpo_fallback(self: Any, pixels: int) -> bool:
        browser_session = getattr(self, "browser_session", None)
        if not getattr(browser_session, "cdp_url", None):
            return await original(self, pixels)
        return await _pvpo_scroll_with_wheel_fallback(self, pixels)

    _worldsim_scroll_with_pvpo_fallback.__name__ = "_scroll_with_cdp_gesture"
    DefaultActionWatchdog._scroll_with_cdp_gesture = _worldsim_scroll_with_pvpo_fallback
    _PVPO_SCROLL_PATCHED = True


def _install_pvpo_navigation_tick_patch() -> None:
    """Patch Browser Use navigation to tick begin-frame-controlled PVPO pages.

    Browser Use waits for lifecycle events after ``Page.navigate``. In
    ``chrome-headless-shell --enable-begin-frame-control`` those events can
    stall unless the harness sends compositor frames. The old always-on pump
    solved navigation but could race later PVPO screenshots. This patch scopes
    the ticks to Browser Use's own navigation wait, which is the only period
    that needs a synthetic clock.
    """
    global _PVPO_NAVIGATION_TICK_PATCHED
    if _PVPO_NAVIGATION_TICK_PATCHED:
        return

    from browser_use.browser.session import BrowserSession

    original = BrowserSession._navigate_and_wait

    async def _worldsim_navigate_and_tick_pvpo(
        self: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if not getattr(self, "cdp_url", None):
            return await original(self, *args, **kwargs)

        from worldsim.phase_4.pvpo_beginframe import BeginFrameCoordinator

        target_id = kwargs.get("target_id")
        if target_id is None and len(args) >= 2:
            target_id = args[1]
        coordinator = getattr(self, "_worldsim_pvpo_beginframe_controller", None)
        interval_s = _pvpo_navigation_tick_interval_s()
        if (
            not isinstance(coordinator, BeginFrameCoordinator)
            or interval_s <= 0
            or not isinstance(target_id, str)
            or not target_id
        ):
            return await original(self, *args, **kwargs)

        _increment_session_counter(self, "_worldsim_pvpo_navigation_tick_navigations")
        stop = asyncio.Event()
        tick_task = asyncio.create_task(
            _pvpo_navigation_tick_loop(
                self,
                target_id=target_id,
                stop=stop,
                coordinator=coordinator,
                interval_s=interval_s,
            ),
            name="worldsim-pvpo-navigation-tick",
        )
        try:
            return await original(self, *args, **kwargs)
        finally:
            await _stop_pvpo_navigation_tick(
                self,
                tick_task=tick_task,
                stop=stop,
                coordinator=coordinator,
            )

    _worldsim_navigate_and_tick_pvpo.__name__ = "_navigate_and_wait"
    BrowserSession._navigate_and_wait = _worldsim_navigate_and_tick_pvpo
    _PVPO_NAVIGATION_TICK_PATCHED = True


async def _pvpo_navigation_tick_loop(
    browser_session: Any,
    *,
    target_id: str,
    stop: asyncio.Event,
    coordinator: Any,
    interval_s: float,
) -> None:
    from worldsim.phase_4.pvpo_beginframe import BeginFrameCoordinator, BeginFrameTimeout
    from worldsim.phase_4.pvpo_cdp import normalize_cdp_session

    if not isinstance(coordinator, BeginFrameCoordinator):
        return
    timeout_s = _pvpo_cdp_timeout_s()
    while not stop.is_set():
        try:
            capturing = getattr(browser_session, "_worldsim_pvpo_capturing_event", None)
            if isinstance(capturing, asyncio.Event) and capturing.is_set():
                _increment_session_counter(
                    browser_session,
                    "_worldsim_pvpo_navigation_tick_skipped_captures",
                )
            else:
                cdp_session = await browser_session.get_or_create_cdp_session(
                    target_id=target_id,
                    focus=False,
                )
                await coordinator.send(
                    normalize_cdp_session(cdp_session),
                    _pvpo_navigation_tick_beginframe_params(),
                    timeout_s=timeout_s,
                    label="navigation-tick",
                )
                _increment_session_counter(
                    browser_session,
                    "_worldsim_pvpo_navigation_tick_frames",
                )
        except asyncio.CancelledError:
            raise
        except BeginFrameTimeout as exc:
            _increment_session_counter(
                browser_session,
                "_worldsim_pvpo_navigation_tick_timeouts",
            )
            logger.debug("pvpo navigation tick: beginFrame timed out: %s", exc)
            return
        except Exception as exc:
            _increment_session_counter(
                browser_session,
                "_worldsim_pvpo_navigation_tick_failures",
            )
            logger.debug("pvpo navigation tick: beginFrame failed: %s", exc)

        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_s)
        except TimeoutError:
            pass


async def _stop_pvpo_navigation_tick(
    browser_session: Any,
    *,
    tick_task: asyncio.Task[Any],
    stop: asyncio.Event,
    coordinator: Any,
) -> None:
    try:
        if tick_task.done():
            with suppress(asyncio.CancelledError, Exception):
                await tick_task
            return

        stop.set()
        stop_grace_s = _pvpo_navigation_tick_stop_grace_s()
        try:
            await asyncio.wait_for(asyncio.shield(tick_task), timeout=stop_grace_s)
        except TimeoutError:
            _increment_session_counter(
                browser_session,
                "_worldsim_pvpo_navigation_tick_stop_timeouts",
            )
            reason = (
                "pvpo navigation beginFrame tick did not stop after "
                f"{stop_grace_s:.2f}s"
            )
            mark_dirty = getattr(coordinator, "mark_dirty", None)
            if callable(mark_dirty):
                mark_dirty(reason)
            tick_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await tick_task
        except asyncio.CancelledError:
            tick_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await tick_task
            raise
        except Exception as exc:
            logger.debug("pvpo navigation tick: teardown raised: %s", exc)
    finally:
        await _drain_pvpo_beginframe_after_auxiliary_frame(
            browser_session,
            coordinator,
            label="post-navigation-tick",
            timeout_counter_name="_worldsim_pvpo_navigation_tick_drain_timeouts",
            failure_counter_name="_worldsim_pvpo_navigation_tick_drain_failures",
        )


async def _pvpo_scroll_with_wheel_fallback(watchdog: Any, pixels: int) -> bool:
    browser_session = watchdog.browser_session
    try:
        cdp_session = await browser_session.get_or_create_cdp_session()
    except Exception as exc:
        logger.debug("PVPO scroll: CDP session unavailable: %s", exc)
        _increment_session_counter(browser_session, "_worldsim_pvpo_scroll_failures")
        return False

    timeout_s = _pvpo_scroll_action_timeout_s()
    before_state = await _pvpo_get_scroll_state(browser_session, cdp_session)
    try:
        await _pvpo_await_cdp_action(
            _pvpo_dispatch_mouse_wheel(browser_session, cdp_session, pixels),
            timeout_s=timeout_s,
            browser_session=browser_session,
            label="PVPO mouseWheel scroll",
        )
        after_state = await _pvpo_get_scroll_state(browser_session, cdp_session)
        if _pvpo_scroll_state_satisfies_request(before_state, after_state, pixels):
            _increment_session_counter(browser_session, "_worldsim_pvpo_scroll_wheel_successes")
            return True
        _increment_session_counter(browser_session, "_worldsim_pvpo_scroll_wheel_noops")
        _increment_session_counter(browser_session, "_worldsim_pvpo_scroll_js_fallbacks")
        logger.debug(
            "PVPO scroll: mouseWheel returned without expected root scroll; using JS fallback "
            "(before=%s after=%s pixels=%s)",
            before_state,
            after_state,
            pixels,
        )
        return await _pvpo_scroll_with_js(browser_session, cdp_session, pixels)
    except TimeoutError:
        _increment_session_counter(browser_session, "_worldsim_pvpo_scroll_wheel_timeouts")
        after_timeout_state = await _pvpo_get_scroll_state(browser_session, cdp_session)
        if _pvpo_scroll_state_satisfies_request(before_state, after_timeout_state, pixels):
            _increment_session_counter(browser_session, "_worldsim_pvpo_scroll_wheel_late_successes")
            logger.debug(
                "PVPO scroll: mouseWheel timed out after %.2fs but root scroll position changed; "
                "skipping JS fallback (before=%s after=%s pixels=%s)",
                timeout_s,
                before_state,
                after_timeout_state,
                pixels,
            )
            return True
        fallback_count = _increment_session_counter(browser_session, "_worldsim_pvpo_scroll_js_fallbacks")
        if fallback_count == 1:
            logger.warning(
                "PVPO scroll: Input.dispatchMouseEvent(mouseWheel) timed out after %.2fs; "
                "using JS scroll fallback for this begin-frame-controlled session",
                timeout_s,
            )
        else:
            logger.debug(
                "PVPO scroll: Input.dispatchMouseEvent(mouseWheel) timed out after %.2fs; "
                "using JS scroll fallback",
                timeout_s,
            )
        return await _pvpo_scroll_with_js(browser_session, cdp_session, pixels)
    except Exception as exc:
        _increment_session_counter(browser_session, "_worldsim_pvpo_scroll_wheel_failures")
        logger.debug("PVPO scroll: CDP mouseWheel failed (%s); using JS fallback", exc)
        _increment_session_counter(browser_session, "_worldsim_pvpo_scroll_js_fallbacks")
        return await _pvpo_scroll_with_js(browser_session, cdp_session, pixels)


async def _pvpo_dispatch_mouse_wheel(
    browser_session: Any,
    cdp_session: Any,
    pixels: int,
) -> None:
    cdp_client = cdp_session.cdp_client
    session_id = cdp_session.session_id
    if getattr(browser_session, "_original_viewport_size", None):
        viewport_width, viewport_height = browser_session._original_viewport_size
    else:
        layout_metrics = await cdp_client.send.Page.getLayoutMetrics(session_id=session_id)
        viewport_width = layout_metrics["layoutViewport"]["clientWidth"]
        viewport_height = layout_metrics["layoutViewport"]["clientHeight"]

    await cdp_client.send.Input.dispatchMouseEvent(
        params={
            "type": "mouseWheel",
            "x": viewport_width / 2,
            "y": viewport_height / 2,
            "deltaX": 0,
            "deltaY": pixels,
        },
        session_id=session_id,
    )


async def _pvpo_await_cdp_action(
    awaitable: Any,
    *,
    timeout_s: float,
    browser_session: Any | None = None,
    label: str | None = None,
) -> Any:
    return await _await_pvpo_cdp_deadline(
        awaitable,
        timeout_s=timeout_s,
        label=label,
        browser_session=browser_session,
    )


async def _await_pvpo_cdp_deadline(
    awaitable: Any,
    *,
    timeout_s: float,
    label: str | None = None,
    browser_session: Any | None = None,
) -> Any:
    """Bound a CDP-adjacent awaitable without cancelling the protocol future.

    ``cdp_use`` stores one future per request id. Plain ``asyncio.wait_for``
    cancels that future on timeout, so a later Chrome response is logged by
    cdp_use as a misleading "duplicate response". WorldSim timeouts are local
    deadlines: leave the CDP future alive, attach a late-result drain callback,
    and let the owning browser recycle clean up genuinely wedged commands.
    """
    task = asyncio.ensure_future(awaitable)
    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout=timeout_s)
    except TimeoutError as exc:
        if browser_session is not None:
            _increment_session_counter(browser_session, "_worldsim_pvpo_cdp_timeouts")
        task.add_done_callback(
            lambda done_task: _suppress_late_cdp_task_result(
                done_task,
                browser_session=browser_session,
            )
        )
        if label:
            raise TimeoutError(f"{label} timed out after {timeout_s:.2f}s") from exc
        raise
    except asyncio.CancelledError:
        if not task.done():
            task.add_done_callback(
                lambda done_task: _suppress_late_cdp_task_result(
                    done_task,
                    browser_session=browser_session,
                )
            )
        raise


def _suppress_late_cdp_task_result(
    task: asyncio.Future[Any],
    *,
    browser_session: Any | None = None,
) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        if browser_session is not None:
            _increment_session_counter(
                browser_session,
                "_worldsim_pvpo_cdp_late_cancellations",
            )
    except Exception:
        if browser_session is not None:
            _increment_session_counter(
                browser_session,
                "_worldsim_pvpo_cdp_late_failures",
            )
    else:
        if browser_session is not None:
            _increment_session_counter(
                browser_session,
                "_worldsim_pvpo_cdp_late_completions",
            )


async def _pvpo_get_scroll_state(browser_session: Any, cdp_session: Any) -> _PvpoScrollState | None:
    cdp_client = cdp_session.cdp_client
    session_id = cdp_session.session_id
    expression = """
(() => {
  const root = document.scrollingElement || document.documentElement || document.body;
  if (!root) {
    return {success: false, error: "no scrolling element"};
  }
  const scrollHeight = Number(root.scrollHeight || 0);
  const viewportHeight = Number(window.innerHeight || root.clientHeight || 0);
  return {
    success: true,
    x: Number(window.scrollX || root.scrollLeft || 0),
    y: Number(window.scrollY || root.scrollTop || 0),
    maxY: Math.max(0, scrollHeight - viewportHeight)
  };
})()
"""
    try:
        result = await _pvpo_await_cdp_action(
            cdp_client.send.Runtime.evaluate(
                params={"expression": expression, "returnByValue": True, "awaitPromise": True},
                session_id=session_id,
            ),
            timeout_s=_pvpo_scroll_action_timeout_s(),
            browser_session=browser_session,
            label="PVPO scroll state probe",
        )
    except Exception as exc:
        logger.debug("PVPO scroll: could not read root scroll state: %s", exc)
        return None
    if isinstance(result, dict) and result.get("exceptionDetails"):
        logger.debug("PVPO scroll: root scroll state exception: %s", result.get("exceptionDetails"))
        return None
    value = result.get("result", {}).get("value") if isinstance(result, dict) else None
    if not isinstance(value, dict) or value.get("success") is not True:
        logger.debug("PVPO scroll: root scroll state unavailable: %s", value)
        return None
    try:
        return _PvpoScrollState(
            x=float(value.get("x", 0)),
            y=float(value.get("y", 0)),
            max_y=max(0.0, float(value.get("maxY", 0))),
        )
    except (TypeError, ValueError):
        logger.debug("PVPO scroll: invalid root scroll state payload: %s", value)
        return None


def _pvpo_scroll_state_satisfies_request(
    before: _PvpoScrollState | None,
    after: _PvpoScrollState | None,
    pixels: int,
) -> bool:
    """Return whether the observed root scroll movement matches Browser Use's request.

    Unknown state is treated as success because the CDP actuator itself
    returned. Known state lets us avoid both false-positive no-op wheel
    responses and duplicate JS fallback when a timed-out wheel already moved.
    """
    if before is None or after is None or pixels == 0:
        return True
    if before.max_y <= _PVPO_SCROLL_EPSILON_PX and after.max_y <= _PVPO_SCROLL_EPSILON_PX:
        return True
    if pixels > 0:
        if after.y > before.y + _PVPO_SCROLL_EPSILON_PX:
            return True
        return before.y >= before.max_y - _PVPO_SCROLL_EPSILON_PX or after.y >= after.max_y - _PVPO_SCROLL_EPSILON_PX
    if after.y < before.y - _PVPO_SCROLL_EPSILON_PX:
        return True
    return before.y <= _PVPO_SCROLL_EPSILON_PX or after.y <= _PVPO_SCROLL_EPSILON_PX


async def _pvpo_scroll_with_js(browser_session: Any, cdp_session: Any, pixels: int) -> bool:
    cdp_client = cdp_session.cdp_client
    session_id = cdp_session.session_id
    expression = f"""
(() => {{
  const dy = {int(pixels)};
  const root = document.scrollingElement || document.documentElement || document.body;
  if (!root) {{
    return {{success: false, error: "no scrolling element"}};
  }}
  const beforeY = Number(window.scrollY || root.scrollTop || 0);
  const beforeX = Number(window.scrollX || root.scrollLeft || 0);
  window.scrollBy({{left: 0, top: dy, behavior: "instant"}});
  const afterY = Number(window.scrollY || root.scrollTop || 0);
  const afterX = Number(window.scrollX || root.scrollLeft || 0);
  const maxY = Math.max(0, Number(root.scrollHeight || 0) - Number(window.innerHeight || 0));
  return {{
    success: true,
    beforeX,
    beforeY,
    afterX,
    afterY,
    scrolledX: afterX - beforeX,
    scrolledY: afterY - beforeY,
    maxY
  }};
}})()
"""
    try:
        result = await _pvpo_await_cdp_action(
            cdp_client.send.Runtime.evaluate(
                params={"expression": expression, "returnByValue": True, "awaitPromise": True},
                session_id=session_id,
            ),
            timeout_s=_pvpo_scroll_action_timeout_s(),
            browser_session=browser_session,
            label="PVPO JS scroll fallback",
        )
    except Exception as exc:
        logger.debug("PVPO scroll: JS fallback failed: %s", exc)
        _increment_session_counter(browser_session, "_worldsim_pvpo_scroll_js_failures")
        return False
    if isinstance(result, dict) and result.get("exceptionDetails"):
        logger.debug("PVPO scroll: JS fallback exception: %s", result.get("exceptionDetails"))
        _increment_session_counter(browser_session, "_worldsim_pvpo_scroll_js_failures")
        return False
    value = result.get("result", {}).get("value") if isinstance(result, dict) else None
    if isinstance(value, dict) and value.get("success") is False:
        logger.debug("PVPO scroll: JS fallback reported failure: %s", value)
        _increment_session_counter(browser_session, "_worldsim_pvpo_scroll_js_failures")
        return False
    if isinstance(value, dict):
        try:
            before = _PvpoScrollState(
                x=float(value.get("beforeX", 0)),
                y=float(value.get("beforeY", 0)),
                max_y=max(0.0, float(value.get("maxY", 0))),
            )
            after = _PvpoScrollState(
                x=float(value.get("afterX", 0)),
                y=float(value.get("afterY", 0)),
                max_y=max(0.0, float(value.get("maxY", 0))),
            )
        except (TypeError, ValueError):
            before = after = None
        if not _pvpo_scroll_state_satisfies_request(before, after, pixels):
            logger.debug("PVPO scroll: JS fallback did not move root scroll as requested: %s", value)
            _increment_session_counter(browser_session, "_worldsim_pvpo_scroll_js_noops")
            return False
    return True


def _increment_session_counter(session: Any, name: str) -> int:
    if session is None:
        return 0
    current = getattr(session, name, 0)
    if not isinstance(current, int):
        current = 0
    updated = current + 1
    setattr(session, name, updated)
    return updated


async def _drain_pvpo_beginframe_after_auxiliary_frame(
    browser_session: Any,
    coordinator: Any,
    *,
    label: str,
    timeout_counter_name: str,
    failure_counter_name: str,
) -> None:
    """Quiesce non-measurement beginFrame work before PVPO measurement resumes."""
    from worldsim.phase_4.pvpo_beginframe import BeginFrameCoordinator, BeginFrameTimeout

    if not isinstance(coordinator, BeginFrameCoordinator):
        return
    try:
        await coordinator.drain_prior(label=label)
    except BeginFrameTimeout as exc:
        _increment_session_counter(browser_session, timeout_counter_name)
        logger.debug("pvpo %s: prior beginFrame drain timed out: %s", label, exc)
    except Exception as exc:
        _increment_session_counter(browser_session, failure_counter_name)
        logger.debug("pvpo %s: prior beginFrame drain failed: %s", label, exc)


async def _capture_pvpo_beginframe_screenshot(watchdog: Any, event: Any) -> str:
    """Handle Browser Use ``ScreenshotEvent`` using beginFrame screenshots."""
    from browser_use.browser.views import BrowserError

    from worldsim.phase_4.pvpo_beginframe import BeginFrameCoordinator, is_beginframe_pending_error

    browser_session = watchdog.browser_session
    capturing = getattr(browser_session, "_worldsim_pvpo_capturing_event", None)
    if isinstance(capturing, asyncio.Event) and capturing.is_set():
        return _pvpo_browser_use_screenshot_fallback(browser_session)
    focused_target = browser_session.get_focused_target()
    if focused_target and focused_target.target_type in ("page", "tab"):
        target_id = focused_target.target_id
    else:
        page_targets = browser_session.get_page_targets()
        if not page_targets:
            raise BrowserError("[PVPO ScreenshotWatchdog] No page targets available")
        target_id = page_targets[-1].target_id

    cdp_session = await browser_session.get_or_create_cdp_session(target_id, focus=True)
    try:
        await browser_session.remove_highlights()
    except Exception:
        pass
    if isinstance(capturing, asyncio.Event) and capturing.is_set():
        return _pvpo_browser_use_screenshot_fallback(browser_session)

    screenshot: dict[str, Any] = {"format": "png"}
    clip = getattr(event, "clip", None)
    if clip:
        screenshot["clip"] = {
            "x": clip["x"],
            "y": clip["y"],
            "width": clip["width"],
            "height": clip["height"],
            "scale": 1,
        }

    cdp = None

    coordinator = getattr(browser_session, "_worldsim_pvpo_beginframe_controller", None)

    async def _send_raw() -> dict[str, Any]:
        nonlocal cdp
        if cdp is None:
            from worldsim.phase_4.pvpo_cdp import normalize_cdp_session

            cdp = normalize_cdp_session(cdp_session)
        return await cdp.send("HeadlessExperimental.beginFrame", {"screenshot": screenshot})

    async def _capture() -> dict[str, Any]:
        lock = getattr(browser_session, "_worldsim_pvpo_beginframe_lock", None)
        if lock is not None:
            async with lock:
                return await _send_raw()
        return await _send_raw()

    try:
        if isinstance(coordinator, BeginFrameCoordinator):
            if cdp is None:
                from worldsim.phase_4.pvpo_cdp import normalize_cdp_session

                cdp = normalize_cdp_session(cdp_session)
            result = await coordinator.send(
                cdp,
                {"screenshot": screenshot},
                timeout_s=_pvpo_cdp_timeout_s(),
                label="browser-use-screenshot",
            )
        else:
            result = await _await_pvpo_cdp_deadline(
                _capture(),
                timeout_s=_pvpo_cdp_timeout_s(),
                label="PVPO browser-use screenshot",
                browser_session=browser_session,
            )
    except Exception as exc:
        if is_beginframe_pending_error(exc) or isinstance(exc, TimeoutError):
            await _drain_pvpo_beginframe_after_auxiliary_frame(
                browser_session,
                coordinator,
                label="post-browser-use-screenshot",
                timeout_counter_name=(
                    "_worldsim_pvpo_browser_use_screenshot_drain_timeouts"
                ),
                failure_counter_name=(
                    "_worldsim_pvpo_browser_use_screenshot_drain_failures"
                ),
            )
            return _pvpo_browser_use_screenshot_fallback(browser_session)
        raise
    data = result.get("screenshotData") if isinstance(result, dict) else None
    if isinstance(data, str) and data:
        browser_session._worldsim_pvpo_last_browser_use_screenshot = data
        return data
    raise BrowserError("[PVPO ScreenshotWatchdog] beginFrame screenshot missing data")


def _pvpo_browser_use_screenshot_fallback(browser_session: Any) -> str:
    cached = getattr(browser_session, "_worldsim_pvpo_last_browser_use_screenshot", None)
    if isinstance(cached, str) and cached:
        return cached
    return _TRANSPARENT_PNG_BASE64


async def _await_pvpo_cdp(
    awaitable: Any,
    *,
    timeout_s: float,
    label: str | None = None,
    browser_session: Any | None = None,
) -> Any:
    return await _await_pvpo_cdp_deadline(
        awaitable,
        timeout_s=timeout_s,
        label=label,
        browser_session=browser_session,
    )


def _pvpo_issue_message(exc: BaseException, *, timeout_s: float) -> str:
    message = str(exc)
    if isinstance(exc, TimeoutError) and not message:
        return f"timed out after {timeout_s:.2f}s"
    return message or type(exc).__name__


def _pvpo_beginframe_dirty_reason(session: Any) -> str | None:
    try:
        from worldsim.phase_4.pvpo_beginframe import BeginFrameCoordinator

        controller = getattr(session, "_worldsim_pvpo_beginframe_controller", None)
        if isinstance(controller, BeginFrameCoordinator):
            return controller.dirty_reason
    except Exception:
        return None
    return None


# Stable enum of first-batch implementations. Types outside this set are schema-
# legal (validator accepts them) but raise ``NotImplementedError`` at runtime
# until their dispatcher arm is written. See plan §8 rollout order.
_IMPLEMENTED_AUTH_TYPES = frozenset({"storage_state", "http_basic", "http_headers", "none"})
_UNIMPLEMENTED_AUTH_TYPES = frozenset({"form_login", "pre_auth_script", "client_cert"})


def _resolve_auth(
    auth_mechanism: dict[str, Any] | None,
    task: dict[str, Any] | None,
    benchmark_root: Path | None,
    site_url: str | None = None,
    storage_state_runtime_dir: Path | None = None,
    instance_id: str | None = None,
) -> tuple[dict[str, Any], list[Any]]:
    """Translate an ``auth_mechanism`` dict into BrowserSession kwargs + deferred actions.

    Returns ``(session_kwargs, deferred_actions)``. ``session_kwargs`` is merged
    into the ``BrowserSession(...)`` call. ``deferred_actions`` is a list of
    async callables that receive the started session (unused for the first
    batch — reserved for ``form_login`` / ``pre_auth_script``).

    First-batch implementations: ``storage_state``, ``http_basic``, ``http_headers``,
    ``none``.
    ``unknown`` also no-ops (runtime has already been gated by
    ``--allow-unknown-auth`` in ``main.py``).

    Raises ``NotImplementedError`` for stub types (``form_login``,
    ``pre_auth_script``, ``client_cert``, ``http_headers``) with a clear
    pointer to the plan.
    Raises :class:`AuthArtifactMissingError` when a declared storage_state
    path is absent and no generator is declared.
    """
    session_kwargs: dict[str, Any] = {}
    deferred_actions: list[Any] = []

    # No declared auth_mechanism: no-op (legacy pipeline path).
    if not auth_mechanism:
        return session_kwargs, deferred_actions

    mech_type = auth_mechanism.get("type")
    if mech_type in (None, "", "none", "unknown"):
        return session_kwargs, deferred_actions

    if mech_type in _UNIMPLEMENTED_AUTH_TYPES:
        raise NotImplementedError(
            f"auth_mechanism.type={mech_type!r} is schema-legal but the runtime "
            "dispatcher has not been implemented yet. See plan §8 — only "
            "storage_state, http_basic, http_headers, and none ship in the first batch."
        )

    if mech_type == "storage_state":
        sub = auth_mechanism.get("storage_state") or {}
        # per_task_refresh is schema-legal but unimplemented: no current consumer
        # on the roadmap (verified 2026-04 across adapter branches + grep). Defer
        # with a clear runtime error so sites that flip it on see the gap
        # immediately instead of getting silent stale auth.
        if bool(sub.get("per_task_refresh")):
            raise NotImplementedError(
                "auth_mechanism.storage_state.per_task_refresh=True is not yet "
                "implemented at runtime. No benchmark on the current roadmap "
                "requires per-task regeneration; Phase 0d produces one-shot "
                "artifacts. Remove per_task_refresh from AGENT_CONTEXT or wire "
                "it through BrowserUseAgent.run() / Phase 0d before re-enabling."
            )
        raw_path = sub.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise AuthArtifactMissingError(
                "auth_mechanism.storage_state.path is empty; validator should have caught this"
            )
        bootstrap_path = _phase_0d_fallback_path(task, instance_id=instance_id)
        site_name = str(task.get("site") or "") if isinstance(task, dict) else ""
        bootstrap_error: AuthArtifactMissingError | None = None
        declared_error: AuthArtifactMissingError | None = None
        try:
            path = _resolve_storage_state_path(
                raw_path.strip(),
                benchmark_root,
                site_name=site_name,
                instance_id=instance_id,
            )
        except AuthArtifactMissingError as exc:
            path = None
            declared_error = exc
        # storage_state wins over any form_login that may coexist (plan §5 edges).
        if path is not None and path.exists():
            error = _storage_state_site_error(path, site_url)
            if error is None:
                session_kwargs["storage_state"] = _storage_state_context_value(
                    path,
                    runtime_dir=storage_state_runtime_dir,
                )
                return session_kwargs, deferred_actions
            declared_error = AuthArtifactMissingError(error)

        if declared_error is not None and path is None:
            raise declared_error

        if bootstrap_path is not None:
            try:
                _validate_storage_state_for_site(bootstrap_path, site_url)
                session_kwargs["storage_state"] = _storage_state_context_value(
                    bootstrap_path,
                    runtime_dir=storage_state_runtime_dir,
                )
                return session_kwargs, deferred_actions
            except AuthArtifactMissingError as exc:
                bootstrap_error = exc
                logger.warning(
                    "Phase 0d storage_state %s is unusable (%s); checking declared path",
                    bootstrap_path,
                    exc,
                )

        if path is None:
            raise declared_error or AuthArtifactMissingError(
                "storage_state path could not be resolved"
            )

        if not path.exists():
            generator = sub.get("generator_script")
            if bootstrap_error is not None:
                raise AuthArtifactMissingError(str(bootstrap_error))
            # Phase 0d writes the generated artifact to
            # ``logs/phase_0d/<site>/storage_state.json``. Consult that path as
            # a fallback before declaring the artifact missing so the runtime
            # picks up bootstrapped credentials automatically.
            if generator:
                raise AuthArtifactMissingError(
                    f"storage_state artifact missing at {path}; generator_script "
                    f"{generator!r} declared — run Phase 0d (auth-bootstrap) "
                    "before Phase 3."
                )
            raise AuthArtifactMissingError(
                f"storage_state artifact missing at {path} and no generator_script declared"
            )
        raise declared_error or AuthArtifactMissingError(str(path))

    if mech_type == "http_basic":
        sub = auth_mechanism.get("http_basic") or {}
        username = sub.get("username")
        password = sub.get("password")
        if not username or not password:
            raise AuthArtifactMissingError(
                "auth_mechanism.http_basic requires non-empty username/password"
            )
        origin = _origin_from_url(site_url or "")
        if not origin:
            raise AuthArtifactMissingError(
                "auth_mechanism.http_basic requires a valid site_url to scope credentials"
            )
        encoded = base64.b64encode(f"{username}:{password}".encode()).decode("ascii")
        deferred_actions.append(
            _scoped_header_auth_action(origin, {"Authorization": f"Basic {encoded}"})
        )
        return session_kwargs, deferred_actions

    if mech_type == "http_headers":
        try:
            resolved = resolve_agent_auth_headers(auth_mechanism)
        except RuntimeError as exc:
            raise AuthArtifactMissingError(str(exc)) from exc
        origin = _origin_from_url(site_url or "")
        if not origin:
            raise AuthArtifactMissingError(
                "auth_mechanism.http_headers requires a valid site_url to scope headers"
            )
        deferred_actions.append(_scoped_header_auth_action(origin, resolved))
        return session_kwargs, deferred_actions

    # Defensive fallback: an unknown-but-enumerated type slipped through.
    raise NotImplementedError(f"auth_mechanism.type={mech_type!r} has no runtime dispatcher")


def resolve_instance_agent_auth(instance: dict[str, Any]) -> dict[str, Any] | None:
    """Derive a ``_resolve_auth``-compatible auth_mechanism from instance config.

    Returns an auth_mechanism dict that is passed directly as
    ``run_kwargs["auth_mechanism"]`` to ``BrowserUseAgent.run``. This makes
    ``instances.json`` the single source of truth for auth.

    Returns ``None`` if the instance has no ``agent_auth`` configured. In that
    case the task runs without auth (no silent fallback to Phase 0c).
    """
    agent_auth = instance.get("agent_auth")
    if not has_configured_agent_auth(agent_auth):
        return None
    return agent_auth


class BrowserUseAgent:
    """Browser Use-backed :class:`AgentRunner` (spec canonical implementation)."""

    def __init__(
        self,
        llm: Any,
        *,
        use_vision: bool = False,
        max_steps: int = 50,
        timeout: int = 10800,
        llm_timeout: int | None = None,
        step_timeout: int | None = None,
        headless: bool = True,
    ) -> None:
        # ``llm`` is configured by the caller (see worldsim/main.py).
        # To use OpenRouter, pass an OpenRouter-configured ChatOpenAI instance.
        self.llm = llm
        self.use_vision = use_vision
        self.max_steps = max_steps
        self.timeout = timeout
        self.llm_timeout = llm_timeout
        self.step_timeout = step_timeout
        self.headless = headless
        self._session: Any = None
        self._pvpo_cdp_url: str = ""
        self._task_origins: set[str] = set()
        self._owned_target_ids: set[str] = set()
        self._primary_target_id: str | None = None
        self._browser_runtime: dict[str, Any] = {}
        self._preserve_remote_auth_state = False
        # Surface the configured model slug at construction. If the run 404s
        # mid-task, this is the first thing to check against the provider's
        # currently-served model list. An allowlist would rot faster than
        # provider catalogs update, so we only log.
        model_slug = getattr(llm, "model", None)
        if isinstance(model_slug, str) and model_slug:
            logger.info("BrowserUseAgent configured with model=%r", model_slug)
        if llm_timeout is not None:
            logger.info("BrowserUseAgent configured with llm_timeout=%ss", llm_timeout)
        if step_timeout is not None:
            logger.info("BrowserUseAgent configured with step_timeout=%ss", step_timeout)
        if timeout != 10800:
            logger.info("BrowserUseAgent configured with task_timeout=%ss", timeout)

    async def setup(self, server_url: str) -> None:
        # Browser sessions are task-scoped so trajectory artifacts remain
        # isolated per task directory.
        self._session = None
        self._pvpo_cdp_url = ""
        self._task_origins = set()
        self._owned_target_ids = set()
        self._primary_target_id = None
        self._browser_runtime = {}
        self._preserve_remote_auth_state = False

    async def run(
        self,
        task: str,
        server_url: str,
        task_dir: Path,
        *,
        start_urls: list[str] | None = None,
        site_prompt: str | None = None,
        auth_mechanism: dict[str, Any] | None = None,
        benchmark_root: Path | None = None,
        task_site: str | None = None,
        payload_text: str | None = None,
        payload_witnesses: list[str | dict[str, Any]] | None = None,
        pvpo_cdp_url: str | None = None,
        instance_id: str | None = None,
        url_origin_rewrites: dict[str, str] | None = None,
    ) -> AgentResult:
        from browser_use import Agent, BrowserSession

        task_dir = Path(task_dir)
        task_dir.mkdir(parents=True, exist_ok=True)
        self._task_origins = {
            origin for origin in (_origin_from_url(url) for url in (start_urls or [])) if origin
        }
        trusted_origin = _origin_from_url(server_url)
        if has_effective_agent_auth(auth_mechanism) and trusted_origin:
            off_origin = sorted(origin for origin in self._task_origins if origin != trusted_origin)
            if off_origin:
                raise AuthArtifactMissingError(
                    f"authenticated BrowserUse run received off-origin start_urls: {off_origin}"
                )
        self._owned_target_ids = set()
        self._primary_target_id = None
        self._browser_runtime = {}
        if self.llm_timeout is not None:
            self._browser_runtime["browser_use_llm_timeout_s"] = self.llm_timeout
        if self.step_timeout is not None:
            self._browser_runtime["browser_use_step_timeout_s"] = self.step_timeout
        pvpo_endpoint_lease_cm: Any = None
        pvpo_beginframe_controller: Any = None

        async def _release_pvpo_endpoint_lease() -> None:
            nonlocal pvpo_endpoint_lease_cm
            if pvpo_endpoint_lease_cm is None:
                return
            lease = pvpo_endpoint_lease_cm
            pvpo_endpoint_lease_cm = None
            await lease.__aexit__(None, None, None)

        # Resolve the declared auth_mechanism into BrowserSession kwargs +
        # deferred post-start actions. Errors here surface before we spin up a
        # browser so they fail fast and cheaply.
        # ``task_site`` (when present) lets ``_resolve_auth`` fall back to the
        # Phase 0d-bootstrapped artifact at
        # ``logs/phase_0d/<site>/storage_state.json`` when the declared path is
        # missing.
        resolve_task = {"site": task_site} if task_site else None
        session_auth_kwargs, deferred_auth_actions = _resolve_auth(
            auth_mechanism,
            task=resolve_task,
            benchmark_root=benchmark_root,
            site_url=server_url,
            storage_state_runtime_dir=task_dir / "auth",
            instance_id=instance_id,
        )
        if url_origin_rewrites and isinstance(session_auth_kwargs.get("storage_state"), str):
            alias_summary = _augment_storage_state_origin_aliases(
                session_auth_kwargs["storage_state"],
                url_origin_rewrites,
            )
            if alias_summary.get("aliases"):
                self._browser_runtime["storage_state_origin_aliases"] = alias_summary
        auth_sensitive_header_names = _auth_sensitive_header_names(auth_mechanism)
        if deferred_auth_actions and len(start_urls or []) > 1:
            raise AuthArtifactMissingError(
                "authenticated BrowserUse run with header-based auth cannot safely "
                "open multiple start_urls because new-tab first requests may precede "
                "CDP Fetch attachment"
            )
        # PVPO integration: Phase 4 binds each worker to its own chrome-
        # headless-shell endpoint via the instance config. The shared global
        # WORLDSIM_PVPO_CDP_URL path is intentionally removed.
        resolved_pvpo_cdp_url = _resolve_pvpo_cdp_url(pvpo_cdp_url or "")
        self._pvpo_cdp_url = resolved_pvpo_cdp_url
        self._preserve_remote_auth_state = "storage_state" in session_auth_kwargs
        external_cdp_storage_state = (
            session_auth_kwargs.get("storage_state")
            if resolved_pvpo_cdp_url and isinstance(session_auth_kwargs.get("storage_state"), str)
            else None
        )
        session_kwargs: dict[str, Any] = {
            "headless": self.headless,
            "keep_alive": False,
            **session_auth_kwargs,
        }
        if resolved_pvpo_cdp_url:
            session_kwargs["cdp_url"] = resolved_pvpo_cdp_url
            _install_pvpo_beginframe_screenshot_patch()
            _install_pvpo_scroll_patch()
            _install_pvpo_navigation_tick_patch()
            from worldsim.phase_4.pvpo_beginframe import pvpo_endpoint_lease

            pvpo_endpoint_lease_cm = pvpo_endpoint_lease(resolved_pvpo_cdp_url)
            pvpo_beginframe_controller = await pvpo_endpoint_lease_cm.__aenter__()
            self._browser_runtime.update(pvpo_beginframe_controller.stats())
        else:
            session_kwargs["args"] = [
                "--disable-gpu",
                "--disable-extensions",
                "--no-sandbox",  # required for Chrome on EC2/Docker/root
                "--disable-software-rasterizer",  # reduce CPU when GPU unavailable
            ]
        try:
            self._session = BrowserSession(**session_kwargs)
            if resolved_pvpo_cdp_url:
                if pvpo_beginframe_controller is not None:
                    self._session._worldsim_pvpo_beginframe_controller = (
                        pvpo_beginframe_controller
                    )
                    self._session._worldsim_pvpo_beginframe_lock = (
                        pvpo_beginframe_controller.lock
                    )
                else:
                    self._session._worldsim_pvpo_beginframe_lock = asyncio.Lock()
                self._session._worldsim_pvpo_disable_browser_use_screenshots = not bool(
                    self.use_vision
                )
        except Exception:
            await _release_pvpo_endpoint_lease()
            raise

        # Retry browser startup with linear backoff for transient failures
        last_exc: Exception | None = None
        try:
            for attempt in range(1, 4):
                try:
                    await self._session.start()
                    last_exc = None
                    break
                except (TimeoutError, ConnectionError, OSError, RuntimeError) as exc:
                    last_exc = exc
                    if attempt < 3:
                        delay = 3 * attempt
                        logger.warning(
                            "Browser startup failed (attempt %d/3), retrying in %ds: %s",
                            attempt,
                            delay,
                            exc,
                        )
                        await asyncio.sleep(delay)
        except Exception:
            await _release_pvpo_endpoint_lease()
            raise
        if last_exc is not None:
            await _release_pvpo_endpoint_lease()
            raise last_exc

        history = None
        agent = None
        elapsed = 0.0
        network_trace: list[dict[str, Any]] = []
        network_recorder: _NetworkTraceRecorder | None = None
        status = "error"
        extra_errors: list[str] = []
        task_text = site_prompt if site_prompt else task
        try:
            if self._pvpo_cdp_url:
                await self._reset_remote_browser_for_task(self._session)
                if external_cdp_storage_state:
                    await _restore_external_cdp_storage_state(
                        self._session, external_cdp_storage_state
                    )

            # Run any deferred auth actions (e.g. future form_login flow) after
            # session.start() succeeds and after any remote-browser reset has
            # produced a fresh task-owned target. No-op for the first batch
            # (storage_state / http_basic / http_headers / none).
            for action in deferred_auth_actions:
                await action(self._session)
            if url_origin_rewrites:
                await _ensure_scoped_request_mutator(
                    self._session,
                    url_origin_rewrites=url_origin_rewrites,
                )

            network_recorder = _NetworkTraceRecorder(
                self._session,
                task_dir,
                target_filter=self._owned_target_ids,
                sensitive_header_names=auth_sensitive_header_names,
            )
            await network_recorder.start()

            initial_actions = _build_initial_actions(start_urls or [])
            # ``frame_pump`` now owns only the shared capture event/coordinator
            # context in production. Browser Use navigation gets compositor
            # frames from the PVPO navigation-tick patch above, scoped to
            # _navigate_and_wait instead of a broad background clock.
            from worldsim.phase_4.pvpo_frame_pump import frame_pump

            frame_pump_interval_s = 0.0
            async with frame_pump(self._session, interval_s=frame_pump_interval_s) as capturing:
                self._session._worldsim_pvpo_beginframe_lock = getattr(
                    capturing, "beginframe_lock", None
                )
                self._session._worldsim_pvpo_beginframe_controller = getattr(
                    capturing, "beginframe_controller", None
                )
                self._session._worldsim_pvpo_capturing_event = capturing
                pvpo_hook = _make_pvpo_step_callback(
                    self._session,
                    task_dir,
                    payload_text,
                    payload_witnesses=payload_witnesses,
                    owned_target_ids=self._owned_target_ids,
                    capturing=capturing,
                )
                agent_kwargs: dict[str, Any] = {
                    "task": task_text,
                    "llm": self.llm,
                    "browser_session": self._session,
                    "use_vision": self.use_vision,
                    # WorldSim uses its own reward evaluators plus Phase 3/4
                    # diagnosis/judge flows; Browser Use's internal judge only adds
                    # post-hoc logging and currently breaks on the Anthropic-via-
                    # OpenRouter path.
                    "use_judge": False,
                    "save_conversation_path": str(task_dir / "conversations"),
                    "initial_actions": initial_actions,
                    "register_new_step_callback": pvpo_hook,
                }
                if self.llm_timeout is not None:
                    agent_kwargs["llm_timeout"] = self.llm_timeout
                if self.step_timeout is not None:
                    agent_kwargs["step_timeout"] = self.step_timeout
                agent = Agent(**agent_kwargs)

                t0 = time.time()
                try:
                    history = await asyncio.wait_for(
                        agent.run(max_steps=self.max_steps), timeout=self.timeout
                    )
                    elapsed = time.time() - t0
                    status = "success" if history.is_done() else "failure"
                except TimeoutError:
                    elapsed = time.time() - t0
                    status = "timeout"
                    extra_errors.append(f"agent timed out after {self.timeout}s")
                    history = getattr(agent, "history", None)
                    logger.warning("Agent timed out after %ss for %s", self.timeout, task_dir)
                except Exception as e:
                    elapsed = time.time() - t0
                    status = "error"
                    extra_errors.append(str(e))
                    history = getattr(agent, "history", None)
                    logger.exception("Agent run failed for %s", task_dir)
        finally:
            if network_recorder is not None:
                network_trace = await network_recorder.stop()
            self._task_origins.update(_origins_from_network_trace(network_trace))
            self._record_browser_use_patch_runtime()
            self._browser_runtime.update(
                {
                    "network_trace_entries": len(network_trace),
                    "observed_origins": sorted(self._task_origins),
                }
            )
            history = history or (getattr(agent, "history", None) if agent is not None else None)
            _write_agent_artifacts(
                task_dir=task_dir,
                history=history,
                task_instruction=task_text,
                status=status,
                extra_errors=extra_errors,
            )
            try:
                if self._session is not None:
                    await self._shutdown_browser_session(task_dir=task_dir)
                else:
                    _write_browser_runtime_artifact(task_dir, self._browser_runtime)
            finally:
                await _release_pvpo_endpoint_lease()

        steps, is_done, final_result, history_errors = _extract_history_state(history)
        return AgentResult(
            elapsed=round(elapsed, 1),
            steps=steps,
            is_done=is_done,
            final_result=final_result,
            status=status,
            errors=[*history_errors, *extra_errors],
            network_trace=network_trace,
        )

    def _record_browser_use_patch_runtime(self) -> None:
        """Persist per-task counters from WorldSim's Browser Use compatibility patches."""
        if self._session is None:
            return
        counter_names = (
            "_worldsim_pvpo_scroll_wheel_successes",
            "_worldsim_pvpo_scroll_wheel_late_successes",
            "_worldsim_pvpo_scroll_wheel_timeouts",
            "_worldsim_pvpo_scroll_wheel_failures",
            "_worldsim_pvpo_scroll_wheel_noops",
            "_worldsim_pvpo_scroll_js_fallbacks",
            "_worldsim_pvpo_scroll_js_failures",
            "_worldsim_pvpo_scroll_js_noops",
            "_worldsim_pvpo_scroll_failures",
            "_worldsim_pvpo_navigation_tick_navigations",
            "_worldsim_pvpo_navigation_tick_frames",
            "_worldsim_pvpo_navigation_tick_failures",
            "_worldsim_pvpo_navigation_tick_timeouts",
            "_worldsim_pvpo_navigation_tick_stop_timeouts",
            "_worldsim_pvpo_navigation_tick_drain_timeouts",
            "_worldsim_pvpo_navigation_tick_drain_failures",
            "_worldsim_pvpo_navigation_tick_skipped_captures",
            "_worldsim_pvpo_browser_use_screenshot_drain_timeouts",
            "_worldsim_pvpo_browser_use_screenshot_drain_failures",
            "_worldsim_pvpo_cdp_timeouts",
            "_worldsim_pvpo_cdp_late_completions",
            "_worldsim_pvpo_cdp_late_failures",
            "_worldsim_pvpo_cdp_late_cancellations",
        )
        for name in counter_names:
            value = getattr(self._session, name, 0)
            if isinstance(value, int) and value > 0:
                self._browser_runtime[name.removeprefix("_worldsim_")] = value
        try:
            from worldsim.phase_4.pvpo_beginframe import BeginFrameCoordinator

            controller = getattr(self._session, "_worldsim_pvpo_beginframe_controller", None)
            if isinstance(controller, BeginFrameCoordinator):
                for key, value in controller.stats().items():
                    if value not in (None, "", 0, False):
                        self._browser_runtime[f"pvpo_{key}"] = value
        except Exception:
            return

    async def teardown(self) -> None:
        if self._session is not None:
            await self._shutdown_browser_session(task_dir=None)

    async def _shutdown_browser_session(self, *, task_dir: Path | None) -> None:
        """Bounded, idempotent Browser Use session shutdown.

        Browser Use closes its own session in ``Agent.run()`` when
        ``keep_alive=False``. Phase 4 then still needs to clean/recycle the
        external PVPO browser boundary, but a second CDP cleanup/kill against an
        already-reset BrowserSession can leave the event bus alive. Treat that
        state as already disconnected and skip the duplicate kill.
        """
        session = self._session
        if session is None:
            if task_dir is not None:
                _write_browser_runtime_artifact(task_dir, self._browser_runtime)
            return

        try:
            self._cleanup_temp_profile(session)
            try:
                await self._stop_scoped_header_auth()
            except Exception as e:
                logger.warning("scoped http_headers auth shutdown failed: %s", e)

            disconnected = self._browser_session_disconnected(session)
            if disconnected:
                self._browser_runtime["browser_session_disconnect_observed"] = True

            if self._pvpo_cdp_url and disconnected:
                self._browser_runtime["cleanup_skipped_reason"] = (
                    "browser_use_session_already_disconnected"
                )
            else:
                try:
                    await self._cleanup_external_cdp_state(session)
                except Exception as e:
                    logger.warning("Remote PVPO browser cleanup failed: %s", e)

            try:
                await self._recycle_external_pvpo_browser(session)
            except Exception as e:
                logger.warning("Remote PVPO browser recycle failed: %s", e)

            if disconnected:
                self._browser_runtime["browser_session_kill_skipped_reason"] = (
                    "browser_use_session_already_disconnected"
                )
            else:
                try:
                    await asyncio.wait_for(session.kill(), timeout=5)
                except TimeoutError:
                    logger.warning(
                        "BrowserSession kill timed out%s",
                        f" for {task_dir}" if task_dir is not None else "",
                    )
                    await self._force_stop_browser_event_bus(session)
                except Exception as e:
                    logger.warning("BrowserSession kill failed: %s", e)
        finally:
            if task_dir is not None:
                _write_browser_runtime_artifact(task_dir, self._browser_runtime)
            self._session = None
            self._pvpo_cdp_url = ""
            self._task_origins = set()
            self._owned_target_ids = set()
            self._primary_target_id = None
            self._browser_runtime = {}
            self._preserve_remote_auth_state = False

    @staticmethod
    def _browser_session_disconnected(session: Any) -> bool:
        cdp_root = getattr(session, "_cdp_client_root", ...)
        if cdp_root is None:
            return True
        session_manager = getattr(session, "session_manager", ...)
        cdp_client = getattr(session, "cdp_client", ...)
        return session_manager is None and cdp_client is None

    @staticmethod
    async def _force_stop_browser_event_bus(session: Any) -> None:
        for attr_name in ("event_bus", "_event_bus", "browser_event_bus"):
            event_bus = getattr(session, attr_name, None)
            if event_bus is None:
                continue
            for method_name in ("stop", "shutdown", "close"):
                method = getattr(event_bus, method_name, None)
                if not callable(method):
                    continue
                try:
                    try:
                        result = method(timeout=0)
                    except TypeError:
                        result = method()
                    if hasattr(result, "__await__"):
                        await asyncio.wait_for(result, timeout=1)
                except Exception as exc:
                    logger.debug(
                        "BrowserSession event bus %s.%s force-stop failed: %s",
                        attr_name,
                        method_name,
                        exc,
                    )
                return

    async def _stop_scoped_header_auth(self) -> None:
        if self._session is None:
            return
        injector = getattr(self._session, "_worldsim_scoped_header_auth", None)
        if injector is None:
            return
        try:
            await injector.stop()
        except Exception:
            logger.warning("scoped http_headers auth shutdown failed", exc_info=True)
        finally:
            with suppress(AttributeError):
                delattr(self._session, "_worldsim_scoped_header_auth")

    async def _reset_remote_browser_for_task(self, session: Any) -> None:
        """Reset a worker-owned remote browser to one fresh blank target."""
        pages = await _session_pages(session)
        initial_targets = sorted(
            target_id for target_id in (_target_id_for_page(page) for page in pages) if target_id
        )
        focused_page = None
        get_current_page = getattr(session, "get_current_page", None)
        if callable(get_current_page):
            try:
                focused_page = await get_current_page()
            except Exception as exc:
                logger.debug("PVPO reset: could not resolve focused page: %s", exc)
        focused_target_id = _target_id_for_page(focused_page) if focused_page is not None else None

        retained_page = None
        extra_pages: list[Any] = []
        for page in pages:
            target_id = _target_id_for_page(page)
            if retained_page is None and (
                focused_target_id is None or target_id == focused_target_id
            ):
                retained_page = page
                continue
            extra_pages.append(page)
        if retained_page is None and extra_pages:
            retained_page = extra_pages.pop(0)
        if pages:
            if not self._preserve_remote_auth_state:
                await self._clear_page_storage(session, pages=pages, origins=set())
            if retained_page is not None:
                try:
                    await retained_page.goto("about:blank")
                except Exception as exc:
                    logger.debug("PVPO reset: could not reuse existing page: %s", exc)
                    try:
                        await self._close_pages(session, [retained_page])
                    except Exception:
                        logger.debug(
                            "PVPO reset: could not close failed retained page", exc_info=True
                        )
                    retained_page = None
            if extra_pages:
                await self._close_pages(session, extra_pages)

        if not self._preserve_remote_auth_state:
            await self._clear_browser_cookies(session)

        if retained_page is None:
            retained_page = await session.new_page("about:blank")
        new_target_id = _target_id_for_page(retained_page)
        if not new_target_id:
            raise RuntimeError("remote PVPO browser reset created a page without a target id")
        await session.get_or_create_cdp_session(target_id=new_target_id, focus=True)
        self._owned_target_ids = {new_target_id}
        self._primary_target_id = new_target_id
        self._browser_runtime.update(
            {
                "pvpo_cdp_url": self._pvpo_cdp_url,
                "reset_initial_targets": initial_targets,
                "reset_closed_targets": len(extra_pages),
                "primary_target_id": new_target_id,
                "reset_preserved_auth_state": self._preserve_remote_auth_state,
            }
        )

    async def _cleanup_external_cdp_state(self, session: Any) -> None:
        """Clean up worker-owned remote browser state between task-scoped runs."""
        if not self._pvpo_cdp_url:
            return

        pages = await _session_pages(session)
        closed_target_ids = sorted(
            target_id for target_id in (_target_id_for_page(page) for page in pages) if target_id
        )
        await self._clear_page_storage(session, pages=pages, origins=set(self._task_origins))
        await self._clear_browser_cookies(session)
        await self._close_pages(session, pages)
        self._browser_runtime.update(
            {
                "cleanup_closed_targets": len(closed_target_ids),
                "cleanup_target_ids": closed_target_ids,
                "cleanup_origins": sorted(self._task_origins),
                "cleanup_preserved_auth_state": False,
            }
        )

    async def _recycle_external_pvpo_browser(self, session: Any) -> None:
        """Hard-reset the dedicated PVPO browser process after task cleanup."""
        if not self._pvpo_cdp_url:
            return
        from worldsim.phase_4.pvpo_browser_lifecycle import recycle_pvpo_browser_after_task

        recycle_artifact = await recycle_pvpo_browser_after_task(session, self._pvpo_cdp_url)
        controller = getattr(session, "_worldsim_pvpo_beginframe_controller", None)
        if recycle_artifact.get("recycle_status") == "recycled":
            try:
                from worldsim.phase_4.pvpo_beginframe import BeginFrameCoordinator

                if isinstance(controller, BeginFrameCoordinator):
                    controller.reset_after_recycle()
                    recycle_artifact["beginframe_state_reset"] = True
            except Exception as exc:
                recycle_artifact["beginframe_state_reset_error"] = str(exc)
        self._browser_runtime.update(
            {
                "pvpo_browser_recycle": recycle_artifact,
                "pvpo_browser_recycle_status": recycle_artifact.get("recycle_status"),
            }
        )

    async def _clear_page_storage(
        self, session: Any, *, pages: list[Any], origins: set[str]
    ) -> None:
        from worldsim.phase_4.pvpo_cdp import runtime_evaluate

        resolved_origins = set(origins)
        for page in pages:
            get_url = getattr(page, "get_url", None)
            if callable(get_url):
                try:
                    origin = _origin_from_url(await get_url())
                except Exception as exc:
                    logger.debug("PVPO cleanup: could not read page URL: %s", exc)
                else:
                    if origin:
                        resolved_origins.add(origin)

        cdp_client = getattr(session, "cdp_client", None)
        storage_sender = getattr(getattr(cdp_client, "send", None), "Storage", None)
        clear_data_for_origin = getattr(storage_sender, "clearDataForOrigin", None)
        if callable(clear_data_for_origin):
            for origin in sorted(resolved_origins):
                try:
                    await clear_data_for_origin(
                        params={
                            "origin": origin,
                            "storageTypes": "all",
                        }
                    )
                except Exception as exc:
                    logger.debug(
                        "PVPO cleanup: could not clear data for origin %s: %s", origin, exc
                    )

        seen_target_ids: set[str] = set()
        for page in pages:
            target_id = _target_id_for_page(page)
            if not target_id or target_id in seen_target_ids:
                continue
            seen_target_ids.add(target_id)
            try:
                cdp_session = await session.get_or_create_cdp_session(
                    target_id=target_id, focus=False
                )
                await runtime_evaluate(cdp_session, _CLEAR_PAGE_STORAGE_JS)
            except Exception as exc:
                logger.debug(
                    "PVPO cleanup: could not clear storage for target %s: %s",
                    target_id,
                    exc,
                )

    async def _clear_browser_cookies(self, session: Any) -> None:
        clear_cookies = getattr(session, "clear_cookies", None)
        if callable(clear_cookies):
            try:
                await clear_cookies()
            except Exception as exc:
                logger.debug("PVPO cleanup: could not clear cookies: %s", exc)

    async def _close_pages(self, session: Any, pages: list[Any]) -> None:
        close_page = getattr(session, "close_page", None)
        if not callable(close_page):
            return
        for page in pages:
            try:
                await close_page(page)
            except Exception as exc:
                logger.debug("PVPO cleanup: could not close page: %s", exc)

    @staticmethod
    def _cleanup_temp_profile(session: Any) -> None:
        """Remove the browser-use temp user data dir if it lives under /tmp/.

        Browser Use creates ``/tmp/browser-use-user-data-dir-*`` dirs via
        ``BrowserProfile.validate_user_data_dir`` and never cleans them up.
        Over hundreds of tasks these accumulate significantly.
        """
        try:
            user_data_dir = getattr(
                getattr(session, "browser_profile", None),
                "user_data_dir",
                None,
            )
            if user_data_dir and Path(user_data_dir).exists() and "/tmp/" in str(user_data_dir):
                shutil.rmtree(user_data_dir, ignore_errors=True)
        except Exception:
            pass


def _build_initial_actions(start_urls: list[str]) -> list[dict[str, dict[str, Any]]] | None:
    """Convert resolved start URLs into Browser Use initial navigate actions."""
    seen: set[str] = set()
    actions: list[dict[str, dict[str, Any]]] = []
    for index, url in enumerate(start_urls):
        normalized = str(url).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        actions.append(
            {
                "navigate": {
                    "url": normalized,
                    "new_tab": index > 0,
                }
            }
        )
    return actions or None


def _target_id_for_page(page: Any) -> str | None:
    target_id = getattr(page, "_target_id", None)
    if isinstance(target_id, str) and target_id:
        return target_id
    return None


async def _session_pages(session: Any, *, target_filter: set[str] | None = None) -> list[Any]:
    pages: list[Any] = []
    get_pages = getattr(session, "get_pages", None)
    if callable(get_pages):
        try:
            pages = list(await get_pages())
        except Exception as exc:
            logger.debug("PVPO page enumeration failed: %s", exc)
    if not pages:
        try:
            current_page = await session.get_current_page()
        except Exception as exc:
            logger.debug("PVPO current page unavailable: %s", exc)
        else:
            if current_page is not None:
                pages = [current_page]
    if not target_filter:
        return pages
    filtered: list[Any] = []
    for page in pages:
        target_id = _target_id_for_page(page)
        if target_id and target_id in target_filter:
            filtered.append(page)
    return filtered


def _write_browser_runtime_artifact(task_dir: Path, runtime_payload: dict[str, Any]) -> None:
    """Persist browser runtime metadata for debugging concurrency failures."""
    if not runtime_payload:
        return
    try:
        write_json_atomic(
            task_dir / "browser_runtime.json",
            runtime_payload,
        )
    except Exception as exc:
        logger.warning("Failed to write browser_runtime.json for %s: %s", task_dir, exc)


def _write_agent_artifacts(
    *,
    task_dir: Path,
    history: Any,
    task_instruction: str | None = None,
    status: str,
    extra_errors: list[str],
) -> None:
    """Persist trajectory artifacts even when the agent times out or crashes."""
    task_dir.mkdir(parents=True, exist_ok=True)
    steps, is_done, final_result, history_errors = _extract_history_state(history)
    errors = [*history_errors, *extra_errors]

    history_path = task_dir / "history.json"
    if history is not None:
        try:
            _write_history_atomically(history, history_path)
        except Exception as e:
            logger.warning("Failed to write history.json for %s: %s", task_dir, e)
            _write_history_fallback(history_path, errors, status)
    else:
        _write_history_fallback(history_path, errors, status)

    _copy_history_screenshots(task_dir, history)
    if task_instruction:
        try:
            from worldsim.phase_4.aer_trajectory_extract import extract_trajectory
            from worldsim.phase_4.needham_trace import write_trace_artifacts

            extracted = extract_trajectory(task_dir)
            write_trace_artifacts(
                task_dir,
                task_instruction=task_instruction,
                extracted=extracted,
            )
        except Exception as exc:
            logger.warning("Failed to write needham_trace artifacts for %s: %s", task_dir, exc)

    final_response = {
        "status": status.upper(),
        "final_result": final_result,
        "errors": errors,
        "steps": steps,
        "is_done": is_done,
    }
    try:
        write_json_atomic(
            task_dir / "final_response.json",
            final_response,
        )
    except Exception as e:
        logger.warning("Failed to write final_response.json for %s: %s", task_dir, e)


def _write_history_atomically(history: Any, history_path: Path) -> None:
    """Write Browser Use history via temp file + replace to avoid truncation."""
    history_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=history_path.parent, suffix=".history.tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        try:
            history.save_to_file(tmp_path)
        except UnicodeEncodeError:
            _write_history_with_ascii_escapes(history, tmp_path)
        os.replace(tmp_path, history_path)
    except BaseException:
        with suppress(OSError):
            tmp_path.unlink()
        raise


def _write_history_with_ascii_escapes(history: Any, path: Path) -> None:
    """Persist Browser Use history when raw UTF-8 serialization hits surrogates."""
    if not hasattr(history, "model_dump"):
        raise TypeError("history object does not expose model_dump()")
    payload = history.model_dump()
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def _extract_history_state(history: Any) -> tuple[int, bool, str | None, list[str]]:
    """Extract summary fields from a Browser Use history object if present."""
    if history is None:
        return 0, False, None, []

    try:
        steps = len(history.history)
    except Exception:
        steps = 0

    try:
        is_done = bool(history.is_done())
    except Exception:
        is_done = False

    try:
        final_result = history.final_result()
    except Exception:
        final_result = None

    try:
        errors = []
        for error in history.errors():
            if error is None:
                continue
            text = str(error)
            errors.append(text if text else "<empty browser-use step error>")
    except Exception:
        errors = []

    return steps, is_done, final_result, errors


def _normalize_payload_witness_specs(
    payload_witnesses: list[str | dict[str, Any]] | None,
) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    for index, witness in enumerate(payload_witnesses or []):
        if isinstance(witness, str):
            if witness:
                specs.append({"id": f"witness:{index}", "text": witness})
            continue
        if not isinstance(witness, dict):
            continue
        text = witness.get("text")
        if not isinstance(text, str) or not text:
            continue
        witness_id = witness.get("id")
        kind = witness.get("kind")
        spec = {
            "id": witness_id
            if isinstance(witness_id, str) and witness_id
            else f"witness:{index}",
            "text": text,
        }
        if isinstance(kind, str) and kind:
            spec["kind"] = kind
        specs.append(spec)
    return specs


def _make_pvpo_step_callback(
    session: Any,
    task_dir: Path,
    payload_text: str | None,
    *,
    payload_witnesses: list[str | dict[str, Any]] | None = None,
    owned_target_ids: set[str] | None = None,
    capturing: asyncio.Event | None = None,
):
    """Build the per-step callback that captures PVPO artifacts.

    Browser-Use invokes this after each agent step with
    ``(browser_state_summary, agent_output, step_index)``. We use the step
    boundary to:

    1. Idempotently inject the animation-killer stylesheet on the current
       page (once per page load).
    2. Pause virtual time, run the per-char visibility query (which locates
       ``payload_text`` in the live DOM by substring match), capture a
       deterministic ``HeadlessExperimental.beginFrame`` screenshot, and
       write ``pvpo/step_{N}.json`` + ``screenshots/step_{N}.png``.

    The callback is intentionally best-effort: any CDP failure (the most
    common is ``beginFrame`` being unsupported on native macOS Chrome, which
    is why rigor runs use the ``chrome-headless-shell`` Docker container)
    logs at debug and returns without raising. Trajectories without PVPO
    artifacts fall back to the legacy screenshot-copy path via
    ``_copy_history_screenshots`` after the agent finishes, and encounter
    detection reports zero coverage — routing to placement-fix.

    When ``payload_text`` and ``payload_witnesses`` are both absent, PVPO
    capture is disabled (we have nothing to locate in the DOM). This is the
    benign-task / no-seed case.

    See ``docs/handoffs/codex-handoff-paint-verified-oracle.md`` §3 and
    the Implementation Status section documenting the content-match
    anchor strategy.
    """
    if not payload_text and not payload_witnesses:
        from worldsim.phase_4.pvpo_capture import initial_capture_summary, save_capture_summary

        save_capture_summary(task_dir, initial_capture_summary(payload_present=False))

        async def _noop(state_summary: Any, agent_output: Any, step_idx: int) -> None:
            return None

        return _noop

    # Import inside the factory so import-time failure of optional PVPO deps
    # (Pillow, numpy, phase_4 subpackage) does not break the base AgentRunner.
    from worldsim.phase_4.pvpo_browser_config import inject_animation_killer
    from worldsim.phase_4.pvpo_capture import (
        Rect,
        atomic_capture_with_visibility,
        initial_capture_summary,
        save_capture_summary,
        save_step_artifacts,
    )
    from worldsim.phase_4.pvpo_cdp import runtime_evaluate_value

    # Shared per-run state: avoid re-installing the animation-killer
    # stylesheet on every step. No reference-container state and no anchor-
    # attribute lookup after the ink-occupancy + content-match cutover —
    # the JS query walks text nodes for the payload substring directly.
    pages_prepared: set[str] = set()
    warned_issue_classes: set[str] = set()
    capture_summary = initial_capture_summary(payload_present=True)
    non_empty_witnesses = _normalize_payload_witness_specs(payload_witnesses)
    payload_text_present = isinstance(payload_text, str) and bool(payload_text)
    if non_empty_witnesses:
        witness_mode = "curated_witnesses"
    elif payload_text_present and payload_witnesses is not None:
        witness_mode = "payload_text_fallback_empty_witnesses"
    elif payload_text_present:
        witness_mode = "payload_text_fallback"
    else:
        witness_mode = "no_witness"
    capture_summary.update(
        {
            "witness_selection_mode": witness_mode,
            "payload_witness_count": len(non_empty_witnesses),
            "payload_witness_lengths": [
                len(witness["text"]) for witness in non_empty_witnesses
            ],
            "payload_witness_ids": [witness.get("id") for witness in non_empty_witnesses],
            "payload_witness_kinds": [
                witness.get("kind") for witness in non_empty_witnesses
            ],
            "payload_text_present": payload_text_present,
            "payload_text_length": len(payload_text) if isinstance(payload_text, str) else 0,
        }
    )
    save_capture_summary(task_dir, capture_summary)

    def _record_issue(
        issue_class: str,
        step_idx: int,
        message: str,
        *,
        count_as_seen: bool = False,
    ) -> None:
        if count_as_seen:
            capture_summary["steps_seen"] += 1
        capture_summary["status"] = "degraded"
        capture_summary["issue_steps"] += 1
        issue_counts = capture_summary.setdefault("issue_counts", {})
        issue_counts[issue_class] = int(issue_counts.get(issue_class, 0)) + 1
        if capture_summary.get("first_issue_class") is None:
            capture_summary["first_issue_class"] = issue_class
            capture_summary["first_issue_step"] = step_idx
            capture_summary["first_issue_message"] = message
        capture_summary["last_issue_class"] = issue_class
        capture_summary["last_issue_step"] = step_idx
        capture_summary["last_issue_message"] = message
        save_capture_summary(task_dir, capture_summary)
        if issue_class not in warned_issue_classes:
            warned_issue_classes.add(issue_class)
            logger.warning(
                "pvpo: %s at step %d for %s; continuing in degraded mode "
                "(zero coverage may reflect capture failure): %s",
                issue_class,
                step_idx,
                task_dir,
                message,
            )
        else:
            logger.debug("pvpo: %s at step %d: %s", issue_class, step_idx, message)

    def _record_capture_success(step_idx: int, capture_issue_class: str | None) -> None:
        capture_summary["steps_captured"] += 1
        if capture_summary["issue_steps"] == 0:
            capture_summary["status"] = "ok"
        if capture_issue_class is not None and capture_summary["status"] != "degraded":
            capture_summary["status"] = "degraded"
        save_capture_summary(task_dir, capture_summary)

    async def _callback(state_summary: Any, agent_output: Any, step_idx: int) -> None:
        timeout_s = _pvpo_cdp_timeout_s()
        capture_summary["steps_seen"] += 1
        save_capture_summary(task_dir, capture_summary)
        try:
            page = await _await_pvpo_cdp(
                session.get_current_page(),
                timeout_s=timeout_s,
                label="BrowserSession.get_current_page",
                browser_session=session,
            )
        except Exception as exc:  # pragma: no cover - CDP unavailable
            _record_issue(
                "current_page_unavailable",
                step_idx,
                _pvpo_issue_message(exc, timeout_s=timeout_s),
            )
            return
        if page is None:
            _record_issue(
                "current_page_unavailable", step_idx, "Browser session has no current page"
            )
            return

        target_id = _target_id_for_page(page)
        if not target_id:
            _record_issue("cdp_session_unavailable", step_idx, "current page has no target id")
            return
        if owned_target_ids is not None:
            owned_target_ids.add(target_id)
        try:
            cdp_session = await _await_pvpo_cdp(
                session.get_or_create_cdp_session(target_id=target_id, focus=False),
                timeout_s=timeout_s,
                label="BrowserSession.get_or_create_cdp_session",
                browser_session=session,
            )
        except Exception as exc:  # pragma: no cover - CDP unavailable
            _record_issue(
                "cdp_session_unavailable",
                step_idx,
                _pvpo_issue_message(exc, timeout_s=timeout_s),
            )
            return

        dirty_reason = _pvpo_beginframe_dirty_reason(session)
        if dirty_reason:
            _record_issue("beginframe_endpoint_dirty", step_idx, dirty_reason)
            return

        try:
            if target_id not in pages_prepared:
                await _await_pvpo_cdp(
                    inject_animation_killer(page, cdp_session),
                    timeout_s=timeout_s,
                    label="PVPO animation killer",
                    browser_session=session,
                )
                pages_prepared.add(target_id)
        except Exception as exc:
            _record_issue(
                "animation_killer_failed",
                step_idx,
                _pvpo_issue_message(exc, timeout_s=timeout_s),
            )

        try:
            viewport = await _await_pvpo_cdp(
                runtime_evaluate_value(cdp_session, _CDP_VIEWPORT_JS),
                timeout_s=timeout_s,
                label="PVPO viewport probe",
                browser_session=session,
            )
            if not isinstance(viewport, dict):
                raise RuntimeError(
                    f"viewport probe returned {type(viewport).__name__}, expected object"
                )
            viewport_rect = Rect(
                x=0,
                y=0,
                w=int(viewport.get("w", 0)) or 1280,
                h=int(viewport.get("h", 0)) or 720,
            )
            capture = await atomic_capture_with_visibility(
                cdp_session,
                viewport_rect=viewport_rect,
                payload_text=payload_text,
                witness_texts=non_empty_witnesses if payload_witnesses is not None else None,
                scroll_to_match=False,
                capturing=capturing,
                cdp_timeout_s=timeout_s,
            )
            save_step_artifacts(task_dir, step_idx, capture)
            if capture.issue_class is not None:
                _record_issue(
                    capture.issue_class,
                    step_idx,
                    capture.issue_message or capture.issue_class,
                )
            _record_capture_success(step_idx, capture.issue_class)
        except Exception as exc:
            _record_issue(
                "capture_failed",
                step_idx,
                _pvpo_issue_message(exc, timeout_s=timeout_s),
            )

    return _callback


def _copy_history_screenshots(task_dir: Path, history: Any) -> None:
    """Copy any screenshots referenced by the partial or final history."""
    if history is None:
        return

    try:
        screenshot_paths = history.screenshot_paths()
    except Exception:
        return

    screenshots_dir = task_dir / "screenshots"
    for step_idx, path_str in enumerate(screenshot_paths):
        if path_str and Path(path_str).exists():
            screenshots_dir.mkdir(parents=True, exist_ok=True)
            destination = screenshots_dir / f"step_{step_idx}.png"
            if destination.exists() and destination.stat().st_size > 0:
                # Phase 4 PVPO already wrote the deterministic frame for this
                # step. Keep that artifact; Browser Use screenshots are only a
                # fallback for phases that do not populate screenshots eagerly.
                continue
            shutil.copy2(path_str, destination)


def _write_history_fallback(history_path: Path, errors: list[str], status: str) -> None:
    """Write a minimal history artifact when Browser Use exposes no history object."""
    payload = {
        "history": [],
        "partial": True,
        "status": status,
        "errors": errors,
    }
    try:
        write_json_atomic(history_path, payload)
    except Exception as e:
        logger.warning("Failed to write fallback history.json for %s: %s", history_path, e)
