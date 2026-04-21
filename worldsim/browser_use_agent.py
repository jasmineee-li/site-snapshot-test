"""Browser Use agent runner.

Canonical source: ``docs/worldsim-v5-technical-specifcation.md`` "Browser Use Integration".

We use Browser Use as an async Python library (not a subprocess) for running
browser agents against pre-running benchmark environments. Each worker owns
the runner object, and each task gets a fresh ``BrowserSession`` so trajectory
artifacts such as network traces stay isolated per task directory.
"""

from __future__ import annotations

import asyncio
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

from worldsim.atomic_io import write_json_atomic
from worldsim.config import has_configured_agent_auth
from worldsim.pvpo_endpoint import validate_pvpo_cdp_url

logger = logging.getLogger(__name__)

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
    "cookie",
    "csrf",
    "key",
)
_PHASE_0D_SITE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_CDP_VIEWPORT_JS = """
(() => ({
  w: Math.max(0, Number(window.innerWidth || 0)),
  h: Math.max(0, Number(window.innerHeight || 0))
}))()
"""
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
        pvpo_cdp_url: str | None = None,
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
    ) -> None:
        self._browser_session = browser_session
        self._task_dir = Path(task_dir)
        self._client = getattr(browser_session, "cdp_client", None)
        self._recording = False
        self._poll_task: asyncio.Task | None = None
        self._enabled_targets: set[str] = set()
        self._target_filter = target_filter
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
        # Persist only redacted wire artifacts; downstream sandboxes may stage
        # network.har wholesale from the trajectory directory.
        self._write_trace([self._redact_trace_entry(entry) for entry in trace])
        return trace

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
                self._nav_events,
            )
        except Exception as e:
            logger.warning("Failed to write navigation_trace.json: %s", e)

        har_entries = flat_events_to_har_entries(trace)
        har_pages = nav_events_to_har_pages(self._nav_events)
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
    def _redact_trace_entry(cls, entry: dict[str, Any]) -> dict[str, Any]:
        """Redact sensitive wire data before persisting trajectory artifacts."""
        redacted = dict(entry)
        redacted["url"] = cls._redact_url(redacted.get("url", ""))
        redacted["query_params"] = cls._redact_query_params(redacted.get("query_params", {}))
        redacted["headers"] = cls._redact_headers(redacted.get("headers", {}))
        redacted["response_headers"] = cls._redact_headers(redacted.get("response_headers", {}))
        if redacted.get("post_data") is not None:
            redacted["post_data"] = "<redacted>"
        if redacted.get("response_cookies"):
            redacted["response_cookies"] = {
                key: "<redacted>" for key in redacted["response_cookies"]
            }
        return redacted

    @classmethod
    def _redact_headers(cls, headers: dict[str, Any]) -> dict[str, Any]:
        """Redact sensitive header values while preserving non-secret metadata."""
        redacted: dict[str, Any] = {}
        for key, value in headers.items():
            lower = str(key).lower()
            if lower in _SENSITIVE_HEADER_NAMES or any(
                marker in lower for marker in _SENSITIVE_HEADER_SUBSTRINGS
            ):
                redacted[str(key)] = "<redacted>"
            else:
                redacted[str(key)] = value
        return redacted

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


def _phase_0d_fallback_path(task: dict[str, Any] | None) -> Path | None:
    """Return the Phase 0d-bootstrapped storage_state.json path for ``task``'s site.

    Phase 0d writes artifacts to ``<state_dir>/phase_0d/<site>/storage_state.json``.
    We consult it as a fallback when the benchmark-declared storage_state path is
    missing, so operators who bootstrap via ``worldsim phase 0d`` do not need to
    re-edit AGENT_CONTEXT to point at the generated artifact.

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
        )
    except ImportError:  # pragma: no cover — only triggers on misinstalled env.
        return None
    artifact_path = phase_0d_artifact_path(site.strip())
    completion_path = phase_0d_completion_path(site.strip())
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


def _resolve_storage_state_path(raw_path: str, benchmark_root: Path | None) -> Path:
    """Resolve a storage-state artifact path and enforce benchmark-root containment."""
    path = Path(raw_path)
    if path.is_absolute():
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


# Stable enum of first-batch implementations. Types outside this set are schema-
# legal (validator accepts them) but raise ``NotImplementedError`` at runtime
# until their dispatcher arm is written. See plan §8 rollout order.
_IMPLEMENTED_AUTH_TYPES = frozenset({"storage_state", "http_basic", "none"})
_UNIMPLEMENTED_AUTH_TYPES = frozenset({"form_login", "pre_auth_script", "client_cert"})


def _resolve_auth(
    auth_mechanism: dict[str, Any] | None,
    task: dict[str, Any] | None,
    benchmark_root: Path | None,
) -> tuple[dict[str, Any], list[Any]]:
    """Translate an ``auth_mechanism`` dict into BrowserSession kwargs + deferred actions.

    Returns ``(session_kwargs, deferred_actions)``. ``session_kwargs`` is merged
    into the ``BrowserSession(...)`` call. ``deferred_actions`` is a list of
    async callables that receive the started session (unused for the first
    batch — reserved for ``form_login`` / ``pre_auth_script``).

    First-batch implementations: ``storage_state``, ``http_basic``, ``none``.
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
            "storage_state, http_basic, and none ship in the first batch."
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
        # If the declared path resolves outside benchmark_root (e.g. the
        # operator re-routed it to logs/phase_0d/<site>/storage_state.json
        # under state_dir), fall back to the phase_0d artifact when present.
        # The preflight in worldsim/storage_state_preflight.py already does
        # this; mirror the fallback here so agent launch stays consistent.
        try:
            path = _resolve_storage_state_path(raw_path.strip(), benchmark_root)
        except AuthArtifactMissingError:
            bootstrap_path = _phase_0d_fallback_path(task)
            if bootstrap_path is not None:
                session_kwargs["storage_state"] = str(bootstrap_path)
                return session_kwargs, deferred_actions
            raise
        # storage_state wins over any form_login that may coexist (plan §5 edges).
        if not path.exists():
            generator = sub.get("generator_script")
            # Phase 0d writes the generated artifact to
            # ``logs/phase_0d/<site>/storage_state.json``. Consult that path as
            # a fallback before declaring the artifact missing so the runtime
            # picks up bootstrapped credentials automatically.
            bootstrap_path = _phase_0d_fallback_path(task)
            if bootstrap_path is not None:
                session_kwargs["storage_state"] = str(bootstrap_path)
                return session_kwargs, deferred_actions
            if generator:
                raise AuthArtifactMissingError(
                    f"storage_state artifact missing at {path}; generator_script "
                    f"{generator!r} declared — run Phase 0d (auth-bootstrap) "
                    "before Phase 3."
                )
            raise AuthArtifactMissingError(
                f"storage_state artifact missing at {path} and no generator_script declared"
            )
        session_kwargs["storage_state"] = str(path)
        return session_kwargs, deferred_actions

    if mech_type == "http_basic":
        sub = auth_mechanism.get("http_basic") or {}
        username = sub.get("username")
        password = sub.get("password")
        if not username or not password:
            raise AuthArtifactMissingError(
                "auth_mechanism.http_basic requires non-empty username/password"
            )
        session_kwargs["http_credentials"] = {"username": username, "password": password}
        return session_kwargs, deferred_actions

    if mech_type == "http_headers":
        sub = auth_mechanism.get("http_headers") or {}
        headers = sub.get("headers")
        if not isinstance(headers, dict) or not headers:
            raise AuthArtifactMissingError(
                "auth_mechanism.http_headers requires a non-empty headers dict"
            )
        # Interpolate ${credentials.username}/${credentials.password} tokens so
        # benchmarks that declare an auto-login header with variable credentials
        # (e.g. Magento's X-M2-Customer-Auto-Login: <user>:<pass>) resolve at
        # runtime. Missing credentials fail closed only when the template
        # actually references them; static headers are preserved verbatim.
        creds = (auth_mechanism.get("authentication") or {}).get("credentials") or {}
        u, p = creds.get("username", ""), creds.get("password", "")
        resolved: dict[str, str] = {}
        for name, value in headers.items():
            if not isinstance(name, str) or not isinstance(value, str):
                raise AuthArtifactMissingError(
                    "auth_mechanism.http_headers.headers must be a string->string map"
                )
            needs_username = "${credentials.username}" in value
            needs_password = "${credentials.password}" in value
            if (needs_username or needs_password) and (
                not isinstance(creds, dict) or not u or not p
            ):
                raise AuthArtifactMissingError(
                    "auth_mechanism.http_headers references credentials placeholders but "
                    "authentication.credentials is missing username/password"
                )
            if needs_username or needs_password:
                resolved[name] = value.replace("${credentials.username}", u).replace(
                    "${credentials.password}", p
                )
            else:
                resolved[name] = value
        # NB: Browser Use's BrowserSession uses `headers=` (not Playwright's
        # `extra_http_headers=`); the kwarg is forwarded to the Playwright
        # context internally.
        session_kwargs["headers"] = resolved
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
        headless: bool = True,
    ) -> None:
        # ``llm`` is configured by the caller (see worldsim/main.py).
        # To use OpenRouter, pass an OpenRouter-configured ChatOpenAI instance.
        self.llm = llm
        self.use_vision = use_vision
        self.max_steps = max_steps
        self.timeout = timeout
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
        pvpo_cdp_url: str | None = None,
    ) -> AgentResult:
        from browser_use import Agent, BrowserSession

        task_dir = Path(task_dir)
        task_dir.mkdir(parents=True, exist_ok=True)
        self._task_origins = {
            origin for origin in (_origin_from_url(url) for url in (start_urls or [])) if origin
        }
        self._owned_target_ids = set()
        self._primary_target_id = None
        self._browser_runtime = {}

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
        )
        self._preserve_remote_auth_state = "storage_state" in session_auth_kwargs

        # PVPO integration: Phase 4 binds each worker to its own chrome-
        # headless-shell endpoint via the instance config. The shared global
        # WORLDSIM_PVPO_CDP_URL path is intentionally removed.
        resolved_pvpo_cdp_url = _resolve_pvpo_cdp_url(pvpo_cdp_url or "")
        self._pvpo_cdp_url = resolved_pvpo_cdp_url
        session_kwargs: dict[str, Any] = {
            "headless": self.headless,
            "keep_alive": False,
            **session_auth_kwargs,
        }
        if resolved_pvpo_cdp_url:
            session_kwargs["cdp_url"] = resolved_pvpo_cdp_url
        else:
            session_kwargs["args"] = [
                "--disable-gpu",
                "--disable-extensions",
                "--no-sandbox",  # required for Chrome on EC2/Docker/root
                "--disable-software-rasterizer",  # reduce CPU when GPU unavailable
            ]
        self._session = BrowserSession(**session_kwargs)

        # Retry browser startup with linear backoff for transient failures
        last_exc: Exception | None = None
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
        if last_exc is not None:
            raise last_exc

        history = None
        agent = None
        elapsed = 0.0
        network_trace: list[dict[str, Any]] = []
        network_recorder: _NetworkTraceRecorder | None = None
        status = "error"
        extra_errors: list[str] = []
        try:
            if self._pvpo_cdp_url:
                await self._reset_remote_browser_for_task(self._session)

            # Run any deferred auth actions (e.g. future form_login flow) after
            # session.start() succeeds and after any remote-browser reset has
            # produced a fresh task-owned target. No-op for the first batch
            # (storage_state / http_basic / none).
            for action in deferred_auth_actions:
                await action(self._session)

            network_recorder = _NetworkTraceRecorder(
                self._session,
                task_dir,
                target_filter=self._owned_target_ids,
            )
            await network_recorder.start()

            initial_actions = _build_initial_actions(start_urls or [])
            task_text = site_prompt if site_prompt else task
            # Per-session HeadlessExperimental.beginFrame pump. Required when
            # Chrome is launched with --enable-begin-frame-control (PVPO
            # rigor): browser-use 0.12.6 never issues beginFrame, so without
            # this pump step-1 navigation stalls the compositor and times
            # out. Pump is gated off during atomic PVPO capture via the
            # yielded ``capturing`` Event so the capture remains atomic.
            from worldsim.phase_4.pvpo_frame_pump import frame_pump

            async with frame_pump(self._session) as capturing:
                pvpo_hook = _make_pvpo_step_callback(
                    self._session,
                    task_dir,
                    payload_text,
                    owned_target_ids=self._owned_target_ids,
                    capturing=capturing,
                )
                agent = Agent(
                    task=task_text,
                    llm=self.llm,
                    browser_session=self._session,
                    use_vision=self.use_vision,
                    # WorldSim uses its own reward evaluators plus Phase 3/4
                    # diagnosis/judge flows; Browser Use's internal judge only adds
                    # post-hoc logging and currently breaks on the Anthropic-via-
                    # OpenRouter path.
                    use_judge=False,
                    save_conversation_path=str(task_dir / "conversations"),
                    initial_actions=initial_actions,
                    register_new_step_callback=pvpo_hook,
                )

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
                status=status,
                extra_errors=extra_errors,
            )
            if self._session is not None:
                # Clean up temp profile dir before killing to avoid /tmp accumulation
                self._cleanup_temp_profile(self._session)
                try:
                    await self._cleanup_external_cdp_state(self._session)
                except Exception as e:
                    logger.warning("Remote PVPO browser cleanup failed: %s", e)
                finally:
                    _write_browser_runtime_artifact(task_dir, self._browser_runtime)
                try:
                    await self._session.kill()
                except Exception as e:
                    logger.warning("BrowserSession kill failed: %s", e)
                self._session = None
                self._pvpo_cdp_url = ""
                self._task_origins = set()
                self._owned_target_ids = set()
                self._primary_target_id = None
                self._browser_runtime = {}
                self._preserve_remote_auth_state = False
            else:
                _write_browser_runtime_artifact(task_dir, self._browser_runtime)

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

    async def teardown(self) -> None:
        if self._session is not None:
            # Clean up temp profile dir before killing to avoid /tmp accumulation
            self._cleanup_temp_profile(self._session)
            try:
                await self._cleanup_external_cdp_state(self._session)
                await self._session.kill()
            except Exception as e:
                logger.warning("BrowserSession kill failed: %s", e)
            self._session = None
            self._pvpo_cdp_url = ""
            self._task_origins = set()
            self._owned_target_ids = set()
            self._primary_target_id = None
            self._browser_runtime = {}
            self._preserve_remote_auth_state = False

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
        if not self._preserve_remote_auth_state:
            await self._clear_page_storage(session, pages=pages, origins=set(self._task_origins))
            await self._clear_browser_cookies(session)
        await self._close_pages(session, pages)
        self._browser_runtime.update(
            {
                "cleanup_closed_targets": len(closed_target_ids),
                "cleanup_target_ids": closed_target_ids,
                "cleanup_origins": sorted(self._task_origins),
                "cleanup_preserved_auth_state": self._preserve_remote_auth_state,
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
        history.save_to_file(tmp_path)
        os.replace(tmp_path, history_path)
    except BaseException:
        with suppress(OSError):
            tmp_path.unlink()
        raise


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
        errors = [str(error) for error in history.errors()]
    except Exception:
        errors = []

    return steps, is_done, final_result, errors


def _make_pvpo_step_callback(
    session: Any,
    task_dir: Path,
    payload_text: str | None,
    *,
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

    When ``payload_text`` is None or empty, PVPO capture is disabled (we
    have nothing to locate in the DOM). This is the benign-task / no-seed
    case.

    See ``docs/handoffs/codex-handoff-paint-verified-oracle.md`` §3 and
    the Implementation Status section documenting the content-match
    anchor strategy.
    """
    if not payload_text:
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
        capture_summary["steps_seen"] += 1
        save_capture_summary(task_dir, capture_summary)
        try:
            page = await session.get_current_page()
        except Exception as exc:  # pragma: no cover - CDP unavailable
            _record_issue("current_page_unavailable", step_idx, str(exc))
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
            cdp_session = await session.get_or_create_cdp_session(target_id=target_id, focus=False)
        except Exception as exc:  # pragma: no cover - CDP unavailable
            _record_issue("cdp_session_unavailable", step_idx, str(exc))
            return

        try:
            if target_id not in pages_prepared:
                await inject_animation_killer(page, cdp_session)
                pages_prepared.add(target_id)
        except Exception as exc:
            _record_issue("animation_killer_failed", step_idx, str(exc))

        try:
            viewport = await runtime_evaluate_value(cdp_session, _CDP_VIEWPORT_JS)
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
                capturing=capturing,
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
            _record_issue("capture_failed", step_idx, str(exc))

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
            if destination.exists():
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
