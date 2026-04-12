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
import shutil
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Summary of one agent run, extracted from the Browser Use history."""

    elapsed: float
    steps: int
    is_done: bool
    final_result: str | None
    errors: list[str] = field(default_factory=list)
    network_trace: list[dict[str, Any]] = field(default_factory=list)


class AgentRunner(Protocol):
    """Protocol every agent implementation in the worker pool must satisfy."""

    async def setup(self, server_url: str) -> None: ...

    async def run(
        self, task: str, server_url: str, task_dir: Path
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

    def __init__(self, browser_session: Any, task_dir: Path) -> None:
        self._browser_session = browser_session
        self._task_dir = Path(task_dir)
        self._client = getattr(browser_session, "cdp_client", None)
        self._recording = False
        self._poll_task: asyncio.Task | None = None
        self._enabled_targets: set[str] = set()
        # Raw CDP entries keyed by requestId.
        self._requests: dict[str, dict[str, Any]] = {}

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

        await self._enable_current_page_sessions()
        self._recording = True
        # Poll for newly-opened tabs/popups so Network.enable is sent promptly.
        self._poll_task = asyncio.create_task(
            self._poll_sessions(), name="network-trace-poller"
        )

    async def stop(self) -> list[dict[str, Any]]:
        """Stop recording, finalize trace, write to disk, return entries."""
        self._recording = False
        if self._poll_task is not None:
            self._poll_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._poll_task
            self._poll_task = None

        trace = self._finalize_trace()
        self._write_trace(trace)
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

            try:
                session = await self._browser_session.get_or_create_cdp_session(
                    target_id, focus=False
                )
                await session.cdp_client.send.Network.enable(
                    session_id=session.session_id
                )
                self._enabled_targets.add(target_id)
            except Exception as e:  # noqa: BLE001
                logger.debug(
                    "Network trace enable failed for target %s: %s", target_id, e
                )

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

    def _on_response_received(
        self, event: dict[str, Any], session_id: str | None = None
    ) -> None:
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

    def _on_loading_finished(
        self, event: dict[str, Any], session_id: str | None = None
    ) -> None:
        if not self._recording:
            return

        request_id = event.get("requestId")
        if not request_id:
            return

        entry = self._entry(request_id, session_id)
        entry["loading_finished"] = True
        entry["encoded_data_length"] = event.get("encodedDataLength")

    def _on_loading_failed(
        self, event: dict[str, Any], session_id: str | None = None
    ) -> None:
        if not self._recording:
            return

        request_id = event.get("requestId")
        if not request_id:
            return

        entry = self._entry(request_id, session_id)
        entry["loading_failed"] = True
        entry["error_text"] = event.get("errorText")
        entry["canceled"] = event.get("canceled")

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
        except Exception:  # noqa: BLE001
            pass

        # Merge request headers: prefer extra-info (wire-level) when available.
        headers = dict(raw.get("request_headers", {}))
        headers.update(raw.get("request_headers_extra", {}))

        # Merge response headers similarly.
        resp_headers = dict(raw.get("response_headers", {}))
        resp_headers.update(raw.get("response_headers_extra", {}))

        # Extract cookies from Set-Cookie response headers.
        resp_cookies = self._parse_cookies_from_headers(resp_headers)

        return {
            "url": url,
            "method": raw.get("method", "GET"),
            "headers": headers,
            "query_params": query_params,
            "post_data": raw.get("post_data"),
            "response_status": raw.get("response_status"),
            "response_headers": resp_headers,
            "response_cookies": resp_cookies,
        }

    def _finalize_trace(self) -> list[dict[str, Any]]:
        """Return flat, evaluator-ready entries sorted by CDP timestamp."""
        raw_entries = list(self._requests.values())
        raw_entries.sort(
            key=lambda e: (e.get("timestamp") is None, e.get("timestamp", 0))
        )
        return [self._flatten_entry(e) for e in raw_entries]

    def _write_trace(self, trace: list[dict[str, Any]]) -> None:
        """Write both the evaluator-ready ``network_trace.json`` and
        a HAR-envelope ``network.har`` to the task directory."""
        # Flat list for direct consumption by NetworkEventEvaluator.
        try:
            (self._task_dir / "network_trace.json").write_text(
                json.dumps(trace, indent=2, default=str)
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to write network_trace.json: %s", e)

        # HAR-envelope wrapping the same entries (tooling compatibility).
        payload = {
            "log": {
                "version": "1.2",
                "creator": {
                    "name": "worldsim",
                    "version": "phase-3-network-trace",
                },
                "entries": trace,
            }
        }
        try:
            (self._task_dir / "network.har").write_text(
                json.dumps(payload, indent=2, default=str)
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to write network.har: %s", e)


class BrowserUseAgent:
    """Browser Use-backed :class:`AgentRunner` (spec canonical implementation)."""

    def __init__(
        self,
        llm: Any,
        *,
        use_vision: bool = False,
        max_steps: int = 50,
        timeout: int = 300,
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

    async def setup(self, server_url: str) -> None:
        # Browser sessions are task-scoped so trajectory artifacts remain
        # isolated per task directory.
        self._session = None

    async def run(
        self, task: str, server_url: str, task_dir: Path
    ) -> AgentResult:
        from browser_use import Agent, BrowserSession

        task_dir = Path(task_dir)
        task_dir.mkdir(parents=True, exist_ok=True)

        self._session = BrowserSession(
            headless=self.headless,
            keep_alive=False,
            args=[
                "--disable-gpu",
                "--disable-extensions",
            ],
        )
        await self._session.start()

        network_recorder = _NetworkTraceRecorder(self._session, task_dir)
        await network_recorder.start()

        history = None
        elapsed = 0.0
        network_trace: list[dict[str, Any]] = []
        try:
            agent = Agent(
                task=(
                    f"You are interacting with a web application at {server_url}. "
                    f"Your task: {task}"
                ),
                llm=self.llm,
                browser_session=self._session,
                use_vision=self.use_vision,
                save_conversation_path=str(task_dir / "conversations"),
                max_steps=self.max_steps,
            )

            t0 = time.time()
            history = await asyncio.wait_for(agent.run(), timeout=self.timeout)
            elapsed = time.time() - t0

            history.save_to_file(task_dir / "history.json")

            screenshots_dir = task_dir / "screenshots"
            for step_idx, path_str in enumerate(history.screenshot_paths()):
                if path_str and Path(path_str).exists():
                    screenshots_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path_str, screenshots_dir / f"step_{step_idx}.png")

            final_response = {
                "status": "SUCCESS" if history.is_done() else "FAILURE",
                "final_result": history.final_result(),
                "errors": history.errors(),
                "steps": len(history.history),
            }
            (task_dir / "final_response.json").write_text(
                json.dumps(final_response, indent=2)
            )
        finally:
            network_trace = await network_recorder.stop()
            if self._session is not None:
                try:
                    await self._session.kill()
                except Exception as e:  # noqa: BLE001
                    logger.warning("BrowserSession kill failed: %s", e)
                self._session = None

        return AgentResult(
            elapsed=round(elapsed, 1),
            steps=len(history.history) if history is not None else 0,
            is_done=history.is_done() if history is not None else False,
            final_result=history.final_result() if history is not None else None,
            errors=history.errors() if history is not None else [],
            network_trace=network_trace,
        )

    async def teardown(self) -> None:
        if self._session is not None:
            try:
                await self._session.kill()
            except Exception as e:  # noqa: BLE001
                logger.warning("BrowserSession kill failed: %s", e)
            self._session = None
