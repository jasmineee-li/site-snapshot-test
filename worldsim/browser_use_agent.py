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
from urllib.parse import parse_qs, urlencode, urlparse, urlsplit, urlunsplit

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

            try:
                session = await self._browser_session.get_or_create_cdp_session(
                    target_id, focus=False
                )
                await session.cdp_client.send.Network.enable(session_id=session.session_id)
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
        raw_entries.sort(key=lambda e: (e.get("timestamp") is None, e.get("timestamp", 0)))
        return [self._flatten_entry(e) for e in raw_entries]

    def _write_trace(self, trace: list[dict[str, Any]]) -> None:
        """Write both the evaluator-ready ``network_trace.json`` and
        a HAR-envelope ``network.har`` to the task directory."""
        # Flat list for direct consumption by NetworkEventEvaluator.
        try:
            (self._task_dir / "network_trace.json").write_text(
                json.dumps(trace, indent=2, default=str)
            )
        except Exception as e:
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
            (self._task_dir / "network.har").write_text(json.dumps(payload, indent=2, default=str))
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
    try:
        from worldsim.phases.phase_0d_auth_bootstrap import phase_0d_artifact_path
    except ImportError:  # pragma: no cover — only triggers on misinstalled env.
        return None
    return phase_0d_artifact_path(site.strip())


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
        path = Path(raw_path)
        if not path.is_absolute() and benchmark_root is not None:
            path = Path(benchmark_root) / path
        # storage_state wins over any form_login that may coexist (plan §5 edges).
        if not path.exists():
            generator = sub.get("generator_script")
            # Phase 0d writes the generated artifact to
            # ``logs/phase_0d/<site>/storage_state.json``. Consult that path as
            # a fallback before declaring the artifact missing so the runtime
            # picks up bootstrapped credentials automatically.
            bootstrap_path = _phase_0d_fallback_path(task)
            if bootstrap_path is not None and bootstrap_path.exists():
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
        # runtime. Absent credentials leave the header value as-is.
        creds = (auth_mechanism.get("authentication") or {}).get("credentials") or {}
        u, p = creds.get("username", ""), creds.get("password", "")
        resolved: dict[str, str] = {}
        for name, value in headers.items():
            if not isinstance(name, str) or not isinstance(value, str):
                raise AuthArtifactMissingError(
                    "auth_mechanism.http_headers.headers must be a string->string map"
                )
            resolved[name] = value.replace("${credentials.username}", u).replace(
                "${credentials.password}", p
            )
        # NB: Browser Use's BrowserSession uses `headers=` (not Playwright's
        # `extra_http_headers=`); the kwarg is forwarded to the Playwright
        # context internally.
        session_kwargs["headers"] = resolved
        return session_kwargs, deferred_actions

    # Defensive fallback: an unknown-but-enumerated type slipped through.
    raise NotImplementedError(f"auth_mechanism.type={mech_type!r} has no runtime dispatcher")


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

    async def setup(self, server_url: str) -> None:
        # Browser sessions are task-scoped so trajectory artifacts remain
        # isolated per task directory.
        self._session = None

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
    ) -> AgentResult:
        from browser_use import Agent, BrowserSession

        task_dir = Path(task_dir)
        task_dir.mkdir(parents=True, exist_ok=True)

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

        self._session = BrowserSession(
            headless=self.headless,
            keep_alive=False,
            args=[
                "--disable-gpu",
                "--disable-extensions",
                "--no-sandbox",  # required for Chrome on EC2/Docker/root
                "--disable-software-rasterizer",  # reduce CPU when GPU unavailable
            ],
            **session_auth_kwargs,
        )

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

        # Run any deferred auth actions (e.g. future form_login flow) after
        # session.start() succeeds, before the Agent is constructed. No-op for
        # the first batch (storage_state / http_basic / none).
        for action in deferred_auth_actions:
            await action(self._session)

        network_recorder = _NetworkTraceRecorder(self._session, task_dir)
        await network_recorder.start()

        history = None
        agent = None
        elapsed = 0.0
        network_trace: list[dict[str, Any]] = []
        status = "error"
        extra_errors: list[str] = []
        try:
            initial_actions = _build_initial_actions(start_urls or [])
            task_text = site_prompt if site_prompt else task
            agent = Agent(
                task=task_text,
                llm=self.llm,
                browser_session=self._session,
                use_vision=self.use_vision,
                save_conversation_path=str(task_dir / "conversations"),
                initial_actions=initial_actions,
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
            network_trace = await network_recorder.stop()
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
                    await self._session.kill()
                except Exception as e:
                    logger.warning("BrowserSession kill failed: %s", e)
                self._session = None

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
                await self._session.kill()
            except Exception as e:
                logger.warning("BrowserSession kill failed: %s", e)
            self._session = None

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
            history.save_to_file(history_path)
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
        (task_dir / "final_response.json").write_text(
            json.dumps(final_response, indent=2, default=str)
        )
    except Exception as e:
        logger.warning("Failed to write final_response.json for %s: %s", task_dir, e)


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
            shutil.copy2(path_str, screenshots_dir / f"step_{step_idx}.png")


def _write_history_fallback(history_path: Path, errors: list[str], status: str) -> None:
    """Write a minimal history artifact when Browser Use exposes no history object."""
    payload = {
        "history": [],
        "partial": True,
        "status": status,
        "errors": errors,
    }
    try:
        history_path.write_text(json.dumps(payload, indent=2))
    except Exception as e:
        logger.warning("Failed to write fallback history.json for %s: %s", history_path, e)
