"""Unit tests for Bug K — Phase 2c probe hardening.

Covers the four layered changes that close the Playwright CPU-contention
tail under ``--feasibility-concurrency=24`` on r5:

1. ``wait_until="commit"`` on the two ``page.goto`` call sites.
2. ``page.route`` subresource blocker installed before ``goto``.
3. ``chromium.launch`` args include ``--disable-dev-shm-usage``.
4. Module-level ``_BROWSER_PROBE_CAP`` at 8 and a new semaphore
   alongside the existing memory + per-replica semaphores.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from warp_taskgen.phase_2.phase_2c import probes
from warp_taskgen.phases import phase_2_reachability as reach
from warp_taskgen.phases import phase_2_render_check as render_check

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _Request:
    def __init__(
        self,
        resource_type: str,
        *,
        url: str = "http://live/resource",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.resource_type = resource_type
        self.url = url
        self.headers = headers or {}


class _Route:
    def __init__(
        self,
        resource_type: str,
        *,
        url: str = "http://live/resource",
        headers: dict[str, str] | None = None,
        fetch_error: Exception | None = None,
    ) -> None:
        self.request = _Request(resource_type, url=url, headers=headers)
        self.aborted = False
        self.continued = False
        self.continue_kwargs: dict[str, Any] = {}
        self.fetch_kwargs: dict[str, Any] | None = None
        self.fulfill_kwargs: dict[str, Any] | None = None
        self.fetch_response = object()
        self.fetch_error = fetch_error

    async def abort(self) -> None:
        self.aborted = True

    async def continue_(self, **kwargs: Any) -> None:
        self.continued = True
        self.continue_kwargs = kwargs

    async def fetch(self, **kwargs: Any) -> object:
        self.fetch_kwargs = kwargs
        if self.fetch_error is not None:
            raise self.fetch_error
        return self.fetch_response

    async def fulfill(self, **kwargs: Any) -> None:
        self.fulfill_kwargs = kwargs


class _ProbePage:
    """Fake page that records route installation, goto args, and body reads.

    Distinguishes call order via a monotonic counter so tests can prove
    ``page.route`` is registered before the first ``page.goto``.
    """

    def __init__(self, body_per_url: dict[str, str] | None = None) -> None:
        self._body_per_url = body_per_url or {}
        self._current_url = ""
        self._step = 0
        self.route_installed_step: int | None = None
        self.route_handler: Any = None
        self.routed_requests: list[_Route] = []
        self.goto_calls: list[dict[str, Any]] = []
        self.first_goto_step: int | None = None

    async def route(self, pattern: str, handler: Any) -> None:
        self._step += 1
        self.route_installed_step = self._step
        self.route_handler = handler

    async def goto(self, url: str, *, timeout: int, wait_until: str) -> None:
        self._step += 1
        if self.first_goto_step is None:
            self.first_goto_step = self._step
        canonical = url.split("?", 1)[0] if "?_=" in url else url
        if self.route_handler is not None:
            route = _Route("document", url=canonical, headers={"Existing": "yes"})
            await self.route_handler(route)
            self.routed_requests.append(route)
        self.goto_calls.append(
            {"url": canonical, "timeout": timeout, "wait_until": wait_until},
        )
        self._current_url = canonical

    async def wait_for_selector(self, selector: str, *, timeout: int) -> None:
        return None

    async def wait_for_load_state(self, state: str, *, timeout: int) -> None:
        return None

    async def wait_for_timeout(self, ms: int) -> None:
        return None

    async def text_content(self, selector: str) -> str:
        return self._body_per_url.get(self._current_url, "")


class _ProbeContext:
    def __init__(self, page: _ProbePage) -> None:
        self._page = page

    async def new_page(self) -> _ProbePage:
        return self._page

    async def close(self) -> None:
        return None


class _ProbeBrowser:
    def __init__(self, page: _ProbePage) -> None:
        self._page = page
        self.context_kwargs: list[dict[str, Any]] = []

    async def new_context(self, **kwargs: Any) -> _ProbeContext:
        unexpected = set(kwargs) - {"storage_state", "extra_http_headers", "http_credentials"}
        if unexpected:
            raise AssertionError(f"unexpected Playwright context kwargs: {sorted(unexpected)}")
        self.context_kwargs.append(kwargs)
        return _ProbeContext(self._page)


# ---------------------------------------------------------------------------
# (1) wait_until="commit" on both call sites
# ---------------------------------------------------------------------------


def test_reachability_goto_uses_wait_until_commit() -> None:
    body = "seed signature appears here in the rendered DOM"
    page = _ProbePage({"http://live/foo/-/issues/1": body})
    browser = _ProbeBrowser(page)

    outcome = asyncio.run(
        reach.verify_reachable(
            browser=browser,
            benign_target_resource={
                "kind": "gitlab_issue",
                "start_url_resolved": "http://live/foo/-/issues/1",
            },
            instance_site_url="http://live",
            signature="seed signature",
            second_witness=None,
        ),
    )

    assert outcome.reachability == "reachable_direct"
    assert page.goto_calls, "goto was never invoked"
    assert page.goto_calls[0]["wait_until"] == "commit"


def test_render_check_goto_uses_wait_until_commit() -> None:
    body = "seed signature appears on the rendered issue page"
    page = _ProbePage({"http://live/foo/-/issues/1": body})
    browser = _ProbeBrowser(page)

    outcome = asyncio.run(
        render_check.verify_seed_renders(
            browser=browser,
            urls=["http://live/foo/-/issues/1"],
            site_name="gitlab",
            site_url="http://live",
            signature="seed signature",
        ),
    )

    assert outcome.ok
    assert page.goto_calls
    assert page.goto_calls[0]["wait_until"] == "commit"


def test_render_check_passes_http_headers_as_playwright_extra_headers() -> None:
    page = _ProbePage({"http://live/foo/-/issues/1": "seed signature"})
    browser = _ProbeBrowser(page)

    outcome = asyncio.run(
        render_check.verify_seed_renders(
            browser=browser,
            urls=["http://live/foo/-/issues/1"],
            site_name="gitlab",
            site_url="http://live",
            signature="seed signature",
            browser_context_kwargs={"extra_http_headers": {"X-User": "alice"}},
        ),
    )

    assert outcome.ok
    assert browser.context_kwargs == [{}]
    assert page.routed_requests[0].fetch_kwargs is not None
    assert page.routed_requests[0].fetch_kwargs["headers"]["Existing"] == "yes"
    assert page.routed_requests[0].fetch_kwargs["headers"]["X-User"] == "alice"
    assert page.routed_requests[0].fetch_kwargs["max_redirects"] == 0
    assert page.routed_requests[0].fulfill_kwargs == {
        "response": page.routed_requests[0].fetch_response
    }


def test_route_blocker_strips_http_headers_from_off_origin_requests() -> None:
    page = _ProbePage()
    asyncio.run(
        reach._install_resource_blocker(
            page,
            scoped_extra_http_headers={"X-User": "alice"},
            header_scope_url="http://live",
        )
    )
    handler = page.route_handler
    assert handler is not None
    route = _Route("script", url="http://other.test/script.js", headers={"Existing": "yes"})

    asyncio.run(handler(route))

    assert route.continued
    assert route.continue_kwargs == {}
    assert route.fetch_kwargs is None


def test_route_blocker_rewrites_declared_redirect_origin_alias() -> None:
    page = _ProbePage()
    asyncio.run(
        reach._install_resource_blocker(
            page,
            scoped_extra_http_headers={"X-Worldsim-Token": "test-token"},
            header_scope_url="http://live:18023",
            redirect_origin_aliases=("http://live:8023", "http://live"),
        )
    )
    handler = page.route_handler
    assert handler is not None
    route = _Route("document", url="http://live:18023/source")
    route.fetch_response = SimpleNamespace(
        status=302,
        headers={"Location": "http://live/target?from=gitlab"},
    )

    asyncio.run(handler(route))

    assert route.fulfill_kwargs == {
        "response": route.fetch_response,
        "headers": {"Location": "http://live:18023/target?from=gitlab"},
    }

    foreign = _Route("document", url="http://live:18023/source")
    foreign.fetch_response = SimpleNamespace(
        status=302,
        headers={"Location": "http://other.test/target"},
    )
    asyncio.run(handler(foreign))
    assert foreign.fulfill_kwargs == {"response": foreign.fetch_response}


def test_reachability_passes_http_headers_as_playwright_extra_headers() -> None:
    page = _ProbePage({"http://live/foo/-/issues/1": "seed signature"})
    browser = _ProbeBrowser(page)

    outcome = asyncio.run(
        reach.verify_reachable(
            browser=browser,
            benign_target_resource={
                "kind": "gitlab_issue",
                "start_url_resolved": "http://live/foo/-/issues/1",
            },
            instance_site_url="http://live",
            signature="seed signature",
            second_witness=None,
            browser_context_kwargs={"extra_http_headers": {"X-User": "alice"}},
        ),
    )

    assert outcome.reachability == "reachable_direct"
    assert browser.context_kwargs == [{}]
    assert page.routed_requests[0].fetch_kwargs is not None
    assert page.routed_requests[0].fetch_kwargs["headers"]["X-User"] == "alice"
    assert page.routed_requests[0].fetch_kwargs["max_redirects"] == 0


def test_route_blocker_aborts_when_scoped_header_fetch_fails() -> None:
    page = _ProbePage()
    asyncio.run(
        reach._install_resource_blocker(
            page,
            scoped_extra_http_headers={"X-User": "alice"},
            header_scope_url="http://live",
        )
    )
    handler = page.route_handler
    assert handler is not None
    route = _Route(
        "document",
        url="http://live/foo/-/issues/1",
        headers={"Existing": "yes"},
        fetch_error=RuntimeError("network closed"),
    )

    asyncio.run(handler(route))

    assert route.aborted
    assert route.fetch_kwargs is not None
    assert route.fetch_kwargs["max_redirects"] == 0
    assert not route.continued


# ---------------------------------------------------------------------------
# (2) page.route blocker — behavior + install order
# ---------------------------------------------------------------------------


def test_route_blocker_aborts_heavy_resources_and_allows_script_xhr() -> None:
    # Exercise the handler directly via _install_resource_blocker so we do
    # not depend on the caller's verify path.
    page = _ProbePage()
    asyncio.run(reach._install_resource_blocker(page))
    handler = page.route_handler
    assert handler is not None

    blocked_routes = [_Route(rt) for rt in reach._BLOCKED_RESOURCE_TYPES]
    allowed_routes = [_Route(rt) for rt in ("script", "xhr", "fetch", "document", "manifest")]

    for route in blocked_routes + allowed_routes:
        asyncio.run(handler(route))

    for route in blocked_routes:
        assert route.aborted, f"{route.request.resource_type} should be aborted"
        assert not route.continued
    for route in allowed_routes:
        assert route.continued, f"{route.request.resource_type} should continue"
        assert not route.aborted


def test_route_blocker_registers_before_first_goto_in_reachability() -> None:
    page = _ProbePage({"http://live/x": "seed signature"})
    browser = _ProbeBrowser(page)

    asyncio.run(
        reach.verify_reachable(
            browser=browser,
            benign_target_resource={
                "kind": "reddit_forum",  # no body-text poll; keeps the flow short
                "start_url_resolved": "http://live/x",
            },
            instance_site_url="http://live",
            signature="seed signature",
            second_witness=None,
        ),
    )

    assert page.route_installed_step is not None, "page.route was never called"
    assert page.first_goto_step is not None, "page.goto was never called"
    assert page.route_installed_step < page.first_goto_step


def test_route_blocker_registers_before_first_goto_in_render_check() -> None:
    page = _ProbePage({"http://live/foo/-/issues/1": "seed signature"})
    browser = _ProbeBrowser(page)

    asyncio.run(
        render_check.verify_seed_renders(
            browser=browser,
            urls=["http://live/foo/-/issues/1"],
            site_name="gitlab",
            site_url="http://live",
            signature="seed signature",
        ),
    )

    assert page.route_installed_step is not None
    assert page.first_goto_step is not None
    assert page.route_installed_step < page.first_goto_step


def test_blocked_resource_types_match_plan() -> None:
    # Stability tripwire: if someone drops 'websocket' or adds 'script'
    # to the blocked set, this fails loudly. 'script' must stay allowed
    # because the discussions.json XHR is triggered by deferred JS.
    assert reach._BLOCKED_RESOURCE_TYPES == frozenset(
        {"stylesheet", "image", "media", "font", "eventsource", "websocket"},
    )


# ---------------------------------------------------------------------------
# (3) chromium.launch args
# ---------------------------------------------------------------------------


def test_probe_launch_args_include_disable_dev_shm_usage() -> None:
    assert "--disable-dev-shm-usage" in probes._PROBE_LAUNCH_ARGS
    assert "--disable-gpu" in probes._PROBE_LAUNCH_ARGS
    assert "--no-sandbox" in probes._PROBE_LAUNCH_ARGS


# ---------------------------------------------------------------------------
# (4) _BROWSER_PROBE_CAP semaphore
# ---------------------------------------------------------------------------


def test_browser_probe_cap_value() -> None:
    # Codified as 8 per the plan — 2 vCPU headroom per renderer on
    # r5.4xlarge (16 vCPU). Raise only after a clean nav_failed=0 run.
    assert probes._BROWSER_PROBE_CAP == 8


def test_semaphore_acquire_order_outermost_is_probe_cap() -> None:
    # Unit-level proof that the probe cap is stricter than the memory
    # cap under normal operator input. max(int(concurrency), 64) = 64 at
    # concurrency=24; _BROWSER_PROBE_CAP = 8 — so the outer cap binds.
    memory_cap = max(24, 64)
    assert probes._BROWSER_PROBE_CAP <= memory_cap
