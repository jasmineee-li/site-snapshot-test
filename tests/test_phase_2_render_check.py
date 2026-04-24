"""Unit tests for Phase 2c post-seed render verification.

These tests use a fake Browser/Context/Page hierarchy so we exercise the
verifier's logic without depending on Playwright. The integration test
``tests/integration/test_phase_2_feasibility_live.py`` is responsible
for end-to-end coverage with a real browser against r5.
"""

from __future__ import annotations

import asyncio

import pytest

from worldsim.phases.phase_2_render_check import (
    RenderOutcome,
    render_signature,
    verify_seed_renders,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakePage:
    def __init__(
        self, *, body_per_url: dict[str, str], goto_raises: dict[str, Exception] | None = None
    ) -> None:
        self._body_per_url = body_per_url
        self._goto_raises = goto_raises or {}
        self._current_url = ""
        self.goto_calls: list[tuple[str, str]] = []  # (url, wait_until)
        self.load_state_calls: list[tuple[str, int]] = []

    async def goto(self, url, *, timeout, wait_until):
        # Strip cache-buster query string before matching against the test
        # body map so tests can key on the canonical URL.
        canonical = url.split("?", 1)[0] if "?_=" in url else url
        self.goto_calls.append((canonical, wait_until))
        for raising_url, exc in self._goto_raises.items():
            if raising_url in canonical:
                raise exc
        self._current_url = canonical

    async def text_content(self, selector):
        return self._body_per_url.get(self._current_url, "")

    async def wait_for_selector(self, selector, *, timeout):
        return None

    async def wait_for_load_state(self, state, *, timeout):
        self.load_state_calls.append((state, timeout))
        return None

    async def wait_for_timeout(self, ms):
        # Bug J: the body-text poll sleeps via page.wait_for_timeout;
        # fake is a no-op so tests stay deterministic.
        return None

    async def route(self, pattern, handler):
        # Bug K: tests preceded the page.route blocker; swallow here so
        # verify_seed_renders can install the handler without exploding
        # the fake. Handler itself is unit-tested in
        # tests/test_phase_2_probe_hardening.py.
        return None

    def wait_for_response(self, predicate, *, timeout):
        async def _await_noop():
            return None

        return _await_noop()


class _FakeContext:
    def __init__(self, page: _FakePage) -> None:
        self._page = page
        self.closed = False

    async def new_page(self):
        return self._page

    async def close(self):
        self.closed = True


class _FakeBrowser:
    def __init__(self, page: _FakePage) -> None:
        self._page = page
        self.contexts: list[_FakeContext] = []

    async def new_context(self):
        ctx = _FakeContext(self._page)
        self.contexts.append(ctx)
        return ctx


# ---------------------------------------------------------------------------
# render_signature
# ---------------------------------------------------------------------------


def test_render_signature_prefers_nickname():
    seed = {
        "editor_calls": [{"args": {"nickname": "SeedNickAdv003", "detail": "long detail text"}}]
    }
    assert render_signature(seed) == "SeedNickAdv003"


def test_render_signature_falls_back_to_detail_first_40_chars():
    seed = {"editor_calls": [{"args": {"detail": "x" * 100}}]}
    sig = render_signature(seed)
    assert sig is not None
    assert len(sig) == 40


def test_render_signature_avoids_gitlab_rewritten_token_boundaries():
    seed = {
        "editor_calls": [
            {
                "args": {
                    "body": (
                        "Raising priority on issue #1404 in "
                        "localhost:8023/a11yproject/a11yproject.com — we need "
                        "to confirm whether this is closed or still open before "
                        "the next deploy window."
                    )
                }
            }
        ]
    }

    sig = render_signature(seed)

    assert sig == "we need to confirm whether this is close"
    assert "localhost:8023" not in sig
    assert "#1404" not in sig


def test_render_signature_tries_alternate_body_fields():
    for field_name in ("body", "description", "note", "content"):
        seed = {"editor_calls": [{"args": {field_name: "the body of the post"}}]}
        assert render_signature(seed) == "the body of the post"


def test_render_signature_falls_back_to_title():
    seed = {"editor_calls": [{"args": {"title": "A reasonable title"}}]}
    assert render_signature(seed) == "A reasonable title"


def test_render_signature_scans_later_editor_calls():
    seed = {
        "editor_calls": [
            {"args": {"project_id": "{benign_project_id}"}},
            {"args": {"body_template": "later call visible payload text"}},
        ]
    }
    assert render_signature(seed) == "later call visible payload text"


def test_render_signature_prefers_last_signature_bearing_call():
    seed = {
        "editor_calls": [
            {"args": {"description_template": "setup resource description"}},
            {"args": {"body_template": "later call visible payload text"}},
        ]
    }
    assert render_signature(seed) == "later call visible payload text"


def test_render_signature_prefers_provenance_contributing_method():
    seed = {
        "editor_calls": [
            {
                "site": "gitlab",
                "method": "create_project",
                "args": {"description_template": "helper setup description"},
            },
            {
                "site": "gitlab",
                "method": "create_issue_note",
                "args": {"body_template": "visible issue note payload"},
            },
        ]
    }
    metadata = {"read_surface_provenance": {"editor_method": ["gitlab.create_issue_note"]}}

    assert render_signature(seed, metadata) == "visible issue note payload"


def test_render_signature_returns_none_when_no_signature_field():
    seed = {"editor_calls": [{"args": {"unrelated_field": "x"}}]}
    assert render_signature(seed) is None


def test_render_signature_returns_none_for_malformed_seed():
    assert render_signature(None) is None  # type: ignore[arg-type]
    assert render_signature({}) is None
    assert render_signature({"editor_calls": []}) is None
    assert render_signature({"editor_calls": [None]}) is None
    assert render_signature({"editor_calls": [{"args": None}]}) is None


# ---------------------------------------------------------------------------
# verify_seed_renders
# ---------------------------------------------------------------------------


def _run(coro):
    return (
        asyncio.get_event_loop().run_until_complete(coro)
        if asyncio._get_running_loop()
        else asyncio.run(coro)
    )


@pytest.mark.asyncio
async def test_verify_returns_render_unverified_when_no_urls():
    browser = _FakeBrowser(_FakePage(body_per_url={}))
    outcome = await verify_seed_renders(
        browser=browser,
        urls=[],
        site_name="shopping",
        site_url="http://shop.test",
        signature="SeedNickAdv003",
    )
    assert not outcome.ok
    assert outcome.kind == "render_unverified"
    assert "no read_surface_urls" in outcome.detail


@pytest.mark.asyncio
async def test_verify_returns_render_unverified_when_no_signature():
    browser = _FakeBrowser(_FakePage(body_per_url={}))
    outcome = await verify_seed_renders(
        browser=browser,
        urls=["/x"],
        site_name="shopping",
        site_url="http://shop.test",
        signature=None,
    )
    assert not outcome.ok
    assert outcome.kind == "render_unverified"
    assert "no extractable render signature" in outcome.detail


@pytest.mark.asyncio
async def test_verify_passes_when_signature_present_in_first_url():
    page = _FakePage(
        body_per_url={
            "http://shop.test/catalog/product/view/id/67": (
                "Product page reviewed by SeedNickAdv003 with stars"
            )
        }
    )
    browser = _FakeBrowser(page)
    outcome = await verify_seed_renders(
        browser=browser,
        urls=["/catalog/product/view/id/67"],
        site_name="shopping",
        site_url="http://shop.test",
        signature="SeedNickAdv003",
    )
    assert outcome.ok
    assert outcome.matched_signature == "SeedNickAdv003"
    assert "SeedNickAdv003" in (outcome.matched_snippet or "")
    assert browser.contexts[0].closed  # context torn down


@pytest.mark.asyncio
async def test_verify_falls_through_to_second_url_when_first_lacks_signature():
    page = _FakePage(
        body_per_url={
            "http://shop.test/catalog/product/view/id/67": "stale FPC PDP without payload",
            "http://shop.test/review/product/listAjax/id/67": '{"reviews": [{"author": "SeedNickAdv003"}]}',
        }
    )
    browser = _FakeBrowser(page)
    outcome = await verify_seed_renders(
        browser=browser,
        urls=[
            "/catalog/product/view/id/67",
            "/review/product/listAjax/id/67",
        ],
        site_name="shopping",
        site_url="http://shop.test",
        signature="SeedNickAdv003",
    )
    assert outcome.ok
    assert "listAjax" in (outcome.matched_url or "")


@pytest.mark.asyncio
async def test_verify_returns_render_unverified_when_no_url_has_signature():
    page = _FakePage(
        body_per_url={
            "http://shop.test/catalog/product/view/id/67": "stale FPC PDP",
            "http://shop.test/review/product/listAjax/id/67": '{"reviews": []}',
        }
    )
    browser = _FakeBrowser(page)
    outcome = await verify_seed_renders(
        browser=browser,
        urls=[
            "/catalog/product/view/id/67",
            "/review/product/listAjax/id/67",
        ],
        site_name="shopping",
        site_url="http://shop.test",
        signature="SeedNickAdv003",
    )
    assert not outcome.ok
    assert outcome.kind == "render_unverified"
    # both URLs were probed and recorded as failures
    assert len(outcome.urls_tried) == 2
    for url in outcome.urls_tried:
        assert "signature_absent" in outcome.per_url_errors[url]


@pytest.mark.asyncio
async def test_verify_classifies_all_timeouts_as_host_unreachable():
    err = TimeoutError("timeout exceeded")
    page = _FakePage(
        body_per_url={},
        goto_raises={
            "http://shop.test/a": err,
            "http://shop.test/b": err,
        },
    )
    browser = _FakeBrowser(page)
    outcome = await verify_seed_renders(
        browser=browser,
        urls=["/a", "/b"],
        site_name="shopping",
        site_url="http://shop.test",
        signature="SeedNickAdv003",
    )
    assert not outcome.ok
    assert outcome.kind == "host_unreachable"


@pytest.mark.asyncio
async def test_verify_mixed_failure_classified_as_render_unverified():
    """One URL refuses connection, another loads but lacks payload — the
    "real bug we're catching" classification (render_unverified) wins
    because at least one URL did render, just without the signature."""

    class _ConnRefused(Exception):
        def __str__(self) -> str:
            return "net::ERR_CONNECTION_REFUSED"

    page = _FakePage(
        body_per_url={
            "http://shop.test/b": "page loaded but no payload visible",
        },
        goto_raises={
            "http://shop.test/a": _ConnRefused(),
        },
    )
    browser = _FakeBrowser(page)
    outcome = await verify_seed_renders(
        browser=browser,
        urls=["/a", "/b"],
        site_name="shopping",
        site_url="http://shop.test",
        signature="SeedNickAdv003",
    )
    assert not outcome.ok
    assert outcome.kind == "render_unverified"


@pytest.mark.asyncio
async def test_verify_appends_cache_buster_to_url():
    visited: list[str] = []

    class _RecordingPage(_FakePage):
        async def goto(self, url, *, timeout, wait_until):
            visited.append(url)
            self._current_url = url.split("?", 1)[0] if "?_=" in url else url

    page = _RecordingPage(body_per_url={"http://shop.test/x": "contains SeedNickAdv003"})
    browser = _FakeBrowser(page)
    outcome = await verify_seed_renders(
        browser=browser,
        urls=["/x"],
        site_name="shopping",
        site_url="http://shop.test",
        signature="SeedNickAdv003",
    )
    assert outcome.ok
    assert len(visited) == 1
    assert "?_=" in visited[0]


@pytest.mark.asyncio
async def test_render_outcome_evidence_passed_includes_match_fields():
    outcome = RenderOutcome.passed(
        url="http://shop.test/x",
        signature="SeedNickAdv003",
        snippet="...prefix SeedNickAdv003 suffix...",
    )
    evidence = outcome.evidence()
    assert evidence["ok"] is True
    assert evidence["matched_url"] == "http://shop.test/x"
    assert evidence["matched_signature"] == "SeedNickAdv003"
    assert "SeedNickAdv003" in evidence["matched_snippet"]


@pytest.mark.asyncio
async def test_render_outcome_evidence_failed_includes_kind_and_errors():
    outcome = RenderOutcome.failed(
        kind="render_unverified",
        detail="signature absent",
        urls_tried=["http://shop.test/a", "http://shop.test/b"],
        per_url_errors={"http://shop.test/a": "signature_absent"},
    )
    evidence = outcome.evidence()
    assert evidence["ok"] is False
    assert evidence["kind"] == "render_unverified"
    assert evidence["urls_tried"] == ["http://shop.test/a", "http://shop.test/b"]
    assert evidence["per_url_errors"] == {"http://shop.test/a": "signature_absent"}


@pytest.mark.asyncio
async def test_gitlab_render_never_uses_networkidle():
    """Regression guard: ``wait_until="networkidle"`` is Playwright-
    documented as DISCOURAGED and was the root cause of the 2026-04-22
    Phase 2c timeout class. Post-Bug-K (2026-04-23) the probe uses
    ``commit`` — the fastest goto phase — and gates readiness on the
    per-site selector wait + body-text poll.
    """
    url = "http://gitlab.test/myproject/-/issues/1"
    page = _FakePage(body_per_url={url: "notes body containing SeedNickAdv003"})
    browser = _FakeBrowser(page)

    outcome = await verify_seed_renders(
        browser=browser,
        urls=[url],
        site_name="gitlab",
        site_url="http://gitlab.test",
        signature="SeedNickAdv003",
    )
    assert outcome.ok is True
    assert page.goto_calls, "expected a goto call"
    for _url, wait_until in page.goto_calls:
        assert wait_until != "networkidle", (
            f"networkidle is discouraged and was removed on 2026-04-22; "
            f"got wait_until={wait_until!r} for url={_url!r}"
        )
        assert wait_until == "commit"


# ---------------------------------------------------------------------------
# Bug J: render_check body-text polling after GitLab selector wait
# ---------------------------------------------------------------------------


class _PollingFakePage:
    """Fake page whose body changes across successive text_content calls.

    Simulates GitLab's lazy discussions.json batch-2/3 arrival: first
    few polls see only the SPA shell + batch-1 notes (no signature);
    a later poll sees the seeded note appear in the body text.
    """

    def __init__(self, bodies: list[str], final_body: str = "") -> None:
        self._bodies = list(bodies)
        self._final = final_body
        self._current_url = ""
        self.goto_calls: list[tuple[str, str]] = []
        self.load_state_calls: list[tuple[str, int]] = []
        self.text_content_calls = 0

    async def goto(self, url, *, timeout, wait_until):
        canonical = url.split("?", 1)[0] if "?_=" in url else url
        self.goto_calls.append((canonical, wait_until))
        self._current_url = canonical

    async def text_content(self, selector):
        self.text_content_calls += 1
        if self._bodies:
            return self._bodies.pop(0)
        return self._final

    async def wait_for_selector(self, selector, *, timeout):
        return None

    async def wait_for_load_state(self, state, *, timeout):
        self.load_state_calls.append((state, timeout))
        return None

    async def wait_for_timeout(self, ms):
        return None

    async def route(self, pattern, handler):
        return None


class _PollingFakeContext:
    def __init__(self, page: _PollingFakePage) -> None:
        self._page = page

    async def new_page(self):
        return self._page

    async def close(self):
        return None


class _PollingFakeBrowser:
    def __init__(self, page: _PollingFakePage) -> None:
        self._page = page

    async def new_context(self, **kwargs):
        return _PollingFakeContext(self._page)


@pytest.mark.asyncio
async def test_body_poll_detects_late_arriving_signature():
    # First 3 polls return empty body (batch-2 hasn't loaded); 4th
    # returns the populated body with the signature. verify_seed_renders
    # must find it and not iterate to URL 2.
    url1 = "http://gitlab.test/proj/-/issues/5"
    url2 = "http://gitlab.test/proj/-/issues/5/discussions.json"
    body_populated = "full issue page with SEEDSIG123 in notes"
    page = _PollingFakePage(
        bodies=["", "", "", body_populated, body_populated],
        final_body=body_populated,
    )
    browser = _PollingFakeBrowser(page)

    outcome = await verify_seed_renders(
        browser=browser,
        urls=[url1, url2],
        site_name="gitlab",
        site_url="http://gitlab.test",
        signature="SEEDSIG123",
    )
    assert outcome.ok is True
    # Only URL 1 should be visited — the signature arrives mid-poll.
    assert len(page.goto_calls) == 1
    assert page.goto_calls[0][0] == url1


@pytest.mark.asyncio
async def test_body_poll_times_out_falls_through_to_signature_absent():
    # Body never contains the signature within the poll window. Fall
    # through to the existing signature_absent classification.
    url = "http://gitlab.test/proj/-/issues/5"
    page = _PollingFakePage(bodies=[], final_body="body without the signature")
    browser = _PollingFakeBrowser(page)

    outcome = await verify_seed_renders(
        browser=browser,
        urls=[url],
        site_name="gitlab",
        site_url="http://gitlab.test",
        signature="MISSING_SIGNATURE",
        selector_timeout_ms=200,  # keep the test fast
    )
    assert outcome.ok is False
    assert outcome.kind == "render_unverified"


@pytest.mark.asyncio
async def test_body_poll_only_runs_for_gitlab_issue_mr():
    # Reddit and other gitlab URLs must NOT trigger the poll. Fake's
    # text_content_calls == 1 confirms only the final body read ran.
    url = "http://reddit.test/f/books/1"
    page = _PollingFakePage(
        bodies=["body with REDDITSIG once"],
        final_body="body with REDDITSIG once",
    )
    browser = _PollingFakeBrowser(page)

    outcome = await verify_seed_renders(
        browser=browser,
        urls=[url],
        site_name="reddit",
        site_url="http://reddit.test",
        signature="REDDITSIG",
    )
    assert outcome.ok is True
    # Exactly one text_content call — the final body read, no polling.
    assert page.text_content_calls == 1
    assert page.load_state_calls == [("domcontentloaded", 10000)]


@pytest.mark.asyncio
async def test_reddit_waits_for_domcontentloaded_before_body_read():
    class _RedditReadyPage(_FakePage):
        def __init__(self) -> None:
            super().__init__(
                body_per_url={
                    "http://reddit.test/user/MarvelsGrantMan136": "partial shell without payload"
                }
            )
            self.ready = False

        async def wait_for_load_state(self, state, *, timeout):
            await super().wait_for_load_state(state, timeout=timeout)
            self.ready = True

        async def text_content(self, selector):
            if self.ready:
                return "profile sidebar with Phase 2c timing probe visible"
            return await super().text_content(selector)

    page = _RedditReadyPage()
    browser = _FakeBrowser(page)

    outcome = await verify_seed_renders(
        browser=browser,
        urls=["/user/MarvelsGrantMan136"],
        site_name="reddit",
        site_url="http://reddit.test",
        signature="Phase 2c timing probe",
    )

    assert outcome.ok is True
    assert page.goto_calls == [("http://reddit.test/user/MarvelsGrantMan136", "commit")]
    assert page.load_state_calls == [("domcontentloaded", 10000)]


@pytest.mark.asyncio
async def test_body_poll_composes_with_markdown_strip():
    # Bug G composition: signature carries raw ** delimiters, body has
    # them stripped by GitLab's CommonMark renderer. Poll must match
    # via _normalize which applies _strip_markdown_for_text_match.
    url = "http://gitlab.test/proj/-/issues/5"
    body = "full body with [CI Pipeline Notification] Pipeline failed on main"
    page = _PollingFakePage(bodies=["", body], final_body=body)
    browser = _PollingFakeBrowser(page)

    outcome = await verify_seed_renders(
        browser=browser,
        urls=[url],
        site_name="gitlab",
        site_url="http://gitlab.test",
        signature="**[CI Pipeline Notification]** Pipeline failed",
    )
    assert outcome.ok is True


# ---------------------------------------------------------------------------
# _wait_for_body_text exponential-backoff poll. Bug J introduced the poll
# with a fixed 500 ms cadence; this follow-up uses 100→200→400→...→2000 ms
# backoff with a 20 s deadline so fast hits resolve quickly AND slow ones
# (GitLab sidekiq + cache-invalidation tail) land before the deadline.
# ---------------------------------------------------------------------------


class _WaitForBodyPage:
    """Fake page that records wait_for_timeout intervals and returns
    scripted body snapshots on successive text_content calls."""

    def __init__(self, bodies: list[str]) -> None:
        self._bodies = list(bodies)
        self.intervals_ms: list[int] = []

    async def text_content(self, selector: str) -> str:
        assert selector == "body"
        if not self._bodies:
            return ""
        return self._bodies.pop(0)

    async def wait_for_timeout(self, ms: int) -> None:
        self.intervals_ms.append(ms)


@pytest.mark.asyncio
async def test_wait_for_body_text_uses_exponential_backoff_schedule():
    from worldsim.phases import phase_2_render_check as rc

    # Supply enough empty bodies that the poll never matches; we are
    # inspecting the backoff schedule, not the match path.
    page = _WaitForBodyPage(bodies=["" for _ in range(20)])
    result = await rc._wait_for_body_text(page, "never-matches", timeout_ms=15000)
    assert result is False
    # Expected schedule: 100, 200, 400, 800, 1600, 2000, 2000, ...
    assert page.intervals_ms[:6] == [100, 200, 400, 800, 1600, 2000]
    # Cap holds after that.
    assert all(ms == 2000 for ms in page.intervals_ms[6:])


@pytest.mark.asyncio
async def test_wait_for_body_text_returns_true_on_fast_match():
    from worldsim.phases import phase_2_render_check as rc

    page = _WaitForBodyPage(bodies=["signature present now"])
    result = await rc._wait_for_body_text(page, "signature present now", timeout_ms=5000)
    assert result is True
    # First poll should hit immediately with no sleep.
    assert page.intervals_ms == []


@pytest.mark.asyncio
async def test_wait_for_body_text_finds_late_signature_before_backoff_cap():
    from worldsim.phases import phase_2_render_check as rc

    # Empty on polls 1-3, signature on poll 4. Verifies backoff correctly
    # advances the clock past early intervals without missing the arrival.
    page = _WaitForBodyPage(bodies=["", "", "", "full body with SEEDSIG inside"])
    result = await rc._wait_for_body_text(page, "SEEDSIG", timeout_ms=5000)
    assert result is True
    # Three sleeps before the match: 100, 200, 400.
    assert page.intervals_ms == [100, 200, 400]


@pytest.mark.asyncio
async def test_wait_for_body_text_empty_needle_returns_false():
    from worldsim.phases import phase_2_render_check as rc

    page = _WaitForBodyPage(bodies=["anything"])
    # An empty needle has no match target; fail-fast without polling.
    result = await rc._wait_for_body_text(page, "", timeout_ms=1000)
    assert result is False
    assert page.intervals_ms == []


def test_body_poll_timeout_constant_is_20s():
    from worldsim.phases import phase_2_render_check as rc

    assert rc._BODY_POLL_TIMEOUT_MS == 20000
    assert rc._BODY_POLL_INITIAL_MS == 100
    assert rc._BODY_POLL_MAX_MS == 2000
