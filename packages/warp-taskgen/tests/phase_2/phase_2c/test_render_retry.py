"""``render_unverified`` retry-once behaviour after a short breather."""

from __future__ import annotations

import asyncio

from warp_taskgen.phase_2.phase_2c import probes
from warp_taskgen.phase_2.phase_2c import runner as feas

from ._fixtures import (
    _STUB_SEED_REGISTRY,
    _bypass_preflight,  # noqa: F401
    _FakePlaywrightBrowser,
    _FakePlaywrightPage,
    _install_fake_playwright,
    _metadata_bundle,
    _shopping_instance,
    _shopping_review_task,
    _stable_git_fingerprint,  # noqa: F401
    _write_tasks,
)

# ---------------------------------------------------------------------------
# render_unverified retry-once. A single 3-second breather between the
# first miss and the second attempt covers GitLab's slow sidekiq +
# page-cache invalidation tail without ballooning Phase 2c wall time.
# ---------------------------------------------------------------------------


class _GotoCyclingPage(_FakePlaywrightPage):
    """Fake page whose body advances across successive goto cycles.

    First call to goto → body is ``bodies[0]``; second goto → ``bodies[1]``,
    etc. Lets a test simulate ``first render_check misses, retry renders
    cleanly`` without standing up two separate browsers.
    """

    def __init__(self, bodies: list[str]) -> None:
        super().__init__(body=bodies[0] if bodies else "")
        self._cycle = 0
        self._bodies = list(bodies)

    async def goto(self, url, *, timeout, wait_until):
        self.body = self._bodies[min(self._cycle, len(self._bodies) - 1)]
        self._cycle += 1
        return None


def test_render_unverified_retries_once_after_short_sleep(tmp_path, monkeypatch):
    """First render_check sees a bare shell (signature absent); after the
    3 s breather the second render_check sees the hydrated body with the
    signature. Task lands as verified, not infeasible."""
    monkeypatch.delenv("WORLDSIM_PHASE_2C_SKIP_RENDER_CHECK", raising=False)

    # First goto: shell without signature. Second goto: post-hydration
    # with signature embedded.
    page = _GotoCyclingPage(
        bodies=[
            "issue shell only — no notes rendered yet",
            "issue body now includes seeded SeedNickAdv003 note",
        ]
    )
    fake_browser = _FakePlaywrightBrowser(page)
    _install_fake_playwright(monkeypatch, fake_browser)

    # Zero out the retry sleep so the test is fast but the code path
    # still exercises the gate.
    sleep_calls: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    bundle = _metadata_bundle(
        ["http://shop.example/catalog/product/view/id/67"], retry_sleep=_fake_sleep
    )
    tasks_path = _write_tasks(tmp_path, [_shopping_review_task()])

    report = asyncio.run(
        feas.verify_feasibility(
            tasks_path,
            probes=bundle,
            seed_registry=_STUB_SEED_REGISTRY,
            instances=[_shopping_instance(site_url="http://shop.example/")],
            concurrency=1,
            retry_count=0,
        )
    )
    assert len(report.verified) == 1, report.infeasible
    # The retry breather must have fired exactly once, at the canonical
    # delay. Multiple sleeps indicate over-retrying.
    assert sleep_calls == [probes._RENDER_UNVERIFIED_RETRY_DELAY_S]


def test_render_unverified_retry_respects_final_miss(tmp_path, monkeypatch):
    """If the second render_check still misses, the task is infeasible
    with render_unverified — retry doesn't silently convert a miss."""
    monkeypatch.delenv("WORLDSIM_PHASE_2C_SKIP_RENDER_CHECK", raising=False)

    page = _GotoCyclingPage(
        bodies=[
            "shell — no signature",
            "still no signature on retry either",
        ]
    )
    fake_browser = _FakePlaywrightBrowser(page)
    _install_fake_playwright(monkeypatch, fake_browser)

    async def _fake_sleep(seconds: float) -> None:
        return None

    bundle = _metadata_bundle(
        ["http://shop.example/catalog/product/view/id/67"], retry_sleep=_fake_sleep
    )
    tasks_path = _write_tasks(tmp_path, [_shopping_review_task()])

    report = asyncio.run(
        feas.verify_feasibility(
            tasks_path,
            probes=bundle,
            seed_registry=_STUB_SEED_REGISTRY,
            instances=[_shopping_instance(site_url="http://shop.example/")],
            concurrency=1,
            retry_count=0,
        )
    )
    assert len(report.infeasible) == 1
    error = report.infeasible[0]["feasibility"]["errors"][0]
    assert error["kind"] == "render_unverified"
