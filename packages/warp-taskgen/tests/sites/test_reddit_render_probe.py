"""Positive coverage for the Reddit Site render probe at its new seam.

Mirrors ``tests/test_phase_2_render_check.py::
test_verify_reddit_comment_passes_with_seeded_first_visible_comment``
one layer down: the exact-comment entry-viewport evidence is now
Site-owned, so it must be provable without ``verify_seed_renders``.
"""

from __future__ import annotations

import pytest

from warp_taskgen.sites.reddit_render_probe import RedditRenderProbe


class _FakeEvaluatePage:
    def __init__(self, result: dict | None) -> None:
        self._result = result
        self.evaluate_args: list[object] = []
        self.load_state_calls: list[tuple[str, int]] = []

    async def evaluate(self, script, arg=None):
        self.evaluate_args.append(arg)
        return self._result

    async def wait_for_load_state(self, state, *, timeout):
        self.load_state_calls.append((state, timeout))


@pytest.mark.asyncio
async def test_seed_comment_first_visible():
    page = _FakeEvaluatePage(
        {
            "ok": True,
            "reason": "seed_comment_first_visible_and_painted",
            "comment_id": "901",
            "first_visible_comment_id": "901",
            "match_found_in_comment": True,
            "visible_at_entry": True,
            "visible_nonspace_chars": 31,
            "requires_expand": False,
        }
    )
    probe = RedditRenderProbe()

    comment_id = probe.exact_visibility_comment_id({"comment_id": "901"})
    assert comment_id == "901"

    evidence = await probe.exact_visibility_probe(
        page,
        comment_id=comment_id,
        normalized_needle="https://attacker.invalid/seeded",
    )

    assert evidence is not None
    assert evidence["ok"] is True
    assert evidence["first_visible_comment_id"] == "901"
    assert page.evaluate_args == [{"commentId": "901", "needle": "https://attacker.invalid/seeded"}]


@pytest.mark.asyncio
async def test_exact_visibility_probe_requires_a_comment_id_and_needle():
    probe = RedditRenderProbe()
    page = _FakeEvaluatePage({"ok": True})

    assert probe.exact_visibility_comment_id({}) is None
    assert probe.exact_visibility_comment_id(None) is None
    assert await probe.exact_visibility_probe(page, comment_id="", normalized_needle="x") is None
    assert await probe.exact_visibility_probe(page, comment_id="901", normalized_needle="") is None
    assert page.evaluate_args == []


@pytest.mark.asyncio
async def test_wait_for_render_waits_for_postmill_dom_readiness():
    page = _FakeEvaluatePage(None)

    await RedditRenderProbe().wait_for_render(
        page,
        target_url="http://reddit.test/f/books/42",
        signature="SEEDSIG",
        selector_timeout_ms=10000,
        body_poll_timeout_ms=20000,
    )

    assert page.load_state_calls == [("domcontentloaded", 10000)]
