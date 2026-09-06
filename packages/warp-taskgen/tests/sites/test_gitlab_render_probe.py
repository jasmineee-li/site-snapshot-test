"""Positive coverage for the GitLab Site render probe at its new seam.

Mirrors ``tests/test_phase_2_render_check.py::
test_ryw_fastpath_matches_note_id_in_discussions_json`` one layer down:
the read-your-write fast path is now Site-owned, so it must prove the
note_id readback on its own, without going through
``verify_seed_renders``.
"""

from __future__ import annotations

import pytest

from warp_taskgen.sites.gitlab_render_probe import GitLabRenderProbe


class _FakeResponse:
    def __init__(self, *, status: int, body: str) -> None:
        self.status = status
        self._body = body

    async def text(self) -> str:
        return self._body

    async def json(self):
        import json

        return json.loads(self._body)


class _FakeRequestContext:
    def __init__(self, responses: dict[str, _FakeResponse]) -> None:
        self._responses = dict(responses)
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def get(self, url, **kwargs):
        self.calls.append((url, dict(kwargs)))
        return self._responses.get(url, _FakeResponse(status=404, body="not found"))


class _FakeRequestPage:
    def __init__(self, responses: dict[str, _FakeResponse]) -> None:
        self.request = _FakeRequestContext(responses)


class _WaitPage:
    def __init__(self, body: str) -> None:
        self._body = body
        self.selector_calls: list[tuple[str, int]] = []
        self.text_content_calls = 0

    async def wait_for_selector(self, selector, *, timeout):
        self.selector_calls.append((selector, timeout))

    async def text_content(self, selector):
        self.text_content_calls += 1
        return self._body

    async def wait_for_timeout(self, ms):
        return None


@pytest.mark.asyncio
async def test_note_ryw_matches_note_id():
    url = "http://gitlab.test/proj/-/issues/5"
    json_url = url + "/discussions.json"
    page = _FakeRequestPage(
        {
            json_url: _FakeResponse(
                status=200,
                body='[{"id":"abc","notes":[{"id":42,"body":"Raising priority..."}]}]',
            )
        }
    )

    outcome = await GitLabRenderProbe().read_your_write(
        page=page,
        target_url=url,
        site_name="gitlab",
        signature="Raising priority on issue",
        write_tokens={"note_id": 42},
        timeout_ms=10000,
    )

    assert outcome is not None
    assert outcome.ok is True
    assert outcome.matched_url == json_url
    assert outcome.matched_signature == "note_id=42"
    # Exactly one read-your-write fetch — no spurious retries.
    assert [call[0] for call in page.request.calls] == [json_url]
    assert page.request.calls[0][1]["max_redirects"] == 0


@pytest.mark.asyncio
async def test_note_ryw_skips_when_note_id_absent():
    page = _FakeRequestPage({})

    outcome = await GitLabRenderProbe().read_your_write(
        page=page,
        target_url="http://gitlab.test/proj/-/issues/5",
        site_name="gitlab",
        signature="Raising priority on issue",
        write_tokens={},
        timeout_ms=10000,
    )

    assert outcome is None
    assert page.request.calls == []


@pytest.mark.asyncio
async def test_wait_for_render_polls_only_on_issuable_surfaces():
    probe = GitLabRenderProbe()

    issuable = _WaitPage("shell with SEEDSIG rendered")
    await probe.wait_for_render(
        issuable,
        target_url="http://gitlab.test/proj/-/issues/5",
        signature="SEEDSIG",
        selector_timeout_ms=10000,
        body_poll_timeout_ms=1,
    )
    assert issuable.selector_calls == [
        (".notes .note, .discussion-notes .note, ul.notes-list .note", 10000)
    ]
    assert issuable.text_content_calls == 1

    dashboard = _WaitPage("shell with SEEDSIG rendered")
    await probe.wait_for_render(
        dashboard,
        target_url="http://gitlab.test/proj/-/wikis/home",
        signature="SEEDSIG",
        selector_timeout_ms=10000,
        body_poll_timeout_ms=1,
    )
    assert dashboard.selector_calls == []
    assert dashboard.text_content_calls == 0
