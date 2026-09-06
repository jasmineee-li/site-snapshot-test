"""``probes._run_render_check``: auth resolution, canonical patch, injection."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from warp_taskgen.phase_2.phase_2c import probes
from warp_taskgen.phases.phase_2_render_check import RenderOutcome
from warp_taskgen.sites import default_catalog

from ._fixtures import (
    _bypass_preflight,  # noqa: F401
    _gitlab_instance,
    _stable_git_fingerprint,  # noqa: F401
)


@pytest.mark.asyncio
async def test_run_render_check_passes_resolved_agent_auth(monkeypatch):
    captured: dict[str, Any] = {}

    async def fake_verify_seed_renders(**kwargs):
        captured.update(kwargs)
        return RenderOutcome.passed(
            url="https://gitlab.example/project/-/issues/1",
            signature="auth probe body",
            snippet="auth probe body",
        )

    outcome = await probes._run_render_check(
        site_catalog=default_catalog(),
        strict_site_planning=False,
        browser=object(),
        render_semaphore=None,
        seed={
            "editor_calls": [
                {
                    "site": "gitlab",
                    "method": "create_issue_note",
                    "args": {"note_body": "auth probe body"},
                }
            ]
        },
        metadata={"read_surface_urls": ["https://gitlab.example/project/-/issues/1"]},
        instance=_gitlab_instance(
            agent_auth={
                "type": "http_headers",
                "http_headers": {"headers": {"X-User": "${credentials.username}"}},
                "authentication": {"credentials": {"username": "alice", "password": "pw"}},
            }
        ),
        verify_seed_renders=fake_verify_seed_renders,
    )

    assert outcome.ok
    assert captured["browser_context_kwargs"] == {"extra_http_headers": {"X-User": "alice"}}


@pytest.mark.asyncio
async def test_probes_render_check_uses_canonical_patch(monkeypatch):
    captured: dict[str, Any] = {}

    async def fake_verify_seed_renders(**kwargs):
        captured.update(kwargs)
        return RenderOutcome.passed(
            url="https://gitlab.example/project/-/issues/1",
            signature="canonical body",
            snippet="canonical body",
        )

    monkeypatch.setattr(probes, "verify_seed_renders", fake_verify_seed_renders)

    outcome = await probes._run_render_check(
        site_catalog=default_catalog(),
        strict_site_planning=False,
        browser=object(),
        render_semaphore=None,
        seed={
            "editor_calls": [
                {
                    "site": "gitlab",
                    "method": "create_issue_note",
                    "args": {"note_body": "canonical body"},
                }
            ]
        },
        metadata={"read_surface_urls": ["https://gitlab.example/project/-/issues/1"]},
        instance=_gitlab_instance(),
    )

    assert outcome.ok
    assert captured["signature"] == "canonical body"


@pytest.mark.asyncio
async def test_render_check_uses_injected_verify_seed_renders(monkeypatch):
    captured: dict[str, Any] = {}

    async def fake_verify_seed_renders(**kwargs):
        captured.update(kwargs)
        return RenderOutcome.passed(
            url="https://gitlab.example/project/-/issues/1",
            signature="impl body",
            snippet="impl body",
        )

    outcome = await probes._run_render_check(
        site_catalog=default_catalog(),
        strict_site_planning=False,
        browser=object(),
        render_semaphore=None,
        seed={
            "editor_calls": [
                {
                    "site": "gitlab",
                    "method": "create_issue_note",
                    "args": {"note_body": "impl body"},
                }
            ]
        },
        metadata={"read_surface_urls": ["https://gitlab.example/project/-/issues/1"]},
        instance=_gitlab_instance(),
        verify_seed_renders=fake_verify_seed_renders,
    )

    assert outcome.ok
    assert captured["signature"] == "impl body"


@pytest.mark.asyncio
async def test_concurrent_render_checks_keep_their_injected_probe(monkeypatch):
    calls: list[str] = []

    async def fake_verify_seed_renders(**kwargs):
        calls.append(str(kwargs["signature"]))
        await asyncio.sleep(0)
        return RenderOutcome.passed(
            url="https://gitlab.example/project/-/issues/1",
            signature=str(kwargs["signature"]),
            snippet=str(kwargs["signature"]),
        )

    semaphore = asyncio.Semaphore(1)

    async def call(body: str):
        return await probes._run_render_check(
            site_catalog=default_catalog(),
            strict_site_planning=False,
            browser=object(),
            render_semaphore=semaphore,
            seed={
                "editor_calls": [
                    {
                        "site": "gitlab",
                        "method": "create_issue_note",
                        "args": {"note_body": body},
                    }
                ]
            },
            metadata={"read_surface_urls": ["https://gitlab.example/project/-/issues/1"]},
            instance=_gitlab_instance(),
            verify_seed_renders=fake_verify_seed_renders,
        )

    outcomes = await asyncio.gather(call("first body"), call("second body"))

    assert [outcome.ok for outcome in outcomes] == [True, True]
    assert calls == ["first body", "second body"]
