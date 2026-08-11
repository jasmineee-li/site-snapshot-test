from __future__ import annotations

from typing import Any

import pytest

from tests.sites.test_read_surface import FakeReadSurfaceSite
from worldsim.phase_2.phase_2c import _impl as phase_2c_impl
from worldsim.phase_2.phase_2c import probes
from worldsim.seeding import SeedSiteRegistration, SeedSiteRegistry
from worldsim.sites import SiteCatalog


@pytest.mark.asyncio
async def test_render_check_consumes_injected_site_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_verify_seed_renders(**kwargs: Any):
        captured.update(kwargs)
        return probes.RenderOutcome.passed(
            url=kwargs["urls"][0],
            signature=kwargs["signature"],
            snippet=kwargs["signature"],
        )

    monkeypatch.setattr(probes, "verify_seed_renders", fake_verify_seed_renders)
    outcome = await probes._run_render_check(
        browser=object(),
        render_semaphore=None,
        seed={
            "editor_calls": [
                {
                    "site": "fake",
                    "method": "create_message",
                    "args": {"body": "fake payload body"},
                }
            ]
        },
        metadata={
            "read_surface_urls": [
                "https://attacker.invalid/messages/17",
                "/messages/17",
            ],
            "comment_id": "17",
            "editor_call_results": [
                {
                    "call_index": 0,
                    "editor_method": "create_message",
                    "read_surface_urls": [
                        "https://attacker.invalid/messages/17",
                        "/messages/17",
                    ],
                    "write_tokens": {"comment_id": "17"},
                    "read_surface_provenance_source": "editor_api_response",
                }
            ],
        },
        instance={
            "site_name": "fake",
            "site_url": "https://fake.local",
            "benchmark": "webarena_verified",
        },
        site_catalog=SiteCatalog([FakeReadSurfaceSite()]),
    )

    assert outcome.ok
    assert captured["urls"] == ["https://fake.local/messages/17"]
    assert captured["signature"] == "fake payload body"
    assert captured["write_tokens"] == {"comment_id": "17"}
    assert captured["diagnostics"]["read_surface_plan"] == {
        "site": "fake",
        "verification_mode": "seed_resource",
        "provenance_source": "editor_api_response",
    }


@pytest.mark.asyncio
async def test_render_check_fails_before_browser_for_foreign_only_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def should_not_run(**kwargs: Any):
        raise AssertionError(f"browser verifier received unsafe evidence: {kwargs!r}")

    monkeypatch.setattr(probes, "verify_seed_renders", should_not_run)
    outcome = await probes._run_render_check(
        browser=object(),
        render_semaphore=None,
        seed={
            "editor_calls": [
                {
                    "site": "fake",
                    "method": "create_message",
                    "args": {"body": "fake payload body"},
                }
            ]
        },
        metadata={"read_surface_urls": ["https://attacker.invalid/messages/17"]},
        instance={
            "site_name": "fake",
            "site_url": "https://fake.local",
            "benchmark": "webarena_verified",
        },
        site_catalog=SiteCatalog([FakeReadSurfaceSite()]),
    )

    assert not outcome.ok
    assert outcome.kind == "render_unverified"
    assert "foreign_read_surface" in outcome.detail


@pytest.mark.asyncio
async def test_active_site_missing_benchmark_fails_before_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def should_not_run(**kwargs: Any):
        raise AssertionError(f"browser verifier received unbound evidence: {kwargs!r}")

    monkeypatch.setattr(probes, "verify_seed_renders", should_not_run)
    outcome = await probes._run_render_check(
        browser=object(),
        render_semaphore=None,
        seed={
            "editor_calls": [
                {
                    "site": "gitlab",
                    "method": "create_issue_note",
                    "args": {"note_body": "payload body unique"},
                }
            ]
        },
        metadata={"read_surface_urls": ["https://attacker.invalid/issues/17"]},
        instance={
            "site_name": "gitlab",
            "site_url": "https://gitlab.local",
        },
    )

    assert not outcome.ok
    assert outcome.kind == "render_unverified"
    assert "benchmark metadata" in outcome.detail


@pytest.mark.asyncio
async def test_payload_call_cannot_borrow_setup_call_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def should_not_run(**kwargs: Any):
        raise AssertionError(f"browser verifier received cross-call evidence: {kwargs!r}")

    monkeypatch.setattr(probes, "verify_seed_renders", should_not_run)
    outcome = await probes._run_render_check(
        browser=object(),
        render_semaphore=None,
        seed={
            "editor_calls": [
                {
                    "site": "gitlab",
                    "method": "create_issue",
                    "args": {"issue_title": "setup title"},
                },
                {
                    "site": "gitlab",
                    "method": "create_issue_note",
                    "args": {"note_body": "payload body unique"},
                },
            ]
        },
        metadata={
            "read_surface_urls": ["https://gitlab.local/project/-/issues/1"],
            "editor_call_results": [
                {
                    "call_index": 0,
                    "editor_method": "create_issue",
                    "read_surface_urls": ["/project/-/issues/1"],
                },
                {
                    "call_index": 1,
                    "editor_method": "create_issue_note",
                    "read_surface_urls": [],
                },
            ],
        },
        instance={
            "site_name": "gitlab",
            "site_url": "https://gitlab.local",
            "benchmark": "webarena_verified",
        },
    )

    assert not outcome.ok
    assert outcome.kind == "render_unverified"
    assert "payload editor call" in outcome.detail


@pytest.mark.asyncio
async def test_multicall_payload_requires_per_call_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def should_not_run(**kwargs: Any):
        raise AssertionError(f"browser verifier received aggregate evidence: {kwargs!r}")

    monkeypatch.setattr(probes, "verify_seed_renders", should_not_run)
    outcome = await probes._run_render_check(
        browser=object(),
        render_semaphore=None,
        seed={
            "editor_calls": [
                {
                    "site": "gitlab",
                    "method": "create_issue",
                    "args": {"issue_title": "setup title"},
                },
                {
                    "site": "gitlab",
                    "method": "create_issue_note",
                    "args": {"note_body": "payload body unique"},
                },
            ]
        },
        metadata={"read_surface_urls": ["https://gitlab.local/project/-/issues/1"]},
        instance={
            "site_name": "gitlab",
            "site_url": "https://gitlab.local",
            "benchmark": "webarena_verified",
        },
    )

    assert not outcome.ok
    assert outcome.kind == "render_unverified"
    assert "payload editor call" in outcome.detail


@pytest.mark.asyncio
async def test_verify_one_threads_per_run_seed_and_site_catalogs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeEditor:
        pass

    class FakeHandle:
        def cleanup(self) -> None:
            return None

    seed_registry = SeedSiteRegistry.from_registrations(
        [SeedSiteRegistration("webarena_verified", "fake", FakeEditor)]
    )
    site_catalog = SiteCatalog([FakeReadSurfaceSite()])
    captured: dict[str, Any] = {}

    async def fake_apply(seed: dict[str, Any], instance: dict[str, Any], **kwargs: Any):
        del seed, instance
        captured["seed_registry"] = kwargs.get("seed_registry")
        return FakeHandle(), {
            "read_surface_urls": ["/messages/17"],
            "comment_id": "17",
        }

    async def fake_render_check(**kwargs: Any):
        captured["site_catalog"] = kwargs.get("site_catalog")
        return phase_2c_impl.RenderOutcome.passed(
            url="https://fake.local/messages/17",
            signature="fake payload body",
            snippet="fake payload body",
        )

    monkeypatch.setattr(phase_2c_impl, "apply_data_seed_async", fake_apply)
    monkeypatch.setattr(phase_2c_impl, "_run_render_check", fake_render_check)

    result = await phase_2c_impl._verify_one(
        {
            "id": "fake-task",
            "site": "fake",
            "benchmark": "webarena_verified",
            "adversarial_data_seed": {
                "editor_calls": [
                    {
                        "site": "fake",
                        "method": "create_message",
                        "args": {"body": "fake payload body"},
                    }
                ]
            },
        },
        {
            "site_name": "fake",
            "site_url": "https://fake.local",
            "benchmark": "webarena_verified",
        },
        retry_count=0,
        fingerprint_base={"host_config": "test", "editor_commit": "test"},
        ttl_hours=None,
        force_reverify=True,
        cleanup_warnings=[],
        browser=object(),
        seed_registry=seed_registry,
        site_catalog=site_catalog,
    )

    assert result["feasibility"]["status"] == "verified"
    assert captured == {
        "seed_registry": seed_registry,
        "site_catalog": site_catalog,
    }
