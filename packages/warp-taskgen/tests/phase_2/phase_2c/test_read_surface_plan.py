from __future__ import annotations

from typing import Any

import pytest

from tests.sites.test_read_surface import FakeReadSurfaceSite
from warp_taskgen.phase_2.phase_2c import probes, verifier
from warp_taskgen.phase_2.phase_2c.policy import default_feasibility_policy_catalog
from warp_taskgen.phase_2.phase_2c.probe_bundle import Phase2cProbeBundle
from warp_taskgen.runtime_composition import (
    DEFAULT_RUNTIME_COMPOSITION,
    RuntimeComposition,
    classifieds_listing_reply_poc,
)
from warp_taskgen.seeding import SeedSiteRegistration, SeedSiteRegistry
from warp_taskgen.sites import ReadbackDecision, ReadbackObservation, SiteCatalog, default_catalog
from warp_taskgen.sites.read_surface import build_read_surface_plan


@pytest.mark.asyncio
async def test_render_check_consumes_injected_site_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeReadbackSite(FakeReadSurfaceSite):
        def interpret_readback(self, observation: ReadbackObservation) -> ReadbackDecision:
            return ReadbackDecision(observation.payload == "fake observation", "fake_decision")

    async def fake_verify_seed_renders(**kwargs: Any):
        captured.update(kwargs)
        return probes.RenderOutcome.passed(
            url=kwargs["urls"][0],
            signature=kwargs["signature"],
            snippet=kwargs["signature"],
        )

    monkeypatch.setattr(probes, "verify_seed_renders", fake_verify_seed_renders)
    outcome = await probes._run_render_check(
        strict_site_planning=True,
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
        site_catalog=SiteCatalog([FakeReadbackSite()]),
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
    decision = captured["readback_site"].interpret_readback(
        ReadbackObservation("resource_identity", {"comment_id": "17"}, "fake observation")
    )
    assert decision == ReadbackDecision(True, "fake_decision")


@pytest.mark.asyncio
async def test_render_check_uses_anonymous_reader_contract_without_writer_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    reader_instances: list[dict[str, Any]] = []

    async def fake_verify_seed_renders(**kwargs: Any):
        captured.update(kwargs)
        return probes.RenderOutcome.passed(
            url=kwargs["urls"][0],
            signature=kwargs["signature"],
            snippet=kwargs["signature"],
        )

    def reader_preflight(instance: dict[str, Any]) -> Any:
        reader_instances.append(instance)
        return type("ReaderResult", (), {"ok": True})()

    monkeypatch.setattr(probes, "verify_seed_renders", fake_verify_seed_renders)
    monkeypatch.setattr(
        probes,
        "_resolve_benign_browser_context_auth",
        lambda _instance: (_ for _ in ()).throw(AssertionError("writer auth reused")),
    )
    outcome = await probes._run_render_check(
        strict_site_planning=True,
        browser=object(),
        render_semaphore=None,
        seed={
            "editor_calls": [
                {
                    "site": "fake",
                    "method": "create_message",
                    "args": {"body": "anonymous reader body"},
                }
            ]
        },
        metadata={"read_surface_urls": ["/messages/17"]},
        instance={
            "site_name": "fake",
            "site_url": "https://fake.local",
            "benchmark": "webarena_verified",
        },
        site_catalog=SiteCatalog([FakeReadSurfaceSite()]),
        reader_preflight=reader_preflight,
    )

    assert outcome.ok
    assert reader_instances and reader_instances[0]["site_name"] == "fake"
    assert captured["browser_context_kwargs"] == {}


@pytest.mark.asyncio
async def test_classifieds_composition_reader_gate_ignores_failing_writer_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The named POC reaches render only through its anonymous reader gate."""

    composition = classifieds_listing_reply_poc()
    captured: dict[str, Any] = {}

    async def fake_verify_seed_renders(**kwargs: Any):
        captured.update(kwargs)
        return probes.RenderOutcome.passed(
            url=kwargs["urls"][0],
            signature=kwargs["signature"],
            snippet=kwargs["signature"],
        )

    monkeypatch.setattr(probes, "verify_seed_renders", fake_verify_seed_renders)
    monkeypatch.setattr(
        probes,
        "_resolve_benign_browser_context_auth",
        lambda _instance: (_ for _ in ()).throw(AssertionError("writer auth reused")),
    )
    outcome = await probes._run_render_check(
        strict_site_planning=True,
        browser=object(),
        render_semaphore=None,
        seed={
            "editor_calls": [
                {
                    "site": "classifieds",
                    "method": "create_listing_reply",
                    "args": {"body": "anonymous reader body"},
                }
            ]
        },
        metadata={
            "read_surface_urls": ["/index.php?page=item&id=17"],
            "write_tokens": {
                "listing_id": "17",
                "reply_id": "19",
                "actor_name": "alice",
                "reply_body_sha256": "body-hash",
            },
            "editor_call_results": [
                {
                    "call_index": 0,
                    "editor_method": "create_listing_reply",
                    "read_surface_urls": ["/index.php?page=item&id=17"],
                    "read_surface_provenance_source": "classifieds.regular_participant",
                }
            ],
        },
        instance={
            "site_name": "classifieds",
            "site_url": "https://classifieds.local",
            "benchmark": "visualwebarena",
            "reader_auth": {"type": "none"},
            "agent_auth": {
                "type": "storage_state",
                "storage_state": {"path": "/deliberately/missing-writer-state.json"},
            },
        },
        site_catalog=composition.site_catalog,
        reader_preflight=composition.reader_preflight,
    )

    assert outcome.ok
    assert captured["browser_context_kwargs"] == {}


@pytest.mark.asyncio
async def test_classifieds_composition_missing_reader_auth_fails_before_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    composition = classifieds_listing_reply_poc()
    render_called = False

    async def should_not_render(**kwargs: Any):
        nonlocal render_called
        render_called = True
        raise AssertionError(f"render ran without reader auth: {kwargs!r}")

    monkeypatch.setattr(probes, "verify_seed_renders", should_not_render)
    monkeypatch.setattr(
        probes,
        "_resolve_benign_browser_context_auth",
        lambda _instance: (_ for _ in ()).throw(AssertionError("writer auth reused")),
    )
    outcome = await probes._run_render_check(
        strict_site_planning=True,
        browser=object(),
        render_semaphore=None,
        seed={
            "editor_calls": [
                {
                    "site": "classifieds",
                    "method": "create_listing_reply",
                    "args": {"body": "anonymous reader body"},
                }
            ]
        },
        metadata={
            "read_surface_urls": ["/index.php?page=item&id=17"],
            "write_tokens": {
                "listing_id": "17",
                "reply_id": "19",
                "actor_name": "alice",
                "reply_body_sha256": "body-hash",
            },
            "editor_call_results": [
                {
                    "call_index": 0,
                    "editor_method": "create_listing_reply",
                    "read_surface_urls": ["/index.php?page=item&id=17"],
                    "read_surface_provenance_source": "classifieds.regular_participant",
                }
            ],
        },
        instance={
            "site_name": "classifieds",
            "site_url": "https://classifieds.local",
            "benchmark": "visualwebarena",
            "agent_auth": {
                "type": "storage_state",
                "storage_state": {"path": "/deliberately/missing-writer-state.json"},
            },
        },
        site_catalog=composition.site_catalog,
        reader_preflight=composition.reader_preflight,
    )

    assert not outcome.ok
    assert outcome.kind == "auth_missing"
    assert not render_called


@pytest.mark.asyncio
async def test_classifieds_composition_missing_reader_auth_skips_seed_and_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    composition = classifieds_listing_reply_poc()
    seed_called = False
    render_called = False

    async def should_not_seed(*args: Any, **kwargs: Any):
        nonlocal seed_called
        seed_called = True
        raise AssertionError(f"seed ran without reader auth: {args!r} {kwargs!r}")

    async def should_not_render(*args: Any, **kwargs: Any):
        nonlocal render_called
        render_called = True
        raise AssertionError(f"render ran without reader auth: {args!r} {kwargs!r}")

    bundle = Phase2cProbeBundle.default(apply_seed=should_not_seed, render_check=should_not_render)
    result = await verifier._verify_one(
        {
            "id": "classifieds-reader-auth-missing",
            "site": "classifieds",
            "benchmark": "visualwebarena",
            "adversarial_data_seed": {
                "editor_calls": [
                    {
                        "site": "classifieds",
                        "method": "create_listing_reply",
                        "args": {"body": "anonymous reader body"},
                    }
                ]
            },
        },
        {
            "site_name": "classifieds",
            "site_url": "https://classifieds.local",
            "benchmark": "visualwebarena",
            "agent_auth": {
                "type": "storage_state",
                "storage_state": {"path": "/deliberately/missing-writer-state.json"},
            },
        },
        retry_count=0,
        fingerprint_base={"host_config": "test", "editor_commit": "test"},
        ttl_hours=None,
        force_reverify=True,
        cleanup_warnings=[],
        browser=object(),
        runtime_composition=composition,
        probes=bundle,
    )

    assert result["feasibility"]["status"] == "infeasible"
    assert result["feasibility"]["errors"][0]["kind"] == "auth_missing"
    assert not seed_called
    assert not render_called


@pytest.mark.asyncio
async def test_classifieds_composition_cleanup_failure_invalidates_phase2c_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from warp_taskgen.editors.base import EditorError

    composition = classifieds_listing_reply_poc()

    class FailingCleanup:
        def cleanup(self) -> None:
            raise EditorError("cleanup_failed", "reply id remains visible")

    async def fake_seed(*args: Any, **kwargs: Any):
        return FailingCleanup(), {
            "read_surface_urls": ["/index.php?page=item&id=17"],
            "write_tokens": {
                "listing_id": "17",
                "reply_id": "19",
                "actor_name": "alice",
                "reply_body_sha256": "a" * 64,
            },
        }

    async def fake_render(**kwargs: Any):
        return probes.RenderOutcome.passed(
            url="https://classifieds.local/index.php?page=item&id=17",
            signature="canary",
            snippet="canary",
        )

    async def fake_reachability(**kwargs: Any):
        return None

    bundle = Phase2cProbeBundle.default(
        apply_seed=fake_seed,
        render_check=fake_render,
        reachability_check=fake_reachability,
    )

    warnings: list[str] = []
    with pytest.raises(RuntimeError, match="cleanup_failed"):
        await verifier._verify_one(
            {
                "id": "classifieds-cleanup-failure",
                "site": "classifieds",
                "benchmark": "visualwebarena",
                "adversarial_data_seed": {
                    "editor_calls": [
                        {
                            "site": "classifieds",
                            "method": "create_listing_reply",
                            "args": {"listing_id": "17", "body": "canary"},
                        }
                    ]
                },
            },
            {
                "site_name": "classifieds",
                "site_url": "https://classifieds.local",
                "benchmark": "visualwebarena",
                "reader_auth": {"type": "none"},
            },
            retry_count=0,
            fingerprint_base={"host_config": "test", "editor_commit": "test"},
            ttl_hours=None,
            force_reverify=True,
            cleanup_warnings=warnings,
            browser=object(),
            runtime_composition=composition,
            probes=bundle,
        )
    assert warnings and "cleanup_failed" in warnings[0]


@pytest.mark.asyncio
async def test_render_check_preserves_generic_identity_tokens_in_bound_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class GenericIdentitySite(FakeReadSurfaceSite):
        def build_read_surface_plan(
            self,
            *,
            seed_result: Any,
            signature: str,
            origin: str,
        ):
            return build_read_surface_plan(
                site=self.site,
                seed_result=seed_result,
                signature=signature,
                origin=origin,
                identity_keys=("listing_id", "reply_id", "actor_name"),
            )

    async def fake_verify_seed_renders(**kwargs: Any):
        captured.update(kwargs)
        return probes.RenderOutcome.passed(
            url=kwargs["urls"][0],
            signature=kwargs["signature"],
            snippet=kwargs["signature"],
        )

    monkeypatch.setattr(probes, "verify_seed_renders", fake_verify_seed_renders)
    outcome = await probes._run_render_check(
        strict_site_planning=True,
        browser=object(),
        render_semaphore=None,
        seed={
            "editor_calls": [
                {
                    "site": "fake",
                    "method": "create_message",
                    "args": {"body": "generic identity payload"},
                }
            ]
        },
        metadata={
            "read_surface_urls": ["/messages/17"],
            "write_tokens": {
                "listing_id": "17",
                "reply_id": "88",
                "actor_name": "alice",
            },
        },
        instance={
            "site_name": "fake",
            "site_url": "https://fake.local",
            "benchmark": "webarena_verified",
        },
        site_catalog=SiteCatalog([GenericIdentitySite()]),
    )

    assert outcome.ok
    assert captured["write_tokens"] == {
        "listing_id": "17",
        "reply_id": "88",
        "actor_name": "alice",
    }
    assert captured["readback_plan"].identity_tokens == captured["write_tokens"]
    assert captured["readback_plan"].persist_readback_identity_tokens is False


@pytest.mark.asyncio
async def test_render_check_fails_before_browser_for_foreign_only_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def should_not_run(**kwargs: Any):
        raise AssertionError(f"browser verifier received unsafe evidence: {kwargs!r}")

    monkeypatch.setattr(probes, "verify_seed_renders", should_not_run)
    outcome = await probes._run_render_check(
        strict_site_planning=True,
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
        site_catalog=default_catalog(),
        strict_site_planning=False,
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
        site_catalog=default_catalog(),
        strict_site_planning=False,
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
        site_catalog=default_catalog(),
        strict_site_planning=False,
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
        return probes.RenderOutcome.passed(
            url="https://fake.local/messages/17",
            signature="fake payload body",
            snippet="fake payload body",
        )

    bundle = Phase2cProbeBundle.default(apply_seed=fake_apply, render_check=fake_render_check)

    result = await verifier._verify_one(
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
        runtime_composition=RuntimeComposition(
            name=DEFAULT_RUNTIME_COMPOSITION,
            site_catalog=site_catalog,
            seed_registry=seed_registry,
            feasibility_policy_catalog=default_feasibility_policy_catalog(),
            seed_token_scope="method",
            strict_site_planning=True,
        ),
        probes=bundle,
    )

    assert result["feasibility"]["status"] == "verified"
    assert captured == {
        "seed_registry": seed_registry,
        "site_catalog": site_catalog,
    }


# Planning strictness is a Runtime Composition property (#276). The default
# composition keeps the historical fall-through for the three cases where a
# read surface cannot be planned; a named POC composition fails closed on the
# same three. These pin behaviour that was previously implied by whether a
# caller happened to pass a Site catalog.

_FALL_THROUGH_CASES = {
    "benchmark_missing_and_site_not_in_catalog": (
        {
            "editor_calls": [
                {"site": "fake", "method": "create_message", "args": {"body": "unplanned body"}}
            ]
        },
        {"read_surface_urls": ["https://fake.local/messages/17"]},
        {"site_name": "fake", "site_url": "https://fake.local"},
    ),
    "blank_signature": (
        {
            "editor_calls": [
                {"site": "gitlab", "method": "create_issue_note", "args": {"note_body": ""}}
            ]
        },
        {"read_surface_urls": ["https://gitlab.local/p/-/issues/1"]},
        {
            "site_name": "gitlab",
            "site_url": "https://gitlab.local",
            "benchmark": "webarena_verified",
        },
    ),
    "targeting_error_for_site_not_in_catalog": (
        {
            "editor_calls": [
                {
                    "site": "shopping",
                    "method": "create_product_review",
                    "args": {"detail": "unplanned review body"},
                }
            ]
        },
        {"read_surface_urls": ["https://shopping.local/review/1"]},
        {
            "site_name": "shopping",
            "site_url": "https://shopping.local",
            "benchmark": "webarena_verified",
        },
    ),
}


async def _render_check_for_case(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    *,
    site_catalog: Any,
    strict_site_planning: bool,
) -> tuple[Any, list[dict[str, Any]]]:
    seed, metadata, instance = _FALL_THROUGH_CASES[case]
    probe_calls: list[dict[str, Any]] = []

    async def fake_verify_seed_renders(**kwargs: Any):
        probe_calls.append(kwargs)
        return probes.RenderOutcome.passed(
            url=kwargs["urls"][0],
            signature="observed",
            snippet="observed",
        )

    monkeypatch.setattr(probes, "verify_seed_renders", fake_verify_seed_renders)
    outcome = await probes._run_render_check(
        browser=object(),
        render_semaphore=None,
        seed=seed,
        metadata=metadata,
        instance=instance,
        site_catalog=site_catalog,
        strict_site_planning=strict_site_planning,
    )
    return outcome, probe_calls


@pytest.mark.asyncio
@pytest.mark.parametrize("case", sorted(_FALL_THROUGH_CASES))
async def test_default_composition_falls_through_unplannable_read_surface(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    composition = RuntimeComposition.default()
    assert composition.strict_site_planning is False

    outcome, probe_calls = await _render_check_for_case(
        monkeypatch,
        case,
        site_catalog=composition.site_catalog,
        strict_site_planning=composition.strict_site_planning,
    )

    assert outcome.kind != "render_unverified"
    assert outcome.ok is True
    assert len(probe_calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("case", sorted(_FALL_THROUGH_CASES))
async def test_poc_composition_fails_closed_on_unplannable_read_surface(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    composition = classifieds_listing_reply_poc()
    assert composition.strict_site_planning is True

    outcome, probe_calls = await _render_check_for_case(
        monkeypatch,
        case,
        site_catalog=composition.site_catalog,
        strict_site_planning=composition.strict_site_planning,
    )

    assert outcome.ok is False
    assert outcome.kind == "render_unverified"
    assert probe_calls == []
