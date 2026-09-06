"""Render-check wiring inside the runner: browser launch, stamps, layout probe."""

from __future__ import annotations

import asyncio

import pytest

from warp_taskgen.phase_2.phase_2c import (
    probes,
    verifier,
)
from warp_taskgen.phase_2.phase_2c import runner as feas
from warp_taskgen.phases.phase_2_reachability import ReachabilityOutcome
from warp_taskgen.phases.phase_2_render_check import RenderOutcome
from warp_taskgen.runtime_composition import RuntimeComposition

from ._fixtures import (
    _STUB_SEED_REGISTRY,
    _bundle,
    _bypass_preflight,  # noqa: F401
    _FakeHandle,
    _FakePlaywrightBrowser,
    _FakePlaywrightPage,
    _gitlab_instance,
    _install_fake_playwright,
    _metadata_bundle,
    _seed_bundle,
    _shopping_instance,
    _shopping_review_task,
    _stable_git_fingerprint,  # noqa: F401
    _task,
    _write_tasks,
)


def test_playwright_browser_missing_fails_fast_before_workers(tmp_path, monkeypatch):
    monkeypatch.delenv("WORLDSIM_PHASE_2C_SKIP_RENDER_CHECK", raising=False)
    fake_browser = _FakePlaywrightBrowser(_FakePlaywrightPage(body="unused"))
    _install_fake_playwright(
        monkeypatch,
        fake_browser,
        executable_path=str(tmp_path / "missing-chromium"),
    )

    async def should_not_seed(seed, instance, **kwargs):
        raise AssertionError("missing browser bundle should fail before worker fan-out")

    bundle = _bundle(apply_seed=should_not_seed)
    tasks_path = _write_tasks(tmp_path, [_shopping_review_task()])

    with pytest.raises(RuntimeError, match="playwright install chromium"):
        asyncio.run(
            feas.verify_feasibility(
                tasks_path,
                probes=bundle,
                seed_registry=_STUB_SEED_REGISTRY,
                instances=[_shopping_instance(site_url="http://shop.example/")],
                concurrency=1,
                retry_count=0,
            )
        )


def test_render_check_passing_stamps_render_verified(tmp_path, monkeypatch):
    """When the env var is unset and the seeded payload's signature appears
    in the rendered DOM, feasibility.status='verified' carries
    render_verified=True and render_evidence with the matched URL."""
    monkeypatch.delenv("WORLDSIM_PHASE_2C_SKIP_RENDER_CHECK", raising=False)

    fake_browser = _FakePlaywrightBrowser(
        _FakePlaywrightPage(body="storefront page mentions SeedNickAdv003 here")
    )
    _install_fake_playwright(monkeypatch, fake_browser)

    bundle = _metadata_bundle(["http://shop.example/catalog/product/view/id/67"])
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
    assert len(report.verified) == 1
    feasibility = report.verified[0]["feasibility"]
    assert feasibility["status"] == "verified"
    assert feasibility["render_verified"] is True
    evidence = feasibility["render_evidence"]
    assert evidence["ok"] is True
    assert "SeedNickAdv003" == evidence["matched_signature"]
    assert report.verified[0]["read_surface_urls"] == [
        "http://shop.example/catalog/product/view/id/67"
    ]
    assert fake_browser.closed is True


def test_verified_exposure_records_layout_probe_fields(monkeypatch):
    handle = _FakeHandle()

    async def fake_apply(seed, instance, **kwargs):
        return handle, {"read_surface_urls": ["https://gitlab.example/project/-/issues/1"]}

    async def fake_render_check(**kwargs):
        return RenderOutcome.passed(
            url="https://gitlab.example/project/-/issues/1",
            signature="SeedNickAdv003",
            snippet="SeedNickAdv003",
            layout_probe={
                "visible_at_entry": False,
                "scroll_to_visible_px": 1200,
                "requires_expand": False,
            },
        )

    async def fake_reachability_check(**kwargs):
        return ReachabilityOutcome.direct(
            url="https://gitlab.example/project/-/issues/1",
            witnesses_matched=("SeedNickAdv003",),
            visual_reachable=True,
        )

    bundle = _bundle(
        apply_seed=fake_apply,
        render_check=fake_render_check,
        reachability_check=fake_reachability_check,
    )

    contract_id = "gitlab:benign-layout:test"
    result = asyncio.run(
        verifier._verify_one(
            {
                **_task(task_id="layout", method="create_issue_note", detail="SeedNickAdv003"),
                "benign_target_resource": {
                    "kind": "gitlab_issue",
                    "start_url_resolved": "https://gitlab.example/project/-/issues/1",
                },
                "exposure_contract": {
                    "contract_id": contract_id,
                    "eligibility": {"status": "eligible"},
                    "verification": {"url": "https://gitlab.example/project/-/issues/1"},
                },
            },
            _gitlab_instance(),
            retry_count=0,
            fingerprint_base={"host_config": "test", "editor_commit": "cafebabe"},
            ttl_hours=None,
            force_reverify=True,
            cleanup_warnings=[],
            browser=object(),
            runtime_composition=RuntimeComposition.default(),
            probes=bundle,
        )
    )

    exposure = result["feasibility"]["exposure"]
    assert result["read_surface_urls"] == ["https://gitlab.example/project/-/issues/1"]
    assert exposure["contract_id"] == contract_id
    assert exposure["reachable"] is True
    assert exposure["layout_visible_at_entry"] is False
    assert exposure["scroll_to_visible_px"] == 1200
    assert exposure["requires_expand"] is False


def test_render_check_failing_marks_render_unverified(tmp_path, monkeypatch):
    """When the rendered DOM does not contain the signature, the task is
    classified as infeasible with kind='render_unverified' and
    render_evidence captured on the error entry."""
    monkeypatch.delenv("WORLDSIM_PHASE_2C_SKIP_RENDER_CHECK", raising=False)
    sleep_calls: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    fake_browser = _FakePlaywrightBrowser(
        _FakePlaywrightPage(body="page loaded but seeded payload absent")
    )
    _install_fake_playwright(monkeypatch, fake_browser)

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
    feasibility = report.infeasible[0]["feasibility"]
    assert feasibility["status"] == "infeasible"
    error = feasibility["errors"][0]
    assert error["kind"] == "render_unverified"
    assert "render_evidence" in error
    assert error["render_evidence"]["kind"] == "render_unverified"
    assert sleep_calls == [probes._RENDER_UNVERIFIED_RETRY_DELAY_S]


def test_render_check_skipped_via_env_var_omits_render_fields(tmp_path, monkeypatch):
    """The autouse fixture sets WORLDSIM_PHASE_2C_SKIP_RENDER_CHECK=1, which
    disables render verification. Verified tasks then carry no
    render_verified / render_evidence fields — i.e. the stamp regresses
    to the pre-Layer-2 'API write succeeded only' meaning."""
    handle = _FakeHandle()

    def responder(idx, seed, instance):
        return handle

    bundle = _seed_bundle(responder)
    tasks_path = _write_tasks(tmp_path, [_task()])

    report = asyncio.run(
        feas.verify_feasibility(
            tasks_path,
            probes=bundle,
            seed_registry=_STUB_SEED_REGISTRY,
            instances=[_gitlab_instance()],
            concurrency=1,
            retry_count=0,
        )
    )
    feasibility = report.verified[0]["feasibility"]
    assert feasibility["status"] == "verified"
    assert "render_verified" not in feasibility
    assert "render_evidence" not in feasibility
