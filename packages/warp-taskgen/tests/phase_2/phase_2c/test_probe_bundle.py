"""``Phase2cProbeBundle``: default wiring, injected failures, checkpoint envelope."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from warp_taskgen.auth_tokens import acquire_tokens_for_instances
from warp_taskgen.editors import EditorError
from warp_taskgen.phase_2.phase_2c import fingerprints, probes, source_data_admission, verifier
from warp_taskgen.phase_2.phase_2c.probe_bundle import Phase2cProbeBundle
from warp_taskgen.phase_2.phase_2c.retry_timing import phase_2c_retry_sleep
from warp_taskgen.phase_2.phase_2c.runner import verify_feasibility
from warp_taskgen.phases.phase_2_reachability import verify_reachable
from warp_taskgen.phases.phase_2_render_check import RenderOutcome, verify_seed_renders
from warp_taskgen.seeding import apply_data_seed_async
from warp_taskgen.seeding.site_contracts import SeedSiteRegistration, SeedSiteRegistry

_FINGERPRINT = {"host_config": "instances.test.json", "editor_commit": "test"}


class _Handle:
    def __init__(self) -> None:
        self.cleanups = 0

    def cleanup(self) -> None:
        self.cleanups += 1


class _StubEditor:
    @classmethod
    def probe_base_state(cls, _instance: dict[str, Any]) -> None:
        return None


def _task(task_id: str = "bundle-task") -> dict[str, Any]:
    return {
        "id": task_id,
        "site": "gitlab",
        "benchmark": "webarena_verified",
        "adversarial_data_seed": {
            "editor_calls": [{"site": "gitlab", "method": "create_issue", "args": {"title": "t"}}]
        },
    }


def _instance() -> dict[str, Any]:
    return {
        "site_name": "gitlab",
        "site_url": "https://gitlab.example",
        "benchmark": "webarena_verified",
    }


# ---------------------------------------------------------------------------
# Positive: the default bundle is the production wiring
# ---------------------------------------------------------------------------


def test_default_bundle_wires_the_real_siblings() -> None:
    bundle = Phase2cProbeBundle.default()

    assert bundle.acquire_tokens is acquire_tokens_for_instances
    assert bundle.source_data_preflight is source_data_admission._run_preflight_and_filter_raw
    assert bundle.apply_seed is apply_data_seed_async
    assert bundle.render_check is probes._run_render_check
    assert bundle.reachability_check is probes._run_reachability_check
    assert bundle.verify_seed_renders is verify_seed_renders
    assert bundle.verify_reachable is verify_reachable
    assert bundle.retry_sleep is phase_2c_retry_sleep
    assert bundle.host_fingerprint is fingerprints._host_fingerprint
    assert bundle.ensure_chromium_ready is probes._ensure_playwright_chromium_ready
    assert callable(bundle.playwright_factory)


def test_default_overrides_replace_only_the_named_fields() -> None:
    async def fake_sleep(_delay: float) -> None:
        return None

    bundle = Phase2cProbeBundle.default(retry_sleep=fake_sleep)

    assert bundle.retry_sleep is fake_sleep
    assert bundle.apply_seed is apply_data_seed_async
    with pytest.raises(dataclasses.FrozenInstanceError):
        bundle.retry_sleep = phase_2c_retry_sleep  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Negative: injected failures reach the verifier's infeasible taxonomy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_seed_editor_error_yields_that_infeasible_kind() -> None:
    sleeps: list[float] = []

    async def deny(_seed: dict[str, Any], _instance: dict[str, Any], **_kwargs: Any):
        raise EditorError("permission_denied", "regular user may not create", http_status=403)

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    bundle = Phase2cProbeBundle.default(apply_seed=deny, retry_sleep=record_sleep)

    result = await verifier._verify_one(
        _task(),
        _instance(),
        retry_count=1,
        fingerprint_base=_FINGERPRINT,
        ttl_hours=None,
        force_reverify=True,
        cleanup_warnings=[],
        browser=None,
        probes=bundle,
    )

    feasibility = result["feasibility"]
    assert feasibility["status"] == "infeasible"
    assert feasibility["errors"][0]["kind"] == "permission_denied"
    assert feasibility["errors"][0]["http_status"] == 403
    # permission_denied is a platform answer: one attempt, no retry sleep.
    assert len(feasibility["attempts"]) == 1
    assert sleeps == []


@pytest.mark.asyncio
async def test_render_unverified_retries_once_then_cleans_up_once() -> None:
    handle = _Handle()
    sleeps: list[float] = []
    render_calls = 0

    async def seed(_seed: dict[str, Any], _instance: dict[str, Any], **_kwargs: Any):
        return handle, {"read_surface_urls": ["https://gitlab.example/p/-/issues/1"]}

    async def unverified(**_kwargs: Any) -> RenderOutcome:
        nonlocal render_calls
        render_calls += 1
        return RenderOutcome.failed(
            kind=probes._RENDER_UNVERIFIED_KIND,
            detail="signature absent",
            urls_tried=["https://gitlab.example/p/-/issues/1"],
            per_url_errors={},
        )

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    async def should_not_reach(**_kwargs: Any):
        raise AssertionError("reachability must not run after a failed render check")

    bundle = Phase2cProbeBundle.default(
        apply_seed=seed,
        render_check=unverified,
        reachability_check=should_not_reach,
        retry_sleep=record_sleep,
    )

    warnings: list[str] = []
    result = await verifier._verify_one(
        _task(),
        _instance(),
        retry_count=0,
        fingerprint_base=_FINGERPRINT,
        ttl_hours=None,
        force_reverify=True,
        cleanup_warnings=warnings,
        browser=object(),
        probes=bundle,
    )

    assert result["feasibility"]["status"] == "infeasible"
    assert result["feasibility"]["errors"][0]["kind"] == probes._RENDER_UNVERIFIED_KIND
    assert render_calls == 2
    assert sleeps == [probes._RENDER_UNVERIFIED_RETRY_DELAY_S]
    assert handle.cleanups == 1
    assert warnings == []


# ---------------------------------------------------------------------------
# Envelope: two runs with the same identity write the same checkpoint
# ---------------------------------------------------------------------------


def _strip_timestamps(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop the two wall-clock stamps: ``completed_at`` from ``write_checkpoint``
    and ``result.feasibility.verified_at`` from the verifier."""
    stripped = json.loads(json.dumps(payload))
    stripped.pop("completed_at")
    stripped["result"]["feasibility"].pop("verified_at")
    return stripped


@pytest.mark.asyncio
async def test_checkpoint_envelope_is_reproducible_across_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WORLDSIM_EDITOR_COMMIT_OVERRIDE", "cafebabe1234")
    monkeypatch.setenv("WORLDSIM_PHASE_2C_SKIP_RENDER_CHECK", "1")
    tasks_path = tmp_path / "adversarial_tasks.json"
    tasks_path.write_text(json.dumps([_task()]))

    async def seed(_seed: dict[str, Any], _instance: dict[str, Any], **_kwargs: Any):
        return _Handle(), {"read_surface_urls": ["https://gitlab.example/p/-/issues/1"]}

    async def no_preflight(_raw: list[dict[str, Any]], **_kwargs: Any) -> list[dict[str, Any]]:
        return []

    bundle = Phase2cProbeBundle.default(
        acquire_tokens=lambda _instances: [],
        source_data_preflight=no_preflight,
        apply_seed=seed,
    )
    registry = SeedSiteRegistry.from_registrations(
        (SeedSiteRegistration("webarena_verified", "gitlab", _StubEditor),)
    )
    checkpoint_dir = tmp_path / "checkpoints"
    envelopes: list[bytes] = []
    for _ in range(2):
        report = await verify_feasibility(
            tasks_path,
            instances=[_instance()],
            concurrency=1,
            retry_count=0,
            force_reverify=True,
            checkpoint_dir=checkpoint_dir,
            run_id="run-bundle",
            definition_digest="b" * 64,
            seed_registry=registry,
            probes=bundle,
        )
        assert len(report.verified) == 1
        assert report.reused_checkpoints == 0
        (checkpoint_file,) = checkpoint_dir.glob("*.json")
        envelopes.append(checkpoint_file.read_bytes())

    first, second = (json.loads(raw) for raw in envelopes)
    assert first["run_id"] == "run-bundle"
    assert first["topology_fingerprint"]["editor_commit"] == "cafebabe1234"
    assert _strip_timestamps(first) == _strip_timestamps(second)
    assert (
        json.dumps(_strip_timestamps(first), sort_keys=True).encode()
        == json.dumps(_strip_timestamps(second), sort_keys=True).encode()
    )
