"""Named Runtime Composition: cleanup failures abort the public Phase 2c runner."""

from __future__ import annotations

import asyncio

import pytest

from warp_taskgen.editors import EditorError
from warp_taskgen.phase_2.phase_2c import runner as feas

from ._fixtures import (
    _bundle,
    _bypass_preflight,  # noqa: F401
    _FakeHandle,
    _gitlab_instance,
    _stable_git_fingerprint,  # noqa: F401
    _task,
    _write_tasks,
)


def test_named_composition_cleanup_failure_aborts_public_phase2c_runner(tmp_path, monkeypatch):
    from warp_taskgen.editors import GitlabEditor
    from warp_taskgen.phase_2.phase_2c.policy import default_feasibility_policy_catalog
    from warp_taskgen.runtime_composition import (
        RequiredSeedCleanupError,
        RuntimeComposition,
    )
    from warp_taskgen.seeding.site_contracts import SeedSiteRegistration, SeedSiteRegistry
    from warp_taskgen.sites.catalog import SiteCatalog

    async def seed_with_failing_cleanup(seed, instance, **kwargs):
        assert kwargs["strict_cleanup"] is True
        return _FakeHandle(raises=True), {}

    bundle = _bundle(apply_seed=seed_with_failing_cleanup)
    tasks_path = _write_tasks(tmp_path, [_task()])
    editor = GitlabEditor
    monkeypatch.setattr(editor, "probe_base_state", classmethod(lambda _cls, _instance: None))
    composition = RuntimeComposition(
        name="strict-cleanup-test",
        site_catalog=SiteCatalog(),
        seed_registry=SeedSiteRegistry.from_registrations(
            (
                SeedSiteRegistration(
                    "webarena_verified",
                    "gitlab",
                    editor,
                ),
            )
        ),
        feasibility_policy_catalog=default_feasibility_policy_catalog(),
        strict_seed_cleanup=True,
        seed_token_scope="method",
        strict_site_planning=True,
    )

    with pytest.raises(RequiredSeedCleanupError, match="cleanup_failed"):
        asyncio.run(
            feas.verify_feasibility(
                tasks_path,
                probes=bundle,
                instances=[_gitlab_instance()],
                concurrency=1,
                retry_count=0,
                runtime_composition=composition,
            )
        )


def test_named_composition_partial_seed_cleanup_failure_aborts_public_phase2c_runner(
    tmp_path, monkeypatch
):
    from warp_taskgen.editors import GitlabEditor
    from warp_taskgen.phase_2.phase_2c.policy import default_feasibility_policy_catalog
    from warp_taskgen.runtime_composition import RequiredSeedCleanupError, RuntimeComposition
    from warp_taskgen.seeding.site_contracts import SeedSiteRegistration, SeedSiteRegistry
    from warp_taskgen.sites.catalog import SiteCatalog

    async def fake_apply(seed, instance, **kwargs):
        assert kwargs["strict_cleanup"] is True
        primary = EditorError("request_failed", "second call failed after a partial write")
        raise RequiredSeedCleanupError(
            "required seed cleanup failed after seed execution error",
            primary_error=primary,
            cleanup_error=RuntimeError("delete witness failed"),
        ) from primary

    async def no_source_preflight(raw, **kwargs):
        return []

    bundle = _bundle(apply_seed=fake_apply, source_data_preflight=no_source_preflight)
    tasks_path = _write_tasks(tmp_path, [_task()])
    editor = GitlabEditor
    monkeypatch.setattr(editor, "probe_base_state", classmethod(lambda _cls, _instance: None))
    composition = RuntimeComposition(
        name="strict-partial-seed-test",
        site_catalog=SiteCatalog(),
        seed_registry=SeedSiteRegistry.from_registrations(
            (SeedSiteRegistration("webarena_verified", "gitlab", editor),)
        ),
        feasibility_policy_catalog=default_feasibility_policy_catalog(),
        strict_seed_cleanup=True,
        seed_token_scope="method",
        strict_site_planning=True,
    )

    with pytest.raises(RequiredSeedCleanupError) as raised:
        asyncio.run(
            feas.verify_feasibility(
                tasks_path,
                probes=bundle,
                instances=[_gitlab_instance()],
                concurrency=1,
                retry_count=0,
                runtime_composition=composition,
            )
        )

    assert isinstance(raised.value.primary_error, EditorError)
    assert raised.value.primary_error.kind == "request_failed"
