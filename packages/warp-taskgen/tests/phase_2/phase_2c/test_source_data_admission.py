"""Source-data preflight filtering, context creation, and stale-auth refresh."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from warp_taskgen.phase_2.phase_2c import (
    auth_preflight,
    source_data_admission,
)
from warp_taskgen.phase_2.phase_2c.policy import (
    PreflightClassification,
    default_feasibility_policy_catalog,
)

from ._fixtures import (
    _bypass_preflight,  # noqa: F401
    _gitlab_instance,
    _stable_git_fingerprint,  # noqa: F401
    _task,
)


@pytest.mark.asyncio
async def test_preflight_filter_removes_stale_storage_state_when_auth_is_non_storage(
    tmp_path, monkeypatch
):
    task = _task("AT-auth", feasibility={"status": "verified"})
    task["benign_target_resource"] = {
        "kind": "gitlab_issue",
        "start_url_resolved": "https://gitlab.local/project/-/issues/1",
    }
    raw = [task]
    stale_path = tmp_path / "stale.json"
    stale_path.write_text(json.dumps({"cookies": []}))
    seen_contexts: list[dict[str, Any] | None] = []

    async def fake_preflight_benign_targets(
        tasks, *, instances_by_site, request_context_factory, feasibility_policy_catalog
    ):
        seen_contexts.extend(
            instance.get("preflight_request_context") for instance in instances_by_site["gitlab"]
        )
        return tasks, []

    class _FakeRequest:
        async def new_context(self, **kwargs):
            raise AssertionError("fake preflight should not create request contexts")

    class _FakePlaywright:
        request = _FakeRequest()

        async def stop(self):
            return None

    class _FakePlaywrightStarter:
        async def start(self):
            return _FakePlaywright()

    monkeypatch.setattr(
        "playwright.async_api.async_playwright",
        lambda: _FakePlaywrightStarter(),
    )

    dropped = await source_data_admission._run_preflight_and_filter_raw(
        raw,
        instances_by_site={
            "gitlab": [
                _gitlab_instance(
                    storage_state_path=str(stale_path),
                    agent_auth={
                        "type": "none",
                        "storage_state": {"path": str(stale_path)},
                    },
                )
            ]
        },
        probe_targets=fake_preflight_benign_targets,
        feasibility_policy_catalog=default_feasibility_policy_catalog(),
    )

    assert dropped == []
    assert seen_contexts == [{}]


@pytest.mark.asyncio
async def test_preflight_context_creation_failure_does_not_probe_anonymously(tmp_path, monkeypatch):
    task = _task("AT-auth", feasibility={"status": "verified"})
    task["benign_target_resource"] = {
        "kind": "gitlab_issue",
        "start_url_resolved": "https://gitlab.local/project/-/issues/1",
    }
    raw = [task]
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps({"cookies": [{"name": "s", "value": "1", "domain": "gitlab.example"}]})
    )

    async def fake_preflight_benign_targets(
        tasks, *, instances_by_site, request_context_factory, feasibility_policy_catalog
    ):
        context_options = instances_by_site["gitlab"][0]["preflight_request_context"]
        await request_context_factory(context_options)
        return tasks, []

    class _FakeRequest:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def new_context(self, **kwargs):
            self.calls.append(kwargs)
            raise RuntimeError("synthetic Playwright transport failure")

    fake_request = _FakeRequest()

    class _FakePlaywright:
        request = fake_request

        async def stop(self):
            return None

    class _FakePlaywrightStarter:
        async def start(self):
            return _FakePlaywright()

    monkeypatch.setattr(
        "playwright.async_api.async_playwright",
        lambda: _FakePlaywrightStarter(),
    )

    with pytest.raises(RuntimeError, match="synthetic Playwright transport failure"):
        await source_data_admission._run_preflight_and_filter_raw(
            raw,
            instances_by_site={
                "gitlab": [
                    _gitlab_instance(
                        benchmark_root=str(tmp_path),
                        storage_state_path=str(state_path),
                        agent_auth={
                            "type": "storage_state",
                            "storage_state": {"path": str(state_path)},
                        },
                    )
                ]
            },
            benchmark_root=tmp_path,
            probe_targets=fake_preflight_benign_targets,
            feasibility_policy_catalog=default_feasibility_policy_catalog(),
        )

    assert len(fake_request.calls) == 1
    storage_state = fake_request.calls[0]["storage_state"]
    assert isinstance(storage_state, dict)
    assert storage_state["cookies"][0]["sameSite"] == "Lax"


@pytest.mark.asyncio
async def test_preflight_threads_benchmark_root_into_request_context_options(monkeypatch):
    task = _task("AT-patch", feasibility={"status": "verified"})
    task["benign_target_resource"] = {
        "kind": "gitlab_issue",
        "start_url_resolved": "https://gitlab.local/project/-/issues/1",
    }
    seen_context_options: list[dict[str, Any] | None] = []

    def fake_context_options(instance, *, benchmark_root=None):
        assert benchmark_root == Path("/tmp/benchmark-root")
        assert instance["site_name"] == "gitlab"
        return {"extra_http_headers": {"X-Test": "patched"}}, None

    async def fake_preflight_benign_targets(
        tasks, *, instances_by_site, request_context_factory, feasibility_policy_catalog
    ):
        seen_context_options.append(instances_by_site["gitlab"][0]["preflight_request_context"])
        return tasks, []

    class _FakeRequest:
        async def new_context(self, **_kwargs):
            raise AssertionError("fake preflight should not create request contexts")

    class _FakePlaywright:
        request = _FakeRequest()

        async def stop(self):
            return None

    class _FakePlaywrightStarter:
        async def start(self):
            return _FakePlaywright()

    monkeypatch.setattr(auth_preflight, "_preflight_request_context_options", fake_context_options)
    monkeypatch.setattr(
        "playwright.async_api.async_playwright",
        lambda: _FakePlaywrightStarter(),
    )

    dropped = await source_data_admission._run_preflight_and_filter_raw(
        [task],
        instances_by_site={"gitlab": [_gitlab_instance(agent_auth={"type": "none"})]},
        benchmark_root=Path("/tmp/benchmark-root"),
        probe_targets=fake_preflight_benign_targets,
        feasibility_policy_catalog=default_feasibility_policy_catalog(),
    )

    assert dropped == []
    assert seen_context_options == [{"extra_http_headers": {"X-Test": "patched"}}]


@pytest.mark.asyncio
async def test_preflight_refreshes_stale_gitlab_storage_state(tmp_path, monkeypatch):
    task = _task("AT-auth", feasibility={"status": "verified"})
    task["benign_target_resource"] = {
        "kind": "gitlab_issue",
        "start_url_resolved": "https://gitlab.local/project/-/issues/1",
    }
    raw = [task]
    old_state = tmp_path / "old.json"
    old_state.write_text(
        json.dumps({"cookies": [{"name": "s", "value": "old", "domain": "gitlab.example"}]})
    )
    new_state = tmp_path / "new.json"
    new_state.write_text(
        json.dumps({"cookies": [{"name": "s", "value": "new", "domain": "gitlab.example"}]})
    )
    seen_context_options: dict[str, Any] = {}
    reacquire_calls: list[str] = []
    self_test_results = [
        PreflightClassification(
            kind="login_redirect",
            quarantine=True,
            http_status=302,
            detail="302 redirect to /users/sign_in",
        ),
        PreflightClassification(
            kind="reachable",
            quarantine=False,
            http_status=200,
            detail="200 OK",
        ),
    ]

    async def fake_self_test_auth(**_kwargs):
        return self_test_results.pop(0)

    async def fake_reacquire_storage_state(*, site_name, instance, benchmark_root):
        reacquire_calls.append(site_name)
        return new_state

    async def fake_preflight_benign_targets(
        tasks, *, instances_by_site, request_context_factory, feasibility_policy_catalog
    ):
        seen_context_options.update(instances_by_site["gitlab"][0]["preflight_request_context"])
        return tasks, []

    class _FakeContext:
        async def dispose(self):
            return None

    class _FakeRequest:
        async def new_context(self, **_kwargs):
            return _FakeContext()

    class _FakePlaywright:
        request = _FakeRequest()

        async def stop(self):
            return None

    class _FakePlaywrightStarter:
        async def start(self):
            return _FakePlaywright()

    from warp_taskgen.phases import phase_0d_auth_bootstrap

    monkeypatch.setattr(
        phase_0d_auth_bootstrap, "reacquire_storage_state", fake_reacquire_storage_state
    )
    monkeypatch.setattr(
        "playwright.async_api.async_playwright",
        lambda: _FakePlaywrightStarter(),
    )

    dropped = await source_data_admission._run_preflight_and_filter_raw(
        raw,
        instances_by_site={
            "gitlab": [
                _gitlab_instance(
                    benchmark_root=str(tmp_path),
                    storage_state_path=str(old_state),
                    agent_auth={
                        "type": "storage_state",
                        "storage_state": {"path": str(old_state)},
                    },
                )
            ]
        },
        benchmark_root=tmp_path,
        probe_targets=fake_preflight_benign_targets,
        self_test_auth=fake_self_test_auth,
        feasibility_policy_catalog=default_feasibility_policy_catalog(),
    )

    assert dropped == []
    assert reacquire_calls == ["gitlab"]
    assert seen_context_options["storage_state"]["cookies"][0]["value"] == "new"


@pytest.mark.asyncio
async def test_preflight_skips_source_data_quarantine_when_gitlab_refresh_still_stale(
    tmp_path, monkeypatch
):
    task = _task("AT-auth", feasibility={"status": "verified"})
    task["benign_target_resource"] = {
        "kind": "gitlab_issue",
        "start_url_resolved": "https://gitlab.local/project/-/issues/1",
    }
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps({"cookies": [{"name": "s", "value": "1", "domain": "gitlab.example"}]})
    )
    self_test_result = PreflightClassification(
        kind="login_redirect",
        quarantine=True,
        http_status=302,
        detail="302 redirect to /users/sign_in",
    )

    async def fake_self_test_auth(**_kwargs):
        return self_test_result

    async def fake_reacquire_storage_state(*, site_name, instance, benchmark_root):
        return state_path

    class _FakeContext:
        async def dispose(self):
            return None

    class _FakeRequest:
        async def new_context(self, **_kwargs):
            return _FakeContext()

    class _FakePlaywright:
        request = _FakeRequest()

        async def stop(self):
            return None

    class _FakePlaywrightStarter:
        async def start(self):
            return _FakePlaywright()

    from warp_taskgen.phases import phase_0d_auth_bootstrap

    monkeypatch.setattr(
        phase_0d_auth_bootstrap, "reacquire_storage_state", fake_reacquire_storage_state
    )
    monkeypatch.setattr(
        "playwright.async_api.async_playwright",
        lambda: _FakePlaywrightStarter(),
    )

    raw = [task]
    dropped = await source_data_admission._run_preflight_and_filter_raw(
        raw,
        instances_by_site={
            "gitlab": [
                _gitlab_instance(
                    benchmark_root=str(tmp_path),
                    storage_state_path=str(state_path),
                    agent_auth={
                        "type": "storage_state",
                        "storage_state": {"path": str(state_path)},
                    },
                )
            ]
        },
        benchmark_root=tmp_path,
        self_test_auth=fake_self_test_auth,
        feasibility_policy_catalog=default_feasibility_policy_catalog(),
    )

    assert dropped == []
    assert raw == [task]
