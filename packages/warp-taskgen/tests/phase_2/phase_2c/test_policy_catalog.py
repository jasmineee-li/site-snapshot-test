from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from warp_taskgen.phase_2.phase_2c import source_data_preflight
from warp_taskgen.phase_2.phase_2c.policy import (
    FeasibilityPolicyCatalog,
    PreflightClassification,
    ProbeTarget,
    SourceDataDecision,
    default_feasibility_policy_catalog,
    task_probe_targets,
)
from warp_taskgen.phase_2.phase_2c.source_data_preflight import preflight_benign_targets
from warp_taskgen.phase_2.phase_2c.webarena_policy import WebArenaFeasibilityPolicy


@dataclass(frozen=True)
class _FakePolicy:
    benchmark: str = "WebArena Verified"
    site: str = "fake"
    auth_path: str | None = None

    def auth_self_test_path(self) -> str | None:
        return self.auth_path

    def requires_authenticated_preflight(self) -> bool:
        return False

    def probe_targets(self, task: dict[str, Any], instance_site_url: str) -> list[ProbeTarget]:
        del task
        return [ProbeTarget(f"{instance_site_url}/fake-surface", "fake.surface")]

    def classify_probe(
        self,
        *,
        status: int | None,
        headers: dict[str, str] | None,
        body_snippet: str,
        exception_name: str | None,
    ) -> PreflightClassification:
        del headers, body_snippet
        if exception_name:
            return PreflightClassification("host_unreachable", False, None, exception_name)
        if status == 404:
            return PreflightClassification("fake_missing", True, status, "fake surface missing")
        return PreflightClassification("reachable", False, status, "fake surface reachable")

    def decide_source_data(
        self,
        *,
        task: dict[str, Any],
        classifications_by_target: dict[int, list[PreflightClassification]],
        target_audit: dict[int, ProbeTarget],
        candidate_replica_count: int,
        login_redirect_count: int,
        probed_count: int,
        bailout_ratio: float,
    ) -> SourceDataDecision:
        del task, candidate_replica_count, login_redirect_count, probed_count, bailout_ratio
        for target_index, classifications in classifications_by_target.items():
            for classification in classifications:
                if classification.quarantine:
                    return SourceDataDecision(
                        "drop",
                        classification,
                        target_audit[target_index],
                    )
        return SourceDataDecision("keep")

    def counts_toward_run_bailout(self, classification: PreflightClassification) -> bool:
        del classification
        return False

    def should_bailout_source_data_run(
        self,
        *,
        bailout_count: int,
        probed_count: int,
        bailout_ratio: float,
    ) -> bool:
        del bailout_count, probed_count, bailout_ratio
        return False

    def restore_drop_on_run_bailout(self, issue: dict[str, Any]) -> bool:
        del issue
        return False


class _FakeResponse:
    def __init__(self, status: int) -> None:
        self.status = status
        self.headers: dict[str, str] = {}

    async def text(self) -> str:
        return ""

    async def dispose(self) -> None:
        return None


class _FakeRequestContext:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.calls: list[str] = []

    async def get(self, url: str, *, timeout: float, max_redirects: int) -> _FakeResponse:
        del timeout, max_redirects
        self.calls.append(url)
        return self.response


def _fake_task() -> dict[str, Any]:
    return {
        "id": "fake-task",
        "site": "fake",
        "benchmark": "WebArena Verified",
        "benign_target_resource": {"start_url_resolved": "https://fake.local/start"},
    }


def test_catalog_normalizes_aliases_and_is_immutable() -> None:
    catalog = FeasibilityPolicyCatalog.from_policies([_FakePolicy()])

    assert catalog.get("webarena_verified", "FAKE") is not None
    with pytest.raises(TypeError):
        catalog.policies["webarena_verified", "other"] = _FakePolicy(site="other")


def test_catalog_rejects_duplicate_aliases_and_malformed_policies() -> None:
    with pytest.raises(ValueError, match="duplicate feasibility policy"):
        FeasibilityPolicyCatalog.from_policies(
            [_FakePolicy(), _FakePolicy(benchmark="webarena_verified", site="FAKE")]
        )

    with pytest.raises(ValueError, match="non-empty site"):
        FeasibilityPolicyCatalog.from_policies([_FakePolicy(site=" ")])

    with pytest.raises(TypeError, match="must expose benchmark and site"):
        FeasibilityPolicyCatalog.from_policies([object()])  # type: ignore[list-item]


def test_catalog_unknown_binding_and_active_webarena_parity() -> None:
    catalog = default_feasibility_policy_catalog()
    assert catalog.get("WebArena Verified", "gitlab") is not None
    assert catalog.get("WebArena Verified", "reddit") is not None
    assert catalog.get("WebArena Verified", "removed-site") is None

    gitlab_task = {
        "site": "gitlab",
        "benchmark": "WebArena Verified",
        "benign_target_resource": {
            "start_url_resolved": "https://gitlab.local/project/-/issues/1",
            "anchors": {"project_path": "project", "issue_iid": "1"},
        },
        "adversarial_data_seed": {
            "editor_calls": [
                {
                    "site": "gitlab",
                    "method": "create_issue_note",
                    "args": {
                        "project_path": "{benign_project_path}",
                        "issue_iid": "{benign_issue_iid}",
                    },
                }
            ]
        },
    }
    reddit_task = {
        "site": "reddit",
        "benchmark": "WebArena Verified",
        "benign_target_resource": {
            "start_url_resolved": "https://reddit.local/f/news",
            "anchors": {"forum_name": "news", "submission_id": "5"},
        },
        "adversarial_data_seed": {
            "editor_calls": [
                {
                    "site": "reddit",
                    "method": "create_comment",
                    "args": {
                        "forum_name": "{benign_forum_name}",
                        "submission_id": "{benign_submission_id}",
                    },
                }
            ]
        },
    }
    for task, instance_url, policy in (
        (gitlab_task, "http://gitlab.test", WebArenaFeasibilityPolicy("gitlab", "/-/profile")),
        (reddit_task, "http://reddit.test", WebArenaFeasibilityPolicy("reddit")),
    ):
        expected = policy.probe_targets(task, instance_url)
        actual = task_probe_targets(
            task,
            instance_url,
            feasibility_policy_catalog=catalog,
        )
        assert actual == expected
    assert catalog.get("WebArena Verified", "gitlab").auth_self_test_path() == "/-/profile"
    assert catalog.get("WebArena Verified", "reddit").auth_self_test_path() is None


def test_fake_policy_preflight_is_per_run_and_does_not_leak() -> None:
    task = _fake_task()
    catalog = FeasibilityPolicyCatalog.from_policies([_FakePolicy()])
    context = _FakeRequestContext(_FakeResponse(404))

    async def factory(_options: dict[str, Any]) -> _FakeRequestContext:
        return context

    instance = {
        "site_name": "fake",
        "benchmark": "webarena_verified",
        "site_url": "http://fake.test",
    }
    keep, dropped = asyncio.run(
        preflight_benign_targets(
            [task],
            instances_by_site={"fake": [instance]},
            request_context_factory=factory,
            feasibility_policy_catalog=catalog,
        )
    )
    assert keep == []
    assert dropped[0]["source_data_issue"]["kind"] == "fake_missing"
    assert context.calls == ["http://fake.test/fake-surface"]

    keep, dropped = asyncio.run(
        preflight_benign_targets(
            [task],
            instances_by_site={"fake": [instance]},
            request_context_factory=factory,
        )
    )
    assert keep == [task]
    assert dropped == []
    assert context.calls == ["http://fake.test/fake-surface"]


def test_canonical_source_preflight_threads_policy_catalog_to_probe(monkeypatch) -> None:
    catalog = FeasibilityPolicyCatalog.from_policies([_FakePolicy()])
    raw = [_fake_task()]
    context = _FakeRequestContext(_FakeResponse(404))

    class _FakeRequest:
        async def new_context(self, **_kwargs: Any) -> _FakeRequestContext:
            return context

    class _FakePlaywright:
        request = _FakeRequest()

        async def stop(self) -> None:
            return None

    class _FakeStarter:
        async def start(self) -> _FakePlaywright:
            return _FakePlaywright()

    monkeypatch.setattr(
        "playwright.async_api.async_playwright",
        lambda: _FakeStarter(),
    )

    dropped = asyncio.run(
        source_data_preflight._run_preflight_and_filter_raw(
            raw,
            instances_by_site={
                "fake": [
                    {
                        "site_name": "fake",
                        "benchmark": "webarena_verified",
                        "site_url": "http://fake.test",
                    }
                ]
            },
            feasibility_policy_catalog=catalog,
        )
    )
    assert raw == []
    assert dropped[0]["source_data_issue"]["kind"] == "fake_missing"
    assert context.calls == ["http://fake.test/fake-surface"]
