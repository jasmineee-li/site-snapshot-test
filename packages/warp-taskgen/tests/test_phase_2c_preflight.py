"""Unit tests for the Phase 2c source-data preflight and its auth self-test.

Bug I (2026-04-23): pre-seed HTTP probe + source-data quarantine.
These tests exercise the classifier rules + the whole-run bailout.
"""

from __future__ import annotations

import asyncio
from typing import Any

from warp_taskgen.phase_2.phase_2c.policy import default_feasibility_policy_catalog
from warp_taskgen.phase_2.phase_2c.preflight_auth_self_test import self_test_preflight_auth
from warp_taskgen.phase_2.phase_2c.source_data_preflight import preflight_benign_targets
from warp_taskgen.phase_2.phase_2c.webarena_policy import (
    classify_webarena_probe,
    location_is_login,
    looks_like_login_stub,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(
        self,
        *,
        status: int,
        headers: dict[str, str] | None = None,
        body: str = "",
    ) -> None:
        self.status = status
        self.headers = headers or {}
        self._body = body
        self.disposed = False

    async def text(self) -> str:
        return self._body

    async def dispose(self) -> None:
        self.disposed = True


class _FakeRequestContext:
    def __init__(self, *, response_map: dict[str, _FakeResponse] | None = None) -> None:
        self._response_map = response_map or {}
        self.calls: list[tuple[str, float, int]] = []

    async def get(self, url: str, *, timeout: float, max_redirects: int) -> _FakeResponse:
        self.calls.append((url, timeout, max_redirects))
        if url in self._response_map:
            resp = self._response_map[url]
            if isinstance(resp, Exception):
                raise resp
            return resp
        return _FakeResponse(status=200, body="ok")


# ---------------------------------------------------------------------------
# classify_webarena_probe — table-driven
# ---------------------------------------------------------------------------


def test_classify_200_clean_reachable():
    c = classify_webarena_probe(
        status=200, headers={}, body_snippet="<html>welcome</html>", exception_name=None
    )
    assert c.kind == "reachable" and not c.quarantine


def test_classify_200_login_stub_short_body():
    c = classify_webarena_probe(
        status=200,
        headers={},
        body_snippet="please sign in to continue" * 2,
        exception_name=None,
    )
    assert c.kind == "login_redirect" and c.quarantine


def test_classify_200_login_stub_form_marker():
    c = classify_webarena_probe(
        status=200,
        headers={},
        body_snippet='<form action="/users/sign_in" method="post">',
        exception_name=None,
    )
    assert c.kind == "login_redirect" and c.quarantine


def test_classify_302_to_sign_in_is_login_redirect():
    # Playwright lowercases response header names so the classifier only
    # looks up ``location`` (not ``Location``). Keep the test lowercase
    # to match the real Playwright contract.
    c = classify_webarena_probe(
        status=302,
        headers={"location": "http://host/users/sign_in"},
        body_snippet="",
        exception_name=None,
    )
    assert c.kind == "login_redirect" and c.quarantine


def test_classify_login_redirect_redacts_location_query():
    c = classify_webarena_probe(
        status=302,
        headers={"location": "http://host/users/sign_in?token=secret&return_to=/private"},
        body_snippet="",
        exception_name=None,
    )

    assert c.kind == "login_redirect"
    assert "secret" not in c.detail
    assert "token=" not in c.detail
    assert c.detail == "302 redirect to http://host/users/sign_in"


def test_classify_302_non_login_is_redirect_noncritical():
    c = classify_webarena_probe(
        status=301,
        headers={"location": "/projects/foo/bar"},
        body_snippet="",
        exception_name=None,
    )
    assert c.kind == "redirect_noncritical" and not c.quarantine


def test_classify_404_quarantines_as_not_found():
    c = classify_webarena_probe(status=404, headers={}, body_snippet="", exception_name=None)
    assert c.kind == "not_found" and c.quarantine


def test_classify_403_quarantines_as_forbidden():
    c = classify_webarena_probe(status=403, headers={}, body_snippet="", exception_name=None)
    assert c.kind == "forbidden" and c.quarantine


def test_classify_401_quarantines_as_auth_missing():
    c = classify_webarena_probe(status=401, headers={}, body_snippet="", exception_name=None)
    assert c.kind == "auth_missing" and c.quarantine


def test_classify_410_quarantines_as_gone():
    c = classify_webarena_probe(status=410, headers={}, body_snippet="", exception_name=None)
    assert c.kind == "gone" and c.quarantine


def test_classify_429_does_not_quarantine():
    c = classify_webarena_probe(status=429, headers={}, body_snippet="", exception_name=None)
    assert c.kind == "rate_limited" and not c.quarantine


def test_classify_500_does_not_quarantine():
    c = classify_webarena_probe(status=503, headers={}, body_snippet="", exception_name=None)
    assert c.kind == "server_error" and not c.quarantine


def test_classify_connection_error_does_not_quarantine():
    c = classify_webarena_probe(
        status=None,
        headers=None,
        body_snippet="",
        exception_name="ConnectionRefusedError",
    )
    assert c.kind == "host_unreachable" and not c.quarantine


def test_classify_timeout_does_not_quarantine():
    c = classify_webarena_probe(
        status=None, headers=None, body_snippet="", exception_name="TimeoutError"
    )
    assert c.kind == "probe_timeout" and not c.quarantine


def test_location_is_login_helper():
    assert location_is_login("http://host/users/sign_in?foo=1")
    assert location_is_login("/login")
    assert not location_is_login("/projects/foo/issues/1")
    assert not location_is_login(None)


def test_looks_like_login_stub_boundaries():
    # Short body + "sign in" → stub.
    assert looks_like_login_stub("please sign in")
    # Long prose body mentioning "sign in" → NOT stub (body_len gate).
    long_body = "This README mentions how to sign in for local dev." + (" " * 700)
    assert not looks_like_login_stub(long_body)
    # Form-action marker → stub regardless of body length.
    assert looks_like_login_stub("x" * 2000 + '<form action="/users/sign_in"')


def test_self_test_preflight_auth_gitlab_alive():
    context = _FakeRequestContext(
        response_map={"http://gitlab.test/-/profile": _FakeResponse(status=200, body="profile")}
    )

    result = asyncio.run(
        self_test_preflight_auth(
            request_context=context,
            site="gitlab",
            site_url="http://gitlab.test",
        )
    )

    assert result is not None
    assert result.kind == "reachable"
    assert context.calls[0][0] == "http://gitlab.test/-/profile"


def test_self_test_preflight_auth_gitlab_detects_login_redirect():
    context = _FakeRequestContext(
        response_map={
            "http://gitlab.test/-/profile": _FakeResponse(
                status=302, headers={"location": "/users/sign_in"}
            )
        }
    )

    result = asyncio.run(
        self_test_preflight_auth(
            request_context=context,
            site="gitlab",
            site_url="http://gitlab.test",
        )
    )

    assert result is not None
    assert result.kind == "login_redirect"


def test_self_test_preflight_auth_gitlab_detects_unauthorized():
    context = _FakeRequestContext(
        response_map={"http://gitlab.test/-/profile": _FakeResponse(status=401)}
    )

    result = asyncio.run(
        self_test_preflight_auth(
            request_context=context,
            site="gitlab",
            site_url="http://gitlab.test",
        )
    )

    assert result is not None
    assert result.kind == "auth_missing"


def test_self_test_preflight_auth_skips_reddit():
    context = _FakeRequestContext()

    result = asyncio.run(
        self_test_preflight_auth(
            request_context=context,
            site="reddit",
            site_url="http://reddit.test",
        )
    )

    assert result is None
    assert context.calls == []


# ---------------------------------------------------------------------------
# preflight_benign_targets — integration via fakes
# ---------------------------------------------------------------------------


def _make_task(task_id: str, site: str, start_url: str) -> dict[str, Any]:
    return {
        "id": task_id,
        "site": site,
        "benchmark": "webarena_verified",
        "benign_target_resource": {
            "kind": "gitlab_issue",
            "start_url_resolved": start_url,
        },
    }


def _gitlab_instance(site_url: str = "http://gitlab.test") -> dict[str, Any]:
    return {
        "site_name": "gitlab",
        "benchmark": "webarena_verified",
        "site_url": site_url,
        "preflight_request_context": {"extra_http_headers": {"X-Test-Auth": "1"}},
    }


def test_preflight_splits_quarantine_from_keep():
    keep_task = _make_task("adv_keep", "gitlab", "http://gitlab.test/foo/-/issues/1")
    drop_task = _make_task("adv_drop", "gitlab", "http://gitlab.test/foo/-/merge_requests/2")
    instances_by_site = {"gitlab": [_gitlab_instance()]}
    request_context = _FakeRequestContext(
        response_map={
            "http://gitlab.test/foo/-/issues/1": _FakeResponse(status=200, body="clean"),
            "http://gitlab.test/foo/-/merge_requests/2": _FakeResponse(status=404, body=""),
        }
    )

    async def _factory(_storage_state_path):
        return request_context

    keep, dropped = asyncio.run(
        preflight_benign_targets(
            [keep_task, drop_task],
            instances_by_site=instances_by_site,
            request_context_factory=_factory,
        )
    )
    assert [t["id"] for t in keep] == ["adv_keep"]
    assert [t["id"] for t in dropped] == ["adv_drop"]
    assert dropped[0]["source_data_issue"]["kind"] == "not_found"
    assert dropped[0]["source_data_issue"]["http_status"] == 404
    assert "probed_at" in dropped[0]["source_data_issue"]


def test_preflight_uses_canonical_benchmark_for_policy_lookup():
    task = _make_task("adv_alias", "gitlab", "http://gitlab.test/foo/-/issues/99")
    task["benchmark_name"] = "WebArena Verified"
    instances_by_site = {"gitlab": [_gitlab_instance()]}
    request_context = _FakeRequestContext(
        response_map={
            "http://gitlab.test/foo/-/issues/99": _FakeResponse(status=404, body=""),
        }
    )

    async def _factory(_storage_state_path):
        return request_context

    keep, dropped = asyncio.run(
        preflight_benign_targets(
            [task],
            instances_by_site=instances_by_site,
            request_context_factory=_factory,
        )
    )

    assert keep == []
    assert [record["id"] for record in dropped] == ["adv_alias"]
    assert dropped[0]["source_data_issue"]["kind"] == "not_found"


def test_preflight_missing_benchmark_metadata_keeps_without_policy_probe():
    task = _make_task("adv_no_benchmark", "gitlab", "http://gitlab.test/foo/-/issues/99")
    task.pop("benchmark", None)
    instance = _gitlab_instance()
    instance.pop("benchmark", None)
    request_context = _FakeRequestContext(
        response_map={
            "http://gitlab.test/foo/-/issues/99": _FakeResponse(status=404, body=""),
        }
    )

    async def _factory(_storage_state_path):
        return request_context

    keep, dropped = asyncio.run(
        preflight_benign_targets(
            [task],
            instances_by_site={"gitlab": [instance]},
            request_context_factory=_factory,
        )
    )

    assert keep == [task]
    assert dropped == []
    assert request_context.calls == []


def test_policy_lookup_normalizes_benchmark_alias():
    assert default_feasibility_policy_catalog().get("WebArena Verified", "gitlab") is not None


def test_preflight_redacts_sensitive_probe_url_fields():
    task = _make_task(
        "adv_secret_url",
        "reddit",
        "http://reddit.test/f/news/5?token=secret&user=alice",
    )
    instances_by_site = {"reddit": [{"site_name": "reddit", "site_url": "http://reddit.test"}]}
    request_context = _FakeRequestContext(
        response_map={
            "http://reddit.test/f/news/5?token=secret&user=alice": _FakeResponse(status=404),
        }
    )

    async def _factory(_storage_state_path):
        return request_context

    keep, dropped = asyncio.run(
        preflight_benign_targets(
            [task],
            instances_by_site=instances_by_site,
            request_context_factory=_factory,
        )
    )

    assert keep == []
    issue = dropped[0]["source_data_issue"]
    assert issue["probed_url"] == (
        "http://reddit.test/f/news/5?token=%3Credacted%3E&user=%3Credacted%3E"
    )


def test_gitlab_preflight_without_auth_keeps_task_without_probe():
    task = _make_task("adv_no_auth", "gitlab", "http://gitlab.test/foo/-/issues/99")
    instances_by_site = {"gitlab": [{"site_name": "gitlab", "site_url": "http://gitlab.test"}]}
    request_context = _FakeRequestContext(
        response_map={
            "http://gitlab.test/foo/-/issues/99": _FakeResponse(status=404, body=""),
        }
    )

    async def _factory(_storage_state_path):
        return request_context

    keep, dropped = asyncio.run(
        preflight_benign_targets(
            [task],
            instances_by_site=instances_by_site,
            request_context_factory=_factory,
        )
    )

    assert keep == [task]
    assert dropped == []
    assert request_context.calls == []


def test_preflight_quorum_counts_auth_skipped_replicas():
    task = _make_task("adv_one_probe", "gitlab", "http://gitlab.test/foo/-/issues/99")
    probed = _gitlab_instance()
    skipped = _gitlab_instance("http://gitlab2.test")
    skipped["preflight_auth_skip_reason"] = "storage_state refresh failed"
    request_context = _FakeRequestContext(
        response_map={
            "http://gitlab.test/foo/-/issues/99": _FakeResponse(status=404, body=""),
        }
    )

    async def _factory(_storage_state_path):
        return request_context

    keep, dropped = asyncio.run(
        preflight_benign_targets(
            [task],
            instances_by_site={"gitlab": [probed, skipped]},
            request_context_factory=_factory,
        )
    )

    assert keep == [task]
    assert dropped == []


def test_preflight_uses_instance_benchmark_over_task_metadata():
    task = _make_task("adv_instance_policy", "gitlab", "http://gitlab.test/foo/-/issues/99")
    task["benchmark"] = "wasp"
    instances_by_site = {"gitlab": [_gitlab_instance()]}
    request_context = _FakeRequestContext(
        response_map={
            "http://gitlab.test/foo/-/issues/99": _FakeResponse(status=404, body=""),
        }
    )

    async def _factory(_storage_state_path):
        return request_context

    keep, dropped = asyncio.run(
        preflight_benign_targets(
            [task],
            instances_by_site=instances_by_site,
            request_context_factory=_factory,
        )
    )

    assert keep == []
    assert [record["id"] for record in dropped] == ["adv_instance_policy"]
    assert dropped[0]["source_data_issue"]["kind"] == "not_found"


def test_preflight_bailout_when_login_redirect_dominates():
    # >50 % login_redirect → mass-restore + skip quarantine entirely.
    tasks = [
        _make_task(f"adv_{i}", "gitlab", f"http://gitlab.test/p/-/issues/{i}") for i in range(6)
    ]
    instances_by_site = {"gitlab": [_gitlab_instance()]}
    response_map: dict[str, _FakeResponse] = {}
    for i in range(4):  # 4 login-redirects
        response_map[f"http://gitlab.test/p/-/issues/{i}"] = _FakeResponse(
            status=302, headers={"Location": "/users/sign_in"}
        )
    for i in range(4, 6):  # 2 normal
        response_map[f"http://gitlab.test/p/-/issues/{i}"] = _FakeResponse(status=200, body="ok")
    request_context = _FakeRequestContext(response_map=response_map)

    async def _factory(_storage_state_path):
        return request_context

    keep, dropped = asyncio.run(
        preflight_benign_targets(
            tasks,
            instances_by_site=instances_by_site,
            request_context_factory=_factory,
        )
    )
    # Bailout: nothing quarantined, all tasks returned to keep.
    assert dropped == []
    assert sorted(t["id"] for t in keep) == [f"adv_{i}" for i in range(6)]


def test_preflight_bailout_restores_original_task_objects_with_mixed_drops():
    login_task_1 = _make_task("adv_login_1", "gitlab", "http://gitlab.test/p/-/issues/1")
    login_task_2 = _make_task("adv_login_2", "gitlab", "http://gitlab.test/p/-/issues/2")
    not_found_task = _make_task("adv_404", "gitlab", "http://gitlab.test/p/-/issues/404")
    ok_task = _make_task("adv_ok", "gitlab", "http://gitlab.test/p/-/issues/3")
    instances_by_site = {"gitlab": [_gitlab_instance()]}
    request_context = _FakeRequestContext(
        response_map={
            "http://gitlab.test/p/-/issues/1": _FakeResponse(
                status=302, headers={"location": "/users/sign_in"}
            ),
            "http://gitlab.test/p/-/issues/2": _FakeResponse(
                status=302, headers={"location": "/users/sign_in"}
            ),
            "http://gitlab.test/p/-/issues/404": _FakeResponse(status=404),
            "http://gitlab.test/p/-/issues/3": _FakeResponse(status=200, body="ok"),
        }
    )

    async def _factory(_storage_state_path):
        return request_context

    keep, dropped = asyncio.run(
        preflight_benign_targets(
            [login_task_1, login_task_2, not_found_task, ok_task],
            instances_by_site=instances_by_site,
            request_context_factory=_factory,
            bailout_ratio=0.25,
        )
    )

    assert [task["id"] for task in dropped] == ["adv_404"]
    assert login_task_1 in keep
    assert login_task_2 in keep
    assert ok_task in keep
    assert all("source_data_issue" not in task for task in keep)


def test_preflight_skips_tasks_with_no_benign_target():
    malformed = {"id": "adv_x", "site": "gitlab"}  # no benign_target_resource
    instances_by_site = {"gitlab": [{"site_name": "gitlab", "site_url": "http://gitlab.test"}]}

    async def _factory(_):
        return _FakeRequestContext()

    keep, dropped = asyncio.run(
        preflight_benign_targets(
            [malformed],
            instances_by_site=instances_by_site,
            request_context_factory=_factory,
        )
    )
    assert keep == [malformed]
    assert dropped == []


def test_preflight_skips_tasks_whose_site_has_no_instance():
    task = _make_task("adv_orphan", "mystery", "http://nowhere/x")

    async def _factory(_):
        return _FakeRequestContext()

    keep, dropped = asyncio.run(
        preflight_benign_targets(
            [task],
            instances_by_site={},  # no mystery instance
            request_context_factory=_factory,
        )
    )
    assert keep == [task]
    assert dropped == []


def test_preflight_empty_task_list_short_circuits():
    async def _factory(_):
        raise AssertionError("factory should not be called")

    keep, dropped = asyncio.run(
        preflight_benign_targets([], instances_by_site={}, request_context_factory=_factory)
    )
    assert keep == [] and dropped == []


def test_preflight_unanimous_quarantine_across_replicas():
    # All 3 replicas return 404 for the same anchor — classic stale-L4
    # signal. Preflight must quarantine.
    task = _make_task("adv_stale", "reddit", "https://reddit.local/f/books/999")
    instances_by_site = {
        "reddit": [
            {"site_name": "reddit", "site_url": "http://host0:9900"},
            {"site_name": "reddit", "site_url": "http://host1:9900"},
            {"site_name": "reddit", "site_url": "http://host2:9900"},
        ]
    }
    request_context = _FakeRequestContext(
        response_map={
            "http://host0:9900/f/books/999": _FakeResponse(status=404),
            "http://host1:9900/f/books/999": _FakeResponse(status=404),
            "http://host2:9900/f/books/999": _FakeResponse(status=404),
        }
    )

    async def _factory(_):
        return request_context

    _keep, dropped = asyncio.run(
        preflight_benign_targets(
            [task],
            instances_by_site=instances_by_site,
            request_context_factory=_factory,
        )
    )
    assert len(dropped) == 1
    assert dropped[0]["source_data_issue"]["kind"] == "not_found"
    assert dropped[0]["source_data_issue"]["replicas_probed"] == 3


def test_preflight_majority_quarantine_with_replica_0_drift():
    # r5 reddit_0 drift pattern: replica 0 has legacy data (200 OK)
    # but 9/10 replicas return 404 after a fleet reset. The editor's
    # P2C selection lands on a broken replica 90 % of the time and
    # the task fails in-run. Majority rule (> 50 %) correctly
    # quarantines here; the earlier unanimity rule did not, which
    # caused the first Bug I r5 run to miss the 7 reddit stale-L4
    # tasks.
    task = _make_task("adv_drift", "reddit", "https://reddit.local/f/headphones/4")
    instances_by_site = {
        "reddit": [{"site_name": "reddit", "site_url": f"http://host{i}:9900"} for i in range(10)]
    }
    response_map = {"http://host0:9900/f/headphones/4": _FakeResponse(status=200, body="legacy")}
    for i in range(1, 10):
        response_map[f"http://host{i}:9900/f/headphones/4"] = _FakeResponse(status=404)
    request_context = _FakeRequestContext(response_map=response_map)

    async def _factory(_):
        return request_context

    _keep, dropped = asyncio.run(
        preflight_benign_targets(
            [task],
            instances_by_site=instances_by_site,
            request_context_factory=_factory,
        )
    )
    assert len(dropped) == 1
    issue = dropped[0]["source_data_issue"]
    assert issue["kind"] == "not_found"
    assert issue["replicas_probed"] == 10
    assert issue["replicas_agreeing"] == 9
    # All replicas probed (no early-exit — majority rule needs full canvas).
    assert len(request_context.calls) == 10


def test_preflight_minority_quarantine_passes_through():
    # Opposite of the drift case: 1/3 replicas say 404 (transient
    # delivery glitch or stale shard), 2/3 say 200. Not a majority →
    # pass through to the real probe.
    task = _make_task("adv_flaky", "reddit", "https://reddit.local/f/books/100")
    instances_by_site = {
        "reddit": [{"site_name": "reddit", "site_url": f"http://host{i}:9900"} for i in range(3)]
    }
    request_context = _FakeRequestContext(
        response_map={
            "http://host0:9900/f/books/100": _FakeResponse(status=404),
            "http://host1:9900/f/books/100": _FakeResponse(status=200, body="ok"),
            "http://host2:9900/f/books/100": _FakeResponse(status=200, body="ok"),
        }
    )

    async def _factory(_):
        return request_context

    keep, dropped = asyncio.run(
        preflight_benign_targets(
            [task],
            instances_by_site=instances_by_site,
            request_context_factory=_factory,
        )
    )
    assert keep == [task]
    assert dropped == []


def test_preflight_rewrites_synthetic_hostname_to_live_url():
    # Phase 2a emits ``start_url_resolved`` against synthetic hosts
    # (``https://gitlab.local/...``) and Phase 2c rewrites them via
    # ``resolve_start_url(start_url, instance.site_url)``. Regression
    # guard: the preflight must probe the LIVE URL, not the synthetic
    # one, otherwise every request would fail DNS.
    task = {
        "id": "adv_synth",
        "site": "gitlab",
        "benign_target_resource": {
            "kind": "gitlab_issue",
            "start_url_resolved": "https://gitlab.local/foo/-/issues/7",
        },
    }
    instances_by_site = {"gitlab": [_gitlab_instance("http://172.17.0.1:8023")]}
    request_context = _FakeRequestContext(
        response_map={
            "http://172.17.0.1:8023/foo/-/issues/7": _FakeResponse(status=200, body="ok"),
        }
    )

    async def _factory(_):
        return request_context

    keep, dropped = asyncio.run(
        preflight_benign_targets(
            [task],
            instances_by_site=instances_by_site,
            request_context_factory=_factory,
        )
    )
    assert keep == [task]
    assert dropped == []
    # Assert the probe hit the LIVE URL, not the synthetic one.
    assert request_context.calls, "probe was never invoked"
    assert request_context.calls[0][0] == "http://172.17.0.1:8023/foo/-/issues/7"


def test_preflight_cleans_gitlab_editor_surface_project_path_authority():
    task = {
        "id": "adv_gitlab_authority_path",
        "site": "gitlab",
        "benign_target_resource": {
            "kind": "gitlab_issue",
            "start_url_resolved": "https://gitlab.local/a11yproject/a11yproject.com/-/issues/1064",
            "anchors": {
                "project_path": "localhost:8023/a11yproject/a11yproject.com",
                "issue_iid": "1064",
            },
        },
        "adversarial_data_seed": {
            "mechanism": "editor",
            "editor_calls": [
                {
                    "site": "gitlab",
                    "method": "create_issue_note",
                    "args": {
                        "project_path": "{benign_project_path}",
                        "issue_iid": "{benign_issue_iid}",
                        "body": "payload",
                    },
                }
            ],
        },
    }
    instances_by_site = {"gitlab": [_gitlab_instance("http://172.17.0.1:8023")]}
    malformed = "http://172.17.0.1:8023/localhost:8023/a11yproject/a11yproject.com/-/issues/1064"
    expected = "http://172.17.0.1:8023/a11yproject/a11yproject.com/-/issues/1064"
    request_context = _FakeRequestContext(
        response_map={
            expected: _FakeResponse(status=200, body="ok"),
            malformed: _FakeResponse(status=404),
        }
    )

    async def _factory(_):
        return request_context

    keep, dropped = asyncio.run(
        preflight_benign_targets(
            [task],
            instances_by_site=instances_by_site,
            request_context_factory=_factory,
        )
    )

    assert keep == [task]
    assert dropped == []
    probed_urls = [url for url, _, _ in request_context.calls]
    assert probed_urls == [expected]
    assert malformed not in probed_urls


def test_preflight_reuses_request_context_across_same_site_tasks():
    # Memoization invariant: tasks on the same (site, storage_state_path)
    # pair must share a single APIRequestContext so we do not pay the
    # TLS+cookie setup cost per task. The factory is recorded to verify.
    tasks = [
        _make_task(f"adv_{i}", "gitlab", f"http://gitlab.test/p/-/issues/{i}") for i in range(3)
    ]
    instances_by_site = {"gitlab": [_gitlab_instance()]}
    request_context = _FakeRequestContext()
    factory_calls = 0

    async def _factory(_storage_state_path):
        nonlocal factory_calls
        factory_calls += 1
        return request_context

    keep, dropped = asyncio.run(
        preflight_benign_targets(
            tasks,
            instances_by_site=instances_by_site,
            request_context_factory=_factory,
        )
    )
    assert factory_calls == 1, "factory should run once per (site, storage_state)"
    assert len(keep) == 3 and dropped == []
    assert len(request_context.calls) == 3


def test_preflight_context_cache_is_race_safe_for_concurrent_same_key_tasks():
    tasks = [
        _make_task(f"adv_{i}", "gitlab", f"http://gitlab.test/p/-/issues/{i}") for i in range(20)
    ]
    instances_by_site = {"gitlab": [_gitlab_instance()]}
    request_context = _FakeRequestContext()
    factory_calls = 0

    async def _factory(_storage_state_path):
        nonlocal factory_calls
        factory_calls += 1
        await asyncio.sleep(0.01)
        return request_context

    keep, dropped = asyncio.run(
        preflight_benign_targets(
            tasks,
            instances_by_site=instances_by_site,
            request_context_factory=_factory,
        )
    )

    assert factory_calls == 1
    assert len(keep) == 20 and dropped == []


def test_preflight_serializes_shared_request_context_probes():
    class _ContentionContext(_FakeRequestContext):
        def __init__(self, *, response_map: dict[str, _FakeResponse]) -> None:
            super().__init__(response_map=response_map)
            self.active = 0
            self.max_active = 0

        async def get(self, url: str, *, timeout: float, max_redirects: int) -> _FakeResponse:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                await asyncio.sleep(0.001)
                return await super().get(url, timeout=timeout, max_redirects=max_redirects)
            finally:
                self.active -= 1

    tasks = [
        _make_task(f"adv_stale_{i}", "reddit", f"https://reddit.local/f/news/{i}") for i in range(3)
    ]
    instances_by_site = {"reddit": [{"site_name": "reddit", "site_url": "http://host0:9900"}]}
    response_map = {
        f"http://host0:9900/f/news/{task_index}": _FakeResponse(status=404)
        for task_index in range(3)
    }
    request_context = _ContentionContext(response_map=response_map)

    async def _factory(_):
        return request_context

    _keep, dropped = asyncio.run(
        preflight_benign_targets(
            tasks,
            instances_by_site=instances_by_site,
            request_context_factory=_factory,
            concurrency=16,
        )
    )

    assert [record["id"] for record in dropped] == [
        "adv_stale_0",
        "adv_stale_1",
        "adv_stale_2",
    ]
    assert request_context.max_active == 1


def test_preflight_transient_error_passes_task_through():
    task = _make_task("adv_transient", "gitlab", "http://gitlab.test/p/-/issues/1")
    instances_by_site = {"gitlab": [_gitlab_instance()]}
    request_context = _FakeRequestContext(
        response_map={
            "http://gitlab.test/p/-/issues/1": _FakeResponse(status=503, body=""),
        }
    )

    async def _factory(_):
        return request_context

    keep, dropped = asyncio.run(
        preflight_benign_targets(
            [task],
            instances_by_site=instances_by_site,
            request_context_factory=_factory,
        )
    )
    # 5xx is transient — task passes through, no quarantine.
    assert keep == [task] and dropped == []


def test_preflight_quarantines_stale_editor_surface_even_when_start_url_is_ok():
    task = _make_task("adv_comment", "reddit", "https://reddit.local/f/news")
    task["benign_target_resource"]["anchors"] = {
        "forum_name": "news",
        "submission_id": "5",
    }
    task["adversarial_data_seed"] = {
        "mechanism": "editor",
        "editor_calls": [
            {
                "site": "reddit",
                "method": "create_comment",
                "args": {
                    "forum_name": "{benign_forum_name}",
                    "submission_id": "{benign_submission_id}",
                    "body": "payload",
                },
            }
        ],
    }
    instances_by_site = {"reddit": [{"site_name": "reddit", "site_url": "http://reddit.test"}]}
    request_context = _FakeRequestContext(
        response_map={
            "http://reddit.test/f/news": _FakeResponse(status=200, body="forum ok"),
            "http://reddit.test/f/news/5": _FakeResponse(status=404),
        }
    )

    async def _factory(_context_options):
        return request_context

    keep, dropped = asyncio.run(
        preflight_benign_targets(
            [task],
            instances_by_site=instances_by_site,
            request_context_factory=_factory,
        )
    )

    assert keep == []
    assert [record["id"] for record in dropped] == ["adv_comment"]
    issue = dropped[0]["source_data_issue"]
    assert issue["kind"] == "not_found"
    assert issue["probed_url"] == "http://reddit.test/f/news/5"
    assert issue["probe_source"] == "editor_call[0].reddit.create_comment"
