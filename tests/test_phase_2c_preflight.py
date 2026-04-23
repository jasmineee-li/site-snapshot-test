"""Unit tests for :mod:`worldsim.phases.phase_2c_preflight`.

Bug I (2026-04-23): pre-seed HTTP probe + source-data quarantine.
These tests exercise the classifier rules + the whole-run bailout.
"""

from __future__ import annotations

import asyncio
from typing import Any

from worldsim.phases.phase_2c_preflight import (
    _classify_probe,
    _location_is_login,
    _looks_like_login_stub,
    preflight_benign_targets,
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
# _classify_probe — table-driven
# ---------------------------------------------------------------------------


def test_classify_200_clean_reachable():
    c = _classify_probe(
        status=200, headers={}, body_snippet="<html>welcome</html>", exception_name=None
    )
    assert c.kind == "reachable" and not c.quarantine


def test_classify_200_login_stub_short_body():
    c = _classify_probe(
        status=200,
        headers={},
        body_snippet="please sign in to continue" * 2,
        exception_name=None,
    )
    assert c.kind == "login_redirect" and c.quarantine


def test_classify_200_login_stub_form_marker():
    c = _classify_probe(
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
    c = _classify_probe(
        status=302,
        headers={"location": "http://host/users/sign_in"},
        body_snippet="",
        exception_name=None,
    )
    assert c.kind == "login_redirect" and c.quarantine


def test_classify_302_non_login_is_redirect_noncritical():
    c = _classify_probe(
        status=301,
        headers={"location": "/projects/foo/bar"},
        body_snippet="",
        exception_name=None,
    )
    assert c.kind == "redirect_noncritical" and not c.quarantine


def test_classify_404_quarantines_as_not_found():
    c = _classify_probe(status=404, headers={}, body_snippet="", exception_name=None)
    assert c.kind == "not_found" and c.quarantine


def test_classify_403_quarantines_as_forbidden():
    c = _classify_probe(status=403, headers={}, body_snippet="", exception_name=None)
    assert c.kind == "forbidden" and c.quarantine


def test_classify_401_quarantines_as_auth_missing():
    c = _classify_probe(status=401, headers={}, body_snippet="", exception_name=None)
    assert c.kind == "auth_missing" and c.quarantine


def test_classify_410_quarantines_as_gone():
    c = _classify_probe(status=410, headers={}, body_snippet="", exception_name=None)
    assert c.kind == "gone" and c.quarantine


def test_classify_429_does_not_quarantine():
    c = _classify_probe(status=429, headers={}, body_snippet="", exception_name=None)
    assert c.kind == "rate_limited" and not c.quarantine


def test_classify_500_does_not_quarantine():
    c = _classify_probe(status=503, headers={}, body_snippet="", exception_name=None)
    assert c.kind == "server_error" and not c.quarantine


def test_classify_connection_error_does_not_quarantine():
    c = _classify_probe(
        status=None,
        headers=None,
        body_snippet="",
        exception_name="ConnectionRefusedError",
    )
    assert c.kind == "host_unreachable" and not c.quarantine


def test_classify_timeout_does_not_quarantine():
    c = _classify_probe(status=None, headers=None, body_snippet="", exception_name="TimeoutError")
    assert c.kind == "probe_timeout" and not c.quarantine


def test_location_is_login_helper():
    assert _location_is_login("http://host/users/sign_in?foo=1")
    assert _location_is_login("/login")
    assert not _location_is_login("/projects/foo/issues/1")
    assert not _location_is_login(None)


def test_looks_like_login_stub_boundaries():
    # Short body + "sign in" → stub.
    assert _looks_like_login_stub("please sign in")
    # Long prose body mentioning "sign in" → NOT stub (body_len gate).
    long_body = "This README mentions how to sign in for local dev." + (" " * 700)
    assert not _looks_like_login_stub(long_body)
    # Form-action marker → stub regardless of body length.
    assert _looks_like_login_stub("x" * 2000 + '<form action="/users/sign_in"')


# ---------------------------------------------------------------------------
# preflight_benign_targets — integration via fakes
# ---------------------------------------------------------------------------


def _make_task(task_id: str, site: str, start_url: str) -> dict[str, Any]:
    return {
        "id": task_id,
        "site": site,
        "benign_target_resource": {
            "kind": "gitlab_issue",
            "start_url_resolved": start_url,
        },
    }


def test_preflight_splits_quarantine_from_keep():
    keep_task = _make_task("adv_keep", "gitlab", "http://gitlab.test/foo/-/issues/1")
    drop_task = _make_task("adv_drop", "gitlab", "http://gitlab.test/foo/-/merge_requests/2")
    instances_by_site = {"gitlab": [{"site_name": "gitlab", "site_url": "http://gitlab.test"}]}
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


def test_preflight_bailout_when_login_redirect_dominates():
    # >50 % login_redirect → mass-restore + skip quarantine entirely.
    tasks = [
        _make_task(f"adv_{i}", "gitlab", f"http://gitlab.test/p/-/issues/{i}") for i in range(6)
    ]
    instances_by_site = {"gitlab": [{"site_name": "gitlab", "site_url": "http://gitlab.test"}]}
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
    instances_by_site = {"gitlab": [{"site_name": "gitlab", "site_url": "http://172.17.0.1:8023"}]}
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


def test_preflight_reuses_request_context_across_same_site_tasks():
    # Memoization invariant: tasks on the same (site, storage_state_path)
    # pair must share a single APIRequestContext so we do not pay the
    # TLS+cookie setup cost per task. The factory is recorded to verify.
    tasks = [
        _make_task(f"adv_{i}", "gitlab", f"http://gitlab.test/p/-/issues/{i}") for i in range(3)
    ]
    instances_by_site = {"gitlab": [{"site_name": "gitlab", "site_url": "http://gitlab.test"}]}
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


def test_preflight_transient_error_passes_task_through():
    task = _make_task("adv_transient", "gitlab", "http://gitlab.test/p/-/issues/1")
    instances_by_site = {"gitlab": [{"site_name": "gitlab", "site_url": "http://gitlab.test"}]}
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
