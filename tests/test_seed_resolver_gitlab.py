from __future__ import annotations

import pytest
import requests

from worldsim.seed_resolvers import gitlab
from worldsim.seed_resolvers.types import ResolverError


def test_create_project_routes_to_api_endpoint():
    resolved = gitlab.create(
        {
            "mechanism": "api",
            "method": "POST",
            "resource_type": "project",
            "create": {"project": {"name_template": "webagent-task-1"}},
        },
        {"site_url": "http://gitlab.test"},
        {"task_id": "1"},
    )

    assert resolved.method == "POST"
    assert resolved.url == "http://gitlab.test/api/v4/projects"


def test_create_mr_note_uses_preview_parent_ids_in_dry_run():
    resolved = gitlab.create(
        {
            "mechanism": "api",
            "method": "POST",
            "resource_type": "mr_note",
            "create": {
                "project": {"name_template": "webagent-task-{task_id}"},
                "mr": {"title_template": "Seed MR {task_id}"},
            },
        },
        {"site_url": "http://gitlab.test", "seed_resolver_dry_run": True},
        {"task_id": "1"},
    )

    assert resolved.url == "http://gitlab.test/api/v4/projects/0/merge_requests/1/notes"
    assert resolved.context_additions["project_id"] == 0
    assert resolved.context_additions["mr_iid"] == 1


def test_create_issue_propagates_preview_issue_id_in_dry_run():
    resolved = gitlab.create(
        {
            "mechanism": "api",
            "method": "POST",
            "resource_type": "issue",
            "create": {
                "project": {"name_template": "webagent-task-{task_id}"},
                "issue": {"title_template": "Seed issue {task_id}"},
            },
        },
        {"site_url": "http://gitlab.test", "seed_resolver_dry_run": True},
        {"task_id": "1"},
    )

    assert resolved.context_additions["issue_iid"] == "1"


def test_create_mr_propagates_preview_merge_request_id_in_dry_run():
    resolved = gitlab.create(
        {
            "mechanism": "api",
            "method": "POST",
            "resource_type": "mr",
            "create": {
                "project": {"name_template": "webagent-task-{task_id}"},
                "mr": {"title_template": "Seed MR {task_id}"},
            },
        },
        {"site_url": "http://gitlab.test", "seed_resolver_dry_run": True},
        {"task_id": "1"},
    )

    assert resolved.context_additions["mr_iid"] == "1"


def test_update_routes_user_targets():
    profile = gitlab.update(
        {"resource_type": "user_profile"},
        {"site_url": "http://gitlab.test"},
        {},
    )
    status = gitlab.update(
        {"resource_type": "user_status"},
        {"site_url": "http://gitlab.test"},
        {},
    )

    assert profile.method == "PUT"
    assert profile.url == "http://gitlab.test/api/v4/user"
    assert status.method == "PUT"
    assert status.url == "http://gitlab.test/api/v4/user/status"


def test_update_rejects_unknown_resource_type():
    with pytest.raises(ResolverError, match="unsupported_resource_type"):
        gitlab.update(
            {"resource_type": "unknown"},
            {"site_url": "http://gitlab.test"},
            {},
        )


def test_create_mr_note_only_reuses_matching_merge_request(monkeypatch):
    responses = [
        [
            {
                "iid": 1,
                "title": "Seed MR 1",
                "description": "wrong description",
                "source_branch": "webagent-1",
                "target_branch": "main",
            }
        ],
        {"iid": 2, "title": "Seed MR 1"},
    ]

    def fake_request_json(instance, method, path, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr(gitlab, "_gitlab_request_json", fake_request_json)
    monkeypatch.setattr(gitlab, "_gitlab_get_json", lambda *args, **kwargs: {"id": 7, "path_with_namespace": "current-user/webagent-task-1", "default_branch": "main"})
    monkeypatch.setattr(gitlab, "_current_user", lambda instance: {"username": "current-user"})
    monkeypatch.setattr(gitlab, "_ensure_branch_and_commit", lambda *args, **kwargs: None)

    resolved = gitlab.create(
        {
            "mechanism": "api",
            "method": "POST",
            "resource_type": "mr_note",
            "create": {
                "project": {"name_template": "webagent-task-1"},
                "mr": {
                    "title_template": "Seed MR 1",
                    "body_template": "expected description",
                    "source_branch_template": "webagent-1",
                    "target_branch_template": "main",
                },
            },
        },
        {"site_url": "http://gitlab.test"},
        {"task_id": "1"},
    )

    assert resolved.url == "http://gitlab.test/api/v4/projects/7/merge_requests/2/notes"


def test_create_issue_note_uses_preview_parent_ids_in_dry_run():
    resolved = gitlab.create(
        {
            "mechanism": "api",
            "method": "POST",
            "resource_type": "issue_note",
            "create": {
                "project": {"name_template": "webagent-task-{task_id}"},
                "issue": {"title_template": "Seed issue {task_id}"},
            },
        },
        {"site_url": "http://gitlab.test", "seed_resolver_dry_run": True},
        {"task_id": "1"},
    )

    assert resolved.url == "http://gitlab.test/api/v4/projects/0/issues/1/notes"
    assert resolved.context_additions["project_id"] == 0
    assert resolved.context_additions["issue_iid"] == 1


def test_create_issue_note_prefers_issue_id_from_seed_context(monkeypatch):
    calls: list[str] = []

    def fake_get_json(instance, path, **kwargs):
        calls.append(path)
        if path == "/api/v4/projects/current-user%2Fwebagent-task-1":
            return {"id": 7, "path_with_namespace": "current-user/webagent-task-1", "default_branch": "main"}
        if path == "/api/v4/projects/7/issues/17":
            return {"iid": 17, "title": "Seed issue 1"}
        raise AssertionError(f"unexpected path: {path}")

    monkeypatch.setattr(gitlab, "_gitlab_get_json", fake_get_json)
    monkeypatch.setattr(gitlab, "_current_user", lambda instance: {"username": "current-user"})

    resolved = gitlab.create(
        {
            "mechanism": "api",
            "method": "POST",
            "resource_type": "issue_note",
            "create": {
                "project": {"name_template": "webagent-task-1"},
                "issue": {"title_template": "Seed issue 1"},
            },
        },
        {"site_url": "http://gitlab.test"},
        {"task_id": "1", "issue_iid": 17},
    )

    assert resolved.url == "http://gitlab.test/api/v4/projects/7/issues/17/notes"
    assert calls[-1] == "/api/v4/projects/7/issues/17"


def test_create_mr_note_prefers_merge_request_id_from_seed_context(monkeypatch):
    calls: list[str] = []

    def fake_get_json(instance, path, **kwargs):
        calls.append(path)
        if path == "/api/v4/projects/current-user%2Fwebagent-task-1":
            return {"id": 7, "path_with_namespace": "current-user/webagent-task-1", "default_branch": "main"}
        if path == "/api/v4/projects/7/merge_requests/23":
            return {"iid": 23, "title": "Seed MR 1"}
        raise AssertionError(f"unexpected path: {path}")

    monkeypatch.setattr(gitlab, "_gitlab_get_json", fake_get_json)
    monkeypatch.setattr(gitlab, "_current_user", lambda instance: {"username": "current-user"})

    resolved = gitlab.create(
        {
            "mechanism": "api",
            "method": "POST",
            "resource_type": "mr_note",
            "create": {
                "project": {"name_template": "webagent-task-1"},
                "mr": {
                    "title_template": "Seed MR 1",
                    "body_template": "Seed MR context for task 1.",
                    "source_branch_template": "webagent-1",
                    "target_branch_template": "main",
                },
            },
        },
        {"site_url": "http://gitlab.test"},
        {"task_id": "1", "mr_iid": 23},
    )

    assert resolved.url == "http://gitlab.test/api/v4/projects/7/merge_requests/23/notes"
    assert calls[-1] == "/api/v4/projects/7/merge_requests/23"


def test_create_group_routes_to_group_endpoint():
    resolved = gitlab.create(
        {
            "mechanism": "api",
            "method": "POST",
            "resource_type": "group",
            "create": {"group": {"name_template": "seed-group", "path_template": "seed-group"}},
        },
        {"site_url": "http://gitlab.test"},
        {"task_id": "1"},
    )

    assert resolved.method == "POST"
    assert resolved.url == "http://gitlab.test/api/v4/groups"


def test_create_repo_file_routes_to_commit_endpoint_in_preview():
    resolved = gitlab.create(
        {
            "mechanism": "api",
            "method": "POST",
            "resource_type": "repo_file",
            "create": {"project": {"name_template": "webagent-task-{task_id}"}},
        },
        {"site_url": "http://gitlab.test", "seed_resolver_dry_run": True},
        {"task_id": "1"},
    )

    assert resolved.method == "POST"
    assert resolved.url == "http://gitlab.test/api/v4/projects/0/repository/commits"


def test_ensure_branch_and_commit_raises_on_unexpected_400(monkeypatch):
    response = requests.Response()
    response.status_code = 400
    response._content = b'{"message":"invalid ref"}'
    response.url = "http://gitlab.test/api/v4/projects/7/repository/branches"
    error = requests.HTTPError("bad request", response=response)

    monkeypatch.setattr(
        gitlab,
        "_gitlab_request_json",
        lambda instance, method, path, **kwargs: (_ for _ in ()).throw(error),
    )

    with pytest.raises(requests.HTTPError):
        gitlab._ensure_branch_and_commit(
            {"id": 7, "default_branch": "main"},
            {"source_branch": "seed-1", "target_branch": "main"},
            {"site_url": "http://gitlab.test"},
        )


def test_ensure_branch_and_commit_short_circuits_when_marker_file_matches(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(
        gitlab,
        "_gitlab_get_file_content",
        lambda instance, **kwargs: "Seed branch for seed-1\n",
    )

    def fake_request_json(instance, method, path, **kwargs):
        calls.append(path)
        if path.endswith("/repository/branches"):
            return {}
        raise AssertionError("should not write commit")

    monkeypatch.setattr(gitlab, "_gitlab_request_json", fake_request_json)

    gitlab._ensure_branch_and_commit(
        {"id": 7, "default_branch": "main"},
        {"source_branch": "seed-1", "target_branch": "main"},
        {"site_url": "http://gitlab.test"},
    )

    assert calls == ["/api/v4/projects/7/repository/branches"]
