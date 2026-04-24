from __future__ import annotations

from typing import Any

import pytest

from worldsim.editors.base import EditorError
from worldsim.editors.gitlab import GitlabEditor


def _editor() -> GitlabEditor:
    return GitlabEditor(
        {
            "site_url": "http://gitlab.test",
            "agent_auth": {"authentication": {"credentials": {"username": "current-user"}}},
        },
        session=None,
    )


def test_validate_args_rejects_missing_note_body_before_auth_lookup(monkeypatch):
    editor = _editor()
    monkeypatch.setattr(
        editor, "_current_user", lambda: (_ for _ in ()).throw(AssertionError("should not run"))
    )

    with pytest.raises(EditorError, match="missing required args: note_body"):
        editor.validate_args("create_mr_note", {})


def test_preview_context_for_mr_note_includes_preview_parent_ids():
    preview = _editor().preview_context(
        "create_mr_note",
        {
            "project_name_template": "webagent-task-{task_id}",
            "source_branch": "seed-branch",
        },
    )

    assert preview["project_id"] == 0
    assert preview["mr_iid"] == 1
    assert preview["project_path"].endswith("/webagent-task-task-id")


def test_create_project_reuses_project_created_in_same_editor_session(monkeypatch):
    editor = _editor()
    monkeypatch.setattr(editor, "_gitlab_get_json", lambda path, allow_missing=False: None)
    monkeypatch.setattr(editor, "_current_user", lambda: {"id": 1, "username": "current-user"})
    monkeypatch.setattr(editor, "_find_accessible_project", lambda **kwargs: None)
    monkeypatch.setattr(
        editor,
        "_gitlab_request_json",
        lambda method, path, **kwargs: {
            "id": 7,
            "path_with_namespace": "current-user/webagent-task-1",
        },
    )

    created = editor.create_project(name_template="webagent-task-1")
    result = editor.create_project(name_template="webagent-task-1")

    expected = {
        "project_id": 7,
        "project_path": "current-user/webagent-task-1",
        "read_surface_urls": [
            "http://gitlab.test/current-user/webagent-task-1",
            "/current-user/webagent-task-1",
        ],
        "read_surface_provenance_source": "editor_api_response",
    }
    assert created == expected
    # Re-creating a same-path project returns the cached copy, which now
    # includes read_surface_urls too (commit 2 of the C1 migration).
    assert result == expected


def test_create_project_classifies_duplicate_path_errors(monkeypatch):
    editor = _editor()
    monkeypatch.setattr(editor, "_gitlab_get_json", lambda path, allow_missing=False: None)
    monkeypatch.setattr(editor, "_current_user", lambda: {"id": 1, "username": "current-user"})
    monkeypatch.setattr(editor, "_find_accessible_project", lambda **kwargs: None)

    def _raise(*args, **kwargs):
        raise EditorError("request_failed", "Path has already been taken")

    monkeypatch.setattr(editor, "_gitlab_request_json", _raise)

    with pytest.raises(EditorError) as excinfo:
        editor.create_project(name_template="webagent-task-1")

    assert excinfo.value.kind == "project_already_exists"


def test_create_project_classifies_duplicate_path_errors_from_response_snippet(monkeypatch):
    editor = _editor()
    monkeypatch.setattr(editor, "_gitlab_get_json", lambda path, allow_missing=False: None)
    monkeypatch.setattr(editor, "_current_user", lambda: {"id": 1, "username": "current-user"})
    monkeypatch.setattr(editor, "_find_accessible_project", lambda **kwargs: None)

    def _raise(*args, **kwargs):
        raise EditorError(
            "request_failed",
            "gitlab editor request for /api/v4/projects returned HTTP 400",
            http_status=400,
            response_snippet='{"message":{"path":["has already been taken"]}}',
        )

    monkeypatch.setattr(editor, "_gitlab_request_json", _raise)

    with pytest.raises(EditorError) as excinfo:
        editor.create_project(name_template="webagent-verify-1")

    assert excinfo.value.kind == "project_already_exists"


def test_create_project_reaps_safe_project_after_duplicate_post_error(monkeypatch):
    editor = _editor()
    current_user = {"id": 1, "username": "current-user"}
    existing = {
        "id": 7,
        "path_with_namespace": "current-user/webagent-verify-1",
        "namespace": {"full_path": "current-user"},
    }
    seen = {"deleted": False, "post_calls": 0, "find_calls": 0}

    monkeypatch.setattr(editor, "_current_user", lambda: current_user)
    monkeypatch.setattr(editor, "_gitlab_get_json", lambda path, allow_missing=False: None)

    def fake_find(**kwargs):
        seen["find_calls"] += 1
        if seen["find_calls"] == 1:
            return None
        return existing

    monkeypatch.setattr(editor, "_find_accessible_project", fake_find)
    monkeypatch.setattr(
        editor,
        "delete_project",
        lambda project_id: seen.__setitem__("deleted", project_id == 7),
    )

    def fake_request(method, path, **kwargs):
        seen["post_calls"] += 1
        if seen["post_calls"] == 1:
            raise EditorError(
                "request_failed",
                "gitlab editor request for /api/v4/projects returned HTTP 400",
                http_status=400,
                response_snippet='{"message":{"path":["has already been taken"]}}',
            )
        return {"id": 11, "path_with_namespace": "current-user/webagent-verify-1"}

    monkeypatch.setattr(editor, "_gitlab_request_json", fake_request)

    created = editor.create_project(name_template="webagent-verify-1")

    assert seen["deleted"] is True
    assert seen["post_calls"] == 2
    assert created["project_id"] == 11


def test_create_project_rejects_preexisting_accessible_project(monkeypatch):
    editor = _editor()
    monkeypatch.setattr(editor, "_current_user", lambda: {"id": 1, "username": "current-user"})
    monkeypatch.setattr(
        editor,
        "_gitlab_get_json",
        lambda path, allow_missing=False: {
            "id": 7,
            "path_with_namespace": "current-user/webagent-task-1",
        },
    )

    with pytest.raises(EditorError) as excinfo:
        editor.create_project(name_template="webagent-task-1")

    assert excinfo.value.kind == "project_already_exists"


def test_create_project_reaps_stale_disposable_project_in_current_user_namespace(monkeypatch):
    editor = _editor()
    current_user = {"id": 1, "username": "current-user"}
    existing = {
        "id": 7,
        "path_with_namespace": "current-user/webagent-task-1",
        "namespace": {"full_path": "current-user"},
    }
    seen = {"deleted": False}

    def fake_get(path, allow_missing=False):
        if "current-user%2Fwebagent-task-1" not in path:
            return None
        if not seen["deleted"]:
            return existing
        return None

    monkeypatch.setattr(editor, "_current_user", lambda: current_user)
    monkeypatch.setattr(editor, "_gitlab_get_json", fake_get)
    monkeypatch.setattr(editor, "_find_accessible_project", lambda **kwargs: None)
    monkeypatch.setattr(
        editor, "delete_project", lambda project_id: seen.__setitem__("deleted", True)
    )
    monkeypatch.setattr(
        editor,
        "_gitlab_request_json",
        lambda method, path, **kwargs: {
            "id": 11,
            "path_with_namespace": "current-user/webagent-task-1",
        },
    )

    created = editor.create_project(name_template="webagent-task-1")

    assert created == {
        "project_id": 11,
        "project_path": "current-user/webagent-task-1",
        "read_surface_urls": [
            "http://gitlab.test/current-user/webagent-task-1",
            "/current-user/webagent-task-1",
        ],
        "read_surface_provenance_source": "editor_api_response",
    }


def test_create_project_reaps_stale_verification_project_in_current_user_namespace(monkeypatch):
    editor = _editor()
    current_user = {"id": 1, "username": "current-user"}
    existing = {
        "id": 7,
        "path_with_namespace": "current-user/webagent-verify-1",
        "namespace": {"full_path": "current-user"},
    }
    seen = {"deleted": False}

    def fake_get(path, allow_missing=False):
        if "current-user%2Fwebagent-verify-1" not in path:
            return None
        if not seen["deleted"]:
            return existing
        return None

    monkeypatch.setattr(editor, "_current_user", lambda: current_user)
    monkeypatch.setattr(editor, "_gitlab_get_json", fake_get)
    monkeypatch.setattr(editor, "_find_accessible_project", lambda **kwargs: None)
    monkeypatch.setattr(
        editor, "delete_project", lambda project_id: seen.__setitem__("deleted", True)
    )
    monkeypatch.setattr(
        editor,
        "_gitlab_request_json",
        lambda method, path, **kwargs: {
            "id": 11,
            "path_with_namespace": "current-user/webagent-verify-1",
        },
    )

    created = editor.create_project(name_template="webagent-verify-1")

    assert seen["deleted"] is True
    assert created["project_id"] == 11
    assert created["project_path"] == "current-user/webagent-verify-1"


def test_create_project_does_not_reap_non_disposable_preexisting_project(monkeypatch):
    editor = _editor()
    monkeypatch.setattr(editor, "_current_user", lambda: {"id": 1, "username": "current-user"})
    monkeypatch.setattr(
        editor,
        "_gitlab_get_json",
        lambda path, allow_missing=False: {
            "id": 7,
            "path_with_namespace": "current-user/important-project",
            "namespace": {"full_path": "current-user"},
        },
    )

    with pytest.raises(EditorError) as excinfo:
        editor.create_project(name_template="important-project")

    assert excinfo.value.kind == "project_already_exists"


def test_create_issue_does_not_reuse_existing_issue(monkeypatch):
    editor = _editor()
    monkeypatch.setattr(
        editor,
        "_ensure_project",
        lambda **kwargs: {
            "project_id": 174,
            "project_path": "current-user/webagent-task-1",
            "default_branch": "main",
        },
    )
    monkeypatch.setattr(
        editor,
        "_find_existing_issue",
        lambda **kwargs: {"iid": 41, "title": "Seeded", "description": "body"},
    )
    calls = []

    def fake_gitlab_request_json(method, path, **kwargs):
        calls.append((method, path, kwargs))
        return {"iid": 42, "web_url": "http://gitlab.test/current-user/webagent-task-1/-/issues/42"}

    monkeypatch.setattr(editor, "_gitlab_request_json", fake_gitlab_request_json)

    result = editor.create_issue(title_template="Seeded", body_template="body")

    assert result["issue_iid"] == 42
    assert calls == [
        (
            "POST",
            "/api/v4/projects/174/issues",
            {"json_body": {"title": "Seeded", "description": "body"}},
        )
    ]


def test_create_issue_note_does_not_reuse_existing_note(monkeypatch):
    editor = _editor()
    monkeypatch.setattr(
        editor,
        "_ensure_project",
        lambda **kwargs: {"project_id": 174, "project_path": "current-user/webagent-task-1"},
    )
    monkeypatch.setattr(editor, "_ensure_issue", lambda **kwargs: {"issue_iid": 7})
    monkeypatch.setattr(
        editor,
        "_find_existing_issue_note",
        lambda *args, **kwargs: {"id": 9, "body": "payload"},
    )
    created = []
    monkeypatch.setattr(editor, "_push_cleanup", lambda fn: None)

    def fake_gitlab_request_json(method, path, **kwargs):
        created.append((method, path, kwargs))
        return {"id": 10}

    monkeypatch.setattr(editor, "_gitlab_request_json", fake_gitlab_request_json)

    result = editor.create_issue_note(
        project_name_template="webagent-task-1",
        issue_title_template="Issue",
        note_body="payload",
    )

    assert result["note_id"] == 10
    assert created == [
        (
            "POST",
            "/api/v4/projects/174/issues/7/notes",
            {"json_body": {"body": "payload"}},
        )
    ]


def test_update_user_profile_falls_back_to_profile_form_when_api_missing(monkeypatch):
    editor = _editor()
    captured = {}
    state = {"name": "old name", "bio": "old bio"}

    def fake_gitlab_request_json(method, path, **kwargs):
        if method == "GET" and path == "/api/v4/user":
            return {"id": 42, "username": "current-user", **state}
        raise EditorError(
            "request_failed", "gitlab editor request for /api/v4/user returned HTTP 404"
        )

    monkeypatch.setattr(editor, "_gitlab_request_json", fake_gitlab_request_json)
    monkeypatch.setattr(
        editor,
        "_fetch_form_state",
        lambda *args, **kwargs: {
            "action": "/-/profile",
            "fields": {
                "_method": "put",
                "authenticity_token": "csrf-token",
                "user[id]": "42",
                "user[name]": "old name",
                "user[bio]": "old bio",
            },
        },
    )

    def fake_submit_exact_form(
        action_path, form_fields, *, multipart=False, refresh_on_rejection=None
    ):
        captured["action_path"] = action_path
        captured["form_fields"] = form_fields
        captured["multipart"] = multipart
        captured["refresh_on_rejection"] = refresh_on_rejection
        state["name"] = form_fields["user[name]"]
        state["bio"] = form_fields["user[bio]"]
        return {}

    monkeypatch.setattr(editor, "_submit_exact_form", fake_submit_exact_form)
    monkeypatch.setattr(editor, "_ensure_profile_form_session", lambda: None)

    assert editor.update_user_profile(name="new name", bio="new bio") == {}
    assert captured["action_path"] == "/-/profile"
    assert captured["form_fields"] == {
        "_method": "put",
        "authenticity_token": "csrf-token",
        "user[id]": "42",
        "user[name]": "new name",
        "user[bio]": "new bio",
    }
    assert captured["multipart"] is False
    assert callable(captured["refresh_on_rejection"])


def test_update_user_profile_raises_when_verified_state_does_not_change(monkeypatch):
    editor = _editor()
    state = {"id": 42, "username": "current-user", "name": "old name", "bio": "old bio"}

    def fake_gitlab_request_json(method, path, **kwargs):
        if method == "PUT" and path == "/api/v4/user":
            return {}
        if method == "GET" and path == "/api/v4/user":
            return dict(state)
        raise AssertionError((method, path, kwargs))

    monkeypatch.setattr(editor, "_gitlab_request_json", fake_gitlab_request_json)

    with pytest.raises(EditorError, match="did not persist field"):
        editor.update_user_profile(name="new name")

def test_update_user_profile_form_fallback_logs_in_with_seed_credentials():
    class _Response:
        def __init__(
            self,
            status_code: int,
            text: str = "",
            headers: dict[str, str] | None = None,
        ) -> None:
            self.status_code = status_code
            self.text = text
            self.headers = headers or {}

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

    class _Session:
        def __init__(self) -> None:
            self.login_post = None

        def get(self, url, headers=None, timeout=None, allow_redirects=True):
            assert url == "http://gitlab.test/users/sign_in"
            return _Response(
                200,
                '<input type="hidden" name="authenticity_token" value="csrf-token">',
            )

        def post(self, url, headers=None, data=None, timeout=None, allow_redirects=False):
            assert url == "http://gitlab.test/users/sign_in"
            self.login_post = {"headers": headers, "data": data, "allow_redirects": allow_redirects}
            return _Response(302, "", headers={"Location": "/"})

    editor = GitlabEditor(
        {
            "site_name": "gitlab",
            "site_url": "http://gitlab.test",
            "auth": {"credentials": {"username": "seed-user", "password": "seed-pass"}},
        },
        session=_Session(),
    )

    editor._ensure_profile_form_session()

    assert editor.session.login_post == {
        "headers": {},
        "data": {
            "authenticity_token": "csrf-token",
            "user[login]": "seed-user",
            "user[password]": "seed-pass",
        },
        "allow_redirects": False,
    }


def test_update_user_profile_form_login_rejects_redirect_back_to_login():
    class _Response:
        def __init__(self, status_code: int, text: str = "", headers: dict[str, str] | None = None) -> None:
            self.status_code = status_code
            self.text = text
            self.headers = headers or {}

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

    class _Session:
        def get(self, url, headers=None, timeout=None, allow_redirects=True):
            return _Response(
                200,
                '<input type="hidden" name="authenticity_token" value="csrf-token">',
            )

        def post(self, url, headers=None, data=None, timeout=None, allow_redirects=False):
            return _Response(302, "", headers={"Location": "/users/sign_in"})

    editor = GitlabEditor(
        {
            "site_name": "gitlab",
            "site_url": "http://gitlab.test",
            "auth": {"credentials": {"username": "seed-user", "password": "seed-pass"}},
        },
        session=_Session(),
    )

    with pytest.raises(EditorError, match="did not establish a session"):
        editor._ensure_profile_form_session()


def test_update_user_profile_form_fallback_requires_seed_credentials():
    editor = GitlabEditor(
        {
            "site_name": "gitlab",
            "site_url": "http://gitlab.test",
            "agent_auth": {"authentication": {"credentials": {"username": "browser-user"}}},
        },
        session=object(),
    )

    with pytest.raises(EditorError) as excinfo:
        editor._ensure_profile_form_session()

    assert excinfo.value.kind == "auth_missing"


# ---------------------------------------------------------------------------
# _validate_*_args — must accept project_path_template as a valid selector.
# GitLab editor methods declare project_path_template in their @editor_method
# bindings alongside project_id and project_name_template; _ensure_project
# consumes all three. The arg-validators were missing path_template from the
# _require_any_selector list, rejecting Option-A tasks (anchored to an
# existing benign project by path) with "project_id or project_name_template
# is required". Regression for adv-446 / adv_804 / adv_810.
# ---------------------------------------------------------------------------


def test_validate_issue_note_args_accepts_project_path_template():
    editor = _editor()
    # Must not raise — path_template is a valid project selector.
    editor._validate_issue_note_args(
        {
            "note_body": "hello",
            "project_path_template": "a11yproject/a11yproject.com",
            "issue_iid": "1478",
        }
    )


def test_validate_issue_note_args_still_rejects_when_no_project_selector():
    editor = _editor()
    with pytest.raises(EditorError, match="project_id"):
        editor._validate_issue_note_args(
            {
                "note_body": "hello",
                "issue_iid": "1478",
            }
        )


def test_validate_issue_args_accepts_project_path_template():
    editor = _editor()
    editor._validate_issue_args(
        {
            "title_template": "hello world",
            "project_path_template": "a11yproject/a11yproject.com",
        }
    )


def test_validate_merge_request_args_accepts_project_path_template():
    editor = _editor()
    editor._validate_merge_request_args(
        {
            "title_template": "hello",
            "source_branch": "feat/x",
            "project_path_template": "primer/design",
        }
    )


def test_validate_merge_request_note_args_accepts_project_path_template():
    editor = _editor()
    editor._validate_merge_request_note_args(
        {
            "note_body": "hello",
            "project_path_template": "primer/design",
            "mr_iid": "42",
        }
    )


def test_validate_repo_file_args_accepts_project_path_template():
    editor = _editor()
    editor._validate_repo_file_args(
        {
            "branch": "main",
            "path": "README.md",
            "content": "hi",
            "project_path_template": "primer/design",
        }
    )


def test_project_selectors_constant_is_canonical_three():
    # Guard against silent drift if new selectors are added without
    # updating the validators.
    assert set(_editor()._PROJECT_SELECTORS) == {
        "project_id",
        "project_name_template",
        "project_path_template",
    }


# ---------------------------------------------------------------------------
# _ensure_project path-template lookup. The @editor_method bindings,
# the method signatures, and the arg-validators all accept
# project_path_template as a valid project selector — but _ensure_project
# was only wired up to look up by project_id and fall through to
# create-on-demand via project_name_template. Option-A tasks anchored to
# an existing benign project carry path_template (no project_id on disk)
# and expected attach-by-path semantics, not create-on-demand.
# ---------------------------------------------------------------------------


def test_ensure_project_resolves_existing_project_by_path_template(monkeypatch):
    editor = _editor()
    captured_paths: list[str] = []

    def fake_get(path, *, allow_missing=False):
        captured_paths.append(path)
        return {
            "id": 42,
            "path_with_namespace": "a11yproject/a11yproject.com",
            "default_branch": "main",
        }

    monkeypatch.setattr(editor, "_gitlab_get_json", fake_get)
    # Must NOT call create_project when path_template resolves.
    monkeypatch.setattr(
        editor,
        "create_project",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("create_project must not run")),
    )

    result = editor._ensure_project(
        project_id=None,
        project_name_template=None,
        project_path_template="a11yproject/a11yproject.com",
        project_description_template=None,
    )
    assert result == {
        "project_id": 42,
        "project_path": "a11yproject/a11yproject.com",
        "default_branch": "main",
    }
    # URL-encoded — GitLab accepts namespace%2Fpath on /api/v4/projects/:id.
    assert captured_paths == ["/api/v4/projects/a11yproject%2Fa11yproject.com"]


def test_ensure_project_prefers_id_over_path_template(monkeypatch):
    """When both are present, project_id wins — numeric IDs are
    authoritative and cheaper to resolve (no URL encoding)."""
    editor = _editor()
    captured_paths: list[str] = []

    def fake_get(path, *, allow_missing=False):
        captured_paths.append(path)
        return {
            "id": 99,
            "path_with_namespace": "byteblaze/dotfiles",
            "default_branch": "main",
        }

    monkeypatch.setattr(editor, "_gitlab_get_json", fake_get)

    editor._ensure_project(
        project_id=99,
        project_name_template=None,
        project_path_template="different/path",
        project_description_template=None,
    )
    # Only the id lookup should fire; path_template lookup must not.
    assert captured_paths == ["/api/v4/projects/99"]


def test_ensure_project_falls_through_to_create_when_no_lookup_keys(monkeypatch):
    """With only project_name_template set, the method must create a
    new project — path_template lookup is optional, not a blocker."""
    editor = _editor()

    def fake_create(**kwargs):
        return {
            "project_id": 7,
            "project_path": kwargs["name_template"],
        }

    monkeypatch.setattr(editor, "create_project", fake_create)
    monkeypatch.setattr(
        editor,
        "_gitlab_get_json",
        lambda path, *, allow_missing=False: {
            "id": 7,
            "path_with_namespace": "user/webagent-task-x",
            "default_branch": "main",
        },
    )

    result = editor._ensure_project(
        project_id=None,
        project_name_template="webagent-task-x",
        project_path_template=None,
        project_description_template=None,
    )
    assert result["project_id"] == 7


def test_ensure_project_error_message_lists_all_three_selectors(monkeypatch):
    """Negative case: none of the three selectors populated → error
    mentions all three so the operator can fix the task shape."""
    editor = _editor()
    monkeypatch.setattr(editor, "_gitlab_get_json", lambda path, *, allow_missing=False: None)

    with pytest.raises(EditorError) as excinfo:
        editor._ensure_project(
            project_id=None,
            project_name_template=None,
            project_path_template=None,
            project_description_template=None,
        )
    msg = str(excinfo.value)
    assert "project_id" in msg
    assert "project_name_template" in msg
    assert "project_path_template" in msg


def test_ensure_project_falls_back_to_create_when_path_template_lookup_404s(monkeypatch):
    """If the path points at a project that doesn't exist on the live
    replica and name_template is also populated, fall through and
    create. This matches the existing behavior for project_id misses.
    """
    editor = _editor()
    # GET returns None (404 / not-a-dict) for the path lookup.
    monkeypatch.setattr(
        editor,
        "_gitlab_get_json",
        lambda path, *, allow_missing=False: None
        if "missing" in path
        else {"id": 5, "path_with_namespace": "x/fallback", "default_branch": "main"},
    )

    created: list[dict[str, Any]] = []

    def fake_create(**kwargs):
        created.append(kwargs)
        return {"project_id": 5, "project_path": "x/fallback"}

    monkeypatch.setattr(editor, "create_project", fake_create)

    result = editor._ensure_project(
        project_id=None,
        project_name_template="fallback-name",
        project_path_template="ns/missing",
        project_description_template=None,
    )
    assert created and created[0]["name_template"] == "fallback-name"
    assert result["project_id"] == 5
