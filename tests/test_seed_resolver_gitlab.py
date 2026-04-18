from __future__ import annotations

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


def test_update_user_profile_falls_back_to_profile_form_when_api_missing(monkeypatch):
    editor = _editor()
    captured = {}

    def fake_gitlab_request_json(method, path, **kwargs):
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


def test_update_user_profile_form_fallback_logs_in_with_seed_credentials():
    class _Response:
        def __init__(self, status_code: int, text: str = "") -> None:
            self.status_code = status_code
            self.text = text

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
            return _Response(302, "")

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
