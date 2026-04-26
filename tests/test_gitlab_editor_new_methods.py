"""Unit tests for the six new gitlab Mode B editor methods.

Each method addresses a Phase 2 kind that previously had no editor
attach point:

- ``update_user_profile_by_id`` -> ``gitlab_user_profile`` (bio surface)
- ``update_user_status_by_id`` -> ``gitlab_user_profile`` (status surface)
- ``update_snippet`` -> ``gitlab_snippet``
- ``create_snippet`` -> ``gitlab_snippets_index``
- ``update_milestone`` -> ``gitlab_project_milestone``
- ``create_label`` -> ``gitlab_project_labels``
- ``update_group`` -> ``gitlab_group``

Update methods MUST capture the original field value and register a
cleanup that restores it; create methods MUST register a delete cleanup.
Without cleanup, Phase 2c idempotency breaks across tasks on the live
r5 stack.
"""

from __future__ import annotations

from typing import Any

from worldsim.editors.gitlab import GitlabEditor


def _gitlab_editor() -> GitlabEditor:
    return GitlabEditor(
        {
            "site_url": "http://gitlab.test",
            "agent_auth": {"authentication": {"credentials": {"username": "byteblaze"}}},
        },
        session=None,  # type: ignore[arg-type]
    )


def _record_calls(editor: GitlabEditor, monkeypatch, *, responses: dict[tuple[str, str], Any]):
    """Stub _gitlab_request_json with deterministic per-(method, path-prefix) responses."""
    calls: list[tuple[str, str, dict[str, Any]]] = []

    def fake(method: str, path: str, *, json_body: dict | None = None, **kwargs):
        calls.append((method, path, dict(json_body or {})))
        # Prefer the longest matching prefix so /api/v4/users/5/status wins
        # over /api/v4/users.
        best_match: tuple[int, Any] | None = None
        for (m, prefix), resp in responses.items():
            if m == method and path.startswith(prefix):
                if best_match is None or len(prefix) > best_match[0]:
                    best_match = (len(prefix), resp)
        return best_match[1] if best_match else None

    monkeypatch.setattr(editor, "_gitlab_request_json", fake)
    return calls


# ---------------------------------------------------------------------------
# update_user_profile_by_id
# ---------------------------------------------------------------------------


def test_update_user_profile_by_id_captures_and_restores_bio(monkeypatch):
    editor = _gitlab_editor()
    calls = _record_calls(
        editor,
        monkeypatch,
        responses={
            ("GET", "/api/v4/users"): [{"id": 2, "username": "byteblaze", "bio": "old bio"}],
            ("PUT", "/api/v4/users/2"): {"id": 2, "bio": "new bio"},
        },
    )

    result = editor.update_user_profile_by_id(username="byteblaze", bio="new bio")

    assert any(
        method == "PUT" and path == "/api/v4/users/2" and body == {"bio": "new bio"}
        for method, path, body in calls
    )
    assert result["read_surface_provenance_source"] == "editor_constructed"
    assert any("byteblaze" in url for url in result["read_surface_urls"])

    # Run cleanup: must restore original bio.
    editor.cleanup()
    restore_calls = [(m, p, b) for m, p, b in calls if b.get("bio") == "old bio"]
    assert restore_calls, f"cleanup did not restore original bio; calls={calls}"


def test_update_user_profile_by_id_rejects_blank_username():
    editor = _gitlab_editor()
    try:
        editor.update_user_profile_by_id(username="  ", bio="x")
    except Exception as exc:
        assert "username is required" in str(exc)
    else:
        raise AssertionError("expected EditorError for blank username")


# ---------------------------------------------------------------------------
# update_user_status_by_id
# ---------------------------------------------------------------------------


def test_update_user_status_by_id_captures_and_restores_status(monkeypatch):
    editor = _gitlab_editor()
    calls = _record_calls(
        editor,
        monkeypatch,
        responses={
            ("GET", "/api/v4/users"): [{"id": 5, "username": "abi"}],
            ("GET", "/api/v4/users/5/status"): {"message": "old", "emoji": "wave"},
            ("PUT", "/api/v4/users/5/status"): {"message": "new"},
        },
    )

    editor.update_user_status_by_id(username="abi", message="new", emoji="rocket")

    put_calls = [body for method, path, body in calls if method == "PUT" and "/status" in path]
    assert {"message": "new", "emoji": "rocket"} in put_calls

    editor.cleanup()
    put_calls = [body for method, path, body in calls if method == "PUT" and "/status" in path]
    restore_calls = [body for body in put_calls if body.get("message") == "old"]
    assert restore_calls, f"cleanup did not restore original status; calls={calls}"


# ---------------------------------------------------------------------------
# update_snippet
# ---------------------------------------------------------------------------


def test_update_snippet_captures_original_content_and_restores(monkeypatch):
    editor = _gitlab_editor()
    calls = _record_calls(
        editor,
        monkeypatch,
        responses={
            ("GET", "/api/v4/snippets/42"): {
                "id": 42,
                "title": "old title",
                "description": "old desc",
            },
            ("PUT", "/api/v4/snippets/42"): {"id": 42},
        },
    )
    monkeypatch.setattr(editor, "_gitlab_snippet_raw_content", lambda sid: "old content")

    result = editor.update_snippet(snippet_id="42", content="new content", title="new title")

    put_bodies = [body for method, path, body in calls if method == "PUT"]
    assert any(b.get("content") == "new content" for b in put_bodies)
    assert any(
        ".../snippets/42" in url or "/snippets/42" in url for url in result["read_surface_urls"]
    )

    editor.cleanup()
    put_bodies = [body for method, path, body in calls if method == "PUT"]
    restored = [b for b in put_bodies if b.get("content") == "old content"]
    assert restored, f"cleanup did not restore original snippet content; bodies={put_bodies}"


# ---------------------------------------------------------------------------
# create_snippet
# ---------------------------------------------------------------------------


def test_create_snippet_registers_delete_cleanup(monkeypatch):
    editor = _gitlab_editor()

    api_responses: dict[tuple[str, str], Any] = {
        ("POST", "/api/v4/snippets"): {"id": 99, "web_url": "http://gitlab.test/-/snippets/99"},
    }
    json_calls = _record_calls(editor, monkeypatch, responses=api_responses)

    deleted: list[str] = []

    def fake_resp(method, path, **kwargs):
        if method == "DELETE":
            deleted.append(path)
        return None

    monkeypatch.setattr(editor, "_api_request_response", fake_resp)

    result = editor.create_snippet(
        title="Released by WorldSim",
        content="placeholder content",
        description="seed",
    )

    assert result["snippet_id"] == 99
    assert any(method == "POST" for method, _, _ in json_calls)
    assert any("/-/snippets/99" in url for url in result["read_surface_urls"])

    editor.cleanup()
    assert any("/api/v4/snippets/99" in p for p in deleted), (
        f"cleanup did not DELETE the created snippet; deleted={deleted}"
    )


# ---------------------------------------------------------------------------
# update_milestone
# ---------------------------------------------------------------------------


def test_update_milestone_captures_and_restores_description(monkeypatch):
    editor = _gitlab_editor()
    calls = _record_calls(
        editor,
        monkeypatch,
        responses={
            ("GET", "/api/v4/projects/byteblaze%2Fdotfiles"): {
                "id": 4,
                "path_with_namespace": "byteblaze/dotfiles",
            },
            ("GET", "/api/v4/projects/4/milestones"): [
                {"id": 11, "iid": 1, "title": "v1", "description": "old desc"}
            ],
            ("PUT", "/api/v4/projects/4/milestones/11"): {"id": 11},
        },
    )

    result = editor.update_milestone(
        milestone_iid="1",
        description="new desc",
        project_path_template="byteblaze/dotfiles",
    )

    put_bodies = [body for method, path, body in calls if method == "PUT"]
    assert any(b.get("description") == "new desc" for b in put_bodies)
    assert result["milestone_id"] == 11
    assert result["project_path"] == "byteblaze/dotfiles"

    editor.cleanup()
    put_bodies = [body for method, path, body in calls if method == "PUT"]
    restored = [b for b in put_bodies if b.get("description") == "old desc"]
    assert restored, "cleanup did not restore original milestone description"


# ---------------------------------------------------------------------------
# create_label
# ---------------------------------------------------------------------------


def test_create_label_registers_delete_cleanup(monkeypatch):
    editor = _gitlab_editor()
    json_calls = _record_calls(
        editor,
        monkeypatch,
        responses={
            ("GET", "/api/v4/projects/primer%2Fdesign"): {
                "id": 7,
                "path_with_namespace": "primer/design",
            },
            ("POST", "/api/v4/projects/7/labels"): {"id": 33, "name": "WorldSim Label"},
        },
    )

    deleted: list[str] = []

    def fake_resp(method, path, **kwargs):
        if method == "DELETE":
            deleted.append(path)
        return None

    monkeypatch.setattr(editor, "_api_request_response", fake_resp)

    result = editor.create_label(
        name="WorldSim Label",
        description="seed",
        project_path_template="primer/design",
    )

    assert result["label_id"] == 33
    assert any(method == "POST" for method, _, _ in json_calls)

    editor.cleanup()
    assert any("/projects/7/labels/33" in p for p in deleted), (
        f"cleanup did not DELETE the created label; deleted={deleted}"
    )


# ---------------------------------------------------------------------------
# update_group
# ---------------------------------------------------------------------------


def test_update_group_captures_and_restores_description(monkeypatch):
    editor = _gitlab_editor()
    calls = _record_calls(
        editor,
        monkeypatch,
        responses={
            ("GET", "/api/v4/groups/a11yproject"): {
                "id": 12,
                "full_path": "a11yproject",
                "description": "old desc",
            },
            ("PUT", "/api/v4/groups/12"): {"id": 12},
        },
    )

    result = editor.update_group(group_path="a11yproject", description="new desc")

    put_bodies = [body for method, path, body in calls if method == "PUT"]
    assert any(b.get("description") == "new desc" for b in put_bodies)
    assert result["group_id"] == 12

    editor.cleanup()
    put_bodies = [body for method, path, body in calls if method == "PUT"]
    restored = [b for b in put_bodies if b.get("description") == "old desc"]
    assert restored, "cleanup did not restore original group description"


# ---------------------------------------------------------------------------
# Decorator wiring sanity
# ---------------------------------------------------------------------------


def test_all_new_methods_register_kinds():
    """Each new method must address exactly the kind the resolver will emit."""
    spec_kinds = {
        "update_user_profile_by_id": {"gitlab_user_profile"},
        "update_user_status_by_id": {"gitlab_user_profile"},
        "update_snippet": {"gitlab_snippet"},
        "create_snippet": {"gitlab_snippets_index"},
        "update_milestone": {"gitlab_project_milestone"},
        "create_label": {"gitlab_project_labels"},
        "update_group": {"gitlab_group"},
    }
    for method_name, expected in spec_kinds.items():
        method = getattr(GitlabEditor, method_name)
        spec = getattr(method, "_editor_method_spec", None)
        assert spec is not None, f"{method_name} missing @editor_method spec"
        assert set(spec["kinds"]) == expected, (
            f"{method_name} addresses {set(spec['kinds'])}, expected {expected}"
        )
