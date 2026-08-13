from __future__ import annotations

import types

import pytest

from warp_taskgen.editors.base import EditorError
from warp_taskgen.editors.reddit import RedditEditor


def _editor(instance: dict | None = None) -> RedditEditor:
    return RedditEditor(
        instance or {"site_url": "http://reddit.test"}, session=types.SimpleNamespace()
    )


def test_resolve_current_username_uses_agent_auth_fallback():
    username = RedditEditor._resolve_current_username(
        {
            "agent_auth": {
                "authentication": {
                    "credentials": {"username": "reddit-user"},
                }
            }
        }
    )

    assert username == "reddit-user"


def test_validate_args_requires_forum_name_for_comments():
    with pytest.raises(EditorError, match="missing required args: forum_name"):
        _editor(
            {"site_url": "http://reddit.test", "agent_auth": {"credentials": {"username": "user"}}}
        ).validate_args(
            "create_comment",
            {"submission_id": "123", "body": "payload"},
        )


def test_create_forum_allocates_fresh_name_when_requested_name_exists(monkeypatch):
    editor = _editor(
        {"site_url": "http://reddit.test", "agent_auth": {"credentials": {"username": "user"}}}
    )
    existing = {"books"}
    monkeypatch.setattr(editor, "_forum_exists", lambda forum_name: forum_name in existing)
    monkeypatch.setattr(
        editor,
        "_fetch_form_state",
        lambda *args, **kwargs: {
            "action": "",
            "fields": {
                "forum[name]": "",
                "forum[title]": "",
                "forum[description]": "",
                "forum[sidebar]": "",
                "forum[_token]": "csrf-token",
            },
        },
    )
    captured = {}

    def fake_submit_exact_form(
        action_path, form_fields, *, multipart=False, refresh_on_rejection=None
    ):
        captured["action_path"] = action_path
        captured["form_fields"] = form_fields
        existing.add(form_fields["forum[name]"])
        return {}

    monkeypatch.setattr(editor, "_submit_exact_form", fake_submit_exact_form)

    result = editor.create_forum(name_template="books")

    # Collision suffix uses ``_`` (not ``-``) because Postmill silently
    # rejects POST /create_forum submissions whose forum[name] contains
    # a dash — see warp_taskgen/editors/reddit.py::_allocate_forum_name.
    assert result == {
        "forum_name": "books_2",
        "read_surface_urls": ["http://reddit.test/f/books_2", "/f/books_2"],
        "read_surface_provenance_source": "editor_constructed",
    }
    assert captured["action_path"] == "/create_forum"
    assert captured["form_fields"]["forum[name]"] == "books_2"


def test_create_forum_posts_exact_form_fields(monkeypatch):
    editor = _editor(
        {"site_url": "http://reddit.test", "agent_auth": {"credentials": {"username": "user"}}}
    )
    captured = {}
    existing = set()

    monkeypatch.setattr(editor, "_forum_exists", lambda forum_name: forum_name in existing)
    monkeypatch.setattr(
        editor,
        "_fetch_form_state",
        lambda *args, **kwargs: {
            "action": "",
            "fields": {
                "forum[name]": "",
                "forum[title]": "",
                "forum[description]": "",
                "forum[sidebar]": "",
                "forum[_token]": "csrf-token",
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
        existing.add(form_fields["forum[name]"])
        return {}

    monkeypatch.setattr(editor, "_submit_exact_form", fake_submit_exact_form)

    assert editor.create_forum(name_template="books", description_template="A forum") == {
        "forum_name": "books",
        "read_surface_urls": ["http://reddit.test/f/books", "/f/books"],
        "read_surface_provenance_source": "editor_constructed",
    }
    assert captured["action_path"] == "/create_forum"
    assert captured["form_fields"] == {
        "forum[name]": "books",
        "forum[title]": "books",
        "forum[description]": "A forum",
        "forum[sidebar]": "A forum",
        "forum[_token]": "csrf-token",
    }
    assert captured["multipart"] is False
    assert callable(captured["refresh_on_rejection"])


def test_create_forum_raises_when_created_forum_is_not_visible_after_submit(monkeypatch):
    editor = _editor(
        {"site_url": "http://reddit.test", "agent_auth": {"credentials": {"username": "user"}}}
    )
    monkeypatch.setattr(editor, "_forum_exists", lambda forum_name: False)
    monkeypatch.setattr(
        editor,
        "_fetch_form_state",
        lambda *args, **kwargs: {
            "action": "",
            "fields": {
                "forum[name]": "",
                "forum[title]": "",
                "forum[description]": "",
                "forum[sidebar]": "",
                "forum[_token]": "csrf-token",
            },
        },
    )
    monkeypatch.setattr(editor, "_submit_exact_form", lambda *args, **kwargs: {})

    with pytest.raises(EditorError, match="did not make 'books' accessible"):
        editor.create_forum(name_template="books")


def test_preview_context_for_submission_includes_submission_id():
    preview = _editor({"site_url": "http://reddit.test"}).preview_context(
        "create_submission",
        {"forum_name": "books", "title_template": "Thread"},
    )

    assert preview["forum_name"] == "books"
    assert preview["submission_id"] == "1"
    assert preview["submission_url"] == "http://reddit.test/f/books/1"


def test_create_submission_backfills_required_media_type(monkeypatch):
    editor = _editor(
        {"site_url": "http://reddit.test", "agent_auth": {"credentials": {"username": "user"}}}
    )
    captured = {}

    monkeypatch.setattr(
        editor,
        "_fetch_form_state",
        lambda *args, **kwargs: {
            "action": "",
            "fields": {
                "submission[_token]": "csrf-token",
                "submission[forum]": "",
                "submission[title]": "",
                "submission[body]": "",
            },
            "select_options": {
                "submission[forum]": [
                    {"data-name": "books", "value": "7"},
                ]
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
        return types.SimpleNamespace(headers={"Location": "/f/books/123"}, json=lambda: {})

    monkeypatch.setattr(editor, "_submit_exact_form", fake_submit_exact_form)

    result = editor.create_submission(
        forum_name="books",
        title_template="Thread",
        body_template="Body",
    )

    assert result == {
        "forum_name": "books",
        "submission_id": "123",
        "submission_url": "http://reddit.test/f/books/123",
        "created_resource": {
            "role": "seed_render_surface",
            "kind": "submission",
            "id": "123",
            "url": "http://reddit.test/f/books/123",
            "parent_url": "http://reddit.test/f/books",
        },
        "read_surface_urls": [
            "http://reddit.test/f/books/123",
            "/f/books/123",
        ],
        "read_surface_provenance_source": "editor_constructed",
    }
    assert captured["action_path"] == "/submit/books"
    assert captured["form_fields"] == {
        "submission[_token]": "csrf-token",
        "submission[forum]": "7",
        "submission[title]": "Thread",
        "submission[body]": "Body",
        "submission[mediaType]": "url",
    }
    assert captured["multipart"] is False
    assert callable(captured["refresh_on_rejection"])


def test_create_comment_posts_exact_form_fields(monkeypatch):
    editor = _editor(
        {"site_url": "http://reddit.test", "agent_auth": {"credentials": {"username": "user"}}}
    )
    captured = {}

    monkeypatch.setattr(
        editor,
        "_fetch_form_state",
        lambda *args, **kwargs: {
            "action": "/f/books/123/-/comment",
            "fields": {
                "reply_to_submission_123[_token]": "csrf-token",
                "reply_to_submission_123[comment]": "",
                "reply_to_submission_123[email]": "",
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
        return types.SimpleNamespace(json=lambda: {"id": 9})

    monkeypatch.setattr(editor, "_submit_exact_form", fake_submit_exact_form)

    result = editor.create_comment(
        forum_name="books",
        submission_id="123",
        body="Comment body",
    )

    assert result == {
        "forum_name": "books",
        "submission_id": "123",
        "comment_id": "9",
        "read_surface_urls": [
            "http://reddit.test/f/books/123#comment_9",
            "/f/books/123#comment_9",
            "http://reddit.test/f/books/123",
            "/f/books/123",
        ],
        "read_surface_provenance_source": "editor_constructed",
    }
    assert captured["action_path"] == "/f/books/123/-/comment"
    assert captured["form_fields"] == {
        "reply_to_submission_123[_token]": "csrf-token",
        "reply_to_submission_123[comment]": "Comment body",
        "reply_to_submission_123[email]": "",
    }
    assert captured["multipart"] is False
    assert callable(captured["refresh_on_rejection"])


def test_create_comment_extracts_comment_id_from_redirect(monkeypatch):
    editor = _editor(
        {"site_url": "http://reddit.test", "agent_auth": {"credentials": {"username": "user"}}}
    )
    monkeypatch.setattr(
        editor,
        "_fetch_form_state",
        lambda *args, **kwargs: {
            "action": "/f/books/123/-/comment",
            "fields": {
                "reply_to_submission_123[_token]": "csrf-token",
                "reply_to_submission_123[comment]": "",
            },
        },
    )
    monkeypatch.setattr(
        editor,
        "_submit_exact_form",
        lambda *args, **kwargs: types.SimpleNamespace(
            headers={"Location": "/f/books/123#comment-901"},
            json=lambda: (_ for _ in ()).throw(ValueError("not json")),
        ),
    )

    result = editor.create_comment(forum_name="books", submission_id="123", body="body")

    assert result["comment_id"] == "901"
    assert result["read_surface_urls"][:2] == [
        "http://reddit.test/f/books/123#comment_901",
        "/f/books/123#comment_901",
    ]


def test_create_comment_extracts_comment_id_from_rendered_comment_block(monkeypatch):
    editor = _editor(
        {"site_url": "http://reddit.test", "agent_auth": {"credentials": {"username": "user"}}}
    )
    monkeypatch.setattr(
        editor,
        "_fetch_form_state",
        lambda *args, **kwargs: {
            "action": "/f/books/123/-/comment",
            "fields": {
                "reply_to_submission_123[_token]": "csrf-token",
                "reply_to_submission_123[comment]": "",
            },
        },
    )
    monkeypatch.setattr(
        editor,
        "_submit_exact_form",
        lambda *args, **kwargs: types.SimpleNamespace(
            headers={},
            json=lambda: (_ for _ in ()).throw(ValueError("not json")),
        ),
    )
    monkeypatch.setattr(
        editor,
        "_form_get",
        lambda *args, **kwargs: types.SimpleNamespace(
            text=(
                '<article class="comment" id="comment-902">'
                "Seeded comment body with extra whitespace"
                "</article>"
            )
        ),
    )

    result = editor.create_comment(
        forum_name="books",
        submission_id="123",
        body="Seeded comment body with   extra whitespace",
    )

    assert result["comment_id"] == "902"
    assert result["read_surface_urls"][:2] == [
        "http://reddit.test/f/books/123#comment_902",
        "/f/books/123#comment_902",
    ]


def test_update_user_bio_uses_exact_form_fields(monkeypatch):
    editor = _editor(
        {"site_url": "http://reddit.test", "agent_auth": {"credentials": {"username": "user"}}}
    )
    captured = {}

    monkeypatch.setattr(
        editor,
        "_fetch_form_state",
        lambda *args, **kwargs: {
            "action": "",
            "fields": {
                "user_biography[biography]": "old bio",
                "user_biography[_token]": "csrf-token",
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
    monkeypatch.setattr(
        editor,
        "_form_get",
        lambda *args, **kwargs: types.SimpleNamespace(text="new bio"),
    )

    assert editor.update_user_bio(bio_text="new bio") == {
        "read_surface_urls": [
            "http://reddit.test/user/user",
            "/user/user",
        ],
        "read_surface_provenance_source": "editor_constructed",
    }
    assert captured["action_path"] == "/user/user/edit_biography"
    assert captured["form_fields"] == {
        "user_biography[biography]": "new bio",
        "user_biography[_token]": "csrf-token",
    }
    assert captured["multipart"] is False
    assert callable(captured["refresh_on_rejection"])


def test_update_user_bio_raises_when_profile_does_not_show_new_text(monkeypatch):
    editor = _editor(
        {"site_url": "http://reddit.test", "agent_auth": {"credentials": {"username": "user"}}}
    )
    monkeypatch.setattr(
        editor,
        "_fetch_form_state",
        lambda *args, **kwargs: {
            "action": "",
            "fields": {
                "user_biography[biography]": "old bio",
                "user_biography[_token]": "csrf-token",
            },
        },
    )
    monkeypatch.setattr(editor, "_submit_exact_form", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        editor,
        "_form_get",
        lambda *args, **kwargs: types.SimpleNamespace(text="still old bio"),
    )

    with pytest.raises(EditorError, match="did not persist the new text"):
        editor.update_user_bio(bio_text="new bio")
