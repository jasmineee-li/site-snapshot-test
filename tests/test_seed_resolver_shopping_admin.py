from __future__ import annotations

from worldsim.editors.base import BaseSiteEditor
from worldsim.editors.shopping_admin import ShoppingAdminEditor


def test_update_admin_profile_writes_expected_payload(monkeypatch):
    editor = ShoppingAdminEditor({"site_url": "http://shopping-admin.test"}, session=None)
    captured = {}

    monkeypatch.setattr(
        editor,
        "_fetch_form_state",
        lambda *args, **kwargs: {
            "action": "/admin/admin/user/save/",
            "fields": {
                "form_key": "xyz789",
                "username": "admin",
                "email": "admin@example.com",
            },
        },
    )

    def fake_submit_exact_form(action_path, form_fields, *, multipart=False, refresh_on_rejection=None):
        captured["action_path"] = action_path
        captured["form_fields"] = form_fields
        captured["multipart"] = multipart
        captured["refresh_on_rejection"] = refresh_on_rejection
        return {}

    monkeypatch.setattr(editor, "_submit_exact_form", fake_submit_exact_form)

    assert editor.update_admin_profile(field="firstname", value="Alice") == {}
    assert captured["action_path"] == "/admin/admin/user/save/"
    assert captured["form_fields"] == {
        "form_key": "xyz789",
        "username": "admin",
        "email": "admin@example.com",
        "firstname": "Alice",
    }
    assert captured["multipart"] is False
    assert callable(captured["refresh_on_rejection"])


def test_cleanup_runs_lifo_and_clears_stack():
    editor = BaseSiteEditor.__new__(BaseSiteEditor)
    editor.site_name = "test"
    editor._cleanup_stack = []
    order: list[str] = []

    BaseSiteEditor._push_cleanup(editor, lambda: order.append("first"))
    BaseSiteEditor._push_cleanup(editor, lambda: order.append("second"))
    BaseSiteEditor.cleanup(editor)

    assert order == ["second", "first"]
    assert editor._cleanup_stack == []


def test_cleanup_raises_when_any_cleanup_op_fails():
    editor = BaseSiteEditor.__new__(BaseSiteEditor)
    editor.site_name = "test"
    editor._cleanup_stack = []
    order: list[str] = []

    def fail() -> None:
        order.append("second")
        raise RuntimeError("boom")

    BaseSiteEditor._push_cleanup(editor, lambda: order.append("first"))
    BaseSiteEditor._push_cleanup(editor, fail)

    try:
        BaseSiteEditor.cleanup(editor)
    except Exception as exc:
        assert "cleanup failed" in str(exc)
    else:
        raise AssertionError("cleanup should have raised")

    assert order == ["second", "first"]
    assert editor._cleanup_stack == []
