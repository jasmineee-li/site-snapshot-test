from __future__ import annotations

import pytest

from worldsim.editors.base import EditorError
from worldsim.editors.shopping import ShoppingEditor


def test_validate_args_requires_review_detail():
    with pytest.raises(EditorError, match="missing required args: detail"):
        ShoppingEditor({"site_url": "http://shopping.test"}, session=None).validate_args(
            "create_product_review",
            {},
        )


def test_create_product_review_builds_expected_review_payload(monkeypatch):
    editor = ShoppingEditor({"site_url": "http://shopping.test"}, session=None)
    captured = {}

    def fake_api_request_json(method, path, *, json_body=None, params=None, allow_missing=False):
        captured["method"] = method
        captured["path"] = path
        captured["json_body"] = json_body
        return {"id": 42}

    monkeypatch.setattr(editor, "_api_request_json", fake_api_request_json)

    result = editor.create_product_review(
        title="Review title",
        detail="Payload text",
        nickname="reviewer",
        entity_pk_value=7,
        rating=5,
    )

    assert result == {
        "review_id": 42,
        "read_surface_urls": [
            "http://shopping.test/catalog/product/view/id/7",
            "/catalog/product/view/id/7",
            "http://shopping.test/review/product/listAjax/id/7",
            "/review/product/listAjax/id/7",
        ],
        "read_surface_provenance_source": "editor_constructed",
    }
    assert captured["method"] == "POST"
    assert captured["path"] == "/rest/V1/reviews"
    assert captured["json_body"] == {
        "review": {
            "review_entity": "product",
            "title": "Review title",
            "detail": "Payload text",
            "nickname": "reviewer",
            "entity_pk_value": 7,
            "stores": [1],
            "ratings": [{"rating_name": "Quality", "value": 5}],
        }
    }


def test_update_customer_profile_posts_account_edit_form(monkeypatch):
    editor = ShoppingEditor({"site_url": "http://shopping.test"}, session=None)
    captured = {}

    monkeypatch.setattr(
        editor,
        "_fetch_form_state",
        lambda *args, **kwargs: {
            "action": "/customer/account/editpost/",
            "fields": {
                "form_key": "abc123",
                "firstname": "Emma",
                "lastname": "Lopez",
                "email": "emma@example.com",
                "change_email": "1",
                "current_password": "",
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

    assert editor.update_customer_profile(field="firstname", value="Alice") == {
        "read_surface_urls": [
            "http://shopping.test/customer/account/index/",
            "/customer/account/index/",
        ],
        "read_surface_provenance_source": "editor_constructed",
    }
    assert captured["action_path"] == "/customer/account/editpost/"
    assert captured["form_fields"] == {
        "form_key": "abc123",
        "firstname": "Alice",
        "lastname": "Lopez",
        "email": "emma@example.com",
    }
    assert captured["multipart"] is False
    assert callable(captured["refresh_on_rejection"])
