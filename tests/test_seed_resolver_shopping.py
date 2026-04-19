from __future__ import annotations

import pytest
import requests  # used by _fake_response in classifier unit tests below

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


def _fake_response(body: str, status: int = 400, *, as_json: bool = False) -> requests.Response:
    """Build a minimal `requests.Response` for classifier unit tests."""
    r = requests.Response()
    r.status_code = status
    if as_json:
        r._content = body.encode("utf-8")
        r.headers["Content-Type"] = "application/json"
    else:
        r._content = body.encode("utf-8")
        r.headers["Content-Type"] = "text/plain"
    return r


# Magento 2 rephrases length errors across minor versions, auth modes, and
# store-scope locales. The r5 deployment specifically accepts overlong
# review fields silently on `/rest/V1/reviews` (verified 2026-04-19), so
# the live integration test can't assert the classifier's contract on
# Magento. Pin the contract here instead: every plausible wording must
# bucket as `length_exceeded`.
@pytest.mark.parametrize(
    "message",
    [
        "Please enter a value with a maximum length of 255 characters.",
        "The title is too long. Allowed length is 255.",
        "Please enter no more than 255 characters.",
        "Title exceeds the maximum of 255 characters.",
        "Please enter up to 255 characters.",
        "Value too long for column title",
        "Title character limit is 255.",
        "Too many characters in title.",
        "Field cannot be longer than 255 characters.",
        "Please enter a value less than 255 characters.",
    ],
)
def test_classify_4xx_length_exceeded_recognizes_magento_wordings(message):
    editor = ShoppingEditor({"site_url": "http://shopping.test"}, session=None)
    resp = _fake_response(message)

    classification = editor._classify_4xx_response("POST", "/rest/V1/reviews", resp)

    assert classification is not None, (
        f"Magento length wording {message!r} fell through to request_failed; "
        "add a matching token to worldsim/editors/shopping.py::_MAGENTO_LENGTH_TOKENS."
    )
    kind, detail = classification
    assert kind == "length_exceeded"
    assert detail == message


@pytest.mark.parametrize(
    "message",
    [
        "The first name is a required field.",
        "Nickname cannot be empty.",
        "Please enter a value.",
        "Missing parameter: title",
        "Email is required.",
    ],
)
def test_classify_4xx_field_required_recognizes_magento_wordings(message):
    editor = ShoppingEditor({"site_url": "http://shopping.test"}, session=None)
    resp = _fake_response(message)

    classification = editor._classify_4xx_response("POST", "/rest/V1/reviews", resp)

    assert classification is not None
    kind, _detail = classification
    assert kind == "field_required"


@pytest.mark.parametrize(
    "message",
    [
        "Your review was flagged as spam.",
        "This action is forbidden.",
        "Review rejected by content policy.",
        "Submission not allowed.",
        "Request denied by site policy.",
    ],
)
def test_classify_4xx_content_policy_recognizes_magento_wordings(message):
    editor = ShoppingEditor({"site_url": "http://shopping.test"}, session=None)
    resp = _fake_response(message)

    classification = editor._classify_4xx_response("POST", "/rest/V1/reviews", resp)

    assert classification is not None
    kind, _detail = classification
    assert kind == "content_policy"


def test_classify_4xx_unrecognized_message_falls_through_and_logs(caplog):
    """Catch-all 4xx with a message we haven't token-matched should return
    None so the caller falls through to generic `request_failed`, AND
    emit a DEBUG log so future wording drift is diagnosable."""
    import logging

    editor = ShoppingEditor({"site_url": "http://shopping.test"}, session=None)
    resp = _fake_response("Unexpected platform-specific error not yet in tokens.")

    with caplog.at_level(logging.DEBUG, logger="worldsim.editors.shopping"):
        classification = editor._classify_4xx_response("POST", "/rest/V1/reviews", resp)

    assert classification is None
    # The DEBUG line includes the first message for diagnosability.
    assert any("did not match any _MAGENTO_*_TOKENS" in rec.message for rec in caplog.records)


def test_classify_4xx_empty_body_returns_none_and_logs(caplog):
    import logging

    editor = ShoppingEditor({"site_url": "http://shopping.test"}, session=None)
    resp = _fake_response("")

    with caplog.at_level(logging.DEBUG, logger="worldsim.editors.shopping"):
        classification = editor._classify_4xx_response("POST", "/rest/V1/reviews", resp)

    assert classification is None
    assert any("empty / unparseable body" in rec.message for rec in caplog.records)
