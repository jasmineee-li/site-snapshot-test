"""Tests for the editor-side HTTP auth + form-login plumbing.

These helpers used to live as private symbols in ``warp_taskgen.seeding`` and
were tested via ``tests/test_seeding.py``. After the editor migration the
helpers moved to ``warp_taskgen.auth_tokens.build_auth_headers`` and
``warp_taskgen.editors._form_login`` (their sole callers); this file covers
the new public surface.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
import requests

from warp_taskgen import auth_tokens
from warp_taskgen.auth_tokens import (
    _BLOCKED_CALL_HEADER_NAMES,
    build_auth_headers,
    pick_auth_lane,
)
from warp_taskgen.editors._form_login import (
    extract_csrf_token,
    looks_like_login_page,
    perform_web_login_if_needed,
    prepare_form_body,
)


@pytest.fixture
def stub_resolve_bearer_token(monkeypatch):
    """Stub auth_tokens.resolve_bearer_token to return the inline token verbatim.

    The real resolver requires a live validation_endpoint on every bearer
    config. These tests are exercising header construction, not token
    acquisition, so we short-circuit the resolver to keep fixtures terse.
    """
    monkeypatch.setattr(
        auth_tokens, "resolve_bearer_token", lambda auth, *, site_url="": str(auth["token"])
    )


# ── pick_auth_lane ───────────────────────────────────────────────────────


def test_pick_auth_lane_prefers_api_auth_for_api_mechanism():
    instance = {
        "auth": {"type": "web_login", "credentials": {"username": "u", "password": "p"}},
        "api_auth": {"type": "bearer_token", "token": "tok"},
    }
    assert pick_auth_lane(instance, "api") == instance["api_auth"]


def test_pick_auth_lane_falls_back_to_auth_for_form_mechanism():
    instance = {
        "auth": {"type": "web_login", "credentials": {"username": "u", "password": "p"}},
        "api_auth": {"type": "bearer_token", "token": "tok"},
    }
    assert pick_auth_lane(instance, "form") == instance["auth"]


def test_pick_auth_lane_returns_none_when_auth_missing():
    assert pick_auth_lane({}, "api") is None
    assert pick_auth_lane({"auth": "not-a-dict"}, "form") is None


# ── build_auth_headers ───────────────────────────────────────────────────


def test_build_auth_headers_bearer_token_authorization(stub_resolve_bearer_token):
    instance = {
        "site_url": "https://example.test",
        "api_auth": {"type": "bearer_token", "token": "glpat-secret"},
    }
    headers = build_auth_headers(instance, {}, mechanism="api")
    assert headers == {"Authorization": "Bearer glpat-secret"}


def test_build_auth_headers_bearer_token_preserves_existing_bearer_prefix(
    stub_resolve_bearer_token,
):
    instance = {
        "site_url": "https://example.test",
        "api_auth": {"type": "bearer_token", "token": "Bearer already-prefixed"},
    }
    headers = build_auth_headers(instance, {}, mechanism="api")
    assert headers == {"Authorization": "Bearer already-prefixed"}


def test_build_auth_headers_bearer_token_custom_header_name(stub_resolve_bearer_token):
    instance = {
        "site_url": "https://example.test",
        "api_auth": {
            "type": "bearer_token",
            "token": "glpat-secret",
            "header_name": "PRIVATE-TOKEN",
        },
    }
    headers = build_auth_headers(instance, {}, mechanism="api")
    assert headers == {"PRIVATE-TOKEN": "glpat-secret"}


def test_build_auth_headers_http_headers_with_env_resolution(monkeypatch):
    monkeypatch.setenv("WORLDSIM_TEST_AUTH", "value-from-env")
    instance = {
        "site_url": "https://example.test",
        "api_auth": {
            "type": "http_headers",
            "headers": {
                "X-Auth": "literal-value",
                "X-Env": {"from_env": "WORLDSIM_TEST_AUTH"},
            },
        },
    }
    headers = build_auth_headers(instance, {}, mechanism="api")
    assert headers == {"X-Auth": "literal-value", "X-Env": "value-from-env"}


def test_build_auth_headers_http_headers_missing_env_raises(monkeypatch):
    monkeypatch.delenv("WORLDSIM_MISSING_ENV", raising=False)
    instance = {
        "site_url": "https://example.test",
        "api_auth": {
            "type": "http_headers",
            "headers": {"X-Env": {"from_env": "WORLDSIM_MISSING_ENV"}},
        },
    }
    with pytest.raises(RuntimeError, match="WORLDSIM_MISSING_ENV"):
        build_auth_headers(instance, {}, mechanism="api")


def test_build_auth_headers_merges_safe_call_headers(stub_resolve_bearer_token):
    instance = {
        "site_url": "https://example.test",
        "api_auth": {"type": "bearer_token", "token": "tok"},
    }
    call = {"headers": {"X-Custom": "ok", "Accept": "application/json"}}
    headers = build_auth_headers(instance, call, mechanism="api")
    assert headers["Authorization"] == "Bearer tok"
    assert headers["X-Custom"] == "ok"
    assert headers["Accept"] == "application/json"


def test_build_auth_headers_strips_blocked_call_headers(stub_resolve_bearer_token):
    instance = {
        "site_url": "https://example.test",
        "api_auth": {"type": "bearer_token", "token": "tok"},
    }
    call = {
        "headers": {
            "Authorization": "Bearer hostile",  # would override auth lane
            "Cookie": "session=hostile",
            "X-Forwarded-For": "10.0.0.1",
            "Host": "evil.test",
            "X-Allowed": "kept",
        },
    }
    headers = build_auth_headers(instance, call, mechanism="api")
    assert headers == {
        "Authorization": "Bearer tok",
        "X-Allowed": "kept",
    }


def test_blocked_call_header_names_blocks_csrf_and_proxy():
    blocked = {
        "authorization",
        "cookie",
        "host",
        "x-csrf-token",
        "proxy-authorization",
        "x-forwarded-for",
    }
    assert blocked.issubset(_BLOCKED_CALL_HEADER_NAMES)


def test_build_auth_headers_with_no_auth_returns_empty():
    headers = build_auth_headers({"site_url": "https://example.test"}, {}, mechanism="form")
    assert headers == {}


def test_build_auth_headers_with_no_call_returns_auth_only(stub_resolve_bearer_token):
    instance = {
        "site_url": "https://example.test",
        "api_auth": {"type": "bearer_token", "token": "tok"},
    }
    headers = build_auth_headers(instance, mechanism="api")
    assert headers == {"Authorization": "Bearer tok"}


# ── extract_csrf_token + looks_like_login_page ──────────────────────────


def test_extract_csrf_token_input_pattern():
    html = '<input name="authenticity_token" value="abc123">'
    assert extract_csrf_token(html) == ("authenticity_token", "abc123")


def test_extract_csrf_token_meta_pattern_uses_csrf_param():
    html = (
        '<meta name="csrf-token" content="metavalue">'
        '<meta name="csrf-param" content="custom_param">'
    )
    assert extract_csrf_token(html) == ("custom_param", "metavalue")


def test_extract_csrf_token_meta_pattern_default_param_name():
    html = '<meta name="csrf-token" content="metavalue">'
    assert extract_csrf_token(html) == ("csrf_token", "metavalue")


def test_extract_csrf_token_no_match():
    assert extract_csrf_token("<p>nothing here</p>") == (None, None)


def test_looks_like_login_page_detects_password_input():
    assert looks_like_login_page('<input type="password" name="user[password]">')


def test_looks_like_login_page_detects_sign_in():
    assert looks_like_login_page("<h1>Sign in</h1>")


def test_looks_like_login_page_returns_false_on_unrelated_html():
    assert not looks_like_login_page("<p>not a login page</p>")


# ── prepare_form_body ────────────────────────────────────────────────────


class _FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        text: str = "",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)


def test_prepare_form_body_returns_none_for_non_dict_body():
    session = MagicMock(spec=requests.Session)
    result = prepare_form_body(
        "POST",
        "https://example.test/x",
        {},
        ["not", "a", "dict"],
        {"site_name": "x"},
        session,
    )
    assert result is None


def test_prepare_form_body_skips_csrf_for_get_method():
    session = MagicMock(spec=requests.Session)
    body = {"a": "1"}
    result = prepare_form_body(
        "GET", "https://example.test/x", {}, body, {"site_name": "x"}, session
    )
    assert result == body
    session.get.assert_not_called()


def test_prepare_form_body_injects_csrf_for_post(monkeypatch):
    session = requests.Session()
    response_html = '<input name="authenticity_token" value="csrf-tok">'
    monkeypatch.setattr(
        session, "get", lambda *a, **kw: _FakeResponse(status_code=200, text=response_html)
    )
    result = prepare_form_body(
        "POST",
        "https://example.test/posts",
        {},
        {"title": "hi"},
        {"site_name": "x"},
        session,
    )
    assert result == {"title": "hi", "authenticity_token": "csrf-tok"}


def test_prepare_form_body_returns_body_unchanged_when_csrf_missing(monkeypatch):
    session = requests.Session()
    monkeypatch.setattr(
        session, "get", lambda *a, **kw: _FakeResponse(status_code=200, text="<p>nope</p>")
    )
    result = prepare_form_body(
        "POST",
        "https://example.test/posts",
        {},
        {"title": "hi"},
        {"site_name": "x"},
        session,
    )
    assert result == {"title": "hi"}


# ── perform_web_login_if_needed ──────────────────────────────────────────


def test_perform_web_login_if_needed_skips_when_not_web_login():
    session = MagicMock(spec=requests.Session)
    instance = {"auth": {"type": "bearer_token", "token": "tok"}}
    perform_web_login_if_needed(session, instance, "form")
    session.get.assert_not_called()
    session.post.assert_not_called()


def test_perform_web_login_if_needed_skips_when_no_auth():
    session = MagicMock(spec=requests.Session)
    perform_web_login_if_needed(session, {}, "form")
    session.get.assert_not_called()


def test_perform_web_login_if_needed_requires_credentials():
    session = MagicMock(spec=requests.Session)
    instance: dict[str, Any] = {
        "site_url": "https://example.test",
        "site_name": "demo",
        "auth": {"type": "web_login"},
    }
    with pytest.raises(RuntimeError, match="requires credentials"):
        perform_web_login_if_needed(session, instance, "form")


def test_perform_web_login_if_needed_happy_path(monkeypatch):
    session = requests.Session()
    login_get_html = '<input name="authenticity_token" value="csrf">'
    captured: dict[str, Any] = {}

    def fake_get(url, *args, **kwargs):
        captured["get_url"] = url
        return _FakeResponse(status_code=200, text=login_get_html)

    def fake_post(url, *args, **kwargs):
        captured["post_url"] = url
        captured["post_data"] = kwargs.get("data")
        return _FakeResponse(status_code=302)

    monkeypatch.setattr(session, "get", fake_get)
    monkeypatch.setattr(session, "post", fake_post)

    instance = {
        "site_url": "https://example.test",
        "site_name": "demo",
        "auth": {
            "type": "web_login",
            "login_url": "/login",
            "credentials": {"username": "alice", "password": "wonder"},
        },
    }
    perform_web_login_if_needed(session, instance, "form")
    assert captured["post_url"] == "https://example.test/login"
    assert captured["post_data"]["username"] == "alice"
    assert captured["post_data"]["authenticity_token"] == "csrf"


def test_perform_web_login_if_needed_raises_on_login_redirect_loop(monkeypatch):
    session = requests.Session()

    def fake_get(*args, **kwargs):
        return _FakeResponse(status_code=200, text='<input name="csrf" value="t">')

    def fake_post(*args, **kwargs):
        return _FakeResponse(status_code=302, headers={"Location": "/users/sign_in"})

    monkeypatch.setattr(session, "get", fake_get)
    monkeypatch.setattr(session, "post", fake_post)

    instance = {
        "site_url": "https://example.test",
        "site_name": "demo",
        "auth": {
            "type": "web_login",
            "login_url": "/login",
            "credentials": {"username": "alice", "password": "wonder"},
        },
    }
    with pytest.raises(RuntimeError, match="redirected back to login"):
        perform_web_login_if_needed(session, instance, "form")


def test_perform_web_login_if_needed_raises_on_form_re_render(monkeypatch):
    session = requests.Session()

    def fake_get(*args, **kwargs):
        return _FakeResponse(status_code=200, text='<input name="csrf" value="t">')

    def fake_post(*args, **kwargs):
        return _FakeResponse(status_code=200, text='<input type="password">')

    monkeypatch.setattr(session, "get", fake_get)
    monkeypatch.setattr(session, "post", fake_post)

    instance = {
        "site_url": "https://example.test",
        "site_name": "demo",
        "auth": {
            "type": "web_login",
            "login_url": "/login",
            "credentials": {"username": "alice", "password": "wonder"},
        },
    }
    with pytest.raises(RuntimeError, match="login form was re-rendered"):
        perform_web_login_if_needed(session, instance, "form")
