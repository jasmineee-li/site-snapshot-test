from __future__ import annotations

import types
from pathlib import Path

import pytest

from worldsim import seeding


class _FakeResponse:
    def __init__(self, *, status_code: int = 200, text: str = "ok"):
        self.status_code = status_code
        self.text = text

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise seeding.requests.HTTPError(f"status={self.status_code}")


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def __enter__(self) -> _FakeSession:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def request(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        return self._responses.pop(0)

    def get(self, url, **kwargs):
        self.calls.append({"method": "GET", "url": url, **kwargs})
        return self._responses.pop(0)


class _FakeDbCursor:
    def __init__(self, rows, executed):
        self._rows = rows
        self._executed = executed

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query, params=None):
        self._executed.append((query, params))

    def fetchall(self):
        return self._rows


class _FakeDbConnection:
    def __init__(self, rows):
        self._rows = rows
        self.executed: list[tuple[object, object]] = []
        self.closed = False

    def cursor(self):
        return _FakeDbCursor(self._rows, self.executed)

    def rollback(self):
        return None

    def close(self):
        self.closed = True


def _review_postcondition_profile() -> dict:
    return {
        "injection_surface": [
            {
                "id": "review_detail_pdp",
                "delivery_channels": [
                    {
                        "mechanism": "form",
                        "privileged_seed": False,
                        "path_template": "/review/product/post/id/{product_id}/",
                        "method": "POST",
                        "body_field": "detail",
                        "table": None,
                        "column": None,
                        "postcondition": {
                            "type": "db_row_value",
                            "table": "review_detail",
                            "value_column": "detail",
                            "where": {"product_id": {"path_param": "product_id"}},
                        },
                    }
                ],
            }
        ]
    }


def _api_postcondition_profile(
    *,
    path_template: str = "/submit",
    body_field: str = "detail",
    table: str = "submissions",
    value_column: str = "detail",
    where: dict[str, object] | None = None,
) -> dict:
    return {
        "injection_surface": [
            {
                "id": "surface-1",
                "delivery_channels": [
                    {
                        "mechanism": "api",
                        "privileged_seed": False,
                        "path_template": path_template,
                        "method": "POST",
                        "body_field": body_field,
                        "table": None,
                        "column": None,
                        "postcondition": {
                            "type": "db_row_value",
                            "table": table,
                            "value_column": value_column,
                            "where": where or {"id": {"literal": 1}},
                        },
                    }
                ],
            }
        ]
    }


def _install_fake_mysql(monkeypatch, rows):
    fake_conn = _FakeDbConnection(rows)
    monkeypatch.setitem(
        __import__("sys").modules,
        "pymysql",
        types.SimpleNamespace(connect=lambda **kwargs: fake_conn),
    )
    return fake_conn


def test_validate_data_seed_accepts_form_alias():
    seeding.validate_data_seed(
        {
            "mechanism": "form",
            "api_calls": [{"method": "POST", "path": "/review", "body_form": {"title": "ok"}}],
        }
    )


def test_apply_data_seed_sql_requires_db_connection():
    with pytest.raises(RuntimeError, match="SQL seed requires instance\\['db_connection'\\]"):
        seeding.apply_data_seed(
            {
                "mechanism": "sql",
                "statements": ["INSERT INTO products (id) VALUES (1)"],
            },
            {
                "site_name": "shopping",
                "site_url": "http://shopping.test",
            },
        )


@pytest.mark.parametrize("db_connection", ["", "   "])
def test_apply_data_seed_sql_rejects_blank_db_connection(db_connection):
    with pytest.raises(RuntimeError, match="non-empty URI string"):
        seeding.apply_data_seed(
            {
                "mechanism": "sql",
                "statements": ["INSERT INTO products (id) VALUES (1)"],
            },
            {
                "site_name": "shopping",
                "site_url": "http://shopping.test",
                "db_connection": db_connection,
            },
        )


def test_apply_data_seed_sql_rejects_unsupported_db_scheme():
    with pytest.raises(RuntimeError, match="unsupported scheme 'sqlite'"):
        seeding.apply_data_seed(
            {
                "mechanism": "sql",
                "statements": ["INSERT INTO products (id) VALUES (1)"],
            },
            {
                "site_name": "shopping",
                "site_url": "http://shopping.test",
                "db_connection": "sqlite:///tmp/demo.db",
            },
        )


def test_collect_sql_seed_runtime_errors_deduplicates_by_site_instance():
    errors = seeding.collect_sql_seed_runtime_errors(
        [
            {
                "id": "task-1",
                "site": "shopping",
                "data_seed": {
                    "mechanism": "sql",
                    "statements": ["INSERT INTO products (id) VALUES (1)"],
                },
            },
            {
                "id": "task-2",
                "site": "shopping",
                "data_seed": {
                    "mechanism": "sql",
                    "statements": ["INSERT INTO products (id) VALUES (2)"],
                },
            },
        ],
        [
            {
                "site_name": "shopping",
                "site_url": "http://shopping-a.test",
                "db_connection": None,
            },
            {
                "site_name": "shopping",
                "site_url": "http://shopping-b.test",
                "db_connection": "mysql://user:pass@localhost:3306/shop",
            },
        ],
        seed_field="data_seed",
    )

    assert errors == [
        "site 'shopping' has 2 SQL-seeded task(s) but instance 'http://shopping-a.test' "
        "is not SQL-ready: SQL seed requires instance['db_connection'] must be a non-empty URI string"
    ]


def test_apply_data_seed_resolves_placeholders_and_http_headers(monkeypatch):
    fake_session = _FakeSession([_FakeResponse()])
    monkeypatch.setattr(seeding.requests, "Session", lambda: fake_session)
    monkeypatch.setenv("WORLDSIM_SHOPPING_AUTO_LOGIN", "demo:user")
    _install_fake_mysql(monkeypatch, [("ok",)])

    seeding.apply_data_seed(
        {
            "mechanism": "api",
            "api_calls": [
                {
                    "method": "POST",
                    "path": "/submit?next=__SHOPPING__/orders",
                    "body": {"detail": "ok"},
                }
            ],
        },
        {
            "site_name": "shopping",
            "site_url": "http://shopping.test",
            "url_placeholders": {"__SHOPPING__": "http://shopping.test"},
            "auth": {"type": "http_headers", "headers": {"X-Test-Auth": {"from_env": "WORLDSIM_SHOPPING_AUTO_LOGIN"}}},
            "db_connection": "mysql://user:pass@localhost:3306/db",
            "site_profile": _api_postcondition_profile(),
        },
    )

    assert fake_session.calls[0]["url"] == "http://shopping.test/submit?next=http://shopping.test/orders"
    assert fake_session.calls[0]["headers"]["X-Test-Auth"] == "demo:user"


def test_apply_data_seed_loads_bearer_token_from_file(monkeypatch, tmp_path):
    fake_session = _FakeSession([_FakeResponse()])
    monkeypatch.setattr(seeding.requests, "Session", lambda: fake_session)
    _install_fake_mysql(monkeypatch, [("ok",)])
    token_dir = tmp_path / "logs" / "phase_0d" / "gitlab"
    token_dir.mkdir(parents=True)
    token_path = token_dir / "token.txt"
    token_path.write_text("secret-token\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    seeding.apply_data_seed(
        {
            "mechanism": "api",
            "api_calls": [
                {
                    "method": "POST",
                    "path": "/api/v4/projects/1/issues",
                    "body": {"detail": "ok"},
                }
            ],
        },
        {
            "site_name": "gitlab",
            "site_url": "http://gitlab.test",
            "auth": {
                "type": "bearer_token",
                "header_name": "PRIVATE-TOKEN",
                "token_source": str(token_path),
            },
            "db_connection": "mysql://user:pass@localhost:3306/db",
            "site_profile": _api_postcondition_profile(
                path_template="/api/v4/projects/{id}/issues",
                where={"project_id": {"path_param": "id"}},
            ),
        },
    )

    assert fake_session.calls[0]["headers"]["PRIVATE-TOKEN"] == "secret-token"


def test_apply_data_seed_rejects_token_source_outside_phase_0d(monkeypatch, tmp_path):
    fake_session = _FakeSession([_FakeResponse()])
    monkeypatch.setattr(seeding.requests, "Session", lambda: fake_session)
    token_path = tmp_path / "secret.txt"
    token_path.write_text("secret-token\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError, match="token_source must be under one of"):
        seeding.apply_data_seed(
            {
                "mechanism": "api",
                "api_calls": [{"method": "POST", "path": "/api/v4/projects/1/issues"}],
            },
            {
                "site_name": "gitlab",
                "site_url": "http://gitlab.test",
                "auth": {
                    "type": "bearer_token",
                    "header_name": "PRIVATE-TOKEN",
                    "token_source": str(token_path),
                },
            },
        )


def test_apply_data_seed_form_adds_csrf_and_retries_once(monkeypatch):
    fake_session = _FakeSession(
        [
            _FakeResponse(text='<input name="form_key" value="first-token">'),
            _FakeResponse(status_code=422, text="retry"),
            _FakeResponse(text='<input name="form_key" value="second-token">'),
            _FakeResponse(status_code=200, text="ok"),
        ]
    )
    monkeypatch.setattr(seeding.requests, "Session", lambda: fake_session)
    _install_fake_mysql(monkeypatch, [("Great product",)])
    seeding._CSRF_TOKEN_CACHE.clear()

    seeding.apply_data_seed(
        {
            "mechanism": "form",
            "api_calls": [
                {"method": "POST", "path": "/review", "body_form": {"title": "Great product"}}
            ],
        },
        {
            "site_name": "shopping",
            "site_url": "http://shopping.test",
            "auth": {"type": "http_headers", "headers": {"X-Test-Auth": "demo"}},
            "db_connection": "mysql://user:pass@localhost:3306/db",
            "site_profile": {
                "injection_surface": [
                    {
                        "id": "review_detail_pdp",
                        "delivery_channels": [
                            {
                                "mechanism": "form",
                                "privileged_seed": False,
                                "path_template": "/review",
                                "method": "POST",
                                "body_field": "title",
                                "table": None,
                                "column": None,
                                "postcondition": {
                                    "type": "db_row_value",
                                    "table": "reviews",
                                    "value_column": "title",
                                    "where": {"id": {"literal": 1}},
                                },
                            }
                        ],
                    }
                ]
            },
        },
    )

    post_calls = [call for call in fake_session.calls if call["method"] == "POST"]
    assert len(post_calls) == 2
    assert post_calls[0]["data"]["form_key"] == "first-token"
    assert post_calls[1]["data"]["form_key"] == "second-token"


def test_apply_data_seed_form_retries_on_forbidden_csrf_response(monkeypatch):
    fake_session = _FakeSession(
        [
            _FakeResponse(text='<input name="form_key" value="first-token">'),
            _FakeResponse(status_code=403, text="retry"),
            _FakeResponse(text='<input name="form_key" value="second-token">'),
            _FakeResponse(status_code=200, text="ok"),
        ]
    )
    monkeypatch.setattr(seeding.requests, "Session", lambda: fake_session)
    _install_fake_mysql(monkeypatch, [("Great product",)])
    seeding._CSRF_TOKEN_CACHE.clear()

    seeding.apply_data_seed(
        {
            "mechanism": "form",
            "api_calls": [
                {"method": "POST", "path": "/review", "body_form": {"title": "Great product"}}
            ],
        },
        {
            "site_name": "shopping",
            "site_url": "http://shopping.test",
            "auth": {"type": "http_headers", "headers": {"X-Test-Auth": "demo"}},
            "db_connection": "mysql://user:pass@localhost:3306/db",
            "site_profile": {
                "injection_surface": [
                    {
                        "id": "review_detail_pdp",
                        "delivery_channels": [
                            {
                                "mechanism": "form",
                                "privileged_seed": False,
                                "path_template": "/review",
                                "method": "POST",
                                "body_field": "title",
                                "table": None,
                                "column": None,
                                "postcondition": {
                                    "type": "db_row_value",
                                    "table": "reviews",
                                    "value_column": "title",
                                    "where": {"id": {"literal": 1}},
                                },
                            }
                        ],
                    }
                ]
            },
        },
    )

    post_calls = [call for call in fake_session.calls if call["method"] == "POST"]
    assert len(post_calls) == 2
    assert post_calls[0]["data"]["form_key"] == "first-token"
    assert post_calls[1]["data"]["form_key"] == "second-token"


def test_resolve_call_url_rejects_off_origin_absolute_target():
    with pytest.raises(RuntimeError, match="must stay on origin"):
        seeding._resolve_call_url(
            "__EVIL__",
            {
                "site_name": "shopping",
                "site_url": "http://shopping.test",
                "url_placeholders": {"__EVIL__": "http://evil.test/pwn"},
            },
        )


def test_validate_data_seed_rejects_form_get():
    with pytest.raises(ValueError, match="form data seed method"):
        seeding.validate_data_seed(
            {
                "mechanism": "form",
                "api_calls": [{"method": "GET", "path": "/review", "body_form": {"title": "ok"}}],
            }
        )


def test_validate_data_seed_rejects_form_without_body_form():
    with pytest.raises(ValueError, match="body_form"):
        seeding.validate_data_seed(
            {
                "mechanism": "form",
                "api_calls": [{"method": "POST", "path": "/review"}],
            }
        )


def test_validate_data_seed_rejects_api_body_form():
    with pytest.raises(ValueError, match="must use body, not body_form"):
        seeding.validate_data_seed(
            {
                "mechanism": "api",
                "api_calls": [{"method": "POST", "path": "/review", "body_form": {"title": "ok"}}],
            }
        )


def test_apply_data_seed_does_not_allow_seed_headers_to_override_auth(monkeypatch):
    fake_session = _FakeSession([_FakeResponse()])
    monkeypatch.setattr(seeding.requests, "Session", lambda: fake_session)
    monkeypatch.setenv("WORLDSIM_SHOPPING_AUTO_LOGIN", "trusted:user")
    _install_fake_mysql(monkeypatch, [("ok",)])

    seeding.apply_data_seed(
        {
            "mechanism": "api",
            "api_calls": [
                {
                    "method": "POST",
                    "path": "/submit",
                    "body": {"detail": "ok"},
                    "headers": {"Authorization": "evil", "X-Test-Auth": "evil"},
                }
            ],
        },
        {
            "site_name": "shopping",
            "site_url": "http://shopping.test",
            "auth": {"type": "http_headers", "headers": {"X-Test-Auth": {"from_env": "WORLDSIM_SHOPPING_AUTO_LOGIN"}}},
            "db_connection": "mysql://user:pass@localhost:3306/db",
            "site_profile": _api_postcondition_profile(),
        },
    )

    headers = fake_session.calls[0]["headers"]
    assert headers["X-Test-Auth"] == "trusted:user"
    assert "Authorization" not in headers


def test_apply_data_seed_strips_origin_and_referer_headers(monkeypatch):
    fake_session = _FakeSession([_FakeResponse()])
    monkeypatch.setattr(seeding.requests, "Session", lambda: fake_session)
    _install_fake_mysql(monkeypatch, [("ok",)])

    seeding.apply_data_seed(
        {
            "mechanism": "api",
            "api_calls": [
                {
                    "method": "POST",
                    "path": "/submit",
                    "body": {"detail": "ok"},
                    "headers": {
                        "Origin": "http://evil.test",
                        "Referer": "http://evil.test/phish",
                        "X-CSRF-Token": "evil",
                        "X-Allowed": "ok",
                    },
                }
            ],
        },
        {
            "site_name": "shopping",
            "site_url": "http://shopping.test",
            "db_connection": "mysql://user:pass@localhost:3306/db",
            "site_profile": _api_postcondition_profile(),
        },
    )

    headers = fake_session.calls[0]["headers"]
    assert "Origin" not in headers
    assert "Referer" not in headers
    assert "X-CSRF-Token" not in headers
    assert headers["X-Allowed"] == "ok"


def test_apply_data_seed_http_requires_site_profile_for_postcondition_verification(monkeypatch):
    fake_session = _FakeSession([_FakeResponse()])
    monkeypatch.setattr(seeding.requests, "Session", lambda: fake_session)

    with pytest.raises(RuntimeError, match="requires instance\\['site_profile'\\]"):
        seeding.apply_data_seed(
            {
                "mechanism": "api",
                "api_calls": [{"method": "POST", "path": "/submit", "body": {"detail": "attack"}}],
            },
            {
                "site_name": "shopping",
                "site_url": "http://shopping.test",
                "db_connection": "mysql://user:pass@localhost:3306/db",
            },
        )


def test_apply_data_seed_form_verifies_db_postcondition(monkeypatch):
    fake_session = _FakeSession(
        [
            _FakeResponse(text='<input name="form_key" value="csrf-token">'),
            _FakeResponse(status_code=200, text="ok"),
        ]
    )
    fake_conn = _FakeDbConnection([("Great product",)])
    monkeypatch.setattr(seeding.requests, "Session", lambda: fake_session)
    monkeypatch.setitem(
        __import__("sys").modules,
        "pymysql",
        types.SimpleNamespace(connect=lambda **kwargs: fake_conn),
    )
    seeding._CSRF_TOKEN_CACHE.clear()

    seeding.apply_data_seed(
        {
            "mechanism": "form",
            "api_calls": [
                {
                    "method": "POST",
                    "path": "/review/product/post/id/123/",
                    "body_form": {"detail": "Great product"},
                }
            ],
        },
        {
            "site_name": "shopping",
            "site_url": "http://shopping.test",
            "db_connection": "mysql://user:pass@localhost:3306/db",
            "site_profile": _review_postcondition_profile(),
        },
    )

    assert fake_conn.closed is True
    select_query, params = fake_conn.executed[-1]
    assert "SELECT `detail` FROM `review_detail` WHERE `product_id` = %s LIMIT 5" == select_query
    assert params == ["123"]


def test_apply_data_seed_form_rejects_unsatisfied_db_postcondition(monkeypatch):
    fake_session = _FakeSession(
        [
            _FakeResponse(text='<input name="form_key" value="csrf-token">'),
            _FakeResponse(status_code=200, text="ok"),
        ]
    )
    fake_conn = _FakeDbConnection([("stale",)])
    monkeypatch.setattr(seeding.requests, "Session", lambda: fake_session)
    monkeypatch.setitem(
        __import__("sys").modules,
        "pymysql",
        types.SimpleNamespace(connect=lambda **kwargs: fake_conn),
    )
    seeding._CSRF_TOKEN_CACHE.clear()

    with pytest.raises(RuntimeError, match="did not satisfy postcondition"):
        seeding.apply_data_seed(
            {
                "mechanism": "form",
                "api_calls": [
                    {
                        "method": "POST",
                        "path": "/review/product/post/id/123/",
                        "body_form": {"detail": "Great product"},
                    }
                ],
            },
            {
                "site_name": "shopping",
                "site_url": "http://shopping.test",
                "db_connection": "mysql://user:pass@localhost:3306/db",
                "site_profile": _review_postcondition_profile(),
            },
        )


def test_request_with_context_rejects_redirect(monkeypatch):
    class _RedirectResponse(_FakeResponse):
        def __init__(self):
            super().__init__(status_code=302, text="")
            self.headers = {"Location": "http://evil.test"}

    fake_session = _FakeSession([_RedirectResponse()])

    with pytest.raises(RuntimeError, match="returned redirect"):
        seeding._request_with_context(
            fake_session,
            method="POST",
            url="http://shopping.test/submit",
            headers={},
            json_body=None,
            form_body=None,
            instance={"site_name": "shopping"},
            raw_path="/submit",
        )


def test_request_with_context_rejects_same_origin_redirect():
    class _RedirectResponse(_FakeResponse):
        def __init__(self):
            super().__init__(status_code=302, text="")
            self.headers = {"Location": "/done"}

    fake_session = _FakeSession([_RedirectResponse()])

    with pytest.raises(RuntimeError, match="returned redirect"):
        seeding._request_with_context(
            fake_session,
            method="POST",
            url="http://shopping.test/submit",
            headers={},
            json_body=None,
            form_body=None,
            instance={"site_name": "shopping"},
            raw_path="/submit",
        )


def test_get_csrf_token_uses_no_redirect_fetch(monkeypatch):
    fake_session = _FakeSession(
        [
            _FakeResponse(status_code=302, text=""),
            _FakeResponse(text='<input name="form_key" value="origin-token">'),
        ]
    )
    seeding._CSRF_TOKEN_CACHE.clear()

    token = seeding._get_csrf_token(
        fake_session,
        "http://shopping.test/review",
        {},
        {"site_name": "shopping", "site_url": "http://shopping.test"},
    )

    assert token == ("form_key", "origin-token")
    assert fake_session.calls[0]["allow_redirects"] is False
    assert fake_session.calls[1]["allow_redirects"] is False


def test_get_csrf_token_cache_is_path_scoped():
    fake_session = _FakeSession(
        [
            _FakeResponse(text='<input name="form_key" value="review-token">'),
            _FakeResponse(text='<input name="form_key" value="checkout-token">'),
        ]
    )
    seeding._CSRF_TOKEN_CACHE.clear()
    instance = {"site_name": "shopping", "site_url": "http://shopping.test"}

    review_token = seeding._get_csrf_token(fake_session, "http://shopping.test/review", {}, instance)
    checkout_token = seeding._get_csrf_token(
        fake_session,
        "http://shopping.test/checkout",
        {},
        instance,
    )

    assert review_token == ("form_key", "review-token")
    assert checkout_token == ("form_key", "checkout-token")
    assert [call["url"] for call in fake_session.calls] == [
        "http://shopping.test/review",
        "http://shopping.test/checkout",
    ]


def test_get_csrf_token_cache_normalizes_numeric_paths():
    fake_session = _FakeSession(
        [
            _FakeResponse(text='<input name="form_key" value="review-token">'),
        ]
    )
    seeding._CSRF_TOKEN_CACHE.clear()
    instance = {"site_name": "shopping", "site_url": "http://shopping.test"}

    first = seeding._get_csrf_token(fake_session, "http://shopping.test/review/123", {}, instance)
    second = seeding._get_csrf_token(fake_session, "http://shopping.test/review/456", {}, instance)

    assert first == ("form_key", "review-token")
    assert second == ("form_key", "review-token")
    assert [call["url"] for call in fake_session.calls] == ["http://shopping.test/review/123"]


def test_get_csrf_token_cache_keeps_distinct_query_variants():
    fake_session = _FakeSession(
        [
            _FakeResponse(text='<input name="form_key" value="review-token">'),
            _FakeResponse(text='<input name="form_key" value="checkout-token">'),
        ]
    )
    seeding._CSRF_TOKEN_CACHE.clear()
    instance = {"site_name": "shopping", "site_url": "http://shopping.test"}

    first = seeding._get_csrf_token(
        fake_session,
        "http://shopping.test/review?id=123",
        {},
        instance,
    )
    second = seeding._get_csrf_token(
        fake_session,
        "http://shopping.test/review?id=456",
        {},
        instance,
    )

    assert first == ("form_key", "review-token")
    assert second == ("form_key", "checkout-token")
    assert [call["url"] for call in fake_session.calls] == [
        "http://shopping.test/review?id=123",
        "http://shopping.test/review?id=456",
    ]


def test_prepare_form_body_overwrites_stale_csrf_field():
    fake_session = _FakeSession(
        [
            _FakeResponse(text='<input name="form_key" value="fresh-token">'),
        ]
    )
    seeding._CSRF_TOKEN_CACHE.clear()

    form_body = seeding._prepare_form_body(
        "POST",
        "http://shopping.test/review",
        {},
        {"form_key": "stale-token", "detail": "payload"},
        {"site_name": "shopping", "site_url": "http://shopping.test"},
        fake_session,
    )

    assert form_body == {"form_key": "fresh-token", "detail": "payload"}


def test_get_csrf_token_cache_does_not_cross_sessions():
    session_a = _FakeSession([_FakeResponse(text='<input name="form_key" value="token-a">')])
    session_b = _FakeSession([_FakeResponse(text='<input name="form_key" value="token-b">')])
    seeding._CSRF_TOKEN_CACHE.clear()
    instance = {"site_name": "shopping", "site_url": "http://shopping.test"}

    token_a = seeding._get_csrf_token(session_a, "http://shopping.test/review", {}, instance)
    token_b = seeding._get_csrf_token(session_b, "http://shopping.test/review", {}, instance)

    assert token_a == ("form_key", "token-a")
    assert token_b == ("form_key", "token-b")
    assert [call["url"] for call in session_a.calls] == ["http://shopping.test/review"]
    assert [call["url"] for call in session_b.calls] == ["http://shopping.test/review"]
