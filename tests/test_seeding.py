from __future__ import annotations

import types
from typing import ClassVar

import pytest

from worldsim import _sandbox_validator, seeding


class _FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        text: str = "ok",
        json_data=None,
        headers: dict[str, str] | None = None,
    ):
        self.status_code = status_code
        self.text = text
        self._json_data = json_data
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise seeding.requests.HTTPError(f"status={self.status_code}")

    def json(self):
        if self._json_data is None:
            raise ValueError("no json")
        return self._json_data


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse]):
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.closed = False

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

    def post(self, url, **kwargs):
        self.calls.append({"method": "POST", "url": url, **kwargs})
        return self._responses.pop(0)

    def close(self) -> None:
        self.closed = True


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


def test_web_login_rejects_login_form_rerender():
    session = _FakeSession(
        [
            _FakeResponse(status_code=200, text='<input name="csrf_token" value="t">'),
            _FakeResponse(status_code=200, text='<input type="password" name="_password">'),
        ]
    )

    with pytest.raises(RuntimeError, match="login form was re-rendered"):
        seeding._perform_web_login_if_needed(
            session,
            {
                "site_name": "reddit",
                "site_url": "http://reddit.test",
                "auth": {
                    "type": "web_login",
                    "login_url": "/login",
                    "credentials": {"username": "u", "password": "p"},
                },
            },
            "form",
        )


def test_web_login_uses_validation_endpoint_when_present():
    session = _FakeSession(
        [
            _FakeResponse(status_code=200, text='<input name="csrf_token" value="t">'),
            _FakeResponse(status_code=302, headers={"Location": "/"}),
            _FakeResponse(status_code=200, text="welcome"),
        ]
    )

    seeding._perform_web_login_if_needed(
        session,
        {
            "site_name": "reddit",
            "site_url": "http://reddit.test",
            "auth": {
                "type": "web_login",
                "login_url": "/login",
                "validation_endpoint": "/me",
                "credentials": {"username": "u", "password": "p"},
            },
        },
        "form",
    )

    assert [call["url"] for call in session.calls] == [
        "http://reddit.test/login",
        "http://reddit.test/login",
        "http://reddit.test/me",
    ]


def test_web_login_validation_endpoint_rejects_redirect_back_to_login():
    session = _FakeSession(
        [
            _FakeResponse(status_code=200, text='<input name="csrf_token" value="t">'),
            _FakeResponse(status_code=302, headers={"Location": "/"}),
            _FakeResponse(status_code=302, headers={"Location": "/login"}),
        ]
    )

    with pytest.raises(RuntimeError, match="validation endpoint redirected to login"):
        seeding._perform_web_login_if_needed(
            session,
            {
                "site_name": "reddit",
                "site_url": "http://reddit.test",
                "auth": {
                    "type": "web_login",
                    "login_url": "/login",
                    "validation_endpoint": "/me",
                    "credentials": {"username": "u", "password": "p"},
                },
            },
            "form",
        )


def test_seed_requires_reset_for_editor_only_seed_without_mechanism():
    assert seeding.seed_requires_reset(
        {
            "mechanism": "none",
            "editor_calls": [
                {
                    "benchmark": "webarena_verified",
                    "site": "shopping",
                    "method": "create_product_review",
                    "args": {"entity_pk_value": 7, "detail": "payload"},
                }
            ],
        }
    )


def test_seed_requires_reset_for_editor_seed_mechanism():
    assert seeding.seed_requires_reset(
        {
            "mechanism": "editor",
            "editor_calls": [
                {
                    "benchmark": "webarena_verified",
                    "site": "gitlab",
                    "method": "update_user_status",
                    "args": {"status": "payload"},
                }
            ],
        }
    )


def test_collect_seed_runtime_errors_reports_missing_http_header_env(monkeypatch):
    monkeypatch.delenv("WORLDSIM_SHOPPING_AUTO_LOGIN", raising=False)

    errors = seeding.collect_seed_runtime_errors(
        [
            {
                "id": "task-1",
                "site": "shopping",
                "delivery_channel": {
                    "mechanism": "form",
                    "body_field": "detail",
                    "postcondition": {"type": "db_row_value"},
                },
                "adversarial_data_seed": {
                    "mechanism": "form",
                    "api_calls": [
                        {"method": "POST", "path": "/review", "body_form": {"detail": "x"}}
                    ],
                },
            }
        ],
        [
            {
                "site_name": "shopping",
                "site_url": "http://shopping.test",
                "db_connection": "mysql://user:pass@localhost:3306/shop",
                "auth": {
                    "type": "http_headers",
                    "headers": {"X-Test-Auth": {"from_env": "WORLDSIM_SHOPPING_AUTO_LOGIN"}},
                },
            }
        ],
        seed_field="adversarial_data_seed",
    )

    assert errors == [
        "site 'shopping' has form HTTP-seeded task(s) but instance 'http://shopping.test' "
        "has invalid auth config: required auth header env var 'WORLDSIM_SHOPPING_AUTO_LOGIN' is not set"
    ]


def test_collect_seed_runtime_errors_reports_missing_bearer_token_source(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    errors = seeding.collect_seed_runtime_errors(
        [
            {
                "id": "task-1",
                "site": "gitlab",
                "adversarial_data_seed": {
                    "mechanism": "api",
                    "api_calls": [
                        {"method": "POST", "path": "/api/issues", "body": {"detail": "x"}}
                    ],
                },
            }
        ],
        [
            {
                "site_name": "gitlab",
                "site_url": "http://gitlab.test",
                "auth": {
                    "type": "bearer_token",
                    "header_name": "PRIVATE-TOKEN",
                    "token_source": "logs/phase_0d/gitlab/missing.txt",
                },
            }
        ],
        seed_field="adversarial_data_seed",
    )

    assert errors == [
        "site 'gitlab' has api HTTP-seeded task(s) but instance 'http://gitlab.test' has invalid auth config: "
        f"token_source {(tmp_path / 'logs/phase_0d/gitlab/missing.txt').resolve(strict=False)} does not exist"
    ]


def test_collect_seed_runtime_errors_ignores_stale_token_source_when_generator_selected(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    errors = seeding.collect_seed_runtime_errors(
        [
            {
                "id": "task-1",
                "site": "gitlab",
                "adversarial_data_seed": {
                    "mechanism": "api",
                    "api_calls": [
                        {"method": "POST", "path": "/api/issues", "body": {"detail": "x"}}
                    ],
                },
            }
        ],
        [
            {
                "site_name": "gitlab",
                "site_url": "http://gitlab.test",
                "auth": {
                    "type": "bearer_token",
                    "header_name": "PRIVATE-TOKEN",
                    "token_generator": "gitlab_pat",
                    "credentials": {"username": "byteblaze", "password": "hello1234"},
                    "token_source": "logs/phase_0d/gitlab/missing.txt",
                },
            }
        ],
        seed_field="adversarial_data_seed",
    )

    assert errors == []


def test_collect_seed_runtime_errors_skips_db_check_for_http_db_row_verification():
    """HTTP seeds with db_row_value postconditions do NOT require db_connection at pre-flight.

    The runtime gracefully skips DB verification when db_connection is absent
    (HTTP 2xx confirms the seed landed).
    """
    errors = seeding.collect_seed_runtime_errors(
        [
            {
                "id": "task-1",
                "site": "map",
                "delivery_channel": {
                    "mechanism": "form",
                    "body_field": "detail",
                    "postcondition": {"type": "db_row_value"},
                },
                "adversarial_data_seed": {
                    "mechanism": "form",
                    "api_calls": [
                        {"method": "POST", "path": "/review", "body_form": {"detail": "x"}}
                    ],
                },
            }
        ],
        [
            {
                "site_name": "map",
                "site_url": "http://map.test",
            }
        ],
        seed_field="adversarial_data_seed",
    )

    assert errors == []


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
            "auth": {
                "type": "http_headers",
                "headers": {"X-Test-Auth": {"from_env": "WORLDSIM_SHOPPING_AUTO_LOGIN"}},
            },
            "db_connection": "mysql://user:pass@localhost:3306/db",
            "site_profile": _api_postcondition_profile(),
        },
    )

    assert (
        fake_session.calls[0]["url"]
        == "http://shopping.test/submit?next=http://shopping.test/orders"
    )
    assert fake_session.calls[0]["headers"]["X-Test-Auth"] == "demo:user"


def test_apply_data_seed_renders_chained_placeholders_from_response_context(monkeypatch):
    fake_session = _FakeSession(
        [
            _FakeResponse(text='<input name="form_key" value="first-token">'),
            _FakeResponse(json_data={"forum_name": "books", "submission_id": 42}),
            _FakeResponse(text='<input name="form_key" value="second-token">'),
            _FakeResponse(),
        ]
    )
    monkeypatch.setattr(seeding.requests, "Session", lambda: fake_session)
    monkeypatch.setattr(seeding, "_verify_http_seed_postcondition", lambda **kwargs: None)

    seeding.apply_data_seed(
        {
            "mechanism": "form",
            "api_calls": [
                {
                    "method": "POST",
                    "path": "/seed/bootstrap",
                    "body_form": {"detail": "bootstrap"},
                },
                {
                    "method": "POST",
                    "path": "/f/{forum_name}/{submission_id}/-/comment",
                    "body_form": {
                        "reply_to_submission_{submission_id}[comment]": "payload for {forum_name}"
                    },
                },
            ],
        },
        {
            "site_name": "reddit",
            "site_url": "http://reddit.test",
            "site_profile": _review_postcondition_profile(),
        },
    )

    second_call = [call for call in fake_session.calls if call["method"] == "POST"][1]
    assert second_call["url"] == "http://reddit.test/f/books/42/-/comment"
    assert second_call["data"]["reply_to_submission_42[comment]"] == "payload for books"


def test_apply_data_seed_derives_map_way_id_from_task_context(monkeypatch):
    fake_session = _FakeSession([_FakeResponse()])
    monkeypatch.setattr(seeding.requests, "Session", lambda: fake_session)
    monkeypatch.setattr(seeding, "_verify_http_seed_postcondition", lambda **kwargs: None)

    def fake_get(url, *, params, timeout):
        assert url == "http://map.test/nominatim/search"
        assert params["q"] == "Columbia University"
        return _FakeResponse(
            json_data=[
                {"osm_type": "relation", "osm_id": 7, "display_name": "Columbia University"},
                {"osm_type": "way", "osm_id": 19, "display_name": "Columbia University"},
            ]
        )

    monkeypatch.setattr(seeding.requests, "get", fake_get)

    seeding.apply_data_seed(
        {
            "mechanism": "api",
            "api_calls": [
                {
                    "method": "PUT",
                    "path": "/api/0.6/way/{way_id}",
                    "body": {"name": "ok"},
                }
            ],
        },
        {
            "site_name": "map",
            "site_url": "http://map.test",
            "site_profile": _api_postcondition_profile(path_template="/api/0.6/way/{way_id}"),
            "seed_task": {
                "site": "map",
                "instantiation_dict": {"place": "Columbia University"},
            },
        },
    )

    assert fake_session.calls[0]["url"] == "http://map.test/api/0.6/way/19"


def test_apply_data_seed_derives_reddit_submission_placeholders_from_task_context(
    monkeypatch,
):
    fake_session = _FakeSession(
        [
            _FakeResponse(text='<input name="form_key" value="csrf-token">'),
            _FakeResponse(),
        ]
    )
    monkeypatch.setattr(seeding.requests, "Session", lambda: fake_session)
    monkeypatch.setattr(seeding, "_verify_http_seed_postcondition", lambda **kwargs: None)
    seeding._CSRF_TOKEN_CACHE.clear()
    seeding._REDDIT_TABLE_NAME_CACHE.clear()

    class _LookupCursor:
        def __init__(self):
            self._result = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, query, params=None):
            if "to_regclass" in query and params == ["forums"]:
                self._result = ("forums",)
            elif "to_regclass" in query and params == ["forum"]:
                self._result = None
            elif "to_regclass" in query and params == ["submissions"]:
                self._result = ("submissions",)
            elif "to_regclass" in query and params == ["submission"]:
                self._result = None
            elif params == ["Books", "Books", "Books"]:
                self._result = (10067, "books", "Books")
            elif params == ["Best place for a foot rub?", "books", "books"]:
                self._result = (50001,)
            else:
                self._result = None

        def fetchone(self):
            return self._result

    class _LookupConnection:
        def cursor(self):
            return _LookupCursor()

        def rollback(self):
            return None

        def close(self):
            return None

    monkeypatch.setattr(seeding, "_connect_db", lambda parsed: _LookupConnection())

    seeding.apply_data_seed(
        {
            "mechanism": "form",
            "api_calls": [
                {
                    "method": "POST",
                    "path": "/f/{forum_name}/{submission_id}/-/comment",
                    "body_form": {
                        "reply_to_submission_{submission_id}[comment]": "ok",
                    },
                }
            ],
        },
        {
            "site_name": "reddit",
            "site_url": "http://reddit.test",
            "db_connection": "postgresql://user:pass@localhost:5432/reddit",
            "site_profile": _review_postcondition_profile(),
            "seed_task": {
                "site": "reddit",
                "instantiation_dict": {"forum": "Books"},
                "reward_function": {
                    "eval": [
                        {
                            "expected": {
                                "retrieved_data": [{"post_title": "Best place for a foot rub?"}]
                            }
                        }
                    ]
                },
            },
        },
    )

    post_call = next(call for call in fake_session.calls if call["method"] == "POST")
    assert post_call["url"] == "http://reddit.test/f/books/50001/-/comment"
    assert post_call["data"]["reply_to_submission_50001[comment]"] == "ok"


def test_resolve_reddit_forum_supports_plural_table_name(monkeypatch):
    seeding._REDDIT_TABLE_NAME_CACHE.clear()

    class _LookupCursor:
        def __init__(self):
            self._result = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, query, params=None):
            if "to_regclass" in query and params == ["forums"]:
                self._result = ("forums",)
            elif "to_regclass" in query and params == ["forum"]:
                self._result = None
            elif params == ["Books", "Books", "Books"]:
                self._result = (10067, "books", "Books")
            else:
                self._result = None

        def fetchone(self):
            return self._result

    class _LookupConnection:
        def cursor(self):
            return _LookupCursor()

        def rollback(self):
            return None

        def close(self):
            return None

    monkeypatch.setattr(seeding, "_connect_db", lambda parsed: _LookupConnection())

    forum = seeding._resolve_reddit_forum(
        {"instantiation_dict": {"forum": "Books"}},
        {"db_connection": "postgresql://user:pass@localhost:5432/reddit"},
    )

    assert forum == {"id": 10067, "name": "books", "title": "Books"}


def test_resolve_reddit_forum_raises_clear_error_when_no_table_matches(monkeypatch):
    seeding._REDDIT_TABLE_NAME_CACHE.clear()

    class _LookupCursor:
        def __init__(self):
            self._result = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, query, params=None):
            if query in {"BEGIN", "SET TRANSACTION READ ONLY"}:
                self._result = None
            elif "to_regclass" in query:
                self._result = None
            else:
                raise AssertionError("table lookup should fail before the query runs")

        def fetchone(self):
            return self._result

    class _LookupConnection:
        def cursor(self):
            return _LookupCursor()

        def rollback(self):
            return None

        def close(self):
            return None

    monkeypatch.setattr(seeding, "_connect_db", lambda parsed: _LookupConnection())

    with pytest.raises(RuntimeError, match="reddit schema table resolution failed"):
        seeding._resolve_reddit_forum(
            {"instantiation_dict": {"forum": "Books"}},
            {"db_connection": "postgresql://user:pass@localhost:5432/reddit"},
        )


def test_apply_data_seed_loads_bearer_token_from_file(monkeypatch, tmp_path):
    fake_session = _FakeSession([_FakeResponse()])
    monkeypatch.setattr(seeding.requests, "Session", lambda: fake_session)
    monkeypatch.setattr(
        seeding.requests,
        "get",
        lambda *args, **kwargs: _FakeResponse(status_code=200),
    )
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
                "validation_endpoint": "/api/v4/user",
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
                    "validation_endpoint": "/api/v4/user",
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
            "auth": {
                "type": "http_headers",
                "headers": {"X-Test-Auth": {"from_env": "WORLDSIM_SHOPPING_AUTO_LOGIN"}},
            },
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


def test_apply_data_seed_http_does_not_require_site_profile_for_postcondition_verification(
    monkeypatch,
):
    fake_session = _FakeSession([_FakeResponse()])
    monkeypatch.setattr(seeding.requests, "Session", lambda: fake_session)

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

    assert fake_session.calls[0]["url"] == "http://shopping.test/submit"


def test_apply_data_seed_form_skips_db_postcondition_verification(monkeypatch):
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

    assert fake_conn.closed is False
    assert fake_conn.executed == []


def test_apply_data_seed_form_does_not_fail_without_db_postcondition_match(monkeypatch):
    fake_session = _FakeSession(
        [
            _FakeResponse(text='<input name="form_key" value="csrf-token">'),
            _FakeResponse(status_code=200, text="ok"),
        ]
    )
    monkeypatch.setattr(seeding.requests, "Session", lambda: fake_session)
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
            "site_profile": _review_postcondition_profile(),
        },
    )

    assert [call["method"] for call in fake_session.calls] == ["GET", "POST"]


def test_request_with_context_rejects_redirect(monkeypatch):
    class _RedirectResponse(_FakeResponse):
        def __init__(self):
            super().__init__(status_code=302, text="")
            self.headers = {"Location": "http://evil.test"}

    fake_session = _FakeSession([_RedirectResponse()])

    with pytest.raises(RuntimeError, match="returned redirect") as excinfo:
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
    assert "evil.test" not in str(excinfo.value)
    assert "location='<present>'" in str(excinfo.value)


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

    review_token = seeding._get_csrf_token(
        fake_session, "http://shopping.test/review", {}, instance
    )
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


def test_build_seed_context_prefers_seed_task_values():
    context = seeding._build_seed_context(
        {"mechanism": "api", "api_calls": []},
        {
            "seed_task": {
                "id": "adv-123",
                "instruction": "Do the thing",
                "topic": "deploy pipeline",
                "intent": "review the deploy pipeline",
            }
        },
    )

    assert context["task_id"] == "adv-123"
    assert context["instruction"] == "Do the thing"
    assert context["topic"] == "deploy pipeline"
    assert context["intent"] == "review the deploy pipeline"


class _FakeEditor:
    instances: ClassVar[list[_FakeEditor]] = []
    supported_methods = frozenset({"create_submission", "create_comment"})

    def __init__(self, instance, session) -> None:
        self.instance = instance
        self.session = session
        self.cleaned = False
        self.calls: list[tuple[str, dict]] = []
        _FakeEditor.instances.append(self)

    def cleanup(self) -> None:
        self.cleaned = True

    def validate_args(self, method_name: str, args: dict[str, object]) -> None:
        self.calls.append(("validate_args", {"method_name": method_name, "args": dict(args)}))
        required = {
            "create_submission": ("forum_name", "title_template"),
            "create_comment": ("forum_name", "submission_id", "body"),
        }.get(method_name, ())
        missing = [key for key in required if args.get(key) in (None, "")]
        if missing:
            raise seeding.EditorError(
                "invalid_args", "missing required args: " + ", ".join(missing)
            )

    def preview_context(self, method_name: str, args: dict[str, object]) -> dict[str, object]:
        if method_name == "create_submission":
            return {"forum_name": args["forum_name"], "submission_id": "59421"}
        if method_name == "create_comment":
            return {"comment_id": "901"}
        return {}

    def create_submission(
        self, *, forum_name: str, title_template: str, body_template: str | None = None
    ) -> dict[str, object]:
        args = {"forum_name": forum_name, "title_template": title_template}
        if body_template is not None:
            args["body_template"] = body_template
        self.calls.append(("create_submission", args))
        return {"forum_name": forum_name, "submission_id": "59421"}

    def create_comment(
        self, *, forum_name: str, submission_id: str, body: str
    ) -> dict[str, object]:
        self.calls.append(
            (
                "create_comment",
                {"forum_name": forum_name, "submission_id": submission_id, "body": body},
            )
        )
        return {"comment_id": "901"}


class _FailingCleanupEditor(_FakeEditor):
    def __init__(self, instance, session, *, label: str, fail: bool) -> None:
        super().__init__(instance, session)
        self.label = label
        self.fail = fail

    def cleanup(self) -> None:
        self.cleaned = True
        if self.fail:
            raise RuntimeError(f"cleanup failed for {self.label}")


class _RejectingEditor(_FakeEditor):
    def validate_args(self, method_name: str, args: dict[str, object]) -> None:
        raise seeding.EditorError("invalid_args", "bad args")


def test_apply_data_seed_normalizes_editor_call_benchmark_alias(monkeypatch):
    _FakeEditor.instances.clear()
    fake_session = _FakeSession([])
    monkeypatch.setattr(seeding.requests, "Session", lambda: fake_session)
    monkeypatch.setitem(seeding.EDITOR_REGISTRY, ("webarena_verified", "reddit"), _FakeEditor)

    cleanup_handle, _metadata = seeding.apply_data_seed(
        {
            "mechanism": "editor",
            "editor_calls": [
                {
                    "benchmark": "WebArena Verified",
                    "site": "reddit",
                    "method": "create_submission",
                    "args": {"forum_name": "books", "title_template": "Thread"},
                }
            ],
        },
        {"site_name": "reddit", "site_url": "http://reddit.test"},
    )

    assert cleanup_handle is not None
    assert len(_FakeEditor.instances) == 1
    assert _FakeEditor.instances[0].calls[-1][0] == "create_submission"


def test_apply_data_seed_uses_instance_benchmark_when_call_omits_it(monkeypatch):
    _FakeEditor.instances.clear()
    fake_session = _FakeSession([])
    monkeypatch.setattr(seeding.requests, "Session", lambda: fake_session)
    monkeypatch.setitem(seeding.EDITOR_REGISTRY, ("stwebagentbench", "reddit"), _FakeEditor)

    cleanup_handle, _metadata = seeding.apply_data_seed(
        {
            "mechanism": "editor",
            "editor_calls": [
                {
                    "site": "reddit",
                    "method": "create_submission",
                    "args": {"forum_name": "books", "title_template": "Thread"},
                }
            ],
        },
        {
            "benchmark": "stwebagentbench",
            "site_name": "reddit",
            "site_url": "http://reddit.test",
        },
    )

    assert cleanup_handle is not None
    assert len(_FakeEditor.instances) == 1
    assert _FakeEditor.instances[0].calls[-1][0] == "create_submission"


def test_apply_data_seed_supports_editor_calls_and_context_chaining(monkeypatch):
    _FakeEditor.instances.clear()
    fake_session = _FakeSession([])
    monkeypatch.setattr(seeding.requests, "Session", lambda: fake_session)
    monkeypatch.setitem(seeding.EDITOR_REGISTRY, ("webarena_verified", "reddit"), _FakeEditor)

    cleanup_handle, _metadata = seeding.apply_data_seed(
        {
            "mechanism": "editor",
            "editor_calls": [
                {
                    "benchmark": "webarena_verified",
                    "site": "reddit",
                    "method": "create_submission",
                    "args": {"forum_name": "books", "title_template": "Thread"},
                },
                {
                    "benchmark": "webarena_verified",
                    "site": "reddit",
                    "method": "create_comment",
                    "args": {
                        "forum_name": "{forum_name}",
                        "submission_id": "{submission_id}",
                        "body": "payload",
                    },
                },
            ],
        },
        {"site_name": "reddit", "site_url": "http://reddit.test"},
    )

    editor = _FakeEditor.instances[-1]
    assert cleanup_handle is not None
    assert editor.calls == [
        (
            "validate_args",
            {
                "method_name": "create_submission",
                "args": {"forum_name": "books", "title_template": "Thread"},
            },
        ),
        ("create_submission", {"forum_name": "books", "title_template": "Thread"}),
        (
            "validate_args",
            {
                "method_name": "create_comment",
                "args": {"forum_name": "books", "submission_id": "59421", "body": "payload"},
            },
        ),
        (
            "create_comment",
            {"forum_name": "books", "submission_id": "59421", "body": "payload"},
        ),
    ]
    assert editor.cleaned is False
    assert fake_session.closed is False


def test_apply_data_seed_filters_unknown_editor_kwargs_before_validation(monkeypatch):
    _FakeEditor.instances.clear()
    fake_session = _FakeSession([])
    monkeypatch.setattr(seeding.requests, "Session", lambda: fake_session)
    monkeypatch.setitem(seeding.EDITOR_REGISTRY, ("webarena_verified", "reddit"), _FakeEditor)

    cleanup_handle, _metadata = seeding.apply_data_seed(
        {
            "mechanism": "editor",
            "editor_calls": [
                {
                    "benchmark": "webarena_verified",
                    "site": "reddit",
                    "method": "create_comment",
                    "args": {
                        "forum_name": "books",
                        "submission_id": "59421",
                        "body": "payload",
                        "sticky": True,
                        "comment_position": "{missing_position}",
                    },
                }
            ],
        },
        {"site_name": "reddit", "site_url": "http://reddit.test"},
    )

    editor = _FakeEditor.instances[-1]
    assert cleanup_handle is not None
    assert editor.calls == [
        (
            "validate_args",
            {
                "method_name": "create_comment",
                "args": {"forum_name": "books", "submission_id": "59421", "body": "payload"},
            },
        ),
        (
            "create_comment",
            {"forum_name": "books", "submission_id": "59421", "body": "payload"},
        ),
    ]

    cleanup_handle.cleanup()

    assert editor.cleaned is True
    assert fake_session.closed is True


def test_apply_data_seed_maps_reddit_submission_form_field_aliases(monkeypatch):
    _FakeEditor.instances.clear()
    fake_session = _FakeSession([])
    monkeypatch.setattr(seeding.requests, "Session", lambda: fake_session)
    monkeypatch.setitem(seeding.EDITOR_REGISTRY, ("webarena_verified", "reddit"), _FakeEditor)

    cleanup_handle, _metadata = seeding.apply_data_seed(
        {
            "mechanism": "editor",
            "editor_calls": [
                {
                    "benchmark": "webarena_verified",
                    "site": "reddit",
                    "method": "create_submission",
                    "args": {
                        "forum_name": "books",
                        "submission[title]": "Thread title",
                        "submission[body]": "Payload body",
                    },
                }
            ],
        },
        {"site_name": "reddit", "site_url": "http://reddit.test"},
    )

    editor = _FakeEditor.instances[-1]
    assert cleanup_handle is not None
    assert editor.calls == [
        (
            "validate_args",
            {
                "method_name": "create_submission",
                "args": {
                    "forum_name": "books",
                    "title_template": "Thread title",
                    "body_template": "Payload body",
                },
            },
        ),
        (
            "create_submission",
            {
                "forum_name": "books",
                "title_template": "Thread title",
                "body_template": "Payload body",
            },
        ),
    ]
    cleanup_handle.cleanup()


def test_seed_cleanup_handle_cleans_all_editors_before_raising():
    fake_session = _FakeSession([])
    first = _FailingCleanupEditor({}, fake_session, label="first", fail=True)
    second = _FailingCleanupEditor({}, fake_session, label="second", fail=False)
    handle = seeding.SeedCleanupHandle(
        session=fake_session,
        editor_instances={
            ("webarena_verified", "reddit"): first,
            ("webarena_verified", "gitlab"): second,
        },
    )

    with pytest.raises(RuntimeError, match="cleanup failed for first"):
        handle.cleanup()

    assert first.cleaned is True
    assert second.cleaned is True
    assert fake_session.closed is True


class _ReadSurfaceEditor(_FakeEditor):
    """FakeEditor variant that emits ``read_surface_urls`` — for §5.5 tests."""

    site_name = "reddit"

    def create_submission(self, *, forum_name: str, title_template: str) -> dict[str, object]:
        self.calls.append(
            ("create_submission", {"forum_name": forum_name, "title_template": title_template})
        )
        return {
            "forum_name": forum_name,
            "submission_id": "59421",
            "created_resource": {
                "role": "seed_render_surface",
                "kind": "submission",
                "id": "59421",
                "url": "http://reddit.test/f/books/59421",
                "parent_url": "http://reddit.test/f/books",
            },
            "read_surface_urls": [
                "http://reddit.test/f/books/59421",
                "/f/books/59421",
            ],
            "read_surface_provenance_source": "editor_constructed",
        }

    def create_comment(
        self, *, forum_name: str, submission_id: str, body: str
    ) -> dict[str, object]:
        self.calls.append(
            (
                "create_comment",
                {"forum_name": forum_name, "submission_id": submission_id, "body": body},
            )
        )
        return {"comment_id": "901"}


def _reddit_editor_call_submission() -> dict[str, object]:
    return {
        "benchmark": "webarena_verified",
        "site": "reddit",
        "method": "create_submission",
        "args": {"forum_name": "books", "title_template": "Thread"},
    }


def test_apply_data_seed_returns_editor_only_read_surface_urls(monkeypatch):
    """Editor-emitted URLs surface in metadata when no explicit override is set."""
    _ReadSurfaceEditor.instances.clear()
    fake_session = _FakeSession([])
    monkeypatch.setattr(seeding.requests, "Session", lambda: fake_session)
    monkeypatch.setitem(
        seeding.EDITOR_REGISTRY, ("webarena_verified", "reddit"), _ReadSurfaceEditor
    )

    cleanup_handle, metadata = seeding.apply_data_seed(
        {"mechanism": "editor", "editor_calls": [_reddit_editor_call_submission()]},
        {"site_name": "reddit", "site_url": "http://reddit.test"},
    )

    assert cleanup_handle is not None
    cleanup_handle.cleanup()
    assert metadata["read_surface_urls"] == [
        "http://reddit.test/f/books/59421",
        "/f/books/59421",
    ]
    provenance = metadata["read_surface_provenance"]
    assert provenance["source"] == "editor_constructed"
    assert provenance["editor_method"] == ["reddit.create_submission"]


def test_apply_data_seed_merges_explicit_override_with_editor_surface_urls(monkeypatch):
    """Handoff §5.5: explicit task-level URLs union with editor contribution."""
    _ReadSurfaceEditor.instances.clear()
    fake_session = _FakeSession([])
    monkeypatch.setattr(seeding.requests, "Session", lambda: fake_session)
    monkeypatch.setitem(
        seeding.EDITOR_REGISTRY, ("webarena_verified", "reddit"), _ReadSurfaceEditor
    )

    cleanup_handle, metadata = seeding.apply_data_seed(
        {"mechanism": "editor", "editor_calls": [_reddit_editor_call_submission()]},
        {
            "site_name": "reddit",
            "site_url": "http://reddit.test",
            "seed_task": {
                "id": "task_override",
                # Explicit override comes first: both stays, both deduped.
                "read_surface_urls": [
                    "/f/books/59421",  # duplicate of editor path-form
                    "/some/explicit/path",  # unique
                ],
            },
        },
    )

    assert cleanup_handle is not None
    cleanup_handle.cleanup()
    # Union preserving first-occurrence order: explicit first, then editor.
    # /f/books/59421 appeared first in explicit, host-qualified form is unique.
    assert metadata["read_surface_urls"] == [
        "/f/books/59421",
        "/some/explicit/path",
        "http://reddit.test/f/books/59421",
    ]
    provenance = metadata["read_surface_provenance"]
    assert provenance["source"] == "explicit_override+editor"


def test_apply_data_seed_explicit_override_only(monkeypatch):
    """Editor contributes nothing → provenance source is explicit_override."""
    _FakeEditor.instances.clear()
    fake_session = _FakeSession([])
    monkeypatch.setattr(seeding.requests, "Session", lambda: fake_session)
    # Register the non-emitting _FakeEditor so editor_calls succeed but
    # return no read_surface_urls.
    monkeypatch.setitem(seeding.EDITOR_REGISTRY, ("webarena_verified", "reddit"), _FakeEditor)

    cleanup_handle, metadata = seeding.apply_data_seed(
        {"mechanism": "editor", "editor_calls": [_reddit_editor_call_submission()]},
        {
            "site_name": "reddit",
            "site_url": "http://reddit.test",
            "seed_task": {
                "id": "task_override",
                "read_surface_urls": ["/explicit/only"],
            },
        },
    )

    assert cleanup_handle is not None
    cleanup_handle.cleanup()
    assert metadata["read_surface_urls"] == ["/explicit/only"]
    assert metadata["read_surface_provenance"]["source"] == "explicit_override"


class _MultiCallEditor(_FakeEditor):
    """Two methods, each contributing distinct read_surface_urls — §12.9."""

    site_name = "reddit"
    supported_methods = frozenset({"create_submission", "create_comment"})

    def create_submission(self, *, forum_name: str, title_template: str) -> dict[str, object]:
        self.calls.append(("create_submission", {"forum_name": forum_name}))
        return {
            "forum_name": forum_name,
            "submission_id": "59421",
            "read_surface_urls": ["/f/books/59421"],
            "read_surface_provenance_source": "editor_constructed",
        }

    def create_comment(
        self, *, forum_name: str, submission_id: str, body: str
    ) -> dict[str, object]:
        self.calls.append(("create_comment", {"forum_name": forum_name}))
        return {
            "forum_name": forum_name,
            "comment_id": "901",
            # Second call surfaces a distinct URL and uses a stronger source.
            "read_surface_urls": ["/f/books/59421#comment-901"],
            "read_surface_provenance_source": "editor_api_response",
        }


def test_apply_data_seed_multi_call_accumulates_editor_methods_and_picks_strongest_source(
    monkeypatch,
):
    """Every contributing editor call stamps provenance (handoff §12.9)."""
    _MultiCallEditor.instances.clear()
    fake_session = _FakeSession([])
    monkeypatch.setattr(seeding.requests, "Session", lambda: fake_session)
    monkeypatch.setitem(seeding.EDITOR_REGISTRY, ("webarena_verified", "reddit"), _MultiCallEditor)

    cleanup_handle, metadata = seeding.apply_data_seed(
        {
            "mechanism": "editor",
            "editor_calls": [
                {
                    "benchmark": "webarena_verified",
                    "site": "reddit",
                    "method": "create_submission",
                    "args": {"forum_name": "books", "title_template": "Thread"},
                },
                {
                    "benchmark": "webarena_verified",
                    "site": "reddit",
                    "method": "create_comment",
                    "args": {
                        "forum_name": "{forum_name}",
                        "submission_id": "{submission_id}",
                        "body": "payload",
                    },
                },
            ],
        },
        {"site_name": "reddit", "site_url": "http://reddit.test"},
    )

    assert cleanup_handle is not None
    cleanup_handle.cleanup()
    assert metadata["read_surface_urls"] == [
        "/f/books/59421",
        "/f/books/59421#comment-901",
    ]
    provenance = metadata["read_surface_provenance"]
    # Both methods contributed — both appear, first-occurrence order.
    assert provenance["editor_method"] == [
        "reddit.create_submission",
        "reddit.create_comment",
    ]
    # api_response beats constructed — stronger claim wins.
    assert provenance["source"] == "editor_api_response"


def test_apply_data_seed_hoists_write_tokens_into_metadata(monkeypatch):
    """Write identifiers (submission_id, comment_id) land on metadata for the render RYW fastpath."""
    _ReadSurfaceEditor.instances.clear()
    fake_session = _FakeSession([])
    monkeypatch.setattr(seeding.requests, "Session", lambda: fake_session)
    monkeypatch.setitem(
        seeding.EDITOR_REGISTRY, ("webarena_verified", "reddit"), _ReadSurfaceEditor
    )

    cleanup_handle, metadata = seeding.apply_data_seed(
        {
            "mechanism": "editor",
            "editor_calls": [
                _reddit_editor_call_submission(),
                {
                    "benchmark": "webarena_verified",
                    "site": "reddit",
                    "method": "create_comment",
                    "args": {
                        "forum_name": "{forum_name}",
                        "submission_id": "{submission_id}",
                        "body": "hi",
                    },
                },
            ],
        },
        {"site_name": "reddit", "site_url": "http://reddit.test"},
    )

    assert cleanup_handle is not None
    cleanup_handle.cleanup()
    assert metadata.get("submission_id") == "59421"
    assert metadata.get("comment_id") == "901"
    assert metadata["created_resource"] == {
        "role": "seed_render_surface",
        "kind": "submission",
        "id": "59421",
        "url": "http://reddit.test/f/books/59421",
        "parent_url": "http://reddit.test/f/books",
        "editor_method": "reddit.create_submission",
    }
    assert metadata["created_resources"] == [metadata["created_resource"]]
    # Other token slots stay absent when the editor doesn't emit them.
    assert "note_id" not in metadata
    assert "review_id" not in metadata


def test_preflight_editor_seed_calls_chains_preview_context(monkeypatch):
    _FakeEditor.instances.clear()
    monkeypatch.setitem(seeding.EDITOR_REGISTRY, ("webarena_verified", "reddit"), _FakeEditor)

    errors = seeding.preflight_editor_seed_calls(
        {
            "mechanism": "editor",
            "editor_calls": [
                {
                    "benchmark": "webarena_verified",
                    "site": "reddit",
                    "method": "create_submission",
                    "args": {"forum_name": "books", "title_template": "Thread"},
                },
                {
                    "benchmark": "webarena_verified",
                    "site": "reddit",
                    "method": "create_comment",
                    "args": {
                        "forum_name": "{forum_name}",
                        "submission_id": "{submission_id}",
                        "body": "payload",
                    },
                },
            ],
        },
        {"site_name": "reddit", "site_url": "http://reddit.test"},
    )

    assert errors == []
    assert _FakeEditor.instances[-1].cleaned is True


def test_preflight_editor_seed_calls_reports_editor_errors(monkeypatch):
    monkeypatch.setitem(seeding.EDITOR_REGISTRY, ("webarena_verified", "reddit"), _RejectingEditor)

    errors = seeding.preflight_editor_seed_calls(
        {
            "mechanism": "editor",
            "editor_calls": [
                {
                    "benchmark": "webarena_verified",
                    "site": "reddit",
                    "method": "create_comment",
                    "args": {"forum_name": "books", "submission_id": "1", "body": "payload"},
                }
            ],
        },
        {"site_name": "reddit", "site_url": "http://reddit.test"},
    )

    assert errors == [
        {
            "call_index": 0,
            "site": "reddit",
            "kind": "invalid_args",
            "detail": "bad args",
            "method": "create_comment",
        }
    ]


def test_validate_data_seed_rejects_editor_api_calls_mix():
    with pytest.raises(ValueError, match="editor data seed must not include api_calls"):
        seeding.validate_data_seed(
            {
                "mechanism": "editor",
                "editor_calls": [
                    {
                        "benchmark": "webarena_verified",
                        "site": "reddit",
                        "method": "create_submission",
                        "args": {"forum_name": "books", "title_template": "Thread"},
                    }
                ],
                "api_calls": [{"method": "POST", "path": "/legacy", "body": {}}],
            }
        )


def test_preflight_http_seed_calls_validates_concrete_legacy_absolute_url_origin():
    errors = seeding.preflight_http_seed_calls(
        {
            "mechanism": "api",
            "api_calls": [
                {
                    "method": "POST",
                    "url": "http://evil.test/rest/V1/reviews",
                    "body": {"detail": "payload"},
                }
            ],
        },
        {
            "site_name": "shopping",
            "site_url": "http://shopping.test",
        },
    )

    assert errors == [
        "api_calls[0]: HTTP seed target must stay on origin 'http://shopping.test', got 'http://evil.test/rest/V1/reviews'"
    ]


def test_task_seed_site_treats_none_delivery_site_as_empty():
    assert (
        seeding._task_seed_site(
            {
                "site": "shopping",
                "delivery_channel": {"delivery_site": "none"},
            }
        )
        == "shopping"
    )


def test_apply_data_seed_supports_legacy_url_only_api_calls(monkeypatch):
    fake_session = _FakeSession([_FakeResponse()])
    monkeypatch.setattr(seeding.requests, "Session", lambda: fake_session)
    monkeypatch.setattr(
        seeding.requests,
        "get",
        lambda *args, **kwargs: _FakeResponse(status_code=200),
    )
    monkeypatch.setattr(seeding, "_verify_http_seed_postcondition", lambda **kwargs: None)

    seeding.apply_data_seed(
        {
            "mechanism": "api",
            "api_calls": [
                {
                    "method": "POST",
                    "url": "http://shopping.test/rest/V1/reviews",
                    "body": {"review": {"title": "Title", "detail": "Payload", "nickname": "nick"}},
                }
            ],
        },
        {
            "site_name": "shopping",
            "site_url": "http://shopping.test",
            "api_auth": {
                "type": "bearer_token",
                "token": "demo-token",
                "validation_endpoint": "/rest/V1/modules",
            },
        },
    )

    assert fake_session.calls[0]["url"] == "http://shopping.test/rest/V1/reviews"


def test_apply_data_seed_supports_legacy_url_only_form_calls(monkeypatch):
    fake_session = _FakeSession(
        [
            _FakeResponse(text='<input name="form_key" value="csrf-token">'),
            _FakeResponse(),
        ]
    )
    monkeypatch.setattr(seeding.requests, "Session", lambda: fake_session)
    monkeypatch.setattr(seeding, "_verify_http_seed_postcondition", lambda **kwargs: None)

    seeding.apply_data_seed(
        {
            "mechanism": "form",
            "api_calls": [
                {
                    "method": "POST",
                    "url": "http://reddit.test/create_forum",
                    "body_form": {"forum[name]": "books", "forum[description]": "desc"},
                }
            ],
        },
        {"site_name": "reddit", "site_url": "http://reddit.test"},
    )

    post_call = next(call for call in fake_session.calls if call["method"] == "POST")
    assert post_call["url"] == "http://reddit.test/create_forum"
    assert post_call["data"]["form_key"] == "csrf-token"


def test_validate_data_seed_accepts_editor_calls():
    seeding.validate_data_seed(
        {
            "mechanism": "editor",
            "editor_calls": [
                {
                    "benchmark": "webarena_verified",
                    "site": "reddit",
                    "method": "create_comment",
                    "args": {"forum_name": "books", "submission_id": "1", "body": "payload"},
                }
            ],
        }
    )


def test_validate_data_seed_rejects_target_based_calls():
    with pytest.raises(ValueError, match="target-based api_calls are no longer supported"):
        seeding.validate_data_seed(
            {
                "mechanism": "api",
                "api_calls": [
                    {
                        "target": {"site": "gitlab", "resource_type": "project", "create": {}},
                        "body": {"name": "new-project"},
                    }
                ],
            }
        )


def test_sandbox_validate_data_seed_accepts_editor_calls():
    assert (
        _sandbox_validator.validate_data_seed(
            {
                "mechanism": "editor",
                "editor_calls": [
                    {
                        "benchmark": "webarena_verified",
                        "site": "reddit",
                        "method": "create_comment",
                        "args": {"forum_name": "books", "submission_id": "1", "body": "payload"},
                    }
                ],
            }
        )
        == []
    )


def test_sandbox_validate_data_seed_rejects_editor_api_calls_mix():
    assert _sandbox_validator.validate_data_seed(
        {
            "mechanism": "editor",
            "editor_calls": [
                {
                    "benchmark": "webarena_verified",
                    "site": "reddit",
                    "method": "create_comment",
                    "args": {"forum_name": "books", "submission_id": "1", "body": "payload"},
                }
            ],
            "api_calls": [{"method": "POST", "path": "/legacy", "body": {}}],
        }
    ) == ["editor data seed must not include api_calls"]


def test_sandbox_validate_data_seed_rejects_target_based_calls():
    assert _sandbox_validator.validate_data_seed(
        {
            "mechanism": "api",
            "api_calls": [
                {
                    "target": {"site": "shopping", "resource_type": "product_review", "create": {}},
                    "body": {"detail": "payload"},
                }
            ],
        }
    ) == ["target-based api_calls are no longer supported; migrate to editor_calls"]


def test_sandbox_self_contained_adversarial_seed_accepts_editor_prefix_extension():
    benign_seed = {
        "mechanism": "editor",
        "editor_calls": [
            {
                "benchmark": "webarena_verified",
                "site": "reddit",
                "method": "create_submission",
                "args": {"forum_name": "books", "title_template": "Thread"},
            }
        ],
    }
    adversarial_seed = {
        "mechanism": "editor",
        "editor_calls": benign_seed["editor_calls"]
        + [
            {
                "benchmark": "webarena_verified",
                "site": "reddit",
                "method": "create_comment",
                "args": {
                    "forum_name": "{forum_name}",
                    "submission_id": "{submission_id}",
                    "body": "payload",
                },
            }
        ],
    }

    assert (
        _sandbox_validator.self_contained_adversarial_seed_error(benign_seed, adversarial_seed)
        is None
    )


def test_self_contained_adversarial_seed_requires_none_mechanism_editor_prefix():
    benign_seed = {
        "mechanism": "none",
        "editor_calls": [
            {
                "benchmark": "webarena_verified",
                "site": "reddit",
                "method": "create_submission",
                "args": {"forum_name": "books", "title_template": "Thread"},
            }
        ],
    }
    adversarial_seed = {
        "mechanism": "editor",
        "editor_calls": [
            {
                "benchmark": "webarena_verified",
                "site": "reddit",
                "method": "create_comment",
                "args": {"forum_name": "books", "submission_id": "1", "body": "payload"},
            }
        ],
    }

    assert seeding.self_contained_adversarial_seed_error(benign_seed, adversarial_seed) == (
        "adversarial_data_seed must preserve the benign data_seed verbatim before extending it"
    )


def test_sandbox_self_contained_adversarial_seed_requires_none_mechanism_editor_prefix():
    benign_seed = {
        "mechanism": "none",
        "editor_calls": [
            {
                "benchmark": "webarena_verified",
                "site": "reddit",
                "method": "create_submission",
                "args": {"forum_name": "books", "title_template": "Thread"},
            }
        ],
    }
    adversarial_seed = {
        "mechanism": "editor",
        "editor_calls": [
            {
                "benchmark": "webarena_verified",
                "site": "reddit",
                "method": "create_comment",
                "args": {"forum_name": "books", "submission_id": "1", "body": "payload"},
            }
        ],
    }

    assert (
        _sandbox_validator.self_contained_adversarial_seed_error(benign_seed, adversarial_seed)
        == "adversarial_data_seed must preserve the benign data_seed verbatim before extending it"
    )


def test_sandbox_validator_accepts_nested_review_body_field_reference():
    error = _sandbox_validator._find_unresolved_http_seed_reference(
        {
            "mechanism": "api",
            "api_calls": [
                {
                    "method": "POST",
                    "path": "/rest/V1/reviews",
                    "body": {"review": {"detail": "payload", "entity_pk_value": 7}},
                }
            ],
        },
        {
            "body_field": "detail",
            "postcondition": {
                "type": "db_row_value",
                "where": {"product_id": {"body_field": "entity_pk_value"}},
            },
        },
    )

    assert error is None


def test_apply_data_seed_rejects_unresolved_placeholders_in_supported_editor_kwargs(monkeypatch):
    _FakeEditor.instances.clear()
    fake_session = _FakeSession([])
    monkeypatch.setattr(seeding.requests, "Session", lambda: fake_session)
    monkeypatch.setitem(seeding.EDITOR_REGISTRY, ("webarena_verified", "reddit"), _FakeEditor)

    with pytest.raises(
        RuntimeError,
        match="editor call has unresolved template placeholders: missing_submission",
    ):
        seeding.apply_data_seed(
            {
                "mechanism": "editor",
                "editor_calls": [
                    {
                        "benchmark": "webarena_verified",
                        "site": "reddit",
                        "method": "create_comment",
                        "args": {
                            "forum_name": "books",
                            "submission_id": "{missing_submission}",
                            "body": "payload",
                        },
                    }
                ],
            },
            {"site_name": "reddit", "site_url": "http://reddit.test"},
        )


def test_extract_http_body_flattens_nested_review_wrapper():
    body = seeding._extract_http_body(
        {
            "body": {
                "review": {
                    "detail": "payload",
                    "entity_pk_value": 67,
                }
            }
        }
    )

    assert body == {"detail": "payload", "entity_pk_value": 67}


def test_build_seed_context_surfaces_benign_anchors_from_target_resource():
    # Option A: seed context must carry the benign_* keys so
    # {benign_issue_iid} etc. resolve at seed-apply time.
    instance = {
        "seed_task": {
            "id": "adv-1",
            "benign_task_id": "175",
            "instruction": "Get issue status",
            "benign_target_resource": {
                "kind": "gitlab_issue",
                "anchors": {
                    "project_id": 159,
                    "project_path": "primer/design",
                    "issue_iid": 104,
                },
                "attach_surfaces": [],
                "layer": "L1",
            },
            "agent_context": {"authentication": {"credentials": {"username": "byteblaze"}}},
        }
    }
    context = seeding._build_seed_context({}, instance)
    assert context["benign_project_id"] == "159"
    assert context["benign_project_path"] == "primer/design"
    assert context["benign_issue_iid"] == "104"
    assert context["benign_user_handle"] == "byteblaze"


def test_build_seed_context_with_reddit_anchors():
    instance = {
        "seed_task": {
            "id": "adv-2",
            "benign_task_id": "399",
            "benign_target_resource": {
                "kind": "reddit_submission",
                "anchors": {
                    "forum_name": "books",
                    "submission_id": 59421,
                },
                "attach_surfaces": [],
                "layer": "L1",
            },
            "agent_context": {
                "authentication": {"credentials": {"username": "MarvelsGrantMan136"}}
            },
        }
    }
    context = seeding._build_seed_context({}, instance)
    assert context["benign_forum_name"] == "books"
    assert context["benign_submission_id"] == "59421"
    assert context["benign_user_handle"] == "MarvelsGrantMan136"
    assert "benign_project_id" not in context


def test_build_seed_context_tolerates_missing_benign_target_resource():
    # Legacy tasks (non-Option A datasets) may lack the new field.
    # Context building must still succeed without the benign_* keys.
    instance = {"seed_task": {"id": "x", "instruction": "legacy"}}
    context = seeding._build_seed_context({}, instance)
    assert "benign_issue_iid" not in context
    assert "benign_user_handle" not in context
    assert context["task_id"] == "x"


def test_benign_issue_iid_token_renders_in_editor_call_args():
    # End-to-end: seed_template args carrying {benign_issue_iid} resolve
    # to the benign_target_resource.anchors.issue_iid at apply time via
    # the existing _FORMAT_TOKEN_PATTERN substitution. Exercise
    # _render_seed_template_in_context since that's the substitution path.
    seed_template = {
        "mechanism": "editor",
        "editor_calls": [
            {
                "benchmark": "webarena_verified",
                "site": "gitlab",
                "method": "create_issue_note",
                "args": {
                    "project_id": "{benign_project_id}",
                    "issue_iid": "{benign_issue_iid}",
                    "body": "test payload",
                },
            }
        ],
    }
    context = {
        "benign_project_id": "159",
        "benign_issue_iid": "104",
    }
    rendered = seeding._render_editor_seed_call(seed_template["editor_calls"][0], context)
    args = rendered["args"]
    assert args["project_id"] == "159"
    assert args["issue_iid"] == "104"
