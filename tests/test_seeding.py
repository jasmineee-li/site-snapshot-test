from __future__ import annotations

import types

import pytest

from worldsim import _sandbox_validator, seeding
from worldsim.seed_resolvers.types import ResolvedCall


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
        "site 'shopping' has HTTP-seeded task(s) but instance 'http://shopping.test' "
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
        "site 'gitlab' has HTTP-seeded task(s) but instance 'http://gitlab.test' has invalid auth config: "
        f"token_source {(tmp_path / 'logs/phase_0d/gitlab/missing.txt').resolve(strict=False)} does not exist"
    ]


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
                                "retrieved_data": [
                                    {"post_title": "Best place for a foot rub?"}
                                ]
                            }
                        }
                    ]
                },
            },
        },
    )

    post_call = [call for call in fake_session.calls if call["method"] == "POST"][0]
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


def test_apply_data_seed_resolves_targeted_shopping_review_calls(monkeypatch):
    fake_session = _FakeSession([_FakeResponse()])
    monkeypatch.setattr(seeding.requests, "Session", lambda: fake_session)
    monkeypatch.setattr(seeding, "_verify_http_seed_postcondition", lambda **kwargs: None)

    seeding.apply_data_seed(
        {
            "mechanism": "api",
            "api_calls": [
                {
                    "target": {
                        "benchmark": "webarena_verified",
                        "site": "shopping",
                        "resource_type": "product_review",
                        "create": {
                            "product_review": {
                                "title": "Fallback title",
                                "nickname": "seed-user",
                                "entity_pk_value": 67,
                            }
                        },
                    },
                    "body": {
                        "detail": "Payload text",
                    },
                }
            ],
        },
        {
            "site_name": "shopping",
            "site_url": "http://shopping.test",
            "api_auth": {"type": "bearer_token", "token": "demo-token"},
        },
    )

    assert fake_session.calls[0]["url"] == "http://shopping.test/rest/V1/reviews"
    assert fake_session.calls[0]["headers"]["Authorization"] == "Bearer demo-token"
    assert fake_session.calls[0]["json"] == {
        "review": {
            "title": "Fallback title",
            "detail": "Payload text",
            "nickname": "seed-user",
            "entity_pk_value": 67,
            "review_entity": "product",
            "stores": [1],
            "ratings": [{"rating_name": "Quality", "value": 4}],
        }
    }


def test_preflight_http_seed_calls_reports_incomplete_shopping_review_targets():
    errors = seeding.preflight_http_seed_calls(
        {
            "mechanism": "api",
            "api_calls": [
                {
                    "target": {
                        "benchmark": "webarena_verified",
                        "site": "shopping",
                        "resource_type": "product_review",
                        "create": {"product_review": {"entity_pk_value": 67}},
                    },
                    "body": {"detail": "Payload text"},
                }
            ],
        },
        {
            "site_name": "shopping",
            "site_url": "http://shopping.test",
            "api_auth": {"type": "bearer_token", "token": "demo-token"},
        },
    )

    assert errors == [
        "api_calls[0]: missing_required_fields: shopping product_review target is missing required fields: title, nickname"
    ]


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
            "api_auth": {"type": "bearer_token", "token": "demo-token"},
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

    post_call = [call for call in fake_session.calls if call["method"] == "POST"][0]
    assert post_call["url"] == "http://reddit.test/create_forum"
    assert post_call["data"]["form_key"] == "csrf-token"


def test_apply_data_seed_resolves_targeted_gitlab_merge_request_notes(monkeypatch):
    from worldsim.seed_resolvers import gitlab as gitlab_resolver

    fake_session = _FakeSession([_FakeResponse()])
    monkeypatch.setattr(seeding.requests, "Session", lambda: fake_session)
    monkeypatch.setattr(seeding, "_verify_http_seed_postcondition", lambda **kwargs: None)
    monkeypatch.setattr(
        gitlab_resolver,
        "create",
        lambda target, instance, seed_context: ResolvedCall(
            method="POST",
            url="http://gitlab.test/api/v4/projects/5/merge_requests/7/notes",
            headers={"PRIVATE-TOKEN": "glpat-demo"},
            context_additions={"project_id": 5, "mr_iid": 7},
        ),
    )

    seeding.apply_data_seed(
        {
            "mechanism": "api",
            "api_calls": [
                {
                    "method": "POST",
                    "target": {
                        "benchmark": "webarena_verified",
                        "site": "gitlab",
                        "resource_type": "mr_note",
                        "create": {
                            "project": {"name_template": "webagent-task-{task_id}"},
                            "mr": {"title_template": "Seed MR {task_id}"},
                        },
                    },
                    "body": {"body": "Injected note"},
                }
            ],
        },
        {
            "site_name": "gitlab",
            "site_url": "http://gitlab.test",
            "api_auth": {
                "type": "bearer_token",
                "header_name": "PRIVATE-TOKEN",
                "token": "glpat-demo",
            },
            "seed_task": {
                "instantiation_dict": {
                    "repo": "byteblaze/empathy-prompts",
                    "mr": "fix broken external links",
                }
            },
        },
    )

    assert (
        fake_session.calls[0]["url"]
        == "http://gitlab.test/api/v4/projects/5/merge_requests/7/notes"
    )
    assert fake_session.calls[0]["headers"]["PRIVATE-TOKEN"] == "glpat-demo"


def test_apply_data_seed_resolves_targeted_gitlab_calls_from_constraints_without_seed_task(
    monkeypatch,
):
    from worldsim.seed_resolvers import gitlab as gitlab_resolver

    fake_session = _FakeSession([_FakeResponse()])
    monkeypatch.setattr(seeding.requests, "Session", lambda: fake_session)
    monkeypatch.setattr(seeding, "_verify_http_seed_postcondition", lambda **kwargs: None)
    monkeypatch.setattr(
        gitlab_resolver,
        "create",
        lambda target, instance, seed_context: ResolvedCall(
            method="POST",
            url="http://gitlab.test/api/v4/projects/5/merge_requests/7/notes",
            headers={},
            context_additions={"project_id": 5, "mr_iid": 7},
        ),
    )

    seeding.apply_data_seed(
        {
            "mechanism": "api",
            "api_calls": [
                {
                    "method": "POST",
                    "target": {
                        "benchmark": "webarena_verified",
                        "site": "gitlab",
                        "resource_type": "mr_note",
                        "create": {
                            "project": {"name_template": "webagent-task-{task_id}"},
                            "mr": {"title_template": "Seed MR {task_id}"},
                        },
                    },
                    "body": {"body": "Injected note"},
                }
            ],
        },
        {
            "site_name": "gitlab",
            "site_url": "http://gitlab.test",
            "api_auth": {
                "type": "bearer_token",
                "header_name": "PRIVATE-TOKEN",
                "token": "glpat-demo",
            },
        },
    )

    assert (
        fake_session.calls[0]["url"]
        == "http://gitlab.test/api/v4/projects/5/merge_requests/7/notes"
    )


def test_preflight_http_seed_calls_reports_target_resolution_errors():
    errors = seeding.preflight_http_seed_calls(
        {
            "mechanism": "api",
            "api_calls": [
                {
                    "method": "PUT",
                    "target": {
                        "benchmark": "webarena_verified",
                        "site": "map",
                        "resource_type": "way",
                        "update": {},
                    },
                    "body": {"name": "payload"},
                }
            ],
        },
        {
            "site_name": "map",
            "site_url": "http://map.test",
            "seed_task": {"instantiation_dict": {}},
        },
    )

    assert errors == [
        "api_calls[0]: deferred_site: map target-based seeding is deferred; quarantine map tasks instead of migrating them"
    ]


def test_preflight_http_seed_calls_allows_gitlab_project_creation_targets(monkeypatch):
    errors = seeding.preflight_http_seed_calls(
        {
            "mechanism": "api",
            "api_calls": [
                {
                    "method": "POST",
                    "target": {
                        "benchmark": "webarena_verified",
                        "site": "gitlab",
                        "resource_type": "project",
                        "create": {
                            "project": {"name_template": "webagent-task-{task_id}"}
                        },
                    },
                    "body": {"name": "new-project", "path": "new-project"},
                }
            ],
        },
        {
            "site_name": "gitlab",
            "site_url": "http://gitlab.test",
            "api_auth": {
                "type": "bearer_token",
                "header_name": "PRIVATE-TOKEN",
                "token": "glpat-demo",
            },
        },
    )

    assert errors == []


def test_validate_data_seed_rejects_mixed_legacy_and_target_calls():
    with pytest.raises(
        ValueError,
        match="mixed legacy and target-based api_calls are not supported",
    ):
        seeding.validate_data_seed(
            {
                "mechanism": "form",
                "api_calls": [
                    {
                        "method": "POST",
                        "path": "/seed/bootstrap",
                        "body_form": {"detail": "bootstrap"},
                    },
                    {
                        "target": {
                            "benchmark": "webarena_verified",
                            "site": "reddit",
                            "resource_type": "comment",
                            "create": {
                                "forum": {"name_template": "books"},
                                "submission": {"title_template": "Thread"},
                            },
                        },
                        "body_form": {
                            "reply_to_submission_{submission_id}[comment]": "payload for {forum_name}"
                        },
                    },
                ],
            }
        )


def test_preflight_http_seed_calls_rejects_target_site_mismatch():
    errors = seeding.preflight_http_seed_calls(
        {
            "mechanism": "api",
            "api_calls": [
                {
                    "method": "POST",
                    "target": {
                        "benchmark": "webarena_verified",
                        "site": "gitlab",
                        "resource_type": "project",
                        "create": {
                            "project": {"name_template": "webagent-task-{task_id}"}
                        },
                    },
                    "body": {"name": "new-project", "path": "new-project"},
                }
            ],
        },
        {
            "site_name": "shopping",
            "site_url": "http://shopping.test",
        },
    )

    assert errors == [
        "api_calls[0]: target.site 'gitlab' does not match bound seed instance site 'shopping'"
    ]


def test_preflight_http_seed_calls_rejects_mixed_legacy_and_target_chains():
    with pytest.raises(
        ValueError,
        match="mixed legacy and target-based api_calls are not supported",
    ):
        seeding.preflight_http_seed_calls(
            {
                "mechanism": "form",
                "api_calls": [
                    {
                        "method": "POST",
                        "path": "/f/{forum_name}/{submission_id}/-/comment",
                        "body_form": {
                            "reply_to_submission_{submission_id}[comment]": "legacy payload",
                        },
                    },
                    {
                        "target": {
                            "site": "map",
                            "resource_type": "way",
                            "update": {},
                        },
                        "body_form": {"name": "payload"},
                    },
                ],
            },
            {
                "site_name": "map",
                "site_url": "http://map.test",
                "seed_task": {"instantiation_dict": {"forum_name": "books", "submission_id": 123}},
            },
        )

def test_validate_data_seed_requires_exactly_one_target_verb():
    with pytest.raises(
        ValueError,
        match="exactly one of target.create or target.update",
    ):
        seeding.validate_data_seed(
            {
                "mechanism": "api",
                "api_calls": [
                    {
                        "method": "POST",
                        "target": {
                            "site": "gitlab",
                            "resource_type": "project",
                        },
                        "body": {"name": "new-project"},
                    }
                ],
            }
        )


def test_validate_data_seed_requires_target_create_object():
    with pytest.raises(
        ValueError,
        match="targeted data seed calls must include target.create as an object",
    ):
        seeding.validate_data_seed(
            {
                "mechanism": "api",
                "api_calls": [
                    {
                        "target": {
                            "site": "gitlab",
                            "resource_type": "project",
                            "create": [],
                        },
                        "body": {"name": "new-project"},
                    }
                ],
            }
        )


def test_sandbox_validate_data_seed_rejects_form_seed_without_body_form():
    errors = _sandbox_validator.validate_data_seed(
        {
            "mechanism": "form",
            "api_calls": [
                {
                    "method": "POST",
                    "target": {
                        "site": "reddit",
                        "resource_type": "forum",
                        "create": {"forum": {"name_template": "books"}},
                    },
                    "body": {"forum[name]": "books"},
                }
            ],
        }
    )

    assert "form data seed calls must include a non-empty body_form object" in errors
    assert "form data seed calls must not include JSON body" in errors


def test_sandbox_validate_data_seed_rejects_api_seed_with_body_form():
    errors = _sandbox_validator.validate_data_seed(
        {
            "mechanism": "api",
            "api_calls": [
                {
                    "target": {
                        "site": "shopping",
                        "resource_type": "product_review",
                        "create": {"product_review": {"entity_pk_value": 7}},
                    },
                    "body_form": {"detail": "payload"},
                }
            ],
        }
    )

    assert errors == ["api data seed calls must use body, not body_form"]


def test_sandbox_validate_data_seed_rejects_mixed_legacy_and_target_calls():
    errors = _sandbox_validator.validate_data_seed(
        {
            "mechanism": "api",
            "api_calls": [
                {"method": "POST", "path": "/rest/V1/reviews", "body": {"detail": "legacy"}},
                {
                    "target": {
                        "site": "shopping",
                        "resource_type": "product_review",
                        "create": {"product_review": {"entity_pk_value": 7}},
                    },
                    "body": {"detail": "targeted"},
                },
            ],
        }
    )

    assert errors == [
        "mixed legacy and target-based api_calls are not supported; use one shape per seed"
    ]


def test_sandbox_validate_data_seed_requires_target_update_object():
    errors = _sandbox_validator.validate_data_seed(
        {
            "mechanism": "api",
            "api_calls": [
                {
                    "target": {
                        "site": "gitlab",
                        "resource_type": "user_status",
                        "update": [],
                    },
                    "body": {"message": "payload"},
                }
            ],
        }
    )

    assert errors == ["targeted data seed calls must include target.update as an object"]


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


def test_prepare_http_seed_call_rejects_unresolved_target_placeholders():
    with pytest.raises(
        RuntimeError,
        match="seed target has unresolved template placeholders: missing_forum",
    ):
        seeding._prepare_http_seed_call(
            {
                "target": {
                    "site": "reddit",
                    "resource_type": "forum",
                    "create": {"forum": {"name_template": "{missing_forum}"}},
                },
                "body_form": {"forum[name]": "books"},
            },
            {
                "site_name": "reddit",
                "site_url": "http://reddit.test",
            },
            mechanism="form",
            seed_context={},
        )


def test_apply_data_seed_renders_targeted_reddit_comment_keys_after_resolution(monkeypatch):
    from worldsim.seed_resolvers import reddit as reddit_resolver

    fake_session = _FakeSession(
        [
            _FakeResponse(text='<input name="form_key" value="csrf-token">'),
            _FakeResponse(),
        ]
    )
    monkeypatch.setattr(seeding.requests, "Session", lambda: fake_session)
    monkeypatch.setattr(seeding, "_verify_http_seed_postcondition", lambda **kwargs: None)
    monkeypatch.setattr(
        reddit_resolver,
        "create",
        lambda target, instance, seed_context: ResolvedCall(
            method="POST",
            url="http://reddit.test/f/books/59421/-/comment",
            headers={},
            context_additions={"forum_name": "books", "submission_id": 59421},
        ),
    )

    seeding.apply_data_seed(
        {
            "mechanism": "form",
            "api_calls": [
                {
                    "method": "POST",
                    "target": {
                        "benchmark": "webarena_verified",
                        "site": "reddit",
                        "resource_type": "comment",
                        "create": {
                            "forum": {"name_template": "books"},
                            "submission": {"title_template": "Books thread"},
                        },
                    },
                    "body_form": {
                        "reply_to_submission_{submission_id}[comment]": "payload",
                    },
                }
            ],
        },
        {
            "site_name": "reddit",
            "site_url": "http://reddit.test",
            "seed_task": {"instantiation_dict": {"forum": "books"}},
        },
    )

    post_call = [call for call in fake_session.calls if call["method"] == "POST"][0]
    assert post_call["url"] == "http://reddit.test/f/books/59421/-/comment"
    assert post_call["data"] == {
        "reply_to_submission_59421[comment]": "payload",
        "form_key": "csrf-token",
    }


def test_apply_data_seed_preserves_reddit_forum_submission_comment_chain(monkeypatch):
    fake_session = _FakeSession(
        [
            _FakeResponse(text='<input name="form_key" value="csrf-token">'),
            _FakeResponse(),
            _FakeResponse(text='<input name="form_key" value="csrf-token">'),
            _FakeResponse(headers={"Location": "/f/books/59421/seed-thread"}),
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
                    "target": {
                        "site": "reddit",
                        "resource_type": "forum",
                        "create": {"forum": {"name_template": "books"}},
                    },
                    "body_form": {"forum[name]": "books", "forum[description]": "desc"},
                },
                {
                    "target": {
                        "site": "reddit",
                        "resource_type": "submission",
                        "create": {
                            "forum": {"name_template": "books"},
                            "submission": {"title_template": "Books thread", "body_template": "body"},
                        },
                    },
                    "body_form": {"submission[title]": "Books thread", "submission[body]": "body"},
                },
                {
                    "target": {
                        "site": "reddit",
                        "resource_type": "comment",
                        "create": {
                            "forum": {"name_template": "books"},
                            "submission": {"title_template": "Books thread", "body_template": "body"},
                        },
                    },
                    "body_form": {"reply_to_submission_{submission_id}[comment]": "payload"},
                },
            ],
        },
        {"site_name": "reddit", "site_url": "http://reddit.test"},
    )

    post_urls = [call["url"] for call in fake_session.calls if call["method"] == "POST"]
    assert post_urls == [
        "http://reddit.test/create_forum",
        "http://reddit.test/submit/books",
        "http://reddit.test/f/books/59421/-/comment",
    ]


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
