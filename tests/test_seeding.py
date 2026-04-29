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


def test_validate_data_seed_rejects_api_mechanism():
    with pytest.raises(ValueError, match="deprecated"):
        seeding.validate_data_seed(
            {
                "mechanism": "api",
                "api_calls": [{"method": "POST", "path": "/x", "body": {}}],
            }
        )


def test_validate_data_seed_rejects_form_mechanism():
    with pytest.raises(ValueError, match="deprecated"):
        seeding.validate_data_seed(
            {
                "mechanism": "form",
                "api_calls": [{"method": "POST", "path": "/x", "body_form": {"k": "v"}}],
            }
        )


def test_validate_data_seed_rejects_state_push_mechanism():
    with pytest.raises(ValueError, match="deprecated"):
        seeding.validate_data_seed({"mechanism": "state_push", "state": {}})


def test_apply_data_seed_rejects_legacy_mechanisms_before_http(monkeypatch):
    """Legacy mechanisms must fail-fast at the validator, not after issuing HTTP."""
    sentinel: list = []
    monkeypatch.setattr(
        seeding.requests, "request", lambda *a, **kw: sentinel.append((a, kw)) or None
    )
    monkeypatch.setattr(seeding.requests, "Session", lambda: None)
    for mech, payload in (
        ("api", {"api_calls": [{"method": "GET", "path": "/x"}]}),
        ("form", {"api_calls": [{"method": "POST", "path": "/x", "body_form": {"k": "v"}}]}),
        ("state_push", {"state": {"key": "value"}}),
    ):
        with pytest.raises(ValueError, match="deprecated"):
            seeding.apply_data_seed(
                {"mechanism": mech, **payload},
                {"site_url": "http://x", "site_name": "x"},
            )
    assert sentinel == [], "no HTTP request should have been issued for deprecated mechanisms"


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
                    "mechanism": "editor",
                    "editor_calls": [
                        {
                            "benchmark": "webarena_verified",
                            "site": "shopping",
                            "method": "create_product_review",
                            "args": {"entity_pk_value": 7, "detail": "x"},
                        }
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
        "site 'shopping' has api HTTP-seeded task(s) but instance 'http://shopping.test' "
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
                    "mechanism": "editor",
                    "editor_calls": [
                        {
                            "benchmark": "webarena_verified",
                            "site": "gitlab",
                            "method": "create_issue",
                            "args": {
                                "project_id": "{benign_project_id}",
                                "title": "x",
                                "body_template": "x",
                            },
                        }
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
                    "mechanism": "editor",
                    "editor_calls": [
                        {
                            "benchmark": "webarena_verified",
                            "site": "gitlab",
                            "method": "create_issue",
                            "args": {
                                "project_id": "{benign_project_id}",
                                "title": "x",
                                "body_template": "x",
                            },
                        }
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
                "site": "shopping",
                "delivery_channel": {
                    "mechanism": "form",
                    "body_field": "detail",
                    "postcondition": {"type": "db_row_value"},
                },
                "adversarial_data_seed": {
                    "mechanism": "editor",
                    "editor_calls": [
                        {
                            "benchmark": "webarena_verified",
                            "site": "shopping",
                            "method": "create_product_review",
                            "args": {"entity_pk_value": 7, "detail": "x"},
                        }
                    ],
                },
            }
        ],
        [
            {
                "site_name": "shopping",
                "site_url": "http://shopping.test",
            }
        ],
        seed_field="adversarial_data_seed",
    )

    assert errors == []


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


def test_build_seed_context_prefers_seed_task_values():
    context = seeding._build_seed_context(
        {"mechanism": "none"},
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


def test_build_seed_context_exposes_gitlab_label_anchor():
    context = seeding._build_seed_context(
        {"mechanism": "none"},
        {
            "seed_task": {
                "id": "adv-103",
                "benign_target_resource": {
                    "anchors": {
                        "project_path": "kkroening/ffmpeg-python",
                        "label_names": "question",
                    }
                },
            }
        },
    )

    assert context["benign_project_path"] == "kkroening/ffmpeg-python"
    assert context["benign_label_names"] == "question"


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


def test_primary_created_resource_prefers_latest_render_surface():
    resources = [
        {
            "role": "seed_render_surface",
            "kind": "submission",
            "url": "http://reddit.test/f/books/10",
        },
        {
            "role": "seed_render_surface",
            "kind": "submission",
            "url": "http://reddit.test/f/books/11",
        },
    ]

    assert seeding._primary_created_resource(resources) == resources[1]


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
