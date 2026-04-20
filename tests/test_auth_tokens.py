"""Tests for worldsim.auth_tokens -- runtime token acquisition and validation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from worldsim import auth_tokens

# ── Helpers ──────────────────────────────────────────────────────────────


class _FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        text: str = "",
        json_data: object = None,
        content_type: str = "text/html",
    ):
        self.status_code = status_code
        self.text = text
        self._json_data = json_data
        self.headers = {"Content-Type": content_type}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"status={self.status_code}")

    def json(self) -> object:
        if self._json_data is not None:
            return self._json_data
        raise ValueError("no json")


def _gitlab_login_page_html(csrf: str = "csrf-login-token") -> str:
    return f'<meta name="csrf-token" content="{csrf}">'


def _gitlab_pat_form_html(csrf: str = "csrf-pat-token") -> str:
    return f'<meta name="csrf-token" content="{csrf}">'


# ── acquire_token tests ──────────────────────────────────────────────────


def test_acquire_token_with_token_endpoint(monkeypatch):
    """token_endpoint dispatches to EndpointTokenGenerator."""
    auth_tokens._BEARER_API_TOKEN_CACHE = {}

    fake_resp = _FakeResponse(json_data='"admin-token-123"', content_type="application/json")
    mock_post = MagicMock(return_value=fake_resp)
    monkeypatch.setattr(auth_tokens.requests, "post", mock_post)

    token = auth_tokens.acquire_token(
        {
            "type": "bearer_token",
            "token_endpoint": "/rest/V1/integration/admin/token",
            "credentials": {"username": "admin", "password": "admin1234"},
        },
        "http://shopping.test",
    )

    assert token == "admin-token-123"
    mock_post.assert_called_once()
    call_url = mock_post.call_args[0][0]
    assert call_url == "http://shopping.test/rest/V1/integration/admin/token"


def test_acquire_token_with_inline_token():
    """Inline token is returned as-is."""
    token = auth_tokens.acquire_token(
        {"type": "bearer_token", "token": "static-token-value"},
        "http://example.test",
    )
    assert token == "static-token-value"


def test_acquire_token_with_token_source(tmp_path):
    token_path = tmp_path / "token.txt"
    token_path.write_text("file-token\n", encoding="utf-8")

    token = auth_tokens.acquire_token(
        {"type": "bearer_token", "token_source": str(token_path)},
        "http://example.test",
    )

    assert token == "file-token"


def test_acquire_token_with_unknown_generator():
    """Unknown generator name raises RuntimeError."""
    with pytest.raises(RuntimeError, match="unknown token_generator"):
        auth_tokens.acquire_token(
            {
                "type": "bearer_token",
                "token_generator": "nonexistent",
                "credentials": {"username": "x", "password": "y"},
            },
            "http://example.test",
        )


def test_acquire_token_no_strategy_raises():
    """No resolution strategy raises RuntimeError."""
    with pytest.raises(
        RuntimeError, match="no token_generator, token_endpoint, token, or token_source"
    ):
        auth_tokens.acquire_token({"type": "bearer_token"}, "http://example.test")


# ── GitLabPATGenerator tests ────────────────────────────────────────────


def test_gitlab_pat_generator_full_flow(monkeypatch):
    """GitLab PAT generator follows the login -> form -> create flow."""
    responses = [
        # GET /users/sign_in
        _FakeResponse(text=_gitlab_login_page_html("login-csrf")),
        # POST /users/sign_in
        _FakeResponse(status_code=200, text="redirected to dashboard"),
        # GET /-/profile/personal_access_tokens
        _FakeResponse(text=_gitlab_pat_form_html("pat-csrf")),
        # POST /-/profile/personal_access_tokens
        _FakeResponse(
            json_data={"new_token": "glpat-fresh-token-123"},
            content_type="application/json",
        ),
    ]
    call_index = {"i": 0}

    def fake_request(method, url, **kwargs):
        resp = responses[call_index["i"]]
        call_index["i"] += 1
        return resp

    mock_session = MagicMock()
    mock_session.get = lambda url, **kw: fake_request("GET", url, **kw)
    mock_session.post = lambda url, **kw: fake_request("POST", url, **kw)
    monkeypatch.setattr(auth_tokens.requests, "Session", lambda: mock_session)

    token = auth_tokens.acquire_token(
        {
            "type": "bearer_token",
            "token_generator": "gitlab_pat",
            "credentials": {"username": "byteblaze", "password": "hello1234"},
        },
        "http://gitlab.test:8023",
    )

    assert token == "glpat-fresh-token-123"


def test_gitlab_pat_generator_missing_credentials():
    """GitLab PAT generator fails fast on missing credentials."""
    with pytest.raises(RuntimeError, match="requires username and password"):
        gen = auth_tokens.GitLabPATGenerator()
        gen.generate({}, "http://gitlab.test")


def test_gitlab_pat_generator_html_token_fallback(monkeypatch):
    """GitLab PAT generator falls back to HTML input for the created token."""
    responses = [
        _FakeResponse(text=_gitlab_login_page_html()),
        _FakeResponse(status_code=200),
        _FakeResponse(text=_gitlab_pat_form_html()),
        _FakeResponse(
            text='<input id="created-personal-access-token" value="glpat-html-tok">',
            content_type="text/html",
        ),
    ]
    call_index = {"i": 0}

    def fake_request(method, url, **kwargs):
        resp = responses[call_index["i"]]
        call_index["i"] += 1
        return resp

    mock_session = MagicMock()
    mock_session.get = lambda url, **kw: fake_request("GET", url, **kw)
    mock_session.post = lambda url, **kw: fake_request("POST", url, **kw)
    monkeypatch.setattr(auth_tokens.requests, "Session", lambda: mock_session)

    gen = auth_tokens.GitLabPATGenerator()
    token = gen.generate(
        {"username": "root", "password": "pass"},
        "http://gitlab.test",
    )
    assert token == "glpat-html-tok"


# ── validate_token tests ────────────────────────────────────────────────


def test_validate_token_with_gitlab_generator(monkeypatch):
    """validate_token delegates to the generator's validate method."""
    mock_get = MagicMock(return_value=_FakeResponse(status_code=200))
    monkeypatch.setattr(auth_tokens.requests, "get", mock_get)

    result = auth_tokens.validate_token(
        "glpat-valid",
        {"type": "bearer_token", "token_generator": "gitlab_pat"},
        "http://gitlab.test",
    )

    assert result is True
    mock_get.assert_called_once()
    call_headers = mock_get.call_args[1]["headers"]
    assert call_headers["PRIVATE-TOKEN"] == "glpat-valid"


def test_validate_token_with_expired_token(monkeypatch):
    """validate_token returns False for 401 responses."""
    mock_get = MagicMock(return_value=_FakeResponse(status_code=401))
    monkeypatch.setattr(auth_tokens.requests, "get", mock_get)

    result = auth_tokens.validate_token(
        "glpat-expired",
        {"type": "bearer_token", "token_generator": "gitlab_pat"},
        "http://gitlab.test",
    )

    assert result is False


def test_validate_token_with_validation_endpoint(monkeypatch):
    """validate_token uses explicit validation_endpoint when no generator is set."""
    mock_get = MagicMock(return_value=_FakeResponse(status_code=200))
    monkeypatch.setattr(auth_tokens.requests, "get", mock_get)

    result = auth_tokens.validate_token(
        "some-token",
        {"type": "bearer_token", "header_name": "X-Api-Key"},
        "http://api.test",
        validation_endpoint="/health",
    )

    assert result is True
    call_url = mock_get.call_args[0][0]
    assert call_url == "http://api.test/health"
    assert mock_get.call_args[1]["headers"]["X-Api-Key"] == "some-token"


def test_validate_token_without_strategy_returns_false():
    result = auth_tokens.validate_token(
        "some-token",
        {"type": "bearer_token"},
        "http://example.test",
    )
    assert result is False


# ── acquire_tokens_for_instances tests ───────────────────────────────────


def test_acquire_tokens_for_instances_injects_token(monkeypatch):
    """acquire_tokens_for_instances injects fresh tokens into instance auth dicts."""
    auth_tokens.clear_run_token_cache()

    fake_resp = _FakeResponse(json_data='"fresh-admin-token"', content_type="application/json")
    mock_post = MagicMock(return_value=fake_resp)
    mock_get = MagicMock(return_value=_FakeResponse(status_code=200))
    monkeypatch.setattr(auth_tokens.requests, "post", mock_post)
    monkeypatch.setattr(auth_tokens.requests, "get", mock_get)

    instances = [
        {
            "site_name": "shopping",
            "site_url": "http://shopping.test",
            "api_auth": {
                "type": "bearer_token",
                "token_endpoint": "/rest/V1/integration/admin/token",
                "validation_endpoint": "/rest/V1/customers/me",
                "credentials": {"username": "admin", "password": "admin1234"},
            },
        }
    ]

    errors = auth_tokens.acquire_tokens_for_instances(instances)

    assert errors == []
    assert instances[0]["api_auth"]["token"] == "fresh-admin-token"


def test_acquire_tokens_for_instances_skips_non_bearer(monkeypatch):
    """acquire_tokens_for_instances ignores non-bearer_token auth blocks."""
    auth_tokens.clear_run_token_cache()

    instances = [
        {
            "site_name": "shopping",
            "site_url": "http://shopping.test",
            "auth": {
                "type": "http_headers",
                "headers": {"X-Test": "value"},
            },
        }
    ]

    errors = auth_tokens.acquire_tokens_for_instances(instances)
    assert errors == []


def test_acquire_tokens_for_instances_requires_validation_for_legacy_token_source():
    auth_tokens.clear_run_token_cache()

    instances = [
        {
            "site_name": "gitlab",
            "site_url": "http://gitlab.test",
            "auth": {
                "type": "bearer_token",
                "header_name": "PRIVATE-TOKEN",
                "token_source": "logs/phase_0d/gitlab/token.txt",
            },
        }
    ]

    errors = auth_tokens.acquire_tokens_for_instances(instances)
    assert len(errors) == 1
    assert "bearer_token_unvalidated" in errors[0]
    assert "token" not in instances[0]["auth"]


def test_acquire_tokens_for_instances_accepts_validated_token_source(tmp_path, monkeypatch):
    auth_tokens.clear_run_token_cache()
    token_path = tmp_path / "token.txt"
    token_path.write_text("glpat-file\n", encoding="utf-8")
    monkeypatch.setattr(
        auth_tokens.requests,
        "get",
        MagicMock(return_value=_FakeResponse(status_code=200)),
    )

    instances = [
        {
            "site_name": "gitlab",
            "site_url": "http://gitlab.test",
            "auth": {
                "type": "bearer_token",
                "header_name": "PRIVATE-TOKEN",
                "token_source": str(token_path),
                "validation_endpoint": "/api/v4/user",
            },
        }
    ]

    errors = auth_tokens.acquire_tokens_for_instances(instances)
    assert errors == []
    assert instances[0]["auth"]["token"] == "glpat-file"


def test_acquire_tokens_for_instances_reports_failures(monkeypatch):
    """acquire_tokens_for_instances collects errors without raising."""
    auth_tokens.clear_run_token_cache()

    def failing_post(*args, **kwargs):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(auth_tokens.requests, "post", failing_post)

    instances = [
        {
            "site_name": "shopping",
            "site_url": "http://shopping.test",
            "api_auth": {
                "type": "bearer_token",
                "token_endpoint": "/rest/V1/integration/admin/token",
                "validation_endpoint": "/rest/V1/customers/me",
                "credentials": {"username": "admin", "password": "admin1234"},
            },
        }
    ]

    errors = auth_tokens.acquire_tokens_for_instances(instances)
    assert len(errors) == 1
    assert "shopping" in errors[0]
    assert "connection refused" in errors[0]


def test_acquire_tokens_for_instances_caches_across_calls(monkeypatch):
    """Second call for same instance uses cached token without hitting network."""
    auth_tokens.clear_run_token_cache()

    call_count = {"n": 0}

    def counting_post(*args, **kwargs):
        call_count["n"] += 1
        return _FakeResponse(json_data='"cached-tok"', content_type="application/json")

    monkeypatch.setattr(auth_tokens.requests, "post", counting_post)
    monkeypatch.setattr(
        auth_tokens.requests,
        "get",
        MagicMock(return_value=_FakeResponse(status_code=200)),
    )

    instances = [
        {
            "site_name": "shopping",
            "site_url": "http://shopping.test",
            "api_auth": {
                "type": "bearer_token",
                "token_endpoint": "/rest/V1/integration/admin/token",
                "validation_endpoint": "/rest/V1/customers/me",
                "credentials": {"username": "admin", "password": "admin1234"},
            },
        }
    ]

    auth_tokens.acquire_tokens_for_instances(instances)
    auth_tokens.acquire_tokens_for_instances(instances)

    assert call_count["n"] == 1
    assert instances[0]["api_auth"]["token"] == "cached-tok"


def test_acquire_tokens_for_instances_validates_inline_token(monkeypatch):
    auth_tokens.clear_run_token_cache()
    monkeypatch.setattr(
        auth_tokens.requests,
        "get",
        MagicMock(return_value=_FakeResponse(status_code=200)),
    )

    instances = [
        {
            "site_name": "gitlab",
            "site_url": "http://gitlab.test",
            "auth": {
                "type": "bearer_token",
                "header_name": "PRIVATE-TOKEN",
                "token": "glpat-inline",
                "validation_endpoint": "/api/v4/user",
            },
        }
    ]

    errors = auth_tokens.acquire_tokens_for_instances(instances)
    assert errors == []
    assert instances[0]["auth"]["token"] == "glpat-inline"


def test_acquire_tokens_for_instances_rejects_failed_inline_validation(monkeypatch):
    auth_tokens.clear_run_token_cache()
    monkeypatch.setattr(
        auth_tokens.requests,
        "get",
        MagicMock(return_value=_FakeResponse(status_code=401)),
    )

    instances = [
        {
            "site_name": "gitlab",
            "site_url": "http://gitlab.test",
            "auth": {
                "type": "bearer_token",
                "header_name": "PRIVATE-TOKEN",
                "token": "glpat-inline",
                "validation_endpoint": "/api/v4/user",
            },
        }
    ]

    errors = auth_tokens.acquire_tokens_for_instances(instances)
    assert len(errors) == 1
    assert "bearer_token_unvalidated" in errors[0]


# ── Config validation tests ─────────────────────────────────────────────


def test_config_accepts_token_generator():
    """BenchmarkInstance validates bearer_token with token_generator."""
    from worldsim.config import BenchmarkInstance

    instance = BenchmarkInstance(
        site_name="gitlab",
        site_url="http://gitlab.test",
        auth={
            "type": "bearer_token",
            "header_name": "PRIVATE-TOKEN",
            "token_generator": "gitlab_pat",
            "credentials": {"username": "root", "password": "pass"},
        },
    )
    assert instance.auth["token_generator"] == "gitlab_pat"


def test_config_rejects_token_generator_without_credentials():
    """BenchmarkInstance rejects token_generator without credentials."""
    from worldsim.config import BenchmarkInstance

    with pytest.raises(ValueError, match="credentials must be a non-empty object"):
        BenchmarkInstance(
            site_name="gitlab",
            site_url="http://gitlab.test",
            auth={
                "type": "bearer_token",
                "token_generator": "gitlab_pat",
            },
        )


def test_benchmark_config_rejects_empty_instances(tmp_path):
    from worldsim.config import BenchmarkConfig

    with pytest.raises(ValueError, match="at least one instance"):
        BenchmarkConfig.model_validate(
            {
                "benchmark_name": "WebArena Verified",
                "benchmark_codebase": str(tmp_path),
                "instances": [],
            }
        )


# ── Seeding integration tests ───────────────────────────────────────────


def test_seeding_resolve_bearer_token_uses_token_generator(monkeypatch):
    """_resolve_bearer_token dispatches to auth_tokens for token_generator."""
    from worldsim import seeding

    seeding._BEARER_API_TOKEN_CACHE.clear()

    responses = [
        _FakeResponse(text=_gitlab_login_page_html()),
        _FakeResponse(status_code=200),
        _FakeResponse(text=_gitlab_pat_form_html()),
        _FakeResponse(
            json_data={"new_token": "glpat-seeding-test"},
            content_type="application/json",
        ),
    ]
    call_index = {"i": 0}

    def fake_request(method, url, **kwargs):
        resp = responses[call_index["i"]]
        call_index["i"] += 1
        return resp

    mock_session = MagicMock()
    mock_session.get = lambda url, **kw: fake_request("GET", url, **kw)
    mock_session.post = lambda url, **kw: fake_request("POST", url, **kw)
    monkeypatch.setattr(auth_tokens.requests, "Session", lambda: mock_session)

    token = seeding._resolve_bearer_token(
        {
            "type": "bearer_token",
            "header_name": "PRIVATE-TOKEN",
            "token_generator": "gitlab_pat",
            "credentials": {"username": "byteblaze", "password": "hello1234"},
        },
        site_url="http://gitlab.test:8023",
    )

    assert token == "glpat-seeding-test"


def test_seeding_resolve_bearer_token_caches_generated_token(monkeypatch):
    """Generated tokens are cached in _BEARER_API_TOKEN_CACHE."""
    from worldsim import seeding

    seeding._BEARER_API_TOKEN_CACHE.clear()

    call_count = {"n": 0}

    def mock_acquire(auth_config, site_url):
        call_count["n"] += 1
        return "glpat-cached"

    monkeypatch.setattr(auth_tokens, "acquire_token", mock_acquire)

    auth = {
        "type": "bearer_token",
        "token_generator": "gitlab_pat",
        "credentials": {"username": "root", "password": "pass"},
    }

    token1 = seeding._resolve_bearer_token(auth, site_url="http://gitlab.test")
    token2 = seeding._resolve_bearer_token(auth, site_url="http://gitlab.test")

    assert token1 == "glpat-cached"
    assert token2 == "glpat-cached"
    assert call_count["n"] == 1


def test_seeding_runtime_errors_accepts_token_generator():
    """collect_seed_runtime_errors passes for token_generator auth."""
    from worldsim import seeding

    errors = seeding.collect_seed_runtime_errors(
        [
            {
                "id": "task-1",
                "site": "gitlab",
                "adversarial_data_seed": {
                    "mechanism": "api",
                    "api_calls": [
                        {"method": "POST", "path": "/api/v4/issues", "body": {"detail": "x"}}
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
                },
            }
        ],
        seed_field="adversarial_data_seed",
    )

    assert errors == []
