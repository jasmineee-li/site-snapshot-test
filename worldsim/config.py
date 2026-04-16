"""Benchmark configuration schema.

Matches the JSON example in the v5 spec "Prerequisites" section. The user
hands the orchestrator a BenchmarkConfig; the orchestrator never starts,
stops, or provisions benchmark instances — it only connects to the URLs,
DBs, and reset endpoints supplied here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from worldsim.db_urls import parse_supported_db_connection
from worldsim.placeholders import merge_placeholder_maps

_SEEDING_AUTH_TYPES = frozenset({"none", "http_headers", "bearer_token", "web_login"})
_AGENT_AUTH_TYPES = frozenset({"none", "unknown", "storage_state", "http_basic", "http_headers"})


def _require_string_map(value: Any, *, field_name: str) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{field_name} must be a non-empty object")
    normalized: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{field_name} keys must be non-empty strings")
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field_name}[{key!r}] must be a non-empty string")
        normalized[key] = item
    return normalized


def _validate_form_login_recipe(recipe: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(recipe, dict):
        raise ValueError(f"{field_name} must be an object")
    required = (
        "login_url",
        "username_selector",
        "password_selector",
        "submit_selector",
        "success_url_substring",
    )
    normalized: dict[str, Any] = {}
    for key in required:
        value = recipe.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name}.{key} must be a non-empty string")
        normalized[key] = value.strip()
    return normalized


def _validate_seeding_auth_block(value: Any, *, field_name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    auth_type = str(value.get("type", "")).strip()
    if auth_type not in _SEEDING_AUTH_TYPES:
        raise ValueError(
            f"{field_name}.type must be one of {sorted(_SEEDING_AUTH_TYPES)}, got {auth_type!r}"
        )
    if auth_type == "none":
        return value
    if auth_type == "http_headers":
        headers = value.get("headers")
        if not isinstance(headers, dict) or not headers:
            raise ValueError(f"{field_name}.headers must be a non-empty object")
        for key, item in headers.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"{field_name}.headers keys must be non-empty strings")
            if isinstance(item, str) and item.strip():
                continue
            if (
                isinstance(item, dict)
                and isinstance(item.get("from_env"), str)
                and item["from_env"].strip()
            ):
                continue
            raise ValueError(
                f"{field_name}.headers[{key!r}] must be a non-empty string or {{'from_env': 'VAR'}}"
            )
        return value
    if auth_type == "bearer_token":
        header_name = value.get("header_name")
        if header_name is not None and (
            not isinstance(header_name, str) or not header_name.strip()
        ):
            raise ValueError(f"{field_name}.header_name must be a non-empty string when present")
        token = value.get("token")
        token_source = value.get("token_source")
        token_endpoint = value.get("token_endpoint")
        token_generator = value.get("token_generator")
        if isinstance(token, str) and token.strip():
            return value
        if isinstance(token_source, str) and token_source.strip():
            return value
        if isinstance(token_generator, str) and token_generator.strip():
            credentials = value.get("credentials")
            if not isinstance(credentials, dict) or not credentials:
                raise ValueError(
                    f"{field_name}.credentials must be a non-empty object when token_generator is set"
                )
            return value
        if isinstance(token_endpoint, str) and token_endpoint.strip():
            credentials = value.get("credentials")
            if not isinstance(credentials, dict) or not credentials:
                raise ValueError(
                    f"{field_name}.credentials must be a non-empty object when token_endpoint is set"
                )
            return value
        raise ValueError(
            f"{field_name} bearer_token auth requires token, token_source, token_generator, "
            "or token_endpoint"
        )
    if auth_type == "web_login":
        login_url = value.get("login_url")
        if not isinstance(login_url, str) or not login_url.strip():
            raise ValueError(f"{field_name}.login_url must be a non-empty string")
        credentials = _require_string_map(
            value.get("credentials"), field_name=f"{field_name}.credentials"
        )
        if "username" not in credentials or "password" not in credentials:
            raise ValueError(
                f"{field_name}.credentials must include non-empty 'username' and 'password'"
            )
        return value
    return value


def _validate_agent_auth_block(value: Any, *, field_name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    auth_type = str(value.get("type", "")).strip()
    if auth_type not in _AGENT_AUTH_TYPES:
        raise ValueError(
            f"{field_name}.type must be one of {sorted(_AGENT_AUTH_TYPES)}, got {auth_type!r}"
        )
    if auth_type == "none":
        return value
    if auth_type == "unknown":
        notes = value.get("notes")
        if not isinstance(notes, str) or not notes.strip():
            raise ValueError(f"{field_name}.notes must be a non-empty string when type='unknown'")
        return value
    if auth_type == "storage_state":
        storage_state = value.get("storage_state")
        if not isinstance(storage_state, dict):
            raise ValueError(f"{field_name}.storage_state must be an object")
        path = storage_state.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"{field_name}.storage_state.path must be a non-empty string")
        generator_script = storage_state.get("generator_script")
        if generator_script is not None and (
            not isinstance(generator_script, str) or not generator_script.strip()
        ):
            raise ValueError(
                f"{field_name}.storage_state.generator_script must be a non-empty string when present"
            )
        per_task_refresh = storage_state.get("per_task_refresh")
        if per_task_refresh is not None and not isinstance(per_task_refresh, bool):
            raise ValueError(f"{field_name}.storage_state.per_task_refresh must be a boolean")
        form_login = storage_state.get("form_login")
        if form_login is not None:
            _validate_form_login_recipe(
                form_login,
                field_name=f"{field_name}.storage_state.form_login",
            )
        authentication = value.get("authentication")
        if authentication is not None and not isinstance(authentication, dict):
            raise ValueError(f"{field_name}.authentication must be an object when present")
        return value
    if auth_type == "http_basic":
        http_basic = value.get("http_basic")
        credentials = _require_string_map(http_basic, field_name=f"{field_name}.http_basic")
        if "username" not in credentials or "password" not in credentials:
            raise ValueError(
                f"{field_name}.http_basic must include non-empty 'username' and 'password'"
            )
        return value
    if auth_type == "http_headers":
        http_headers = value.get("http_headers")
        if not isinstance(http_headers, dict):
            raise ValueError(f"{field_name}.http_headers must be an object")
        _require_string_map(
            http_headers.get("headers"), field_name=f"{field_name}.http_headers.headers"
        )
        authentication = value.get("authentication")
        if authentication is not None and not isinstance(authentication, dict):
            raise ValueError(f"{field_name}.authentication must be an object when present")
        return value
    return value


def has_configured_agent_auth(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    auth_type = str(value.get("type", "")).strip()
    return bool(auth_type) and auth_type != "unknown"


class BenchmarkInstance(BaseModel):
    """One pre-running benchmark instance.

    For parallel evaluation, supply multiple instances of the same site on
    different ports. Workers are distributed across them in
    ``worldsim.eval_worker_pool.run_eval``.
    """

    site_name: str = Field(..., description="E.g. 'shopping', 'forum', 'gitlab'.")
    site_url: str = Field(..., description="Base URL of the running service.")
    db_connection: str | None = Field(
        None,
        description="Connection string for data seeding, e.g. 'mysql://user:pass@host:3306/db'. "
        "Optional: not every seeding mechanism needs a DB (see worldsim.seeding).",
    )
    reset_endpoint: str | None = Field(
        None,
        description="HTTP endpoint called between tasks to restore initial state. "
        "Optional but strongly recommended for Phase 3/4 reproducibility.",
    )
    url_placeholders: dict[str, str] = Field(
        default_factory=dict,
        description="Maps URL template tokens (e.g. __SHOPPING__) to actual URLs. "
        "Used for resolving cross-site URL references in reward evaluation.",
    )
    auth: dict[str, Any] | None = Field(
        None,
        description="Optional seeding-time auth configuration for form-mechanism seeds, "
        "e.g. static auto-login headers or a bearer token loaded from disk.",
    )
    api_auth: dict[str, Any] | None = Field(
        None,
        description="Optional seeding-time auth for api-mechanism seeds, e.g. an admin "
        "bearer token acquired from a REST endpoint. Falls back to ``auth``.",
    )
    agent_auth: dict[str, Any] | None = Field(
        None,
        description="Agent Playwright auth derived from instance config. Overrides "
        "Phase 0c's auth_mechanism for reliable, static auth injection.",
    )

    @field_validator("db_connection")
    @classmethod
    def _validate_db_connection(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parse_supported_db_connection(value, purpose="BenchmarkInstance.db_connection")
        return value.strip()

    @field_validator("auth", "api_auth")
    @classmethod
    def _validate_seeding_auth(
        cls, value: dict[str, Any] | None, info: Any
    ) -> dict[str, Any] | None:
        return _validate_seeding_auth_block(
            value, field_name=f"BenchmarkInstance.{info.field_name}"
        )

    @field_validator("agent_auth")
    @classmethod
    def _validate_agent_auth(cls, value: dict[str, Any] | None, info: Any) -> dict[str, Any] | None:
        return _validate_agent_auth_block(value, field_name=f"BenchmarkInstance.{info.field_name}")


class VerificationProxy(BaseModel):
    """Optional authenticated reverse proxy for Phase 0c live verification.

    When configured, Phase 0c computes proxy URLs by applying port_offset to
    each site's real port and includes the token as an auth header so Modal
    sandboxes can reach live instances through a token-gated nginx proxy.
    Deployed by ``scripts/deploy_benchmark_proxy.sh``.
    """

    token: str = Field(
        "",
        description="Shared secret for X-Worldsim-Token header. Empty string disables proxy auth.",
    )
    port_offset: int = Field(
        10000,
        description="Added to each site's real port to derive the proxy port. "
        "E.g. real port 7770 -> proxy port 17770.",
    )


class BenchmarkConfig(BaseModel):
    """Top-level benchmark configuration supplied by the user."""

    benchmark_name: str
    instances: list[BenchmarkInstance]
    verification_proxy: VerificationProxy | None = Field(
        None,
        description="Optional authenticated reverse proxy config for Phase 0c. "
        "When present with a non-empty token, Phase 0c uses proxy URLs and "
        "injects the X-Worldsim-Token header for live instance verification.",
    )
    url_placeholders: dict[str, str] = Field(
        default_factory=dict,
        description="Top-level mapping of URL template tokens to live site URLs. "
        "This is the canonical config shape from the v5 spec.",
    )
    benchmark_codebase: Path = Field(
        ...,
        description="Absolute path to the benchmark's source tree. Used in "
        "Phase 0 (read-only reconnaissance). Must exist on disk.",
    )

    @model_validator(mode="after")
    def _merge_legacy_instance_placeholders(self) -> BenchmarkConfig:
        """Preserve backwards compatibility with older per-instance configs."""
        self.url_placeholders = merge_placeholder_maps(
            self.url_placeholders,
            *[instance.url_placeholders for instance in self.instances],
        )
        return self
