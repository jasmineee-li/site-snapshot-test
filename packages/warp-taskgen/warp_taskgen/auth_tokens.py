"""Runtime token acquisition and validation for benchmark instances.

Eliminates stale-token failures by deriving auth tokens from credentials at
pipeline startup.  Three resolution strategies are supported, checked in order:

1. ``token_endpoint`` -- POST credentials to a REST endpoint (e.g. Magento).
2. ``token_generator`` -- protocol-specific token creation (e.g. GitLab PAT
   via web-login + API form submission).
3. ``token_source`` (legacy) -- read a token from disk, but *validate* it
   against the live instance before use.

Tokens acquired at startup are cached for the lifetime of the pipeline run
via ``acquire_tokens_for_instances``, which replaces file-based tokens with
live ones so downstream code never touches disk.
"""

from __future__ import annotations

import concurrent.futures
import datetime as dt
import hashlib
import json
import logging
import os
import re
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

logger = logging.getLogger(__name__)

_GITLAB_RUNTIME_PAT_NAME = "worldsim-runtime"
_GITLAB_LEGACY_PAT_NAMES = (_GITLAB_RUNTIME_PAT_NAME, "worldsim-api")
_TOKEN_VALIDATION_TTL_SECONDS = 1.0

# ── Public interface ─────────────────────────────────────────────────────

# Registry of built-in token generators keyed by name.
_TOKEN_GENERATORS: dict[str, type[TokenGenerator]] = {}


class TokenGenerator:
    """Base class for protocol-specific token generators.

    Subclasses implement ``generate`` and ``validate``.  Registration happens
    automatically via ``__init_subclass__``.
    """

    generator_name: str = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.generator_name:
            _TOKEN_GENERATORS[cls.generator_name] = cls

    def generate(self, credentials: dict[str, str], site_url: str) -> str:
        """Create a fresh token and return it as a string."""
        raise NotImplementedError

    def validate(self, token: str, site_url: str) -> bool:
        """Return True when *token* is accepted by the live instance."""
        raise NotImplementedError


# ── Built-in generators ─────────────────────────────────────────────────


class GitLabPATGenerator(TokenGenerator):
    """Create a GitLab personal access token via web-login + form POST.

    Flow:
      1. POST ``/users/sign_in`` with username/password to get a session cookie.
      2. GET ``/-/profile/personal_access_tokens`` to extract the CSRF token.
      3. POST the PAT form to create a token with ``api`` scope.
      4. Return the ``glpat-...`` string.
    """

    generator_name = "gitlab_pat"

    _AUTH_TOKEN_RE = re.compile(
        r'name=["\']authenticity_token["\'][^>]*value=["\']([^"\']+)["\']',
        re.IGNORECASE,
    )
    _CSRF_META_RE = re.compile(
        r'<meta\s+name=["\']csrf-token["\']\s+content=["\']([^"\']+)["\']',
        re.IGNORECASE,
    )
    _REVOKE_FORM_RE = re.compile(
        r"<tr[^>]*>.*?(?:"
        + "|".join(re.escape(name) for name in _GITLAB_LEGACY_PAT_NAMES)
        + r")[^<]*.*?<form[^>]+action=[\"']([^\"']+/revoke)[\"']",
        re.IGNORECASE | re.DOTALL,
    )

    def generate(self, credentials: dict[str, str], site_url: str) -> str:
        username = str(credentials.get("username", "")).strip()
        password = str(credentials.get("password", ""))
        if not username or not password.strip():
            raise RuntimeError("gitlab_pat generator requires username and password credentials")

        base = site_url.rstrip("/")
        session = requests.Session()

        # Step 1: GET the sign-in page for a CSRF token.
        login_page = session.get(f"{base}/users/sign_in", timeout=30)
        login_page.raise_for_status()
        csrf = self._extract_csrf(login_page.text)
        if not csrf:
            raise RuntimeError("gitlab_pat: could not extract CSRF token from sign-in page")

        # Step 2: POST credentials to sign in.
        sign_in_resp = session.post(
            f"{base}/users/sign_in",
            data={
                "authenticity_token": csrf,
                "user[login]": username,
                "user[password]": password,
            },
            timeout=30,
            allow_redirects=True,
        )
        # GitLab redirects to / on success.
        if sign_in_resp.status_code >= 400:
            raise RuntimeError(f"gitlab_pat: sign-in failed with HTTP {sign_in_resp.status_code}")

        # Step 3: GET the PAT form page.
        pat_url = f"{base}/-/profile/personal_access_tokens"
        form_resp = session.get(pat_url, timeout=30)
        form_resp.raise_for_status()
        form_csrf = self._extract_csrf(form_resp.text)
        if not form_csrf:
            raise RuntimeError("gitlab_pat: could not extract CSRF token from PAT form page")
        self._best_effort_revoke_runtime_tokens(
            session,
            site_url=base,
            html=form_resp.text,
            csrf_token=form_csrf,
        )

        # Step 4: POST to create the PAT.
        create_resp = session.post(
            pat_url,
            data={
                "authenticity_token": form_csrf,
                "personal_access_token[name]": _GITLAB_RUNTIME_PAT_NAME,
                "personal_access_token[expires_at]": self._expires_at(),
                "personal_access_token[scopes][]": "api",
            },
            headers={"X-CSRF-Token": form_csrf},
            timeout=30,
        )
        create_resp.raise_for_status()

        # Parse the response -- modern GitLab returns JSON, older versions
        # embed the token in an HTML input.
        token = self._parse_created_token(create_resp)
        if not token:
            raise RuntimeError("gitlab_pat: PAT creation response did not contain a token value")
        logger.info("gitlab_pat: acquired fresh PAT for %s@%s", username, base)
        return token

    def validate(self, token: str, site_url: str) -> bool:
        base = site_url.rstrip("/")
        try:
            resp = requests.get(
                f"{base}/api/v4/user",
                headers={"PRIVATE-TOKEN": token},
                timeout=15,
            )
            return resp.status_code == 200
        except requests.RequestException:
            return False

    # ── helpers ──────────────────────────────────────────────────────────

    def _extract_csrf(self, html: str) -> str | None:
        match = self._AUTH_TOKEN_RE.search(html) or self._CSRF_META_RE.search(html)
        return match.group(1) if match else None

    @staticmethod
    def _expires_at() -> str:
        return (dt.date.today() + dt.timedelta(days=1)).isoformat()

    def _best_effort_revoke_runtime_tokens(
        self,
        session: requests.Session,
        *,
        site_url: str,
        html: str,
        csrf_token: str,
    ) -> None:
        seen: set[str] = set()
        expected_origin = self._origin(site_url)
        for action in self._REVOKE_FORM_RE.findall(html):
            if action in seen:
                continue
            seen.add(action)
            if action.startswith("http"):
                if self._origin(action) != expected_origin:
                    logger.warning(
                        "gitlab_pat: refusing cross-origin runtime PAT revoke target %s",
                        action,
                    )
                    continue
                revoke_url = action
            else:
                revoke_url = f"{site_url.rstrip('/')}{action}"
            try:
                response = session.post(
                    revoke_url,
                    data={"authenticity_token": csrf_token},
                    headers={"X-CSRF-Token": csrf_token},
                    timeout=30,
                    allow_redirects=False,
                )
            except requests.RequestException as exc:
                logger.warning("gitlab_pat: failed to revoke runtime PAT via %s: %s", action, exc)
                continue
            if response.status_code >= 400:
                logger.warning(
                    "gitlab_pat: runtime PAT revoke via %s returned HTTP %s",
                    action,
                    response.status_code,
                )

    @staticmethod
    def _origin(url: str) -> tuple[str, str]:
        parts = urlsplit(url)
        return (parts.scheme.lower(), parts.netloc.lower())

    @staticmethod
    def _parse_created_token(resp: requests.Response) -> str | None:
        content_type = resp.headers.get("Content-Type", "")
        if "application/json" in content_type:
            try:
                data = resp.json()
                token = data.get("new_token") or data.get("token")
                if isinstance(token, str) and token.strip():
                    return token.strip()
            except ValueError:
                pass
        # Fallback: HTML hidden input.
        match = re.search(
            r'id=["\']created-personal-access-token["\'][^>]*value=["\']([^"\']+)["\']',
            resp.text,
            re.IGNORECASE,
        )
        return match.group(1).strip() if match else None


class EndpointTokenGenerator(TokenGenerator):
    """Acquire a token by POSTing credentials to a REST endpoint.

    This is the Magento-style flow: POST ``{"username": "...", "password": "..."}``
    to ``token_endpoint``, receive the token string as the JSON response body.
    """

    generator_name = "endpoint"

    def __init__(self, *, token_endpoint: str = "") -> None:
        self.token_endpoint = token_endpoint

    def generate(self, credentials: dict[str, str], site_url: str) -> str:
        if not self.token_endpoint:
            raise RuntimeError("endpoint generator requires token_endpoint")
        url = f"{site_url.rstrip('/')}{self.token_endpoint}"
        resp = requests.post(url, json=credentials, timeout=30)
        resp.raise_for_status()
        token_value = resp.json()
        token_text = (
            token_value.strip().strip('"') if isinstance(token_value, str) else str(token_value)
        )
        if not token_text:
            raise RuntimeError(
                f"endpoint token generator got empty response from {self.token_endpoint}"
            )
        logger.info("endpoint: acquired fresh token from %s", self.token_endpoint)
        return token_text

    def validate(self, token: str, site_url: str) -> bool:
        # Generic endpoint tokens have no universal validation path.
        # The caller should use a site-specific validation endpoint.
        return bool(token and token.strip())


# ── Dispatcher ───────────────────────────────────────────────────────────


def _token_strategy(auth_config: dict[str, Any]) -> str | None:
    generator_name = auth_config.get("token_generator")
    if isinstance(generator_name, str) and generator_name.strip():
        return "token_generator"
    token_endpoint = auth_config.get("token_endpoint")
    if isinstance(token_endpoint, str) and token_endpoint.strip():
        return "token_endpoint"
    token = auth_config.get("token")
    if isinstance(token, str) and token.strip():
        return "token"
    token_source = auth_config.get("token_source")
    if isinstance(token_source, str) and token_source.strip():
        return "token_source"
    return None


def acquire_token(auth_config: dict[str, Any], site_url: str) -> str:
    """Acquire a fresh token based on the auth config.

    Resolution order:
    1. ``token_generator`` -- named generator (e.g. ``gitlab_pat``).
    2. ``token_endpoint``  -- POST credentials to a REST endpoint.
    3. ``token``           -- inline static token (returned as-is).
    4. ``token_source``    -- legacy file-backed token read from disk.

    Raises RuntimeError if no strategy can produce a token.
    """
    credentials = auth_config.get("credentials", {})
    if not isinstance(credentials, dict):
        credentials = {}

    # 1. Named generator
    strategy = _token_strategy(auth_config)

    # 1. Named generator
    if strategy == "token_generator":
        generator_name = str(auth_config.get("token_generator")).strip()
        generator_cls = _TOKEN_GENERATORS.get(generator_name.strip())
        if generator_cls is None:
            raise RuntimeError(
                f"unknown token_generator {generator_name!r}, "
                f"available: {sorted(_TOKEN_GENERATORS)}"
            )
        generator = generator_cls()
        return generator.generate(credentials, site_url)

    # 2. Token endpoint
    if strategy == "token_endpoint":
        token_endpoint = str(auth_config.get("token_endpoint")).strip()
        gen = EndpointTokenGenerator(token_endpoint=token_endpoint.strip())
        return gen.generate(credentials, site_url)

    # 3. Inline token
    if strategy == "token":
        token = str(auth_config.get("token")).strip()
        return token.strip()

    # 4. Legacy token_source
    if strategy == "token_source":
        token_source = str(auth_config.get("token_source")).strip()
        from warp_taskgen.seeding.db import _resolve_token_source_path

        token_path = _resolve_token_source_path(token_source.strip())
        try:
            token_text = token_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise RuntimeError(f"could not read token_source {token_path}: {exc}") from exc
        if not token_text:
            raise RuntimeError(f"token_source {token_path} was empty")
        return token_text

    raise RuntimeError(
        "bearer_token auth config has no token_generator, token_endpoint, token, or token_source"
    )


def validate_token(
    token: str,
    auth_config: dict[str, Any],
    site_url: str,
    *,
    validation_endpoint: str | None = None,
) -> bool:
    """Validate a token against the live instance.

    If a ``token_generator`` is configured, delegates to the generator's
    ``validate`` method. Otherwise uses ``validation_endpoint``. Missing
    live validation is treated as a hard failure.
    """
    generator_name = auth_config.get("token_generator")
    if isinstance(generator_name, str) and generator_name.strip():
        generator_cls = _TOKEN_GENERATORS.get(generator_name.strip())
        if generator_cls is not None:
            return generator_cls().validate(token, site_url)

    if validation_endpoint:
        header_name = str(auth_config.get("header_name") or "Authorization")
        header_value = token
        if header_name.lower() == "authorization" and not token.lower().startswith("bearer "):
            header_value = f"Bearer {token}"
        try:
            resp = requests.get(
                f"{site_url.rstrip('/')}{validation_endpoint}",
                headers={header_name: header_value},
                timeout=15,
            )
            return resp.status_code == 200
        except requests.RequestException:
            return False

    return False


def _validation_endpoint_for(auth_config: dict[str, Any]) -> str | None:
    value = auth_config.get("validation_endpoint")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def bearer_token_config_error(auth_config: dict[str, Any]) -> str | None:
    """Return a human-readable config error for bearer token auth, if any."""
    strategy = _token_strategy(auth_config)
    validation_endpoint = _validation_endpoint_for(auth_config)

    if strategy == "token_generator":
        credentials = auth_config.get("credentials")
        if not isinstance(credentials, dict) or not credentials:
            return "token_generator auth requires a non-empty credentials dict"
        return None
    if strategy == "token_endpoint":
        credentials = auth_config.get("credentials")
        if not isinstance(credentials, dict) or not credentials:
            return "token_endpoint auth requires a non-empty credentials dict"
        if validation_endpoint is None:
            return "validation_endpoint is required for token_endpoint auth"
        return None
    if strategy == "token":
        if validation_endpoint is None:
            return "validation_endpoint is required for inline token auth"
        return None
    if strategy == "token_source":
        if validation_endpoint is None:
            return "validation_endpoint is required for token_source auth"
        return None
    return "bearer_token auth config has no token_generator, token_endpoint, token, or token_source"


# ── Pipeline-level helpers ───────────────────────────────────────────────

# Per-run cache: maps (site_url, auth_config_fingerprint) -> (token, last_validated_monotonic).
_RUN_TOKEN_CACHE: dict[str, tuple[str, float]] = {}
_RUN_TOKEN_CACHE_LOCK = threading.Lock()
_TOKEN_ACQUIRE_MAX_WORKERS = 8


def _cache_key(site_url: str, auth_config: dict[str, Any]) -> str:
    """Deterministic cache key from site URL and auth config identity."""
    identity = _cache_identity(auth_config)
    return f"{_canonical_site_url(site_url)}|{json.dumps(identity, sort_keys=True, separators=(',', ':'))}"


def _canonical_site_url(site_url: str) -> str:
    return site_url.strip().rstrip("/")


def _hashed_identity(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_identity(auth_config: dict[str, Any]) -> dict[str, Any]:
    strategy = _token_strategy(auth_config)
    identity: dict[str, Any] = {
        "strategy": strategy,
        "type": auth_config.get("type"),
        "header_name": auth_config.get("header_name"),
        "validation_endpoint": _validation_endpoint_for(auth_config),
    }
    if strategy == "token_generator":
        identity["token_generator"] = auth_config.get("token_generator")
        credentials = auth_config.get("credentials")
        if isinstance(credentials, dict):
            identity["credentials_sha256"] = _hashed_identity(credentials)
    elif strategy == "token_endpoint":
        identity["token_endpoint"] = auth_config.get("token_endpoint")
        credentials = auth_config.get("credentials")
        if isinstance(credentials, dict):
            identity["credentials_sha256"] = _hashed_identity(credentials)
    elif strategy == "token":
        identity["token_sha256"] = _hashed_identity(str(auth_config.get("token") or ""))
    elif strategy == "token_source":
        path = Path(str(auth_config.get("token_source") or "")).expanduser().resolve(strict=False)
        identity["token_source"] = str(path)
        try:
            identity["token_source_mtime_ns"] = path.stat().st_mtime_ns
        except OSError:
            identity["token_source_mtime_ns"] = None
    return identity


def acquire_tokens_for_instances(
    instances: list[Any],
    *,
    auth_fields: tuple[str, ...] = ("auth", "api_auth"),
    force_refresh: bool = False,
) -> list[str]:
    """Acquire and cache tokens for all instances that need them.

    For each instance and each auth field, resolve the bearer token source and
    validate it against the live instance before injecting it for downstream use.

    Returns a list of human-readable error strings (empty on full success).
    """
    errors: list[str] = []
    requirements: list[tuple[Any, str, str, str, dict[str, Any], str]] = []
    unique_requirements: dict[str, tuple[Any, str, str, str, dict[str, Any], str]] = {}

    for instance in instances:
        inst = instance if isinstance(instance, dict) else instance.model_dump()
        site_url = str(inst.get("site_url", ""))
        site_name = str(inst.get("site_name", ""))

        for field in auth_fields:
            auth = inst.get(field)
            if not isinstance(auth, dict):
                continue
            if str(auth.get("type", "")).strip() != "bearer_token":
                continue

            config_error = bearer_token_config_error(auth)
            if config_error is not None:
                errors.append(f"[{site_name}] bearer_token_unvalidated for {field}: {config_error}")
                continue

            requirement = (instance, field, site_url, site_name, auth, _cache_key(site_url, auth))
            requirements.append(requirement)
            unique_requirements.setdefault(requirement[-1], requirement)

    if not unique_requirements:
        return errors

    resolved_tokens: dict[str, str] = {}
    resolution_errors: dict[str, str] = {}
    max_workers = min(max(1, len(unique_requirements)), _TOKEN_ACQUIRE_MAX_WORKERS)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            cache_key: executor.submit(
                resolve_bearer_token,
                auth,
                site_url=site_url,
                force_refresh=force_refresh,
            )
            for cache_key, (_, _, site_url, _, auth, _) in unique_requirements.items()
        }
        for cache_key, future in future_map.items():
            try:
                resolved_tokens[cache_key] = future.result()
            except Exception as exc:
                resolution_errors[cache_key] = str(exc)

    for instance, field, site_url, site_name, _auth, cache_key in requirements:
        token = resolved_tokens.get(cache_key)
        if token is None:
            message = resolution_errors[cache_key]
            if "failed live validation" in message:
                errors.append(f"[{site_name}] bearer_token_unvalidated for {field}: {message}")
            else:
                errors.append(f"[{site_name}] failed to acquire {field} token: {message}")
            continue
        _inject_token(instance, field, token)
        logger.info("Acquired fresh %s token for %s (%s)", field, site_name, site_url)

    return errors


def resolve_bearer_token(
    auth_config: dict[str, Any], *, site_url: str, force_refresh: bool = False
) -> str:
    """Resolve a bearer token and validate it against the live instance.

    This is the single point of truth for bearer auth. Callers should not read
    ``token_source`` directly or keep a side cache of generated tokens.
    """
    config_error = bearer_token_config_error(auth_config)
    if config_error is not None:
        raise RuntimeError(config_error)

    validation_endpoint = _validation_endpoint_for(auth_config)
    cache_key = _cache_key(site_url, auth_config)

    cached = None if force_refresh else _get_cached_token(cache_key)
    if cached is not None:
        cached_token, validated_at = cached
        if time.monotonic() - validated_at <= _TOKEN_VALIDATION_TTL_SECONDS:
            return cached_token
        if validate_token(
            cached_token,
            auth_config,
            site_url,
            validation_endpoint=validation_endpoint,
        ):
            _set_cached_token(cache_key, cached_token)
            return cached_token
        _drop_cached_token(cache_key, expected=cached_token)

    token = acquire_token(auth_config, site_url)
    if not validate_token(
        token,
        auth_config,
        site_url,
        validation_endpoint=validation_endpoint,
    ):
        raise RuntimeError("acquired token failed live validation")
    _set_cached_token(cache_key, token)
    return token


def clear_run_token_cache() -> None:
    """Clear the per-run token cache (useful between test runs)."""
    with _RUN_TOKEN_CACHE_LOCK:
        _RUN_TOKEN_CACHE.clear()


def _get_cached_token(cache_key: str) -> tuple[str, float] | None:
    with _RUN_TOKEN_CACHE_LOCK:
        return _RUN_TOKEN_CACHE.get(cache_key)


def _set_cached_token(cache_key: str, token: str) -> None:
    with _RUN_TOKEN_CACHE_LOCK:
        _RUN_TOKEN_CACHE[cache_key] = (token, time.monotonic())


def _drop_cached_token(cache_key: str, *, expected: str | None = None) -> None:
    with _RUN_TOKEN_CACHE_LOCK:
        current = _RUN_TOKEN_CACHE.get(cache_key)
        current_token = current[0] if isinstance(current, tuple) else current
        if expected is None or current_token == expected:
            _RUN_TOKEN_CACHE.pop(cache_key, None)


def _inject_token(instance: Any, field: str, token: str) -> None:
    """Set ``instance[field]["token"]`` so downstream resolvers use it."""
    if isinstance(instance, dict):
        auth = instance.get(field)
        if isinstance(auth, dict):
            auth["token"] = token
    elif hasattr(instance, field):
        auth = getattr(instance, field)
        if isinstance(auth, dict):
            auth["token"] = token


# ── HTTP request header building ─────────────────────────────────────────
#
# These helpers were previously private to warp_taskgen.seeding (where the
# legacy api/form seed dispatch lived). After the editor migration the
# editors are the sole callers, so they live here adjacent to bearer-token
# resolution. See docs/handoffs/archive/current_progress_pre_wasp_20260417.md for
# the editor-only seed mechanism rationale.

_BLOCKED_CALL_HEADER_NAMES = frozenset(
    {
        "authorization",
        "cookie",
        "origin",
        "referer",
        "x-csrf-token",
        "x-csrftoken",
        "x-xsrf-token",
        "x-xsrftoken",
        "host",
        "forwarded",
        "proxy",
        "proxy-authorization",
        "proxy-authenticate",
        "proxy-connection",
        "transfer-encoding",
        "x-forwarded-for",
        "x-forwarded-host",
        "x-forwarded-proto",
    }
)


def pick_auth_lane(instance: Mapping[str, Any], mechanism: str) -> dict[str, Any] | None:
    """Return the auth config for the given seeding mechanism.

    API-mechanism callers prefer ``instance['api_auth']`` (e.g. admin bearer
    token); form / web-login callers always use ``instance['auth']``.
    """
    if mechanism == "api":
        api_auth = instance.get("api_auth")
        if isinstance(api_auth, dict):
            return api_auth
    auth = instance.get("auth")
    return auth if isinstance(auth, dict) else None


def _resolve_header_value(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        env_name = value.get("from_env")
        if isinstance(env_name, str) and env_name:
            resolved = os.environ.get(env_name)
            if not resolved:
                raise RuntimeError(f"required auth header env var {env_name!r} is not set")
            return resolved
    raise RuntimeError('auth header values must be strings or {"from_env": "VAR_NAME"}')


def _sanitize_call_headers(
    call_headers: Mapping[str, Any],
    *,
    protected_headers: set[str],
) -> dict[str, str]:
    sanitized: dict[str, str] = {}
    for key, value in call_headers.items():
        key_str = str(key)
        lowered = key_str.lower()
        if lowered in _BLOCKED_CALL_HEADER_NAMES or lowered in protected_headers:
            continue
        sanitized[key_str] = str(value)
    return sanitized


def build_auth_headers(
    instance: Mapping[str, Any],
    call: Mapping[str, Any] | None = None,
    *,
    mechanism: str = "form",
) -> dict[str, str]:
    """Build the HTTP request headers for an authenticated benchmark call.

    Picks the auth lane for ``mechanism`` (``api_auth`` for ``"api"``,
    ``auth`` otherwise), materializes either ``http_headers`` declarations
    or a bearer token, and merges in any safe caller-supplied headers from
    ``call.get('headers')``. Sensitive header names (auth, cookie, host,
    forwarded, proxy, csrf) are stripped from the call so a hostile call
    cannot override the auth lane.
    """
    headers: dict[str, str] = {}
    auth = pick_auth_lane(instance, mechanism)
    site_url = str(instance.get("site_url", ""))
    auth_header_names: set[str] = set()
    if isinstance(auth, dict):
        auth_type = str(auth.get("type", "")).strip()
        if auth_type == "http_headers":
            declared_headers = auth.get("headers")
            if isinstance(declared_headers, dict):
                for key, value in declared_headers.items():
                    resolved = _resolve_header_value(value)
                    headers[str(key)] = resolved
                    auth_header_names.add(str(key).lower())
        elif auth_type == "bearer_token":
            token = resolve_bearer_token(auth, site_url=site_url)
            header_name = str(auth.get("header_name") or "Authorization")
            if header_name.lower() == "authorization" and not token.lower().startswith("bearer "):
                token = f"Bearer {token}"
            headers[header_name] = token
            auth_header_names.add(header_name.lower())

    call_headers = (call or {}).get("headers") if call is not None else None
    if isinstance(call_headers, dict):
        sanitized = _sanitize_call_headers(call_headers, protected_headers=auth_header_names)
        merged = dict(sanitized)
        merged.update(headers)
        headers = merged
    return headers
