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

import logging
import re
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

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

    def generate(self, credentials: dict[str, str], site_url: str) -> str:
        username = credentials.get("username", "")
        password = credentials.get("password", "")
        if not username or not password:
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

        # Step 4: POST to create the PAT.
        create_resp = session.post(
            pat_url,
            data={
                "authenticity_token": form_csrf,
                "personal_access_token[name]": "worldsim-runtime",
                "personal_access_token[expires_at]": "",
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
    generator_name = auth_config.get("token_generator")
    if isinstance(generator_name, str) and generator_name.strip():
        generator_cls = _TOKEN_GENERATORS.get(generator_name.strip())
        if generator_cls is None:
            raise RuntimeError(
                f"unknown token_generator {generator_name!r}, "
                f"available: {sorted(_TOKEN_GENERATORS)}"
            )
        generator = generator_cls()
        return generator.generate(credentials, site_url)

    # 2. Token endpoint
    token_endpoint = auth_config.get("token_endpoint")
    if isinstance(token_endpoint, str) and token_endpoint.strip():
        gen = EndpointTokenGenerator(token_endpoint=token_endpoint.strip())
        return gen.generate(credentials, site_url)

    # 3. Inline token
    token = auth_config.get("token")
    if isinstance(token, str) and token.strip():
        return token.strip()

    # 4. Legacy token_source
    token_source = auth_config.get("token_source")
    if isinstance(token_source, str) and token_source.strip():
        token_path = Path(token_source.strip())
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


# ── Pipeline-level helpers ───────────────────────────────────────────────

# Per-run cache: maps (site_url, auth_config_fingerprint) -> token.
_RUN_TOKEN_CACHE: dict[str, str] = {}


def _cache_key(site_url: str, auth_config: dict[str, Any]) -> str:
    """Deterministic cache key from site URL and auth config identity."""
    import json

    # Include enough to distinguish different auth configs on the same site.
    identity = json.dumps(
        {k: auth_config.get(k) for k in sorted(auth_config) if k not in ("token", "token_source")},
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{site_url}|{identity}"


def acquire_tokens_for_instances(
    instances: list[Any],
    *,
    auth_fields: tuple[str, ...] = ("auth", "api_auth"),
) -> list[str]:
    """Acquire and cache tokens for all instances that need them.

    For each instance and each auth field, resolve the bearer token source and
    validate it against the live instance before injecting it for downstream use.

    Returns a list of human-readable error strings (empty on full success).
    """
    errors: list[str] = []

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

            has_generator = (
                isinstance(auth.get("token_generator"), str) and auth["token_generator"].strip()
            )
            has_endpoint = (
                isinstance(auth.get("token_endpoint"), str) and auth["token_endpoint"].strip()
            )
            has_inline_token = isinstance(auth.get("token"), str) and auth["token"].strip()
            has_token_source = (
                isinstance(auth.get("token_source"), str) and auth["token_source"].strip()
            )
            validation_endpoint = _validation_endpoint_for(auth)

            if (has_inline_token or has_token_source or has_endpoint) and validation_endpoint is None:
                errors.append(
                    f"[{site_name}] bearer_token_unvalidated for {field}: "
                    "validation_endpoint is required for inline, token_source, and token_endpoint auth"
                )
                continue

            if not (has_generator or has_endpoint or has_inline_token or has_token_source):
                errors.append(
                    f"[{site_name}] failed to acquire {field} token: bearer_token config has no usable token source"
                )
                continue

            ck = _cache_key(site_url, auth)
            cached = _RUN_TOKEN_CACHE.get(ck)
            if cached:
                if not validate_token(
                    cached,
                    auth,
                    site_url,
                    validation_endpoint=validation_endpoint,
                ):
                    errors.append(
                        f"[{site_name}] bearer_token_unvalidated for {field}: cached token failed live validation"
                    )
                    _RUN_TOKEN_CACHE.pop(ck, None)
                    continue
                _inject_token(instance, field, cached)
                continue

            try:
                token = acquire_token(auth, site_url)
            except Exception as exc:
                errors.append(f"[{site_name}] failed to acquire {field} token: {exc}")
                continue

            if not validate_token(
                token,
                auth,
                site_url,
                validation_endpoint=validation_endpoint,
            ):
                errors.append(
                    f"[{site_name}] bearer_token_unvalidated for {field}: acquired token failed live validation"
                )
                continue

            _RUN_TOKEN_CACHE[ck] = token
            _inject_token(instance, field, token)
            logger.info("Acquired fresh %s token for %s (%s)", field, site_name, site_url)

    return errors


def clear_run_token_cache() -> None:
    """Clear the per-run token cache (useful between test runs)."""
    _RUN_TOKEN_CACHE.clear()


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
