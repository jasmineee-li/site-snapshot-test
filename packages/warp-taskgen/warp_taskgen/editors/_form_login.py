"""Web-login + CSRF helpers for editors that submit HTML forms.

These helpers used to live as private symbols in :mod:`warp_taskgen.seeding`
because the legacy api/form seed dispatch was their other caller. After
the api/form sunset (see
``docs/handoffs/archive/current_progress_pre_wasp_20260417.md``) the editors are
the sole callers, so the helpers moved here.

Layering: this module depends on :mod:`warp_taskgen.auth_tokens` for the
auth-lane picker only. It must NOT import from :mod:`warp_taskgen.seeding`,
otherwise we restore the layering inversion that the editor migration
removed.
"""

from __future__ import annotations

import logging
import re
import urllib.parse
import weakref
from collections.abc import Mapping
from typing import Any

import requests

from warp_taskgen.auth_tokens import pick_auth_lane

logger = logging.getLogger(__name__)

_FORM_METHODS = frozenset({"POST", "PUT", "PATCH"})

_CSRF_TOKEN_CACHE: weakref.WeakKeyDictionary[
    requests.Session,
    dict[tuple[str, str], tuple[str | None, str | None]],
] = weakref.WeakKeyDictionary()

_CSRF_INPUT_PATTERNS = (
    re.compile(
        r'name=["\'](form_key|authenticity_token|csrf_token|token)["\'][^>]*value=["\']([^"\']+)'
    ),
    re.compile(
        r'<meta[^>]+name=["\']csrf-token["\'][^>]+content=["\']([^"\']+)["\']',
        re.IGNORECASE,
    ),
)
_CSRF_PARAM_META = re.compile(
    r'<meta[^>]+name=["\']csrf-param["\'][^>]+content=["\']([^"\']+)["\']',
    re.IGNORECASE,
)


def extract_csrf_token(html: str) -> tuple[str | None, str | None]:
    """Best-effort extract a CSRF (name, value) pair from server-rendered HTML."""
    for pattern in _CSRF_INPUT_PATTERNS[:1]:
        match = pattern.search(html)
        if match:
            return match.group(1), match.group(2)
    meta_match = _CSRF_INPUT_PATTERNS[1].search(html)
    if meta_match:
        # Rails sets <meta name="csrf-param" content="authenticity_token">.
        param_match = _CSRF_PARAM_META.search(html)
        param_name = param_match.group(1) if param_match else "csrf_token"
        return param_name, meta_match.group(1)
    return (None, None)


def looks_like_login_page(html: str) -> bool:
    lowered = (html or "").lower()
    if not lowered:
        return False
    indicators = (
        "user[password]",
        'type="password"',
        'name="_password"',
        "input#login-password",
        "sign in",
        "log in",
    )
    return any(indicator in lowered for indicator in indicators)


def _redirect_path(response: requests.Response) -> str:
    location = response.headers.get("Location")
    if not isinstance(location, str) or not location.strip():
        return ""
    return (urllib.parse.urlparse(location).path or "").strip().lower()


def _redirects_to_login(response: requests.Response) -> bool:
    path = _redirect_path(response)
    if not path:
        return False
    return any(token in path for token in ("/login", "/sign_in", "/users/sign_in", "/session"))


def _origin_for_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def _normalize_delivery_path(path: str) -> str:
    return re.sub(r"/\{[^}/]+\}(?=/|$)", "/{id}", re.sub(r"/\d+(?=/|$)", "/{id}", path))


def _csrf_cache_key(
    instance: Mapping[str, Any],
    url: str,
) -> tuple[str, str]:
    parsed = urllib.parse.urlparse(url)
    normalized_path = _normalize_delivery_path(parsed.path or "/")
    query_suffix = f"?{parsed.query}" if parsed.query else ""
    return (
        str(instance.get("site_name", "")),
        f"{parsed.scheme}://{parsed.netloc}{normalized_path}{query_suffix}",
    )


def get_csrf_token(
    session: requests.Session,
    url: str,
    headers: dict[str, str],
    instance: Mapping[str, Any],
    *,
    force_refresh: bool = False,
) -> tuple[str | None, str | None]:
    origin = _origin_for_url(url)
    cache_key = _csrf_cache_key(instance, url)
    session_cache = _CSRF_TOKEN_CACHE.setdefault(session, {})
    if not force_refresh and cache_key in session_cache:
        return session_cache[cache_key]

    for candidate_url in (url, origin):
        try:
            response = session.get(
                candidate_url,
                headers=headers,
                timeout=30,
                allow_redirects=False,
            )
            if 300 <= response.status_code < 400:
                continue
            response.raise_for_status()
        except requests.RequestException:
            continue
        token = extract_csrf_token(response.text)
        if token != (None, None):
            session_cache[cache_key] = token
            return token

    return (None, None)


def clear_cached_csrf_token(
    session: requests.Session,
    instance: Mapping[str, Any],
    url: str,
) -> None:
    cache_key = _csrf_cache_key(instance, url)
    session_cache = _CSRF_TOKEN_CACHE.get(session)
    if session_cache is not None:
        session_cache.pop(cache_key, None)


def prepare_form_body(
    method: str,
    url: str,
    headers: dict[str, str],
    body_form: object,
    instance: Mapping[str, Any],
    session: requests.Session,
    *,
    force_refresh: bool = False,
) -> dict[str, Any] | None:
    if not isinstance(body_form, dict):
        return None
    form_body = dict(body_form)
    if method.strip().upper() not in _FORM_METHODS:
        return form_body

    token_name, token_value = get_csrf_token(
        session,
        url,
        headers,
        instance,
        force_refresh=force_refresh,
    )
    if token_name and token_value:
        form_body[token_name] = token_value
    return form_body


def perform_web_login_if_needed(
    session: requests.Session,
    instance: Mapping[str, Any],
    mechanism: str,
) -> None:
    """Log in via web form if the effective auth type is ``web_login``.

    Two-step flow: GET the login page to extract a CSRF token, then POST
    credentials with the token. Resulting session cookies are stored on
    ``session`` for subsequent editor requests.
    """
    auth = pick_auth_lane(instance, mechanism)
    if not isinstance(auth, dict) or auth.get("type") != "web_login":
        return
    site_url = str(instance.get("site_url", "")).rstrip("/")
    login_path = str(auth.get("login_url", "/login"))
    login_url = f"{site_url}{login_path}"
    credentials = auth.get("credentials", {})
    if not isinstance(credentials, dict) or not credentials:
        raise RuntimeError(
            f"web_login auth for {instance.get('site_name', '?')} requires credentials"
        )

    resp = session.get(login_url, timeout=30, allow_redirects=True)
    resp.raise_for_status()
    token_name, token_value = extract_csrf_token(resp.text)

    login_data: dict[str, str] = {}
    login_data.update(credentials)
    if token_name and token_value:
        login_data[token_name] = token_value

    post_resp = session.post(login_url, data=login_data, timeout=30, allow_redirects=False)
    if post_resp.status_code not in (200, 302):
        raise RuntimeError(
            f"Web login failed for {instance.get('site_name', '?')}: HTTP {post_resp.status_code}"
        )
    if _redirects_to_login(post_resp):
        raise RuntimeError(
            f"Web login failed for {instance.get('site_name', '?')}: redirected back to login"
        )
    if post_resp.status_code == 200 and looks_like_login_page(post_resp.text):
        raise RuntimeError(
            f"Web login failed for {instance.get('site_name', '?')}: login form was re-rendered"
        )

    validation_endpoint = auth.get("validation_endpoint")
    if isinstance(validation_endpoint, str) and validation_endpoint.strip():
        validation_url = f"{site_url}{validation_endpoint.strip()}"
        validation_resp = session.get(validation_url, timeout=30, allow_redirects=False)
        if validation_resp.status_code in {401, 403}:
            raise RuntimeError(
                f"Web login failed for {instance.get('site_name', '?')}: "
                f"validation endpoint returned HTTP {validation_resp.status_code}"
            )
        if _redirects_to_login(validation_resp):
            raise RuntimeError(
                f"Web login failed for {instance.get('site_name', '?')}: "
                "validation endpoint redirected to login"
            )
        if validation_resp.status_code == 200 and looks_like_login_page(validation_resp.text):
            raise RuntimeError(
                f"Web login failed for {instance.get('site_name', '?')}: "
                "validation endpoint still served the login page"
            )
