#!/usr/bin/env python3
"""Bootstrap a GitLab personal access token from Phase 0d storage-state cookies.

Run Phase 0d first so ``logs/phase_0d/gitlab/storage_state.json`` exists.
This helper reuses those browser cookies to create a PAT and writes it to
``logs/phase_0d/gitlab/personal_access_token.txt`` by default.

This path is legacy/break-glass only. The normal runtime flow should prefer
``token_generator: "gitlab_pat"`` so bearer auth is minted and validated in
memory per run instead of being persisted on disk.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import requests

from worldsim.auth_tokens import GitLabPATGenerator

_AUTH_TOKEN_RE = re.compile(
    r'name=["\']authenticity_token["\'][^>]*value=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_CSRF_META_RE = re.compile(
    r'<meta\s+name=["\']csrf-token["\']\s+content=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_PAT_RE = re.compile(
    r'id=["\']created-personal-access-token["\'][^>]*value=["\']([^"\']+)["\']',
    re.IGNORECASE,
)


def _default_state_path() -> Path:
    return Path("logs/phase_0d/gitlab/storage_state.json")


def _default_output_path() -> Path:
    return Path("logs/phase_0d/gitlab/personal_access_token.txt")


def _load_cookies(storage_state_path: Path) -> list[dict]:
    payload = json.loads(storage_state_path.read_text(encoding="utf-8"))
    cookies = payload.get("cookies")
    if not isinstance(cookies, list) or not cookies:
        raise RuntimeError(f"{storage_state_path} does not contain browser cookies")
    return cookies


def _build_session(cookies: list[dict]) -> requests.Session:
    session = requests.Session()
    for cookie in cookies:
        if not isinstance(cookie, dict):
            continue
        name = cookie.get("name")
        value = cookie.get("value")
        domain = cookie.get("domain")
        if isinstance(name, str) and isinstance(value, str):
            session.cookies.set(name, value, domain=domain)
    return session


def bootstrap_pat(
    *,
    site_url: str,
    storage_state_path: Path,
    output_path: Path,
    token_name: str = "worldsim-runtime",
) -> Path:
    validator = GitLabPATGenerator()
    if output_path.exists():
        existing = output_path.read_text(encoding="utf-8").strip()
        if existing and validator.validate(existing, site_url):
            return output_path

    session = _build_session(_load_cookies(storage_state_path))
    form_url = f"{site_url.rstrip('/')}/-/profile/personal_access_tokens"
    form_response = session.get(form_url, timeout=30)
    form_response.raise_for_status()
    match = _AUTH_TOKEN_RE.search(form_response.text) or _CSRF_META_RE.search(form_response.text)
    if match is None:
        raise RuntimeError("Could not find GitLab authenticity_token on the PAT form page")
    csrf_token = match.group(1)

    response = session.post(
        form_url,
        data={
            "authenticity_token": csrf_token,
            "personal_access_token[name]": token_name,
            "personal_access_token[expires_at]": validator._expires_at(),
            "personal_access_token[scopes][]": "api",
        },
        headers={"X-CSRF-Token": csrf_token},
        timeout=30,
    )
    response.raise_for_status()

    new_token: str | None = None
    content_type = response.headers.get("Content-Type", "")
    if "application/json" in content_type:
        try:
            new_token = response.json().get("new_token")
        except ValueError:
            new_token = None
    if not new_token:
        token_match = _PAT_RE.search(response.text)
        if token_match is not None:
            new_token = token_match.group(1).strip()
    if not new_token:
        raise RuntimeError("GitLab PAT creation response did not expose a created token value")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(new_token.strip() + "\n", encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap a GitLab PAT from storage_state")
    parser.add_argument("--site-url", required=True)
    parser.add_argument("--storage-state", type=Path, default=_default_state_path())
    parser.add_argument("--output", type=Path, default=_default_output_path())
    parser.add_argument("--token-name", default="worldsim-api")
    args = parser.parse_args()

    bootstrap_pat(
        site_url=args.site_url,
        storage_state_path=args.storage_state,
        output_path=args.output,
        token_name=args.token_name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
