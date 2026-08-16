#!/usr/bin/env python3
"""Create regular-participant browser state for the Classifieds canary.

The command logs in through the pinned VisualWebArena Classifieds form and
writes a Playwright storage-state JSON file. Credentials come from the two
named environment variables, never from command-line arguments. The output
must stay outside the source checkout.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from warp_taskgen.classifieds_canary import (
    CanaryConfigError,
    validate_classifieds_loopback_origin,
)

CLASSIFIEDS_USERNAME_ENV = "WARP_TASKGEN_CLASSIFIEDS_USERNAME"
CLASSIFIEDS_PASSWORD_ENV = "WARP_TASKGEN_CLASSIFIEDS_PASSWORD"
LOGIN_PATH = "/index.php?page=login"
DASHBOARD_PATH = "/index.php?page=user&action=dashboard"
LOGOUT_MARKERS = ("page=logout", "action=logout", "/logout")
PERSISTENT_AUTH_COOKIES = frozenset({"oc_userId", "oc_userSecret"})


class ClassifiedsWriterBootstrapError(RuntimeError):
    """Raised when the Classifieds participant state cannot be minted."""


def _required_environment(env: Mapping[str, str]) -> tuple[str, str]:
    username = env.get(CLASSIFIEDS_USERNAME_ENV, "")
    password = env.get(CLASSIFIEDS_PASSWORD_ENV, "")
    if not username:
        raise ClassifiedsWriterBootstrapError(
            f"missing required environment variable {CLASSIFIEDS_USERNAME_ENV}"
        )
    if not password:
        raise ClassifiedsWriterBootstrapError(
            f"missing required environment variable {CLASSIFIEDS_PASSWORD_ENV}"
        )
    return username, password


def _absolute_http_url(site_url: str, path: str) -> str:
    try:
        base = validate_classifieds_loopback_origin(site_url)
    except CanaryConfigError as exc:
        raise ClassifiedsWriterBootstrapError(
            "site URL must be the configured loopback Classifieds origin"
        ) from exc
    return f"{base}{path}"


def _response_status(response: Any) -> int | None:
    status = getattr(response, "status", None)
    if status is None:
        return None
    try:
        return int(status)
    except (TypeError, ValueError):
        return None


def _assert_authenticated_dashboard(page: Any, dashboard_url: str) -> None:
    response = page.goto(dashboard_url, wait_until="domcontentloaded")
    status = _response_status(response)
    if status is not None and status >= 400:
        raise ClassifiedsWriterBootstrapError("authenticated dashboard returned an error")

    current_url = str(getattr(page, "url", ""))
    current = urlsplit(current_url)
    query = parse_qs(current.query, keep_blank_values=True)
    if query.get("page") != ["user"] or query.get("action") not in (
        ["dashboard"],
        ["items"],
    ):
        raise ClassifiedsWriterBootstrapError("login did not reach the authenticated dashboard")

    body = page.content()
    if not isinstance(body, str) or not any(marker in body.casefold() for marker in LOGOUT_MARKERS):
        raise ClassifiedsWriterBootstrapError("authenticated dashboard lacks a logout marker")


def _write_storage_state_atomic(output_path: Path, storage_state: Mapping[str, Any]) -> None:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(storage_state, sort_keys=True, separators=(",", ":")) + "\n"
    temporary_path: Path | None = None
    fd: int | None = None
    try:
        fd, temporary_name = tempfile.mkstemp(
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        os.fchmod(fd, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = None
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
        os.chmod(output_path, stat.S_IRUSR | stat.S_IWUSR)
    finally:
        if fd is not None:
            os.close(fd)
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def _require_external_output(output_path: Path, *, repo_root: Path | None = None) -> None:
    """Keep live session cookies outside the source checkout."""

    if repo_root is None:
        script = Path(__file__).resolve()
        root = next(
            (parent for parent in script.parents if (parent / ".git").exists()),
            script.parents[1],
        )
    else:
        root = repo_root
    root = root.expanduser().resolve()
    output = output_path.expanduser().resolve()
    try:
        output.relative_to(root)
    except ValueError:
        return
    raise ClassifiedsWriterBootstrapError(
        "writer storage state must be outside the source checkout"
    )


def _playwright_factory() -> Any:
    from playwright.sync_api import sync_playwright

    return sync_playwright()


def mint_classifieds_writer_storage_state(
    site_url: str,
    output_path: str | Path,
    *,
    env: Mapping[str, str] | None = None,
    playwright_factory: Callable[[], Any] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Log in and atomically write a regular-participant storage state.

    ``env`` and ``playwright_factory`` are injection seams for unit tests.  The
    function never accepts credentials as arguments; production callers use
    the two named environment variables above.
    """

    credential_env = os.environ if env is None else env
    username, password = _required_environment(credential_env)
    output = Path(output_path).expanduser().resolve()
    _require_external_output(output, repo_root=repo_root)
    login_url = _absolute_http_url(site_url, LOGIN_PATH)
    dashboard_url = _absolute_http_url(site_url, DASHBOARD_PATH)
    factory = _playwright_factory if playwright_factory is None else playwright_factory

    with factory() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            context = browser.new_context()
            try:
                page = context.new_page()
                page.goto(login_url, wait_until="domcontentloaded")
                page.locator("#email").fill(username)
                page.locator("#password").fill(password)
                # The harness recreates the web container during golden-state
                # resets. The pinned app only issues durable auth cookies when
                # its explicit remember control is selected; a PHP session
                # cookie would become invalid after that recreation.
                page.locator("#remember").check()
                page.get_by_role("button", name="Log in", exact=True).click()
                wait_for_load_state = getattr(page, "wait_for_load_state", None)
                if callable(wait_for_load_state):
                    wait_for_load_state("domcontentloaded")
                _assert_authenticated_dashboard(page, dashboard_url)
                state = context.storage_state()
            finally:
                close_context = getattr(context, "close", None)
                if callable(close_context):
                    close_context()
        finally:
            close_browser = getattr(browser, "close", None)
            if callable(close_browser):
                close_browser()

    if not isinstance(state, Mapping) or not state:
        raise ClassifiedsWriterBootstrapError("Playwright returned an empty storage state")
    cookies = state.get("cookies")
    if not isinstance(cookies, list) or not cookies:
        raise ClassifiedsWriterBootstrapError("Playwright storage state has no session cookies")
    persistent: dict[str, Mapping[str, Any]] = {
        str(cookie.get("name")): cookie
        for cookie in cookies
        if isinstance(cookie, Mapping) and cookie.get("name") in PERSISTENT_AUTH_COOKIES
    }
    if set(persistent) != PERSISTENT_AUTH_COOKIES or any(
        not isinstance(cookie.get("value"), str)
        or not cookie.get("value")
        or not isinstance(cookie.get("expires"), (int, float))
        or cookie.get("expires", -1) <= 0
        or str(cookie.get("domain") or "").lstrip(".") != "127.0.0.1"
        for cookie in persistent.values()
    ):
        raise ClassifiedsWriterBootstrapError(
            "Playwright storage state lacks the pinned persistent participant cookies"
        )
    _write_storage_state_atomic(output, state)
    return {
        "output_path": str(output),
        "username_present": bool(username),
        "password_present": bool(password),
        "authenticated": True,
        "cookies_present": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create regular-participant Playwright state for the loopback Classifieds canary.",
        epilog=(
            "Inputs: --site-url and the environment variables "
            f"{CLASSIFIEDS_USERNAME_ENV} and {CLASSIFIEDS_PASSWORD_ENV}. "
            "Output: the authenticated storage-state JSON at --output-path. "
            "Safety: the URL must be the configured loopback origin; credentials "
            "are not accepted on the command line and the output stays outside the checkout."
        ),
    )
    parser.add_argument(
        "--site-url",
        required=True,
        help="Configured loopback Classifieds origin used for login and dashboard checks.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="External path for the 0600 Playwright storage-state JSON file.",
    )
    args = parser.parse_args(argv)
    output_path = str(Path(args.output_path).expanduser().resolve())
    try:
        metadata = mint_classifieds_writer_storage_state(args.site_url, output_path)
    except Exception as exc:
        # Keep command output safe: the error class is useful to operators but
        # credential values and Playwright's raw diagnostics are never echoed.
        print(
            json.dumps(
                {
                    "output_path": output_path,
                    "status": "error",
                    "error": type(exc).__name__,
                },
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
