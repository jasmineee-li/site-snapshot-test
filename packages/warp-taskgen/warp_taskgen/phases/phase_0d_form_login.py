"""Phase 0d form-login bootstrap: the built-in Playwright sign-in path.

Owns the headless Chromium form-login bootstrap and its version-tolerant submit
selector order. See the ``warp_taskgen.phases.phase_0d_auth_bootstrap`` runner
for the phase contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from warp_taskgen.phases.phase_0d_site_auth_specs import AuthBootstrapError, _SiteSpec


async def _bootstrap_via_form_login(
    *,
    spec: _SiteSpec,
    site_url: str,
    output_path: Path,
    timeout_ms: int = 30_000,
    playwright_factory: Any | None = None,
) -> None:
    """Built-in Playwright form-login bootstrap for Phase 0d.

    Launches a headless Chromium, navigates to ``login_url`` (resolved against
    ``site_url`` when relative), fills the discovered credentials, clicks
    submit, waits for ``success_url_substring`` to appear in the final URL,
    and dumps ``storage_state`` to ``output_path``.

    ``playwright_factory`` is injected by tests to mock Playwright without
    requiring the real package to be installed. In production we import
    ``playwright.async_api.async_playwright`` lazily so operators who never
    use form_login do not need Playwright at all.

    Raises :class:`AuthBootstrapError` with a precise diagnostic on any
    failure (missing selectors, timeout, bad credentials, etc.).
    """
    recipe = spec.form_login
    if not isinstance(recipe, dict):
        raise AuthBootstrapError("form_login recipe missing — cannot bootstrap")
    credentials = spec.credentials
    if not isinstance(credentials, dict):
        raise AuthBootstrapError(
            "authentication.credentials must be an object with string username+password"
        )
    username = credentials.get("username")
    password = credentials.get("password")
    if not isinstance(username, str) or not username:
        raise AuthBootstrapError("authentication.credentials.username must be a non-empty string")
    if not isinstance(password, str) or not password:
        raise AuthBootstrapError("authentication.credentials.password must be a non-empty string")

    login_url = recipe["login_url"]
    # Allow relative login_url when site_url is available (e.g. "/login" ->
    # "<site_url>/login"). Absolute URLs pass through unchanged.
    if "://" not in login_url:
        if not site_url:
            raise AuthBootstrapError(
                f"form_login.login_url {login_url!r} is relative but no site_url was "
                "supplied; pass --instances or declare an absolute login_url"
            )
        login_url = site_url.rstrip("/") + "/" + login_url.lstrip("/")

    success_substring = recipe["success_url_substring"]

    # Remove any stale artifact so a partial write cannot satisfy the
    # post-generation non-empty check (mirrors _run_generator).
    if output_path.exists():
        try:
            output_path.unlink()
        except OSError:
            pass
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if playwright_factory is None:
        try:
            from playwright.async_api import async_playwright as _pw_factory
        except ImportError as exc:
            raise AuthBootstrapError(
                "form_login bootstrap requires Playwright: install 'playwright' and run "
                "'playwright install chromium', or switch the site to generator_script / "
                "pre_auth_script. Underlying import error: " + repr(exc)
            ) from exc
        playwright_factory = _pw_factory

    try:
        async with playwright_factory() as pw:
            browser = await pw.chromium.launch(headless=True)
            try:
                context = await browser.new_context()
                try:
                    page = await context.new_page()
                    # GitLab login pages can keep the full load event open on
                    # slow assets even after the form is ready. Selector waits
                    # and the success URL check below still gate correctness.
                    await page.goto(
                        login_url,
                        timeout=timeout_ms,
                        wait_until="domcontentloaded",
                    )
                    await page.fill(recipe["username_selector"], username, timeout=timeout_ms)
                    await page.fill(recipe["password_selector"], password, timeout=timeout_ms)
                    await _click_form_login_submit(
                        page,
                        spec=spec,
                        recipe=recipe,
                        login_url=login_url,
                        timeout_ms=timeout_ms,
                    )
                    # Wait for success_url_substring to appear in page.url.
                    await page.wait_for_url(
                        lambda url: success_substring in url,
                        timeout=timeout_ms,
                    )
                    await context.storage_state(path=str(output_path))
                finally:
                    await context.close()
            finally:
                await browser.close()
    except AuthBootstrapError:
        raise
    except Exception as exc:
        raise AuthBootstrapError(
            f"form_login bootstrap failed for site {spec.site_name!r}: {exc!r}"
        ) from exc

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise AuthBootstrapError(
            f"form_login bootstrap for site {spec.site_name!r} did not write a non-empty "
            f"storage_state at {output_path}"
        )
    try:
        json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuthBootstrapError(
            f"form_login bootstrap for site {spec.site_name!r} wrote invalid JSON to "
            f"{output_path}: {exc}"
        ) from exc


def _form_login_submit_selectors(
    *,
    spec: _SiteSpec,
    recipe: dict[str, Any],
    login_url: str,
) -> list[str]:
    """Return ordered submit selectors for version-tolerant form login.

    GitLab has shipped both Rails-style submit inputs and Pajamas-style submit
    buttons on the sign-in page. A stale recipe should not block auth refresh
    when the rest of the login contract is still valid.
    """
    declared = recipe["submit_selector"]
    selectors: list[str] = []
    is_gitlab_login = spec.site_name.lower() == "gitlab" or "/users/sign_in" in login_url
    if is_gitlab_login:
        selectors.extend(
            [
                "button[type='submit'], input[type='submit']",
                "button.js-sign-in-button",
                "button[data-testid='sign-in-button']",
            ]
        )
    selectors.append(declared)

    unique: list[str] = []
    for selector in selectors:
        if selector not in unique:
            unique.append(selector)
    return unique


async def _click_form_login_submit(
    page: Any,
    *,
    spec: _SiteSpec,
    recipe: dict[str, Any],
    login_url: str,
    timeout_ms: int,
) -> None:
    errors: list[str] = []
    for selector in _form_login_submit_selectors(spec=spec, recipe=recipe, login_url=login_url):
        try:
            await page.click(selector, timeout=timeout_ms)
            return
        except Exception as exc:  # pragma: no cover - real Playwright error types vary.
            errors.append(f"{selector!r}: {exc!r}")
    raise AuthBootstrapError(
        f"form_login submit click failed for site {spec.site_name!r}; tried " + "; ".join(errors)
    )
