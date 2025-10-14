from __future__ import annotations
import os
from urllib.parse import urlparse
import os

from playwright.sync_api import Browser, Page
from ..shared.base_playwright import BasePlaywrightComputer


def _host_allowed(url: str, allowed_hosts: set[str]) -> bool:
    try:
        u = urlparse(url)
        host = (u.hostname or "").lower()
        if u.scheme not in ("http", "https", ""):
            return False
        # Allow bare requests Playwright may make (data:, about:) at resource level
        if u.scheme in ("", "about", "data"):
            return True
        # Exact or subdomain match
        return any(host == h or host.endswith("." + h) for h in allowed_hosts)
    except Exception:
        return False


class LocalPlaywrightBrowser(BasePlaywrightComputer):
    """Launches a local Chromium instance using Playwright with a localhost allowlist."""

    def __init__(
        self,
        headless: bool = False,
        start_url: str | None = None,
        allowed_hosts: set[str] | None = None,
    ):
        super().__init__()
        self.headless = headless
        self.start_url = start_url or os.getenv(
            "CUA_START_URL", "http://localhost:8000/gmail"
        )
        # allow localhost + loopback by default
        self.allowed_hosts = allowed_hosts or {"localhost", "127.0.0.1"}

    def _get_browser_and_page(self) -> tuple[Browser, Page]:
        width, height = self.get_dimensions()
        debug = str(os.getenv("CUA_DEBUG", "0")).lower() in {"1", "true", "yes"}

        # Keep launch args minimal; window size is optional if you set viewport later
        launch_args = [
            f"--window-size={width},{height}",
            "--disable-extensions",
            "--disable-file-system",
        ]

        browser = self._playwright.chromium.launch(
            headless=self.headless,
            args=launch_args,
            # chromium_sandbox may fail on macOS; enable only if your env supports it
            chromium_sandbox=False,
        )

        # If you see TLS issues in corp/VPN, temporarily set ignore_https_errors=True
        context = browser.new_context(
            viewport={"width": width, "height": height},
            ignore_https_errors=False,
        )

        # Strict network allowlist
        context.route(
            "**/*",
            lambda route, request: (
                route.continue_()
                if _host_allowed(request.url, self.allowed_hosts)
                else route.abort()
            ),
        )

        if debug:
            try:
                context.on(
                    "request", lambda req: print(f"[CTX REQ] {req.method} {req.url}")
                )
                context.on(
                    "response", lambda res: print(f"[CTX RES] {res.status} {res.url}")
                )
            except Exception as e:
                print(
                    f"[LocalPlaywrightBrowser] Failed to add context debug handlers: {e}"
                )

        # Track pages
        context.on("page", self._handle_new_page)

        page = context.new_page()
        page.on("close", self._handle_page_close)

        # Use a fully-qualified URL
        try:
            page.goto(self.start_url, wait_until="load", timeout=30000)
        except Exception as e:
            # Helpful fallback & logging
            print(f"[LocalPlaywrightBrowser] Failed to goto {self.start_url}: {e}")
            try:
                page.goto("http://localhost:8000", wait_until="load", timeout=15000)
            except Exception as e2:
                print(f"[LocalPlaywrightBrowser] Fallback goto failed: {e2}")
                raise

        return browser, page

    def _handle_new_page(self, page: Page):
        print("New page created")
        self._page = page
        page.on("close", self._handle_page_close)

    def _handle_page_close(self, page: Page):
        print("Page closed")
        if self._page == page:
            ctx = self._browser.contexts[0] if self._browser.contexts else None
            if ctx and ctx.pages:
                self._page = ctx.pages[-1]
            else:
                print("Warning: All pages have been closed.")
                self._page = None
