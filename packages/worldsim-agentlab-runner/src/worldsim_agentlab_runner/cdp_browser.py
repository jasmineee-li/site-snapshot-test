from __future__ import annotations

import contextlib
import json
import time
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen


@contextlib.contextmanager
def patched_chromium_launch(cdp_url: str | None, runtime: dict[str, Any]):
    """Route BrowserGym's Chromium launch through a dedicated CDP endpoint.

    BrowserGym launches Chromium internally. Phase 4 assigns each worker a
    dedicated chrome-headless-shell endpoint instead. Patching the launch call
    keeps BrowserGym/AgentLab unmodified while preserving WorldSim locality.
    """

    if not cdp_url:
        yield
        return

    from browsergym.core import _get_global_playwright

    playwright = _get_global_playwright()
    browser_type = playwright.chromium
    original_launch = browser_type.launch

    def connect_over_cdp_launch(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        runtime["cdp_url"] = cdp_url
        runtime["cdp_connect_started_at"] = time.time()
        browser = browser_type.connect_over_cdp(cdp_url, timeout=15_000)
        runtime["cdp_connect_finished_at"] = time.time()
        runtime["cdp_browser_version"] = _browser_version(cdp_url)
        return browser

    browser_type.launch = connect_over_cdp_launch  # type: ignore[method-assign]
    try:
        yield
    finally:
        browser_type.launch = original_launch  # type: ignore[method-assign]


def _browser_version(cdp_url: str) -> dict[str, Any] | None:
    try:
        parsed = urlparse(cdp_url)
        version_url = parsed._replace(path="/json/version", query="", fragment="").geturl()
        with urlopen(version_url, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None
