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
    runtime.update(
        {
            "browser_instance_scope": "agent_run",
            "browsergym_launch_patch": "connect_over_cdp",
            "browser_connect_count": 0,
        }
    )
    runtime["pre_run_cdp_targets"] = probe_cdp_targets(cdp_url)
    dirty_reason = _dirty_target_reason(runtime["pre_run_cdp_targets"])
    if dirty_reason:
        runtime["pre_run_cdp_clean"] = False
        runtime["pre_run_cdp_dirty_reason"] = dirty_reason
        raise RuntimeError(f"AgentLab PVPO browser endpoint is dirty before run: {dirty_reason}")
    runtime["pre_run_cdp_clean"] = True

    def connect_over_cdp_launch(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        runtime["browser_connect_count"] = int(runtime.get("browser_connect_count") or 0) + 1
        if runtime["browser_connect_count"] > 1:
            runtime["browser_connect_error"] = "multiple_browsergym_launches"
            raise RuntimeError(
                "AgentLab Phase 4 expected exactly one BrowserGym browser launch per run"
            )
        runtime["cdp_url"] = cdp_url
        runtime["browser_connected_over_cdp"] = True
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


def probe_cdp_targets(cdp_url: str) -> dict[str, Any]:
    """Return compact pre-run target evidence for a PVPO CDP endpoint."""

    if not cdp_url:
        return {"status": "skipped", "reason": "missing cdp_url"}
    try:
        parsed = urlparse(cdp_url)
        list_url = parsed._replace(path="/json/list", query="", fragment="").geturl()
        with urlopen(list_url, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "target_count": None,
            "page_count": None,
        }
    if not isinstance(payload, list):
        return {
            "status": "error",
            "error": "CDP /json/list returned a non-list payload",
            "target_count": None,
            "page_count": None,
        }
    targets = [item for item in payload if isinstance(item, dict)]
    pages = [item for item in targets if str(item.get("type") or "") == "page"]
    return {
        "status": "ok",
        "target_count": len(targets),
        "page_count": len(pages),
        "page_urls": [str(item.get("url") or "") for item in pages[:5]],
    }


def _browser_version(cdp_url: str) -> dict[str, Any] | None:
    try:
        parsed = urlparse(cdp_url)
        version_url = parsed._replace(path="/json/version", query="", fragment="").geturl()
        with urlopen(version_url, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _dirty_target_reason(probe: dict[str, Any]) -> str | None:
    if probe.get("status") != "ok":
        return None
    page_urls = probe.get("page_urls")
    if not isinstance(page_urls, list):
        return None
    if len(page_urls) > 1:
        return f"expected at most one page target, found {len(page_urls)}"
    if not page_urls:
        return None
    url = str(page_urls[0] or "")
    if (
        url == ""
        or url.startswith("about:blank")
        or url.startswith("chrome://newtab")
        or url.startswith("chrome://new-tab-page")
    ):
        return None
    return f"single page target is not blank: {url}"
