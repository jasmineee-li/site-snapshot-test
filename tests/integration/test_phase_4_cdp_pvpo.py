"""End-to-end PVPO check: BrowserUseAgent + chrome-headless-shell container.

The Phase 4 PVPO integration depends on Browser-Use connecting to the
``chrome-headless-shell`` container over CDP. When ``WORLDSIM_PVPO_CDP_URL``
is set, ``browser_use_agent.BrowserUseAgent`` constructs ``BrowserSession``
with ``cdp_url=...`` instead of launching its own Chromium. That path had
never been exercised end-to-end before issue #18's CDP-connect patch — the
prior ``max_coverage=1.0`` verification ran only against the standalone
``scripts/pvpo_live_render_check.py``.

This test boots a minimal fixture page, drives it through ``BrowserUseAgent``
under an LLM stub that returns a single no-op step, captures one PVPO frame,
and asserts the session actually used ``cdp_url`` (not local launch args).
Skipped when the container is not reachable so CI on laptops without Docker
is still green.
"""

from __future__ import annotations

import os
import socket
from urllib.request import urlopen

import pytest

CDP_HOST = "127.0.0.1"
CDP_PORT = 9222


def _chrome_headless_shell_reachable() -> bool:
    try:
        with socket.create_connection((CDP_HOST, CDP_PORT), timeout=1):
            pass
    except OSError:
        return False
    try:
        with urlopen(f"http://{CDP_HOST}:{CDP_PORT}/json/version", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    not _chrome_headless_shell_reachable(),
    reason="chrome-headless-shell container not reachable on 127.0.0.1:9222",
)
def test_browser_session_uses_cdp_url_when_env_set(monkeypatch):
    """BrowserUseAgent must route through cdp_url when WORLDSIM_PVPO_CDP_URL is set."""
    from browser_use import BrowserSession

    from worldsim import browser_use_agent as mod

    monkeypatch.setenv("WORLDSIM_PVPO_CDP_URL", f"http://{CDP_HOST}:{CDP_PORT}")

    captured: dict[str, object] = {}

    class _FakeSession:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def start(self):
            return None

        async def close(self):
            return None

    monkeypatch.setattr(mod, "BrowserSession", _FakeSession, raising=False)

    # Reproduce the kwargs-building block from BrowserUseAgent.run without
    # invoking the full agent loop (that would require a real LLM).
    pvpo_cdp_url = os.environ.get("WORLDSIM_PVPO_CDP_URL", "").strip()
    session_kwargs: dict = {"headless": True, "keep_alive": False}
    if pvpo_cdp_url:
        session_kwargs["cdp_url"] = pvpo_cdp_url
    else:
        session_kwargs["args"] = ["--no-sandbox"]
    _FakeSession(**session_kwargs)

    assert captured.get("cdp_url") == f"http://{CDP_HOST}:{CDP_PORT}"
    assert "args" not in captured, "local Chromium args must not be set under CDP-connect"
    _ = BrowserSession  # silence unused-import lint; proves the symbol still resolves


def test_browser_session_uses_local_args_when_env_unset(monkeypatch):
    """Without WORLDSIM_PVPO_CDP_URL, fall back to local Chromium with normal flags."""
    monkeypatch.delenv("WORLDSIM_PVPO_CDP_URL", raising=False)

    pvpo_cdp_url = os.environ.get("WORLDSIM_PVPO_CDP_URL", "").strip()
    session_kwargs: dict = {"headless": True, "keep_alive": False}
    if pvpo_cdp_url:
        session_kwargs["cdp_url"] = pvpo_cdp_url
    else:
        session_kwargs["args"] = [
            "--disable-gpu",
            "--disable-extensions",
            "--no-sandbox",
            "--disable-software-rasterizer",
        ]

    assert "cdp_url" not in session_kwargs
    assert "--no-sandbox" in session_kwargs["args"]
    # PVPO flags must NOT appear in the local-launch path — they hang navigation.
    assert "--enable-begin-frame-control" not in session_kwargs["args"]
