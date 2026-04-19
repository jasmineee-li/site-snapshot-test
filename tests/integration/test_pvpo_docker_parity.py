"""PVPO Linux-host vs in-container parity.

Launches a fixture page twice — once via the host-direct chromium that
``playwright install`` set up, once via the ``chrome-headless-shell`` Docker
container reachable at ``127.0.0.1:9222`` — and compares the per-step
visibility vectors. On Linux with both environments available, the vectors
must be byte-identical: same Blink+HarfBuzz+Skia paint pipeline inside the
container, so layout measurements and ``refRect`` coordinates match.

Skipped on macOS/Darwin (native ``HeadlessExperimental.beginFrame`` is not
supported; comparing a non-functional host path to an in-container path
isn't meaningful there). Skipped when the Docker container is not reachable.
"""

from __future__ import annotations

import hashlib
import json
import platform
import socket
from pathlib import Path
from urllib.request import urlopen

import pytest

from worldsim.phase_4.pvpo_browser_config import (
    inject_animation_killer,
    inject_reference_container,
)
from worldsim.phase_4.pvpo_capture import (
    Rect,
    atomic_capture_with_visibility,
)

CDP_HOST = "127.0.0.1"
CDP_PORT = 9222


def _on_linux() -> bool:
    return platform.system().lower() == "linux"


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


_FIXTURE_HTML = """
<!doctype html>
<html><head><meta charset="utf-8"><style>
  body { font-family: sans-serif; padding: 20px; }
  article { margin-top: 40px; }
</style></head>
<body>
  <article>
    <h1>Parity fixture</h1>
    <p>
      <span data-worldsim-payload="1">ABCDEFGH</span>
    </p>
  </article>
</body></html>
"""


def _vector_hash(vec: list[dict]) -> str:
    normalized = json.dumps(
        [
            {
                "i": entry.get("i"),
                "char": entry.get("char"),
                "isSpace": entry.get("isSpace"),
                "layoutVisible": entry.get("layoutVisible"),
                "liveRect": entry.get("liveRect"),
                "refRect": entry.get("refRect"),
            }
            for entry in vec
        ],
        sort_keys=True,
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


async def _capture_via_cdp(cdp: object) -> list[dict]:
    from worldsim.phase_4.pvpo_capture import atomic_capture_with_visibility as _capture

    viewport = Rect(x=0, y=0, w=1280, h=720)
    capture = await _capture(cdp, viewport_rect=viewport)
    return capture.visibility_vec


@pytest.mark.integration
@pytest.mark.skipif(
    not _on_linux(),
    reason="parity test only runs on Linux hosts (beginFrame is unsupported natively on macOS/Windows)",
)
@pytest.mark.skipif(
    not _chrome_headless_shell_reachable(),
    reason="chrome-headless-shell not reachable at 127.0.0.1:9222 — start the Docker container first",
)
@pytest.mark.asyncio
async def test_pvpo_visibility_vectors_byte_identical_host_vs_container(tmp_path: Path):
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        pytest.skip("playwright not installed")

    async with async_playwright() as pw:
        host_browser = await pw.chromium.launch(
            headless=True,
            args=[
                "--enable-begin-frame-control",
                "--run-all-compositor-stages-before-draw",
                "--disable-checker-imaging",
                "--no-sandbox",
            ],
        )
        host_context = await host_browser.new_context()
        host_page = await host_context.new_page()
        await host_page.set_content(_FIXTURE_HTML)
        host_cdp = await host_context.new_cdp_session(host_page)
        await inject_animation_killer(host_page, host_cdp)
        await inject_reference_container(host_page, "ABCDEFGH", "[data-worldsim-payload]")
        host_capture = await atomic_capture_with_visibility(
            host_cdp, viewport_rect=Rect(x=0, y=0, w=1280, h=720)
        )
        await host_page.close()
        await host_browser.close()

        container_browser = await pw.chromium.connect_over_cdp(f"http://{CDP_HOST}:{CDP_PORT}")
        container_context = (
            container_browser.contexts[0]
            if container_browser.contexts
            else await container_browser.new_context()
        )
        container_page = await container_context.new_page()
        await container_page.set_content(_FIXTURE_HTML)
        container_cdp = await container_context.new_cdp_session(container_page)
        await inject_animation_killer(container_page, container_cdp)
        await inject_reference_container(container_page, "ABCDEFGH", "[data-worldsim-payload]")
        container_capture = await atomic_capture_with_visibility(
            container_cdp, viewport_rect=Rect(x=0, y=0, w=1280, h=720)
        )
        await container_page.close()
        await container_browser.close()

    host_hash = _vector_hash(host_capture.visibility_vec)
    container_hash = _vector_hash(container_capture.visibility_vec)
    assert host_hash == container_hash, (
        "PVPO visibility vectors diverged between host and container; paint "
        "pipeline parity is broken. Host first entry: "
        f"{host_capture.visibility_vec[:1]!r}; container first entry: "
        f"{container_capture.visibility_vec[:1]!r}"
    )
