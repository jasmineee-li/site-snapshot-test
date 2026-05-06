from __future__ import annotations

import base64
import contextlib
import io
import time
from collections.abc import Iterator
from typing import Any

import numpy as np
from PIL import Image

from worldsim_agentlab_runner.sync_cdp import (
    PumpedSyncCdpSession,
    SyncCdpWorker,
    sync_cdp_deadline,
)

_DEFAULT_SCREENSHOT_TIMEOUT_S = 10.0


@contextlib.contextmanager
def patched_browsergym_screenshot_for_pvpo(
    cdp_url: str | None,
    runtime: dict[str, Any],
    *,
    timeout_s: float = _DEFAULT_SCREENSHOT_TIMEOUT_S,
) -> Iterator[None]:
    """Patch BrowserGym screenshots for begin-frame-controlled PVPO Chrome.

    BrowserGym 0.14.x always captures an observation screenshot during
    ``BrowserEnv.reset()`` via ``Page.captureScreenshot``. On
    ``chrome-headless-shell --enable-begin-frame-control`` that CDP command can
    wait indefinitely because display updates are driven by
    ``HeadlessExperimental.beginFrame``. Browser Use carries the same workaround
    in its watchdog path; the AgentLab sidecar needs it at the BrowserGym
    observation boundary.

    The patch is deliberately local to the sidecar run and restores both module
    references that BrowserGym uses: ``browsergym.core.observation`` and the
    already-imported alias in ``browsergym.core.env``.
    """

    if not cdp_url:
        yield
        return

    import browsergym.core.env as browsergym_env
    import browsergym.core.observation as browsergym_observation

    original_observation = browsergym_observation.extract_screenshot
    original_env = browsergym_env.extract_screenshot

    worker = SyncCdpWorker(
        timeout_s=timeout_s,
        name="worldsim-agentlab-browsergym-screenshot",
    )

    def _extract_screenshot(page: Any) -> np.ndarray:
        return _extract_beginframe_screenshot(
            page,
            runtime,
            cdp_url=cdp_url,
            timeout_s=timeout_s,
            worker=worker,
        )

    browsergym_observation.extract_screenshot = _extract_screenshot
    browsergym_env.extract_screenshot = _extract_screenshot
    runtime["browsergym_screenshot_patch"] = "headless_beginframe"
    runtime["browsergym_screenshot_timeout_s"] = timeout_s
    try:
        yield
    finally:
        browsergym_observation.extract_screenshot = original_observation
        browsergym_env.extract_screenshot = original_env
        worker.close()


def _extract_beginframe_screenshot(
    page: Any,
    runtime: dict[str, Any],
    *,
    cdp_url: str = "",
    timeout_s: float,
    worker: SyncCdpWorker | None = None,
) -> np.ndarray:
    cdp = page.context.new_cdp_session(page)
    metrics_overridden = False
    started_at = time.monotonic()
    try:
        dimensions = _viewport_dimensions(page)
        scale_factor = float(getattr(page, "_bgym_scale_factor", 1.0) or 1.0)
        original_metrics = {
            "width": dimensions["width"],
            "height": dimensions["height"],
            "deviceScaleFactor": 1.0,
            "mobile": False,
        }
        cdp.send(
            "Emulation.setDeviceMetricsOverride",
            {
                "width": dimensions["width"],
                "height": dimensions["height"],
                "deviceScaleFactor": scale_factor,
                "mobile": False,
            },
        )
        metrics_overridden = True
        result = _send_beginframe_screenshot(
            cdp,
            cdp_url=cdp_url,
            timeout_s=timeout_s,
            worker=worker,
        )
        screenshot_data = result.get("screenshotData") if isinstance(result, dict) else None
        if not isinstance(screenshot_data, str) or not screenshot_data:
            raise RuntimeError("HeadlessExperimental.beginFrame returned no screenshotData")
        runtime["browsergym_beginframe_screenshot_count"] = (
            int(runtime.get("browsergym_beginframe_screenshot_count") or 0) + 1
        )
        runtime["browsergym_beginframe_last_elapsed_s"] = round(time.monotonic() - started_at, 3)
        return _png_base64_to_rgb_array(screenshot_data)
    finally:
        if metrics_overridden:
            with contextlib.suppress(Exception):
                cdp.send("Emulation.setDeviceMetricsOverride", original_metrics)
        with contextlib.suppress(Exception):
            cdp.detach()


def _send_beginframe_screenshot(
    cdp: Any,
    *,
    cdp_url: str,
    timeout_s: float,
    worker: SyncCdpWorker | None,
) -> dict[str, Any]:
    params = {"screenshot": {"format": "png"}}
    if worker is None or not cdp_url:
        with sync_cdp_deadline(timeout_s, "BrowserGym PVPO beginFrame screenshot"):
            return cdp.send("HeadlessExperimental.beginFrame", params)

    async def _send(pump: Any) -> dict[str, Any]:
        from worldsim.phase_4.pvpo_beginframe import coordinator_for_pvpo_endpoint

        coordinator = coordinator_for_pvpo_endpoint(cdp_url)
        return await coordinator.send(
            PumpedSyncCdpSession(cdp, pump),
            params,
            timeout_s=timeout_s,
            label="browsergym-screenshot",
        )

    return worker.run(_send)


_sync_cdp_deadline = sync_cdp_deadline


def _viewport_dimensions(page: Any) -> dict[str, int]:
    viewport = getattr(page, "viewport_size", None)
    if isinstance(viewport, dict):
        width = int(viewport.get("width") or 0)
        height = int(viewport.get("height") or 0)
        if width > 0 and height > 0:
            return {"width": width, "height": height}
    dimensions = page.evaluate(
        """() => ({
            width: window.innerWidth,
            height: window.innerHeight
        })"""
    )
    if not isinstance(dimensions, dict):
        return {"width": 1280, "height": 720}
    return {
        "width": int(dimensions.get("width") or 1280),
        "height": int(dimensions.get("height") or 720),
    }


def _png_base64_to_rgb_array(data: str) -> np.ndarray:
    png_bytes = base64.b64decode(data)
    with Image.open(io.BytesIO(png_bytes)) as image:
        return np.array(image.convert("RGB"))
