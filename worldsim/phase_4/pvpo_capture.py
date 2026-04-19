"""Paint-Verified Payload Oracle — atomic CDP capture.

Per-step capture (handoff §3.3, with the ink-occupancy simplification from
the post-handoff review that dropped the hidden reference container):

  1. ``Emulation.setVirtualTimePolicy({"policy": "pause"})``
  2. ``Runtime.evaluate(PVPO_QUERY_JS, returnByValue=True)`` — per-char
     visibility vector (layout-visible, in-viewport, not occluded) plus
     the page's effective background RGB resolved via an ancestor walk.
  3. ``HeadlessExperimental.beginFrame`` with ``screenshot.clip`` covering
     the visible viewport. Host-side ink-occupancy verification
     (:mod:`worldsim.phase_4.ink_occupancy`) measures non-background pixel
     density inside each character's live rect, which is strictly stronger
     than layout-visibility alone and correct under
     ``beginFrame``'s post-composite semantics (no offscreen DOM needed).
  4. ``Emulation.setVirtualTimePolicy({"policy": "advance"})``

The prior byte-equal reference-container oracle was removed in the clean
cutover: it required the reference to paint in the same composited frame
as the live payload, which is not achievable without per-step
``Emulation.setDeviceMetricsOverride`` viewport expansion. See
``docs/handoffs/codex-handoff-paint-verified-oracle.md`` and the
subsequent review.
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


PVPO_QUERY_JS: str = (Path(__file__).parent / "pvpo_query.js").read_text(encoding="utf-8")


@dataclass(frozen=True)
class Rect:
    """Integer rectangle in page / screenshot coordinates."""

    x: int
    y: int
    w: int
    h: int

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Rect:
        return cls(x=int(d["x"]), y=int(d["y"]), w=int(d["w"]), h=int(d["h"]))

    def as_cdp_clip(self, scale: float = 1.0) -> dict[str, Any]:
        """Shape expected by ``HeadlessExperimental.beginFrame.screenshot.clip``."""
        return {
            "x": float(self.x),
            "y": float(self.y),
            "width": float(self.w),
            "height": float(self.h),
            "scale": float(scale),
        }


@dataclass
class StepCapture:
    """One atomic capture: PNG pixels, visibility vector, bg color, clip."""

    screenshot_png: bytes
    visibility_vec: list[dict[str, Any]]
    background_color: tuple[int, int, int]
    has_damage: bool
    clip: Rect


async def atomic_capture_with_visibility(
    cdp_session: Any,
    *,
    viewport_rect: Rect,
) -> StepCapture:
    """Run the virtual-time-paused visibility query + ``beginFrame`` screenshot.

    Args:
        cdp_session: Playwright CDP session
            (``page.context().new_cdp_session(page)``) attached to a
            ``chrome-headless-shell`` launched with
            :data:`worldsim.phase_4.pvpo_browser_config.PVPO_LAUNCH_FLAGS`.
        viewport_rect: the visual viewport in page coordinates. The
            screenshot is clipped to this rect; post-composite capture
            means anything outside the viewport would not be in the frame
            even if the clip were larger.

    Returns:
        :class:`StepCapture` with decoded PNG bytes, the raw visibility
        vector, the resolved ``background_color`` tuple, the ``hasDamage``
        flag, and the clip rect. Host-side ink-occupancy verification runs
        downstream in ``encounter_detection`` using these fields.

    Note on ``hasDamage: false``:
        The visibility query is read-only and the compositor may skip the
        commit. The prior frame's pixels are still semantically correct
        (no layout mutation occurred). We trust them and log for
        observability — scope locked per handoff §9.
    """
    await cdp_session.send("Emulation.setVirtualTimePolicy", {"policy": "pause"})
    try:
        raw = await cdp_session.send(
            "Runtime.evaluate",
            {"expression": PVPO_QUERY_JS, "returnByValue": True},
        )
        visibility_vec, background_color = _unwrap_runtime_evaluate(raw)

        frame = await cdp_session.send(
            "HeadlessExperimental.beginFrame",
            {
                "screenshot": {
                    "format": "png",
                    "quality": 100,
                    "clip": viewport_rect.as_cdp_clip(),
                },
            },
        )

        has_damage = bool(frame.get("hasDamage", True))
        if not has_damage:
            logger.debug(
                "pvpo beginFrame hasDamage=False; trusting prior frame pixels "
                "(read-only visibility query should not dirty the compositor)"
            )
        png_b64 = frame.get("screenshotData") or ""
        screenshot_png = base64.b64decode(png_b64) if png_b64 else b""
    finally:
        await cdp_session.send("Emulation.setVirtualTimePolicy", {"policy": "advance"})

    return StepCapture(
        screenshot_png=screenshot_png,
        visibility_vec=visibility_vec,
        background_color=background_color,
        has_damage=has_damage,
        clip=viewport_rect,
    )


_DEFAULT_BG: tuple[int, int, int] = (255, 255, 255)


def _unwrap_runtime_evaluate(
    raw: dict[str, Any],
) -> tuple[list[dict[str, Any]], tuple[int, int, int]]:
    """Extract ``(entries, background_color)`` from a ``Runtime.evaluate`` result.

    The JS query returns ``{entries: [...], backgroundColor: {r, g, b}}``.
    Any departure from that shape falls back to ``([], _DEFAULT_BG)`` so
    the caller still gets a valid :class:`StepCapture` rather than an
    exception mid-capture.
    """
    result = raw.get("result") or {}
    if result.get("type") != "object" or "value" not in result:
        return [], _DEFAULT_BG
    value = result["value"]
    if not isinstance(value, dict):
        return [], _DEFAULT_BG
    entries = value.get("entries") or []
    if not isinstance(entries, list):
        entries = []
    bg = value.get("backgroundColor")
    if not isinstance(bg, dict):
        return entries, _DEFAULT_BG
    try:
        bg_rgb = (int(bg.get("r", 255)), int(bg.get("g", 255)), int(bg.get("b", 255)))
    except (TypeError, ValueError):
        bg_rgb = _DEFAULT_BG
    return entries, bg_rgb


def save_step_artifacts(
    trajectory_dir: Path,
    step_idx: int,
    capture: StepCapture,
) -> None:
    """Persist the PNG and PVPO metadata under the per-task trajectory dir.

    Layout:
        trajectory_dir / "screenshots" / f"step_{step_idx}.png"
        trajectory_dir / "pvpo"        / f"step_{step_idx}.json"

    The ``screenshots`` directory is the same path existing Phase 4
    consumers (judge trajectory slice, C1-C4 taxonomy classifiers) read —
    PVPO replaces the per-step screenshot capture in place.
    """
    shots = trajectory_dir / "screenshots"
    pvpo_dir = trajectory_dir / "pvpo"
    shots.mkdir(parents=True, exist_ok=True)
    pvpo_dir.mkdir(parents=True, exist_ok=True)
    (shots / f"step_{step_idx}.png").write_bytes(capture.screenshot_png)
    (pvpo_dir / f"step_{step_idx}.json").write_text(
        json.dumps(
            {
                "step_idx": step_idx,
                "visibility_vec": capture.visibility_vec,
                "background_color": list(capture.background_color),
                "has_damage": capture.has_damage,
                "clip": {
                    "x": capture.clip.x,
                    "y": capture.clip.y,
                    "w": capture.clip.w,
                    "h": capture.clip.h,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
