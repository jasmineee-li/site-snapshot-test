"""Paint-Verified Payload Oracle — atomic CDP capture and pixel verification.

Per-step capture (handoff §3.3):

  1. ``Emulation.setVirtualTimePolicy({"policy": "pause"})``
  2. ``Runtime.evaluate(PVPO_QUERY_JS, returnByValue=True)`` — per-char
     visibility vector with live and reference rects.
  3. ``HeadlessExperimental.beginFrame`` with ``screenshot.clip`` covering
     the union of the visual viewport and the hidden reference container.
  4. ``Emulation.setVirtualTimePolicy({"policy": "advance"})``

Host-side verification (handoff §3.5):

  ``paint_verified_vector(png, visibility_vec)`` crops the live and reference
  regions from the same committed frame and returns byte-equal booleans. Both
  regions share the exact same Blink+HarfBuzz+Skia paint pipeline within the
  single committed frame, which is what makes zero-tolerance pixel comparison
  correct by construction (no OCR, no VLM, no external Skia, no fuzzy match).

See ``docs/handoffs/codex-handoff-paint-verified-oracle.md`` for the full
design rationale and references to the Chromium source that confirms each
decision.
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)


# Loaded at import time. The JS source lives next to this module so edits to
# the query travel with the Python that evaluates it (single source of truth
# for PVPO invariants).
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
    """One atomic capture: PNG pixels, the visibility vector, the clip used."""

    screenshot_png: bytes
    visibility_vec: list[dict[str, Any]]
    has_damage: bool
    clip: Rect


async def atomic_capture_with_visibility(
    cdp_session: Any,
    *,
    viewport_rect: Rect,
) -> StepCapture:
    """Run the virtual-time-paused visibility query + ``beginFrame`` screenshot.

    Args:
        cdp_session: a Playwright CDP session (``page.context().new_cdp_session(page)``)
            attached to a ``chrome-headless-shell`` launched with
            :data:`worldsim.phase_4.pvpo_browser_config.PVPO_LAUNCH_FLAGS`.
        viewport_rect: the visual viewport in page coordinates. Combined with
            the reference-container rect derived from the visibility vector
            to form the screenshot clip.

    Returns:
        :class:`StepCapture` with decoded PNG bytes, the raw visibility vector,
        the ``hasDamage`` flag from ``beginFrame``, and the clip used (so
        host-side pixel compare can translate viewport-relative rects into
        screenshot-local coordinates).

    Note on ``hasDamage: false``:
        The visibility query is read-only and the compositor may skip the
        commit, returning ``hasDamage: false``. The prior frame's pixels are
        still semantically correct for the current DOM state (no layout
        mutation occurred since the last committed frame). We trust them and
        log for observability. Do NOT retry or force damage — scope locked
        per handoff §9.
    """
    await cdp_session.send("Emulation.setVirtualTimePolicy", {"policy": "pause"})
    try:
        raw = await cdp_session.send(
            "Runtime.evaluate",
            {"expression": PVPO_QUERY_JS, "returnByValue": True},
        )
        visibility_vec = _unwrap_runtime_evaluate(raw)

        clip_rect = _clip_union(viewport_rect, visibility_vec)
        frame = await cdp_session.send(
            "HeadlessExperimental.beginFrame",
            {
                "screenshot": {
                    "format": "png",
                    "quality": 100,
                    "clip": clip_rect.as_cdp_clip(),
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
        has_damage=has_damage,
        clip=clip_rect,
    )


def _unwrap_runtime_evaluate(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the ``returnByValue`` payload from a ``Runtime.evaluate`` result."""
    result = raw.get("result") or {}
    if result.get("type") == "object" and "value" in result:
        value = result["value"]
        if isinstance(value, list):
            return value
    return []


def _clip_union(viewport_rect: Rect, visibility_vec: list[dict[str, Any]]) -> Rect:
    """Bounding-box union of the viewport and every non-null reference rect.

    If no reference rects are present (no reference container injected yet),
    returns ``viewport_rect`` unchanged.
    """
    x_min = viewport_rect.x
    y_min = viewport_rect.y
    x_max = viewport_rect.x + viewport_rect.w
    y_max = viewport_rect.y + viewport_rect.h
    have_ref = False
    for entry in visibility_vec:
        ref = entry.get("refRect")
        if not ref:
            continue
        have_ref = True
        rx = int(ref["x"])
        ry = int(ref["y"])
        rw = int(ref["w"])
        rh = int(ref["h"])
        x_min = min(x_min, rx)
        y_min = min(y_min, ry)
        x_max = max(x_max, rx + rw)
        y_max = max(y_max, ry + rh)
    if not have_ref:
        return viewport_rect
    return Rect(x=x_min, y=y_min, w=x_max - x_min, h=y_max - y_min)


def paint_verified_vector(
    screenshot_png: bytes,
    visibility_vec: list[dict[str, Any]],
    clip: Rect | None = None,
) -> list[bool]:
    """Per-char: True iff layout-visible AND live crop byte-equal to reference crop.

    Whitespace characters return ``False``. Characters not layout-visible
    (occluded, outside viewport, ancestor hidden) return ``False``. Characters
    missing either a live or reference rect return ``False``. Pixel comparison
    uses ``PIL.Image.tobytes`` byte equality with zero tolerance — the live and
    reference regions are rendered by the same Skia instance in the same
    committed frame, so equality is the correct check.

    Args:
        screenshot_png: PNG bytes from
            :func:`atomic_capture_with_visibility`.
        visibility_vec: the ``visibility_vec`` from the same capture.
        clip: the clip rect used when taking the screenshot. Viewport-relative
            rects in ``visibility_vec`` are translated into screenshot-local
            coordinates by subtracting ``(clip.x, clip.y)``. When ``None``,
            assumes the screenshot origin matches page origin.

    Returns:
        A list of booleans, one per character in the original payload
        (including whitespace). Length equals ``len(visibility_vec)``.
    """
    image = Image.open(BytesIO(screenshot_png)) if screenshot_png else None
    off_x = clip.x if clip else 0
    off_y = clip.y if clip else 0

    out: list[bool] = []
    for entry in visibility_vec:
        if entry.get("isSpace"):
            out.append(False)
            continue
        if not entry.get("layoutVisible"):
            out.append(False)
            continue
        live = entry.get("liveRect")
        ref = entry.get("refRect")
        if not live or not ref or image is None:
            out.append(False)
            continue
        try:
            live_crop = image.crop(_pillow_box(live, off_x, off_y))
            ref_crop = image.crop(_pillow_box(ref, off_x, off_y))
        except (ValueError, OSError):
            out.append(False)
            continue
        if live_crop.size != ref_crop.size:
            # Defensive: same paint pipeline should yield identical sizes.
            out.append(False)
            continue
        out.append(live_crop.tobytes() == ref_crop.tobytes())
    return out


def _pillow_box(rect: dict[str, Any], off_x: int, off_y: int) -> tuple[int, int, int, int]:
    x = int(rect["x"]) - off_x
    y = int(rect["y"]) - off_y
    w = int(rect["w"])
    h = int(rect["h"])
    return (x, y, x + w, y + h)


def save_step_artifacts(
    trajectory_dir: Path,
    step_idx: int,
    capture: StepCapture,
) -> None:
    """Persist the PNG and PVPO metadata under the per-task trajectory dir.

    Layout:
        trajectory_dir / "screenshots" / f"step_{step_idx}.png"
        trajectory_dir / "pvpo"        / f"step_{step_idx}.json"

    The ``screenshots`` directory is the same path existing Phase 4 consumers
    (the judge trajectory slice, C1-C4 taxonomy classifiers) already read —
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
