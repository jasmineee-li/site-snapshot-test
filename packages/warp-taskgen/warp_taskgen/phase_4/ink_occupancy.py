"""Ink-occupancy oracle — per-character paint verification via non-bg pixel density.

Replaces the handoff §3.2-§3.5 byte-equal pixel compare against a hidden-DOM
reference container. That design required a separate reference render; the
current oracle verifies the actual captured page surface directly.

Core contract: for each non-whitespace character of the seeded payload, crop
the live-paint rectangle from the page-surface screenshot and ask
"does the crop contain enough non-background pixels for a real glyph to
have painted here?" Two OR'd thresholds are used so narrow and wide glyphs
both classify correctly:

- ``min_ink_pixels`` (absolute floor) — catches narrow / thin glyphs such as
  ``. , ' ` : ; i I l 1`` whose relative ink fraction is too small for a
  ratio threshold but whose absolute pixel count is still well above zero.
- ``min_occupancy`` (ratio floor) — catches normal-width glyphs (letters,
  digits) without being fooled by subpixel anti-aliasing along glyph edges.

Determinism: numpy L1 per-channel delta against a caller-supplied
``bg_rgb`` tuple. No OCR, no ML, no fuzzy matching, no tolerance knobs on
the delta (fixed at 24/255 per-channel-L1-sum by default — just above
Cleartype AA subpixel noise, just below genuine glyph stroke contrast).
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

from warp_taskgen.phase_4.pvpo_capture import Rect


def char_is_inked(
    image: Image.Image,
    rect: dict[str, int],
    bg_rgb: tuple[int, int, int],
    *,
    delta: int = 24,
    min_ink_pixels: int = 3,
    min_occupancy: float = 0.03,
    clip_offset: tuple[int, int] = (0, 0),
) -> bool:
    """Return ``True`` iff ``rect`` contains enough non-background pixels.

    Args:
        image: Pillow image opened from the page-surface PNG.
        rect: ``{"x", "y", "w", "h"}`` in viewport coordinates as emitted by
            the JS visibility query's ``liveRect``.
        bg_rgb: the page's effective background RGB, resolved by the JS
            query's ``resolveBackgroundColor`` ancestor walk.
        delta: per-channel L1 sum threshold; a pixel counts as "ink" iff
            ``|Δr| + |Δg| + |Δb| > delta``. Default ``24`` is just above
            Cleartype subpixel-AA noise in typical WebArena renders.
        min_ink_pixels: absolute floor. Catches punctuation whose ratio
            ``ink/area`` is below ``min_occupancy``.
        min_occupancy: relative floor. Catches normal-width glyphs. ORed
            with the absolute floor.
        clip_offset: screenshot clip origin in page coords. The ``rect`` is
            translated by subtracting ``(off_x, off_y)`` before cropping.

    Returns:
        Boolean. Zero-sized or unreadable rects return ``False``.
    """
    off_x, off_y = clip_offset
    try:
        x = int(rect["x"]) - off_x
        y = int(rect["y"]) - off_y
        w = int(rect["w"])
        h = int(rect["h"])
    except (KeyError, TypeError, ValueError):
        return False
    if w <= 0 or h <= 0:
        return False
    try:
        crop = image.crop((x, y, x + w, y + h)).convert("RGB")
    except (ValueError, OSError):
        return False
    if crop.size[0] == 0 or crop.size[1] == 0:
        return False
    arr = np.asarray(crop, dtype=np.int16)
    bg = np.array(bg_rgb, dtype=np.int16)
    diff = np.abs(arr - bg).sum(axis=-1)
    ink = int((diff > delta).sum())
    area = w * h
    if ink >= min_ink_pixels:
        return True
    return (ink / area) >= min_occupancy


def ink_occupancy_vector(
    screenshot_png: bytes,
    visibility_vec: list[dict[str, Any]],
    background_color: tuple[int, int, int],
    *,
    clip: Rect | None = None,
    delta: int = 24,
    min_ink_pixels: int = 3,
    min_occupancy: float = 0.03,
) -> list[bool]:
    """Per-char paint classifier. Shape-compatible with the old paint_verified_vector.

    - Whitespace → ``False``.
    - ``layoutVisible=False`` → ``False``.
    - Missing ``liveRect`` → ``False``.
    - Otherwise: :func:`char_is_inked`.

    The ``clip`` argument exists so screenshot-local coordinates can be
    derived from viewport-relative ``liveRect`` coords when page-surface
    returns a clipped PNG. PVPO currently captures the full viewport
    (``clip.x == clip.y == 0``) so translation is a no-op, but the param
    is kept for forward compatibility if clip-union returns.
    """
    if not screenshot_png:
        return [False] * len(visibility_vec)
    image = Image.open(BytesIO(screenshot_png))
    clip_offset = (clip.x, clip.y) if clip else (0, 0)

    out: list[bool] = []
    for entry in visibility_vec:
        if entry.get("isSpace"):
            out.append(False)
            continue
        if not entry.get("layoutVisible"):
            out.append(False)
            continue
        rect = entry.get("liveRect")
        if not rect:
            out.append(False)
            continue
        out.append(
            char_is_inked(
                image,
                rect,
                background_color,
                delta=delta,
                min_ink_pixels=min_ink_pixels,
                min_occupancy=min_occupancy,
                clip_offset=clip_offset,
            )
        )
    return out
