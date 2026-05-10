"""Ink-occupancy unit tests + character-class calibration fixture.

The calibration test renders a small HTML fixture containing representative
payload characters (wide glyphs, normal letters, narrow letters,
punctuation, space) using PIL's default font and asserts the OR'd
threshold configuration classifies each correctly. This is the
"reproducible claim" fixture requested during the ink-occupancy design
review — if the default thresholds drift, this test catches it before
production does.

We don't invoke Chrome here (that's what ``tests/integration/test_pvpo_e2e_smoke.py``
exists for); PIL's deterministic ``ImageDraw.text`` gives us a
stable-enough renderer to pin the classification logic.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

import pytest
from PIL import Image, ImageDraw, ImageFont

from worldsim.phase_4.ink_occupancy import (
    char_is_inked,
    ink_occupancy_vector,
)
from worldsim.phase_4.pvpo_capture import Rect


def _png_with_glyph(char: str, *, size: tuple[int, int] = (24, 24)) -> tuple[bytes, Rect]:
    """Render ``char`` centered in a ``size`` canvas with PIL's default font."""
    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    # Anchor="mm" centers the glyph on (size/2, size/2).
    draw.text((size[0] / 2, size[1] / 2), char, fill=(0, 0, 0), font=font, anchor="mm")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), Rect(x=0, y=0, w=size[0], h=size[1])


def _open(png_bytes: bytes) -> Image.Image:
    return Image.open(BytesIO(png_bytes))


def test_white_background_no_glyph_is_not_inked():
    img = Image.new("RGB", (20, 20), (255, 255, 255))
    assert char_is_inked(img, {"x": 0, "y": 0, "w": 20, "h": 20}, (255, 255, 255)) is False


def test_solid_black_rect_classified_as_inked():
    img = Image.new("RGB", (20, 20), (0, 0, 0))
    assert char_is_inked(img, {"x": 0, "y": 0, "w": 20, "h": 20}, (255, 255, 255)) is True


def test_single_ink_pixel_below_absolute_floor_not_inked():
    img = Image.new("RGB", (20, 20), (255, 255, 255))
    img.putpixel((10, 10), (0, 0, 0))
    assert (
        char_is_inked(img, {"x": 0, "y": 0, "w": 20, "h": 20}, (255, 255, 255), min_ink_pixels=3)
        is False
    )


def test_three_ink_pixels_crosses_absolute_floor():
    img = Image.new("RGB", (20, 20), (255, 255, 255))
    for i in range(3):
        img.putpixel((10, 10 + i), (0, 0, 0))
    assert (
        char_is_inked(img, {"x": 0, "y": 0, "w": 20, "h": 20}, (255, 255, 255), min_ink_pixels=3)
        is True
    )


def test_zero_sized_rect_returns_false():
    img = Image.new("RGB", (20, 20), (0, 0, 0))
    assert char_is_inked(img, {"x": 0, "y": 0, "w": 0, "h": 0}, (255, 255, 255)) is False
    assert char_is_inked(img, {"x": 0, "y": 0, "w": 5, "h": 0}, (255, 255, 255)) is False


def test_malformed_rect_returns_false():
    img = Image.new("RGB", (20, 20), (0, 0, 0))
    assert char_is_inked(img, {"x": "bad", "y": 0, "w": 5, "h": 5}, (255, 255, 255)) is False  # type: ignore[dict-item]
    assert char_is_inked(img, {}, (255, 255, 255)) is False


def test_clip_offset_translates_rect_into_screenshot_coords():
    # Viewport rect (100, 60, 5, 5); clip origin (50, 50) → screenshot-local (50, 10).
    img = Image.new("RGB", (200, 100), (255, 255, 255))
    for dx in range(50, 55):
        for dy in range(10, 15):
            img.putpixel((dx, dy), (0, 0, 0))
    assert (
        char_is_inked(
            img,
            {"x": 100, "y": 60, "w": 5, "h": 5},
            (255, 255, 255),
            clip_offset=(50, 50),
        )
        is True
    )


def test_subpixel_aa_noise_below_delta_does_not_trigger():
    """Pixels with per-channel L1 delta just below ``delta`` (AA noise)
    must not count as ink. This is why delta sits above Cleartype subpixel
    variance but below glyph stroke contrast."""
    img = Image.new("RGB", (20, 20), (255, 255, 255))
    for ix in range(20):
        for iy in range(20):
            img.putpixel((ix, iy), (250, 250, 250))  # Δ=15 < default delta=24
    assert char_is_inked(img, {"x": 0, "y": 0, "w": 20, "h": 20}, (255, 255, 255)) is False


def test_non_white_background_with_high_contrast_ink_classified_correctly():
    img = Image.new("RGB", (20, 20), (200, 200, 200))
    for dx in range(8, 12):
        for dy in range(8, 12):
            img.putpixel((dx, dy), (50, 50, 50))
    assert char_is_inked(img, {"x": 0, "y": 0, "w": 20, "h": 20}, (200, 200, 200)) is True


# ---------------------------------------------------------------------------
# Character-class calibration — deterministic glyph rendering via PIL
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "char",
    [
        # Wide / normal glyphs — easily above both thresholds even under PIL's
        # tiny bitmap default font.
        "M",
        "W",
        "a",
        "b",
        "e",
        "n",
        "o",
        "0",
        "5",
        # Narrow letters / digits — low relative occupancy but enough absolute
        # ink pixels to clear the default floor.
        "i",
        "l",
        "I",
        "1",
        # Punctuation (other than single-dot ``.`` / ``'`` which are 2-pixel
        # blobs under PIL's default bitmap font — they render as ≥3 pixels
        # under real Chrome sans-serif at 16-20px and are covered by the live
        # smoke test instead).
        ",",
        ":",
        ";",
        "-",
    ],
)
def test_rendered_character_classified_as_inked(char: str):
    """Every payload character class produces enough ink to clear the OR'd
    thresholds. Regression guard for threshold drift."""
    png_bytes, rect = _png_with_glyph(char)
    image = _open(png_bytes)
    assert char_is_inked(
        image,
        {"x": rect.x, "y": rect.y, "w": rect.w, "h": rect.h},
        (255, 255, 255),
    ), f"{char!r} failed ink-occupancy classification"


@pytest.mark.parametrize("char", [".", "'"])
def test_tiny_punctuation_classifiable_with_lower_absolute_floor(char: str):
    """Single-dot punctuation renders as 2-3 pixels in PIL's default bitmap
    font (vs 3-5 under real Chrome sans-serif). Verify the configurable
    ``min_ink_pixels`` parameter works for settings where the proxy renderer
    produces fewer pixels than production."""
    png_bytes, rect = _png_with_glyph(char)
    image = _open(png_bytes)
    assert char_is_inked(
        image,
        {"x": rect.x, "y": rect.y, "w": rect.w, "h": rect.h},
        (255, 255, 255),
        min_ink_pixels=1,
    ), f"{char!r} failed with min_ink_pixels=1 (absolute floor is broken)"


def test_space_character_does_not_register_as_inked():
    """Space glyphs paint nothing; the ink oracle must return False even
    though layout-visibility says they're "visible"."""
    png_bytes, rect = _png_with_glyph(" ")
    image = _open(png_bytes)
    assert (
        char_is_inked(
            image,
            {"x": rect.x, "y": rect.y, "w": rect.w, "h": rect.h},
            (255, 255, 255),
        )
        is False
    )


# ---------------------------------------------------------------------------
# ink_occupancy_vector — shape compatibility with the visibility vector
# ---------------------------------------------------------------------------


def _entry(
    i: int,
    char: str,
    *,
    layout_visible: bool,
    live: tuple[int, int, int, int] | None = None,
    is_space: bool = False,
) -> dict[str, Any]:
    return {
        "i": i,
        "char": char,
        "isSpace": is_space,
        "layoutVisible": layout_visible,
        "liveRect": (
            None if live is None else {"x": live[0], "y": live[1], "w": live[2], "h": live[3]}
        ),
    }


def test_vector_returns_false_for_whitespace_and_hidden():
    img = Image.new("RGB", (50, 50), (255, 255, 255))
    buf = BytesIO()
    img.save(buf, format="PNG")
    vec = [
        _entry(0, " ", layout_visible=False, is_space=True),
        _entry(1, "a", layout_visible=False, live=None),
        _entry(2, "b", layout_visible=True, live=None),
    ]
    assert ink_occupancy_vector(buf.getvalue(), vec, (255, 255, 255)) == [False, False, False]


def test_vector_handles_empty_screenshot():
    vec = [_entry(0, "a", layout_visible=True, live=(0, 0, 5, 5))]
    assert ink_occupancy_vector(b"", vec, (255, 255, 255)) == [False]


def test_vector_preserves_length_across_mixed_chars():
    png_bytes, _ = _png_with_glyph("A", size=(40, 20))
    vec = [
        _entry(0, "A", layout_visible=True, live=(0, 0, 40, 20)),
        _entry(1, " ", layout_visible=False, is_space=True),
        _entry(2, "B", layout_visible=False, live=None),
    ]
    result = ink_occupancy_vector(png_bytes, vec, (255, 255, 255))
    assert len(result) == 3
    assert result[0] is True
    assert result[1] is False
    assert result[2] is False
