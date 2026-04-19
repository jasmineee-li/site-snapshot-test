from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from worldsim.phase_4.pvpo_capture import (
    PVPO_QUERY_JS,
    Rect,
    StepCapture,
    atomic_capture_with_visibility,
    paint_verified_vector,
    save_step_artifacts,
)


def _make_png(
    boxes: dict[tuple[int, int, int, int], tuple[int, int, int]],
    *,
    size: tuple[int, int] = (300, 120),
) -> bytes:
    """Build a deterministic PNG with solid-color rectangles at given positions.

    ``boxes`` maps ``(x, y, w, h)`` to ``(r, g, b)``. Everything else is white.
    Sufficient for byte-equal comparisons because PIL's PNG encoder is
    deterministic for the same pixel buffer.
    """
    img = Image.new("RGB", size, (255, 255, 255))
    for (x, y, w, h), color in boxes.items():
        for ix in range(x, x + w):
            for iy in range(y, y + h):
                if 0 <= ix < size[0] and 0 <= iy < size[1]:
                    img.putpixel((ix, iy), color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _entry(
    i: int,
    char: str,
    *,
    layout_visible: bool,
    live: tuple[int, int, int, int] | None = None,
    ref: tuple[int, int, int, int] | None = None,
    is_space: bool = False,
) -> dict[str, Any]:
    def _rect(t: tuple[int, int, int, int] | None) -> dict[str, int] | None:
        if t is None:
            return None
        return {"x": t[0], "y": t[1], "w": t[2], "h": t[3]}

    return {
        "i": i,
        "char": char,
        "isSpace": is_space,
        "layoutVisible": layout_visible,
        "liveRect": _rect(live),
        "refRect": _rect(ref),
    }


def test_all_visible_byte_equal_returns_all_true():
    png = _make_png(
        {
            (10, 10, 5, 5): (255, 0, 0),
            (100, 10, 5, 5): (255, 0, 0),
            (20, 10, 5, 5): (0, 128, 0),
            (110, 10, 5, 5): (0, 128, 0),
        }
    )
    vec = [
        _entry(0, "a", layout_visible=True, live=(10, 10, 5, 5), ref=(100, 10, 5, 5)),
        _entry(1, "b", layout_visible=True, live=(20, 10, 5, 5), ref=(110, 10, 5, 5)),
    ]
    assert paint_verified_vector(png, vec) == [True, True]


def test_all_occluded_returns_all_false():
    png = _make_png({})
    vec = [
        _entry(0, "a", layout_visible=False),
        _entry(1, "b", layout_visible=False),
    ]
    assert paint_verified_vector(png, vec) == [False, False]


def test_partial_overlap_returns_mixed_booleans():
    png = _make_png(
        {
            (10, 10, 5, 5): (255, 0, 0),
            (100, 10, 5, 5): (255, 0, 0),  # matches char 0
            (20, 10, 5, 5): (0, 255, 0),
            (110, 10, 5, 5): (0, 0, 255),  # diverges for char 1
        }
    )
    vec = [
        _entry(0, "a", layout_visible=True, live=(10, 10, 5, 5), ref=(100, 10, 5, 5)),
        _entry(1, "b", layout_visible=True, live=(20, 10, 5, 5), ref=(110, 10, 5, 5)),
    ]
    assert paint_verified_vector(png, vec) == [True, False]


def test_whitespace_chars_excluded():
    png = _make_png({})
    vec = [
        _entry(0, " ", layout_visible=False, is_space=True),
        _entry(1, "\t", layout_visible=False, is_space=True),
    ]
    assert paint_verified_vector(png, vec) == [False, False]


def test_size_mismatch_returns_false_defensively():
    png = _make_png(
        {
            (10, 10, 5, 5): (0, 0, 0),
            (100, 10, 8, 5): (0, 0, 0),  # wider reference rect
        }
    )
    vec = [
        _entry(0, "a", layout_visible=True, live=(10, 10, 5, 5), ref=(100, 10, 8, 5)),
    ]
    assert paint_verified_vector(png, vec) == [False]


def test_missing_rect_returns_false():
    png = _make_png({(10, 10, 5, 5): (0, 0, 0)})
    vec = [
        _entry(0, "a", layout_visible=True, live=(10, 10, 5, 5), ref=None),
        _entry(1, "b", layout_visible=True, live=None, ref=(10, 10, 5, 5)),
    ]
    assert paint_verified_vector(png, vec) == [False, False]


def test_ligature_payload_per_char_rects_independent():
    # Models what the JS query emits for 'fi' / 'ffi' payloads: per-char rects
    # remain distinct (Range-based measurement does not break ligatures) and
    # the pixel compare works independently for each character position.
    png = _make_png(
        {
            (10, 10, 4, 6): (0, 0, 0),  # live 'f'
            (14, 10, 4, 6): (0, 0, 0),  # live 'i'
            (18, 10, 4, 6): (0, 0, 0),  # live 'i' (second 'i' in 'ffi')
            (100, 10, 4, 6): (0, 0, 0),  # ref 'f'
            (104, 10, 4, 6): (0, 0, 0),  # ref 'i'
            (108, 10, 4, 6): (0, 0, 0),  # ref 'i'
        }
    )
    vec = [
        _entry(0, "f", layout_visible=True, live=(10, 10, 4, 6), ref=(100, 10, 4, 6)),
        _entry(1, "i", layout_visible=True, live=(14, 10, 4, 6), ref=(104, 10, 4, 6)),
        _entry(2, "i", layout_visible=True, live=(18, 10, 4, 6), ref=(108, 10, 4, 6)),
    ]
    assert paint_verified_vector(png, vec) == [True, True, True]


def test_clip_offset_translates_viewport_to_screenshot_coords():
    size = (200, 100)
    img = Image.new("RGB", size, (255, 255, 255))
    for ix in range(50, 55):
        for iy in range(10, 15):
            img.putpixel((ix, iy), (255, 0, 0))
    for ix in range(150, 155):
        for iy in range(10, 15):
            img.putpixel((ix, iy), (255, 0, 0))
    buf = BytesIO()
    img.save(buf, format="PNG")
    png = buf.getvalue()

    clip = Rect(x=50, y=50, w=200, h=100)
    vec = [
        _entry(0, "a", layout_visible=True, live=(100, 60, 5, 5), ref=(200, 60, 5, 5)),
    ]
    assert paint_verified_vector(png, vec, clip=clip) == [True]


@pytest.mark.asyncio
async def test_atomic_capture_sequences_cdp_calls():
    png_bytes = _make_png({(10, 10, 5, 5): (255, 0, 0)})
    cdp = AsyncMock()

    vec_value = [
        _entry(0, "a", layout_visible=True, live=(10, 10, 5, 5), ref=(100, 10, 5, 5)),
    ]

    async def _send(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if method == "Emulation.setVirtualTimePolicy":
            return {}
        if method == "Runtime.evaluate":
            return {"result": {"type": "object", "value": vec_value}}
        if method == "HeadlessExperimental.beginFrame":
            return {
                "hasDamage": True,
                "screenshotData": base64.b64encode(png_bytes).decode("ascii"),
            }
        raise AssertionError(f"unexpected CDP method {method}")

    cdp.send = AsyncMock(side_effect=_send)

    viewport = Rect(x=0, y=0, w=200, h=100)
    capture = await atomic_capture_with_visibility(cdp, viewport_rect=viewport)

    assert capture.has_damage is True
    assert capture.screenshot_png == png_bytes
    assert capture.visibility_vec == vec_value

    methods = [c.args[0] for c in cdp.send.call_args_list]
    assert methods == [
        "Emulation.setVirtualTimePolicy",
        "Runtime.evaluate",
        "HeadlessExperimental.beginFrame",
        "Emulation.setVirtualTimePolicy",
    ]
    # First setVirtualTimePolicy must be pause, last must be advance.
    policies = [
        c.args[1]["policy"]
        for c in cdp.send.call_args_list
        if c.args[0] == "Emulation.setVirtualTimePolicy"
    ]
    assert policies == ["pause", "advance"]


@pytest.mark.asyncio
async def test_atomic_capture_resumes_virtual_time_on_error():
    cdp = AsyncMock()

    async def _send(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if method == "Emulation.setVirtualTimePolicy":
            return {}
        if method == "Runtime.evaluate":
            raise RuntimeError("synthetic CDP failure")
        raise AssertionError(f"unexpected CDP method {method}")

    cdp.send = AsyncMock(side_effect=_send)

    with pytest.raises(RuntimeError, match="synthetic CDP failure"):
        await atomic_capture_with_visibility(cdp, viewport_rect=Rect(0, 0, 100, 100))

    # The ``advance`` policy must have been sent in the finally block so the
    # page does not stay paused after a failed capture.
    policies = [
        c.args[1]["policy"]
        for c in cdp.send.call_args_list
        if c.args[0] == "Emulation.setVirtualTimePolicy"
    ]
    assert policies == ["pause", "advance"]


@pytest.mark.asyncio
async def test_atomic_capture_has_damage_false_does_not_retry(caplog):
    png_bytes = _make_png({})
    cdp = AsyncMock()

    begin_frame_count = {"n": 0}

    async def _send(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if method == "Emulation.setVirtualTimePolicy":
            return {}
        if method == "Runtime.evaluate":
            return {"result": {"type": "object", "value": []}}
        if method == "HeadlessExperimental.beginFrame":
            begin_frame_count["n"] += 1
            return {
                "hasDamage": False,
                "screenshotData": base64.b64encode(png_bytes).decode("ascii"),
            }
        raise AssertionError(f"unexpected CDP method {method}")

    cdp.send = AsyncMock(side_effect=_send)

    viewport = Rect(x=0, y=0, w=200, h=100)
    with caplog.at_level("DEBUG", logger="worldsim.phase_4.pvpo_capture"):
        capture = await atomic_capture_with_visibility(cdp, viewport_rect=viewport)

    assert capture.has_damage is False
    assert begin_frame_count["n"] == 1
    assert any("hasDamage=False" in rec.message for rec in caplog.records)


def test_save_step_artifacts_writes_both_files(tmp_path: Path):
    png = _make_png({(0, 0, 10, 10): (100, 150, 200)})
    capture = StepCapture(
        screenshot_png=png,
        visibility_vec=[
            _entry(0, "a", layout_visible=True, live=(10, 10, 5, 5), ref=(100, 10, 5, 5)),
        ],
        has_damage=True,
        clip=Rect(x=0, y=0, w=200, h=100),
    )
    save_step_artifacts(tmp_path, step_idx=3, capture=capture)

    png_out = tmp_path / "screenshots" / "step_3.png"
    pvpo_out = tmp_path / "pvpo" / "step_3.json"
    assert png_out.exists()
    assert png_out.read_bytes() == png

    pvpo_json = json.loads(pvpo_out.read_text())
    assert pvpo_json["step_idx"] == 3
    assert pvpo_json["has_damage"] is True
    assert pvpo_json["clip"] == {"x": 0, "y": 0, "w": 200, "h": 100}
    assert pvpo_json["visibility_vec"][0]["char"] == "a"


def test_pvpo_query_js_contract_markers_present():
    assert "@ts-check" in PVPO_QUERY_JS
    assert "data-worldsim-payload" in PVPO_QUERY_JS
    assert "worldsim-payload-reference" in PVPO_QUERY_JS
    assert "data-worldsim-ref-idx" in PVPO_QUERY_JS
    assert "checkVisibility" in PVPO_QUERY_JS
    assert "elementFromPoint" in PVPO_QUERY_JS
    # Range-based per-char rects (ligature safety, handoff §3.4).
    assert "createRange" in PVPO_QUERY_JS
