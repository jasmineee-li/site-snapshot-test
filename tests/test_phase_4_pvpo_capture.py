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
    Rect,
    StepCapture,
    _unwrap_runtime_evaluate,
    atomic_capture_with_visibility,
    build_pvpo_query_js,
    save_step_artifacts,
)


def _png_bytes() -> bytes:
    img = Image.new("RGB", (100, 50), (255, 255, 255))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _entry(
    i: int,
    char: str,
    *,
    layout_visible: bool,
    live: tuple[int, int, int, int] | None = None,
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
    }


def test_rect_as_cdp_clip_shape():
    rect = Rect(x=10, y=20, w=300, h=200)
    clip = rect.as_cdp_clip()
    assert clip == {"x": 10.0, "y": 20.0, "width": 300.0, "height": 200.0, "scale": 1.0}


def test_unwrap_runtime_evaluate_happy_path():
    raw = {
        "result": {
            "type": "object",
            "value": {
                "entries": [_entry(0, "a", layout_visible=True, live=(0, 0, 5, 5))],
                "backgroundColor": {"r": 240, "g": 241, "b": 242},
            },
        }
    }
    entries, bg = _unwrap_runtime_evaluate(raw)
    assert len(entries) == 1
    assert entries[0]["char"] == "a"
    assert bg == (240, 241, 242)


def test_unwrap_runtime_evaluate_missing_value_falls_back_to_defaults():
    assert _unwrap_runtime_evaluate({"result": {"type": "object"}}) == ([], (255, 255, 255))
    assert _unwrap_runtime_evaluate({}) == ([], (255, 255, 255))
    assert _unwrap_runtime_evaluate({"result": {"type": "object", "value": "not-a-dict"}}) == (
        [],
        (255, 255, 255),
    )


def test_unwrap_runtime_evaluate_malformed_bg_falls_back():
    raw = {
        "result": {
            "type": "object",
            "value": {"entries": [], "backgroundColor": "not-a-dict"},
        }
    }
    entries, bg = _unwrap_runtime_evaluate(raw)
    assert entries == []
    assert bg == (255, 255, 255)


@pytest.mark.asyncio
async def test_atomic_capture_sequences_cdp_calls_and_extracts_bg():
    png = _png_bytes()
    cdp = AsyncMock()

    async def _send(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if method == "Emulation.setVirtualTimePolicy":
            return {}
        if method == "Runtime.evaluate":
            return {
                "result": {
                    "type": "object",
                    "value": {
                        "entries": [_entry(0, "a", layout_visible=True, live=(10, 10, 5, 5))],
                        "backgroundColor": {"r": 250, "g": 251, "b": 252},
                    },
                }
            }
        if method == "HeadlessExperimental.beginFrame":
            return {
                "hasDamage": True,
                "screenshotData": base64.b64encode(png).decode("ascii"),
            }
        raise AssertionError(f"unexpected CDP method {method}")

    cdp.send = AsyncMock(side_effect=_send)

    capture = await atomic_capture_with_visibility(cdp, viewport_rect=Rect(0, 0, 200, 100))

    assert capture.has_damage is True
    assert capture.screenshot_png == png
    assert capture.background_color == (250, 251, 252)
    assert capture.visibility_vec[0]["char"] == "a"

    methods = [c.args[0] for c in cdp.send.call_args_list]
    assert methods == [
        "Emulation.setVirtualTimePolicy",
        "Runtime.evaluate",
        "HeadlessExperimental.beginFrame",
        "Emulation.setVirtualTimePolicy",
    ]
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

    policies = [
        c.args[1]["policy"]
        for c in cdp.send.call_args_list
        if c.args[0] == "Emulation.setVirtualTimePolicy"
    ]
    assert policies == ["pause", "advance"]


@pytest.mark.asyncio
async def test_atomic_capture_has_damage_false_does_not_retry(caplog):
    cdp = AsyncMock()
    begin_frame_count = {"n": 0}

    async def _send(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if method == "Emulation.setVirtualTimePolicy":
            return {}
        if method == "Runtime.evaluate":
            return {
                "result": {
                    "type": "object",
                    "value": {"entries": [], "backgroundColor": {"r": 255, "g": 255, "b": 255}},
                }
            }
        if method == "HeadlessExperimental.beginFrame":
            begin_frame_count["n"] += 1
            return {
                "hasDamage": False,
                "screenshotData": base64.b64encode(_png_bytes()).decode("ascii"),
            }
        raise AssertionError(f"unexpected CDP method {method}")

    cdp.send = AsyncMock(side_effect=_send)

    with caplog.at_level("DEBUG", logger="worldsim.phase_4.pvpo_capture"):
        capture = await atomic_capture_with_visibility(cdp, viewport_rect=Rect(0, 0, 200, 100))

    assert capture.has_damage is False
    assert begin_frame_count["n"] == 1
    assert any("hasDamage=False" in rec.message for rec in caplog.records)


def test_save_step_artifacts_writes_both_files_with_bg(tmp_path: Path):
    capture = StepCapture(
        screenshot_png=_png_bytes(),
        visibility_vec=[_entry(0, "a", layout_visible=True, live=(10, 10, 5, 5))],
        background_color=(128, 129, 130),
        has_damage=True,
        clip=Rect(x=0, y=0, w=200, h=100),
    )
    save_step_artifacts(tmp_path, step_idx=3, capture=capture)

    png_out = tmp_path / "screenshots" / "step_3.png"
    pvpo_out = tmp_path / "pvpo" / "step_3.json"
    assert png_out.exists()
    assert png_out.read_bytes() == capture.screenshot_png

    pvpo_json = json.loads(pvpo_out.read_text())
    assert pvpo_json["step_idx"] == 3
    assert pvpo_json["has_damage"] is True
    assert pvpo_json["clip"] == {"x": 0, "y": 0, "w": 200, "h": 100}
    assert pvpo_json["background_color"] == [128, 129, 130]
    assert pvpo_json["visibility_vec"][0]["char"] == "a"


def test_pvpo_query_js_contract_markers_present():
    js = build_pvpo_query_js("some payload text")
    assert "@ts-check" in js
    assert "entries" in js
    assert "backgroundColor" in js
    assert "checkVisibility" in js
    assert "elementFromPoint" in js
    assert "createRange" in js
    # Content-match anchor strategy replaced the attribute-based anchor.
    assert "createTreeWalker" in js
    assert "matchFound" in js
    # Reference container was removed in the ink-occupancy cutover; the
    # attribute-anchor strategy was removed in the content-match cutover.
    # (The string ``data-worldsim-payload`` survives in a comment explaining
    # why that strategy is no longer used, so we don't assert-not on it.)
    assert "worldsim-payload-reference" not in js
    assert "data-worldsim-ref-idx" not in js
    assert "document.querySelector('[data-worldsim-payload" not in js
    # Payload is inlined JSON-encoded, not as a raw string (quoting-safe).
    assert "some payload text" in js
    assert "__WORLDSIM_PAYLOAD_JSON__" not in js


def test_build_pvpo_query_js_json_encodes_quotes_and_backslashes():
    tricky = 'has "quotes" and \\ backslash and <script>alert(1)</script>'
    js = build_pvpo_query_js(tricky)
    # The literal string must survive JSON encoding; " and \ escaped.
    assert '"has \\"quotes\\"' in js
    assert "\\\\ backslash" in js
    # The substitution token must be gone.
    assert "__WORLDSIM_PAYLOAD_JSON__" not in js


def test_build_pvpo_query_js_empty_payload_still_valid():
    js = build_pvpo_query_js("")
    # Empty payload → query short-circuits to emptyResult() inside the IIFE.
    # But the template must still parse (substitution applied cleanly).
    assert "__WORLDSIM_PAYLOAD_JSON__" not in js
    assert '""' in js
