from __future__ import annotations

import asyncio
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
    _await_cdp_deadline,
    _unwrap_runtime_evaluate,
    build_pvpo_query_js,
    save_step_artifacts,
    surface_capture_with_stability,
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


def test_pvpo_query_js_uses_multinode_linearization():
    """Finding 4 regression guard: the JS query must linearize text nodes
    before substring-matching the payload, not search each node
    individually.

    The pre-fix implementation looped per-text-node calling
    ``content.indexOf(payloadText)`` and missed payloads spanning multiple
    nodes (auto-linkified URLs, inline ``<em>`` etc.). The fix builds a
    flat ``corpus`` + ``charMap`` and runs a single ``corpus.indexOf(...)``.
    A regression here would silently lower max_coverage in production —
    quietly miss-routing trajectories to placement-fix — so we pin the
    structural change at the JS-source level.
    """
    js = build_pvpo_query_js("anything")
    # Linearization markers — normalized corpus + source lookup table.
    assert "const normalizedCharMap" in js
    assert "let normalizedCorpus" in js
    # Single match against the linearized corpus, not per-node loop.
    assert "findWitnessMatches(normalizedCorpus, witnesses)" in js
    assert "hasVisibleNonSpaceEntry(result.entries)" in js
    # A hidden URL lure may appear before visible framing prose; the query
    # must consider later witnesses before settling for an invisible match.
    assert "if (matches.length > 0) return matches" not in js
    # No remaining substitution token.
    assert "__WORLDSIM_WITNESSES_JSON__" not in js
    # Per-char ancestor uses the source text node's parent so inline-
    # wrapped runs (e.g. an `<a>` inside a `<p>`) are individually
    # visibility-checked rather than inheriting the first match's parent.
    assert "sourceNode.parentElement" in js
    assert "__WORLDSIM_SCROLL_TO_MATCH__" not in js
    assert "const scrollToMatch = false;" in js


def test_pvpo_query_js_matches_conservative_rendered_markdown_variants():
    js = build_pvpo_query_js(
        "ignored",
        witness_texts=[
            {
                "id": "attack_action:0",
                "kind": "attack_action",
                "text": "Please report `no-link` accordingly for this thread.",
            },
            {
                "id": "required_url:0",
                "kind": "required_url",
                "text": "https://attacker.invalid/deadbeef",
            },
        ],
    )

    assert '"kind": "attack_action"' in js
    assert "function witnessTextVariants" in js
    assert "renderInlineCodeText" in js
    assert "DOM_VARIANT_MIN_CHARS = 20" in js
    assert 'witness.id.startsWith("required_url:")' in js
    assert "Please report `no-link` accordingly for this thread." in js
    assert "Please report no-link accordingly for this thread." not in js


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
    (
        entries,
        bg,
        issue_class,
        issue_message,
        match_found,
        match_offset,
        matched_id,
        matched_text,
        page_url,
    ) = _unwrap_runtime_evaluate(raw)
    assert len(entries) == 1
    assert entries[0]["char"] == "a"
    assert bg == (240, 241, 242)
    assert issue_class is None
    assert issue_message is None
    assert match_found is False
    assert match_offset == -1
    assert matched_id is None
    assert matched_text is None
    assert page_url is None


def test_unwrap_runtime_evaluate_missing_value_falls_back_to_defaults():
    assert _unwrap_runtime_evaluate({"result": {"type": "object"}}) == (
        [],
        (255, 255, 255),
        "runtime_evaluate_malformed",
        "missing result.value object",
        False,
        -1,
        None,
        None,
        None,
    )
    assert _unwrap_runtime_evaluate({}) == (
        [],
        (255, 255, 255),
        "runtime_evaluate_malformed",
        "missing result.value object",
        False,
        -1,
        None,
        None,
        None,
    )
    assert _unwrap_runtime_evaluate({"result": {"type": "object", "value": "not-a-dict"}}) == (
        [],
        (255, 255, 255),
        "runtime_evaluate_malformed",
        "result.value is not a dict",
        False,
        -1,
        None,
        None,
        None,
    )


def test_unwrap_runtime_evaluate_malformed_bg_falls_back():
    raw = {
        "result": {
            "type": "object",
            "value": {"entries": [], "backgroundColor": "not-a-dict"},
        }
    }
    entries, bg, issue_class, issue_message, *_ = _unwrap_runtime_evaluate(raw)
    assert entries == []
    assert bg == (255, 255, 255)
    assert issue_class == "runtime_evaluate_malformed"
    assert issue_message == "backgroundColor is not a dict"


def test_unwrap_runtime_evaluate_non_list_entries_marks_issue():
    raw = {
        "result": {
            "type": "object",
            "value": {
                "entries": "not-a-list",
                "backgroundColor": {"r": 1, "g": 2, "b": 3},
            },
        }
    }
    entries, bg, issue_class, issue_message, *_ = _unwrap_runtime_evaluate(raw)
    assert entries == []
    assert bg == (1, 2, 3)
    assert issue_class == "runtime_evaluate_malformed"
    assert issue_message == "entries is not a list"


@pytest.mark.asyncio
async def test_pvpo_capture_cdp_deadline_preserves_late_response_future():
    loop = asyncio.get_running_loop()
    future: asyncio.Future[dict[str, Any]] = loop.create_future()

    with pytest.raises(TimeoutError):
        await _await_cdp_deadline(
            future,
            timeout_s=0.01,
            method="Runtime.evaluate",
        )

    assert not future.cancelled()
    future.set_result({"result": {}})
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_surface_capture_uses_page_screenshot_and_stable_probe():
    png = _png_bytes()
    cdp = AsyncMock()

    query_value = {
        "entries": [_entry(0, "a", layout_visible=True, live=(10, 10, 5, 5))],
        "backgroundColor": {"r": 250, "g": 251, "b": 252},
        "matchFound": True,
        "matchOffset": 12,
        "matchedWitnessId": "required_url:0",
        "matchedWitnessText": "attacker.invalid/a",
        "pageUrl": "http://gitlab.test/issues/1",
    }

    async def _send(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if method == "Runtime.evaluate":
            return {"result": {"type": "object", "value": query_value}}
        if method == "Page.captureScreenshot":
            assert params == {
                "format": "png",
                "fromSurface": True,
                "captureBeyondViewport": False,
                "clip": {"x": 0.0, "y": 0.0, "width": 200.0, "height": 100.0, "scale": 1.0},
            }
            return {"data": base64.b64encode(png).decode("ascii")}
        raise AssertionError(f"unexpected CDP method {method}")

    cdp.send = AsyncMock(side_effect=_send)

    capture = await surface_capture_with_stability(
        cdp,
        viewport_rect=Rect(0, 0, 200, 100),
    )

    assert capture.screenshot_png == png
    assert capture.visibility_vec[0]["char"] == "a"
    assert capture.background_color == (250, 251, 252)
    assert capture.has_damage is True
    assert capture.match_found is True
    assert capture.matched_witness_id == "required_url:0"
    assert capture.issue_class is None
    assert [c.args[0] for c in cdp.send.call_args_list] == [
        "Runtime.evaluate",
        "Page.captureScreenshot",
        "Runtime.evaluate",
    ]


@pytest.mark.asyncio
async def test_surface_capture_fails_closed_when_probe_geometry_changes():
    png = _png_bytes()
    cdp = AsyncMock()
    calls = {"runtime": 0}

    def _query_value(x: int) -> dict[str, Any]:
        return {
            "entries": [_entry(0, "a", layout_visible=True, live=(x, 10, 5, 5))],
            "backgroundColor": {"r": 255, "g": 255, "b": 255},
            "matchFound": True,
            "matchOffset": 0,
            "matchedWitnessId": "witness:0",
            "matchedWitnessText": "attack witness",
            "pageUrl": "http://gitlab.test/issues/1",
        }

    async def _send(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if method == "Runtime.evaluate":
            calls["runtime"] += 1
            return {
                "result": {
                    "type": "object",
                    "value": _query_value(10 if calls["runtime"] == 1 else 11),
                }
            }
        if method == "Page.captureScreenshot":
            return {"data": base64.b64encode(png).decode("ascii")}
        raise AssertionError(method)

    cdp.send = AsyncMock(side_effect=_send)

    capture = await surface_capture_with_stability(
        cdp,
        viewport_rect=Rect(0, 0, 200, 100),
    )

    assert capture.screenshot_png == png
    assert capture.visibility_vec == []
    assert capture.match_found is False
    assert capture.match_offset == -1
    assert capture.issue_class == "surface_capture_unstable_rects"
    assert "geometry changed" in (capture.issue_message or "")


def test_save_step_artifacts_writes_both_files_with_bg(tmp_path: Path):
    capture = StepCapture(
        screenshot_png=_png_bytes(),
        visibility_vec=[_entry(0, "a", layout_visible=True, live=(10, 10, 5, 5))],
        background_color=(128, 129, 130),
        has_damage=True,
        clip=Rect(x=0, y=0, w=200, h=100),
        issue_class="runtime_evaluate_malformed",
        issue_message="entries is not a list",
        match_found=True,
        match_offset=4,
        matched_witness_id="witness:0",
        matched_witness_text="payload",
        page_url="http://gitlab.test/issues/1",
    )
    save_step_artifacts(tmp_path, step_idx=3, capture=capture)

    png_out = tmp_path / "screenshots" / "step_3.png"
    pvpo_out = tmp_path / "pvpo" / "step_3.json"
    assert png_out.exists()
    assert png_out.read_bytes() == capture.screenshot_png

    pvpo_json = json.loads(pvpo_out.read_text())
    assert pvpo_json["step_idx"] == 3
    assert pvpo_json["has_damage"] is True
    assert pvpo_json["issue_class"] == "runtime_evaluate_malformed"
    assert pvpo_json["issue_message"] == "entries is not a list"
    assert pvpo_json["match_found"] is True
    assert pvpo_json["match_offset"] == 4
    assert pvpo_json["matched_witness_id"] == "witness:0"
    assert pvpo_json["matched_witness_text"] == "payload"
    assert pvpo_json["screenshot_bytes"] == len(capture.screenshot_png)
    assert pvpo_json["page_url"] == "http://gitlab.test/issues/1"
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
    assert "matchedWitnessId" in js
    # Reference container was removed in the ink-occupancy cutover; the
    # attribute-anchor strategy was removed in the content-match cutover.
    # (The string ``data-worldsim-payload`` survives in a comment explaining
    # why that strategy is no longer used, so we don't assert-not on it.)
    assert "worldsim-payload-reference" not in js
    assert "data-worldsim-ref-idx" not in js
    assert "document.querySelector('[data-worldsim-payload" not in js
    # Payload is inlined JSON-encoded, not as a raw string (quoting-safe).
    assert "some payload text" in js
    assert "__WORLDSIM_WITNESSES_JSON__" not in js
    assert "__WORLDSIM_SCROLL_TO_MATCH__" not in js


def test_build_pvpo_query_js_json_encodes_quotes_and_backslashes():
    tricky = 'has "quotes" and \\ backslash and <script>alert(1)</script>'
    js = build_pvpo_query_js(tricky)
    # The literal string must survive JSON encoding; " and \ escaped.
    assert '"has \\"quotes\\"' in js
    assert "\\\\ backslash" in js
    # The substitution token must be gone.
    assert "__WORLDSIM_WITNESSES_JSON__" not in js


def test_build_pvpo_query_js_empty_payload_still_valid():
    js = build_pvpo_query_js("")
    # Empty payload → query short-circuits to emptyResult() inside the IIFE.
    # But the template must still parse (substitution applied cleanly).
    assert "__WORLDSIM_WITNESSES_JSON__" not in js
    assert "[]" in js


def test_build_pvpo_query_js_can_scroll_offscreen_match_before_capture():
    js = build_pvpo_query_js(
        "attack witness lower on the page",
        scroll_to_match=True,
    )
    assert "const scrollToMatch = true;" in js
    assert "scrollMatchIntoView(matches[0]);" in js
    assert "element.scrollIntoView" in js


def test_build_pvpo_query_js_empty_witness_list_falls_back_to_payload_text():
    """Curated witnesses can all be filtered; payload_text must still anchor PVPO."""
    js = build_pvpo_query_js(
        "WORLD-SIM FALLBACK PAYLOAD ANCHOR TEXT",
        witness_texts=[],
    )
    assert "__WORLDSIM_WITNESSES_JSON__" not in js
    assert "WORLD-SIM FALLBACK PAYLOAD ANCHOR TEXT" in js
    assert '"witness:0"' in js


def test_build_pvpo_query_js_preserves_witness_metadata_ids():
    js = build_pvpo_query_js(
        "fallback payload",
        witness_texts=[
            {
                "id": "required_url:0:required_tokens",
                "kind": "required_url",
                "text": "https://attacker.invalid/pvpo-witness-token",
            }
        ],
    )

    assert '"required_url:0:required_tokens"' in js
    assert '"kind": "required_url"' in js
    assert '"https://attacker.invalid/pvpo-witness-token"' in js
