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

from worldsim.phase_4.pvpo_beginframe import BeginFrameCoordinator, BeginFrameTimeout
from worldsim.phase_4.pvpo_capture import (
    Rect,
    StepCapture,
    _await_cdp_deadline,
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
                        "matchFound": True,
                        "matchOffset": 12,
                        "matchedWitnessId": "required_url:0",
                        "matchedWitnessText": "attacker.invalid/a",
                        "pageUrl": "http://gitlab.test/issues/1",
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
    assert capture.match_found is True
    assert capture.match_offset == 12
    assert capture.matched_witness_id == "required_url:0"
    assert capture.page_url == "http://gitlab.test/issues/1"

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
async def test_atomic_capture_sets_and_clears_capturing_event_on_success():
    png = _png_bytes()
    cdp = AsyncMock()
    capturing = asyncio.Event()
    seen_set_while_running: list[bool] = []

    async def _send(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        seen_set_while_running.append(capturing.is_set())
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
            return {"hasDamage": True, "screenshotData": base64.b64encode(png).decode("ascii")}
        raise AssertionError(method)

    cdp.send = AsyncMock(side_effect=_send)
    assert capturing.is_set() is False
    await atomic_capture_with_visibility(
        cdp, viewport_rect=Rect(0, 0, 100, 100), capturing=capturing
    )
    # All CDP calls saw the gate held set during the capture.
    assert all(seen_set_while_running)
    # And it's cleared after the capture returns.
    assert capturing.is_set() is False


@pytest.mark.asyncio
async def test_atomic_capture_uses_beginframe_controller_from_capturing_event():
    png = _png_bytes()
    cdp = AsyncMock()
    coordinator = BeginFrameCoordinator()
    capturing = asyncio.Event()
    capturing.beginframe_controller = coordinator  # type: ignore[attr-defined]
    capturing.beginframe_lock = coordinator.lock  # type: ignore[attr-defined]

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
            return {
                "hasDamage": True,
                "screenshotData": base64.b64encode(png).decode("ascii"),
            }
        raise AssertionError(method)

    cdp.send = AsyncMock(side_effect=_send)

    capture = await atomic_capture_with_visibility(
        cdp,
        viewport_rect=Rect(0, 0, 100, 100),
        capturing=capturing,
        cdp_timeout_s=0.5,
    )

    assert capture.screenshot_png == png
    assert coordinator.timeout_count == 0
    assert [call.args[0] for call in cdp.send.call_args_list].count(
        "HeadlessExperimental.beginFrame"
    ) == 1
    assert capturing.is_set() is False


@pytest.mark.asyncio
async def test_atomic_capture_drains_prior_frame_before_virtual_time_pause():
    png = _png_bytes()
    cdp = AsyncMock()
    coordinator = BeginFrameCoordinator()
    release_prior = asyncio.Event()
    capturing = asyncio.Event()
    capturing.beginframe_controller = coordinator  # type: ignore[attr-defined]
    call_order: list[str] = []

    async def _send(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        call_order.append(method)
        if method == "Emulation.setVirtualTimePolicy":
            assert coordinator.stats().get("beginframe_inflight_pending") is None
            return {}
        if method == "Runtime.evaluate":
            return {
                "result": {
                    "type": "object",
                    "value": {"entries": [], "backgroundColor": {"r": 255, "g": 255, "b": 255}},
                }
            }
        if method == "HeadlessExperimental.beginFrame":
            if not release_prior.is_set():
                await release_prior.wait()
                return {"hasDamage": True}
            return {
                "hasDamage": True,
                "screenshotData": base64.b64encode(png).decode("ascii"),
            }
        raise AssertionError(method)

    cdp.send = AsyncMock(side_effect=_send)

    with pytest.raises(BeginFrameTimeout):
        await coordinator.send(cdp, {}, timeout_s=0.01, label="navigation-tick")
    release_prior.set()

    capture = await atomic_capture_with_visibility(
        cdp,
        viewport_rect=Rect(0, 0, 100, 100),
        capturing=capturing,
        cdp_timeout_s=0.5,
    )

    assert capture.screenshot_png == png
    assert coordinator.prior_drain_count == 1
    assert call_order.index("Emulation.setVirtualTimePolicy") > call_order.index(
        "HeadlessExperimental.beginFrame"
    )
    assert capturing.is_set() is False


@pytest.mark.asyncio
async def test_atomic_capture_does_not_pause_virtual_time_when_prior_frame_stays_pending(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("WORLDSIM_PVPO_BEGINFRAME_DRAIN_TIMEOUT_S", "0.01")
    cdp = AsyncMock()
    coordinator = BeginFrameCoordinator()
    prior_release = asyncio.Event()
    capturing = asyncio.Event()
    capturing.beginframe_controller = coordinator  # type: ignore[attr-defined]

    async def _send(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if method == "HeadlessExperimental.beginFrame":
            await prior_release.wait()
            return {"hasDamage": True}
        if method == "Emulation.setVirtualTimePolicy":
            raise AssertionError("virtual time must not pause before prior frame drains")
        raise AssertionError(method)

    cdp.send = AsyncMock(side_effect=_send)

    with pytest.raises(BeginFrameTimeout):
        await coordinator.send(cdp, {}, timeout_s=0.01, label="navigation-tick")
    with pytest.raises(BeginFrameTimeout, match="pre-atomic-capture"):
        await atomic_capture_with_visibility(
            cdp,
            viewport_rect=Rect(0, 0, 100, 100),
            capturing=capturing,
            cdp_timeout_s=0.5,
        )

    assert coordinator.prior_drain_timeout_count == 1
    assert capturing.is_set() is False
    prior_release.set()


@pytest.mark.asyncio
async def test_atomic_capture_clears_capturing_event_on_error():
    cdp = AsyncMock()
    capturing = asyncio.Event()

    async def _send(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if method == "Emulation.setVirtualTimePolicy":
            return {}
        if method == "Runtime.evaluate":
            raise RuntimeError("synthetic CDP failure")
        raise AssertionError(method)

    cdp.send = AsyncMock(side_effect=_send)
    with pytest.raises(RuntimeError, match="synthetic CDP failure"):
        await atomic_capture_with_visibility(
            cdp, viewport_rect=Rect(0, 0, 100, 100), capturing=capturing
        )
    # Gate cleared even on exception so the pump resumes.
    assert capturing.is_set() is False


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


@pytest.mark.asyncio
async def test_atomic_capture_retries_empty_beginframe_screenshot():
    png = _png_bytes()
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
            if begin_frame_count["n"] == 1:
                return {"hasDamage": True}
            return {
                "hasDamage": True,
                "screenshotData": base64.b64encode(png).decode("ascii"),
            }
        raise AssertionError(f"unexpected CDP method {method}")

    cdp.send = AsyncMock(side_effect=_send)

    capture = await atomic_capture_with_visibility(cdp, viewport_rect=Rect(0, 0, 200, 100))

    assert begin_frame_count["n"] == 2
    assert capture.screenshot_png == png
    assert capture.issue_class is None


@pytest.mark.asyncio
async def test_atomic_capture_marks_persistent_empty_beginframe_screenshot():
    cdp = AsyncMock()

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
            return {"hasDamage": True}
        raise AssertionError(f"unexpected CDP method {method}")

    cdp.send = AsyncMock(side_effect=_send)

    capture = await atomic_capture_with_visibility(cdp, viewport_rect=Rect(0, 0, 200, 100))

    assert capture.screenshot_png == b""
    assert capture.issue_class == "begin_frame_empty_screenshot"
    assert "empty screenshotData" in (capture.issue_message or "")


@pytest.mark.asyncio
async def test_atomic_capture_accepts_browser_use_cdp_session_surface():
    png = _png_bytes()

    class _Domain:
        def __init__(self, fn):
            self._fn = fn

        def __getattr__(self, name: str):
            async def _call(*, params=None, session_id=None):
                return await self._fn(name, params or {}, session_id)

            return _call

    class _SendRoot:
        def __init__(self, fn):
            self.Emulation = _Domain(fn)
            self.Runtime = _Domain(fn)
            self.HeadlessExperimental = _Domain(fn)

    class _Client:
        def __init__(self, fn):
            self.send = _SendRoot(fn)

    class _BrowserUseCDPSession:
        def __init__(self):
            self.session_id = "session-1"
            self.cdp_client = _Client(self._handle)
            self.calls: list[tuple[str, dict[str, Any], str | None]] = []

        async def _handle(self, method_name: str, params: dict[str, Any], session_id: str | None):
            self.calls.append((method_name, params, session_id))
            if method_name == "setVirtualTimePolicy":
                return {}
            if method_name == "evaluate":
                return {
                    "result": {
                        "type": "object",
                        "value": {
                            "entries": [_entry(0, "a", layout_visible=True, live=(10, 10, 5, 5))],
                            "backgroundColor": {"r": 250, "g": 251, "b": 252},
                        },
                    }
                }
            if method_name == "beginFrame":
                return {
                    "hasDamage": True,
                    "screenshotData": base64.b64encode(png).decode("ascii"),
                }
            raise AssertionError(f"unexpected Browser-Use CDP command {method_name}")

    cdp = _BrowserUseCDPSession()

    capture = await atomic_capture_with_visibility(cdp, viewport_rect=Rect(0, 0, 200, 100))

    assert capture.screenshot_png == png
    assert capture.background_color == (250, 251, 252)
    assert capture.visibility_vec[0]["char"] == "a"
    assert [call[0] for call in cdp.calls] == [
        "setVirtualTimePolicy",
        "evaluate",
        "beginFrame",
        "setVirtualTimePolicy",
    ]


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
