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

import asyncio
import base64
import json
import logging
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worldsim.atomic_io import write_json_atomic
from worldsim.phase_4.pvpo_beginframe import BeginFrameCoordinator
from worldsim.phase_4.pvpo_cdp import normalize_cdp_session

logger = logging.getLogger(__name__)


_PVPO_QUERY_JS_TEMPLATE: str = (Path(__file__).parent / "pvpo_query.js").read_text(encoding="utf-8")
_PAYLOAD_SUBSTITUTION_TOKEN = "__WORLDSIM_WITNESSES_JSON__"
_SCROLL_TO_MATCH_SUBSTITUTION_TOKEN = "__WORLDSIM_SCROLL_TO_MATCH__"
_PVPO_CAPTURE_SUMMARY_FILENAME = "capture_summary.json"
_BEGINFRAME_EMPTY_SCREENSHOT_RETRIES = 1


def _consume_late_cdp_result(task: asyncio.Future[Any]) -> None:
    with suppress(asyncio.CancelledError, Exception):
        task.result()


async def _await_cdp_deadline(
    request: Any,
    *,
    timeout_s: float,
    method: str,
) -> dict[str, Any]:
    task = asyncio.ensure_future(request)
    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout=timeout_s)
    except TimeoutError as exc:
        task.add_done_callback(_consume_late_cdp_result)
        raise TimeoutError(f"PVPO CDP {method} timed out after {timeout_s:.2f}s") from exc
    except asyncio.CancelledError:
        if not task.done():
            task.add_done_callback(_consume_late_cdp_result)
        raise


def _normalize_witness_specs(
    payload_text: str,
    witness_texts: list[str | dict[str, Any]] | None,
) -> list[dict[str, str]]:
    if witness_texts is None:
        values: list[str | dict[str, Any]] = (
            [payload_text] if isinstance(payload_text, str) else []
        )
    else:
        values = [w for w in witness_texts if _witness_text_from_value(w)]
        if not values and isinstance(payload_text, str) and payload_text:
            values = [payload_text]

    witnesses: list[dict[str, str]] = []
    for idx, value in enumerate(values):
        text = _witness_text_from_value(value)
        if not text:
            continue
        if isinstance(value, dict):
            witness_id = value.get("id")
            kind = value.get("kind")
        else:
            witness_id = None
            kind = None
        spec = {
            "id": witness_id
            if isinstance(witness_id, str) and witness_id
            else f"witness:{idx}",
            "text": text,
        }
        if isinstance(kind, str) and kind:
            spec["kind"] = kind
        witnesses.append(spec)
    return witnesses


def _witness_text_from_value(value: str | dict[str, Any]) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str):
            return text
    return ""


def build_pvpo_query_js(
    payload_text: str,
    *,
    witness_texts: list[str | dict[str, Any]] | None = None,
    scroll_to_match: bool = False,
) -> str:
    """Return the PVPO per-step JS with attack witnesses inlined as JSON.

    The query uses normalized substring matching against every text node in
    the DOM to locate an attack-specific witness.  ``payload_text`` is kept as
    a legacy fallback; callers should pass ``witness_texts`` derived from
    Phase 2's required tokens / framing / concealment witnesses.

    When ``witness_texts`` is an empty list (all witnesses filtered by
    :mod:`worldsim.phase_4.payload_witnesses`), we must still fall back to
    ``payload_text``.  Otherwise ``[]`` is not ``None`` and the substituted
    JSON is ``[]``, the JS returns ``matchFound: false`` forever, and
    encounter detection reports ``max_coverage=0`` (spurious ``gate1_invalid``).
    """
    witnesses = _normalize_witness_specs(payload_text, witness_texts)
    return _PVPO_QUERY_JS_TEMPLATE.replace(
        _PAYLOAD_SUBSTITUTION_TOKEN, json.dumps(witnesses)
    ).replace(_SCROLL_TO_MATCH_SUBSTITUTION_TOKEN, "true" if scroll_to_match else "false")


# Back-compat: the old constant returned a non-parameterized JS blob. Anything
# that still imports it now gets a query with no witnesses (no
# match → empty result). New callers should use ``build_pvpo_query_js``.
PVPO_QUERY_JS: str = build_pvpo_query_js("")


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
    page_url: str | None = None
    issue_class: str | None = None
    issue_message: str | None = None
    match_found: bool = False
    match_offset: int = -1
    matched_witness_id: str | None = None
    matched_witness_text: str | None = None


async def atomic_capture_with_visibility(
    cdp_session: Any,
    *,
    viewport_rect: Rect,
    payload_text: str = "",
    witness_texts: list[str | dict[str, Any]] | None = None,
    scroll_to_match: bool = False,
    capturing: asyncio.Event | None = None,
    cdp_timeout_s: float | None = None,
) -> StepCapture:
    """Run the virtual-time-paused visibility query + ``beginFrame`` screenshot.

    Args:
        cdp_session: CDP session object. Supports either Playwright's
            ``send(method, params)`` surface or Browser-Use 0.12.6's
            ``cdp_client.send.<Domain>.<method>(..., session_id=...)`` surface.
        viewport_rect: the visual viewport in page coordinates. The
            screenshot is clipped to this rect; post-composite capture
            means anything outside the viewport would not be in the frame
            even if the clip were larger.
        payload_text: legacy fallback witness. Prefer ``witness_texts``.
        witness_texts: ordered attack-specific witnesses. The JS query
            normalized-substring matches these against text nodes in the DOM
            to locate the rendered payload — no HTML wrapper or data-attribute
            is required.
        capturing: optional :class:`asyncio.Event` shared with the Browser-Use
            PVPO compatibility layer. It marks atomic capture as active so any
            scoped beginFrame ticker can stand down, and it carries the
            endpoint coordinator used to serialize screenshot beginFrames. Set
            on entry, cleared in ``finally`` regardless of success.
        cdp_timeout_s: optional timeout applied to each CDP command. The
            Phase 4 Browser-Use callback sets this so saturated hosts degrade
            PVPO capture instead of spending the whole 180s Browser-Use step.

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
    query_js = build_pvpo_query_js(
        payload_text,
        witness_texts=witness_texts,
        scroll_to_match=scroll_to_match,
    )
    cdp = normalize_cdp_session(cdp_session)

    async def _send(method: str, params: dict[str, Any]) -> dict[str, Any]:
        request = cdp.send(method, params)
        if cdp_timeout_s is None:
            return await request
        return await _await_cdp_deadline(
            request,
            timeout_s=cdp_timeout_s,
            method=method,
        )

    # The Browser-Use PVPO compatibility context attaches a
    # BeginFrameCoordinator to the capturing Event so atomic capture,
    # navigation ticks, and Browser-Use screenshot fallbacks share one
    # timeout-safe beginFrame serialization path. Older callers that passed
    # only a bare Event/lock still work; they just don't get pending-frame
    # draining.
    beginframe_controller = (
        getattr(capturing, "beginframe_controller", None) if capturing is not None else None
    )
    beginframe_lock = getattr(capturing, "beginframe_lock", None) if capturing is not None else None
    if capturing is not None:
        capturing.set()
    virtual_time_paused = False
    try:
        if isinstance(beginframe_controller, BeginFrameCoordinator):
            await beginframe_controller.drain_prior(label="pre-atomic-capture")
        await _send("Emulation.setVirtualTimePolicy", {"policy": "pause"})
        virtual_time_paused = True
        raw = await _send(
            "Runtime.evaluate",
            {"expression": query_js, "returnByValue": True, "awaitPromise": True},
        )
        (
            visibility_vec,
            background_color,
            issue_class,
            issue_message,
            match_found,
            match_offset,
            matched_witness_id,
            matched_witness_text,
            page_url,
        ) = _unwrap_runtime_evaluate(raw)
        if issue_class is not None:
            logger.warning(
                "pvpo: Runtime.evaluate returned malformed visibility payload (%s); "
                "defaulting missing fields and continuing",
                issue_message or issue_class,
            )

        begin_frame_params = {
            "screenshot": {
                "format": "png",
                "quality": 100,
                "clip": viewport_rect.as_cdp_clip(),
            },
        }

        async def _capture_frame() -> tuple[dict[str, Any], bytes]:
            if isinstance(beginframe_controller, BeginFrameCoordinator):
                frame = await beginframe_controller.send(
                    cdp,
                    begin_frame_params,
                    timeout_s=cdp_timeout_s,
                    label="atomic-capture",
                )
            else:
                frame = await _send("HeadlessExperimental.beginFrame", begin_frame_params)
            png_b64 = frame.get("screenshotData") or ""
            return frame, base64.b64decode(png_b64) if png_b64 else b""

        frame: dict[str, Any] = {}
        screenshot_png = b""
        attempts = _BEGINFRAME_EMPTY_SCREENSHOT_RETRIES + 1
        if isinstance(beginframe_controller, BeginFrameCoordinator):
            for attempt in range(attempts):
                frame, screenshot_png = await _capture_frame()
                if screenshot_png:
                    break
                if attempt < attempts - 1:
                    await asyncio.sleep(0)
        elif beginframe_lock is not None:
            async with beginframe_lock:
                for attempt in range(attempts):
                    frame, screenshot_png = await _capture_frame()
                    if screenshot_png:
                        break
                    if attempt < attempts - 1:
                        await asyncio.sleep(0)
        else:
            for attempt in range(attempts):
                frame, screenshot_png = await _capture_frame()
                if screenshot_png:
                    break
                if attempt < attempts - 1:
                    await asyncio.sleep(0)

        has_damage = bool(frame.get("hasDamage", True))
        if not has_damage:
            logger.debug(
                "pvpo beginFrame hasDamage=False; trusting prior frame pixels "
                "(read-only visibility query should not dirty the compositor)"
            )
        if not screenshot_png:
            issue_class = "begin_frame_empty_screenshot"
            issue_message = (
                "HeadlessExperimental.beginFrame returned empty screenshotData "
                f"after {attempts} attempt(s)"
            )
    finally:
        if virtual_time_paused:
            try:
                await _send("Emulation.setVirtualTimePolicy", {"policy": "advance"})
            except Exception as exc:
                logger.warning("pvpo: failed to resume virtual time after capture: %s", exc)
        if capturing is not None:
            capturing.clear()

    return StepCapture(
        screenshot_png=screenshot_png,
        visibility_vec=visibility_vec,
        background_color=background_color,
        has_damage=has_damage,
        clip=viewport_rect,
        page_url=page_url,
        issue_class=issue_class,
        issue_message=issue_message,
        match_found=match_found,
        match_offset=match_offset,
        matched_witness_id=matched_witness_id,
        matched_witness_text=matched_witness_text,
    )


_DEFAULT_BG: tuple[int, int, int] = (255, 255, 255)


def _unwrap_runtime_evaluate(
    raw: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    tuple[int, int, int],
    str | None,
    str | None,
    bool,
    int,
    str | None,
    str | None,
    str | None,
]:
    """Extract ``(entries, background_color)`` from a ``Runtime.evaluate`` result.

    The JS query returns ``{entries: [...], backgroundColor: {r, g, b}}``.
    Any departure from that shape falls back to ``([], _DEFAULT_BG)`` so
    the caller still gets a valid :class:`StepCapture` rather than an
    exception mid-capture.
    """
    result = raw.get("result") or {}
    if result.get("type") != "object" or "value" not in result:
        return (
            [],
            _DEFAULT_BG,
            "runtime_evaluate_malformed",
            "missing result.value object",
            False,
            -1,
            None,
            None,
            None,
        )
    value = result["value"]
    if not isinstance(value, dict):
        return (
            [],
            _DEFAULT_BG,
            "runtime_evaluate_malformed",
            "result.value is not a dict",
            False,
            -1,
            None,
            None,
            None,
        )
    entries = value.get("entries") or []
    issue_message: str | None = None
    if not isinstance(entries, list):
        entries = []
        issue_message = "entries is not a list"
    bg = value.get("backgroundColor")
    if not isinstance(bg, dict):
        if issue_message is None:
            issue_message = "backgroundColor is not a dict"
        return (
            entries,
            _DEFAULT_BG,
            "runtime_evaluate_malformed",
            issue_message,
            bool(value.get("matchFound")),
            _match_offset(value.get("matchOffset")),
            _optional_str(value.get("matchedWitnessId")),
            _optional_str(value.get("matchedWitnessText")),
            _optional_str(value.get("pageUrl")),
        )
    try:
        bg_rgb = (int(bg.get("r", 255)), int(bg.get("g", 255)), int(bg.get("b", 255)))
    except (TypeError, ValueError):
        bg_rgb = _DEFAULT_BG
        if issue_message is None:
            issue_message = "backgroundColor channels are not integers"
    return (
        entries,
        bg_rgb,
        ("runtime_evaluate_malformed" if issue_message else None),
        issue_message,
        bool(value.get("matchFound")),
        _match_offset(value.get("matchOffset")),
        _optional_str(value.get("matchedWitnessId")),
        _optional_str(value.get("matchedWitnessText")),
        _optional_str(value.get("pageUrl")),
    )


def _match_offset(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def capture_summary_path(trajectory_dir: Path) -> Path:
    """Return the canonical per-task PVPO capture summary path."""
    return Path(trajectory_dir) / "pvpo" / _PVPO_CAPTURE_SUMMARY_FILENAME


def initial_capture_summary(*, payload_present: bool) -> dict[str, Any]:
    """Return the default PVPO capture summary structure."""
    return {
        "status": "initialized" if payload_present else "disabled_no_payload",
        "payload_present": bool(payload_present),
        "steps_seen": 0,
        "steps_captured": 0,
        "issue_steps": 0,
        "first_issue_class": None,
        "first_issue_step": None,
        "first_issue_message": None,
        "last_issue_class": None,
        "last_issue_step": None,
        "last_issue_message": None,
        "issue_counts": {},
    }


def save_capture_summary(trajectory_dir: Path, summary: dict[str, Any]) -> None:
    """Persist the per-task PVPO capture summary."""
    write_json_atomic(
        capture_summary_path(trajectory_dir),
        summary,
        failpoint_base="pvpo.save_capture_summary",
    )


def load_capture_summary(trajectory_dir: Path) -> dict[str, Any] | None:
    """Load the per-task PVPO capture summary if it exists and is valid."""
    path = capture_summary_path(trajectory_dir)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("pvpo: could not read capture summary at %s: %s", path, exc)
        return None
    if not isinstance(payload, dict):
        logger.warning("pvpo: capture summary at %s is not a JSON object", path)
        return None
    return payload


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
                "page_url": capture.page_url,
                "issue_class": capture.issue_class,
                "issue_message": capture.issue_message,
                "match_found": capture.match_found,
                "match_offset": capture.match_offset,
                "matched_witness_id": capture.matched_witness_id,
                "matched_witness_text": capture.matched_witness_text,
                "screenshot_bytes": len(capture.screenshot_png),
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
