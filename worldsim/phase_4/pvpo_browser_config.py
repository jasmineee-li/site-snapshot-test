"""Browser configuration for the Paint-Verified Payload Oracle.

This module provides the three one-time setup primitives PVPO needs:

1. ``PVPO_LAUNCH_FLAGS`` — the Chromium command-line flags chrome-headless-shell
   must be started with so ``HeadlessExperimental.beginFrame`` works and the
   screenshot accurately reflects the committed paint frame.
2. ``inject_animation_killer`` — page-level CSS + CDP animation pause that
   eliminates the residual compositor-thread-animation race after
   ``setVirtualTimePolicy("pause")``.
3. ``inject_reference_container`` — inserts a hidden per-character reference
   span array as a sibling of the live injection anchor. Both sides render
   through the same Blink+Skia paint path in the same committed frame, so
   host-side pixel comparison is byte-identical with zero tolerance.

The capture step itself (virtual-time pause, beginFrame, per-char visibility
query, pixel compare) lives in ``pvpo_capture.py``.

See ``docs/handoffs/codex-handoff-paint-verified-oracle.md`` §3 for the full
design rationale and references to Chromium source confirming each decision.
"""

from __future__ import annotations

from typing import Any

# The binary itself is ``chrome-headless-shell`` (from Chrome for Testing, see
# ``worldsim/docker/chrome-headless-shell.Dockerfile``). Flag audit: these four
# are required and active in 2026 Chromium; three flags that appeared in
# earlier proposals (``--enable-surface-synchronization``,
# ``--disable-threaded-scrolling``, ``--disable-threaded-animation``) were
# dropped after verifying they are no-ops or removed in current Chromium.
PVPO_LAUNCH_FLAGS: tuple[str, ...] = (
    "--enable-begin-frame-control",
    "--run-all-compositor-stages-before-draw",
    "--disable-checker-imaging",
)


# Reference container DOM identifiers. The capture step looks these up by
# selector so the Phase-2 text-fill stage does not need to know about them.
REFERENCE_CONTAINER_ID = "worldsim-payload-reference"
REFERENCE_SPAN_ATTR = "data-worldsim-ref-idx"
PAYLOAD_ANCHOR_ATTR = "data-worldsim-payload"


_ANIMATION_KILLER_CSS = """
* {
  animation: none !important;
  animation-duration: 0s !important;
  animation-iteration-count: 1 !important;
  transition: none !important;
  transition-duration: 0s !important;
}
"""


_INJECT_ANIMATION_KILLER_JS = f"""
(() => {{
  if (document.getElementById('worldsim-animation-killer')) return;
  const style = document.createElement('style');
  style.id = 'worldsim-animation-killer';
  style.textContent = {_ANIMATION_KILLER_CSS!r};
  (document.head || document.documentElement).appendChild(style);
}})();
"""


async def inject_animation_killer(page: Any, cdp_session: Any | None = None) -> None:
    """Idempotently install the animation-killer stylesheet.

    The stylesheet forces all CSS ``animation`` and ``transition`` properties
    to zero duration, which prevents off-main-thread compositor-thread
    animations from advancing between the visibility query and the
    ``beginFrame`` capture. Combined with
    ``Emulation.setVirtualTimePolicy("pause")`` (called per-step by
    :func:`worldsim.phase_4.pvpo_capture.atomic_capture_with_visibility`),
    this closes the theoretical race flagged by Chromium's own
    compositor-animation tests (marked "Flaky on all platforms").

    Args:
        page: the Playwright ``Page`` object whose DOM will receive the style
            tag.
        cdp_session: accepted for signature compatibility. Not used — an
            earlier draft of this helper called
            ``Animation.setPaused({"paused": true})`` but that CDP method
            requires a pre-collected ``animations: [id, ...]`` array per
            the current DevTools Protocol and is not a global toggle. There
            is no CDP primitive that pauses all animations regardless of
            source; the CSS route above is the canonical mechanism.

    The WebArena surfaces we target are static HTML forms with no CSS or
    compositor animations, so the residual race reduces to effectively zero
    in our setting.
    """
    await page.evaluate(_INJECT_ANIMATION_KILLER_JS)
    _ = cdp_session  # retained for signature compatibility; see docstring.


_REFERENCE_CONTAINER_OFFSCREEN_Y = 50000


def _build_reference_container_js(
    payload_text: str,
    anchor_selector: str,
) -> str:
    """Build the JS expression that inserts the hidden per-char reference container.

    Placement strategy: the reference container is inserted as a direct sibling
    of the element matching ``anchor_selector`` (the Phase-2b-tagged injection
    anchor). Because it shares the same parent, all cascading CSS (font
    family/weight/size, letter-spacing, writing-mode, etc.) resolves identically
    to the live payload — no style copying required.

    Visual concealment: the container is ``position: absolute`` with a large
    positive ``top`` offset (``_REFERENCE_CONTAINER_OFFSCREEN_Y`` pixels below
    the viewport). This keeps it in the layout + paint tree — so the per-char
    spans actually get rasterized through the same Blink+HarfBuzz+Skia
    pipeline as the live payload, which is what makes zero-tolerance pixel
    comparison sound — while placing it outside any reasonable user scroll
    target. PVPO capture extends ``beginFrame``'s ``clip`` rect to include
    the reference region (see ``pvpo_capture.py::_clip_union``) so both the
    live viewport and the reference crop land in the same committed frame.

    Concealment options explicitly rejected:
      - ``visibility: hidden`` / ``display: none`` — both skip paint entirely
        per CSS spec, so the reference spans would not contribute pixels and
        every byte-equal compare would spuriously return False.
      - ``opacity: 0`` — paints but produces fully transparent pixels, again
        guaranteeing a mismatch against the opaque live text.
      - ``left: -99999px`` — equivalent mechanism but negative offsets are
        more likely to be clipped by negative-coordinate-hostile render
        stacks; a positive ``top`` offset is the conventional choice.
    """
    import json

    payload_chars = list(payload_text)
    style = (
        f"position: absolute; left: 0; top: {_REFERENCE_CONTAINER_OFFSCREEN_Y}px; "
        "pointer-events: none; user-select: none; white-space: pre;"
    )
    return f"""
(() => {{
  const anchor = document.querySelector({json.dumps(anchor_selector)});
  if (!anchor || !anchor.parentElement) {{
    return {{ ok: false, reason: "anchor not found: " + {json.dumps(anchor_selector)} }};
  }}
  if (document.getElementById({json.dumps(REFERENCE_CONTAINER_ID)})) {{
    return {{ ok: true, reason: "already present" }};
  }}
  const container = document.createElement('span');
  container.id = {json.dumps(REFERENCE_CONTAINER_ID)};
  container.style.cssText = {json.dumps(style)};
  container.setAttribute('aria-hidden', 'true');
  const chars = {json.dumps(payload_chars)};
  for (let i = 0; i < chars.length; i++) {{
    const span = document.createElement('span');
    span.setAttribute({json.dumps(REFERENCE_SPAN_ATTR)}, String(i));
    span.textContent = chars[i];
    container.appendChild(span);
  }}
  anchor.parentElement.insertBefore(container, anchor.nextSibling);
  return {{ ok: true }};
}})();
"""


async def inject_reference_container(
    page: Any,
    payload_text: str,
    anchor_selector: str,
) -> None:
    """Idempotently insert the hidden per-character reference container.

    Called once per payload injection (not per step). Place as sibling of the
    Phase-2-tagged anchor element so CSS inheritance is identical to the live
    payload. Subsequent calls with the same container already present are no-ops.

    Args:
        page: the Playwright ``Page`` object.
        payload_text: the exact bytes of the seeded payload, byte-for-byte.
            One ``<span data-worldsim-ref-idx="N">`` is created per character
            (including whitespace — the capture step filters whitespace out of
            the coverage score, not here).
        anchor_selector: a CSS selector locating the element that wraps the
            injected payload on the page. Produced by Phase 2b text-fill by
            setting the ``data-worldsim-payload`` attribute on the injected
            element. The reference container is inserted as its next sibling.

    Raises:
        RuntimeError: if the anchor selector does not match any element on
            the page.
    """
    result = await page.evaluate(_build_reference_container_js(payload_text, anchor_selector))
    if not isinstance(result, dict) or not result.get("ok"):
        reason = result.get("reason") if isinstance(result, dict) else "unknown"
        raise RuntimeError(f"inject_reference_container failed: {reason}")
