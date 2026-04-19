"""Browser configuration for the Paint-Verified Payload Oracle.

This module provides the two one-time setup primitives PVPO needs after the
ink-occupancy cutover:

1. ``PVPO_LAUNCH_FLAGS`` — the Chromium command-line flags
   ``chrome-headless-shell`` must be started with so
   ``HeadlessExperimental.beginFrame`` works and the screenshot accurately
   reflects the committed paint frame.
2. ``inject_animation_killer`` — page-level CSS that eliminates the
   residual compositor-thread-animation race after
   ``setVirtualTimePolicy("pause")``.

The capture step itself (virtual-time pause, beginFrame, per-char
visibility + bg-color query) lives in ``pvpo_capture.py``; host-side
ink-occupancy verification lives in ``ink_occupancy.py``.

See ``docs/handoffs/codex-handoff-paint-verified-oracle.md`` §3 for the
original design. The hidden reference container (handoff §3.2b) and the
byte-equal pixel-compare oracle (handoff §3.5) were removed in the
post-handoff review that settled on ink-occupancy as the practical
verification primitive.
"""

from __future__ import annotations

from typing import Any

# Chromium launch flags. The binary is ``chrome-headless-shell`` from Chrome
# for Testing (see ``worldsim/docker/chrome-headless-shell.Dockerfile``).
# Flag audit: these three are required and active in 2026 Chromium; three
# flags that appeared in earlier proposals
# (``--enable-surface-synchronization``, ``--disable-threaded-scrolling``,
# ``--disable-threaded-animation``) were dropped after verifying they are
# no-ops or removed in current Chromium.
PVPO_LAUNCH_FLAGS: tuple[str, ...] = (
    "--enable-begin-frame-control",
    "--run-all-compositor-stages-before-draw",
    "--disable-checker-imaging",
)


# Public DOM-attribute name used by the JS visibility query to locate the
# seeded payload on the delivery page. ``phase_2_text_fill.py`` emits
# ``<span data-worldsim-payload="1">{rendered_payload}</span>`` at
# materialize time.
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

    The stylesheet forces all CSS ``animation`` and ``transition``
    properties to zero duration, which prevents off-main-thread
    compositor-thread animations from advancing between the visibility
    query and the ``beginFrame`` capture. Combined with
    ``Emulation.setVirtualTimePolicy("pause")`` (called per-step by
    :func:`worldsim.phase_4.pvpo_capture.atomic_capture_with_visibility`),
    this closes the theoretical race flagged by Chromium's own
    compositor-animation tests (marked "Flaky on all platforms").

    Args:
        page: the Playwright ``Page`` object whose DOM will receive the
            style tag.
        cdp_session: accepted for signature compatibility. Not used — an
            earlier draft called ``Animation.setPaused({"paused": true})``
            but that CDP method requires a pre-collected
            ``animations: [id, ...]`` array per the current DevTools
            Protocol and is not a global toggle. The CSS route above is
            the canonical mechanism.

    The WebArena surfaces we target are static HTML forms with no CSS or
    compositor animations, so the residual race reduces to effectively
    zero in our setting.
    """
    await page.evaluate(_INJECT_ANIMATION_KILLER_JS)
    _ = cdp_session  # retained for signature compatibility; see docstring.
