"""Browser configuration for the Paint-Verified Payload Oracle.

This module provides page-level CSS that reduces the residual animation race
between the pre-probe, page-surface screenshot, and post-probe used by PVPO.
The capture step itself lives in ``pvpo_capture.py``; host-side ink-occupancy
verification lives in ``ink_occupancy.py``.

See ``docs/handoffs/codex-handoff-paint-verified-oracle.md`` §3 for the
original design. The hidden reference container (handoff §3.2b) and the
byte-equal pixel-compare oracle (handoff §3.5) were removed in the
post-handoff review that settled on ink-occupancy as the practical
verification primitive.
"""

from __future__ import annotations

from typing import Any

from warp_taskgen.phase_4.pvpo_cdp import runtime_evaluate

PVPO_LAUNCH_FLAGS: tuple[str, ...] = ()


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


_INJECT_ANIMATION_KILLER_FALLBACK_EVAL = f"""() => {{
  if (document.getElementById('worldsim-animation-killer')) return;
  const style = document.createElement('style');
  style.id = 'worldsim-animation-killer';
  style.textContent = {_ANIMATION_KILLER_CSS!r};
  (document.head || document.documentElement).appendChild(style);
}}"""


async def inject_animation_killer(page: Any, cdp_session: Any | None = None) -> None:
    """Idempotently install the animation-killer stylesheet.

    The stylesheet forces all CSS ``animation`` and ``transition`` properties
    to zero duration, which reduces page movement between the pre-probe,
    screenshot, and post-probe. If geometry still changes, page-surface-stable
    capture records degraded evidence and encounter detection fails closed.

    Args:
        page: page-like object used only when no ``cdp_session`` is
            available and the caller falls back to ``page.evaluate``.
        cdp_session: preferred transport. When present, the stylesheet is
            injected with direct CDP ``Runtime.evaluate`` so the Browser-Use
            path does not depend on ``Page.evaluate``'s custom calling
            convention.

    The WebArena surfaces we target are static HTML forms with no CSS or
    compositor animations, so the residual race reduces to effectively
    zero in our setting.
    """
    if cdp_session is not None:
        await runtime_evaluate(cdp_session, _INJECT_ANIMATION_KILLER_JS)
        return
    await page.evaluate(_INJECT_ANIMATION_KILLER_FALLBACK_EVAL)
