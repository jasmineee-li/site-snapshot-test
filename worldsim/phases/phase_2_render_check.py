"""Phase 2c post-seed render verification.

After ``apply_data_seed_async`` succeeds, fetch the editor's emitted
``read_surface_urls`` in a real browser and confirm a unique signature
of the seeded payload appears in the rendered DOM. Without this check,
Phase 2c's ``feasibility.status="verified"`` only proves the seed write
succeeded — not that the payload renders for a Phase 4 agent. The
2026-04-21 Magento review-pending bug shipped 174 tasks under exactly
that lie; this module is the architectural backstop.

The check is a hard gate by default. To opt out (development without
Playwright installed, or unit tests that mock the seed flow), set
``WORLDSIM_PHASE_2C_SKIP_RENDER_CHECK=1`` — this disables verification
entirely and downgrades the ``verified`` stamp to "API write succeeded
only," which is what the pipeline did before this module existed. Use
it sparingly; production runs should leave it unset.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Magento PDPs lazy-load the review block via /review/product/listAjax/.
# Arming a wait_for_response on this fragment before page.goto lets us
# block until the AJAX completes; the static PDP HTML alone does not
# contain the review body.
_SHOPPING_LISTAJAX_FRAGMENT = "/review/product/listAjax/"
_SHOPPING_REVIEW_SELECTOR = ".review-items, .no-reviews, #customer-reviews"

# Markers in Playwright exception strings that mean the host is unreachable
# rather than the payload merely missing. These route to host_unreachable
# (loud, explicit infeasibility) instead of render_unverified.
_HOST_UNREACHABLE_MARKERS: tuple[str, ...] = (
    "TimeoutError",
    "ECONNREFUSED",
    "net::ERR_CONNECTION_REFUSED",
    "net::ERR_NAME_NOT_RESOLVED",
    "net::ERR_CONNECTION_TIMED_OUT",
    "net::ERR_ADDRESS_UNREACHABLE",
)


@dataclass(frozen=True)
class RenderOutcome:
    """Result of a render-check pass.

    ``kind`` is one of ``render_unverified``, ``host_unreachable``,
    ``render_check_error`` (for unexpected Playwright crashes), or empty
    string on success. ``urls_tried`` and ``per_url_errors`` populate
    the ``feasibility.errors[].render_evidence`` field for diagnosability.
    """

    ok: bool
    kind: str
    detail: str
    urls_tried: list[str]
    per_url_errors: dict[str, str]
    matched_url: str | None = None
    matched_signature: str | None = None
    matched_snippet: str | None = None

    @classmethod
    def passed(cls, *, url: str, signature: str, snippet: str) -> RenderOutcome:
        return cls(
            ok=True,
            kind="",
            detail=f"signature {signature!r} present in {url}",
            urls_tried=[url],
            per_url_errors={},
            matched_url=url,
            matched_signature=signature,
            matched_snippet=snippet[:240],
        )

    @classmethod
    def failed(
        cls,
        *,
        kind: str,
        detail: str,
        urls_tried: list[str],
        per_url_errors: dict[str, str],
    ) -> RenderOutcome:
        return cls(
            ok=False,
            kind=kind,
            detail=detail,
            urls_tried=urls_tried,
            per_url_errors=per_url_errors,
        )

    def evidence(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "ok": self.ok,
            "urls_tried": list(self.urls_tried),
        }
        if self.per_url_errors:
            out["per_url_errors"] = dict(self.per_url_errors)
        if self.ok:
            out["matched_url"] = self.matched_url
            out["matched_signature"] = self.matched_signature
            if self.matched_snippet:
                out["matched_snippet"] = self.matched_snippet
        else:
            out["kind"] = self.kind
        return out


def render_signature(seed: dict[str, Any]) -> str | None:
    """Extract a unique substring expected to appear in the rendered DOM.

    Prefers the seed nickname (ASCII, unique by construction in the
    adversarial dataset, and renders in a stable DOM location across
    Magento product reviews / Reddit posts / GitLab notes). Falls back
    in priority order to detail / body / description / note (first 40
    chars), then title.

    Returns None when the editor call carries no signature-bearing
    field at all — caller treats that as render_unverified with a
    clear "no signature available" message.
    """
    if not isinstance(seed, dict):
        return None
    editor_calls = seed.get("editor_calls")
    if not isinstance(editor_calls, list) or not editor_calls:
        return None
    first = editor_calls[0]
    if not isinstance(first, dict):
        return None
    args = first.get("args")
    if not isinstance(args, dict):
        return None
    nickname = args.get("nickname")
    if isinstance(nickname, str) and nickname.strip():
        return nickname.strip()
    for body_field in ("detail", "body", "description", "note", "content"):
        value = args.get(body_field)
        if isinstance(value, str) and value.strip():
            return value.strip()[:40]
    title = args.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()[:40]
    return None


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def _resolve_url(url: str, site_url: str) -> str:
    if url.startswith("/") and site_url:
        return site_url.rstrip("/") + url
    return url


def _with_cache_buster(url: str) -> str:
    """Append a millisecond timestamp to bust Magento's full-page cache.

    The FPC may serve a stale PDP for 60-300s after a status_id flip, so
    every render-check fetch carries a unique query string. Cheaper than
    /rest/V1/cache/flush and doesn't perturb other in-flight tests.
    """
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}_={int(time.time() * 1000)}"


def _classify_failure(
    *, urls_tried: list[str], per_url_errors: dict[str, str], signature: str
) -> RenderOutcome:
    """Decide between host_unreachable and render_unverified.

    Every URL with a network-flavored error → host_unreachable. Any URL
    that loaded successfully but the signature was absent → render_unverified
    (the bug class this module exists to catch).
    """
    if per_url_errors and all(
        any(marker in err for marker in _HOST_UNREACHABLE_MARKERS)
        for err in per_url_errors.values()
    ):
        return RenderOutcome.failed(
            kind="host_unreachable",
            detail=(
                f"all {len(urls_tried)} read_surface_urls timed out or refused "
                "connection during phase 2c render check"
            ),
            urls_tried=urls_tried,
            per_url_errors=per_url_errors,
        )
    return RenderOutcome.failed(
        kind="render_unverified",
        detail=(
            f"signature {signature!r} not found in any of {len(urls_tried)} "
            "read_surface_urls — seed wrote to the platform but the payload "
            "is not rendered. Common causes: pending-approval moderation, "
            "draft state, or the editor's read_surface_urls do not point at "
            "the canonical render surface."
        ),
        urls_tried=urls_tried,
        per_url_errors=per_url_errors,
    )


async def verify_seed_renders(
    *,
    browser: Any,  # playwright.async_api.Browser
    urls: list[str],
    site_name: str,
    site_url: str,
    signature: str | None,
    nav_timeout_ms: int = 15000,
    selector_timeout_ms: int = 5000,
) -> RenderOutcome:
    """Open a fresh context, try each URL until the signature appears.

    Browser handle is owned by the caller (one per Phase 2c run); this
    function creates and tears down a single context (~200ms each, vs
    1.5-3s for a full launch). Caller is responsible for capping
    concurrent contexts at ~8 to avoid Playwright's eviction edge cases.
    """
    if not urls:
        return RenderOutcome.failed(
            kind="render_unverified",
            detail="editor emitted no read_surface_urls — cannot verify rendering",
            urls_tried=[],
            per_url_errors={},
        )
    if not signature:
        return RenderOutcome.failed(
            kind="render_unverified",
            detail=(
                "seed has no extractable render signature "
                "(nickname/detail/body/description/note/content/title all absent)"
            ),
            urls_tried=[],
            per_url_errors={},
        )

    needle = _normalize(signature)
    seen: list[str] = []
    errors: dict[str, str] = {}
    context = await browser.new_context()
    try:
        page = await context.new_page()
        for raw_url in urls:
            target = _with_cache_buster(_resolve_url(raw_url, site_url))
            seen.append(target)
            try:
                listajax_task = None
                if site_name == "shopping" and "/catalog/product/view/" in target:
                    listajax_task = page.wait_for_response(
                        lambda r: _SHOPPING_LISTAJAX_FRAGMENT in r.url and r.status == 200,
                        timeout=nav_timeout_ms,
                    )
                await page.goto(target, timeout=nav_timeout_ms, wait_until="domcontentloaded")
                if listajax_task is not None:
                    try:
                        await listajax_task
                    except Exception:
                        # AJAX may have completed during goto; non-fatal.
                        pass
                if site_name == "shopping":
                    try:
                        await page.wait_for_selector(
                            _SHOPPING_REVIEW_SELECTOR, timeout=selector_timeout_ms
                        )
                    except Exception:
                        pass
                body_text = await page.text_content("body") or ""
                normalized = _normalize(body_text)
                if needle in normalized:
                    pos = normalized.find(needle)
                    raw_pos = 0
                    if pos >= 0 and pos < len(body_text):
                        raw_pos = pos
                    snippet = body_text[max(0, raw_pos - 40) : raw_pos + len(signature) + 40]
                    return RenderOutcome.passed(url=target, signature=signature, snippet=snippet)
                errors[target] = f"signature_absent (body_len={len(body_text)})"
            except Exception as exc:
                msg = f"{exc.__class__.__name__}: {exc}"
                errors[target] = msg
                logger.debug("phase 2c render check error on %s: %s", target, msg)
        return _classify_failure(urls_tried=seen, per_url_errors=errors, signature=signature)
    finally:
        try:
            await context.close()
        except Exception:
            logger.exception("phase 2c render check failed to close context")
