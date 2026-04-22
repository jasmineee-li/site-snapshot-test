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
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Magento PDPs lazy-load the review block via /review/product/listAjax/.
# Arming a wait_for_response on this fragment before page.goto lets us
# block until the AJAX completes; the static PDP HTML alone does not
# contain the review body.
_SHOPPING_LISTAJAX_FRAGMENT = "/review/product/listAjax/"
_SHOPPING_REVIEW_SELECTOR = ".review-items, .no-reviews, #customer-reviews"

# GitLab issue / MR pages load the discussion thread lazily via an
# AJAX call to ``/.../discussions.json`` after DOMContentLoaded fires.
# Without waiting for that response, ``page.text_content("body")``
# captures only the SPA shell (~8kb) and misses every note body the
# seed just wrote — which showed up on the first live WASP run as 98
# of 148 plans failing with render_unverified on the gitlab issue
# page despite the POST /api/v4/.../notes call having succeeded.
_GITLAB_DISCUSSIONS_FRAGMENT = "/discussions.json"
_GITLAB_NOTE_SELECTOR = ".notes .note, .discussion-notes .note, ul.notes-list .note"

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


def render_signature(seed: dict[str, Any], metadata: dict[str, Any] | None = None) -> str | None:
    """Extract a unique substring expected to appear in the rendered DOM.

    Prefers the seed nickname (ASCII, unique by construction in the
    adversarial dataset, and renders in a stable DOM location across
    Magento product reviews / Reddit posts / GitLab notes). Falls back
    in priority order to detail / body / description / note / bio /
    content (first 40 chars), then title.

    Every base field name is also tried with the ``_template`` and
    ``_text`` suffixes the editor-method binding contract uses for
    free-text template args (e.g. ``description_template``,
    ``bio_text``), so feasibility verification works on any plan that
    targets a method whose body lives on a templated arg.

    Returns None when the editor call carries no signature-bearing
    field at all — caller treats that as render_unverified with a
    clear "no signature available" message.
    """
    if not isinstance(seed, dict):
        return None
    editor_calls = seed.get("editor_calls")
    if not isinstance(editor_calls, list) or not editor_calls:
        return None
    call_records = [
        (
            f"{call.get('site')}.{call.get('method')}",
            call.get("args"),
        )
        for call in editor_calls
        if isinstance(call, dict) and isinstance(call.get("args"), dict)
    ]
    if not call_records:
        return None
    preferred_methods: set[str] = set()
    provenance = metadata.get("read_surface_provenance") if isinstance(metadata, dict) else None
    if isinstance(provenance, dict):
        methods = provenance.get("editor_method")
        if isinstance(methods, str) and methods.strip():
            preferred_methods.add(methods.strip())
        elif isinstance(methods, list):
            preferred_methods.update(
                str(method).strip()
                for method in methods
                if isinstance(method, str) and method.strip()
            )
    arg_sets = [
        args
        for method, args in call_records
        if method in preferred_methods and isinstance(args, dict)
    ]
    if not arg_sets:
        arg_sets = [args for _, args in call_records if isinstance(args, dict)]
    # In multi-call seeds the last editor call is typically the one that
    # produces the user-visible note/comment while earlier calls create
    # parent resources or helper setup rows. Prefer later calls so a setup
    # title/description cannot overshadow the actually rendered payload.
    arg_sets = list(reversed(arg_sets))

    def _first_nonempty(fields: tuple[str, ...]) -> str | None:
        for args in arg_sets:
            for base in fields:
                for variant in (base, f"{base}_template", f"{base}_text"):
                    raw = args.get(variant)
                    if isinstance(raw, str) and raw.strip():
                        return raw.strip()
        return None

    nickname = _first_nonempty(("nickname",))
    if nickname is not None:
        return nickname
    body = _first_nonempty(("detail", "body", "description", "note", "bio", "content"))
    if body is not None:
        # Take only the first line so markdown on subsequent lines
        # (``![img](...)``) doesn't bleed into the signature — the
        # image renders as an ``<img>`` tag on the issue page so the
        # literal ``![Alt]`` bytes never appear in body text. First
        # line is almost always plain prose ("Platform Administrator
        # Notice:") that survives rendering.
        first_line = body.split("\n", 1)[0].strip()
        if len(first_line) >= 8:
            return first_line[:40]
        return body[:40]
    title = _first_nonempty(("title", "name"))
    if title is not None:
        return title.split("\n", 1)[0].strip()[:40] or title[:40]
    # Fallback: the LLM sometimes invents arg keys (e.g., reddit's
    # ``reply_to_submission_{submission_id}[comment]``) that don't
    # match the binding-spec vocabulary. Pick the longest string
    # value across all args as the signature — by construction this
    # is the rendered body text (tokens like ``{benign_forum_name}``
    # are short, IDs are shorter still). Skip values that look like
    # ``{benign_*}`` tokens entirely so we don't pick up the token
    # reference itself.
    longest: str | None = None
    longest_len = 0
    for args in arg_sets:
        for value in args.values():
            if not isinstance(value, str):
                continue
            stripped = value.strip()
            if not stripped or stripped.startswith("{benign_"):
                continue
            if len(stripped) > longest_len:
                longest = stripped
                longest_len = len(stripped)
    if longest is not None and longest_len >= 8:
        first_line = longest.split("\n", 1)[0].strip()
        if len(first_line) >= 8:
            return first_line[:40]
        return longest[:40]
    return None


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def _resolve_url(url: str, site_url: str) -> str:
    """Coerce any URL to resolve against the instance's site_url.

    - Path-only (``/foo``) → ``{site_url}/foo``.
    - Fully-qualified URL whose host+port matches site_url → unchanged.
    - Fully-qualified URL on a DIFFERENT host+port → rewritten to
      site_url's host+port. This happens when an editor captured the
      platform API's ``web_url`` and the API reflects the internal
      external_url (e.g. ``http://localhost:8023/byteblaze/foo`` on the
      gitlab image). The ProxyingHTTPAdapter's Location-rewrite handles
      this for ``requests`` calls, but Playwright (used here) has no
      such adapter — rewriting in ``_resolve_url`` keeps the render
      check robust across cross-host replay + api-emitted URLs.
    """
    if not url:
        return url
    if url.startswith("/") and site_url:
        return site_url.rstrip("/") + url
    if site_url and (url.startswith("http://") or url.startswith("https://")):
        from urllib.parse import urlsplit, urlunsplit

        src = urlsplit(url)
        dst = urlsplit(site_url)
        if not dst.netloc or src.netloc == dst.netloc:
            return url
        scheme = dst.scheme or src.scheme or "http"
        path = src.path or "/"
        return urlunsplit((scheme, dst.netloc, path, src.query, src.fragment))
    return url


def _with_cache_buster(url: str) -> str:
    """Append a millisecond timestamp to bust stale-response caches.

    Originally motivated by Magento's full-page cache (which serves a
    stale PDP for 60-300s after a status_id flip). GitLab's Rails cache
    and Postmill's CDN-style response headers exhibit the same class of
    staleness when a newly-seeded note / comment is read back through
    the UI before the cache entry expires. Every render-check and
    reachability fetch carries a unique query string; this is cheaper
    than whatever flush-cache endpoint each site exposes and avoids
    perturbing other in-flight tests.

    Applied to UI page navigations only. API endpoints (``/api/v4/...``,
    ``/sv/*.json``) are tolerated by Playwright's GET but should not
    normally be fetched by this module — the render-check and
    reachability probe both navigate HTML pages.
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
    storage_state_path: str | None = None,
) -> RenderOutcome:
    """Open a fresh context, try each URL until the signature appears.

    Browser handle is owned by the caller (one per Phase 2c run); this
    function creates and tears down a single context (~200ms each, vs
    1.5-3s for a full launch). Caller is responsible for capping
    concurrent contexts at ~8 to avoid Playwright's eviction edge cases.

    ``storage_state_path`` threads the benign user's Phase-0d-bootstrapped
    cookies into Playwright so private projects / authed-only pages are
    reachable. When None (default), opens an anonymous context — fine
    for public content but breaks on private gitlab projects.
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
    context_kwargs: dict[str, Any] = {}
    if storage_state_path:
        path_obj = Path(storage_state_path)
        if path_obj.exists():
            context_kwargs["storage_state"] = str(path_obj)
        else:
            logger.warning(
                "phase 2c render check: storage_state %s not found; falling back to anonymous context",
                storage_state_path,
            )
    context = await browser.new_context(**context_kwargs)
    try:
        page = await context.new_page()
        for raw_url in urls:
            target = _with_cache_buster(_resolve_url(raw_url, site_url))
            seen.append(target)
            try:
                listajax_task = None
                discussions_task = None
                if site_name == "shopping" and "/catalog/product/view/" in target:
                    listajax_task = page.wait_for_response(
                        lambda r: _SHOPPING_LISTAJAX_FRAGMENT in r.url and r.status == 200,
                        timeout=nav_timeout_ms,
                    )
                if site_name == "gitlab" and (
                    "/-/issues/" in target or "/-/merge_requests/" in target
                ):
                    discussions_task = page.wait_for_response(
                        lambda r: _GITLAB_DISCUSSIONS_FRAGMENT in r.url and r.status == 200,
                        timeout=nav_timeout_ms,
                    )
                await page.goto(target, timeout=nav_timeout_ms, wait_until="domcontentloaded")
                if listajax_task is not None:
                    try:
                        await listajax_task
                    except Exception:
                        pass
                if discussions_task is not None:
                    try:
                        await discussions_task
                    except Exception:
                        pass
                if site_name == "shopping":
                    try:
                        await page.wait_for_selector(
                            _SHOPPING_REVIEW_SELECTOR, timeout=selector_timeout_ms
                        )
                    except Exception:
                        pass
                if site_name == "gitlab" and (
                    "/-/issues/" in target or "/-/merge_requests/" in target
                ):
                    try:
                        await page.wait_for_selector(
                            _GITLAB_NOTE_SELECTOR, timeout=selector_timeout_ms
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
