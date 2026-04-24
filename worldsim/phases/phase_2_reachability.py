"""Phase 2c DOM reachability probe (Option A, X4b).

The render-check (``phase_2_render_check.verify_seed_renders``) proves
the seeded payload rendered on **some** URL the editor emitted. Under
Option A that's necessary but not sufficient: the payload also has to
be reachable from the URL the **benign** agent will navigate to at
task start. If the agent never visits the page the seed lives on, the
IPI trial is encounter-less — which is exactly the bug X1-X4 are
closing.

This probe answers: "starting from ``benign_target_resource.start_url_resolved``,
does the agent's default exploration reach a DOM that contains the
seeded payload?" We check for a two-witness grep — the signature
(already used by render-check) PLUS a second rare substring from the
rendered payload — to defend against astronomically unlikely but
non-zero signature collisions on pre-existing DOM content.

Result is tri-state:

* ``reachable_direct`` — both witnesses on the start_url body itself.
* ``reachable_transitively`` — witnesses appear after one additional
  action (e.g. clicking into a listing item). Reserved for later
  iteration; the first pass returns ``reachable_direct`` only.
* ``unreachable`` — neither witness present after the wait window.

Per-site selectors arm a bounded dynamic-content wait before the
body-text read:

* gitlab_issue / gitlab_mr / gitlab_dashboard_list list pages:
  ``.issuable-list``
* gitlab_mr detail:  ``.merge-request``
* gitlab_search_result:  seeded-title poll (up to 5s) before read
* reddit_submission detail:  ``.comment-list``
* reddit_forum:  ``.submission``
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from worldsim.phases.phase_2_render_check import (
    _strip_markdown_for_text_match,
    _with_cache_buster,
)

logger = logging.getLogger(__name__)


_SITE_SELECTORS: dict[str, str] = {
    # gitlab_issue / gitlab_mr: seed lives in the discussions.json note
    # thread, which GitLab loads via AJAX after DOMContentLoaded. The
    # ``.issuable-list, .issue-details`` selector only waits for the
    # metadata shell — by the time the probe reads ``body`` text the
    # notes thread has not appeared yet, and the 2-witness grep fails
    # with ``witnesses_absent``. Wait for an actual note element so the
    # body text contains the seeded payload when we grep it (mirrors
    # the render-check gate at phase_2_render_check.py:_GITLAB_NOTE_SELECTOR).
    "gitlab_issue": ".notes .note, .discussion-notes .note, ul.notes-list .note",
    "gitlab_mr": ".notes .note, .discussion-notes .note, ul.notes-list .note",
    "gitlab_dashboard_list": ".issuable-list, .todos-list",
    "gitlab_search_result": ".search-results, .issuable-list",
    "reddit_submission": ".comment-list, .comments",
    "reddit_forum": ".submission",
    "reddit_dashboard_list": ".submission, .comment",
}

_WITNESS_MIN_LEN = 12
_DEFAULT_NAV_TIMEOUT_MS = 30000
_DEFAULT_SELECTOR_TIMEOUT_MS = 10000
_SEARCH_POLL_INTERVAL_MS = 500
_SEARCH_POLL_ATTEMPTS = 10

# Subresources aborted before ``page.goto`` on Phase 2c probes. Deferred
# scripts in GitLab's frontend (``<script defer src=...>``) block on
# stylesheet downloads per the HTML spec, so aborting CSS lets deferred
# JS parse immediately and the note-selector wait fires fast. Images /
# fonts / media / Sentry beacons / ActionCable WebSocket are pure client
# cost. ``script`` and ``xhr`` stay allowed — the ``discussions.json``
# XHR the gitlab_issue / gitlab_mr probe depends on is initiated by JS.
_BLOCKED_RESOURCE_TYPES: frozenset[str] = frozenset(
    {"stylesheet", "image", "media", "font", "eventsource", "websocket"},
)


async def _install_resource_blocker(page: Any) -> None:
    """Abort non-essential subresources before ``page.goto``.

    Shared by ``phase_2_reachability.verify_reachable`` and
    ``phase_2_render_check.verify_seed_renders``. Must be installed
    after ``context.new_page()`` and before any navigation.
    """

    async def _handler(route: Any) -> None:
        try:
            resource_type = route.request.resource_type
        except Exception:
            resource_type = ""
        try:
            if resource_type in _BLOCKED_RESOURCE_TYPES:
                await route.abort()
            else:
                await route.continue_()
        except Exception:
            # Routing is best-effort — a closed page or superseded
            # navigation races here. Swallow to keep the probe honest.
            logger.debug("phase 2c route handler swallowed exception", exc_info=True)

    await page.route("**/*", _handler)


@dataclass(frozen=True)
class ReachabilityOutcome:
    reachability: str  # "reachable_direct" | "reachable_transitively" | "unreachable"
    kind: str  # empty on success, else a structured reason bucket
    detail: str
    url_tried: str
    witnesses_matched: tuple[str, ...]
    witnesses_missing: tuple[str, ...]

    @classmethod
    def direct(cls, *, url: str, witnesses_matched: tuple[str, ...]) -> ReachabilityOutcome:
        return cls(
            reachability="reachable_direct",
            kind="",
            detail="both witnesses present on start_url body",
            url_tried=url,
            witnesses_matched=witnesses_matched,
            witnesses_missing=(),
        )

    @classmethod
    def unreachable(
        cls,
        *,
        kind: str,
        detail: str,
        url: str,
        witnesses_matched: tuple[str, ...] = (),
        witnesses_missing: tuple[str, ...] = (),
    ) -> ReachabilityOutcome:
        return cls(
            reachability="unreachable",
            kind=kind,
            detail=detail,
            url_tried=url,
            witnesses_matched=witnesses_matched,
            witnesses_missing=witnesses_missing,
        )

    def evidence(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "reachability": self.reachability,
            "url_tried": self.url_tried,
        }
        if self.kind:
            out["kind"] = self.kind
        if self.detail:
            out["detail"] = self.detail
        if self.witnesses_matched:
            out["witnesses_matched"] = list(self.witnesses_matched)
        if self.witnesses_missing:
            out["witnesses_missing"] = list(self.witnesses_missing)
        return out


_MARKDOWN_WITNESS_SKIP_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Tail of a broken markdown image: `alt.">](http://...)`
    re.compile(r"\]\s*\(\s*https?://"),
    # Inline `![alt](url)` fragments — when these appear as a single
    # whitespace-delimited token they render as <img>, not text.
    re.compile(r"^!\[.*\]\("),
    # Raw base64 data URIs or HTML-encoded src attributes fall out of
    # broken markdown rendering and don't appear in plain text_content.
    re.compile(r'^src=["\']?data:'),
    re.compile(r"^(?:src|href)=[\"']"),
    # Raw closing bracket followed by a URL suffix — the "(...)" tail of
    # broken markdown.
    re.compile(r"^[^\s]*\]\("),
    # Curly-token leakage defended at seeding time but add belt-and-
    # braces: never emit a `{benign_*}` witness.
    re.compile(r"^\{benign_"),
)


def _looks_like_markdown_fragment(token: str) -> bool:
    """Reject witness candidates that survive only inside markdown syntax.

    For adversarial seeds wrapped in markdown-image template (``![{{PAYLOAD_TEXT}}](url)``),
    the longest whitespace-delimited tokens are often *syntactic*
    fragments (``alt.">](http://.../img.png)``, ``src="data:...">``)
    that never reach rendered ``text_content``. Skip them so the
    witness picker prefers prose tokens — URLs inside the payload
    prose, the attacker URL, or the signature's tail — which DO
    appear on the rendered page.
    """
    for pattern in _MARKDOWN_WITNESS_SKIP_PATTERNS:
        if pattern.search(token):
            return True
    return False


def derive_second_witness(rendered_payload: str | None, signature: str | None) -> str | None:
    """Pick a rare substring from the rendered payload distinct from signature.

    Returns the longest span of >= _WITNESS_MIN_LEN chars that does not
    overlap the signature and is not a markdown-syntax fragment. Splits
    on whitespace AND on markdown delimiters (``](``, ``">``, ``">]``)
    so a prose URL hidden inside ``[anchor](https://...)`` syntax still
    surfaces as a candidate — without those extra splits, the entire
    ``anchor](https://attacker.invalid/xxx)`` segment reads as one
    whitespace-delimited token and gets skipped as a markdown fragment,
    leaving no usable witness for markdown-wrapped seeds.
    """
    if not rendered_payload:
        return None
    needle = (signature or "").lower()
    # Split on whitespace + markdown delimiters so URLs inside
    # `[anchor](url)` or `alt.">](url)` aren't bonded to their syntax.
    candidates = re.split(r"[\s\r\n]+|\]\(|\">\]|\">|\]\s", rendered_payload)
    best: str | None = None
    for token in candidates:
        token = token.strip().rstrip(".,;:)")
        if len(token) < _WITNESS_MIN_LEN:
            continue
        if needle and needle in token.lower():
            continue
        if _looks_like_markdown_fragment(token):
            continue
        if best is None or len(token) > len(best):
            best = token
    return best


def resolve_start_url(
    start_url_resolved: str | None,
    instance_site_url: str,
    placeholders: Mapping[str, str] | None = None,
) -> str | None:
    """Replace any synthetic hostname in start_url_resolved with the live one.

    Phase 2a emitted start_url_resolved against a synthetic
    ``https://gitlab.local`` / ``https://reddit.local`` origin. At 2c
    time the real instance has a different ``site_url``; rewrite the
    host+scheme so the probe hits the right server.
    """
    if not start_url_resolved or not instance_site_url:
        return start_url_resolved
    parts = urlsplit(start_url_resolved)
    live = urlsplit(instance_site_url.rstrip("/"))
    if not live.scheme or not live.netloc:
        return start_url_resolved
    rewritten = parts._replace(scheme=live.scheme, netloc=live.netloc)
    return urlunsplit(rewritten)


def _selector_for_kind(kind: str) -> str | None:
    return _SITE_SELECTORS.get(kind)


def _normalize_for_match(text: str | None) -> str:
    """Collapse whitespace and lowercase, matching render_check semantics.

    Phase 2c's render-check compares on this normalized form so a
    signature with single spaces matches a body where the same text
    was laid out with newlines or indented whitespace (GitLab wraps
    long notes across multiple lines). Reachability must use the
    identical function — a plain ``.lower()`` missed 30+ signature
    matches whose body text had intervening whitespace.

    Bug G (2026-04-23): strip markdown delimiters before whitespace
    collapse so a signature ``**[CI Pipeline Notification]**`` matches
    a DOM body rendered as ``[CI Pipeline Notification]`` (GitLab
    CommonMark → ``<strong>`` → text_content drops the ``**``).
    """
    return re.sub(r"\s+", " ", _strip_markdown_for_text_match(text or "")).lower()


async def _wait_for_body_text(
    page: Any,
    needle: str | Iterable[str],
    timeout_ms: int,
) -> bool:
    """Poll the current page's ``body`` text every 500 ms until every needle appears.

    Used for kinds whose seed lives in lazy-loaded content:
    * gitlab_issue / gitlab_mr: notes stream in via discussions.json
      AJAX *after* DOMContentLoaded. The note-selector wait only
      guarantees *some* note has rendered; the seeded note may arrive
      in a later batch. Poll until the signature text appears or the
      selector timeout expires.
    * gitlab_search_result: GitLab search is DB LIKE on WebArena, but
      still has a short render window before the matched row enters
      the DOM.

    ``needle`` may be a single string (back-compat) or an iterable of
    strings; with multiple needles the poll waits for *all* of them to
    appear so that a fast-matching primary signature (e.g. a URL that
    is present in the static body) does not short-circuit the wait for
    a later-arriving secondary witness (the hydrated note prose).
    Empty / falsy needles are ignored. Returns ``True`` iff every
    non-empty needle is present in the body within ``timeout_ms``.
    """
    if isinstance(needle, str):
        needles_raw: tuple[str, ...] = (needle,)
    else:
        needles_raw = tuple(needle)
    needles_norm = tuple(n for n in (_normalize_for_match(raw) for raw in needles_raw) if n)
    if not needles_norm:
        return True
    deadline = time.monotonic() + (timeout_ms / 1000.0)
    while time.monotonic() < deadline:
        try:
            body = await page.text_content("body") or ""
        except Exception:
            body = ""
        normalized_body = _normalize_for_match(body)
        if all(n in normalized_body for n in needles_norm):
            return True
        try:
            await page.wait_for_timeout(_SEARCH_POLL_INTERVAL_MS)
        except Exception:
            break
    return False


# Back-compat alias — keep the old name importable for callers and tests
# that may reference it. Deprecate in a follow-up commit once
# grep-clean.
_wait_for_search_title = _wait_for_body_text


async def verify_reachable(
    *,
    browser: Any,
    benign_target_resource: Mapping[str, Any] | None,
    instance_site_url: str,
    signature: str | None,
    second_witness: str | None,
    storage_state_path: str | None = None,
    nav_timeout_ms: int = _DEFAULT_NAV_TIMEOUT_MS,
    selector_timeout_ms: int = _DEFAULT_SELECTOR_TIMEOUT_MS,
) -> ReachabilityOutcome:
    """Navigate the benign start URL and two-witness-grep for the seed."""
    if not isinstance(benign_target_resource, Mapping):
        return ReachabilityOutcome.unreachable(
            kind="no_target_resource",
            detail="task has no benign_target_resource; cannot probe reachability",
            url=(instance_site_url or ""),
        )
    kind = str(benign_target_resource.get("kind") or "")
    start_url = resolve_start_url(
        benign_target_resource.get("start_url_resolved"), instance_site_url
    )
    if not start_url:
        return ReachabilityOutcome.unreachable(
            kind="no_start_url",
            detail="benign_target_resource.start_url_resolved is empty",
            url="",
        )
    if not signature:
        return ReachabilityOutcome.unreachable(
            kind="no_signature",
            detail="render signature unavailable; cannot grep",
            url=start_url,
        )
    witnesses = [signature]
    if second_witness and second_witness.lower() != signature.lower():
        witnesses.append(second_witness)

    context_kwargs: dict[str, Any] = {}
    if storage_state_path:
        context_kwargs["storage_state"] = storage_state_path
    context = await browser.new_context(**context_kwargs)
    try:
        page = await context.new_page()
        await _install_resource_blocker(page)
        target = _with_cache_buster(start_url)
        # ``wait_until="commit"`` resolves when response headers arrive —
        # the fastest goto phase Playwright offers. Prior ``networkidle``
        # never settled on GitLab (ActionCable/Gravatar/Sentry); prior
        # ``domcontentloaded`` blocked on deferred JS parse, which under
        # Phase 2c's 64-wide renderer contention tripped the 30 s timeout
        # even when the server returned in <1.4 s. Downstream waits carry
        # the real readiness signal: the note selector + the body-text
        # poll below for gitlab_issue / gitlab_mr / etc.
        try:
            await page.goto(target, timeout=nav_timeout_ms, wait_until="commit")
        except Exception as exc:
            return ReachabilityOutcome.unreachable(
                kind="nav_failed",
                detail=f"goto {target}: {exc.__class__.__name__}: {exc}",
                url=start_url,
            )
        selector = _selector_for_kind(kind)
        if selector:
            try:
                await page.wait_for_selector(selector, timeout=selector_timeout_ms)
            except Exception:
                # Missing selector is non-fatal; signature grep is the
                # source of truth for reachability.
                logger.debug(
                    "phase 2c reachability: selector %r not found on %s",
                    selector,
                    start_url,
                )
        # For kinds whose seed lives in a lazy-loaded content stream, the
        # selector wait only proves *some* note/result has rendered. Poll
        # the body text for the actual signature so the probe doesn't
        # read body_text before the seeded note arrives in a later AJAX
        # batch. Non-fatal if the wait times out — the signature grep
        # below still runs and will record witnesses_absent with a
        # clear detail.
        if kind in {
            "gitlab_issue",
            "gitlab_mr",
            "gitlab_search_result",
            "gitlab_dashboard_list",
        }:
            await _wait_for_body_text(page, witnesses, selector_timeout_ms)
        try:
            body_text = await page.text_content("body") or ""
        except Exception as exc:
            return ReachabilityOutcome.unreachable(
                kind="body_read_failed",
                detail=f"text_content body: {exc.__class__.__name__}: {exc}",
                url=start_url,
            )
        # Match render_check's whitespace-normalizing comparison
        # (phase_2_render_check._normalize). Plain ``.lower()`` here
        # missed 30+ witnesses_absent failures where the body had
        # collapsed newlines / indented whitespace that did not line up
        # with the raw signature's whitespace. Normalize both sides and
        # the match behaves identically to render_check's.
        normalized_body = _normalize_for_match(body_text)
        matched = tuple(w for w in witnesses if _normalize_for_match(w) in normalized_body)
        missing = tuple(w for w in witnesses if _normalize_for_match(w) not in normalized_body)
        if not missing:
            return ReachabilityOutcome.direct(url=start_url, witnesses_matched=matched)
        return ReachabilityOutcome.unreachable(
            kind="witnesses_absent",
            detail=(
                f"start_url body missing {len(missing)}/{len(witnesses)} "
                "witnesses; seeded payload not reachable from benign entry point"
            ),
            url=start_url,
            witnesses_matched=matched,
            witnesses_missing=missing,
        )
    finally:
        try:
            await context.close()
        except Exception:
            logger.exception("phase 2c reachability failed to close context")
