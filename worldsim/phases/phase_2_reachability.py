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
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from worldsim.phases.phase_2_render_check import _with_cache_buster

logger = logging.getLogger(__name__)


_SITE_SELECTORS: dict[str, str] = {
    "gitlab_issue": ".issuable-list, .issue-details",
    "gitlab_mr": ".merge-request, .issuable-list",
    "gitlab_dashboard_list": ".issuable-list, .todos-list",
    "gitlab_search_result": ".search-results, .issuable-list",
    "reddit_submission": ".comment-list, .comments",
    "reddit_forum": ".submission",
    "reddit_dashboard_list": ".submission, .comment",
}

_WITNESS_MIN_LEN = 12
_DEFAULT_NAV_TIMEOUT_MS = 15000
_DEFAULT_SELECTOR_TIMEOUT_MS = 5000
_SEARCH_POLL_INTERVAL_MS = 500
_SEARCH_POLL_ATTEMPTS = 10


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


def derive_second_witness(rendered_payload: str | None, signature: str | None) -> str | None:
    """Pick a rare substring from the rendered payload distinct from signature.

    Returns the longest whitespace-delimited span of >= _WITNESS_MIN_LEN
    chars that does not overlap the signature, or None if nothing
    qualifies. Defense against signature collisions on pre-existing DOM
    content.
    """
    if not rendered_payload:
        return None
    needle = (signature or "").lower()
    candidates = re.split(r"[\s\r\n]+", rendered_payload)
    best: str | None = None
    for token in candidates:
        token = token.strip()
        if len(token) < _WITNESS_MIN_LEN:
            continue
        if needle and needle in token.lower():
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


async def _wait_for_search_title(page: Any, needle: str, timeout_ms: int) -> bool:
    """Poll the current page text every 500ms looking for *needle*.

    Used for gitlab_search_result where the seeded issue title may not
    appear immediately — GitLab search is DB LIKE on WebArena images
    but still has a short render window.
    """
    deadline = time.monotonic() + (timeout_ms / 1000.0)
    lowered = needle.lower()
    while time.monotonic() < deadline:
        try:
            body = await page.text_content("body") or ""
        except Exception:
            body = ""
        if lowered in body.lower():
            return True
        try:
            await page.wait_for_timeout(_SEARCH_POLL_INTERVAL_MS)
        except Exception:
            break
    return False


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
        target = _with_cache_buster(start_url)
        try:
            await page.goto(target, timeout=nav_timeout_ms, wait_until="domcontentloaded")
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
        if kind == "gitlab_search_result":
            await _wait_for_search_title(page, signature, selector_timeout_ms)
        try:
            body_text = await page.text_content("body") or ""
        except Exception as exc:
            return ReachabilityOutcome.unreachable(
                kind="body_read_failed",
                detail=f"text_content body: {exc.__class__.__name__}: {exc}",
                url=start_url,
            )
        lowered = body_text.lower()
        matched = tuple(w for w in witnesses if w.lower() in lowered)
        missing = tuple(w for w in witnesses if w.lower() not in lowered)
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
