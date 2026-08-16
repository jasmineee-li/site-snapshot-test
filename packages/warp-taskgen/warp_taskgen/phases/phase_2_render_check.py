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

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.parse import quote as urlquote

from warp_taskgen.agent_auth import playwright_storage_state
from warp_taskgen.seeding.site_contracts import normalize_identity_tokens
from warp_taskgen.sites import (
    ReadbackDecision,
    ReadbackFailure,
    ReadbackObservation,
    default_catalog,
)
from warp_taskgen.sites.gitlab_readback import note_html_for_id, rendered_note_text

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
_GITLAB_ISSUABLE_LIST_SELECTOR = ".issuable-list, .issues-list, .merge-requests-list"

# Postmill/Reddit pages are server-rendered, but ``wait_until="commit"``
# returns before Chromium has parsed enough of the document to include
# late sidebar/profile content. Keep the global commit fast-path for
# GitLab's SPA tail and wait only for Reddit's static DOM readiness.
_REDDIT_DOM_READY_STATE = "domcontentloaded"

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
    rendered_body_text: str | None = None
    layout_probe: dict[str, Any] | None = None
    diagnostics: dict[str, Any] | None = None

    @classmethod
    def passed(
        cls,
        *,
        url: str,
        signature: str,
        snippet: str,
        rendered_body_text: str | None = None,
        layout_probe: dict[str, Any] | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> RenderOutcome:
        return cls(
            ok=True,
            kind="",
            detail=f"signature {signature!r} present in {url}",
            urls_tried=[url],
            per_url_errors={},
            matched_url=url,
            matched_signature=signature,
            matched_snippet=snippet[:240],
            rendered_body_text=rendered_body_text,
            layout_probe=layout_probe,
            diagnostics=diagnostics,
        )

    @classmethod
    def failed(
        cls,
        *,
        kind: str,
        detail: str,
        urls_tried: list[str],
        per_url_errors: dict[str, str],
        diagnostics: dict[str, Any] | None = None,
    ) -> RenderOutcome:
        return cls(
            ok=False,
            kind=kind,
            detail=detail,
            urls_tried=urls_tried,
            per_url_errors=per_url_errors,
            diagnostics=diagnostics,
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
            if self.rendered_body_text:
                out["rendered_body_text"] = self.rendered_body_text[:2000]
            if self.layout_probe is not None:
                out["layout_probe"] = dict(self.layout_probe)
        else:
            out["kind"] = self.kind
        if self.diagnostics is not None:
            out["diagnostics"] = dict(self.diagnostics)
        return out


@dataclass(frozen=True)
class RenderSignatureSelection:
    signature: str
    call_index: int | None = None
    editor_method: str | None = None
    source_field: str | None = None


_GITLAB_REWRITTEN_TEXT_TOKEN_RE = re.compile(
    r"(?ix)"
    # Fully-qualified URLs and bare host/path strings. GitLab may
    # autolink or project-linkify these, changing the visible text.
    r"https?://\S+"
    r"|"
    r"\b(?:localhost|(?:[a-z0-9-]+\.)+[a-z]{2,})(?::\d+)?/[^\s<>\])]+"
    r"|"
    # GitLab references / mentions are also rendered through special
    # reference filters. Keep signatures out of those tokens.
    r"(?<![\w/])(?:\#|!|&)\d+\b"
    r"|"
    r"(?<![\w/])@[a-z0-9_.-]+"
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _is_gitlab_issuable_surface(url: str) -> bool:
    return "/-/issues" in url or "/-/merge_requests" in url


def _is_gitlab_issuable_detail_surface(url: str) -> bool:
    return "/-/issues/" in url or "/-/merge_requests/" in url


def _is_gitlab_issue_detail_surface(url: str) -> bool:
    return "/-/issues/" in url


def _trim_signature_candidate(text: str) -> str:
    return text.strip(" \t\r\n-\u2013\u2014:;,.()[]{}<>")


def _stable_render_signature_text(text: str, *, limit: int = 40) -> str | None:
    """Pick rendered-stable text without crossing GitLab rewrite tokens."""
    line = _strip_markdown_for_text_match(text).split("\n", 1)[0].strip()
    if not line:
        line = text.split("\n", 1)[0].strip()
    if not line:
        return None

    spans: list[str] = []
    cursor = 0
    for match in _GITLAB_REWRITTEN_TEXT_TOKEN_RE.finditer(line):
        spans.append(line[cursor : match.start()])
        cursor = match.end()
    spans.append(line[cursor:])

    candidates = [
        candidate[:limit].rstrip()
        for span in spans
        if len(candidate := _trim_signature_candidate(span)) >= 8
    ]
    if candidates:
        # Prefer the longest stable prose run. This keeps signatures
        # unique when the opening phrase is short or generic and avoids
        # straddling GitLab-autolinked project URLs such as
        # ``localhost:8023/group/project``.
        return max(candidates, key=len)

    # Back-compat fallback for terse payloads where the only available
    # text is itself a URL/reference-like token.
    return line[:limit].rstrip() or text[:limit].rstrip()


def render_signature_selection(
    seed: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> RenderSignatureSelection | None:
    """Select a unique rendered substring and the editor call that owns it.

    Prefers the seed nickname (ASCII, unique by construction in the
    adversarial dataset, and renders in a stable DOM location across
    Magento product reviews / Reddit posts / GitLab notes). Falls back
    in priority order to detail / body / description / note / bio /
    content (a rendered-stable first-line substring capped at 40 chars),
    then title.

    Every base field name is also tried with the ``_template`` and
    ``_text`` suffixes the editor-method binding contract uses for
    free-text template args (e.g. ``description_template``,
    ``bio_text``), so feasibility verification works on any plan that
    targets a method whose body lives on a templated arg.

    Returns None when no editor call carries a signature-bearing
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
            index,
            f"{call.get('site')}.{call.get('method')}",
            call.get("args"),
        )
        for index, call in enumerate(editor_calls)
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
    candidate_records = [
        (index, method, args)
        for index, method, args in call_records
        if method in preferred_methods and isinstance(args, dict)
    ]
    if not candidate_records:
        candidate_records = [
            (index, method, args) for index, method, args in call_records if isinstance(args, dict)
        ]
    # In multi-call seeds the last editor call is typically the one that
    # produces the user-visible note/comment while earlier calls create
    # parent resources or helper setup rows. Prefer later calls so a setup
    # title/description cannot overshadow the actually rendered payload.
    candidate_records = list(reversed(candidate_records))

    def _first_nonempty(fields: tuple[str, ...]) -> RenderSignatureSelection | None:
        for index, method, args in candidate_records:
            for base in fields:
                for variant in (base, f"{base}_template", f"{base}_text"):
                    raw = args.get(variant)
                    if isinstance(raw, str) and raw.strip():
                        value = raw.strip()
                        signature = (
                            value if base == "nickname" else _stable_render_signature_text(value)
                        )
                        if signature:
                            return RenderSignatureSelection(
                                signature=signature,
                                call_index=index,
                                editor_method=method,
                                source_field=variant,
                            )
        return None

    nickname = _first_nonempty(("nickname",))
    if nickname is not None:
        return nickname
    body = _first_nonempty(("detail", "body", "description", "note", "bio", "content"))
    if body is not None:
        return body
    title = _first_nonempty(("title", "name"))
    if title is not None:
        return title
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
    longest_index: int | None = None
    longest_method: str | None = None
    for index, method, args in candidate_records:
        for value in args.values():
            if not isinstance(value, str):
                continue
            stripped = value.strip()
            if not stripped or stripped.startswith("{benign_"):
                continue
            if len(stripped) > longest_len:
                longest = stripped
                longest_len = len(stripped)
                longest_index = index
                longest_method = method
    if longest is not None and longest_len >= 8:
        signature = _stable_render_signature_text(longest)
        if signature:
            return RenderSignatureSelection(
                signature=signature,
                call_index=longest_index,
                editor_method=longest_method,
                source_field="<longest_string_arg>",
            )
    return None


def render_signature(seed: dict[str, Any], metadata: dict[str, Any] | None = None) -> str | None:
    """Extract a unique substring expected to appear in the rendered DOM."""
    selection = render_signature_selection(seed, metadata)
    return selection.signature if selection is not None else None


# Sentinel byte for round-tripping escaped delimiters during markdown
# strip. Never legitimately appears in GitLab/Postmill rendered text.
_ESCAPED_STAR = "\x00"
_ESCAPED_UNDERSCORE = "\x01"
_ESCAPED_BACKTICK = "\x02"

# Triple-backtick fence detector. Captures language-tag + inner bytes so
# fenced code bodies round-trip verbatim (GitLab renders them inside
# <pre><code> which preserves content in text_content).
_FENCE_RE = re.compile(r"```[^\n]*\n.*?\n```", re.DOTALL)


def _strip_markdown_for_text_match(text: str) -> str:
    """Collapse markdown delimiters to what GitLab's CommonMark renderer
    emits into ``text_content``.

    Bug G: Phase 2c's reachability and render-check probes grep the
    rendered DOM body for the seeded signature / witnesses. Seeds are
    authored in markdown, but ``**bold**`` renders as ``<strong>bold</strong>``
    whose ``text_content`` is ``bold`` (no asterisks). Stripping the same
    delimiters from the signature before substring match restores
    symmetry.

    Preserves fenced-code regions verbatim (``\\`\\`\\``...``\\`\\`\\```)
    because GitLab renders them in ``<pre><code>`` which keeps inner
    bytes intact. Handles (a) ATX headings / blockquote / list markers
    at line start, (b) inline single-backtick code, (c) inline-link and
    inline-image wrappers + reference-link definitions, (d) bold/italic
    delimiters with CommonMark flanking rules (``**`` and ``__`` before
    ``*`` and ``_``; escape-sentinel round-trip so ``\\*\\*literal\\*\\*``
    survives; guards against ``5 * 3`` and ``*ptr``), (e) table pipe
    separators and divider rules. Idempotent and ``None``-safe.
    """
    if not text:
        return ""

    # Step 1: fence-aware segmentation. Strip only outside fences; keep
    # fence bodies verbatim while dropping the triple-backtick
    # delimiters themselves.
    out_parts: list[str] = []
    cursor = 0
    for match in _FENCE_RE.finditer(text):
        if match.start() > cursor:
            out_parts.append(("outside", text[cursor : match.start()]))
        fence_body = match.group(0)
        # Drop opening fence line (``` + optional lang + \n) and trailing ```.
        first_nl = fence_body.find("\n")
        inner = fence_body[first_nl + 1 : -3].rstrip("\n")
        out_parts.append(("inside", inner))
        cursor = match.end()
    if cursor < len(text):
        out_parts.append(("outside", text[cursor:]))

    transformed: list[str] = []
    for kind, seg in out_parts:
        if kind == "inside":
            # Fence body: preserve content verbatim. GitLab's <pre><code>
            # renders this byte-for-byte in text_content.
            transformed.append(seg)
            continue

        # Step 2: protect escaped delimiters with sentinels so they
        # survive the strip pass.
        seg = (
            seg.replace(r"\*", _ESCAPED_STAR)
            .replace(r"\_", _ESCAPED_UNDERSCORE)
            .replace(r"\`", _ESCAPED_BACKTICK)
        )

        # Step 3: line-leading markers (per line so multi-line structures
        # normalize correctly).
        stripped_lines: list[str] = []
        for line in seg.splitlines(keepends=True):
            stripped_lines.append(
                re.sub(r"^\s{0,3}(?:>\s?|#{1,6}\s+|[-*+]\s+)", "", line),
            )
        seg = "".join(stripped_lines)

        # Step 4: inline single-backticks (outside fences). Lookaround
        # guards prevent the regex from chewing triple-backtick
        # sequences pairwise — ``````system`` must stay ``````system`` so a
        # signature matches the body symmetrically when GitLab renders
        # nested fences (which leaves literal ``` in text_content).
        seg = re.sub(r"(?<!`)`([^`\n]+?)`(?!`)", r"\1", seg)

        # Step 5: link + image wrappers.
        seg = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", seg)  # ![alt](url)
        seg = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", seg)  # [text](url)
        seg = re.sub(r"\[([^\]]+)\]\[[^\]]*\]", r"\1", seg)  # [text][ref]
        seg = re.sub(r"(?m)^\s*\[[^\]]+\]:\s*\S+.*$", "", seg)  # reference defs

        # Step 6: bold + italic. Longest delimiters first so ``**a**``
        # does not get eaten by the single-star pass. CommonMark
        # flanking rules: emphasis runs must border non-whitespace on
        # both inner sides and must not be immediately preceded by or
        # followed by an alphanumeric (which would make them part of an
        # identifier, e.g. ``*ptr``).
        seg = re.sub(r"\*\*(\S(?:.*?\S)?)\*\*", r"\1", seg)
        seg = re.sub(r"__(\S(?:.*?\S)?)__", r"\1", seg)
        seg = re.sub(
            r"(?<![A-Za-z0-9_*])\*(\S(?:.*?\S)?)\*(?![A-Za-z0-9_*])",
            r"\1",
            seg,
        )
        seg = re.sub(
            r"(?<![A-Za-z0-9_])_(\S(?:.*?\S)?)_(?![A-Za-z0-9_])",
            r"\1",
            seg,
        )

        # Step 7: table pipes + divider rules.
        seg = re.sub(r"(?m)^\s*\|?\s*[-:]{3,}\s*(?:\|\s*[-:]{3,}\s*)*\|?\s*$", "", seg)
        seg = seg.replace("|", " ")

        # Step 8: restore escape sentinels to their literal form.
        seg = (
            seg.replace(_ESCAPED_STAR, "*")
            .replace(_ESCAPED_UNDERSCORE, "_")
            .replace(_ESCAPED_BACKTICK, "`")
        )
        transformed.append(seg)

    return "".join(transformed)


def _normalize(text: str | None) -> str:
    return re.sub(r"\s+", " ", _strip_markdown_for_text_match(text or "")).lower()


async def _layout_probe_for_signature(page: Any, normalized_needle: str) -> dict[str, Any] | None:
    """Return initial-viewport geometry for the best rendered text match."""
    if not normalized_needle:
        return None
    try:
        result = await page.evaluate(
            """
            (needle) => {
              const root = document.body || document.documentElement;
              if (!root) return null;
              const walker = document.createTreeWalker(
                root,
                NodeFilter.SHOW_TEXT,
                {
                  acceptNode(node) {
                    const parent = node.parentElement;
                    if (!parent) return NodeFilter.FILTER_REJECT;
                    const tag = parent.tagName;
                    if (tag === "SCRIPT" || tag === "STYLE" || tag === "NOSCRIPT") {
                      return NodeFilter.FILTER_REJECT;
                    }
                    return NodeFilter.FILTER_ACCEPT;
                  },
                },
              );
              const textNodes = [];
              const charMap = [];
              let corpus = "";
              function appendNormalized(content, nodeIndex) {
                const lower = String(content || "").toLowerCase();
                for (let offset = 0; offset < lower.length; offset += 1) {
                  const ch = lower[offset];
                  if (/\\s/.test(ch)) {
                    if (corpus.length === 0 || corpus[corpus.length - 1] === " ") {
                      continue;
                    }
                    corpus += " ";
                    charMap.push({ nodeIndex, offset });
                    continue;
                  }
                  corpus += ch;
                  charMap.push({ nodeIndex, offset });
                }
              }
              while (walker.nextNode()) {
                const node = walker.currentNode;
                const content = node.textContent || "";
                if (!content) continue;
                const nodeIndex = textNodes.length;
                textNodes.push(node);
                appendNormalized(content, nodeIndex);
              }
              const length = String(needle || "").length;
              const viewportH = window.innerHeight || 0;
              const viewportW = window.innerWidth || 0;
              const scrollY = window.scrollY || 0;
              const doc = document.documentElement || root;
              const docH = Math.max(doc.scrollHeight || 0, root.scrollHeight || 0);

              function probeOccurrence(matchOffset, occurrenceIndex) {
                let startInfo = null;
                let endInfo = null;
                for (let i = 0; i < length; i += 1) {
                  const info = charMap[matchOffset + i];
                  if (!info) continue;
                  if (!startInfo) startInfo = info;
                  endInfo = info;
                }
                if (!startInfo || !endInfo) return null;
                const startNode = textNodes[startInfo.nodeIndex];
                const endNode = textNodes[endInfo.nodeIndex];
                if (!startNode || !endNode) return null;
                const range = document.createRange();
                range.setStart(startNode, startInfo.offset);
                range.setEnd(endNode, Math.min((endInfo.offset || 0) + 1, (endNode.textContent || "").length));
                const rect = range.getBoundingClientRect();
                const visibleAtEntry =
                  rect.width > 0 &&
                  rect.height > 0 &&
                  rect.bottom > 0 &&
                  rect.top < viewportH &&
                  rect.right > 0 &&
                  rect.left < viewportW;
                let requiresExpand = false;
                for (let n = startNode.parentElement; n; n = n.parentElement) {
                  const style = window.getComputedStyle(n);
                  if (style.display === "none" || style.visibility === "hidden") {
                    requiresExpand = true;
                    break;
                  }
                  if (n.tagName === "DETAILS" && !n.open) {
                    requiresExpand = true;
                    break;
                  }
                  if (n.classList && n.classList.contains("comment--collapsed")) {
                    requiresExpand = true;
                    break;
                  }
                }
                return {
                  visible_at_entry: visibleAtEntry,
                  rect_top: rect.top,
                  rect_bottom: rect.bottom,
                  viewport_h: viewportH,
                  viewport_w: viewportW,
                  doc_h: docH,
                  scroll_to_visible_px: visibleAtEntry ? 0 : Math.max(0, rect.top + scrollY - 100),
                  requires_expand: requiresExpand,
                  occurrence_index: occurrenceIndex,
                };
              }

              let matchOffset = corpus.indexOf(String(needle || ""));
              if (matchOffset < 0) return null;
              let fallback = null;
              let occurrenceIndex = 0;
              while (matchOffset >= 0) {
                const candidate = probeOccurrence(matchOffset, occurrenceIndex);
                if (candidate) {
                  if (candidate.visible_at_entry) return candidate;
                  if (!fallback) fallback = candidate;
                  else if (fallback.requires_expand && !candidate.requires_expand) fallback = candidate;
                  else if (
                    fallback.requires_expand === candidate.requires_expand &&
                    candidate.scroll_to_visible_px < fallback.scroll_to_visible_px
                  ) {
                    fallback = candidate;
                  }
                }
                occurrenceIndex += 1;
                matchOffset = corpus.indexOf(
                  String(needle || ""),
                  matchOffset + Math.max(length, 1),
                );
              }
              return fallback;
            }
            """,
            normalized_needle,
        )
    except Exception:
        logger.debug("phase 2c render check: layout probe failed", exc_info=True)
        return None
    return result if isinstance(result, dict) else None


async def _exact_selector_layout_probe(page: Any, selector: str) -> dict[str, Any] | None:
    """Prove one exact Site-owned resource marker is rendered and not hidden."""

    try:
        result = await page.evaluate(
            """
            (selector) => {
              let matches;
              try {
                matches = Array.from(document.querySelectorAll(String(selector || "")));
              } catch (_) {
                return { ok: false, reason: "invalid_selector", match_count: 0 };
              }
              if (matches.length !== 1) {
                return { ok: false, reason: "identity_count", match_count: matches.length };
              }
              const marker = matches[0];
              let requiresExpand = false;
              for (let node = marker; node; node = node.parentElement) {
                const style = window.getComputedStyle(node);
                if (
                  style.display === "none" ||
                  style.visibility === "hidden" ||
                  Number(style.opacity || "1") === 0
                ) {
                  requiresExpand = true;
                  break;
                }
                if (node.tagName === "DETAILS" && !node.open) {
                  requiresExpand = true;
                  break;
                }
                if (node.classList && node.classList.contains("comment--collapsed")) {
                  requiresExpand = true;
                  break;
                }
              }
              const rect = marker.getBoundingClientRect();
              const painted = rect.width > 0 && rect.height > 0;
              return {
                ok: painted && !requiresExpand,
                reason: !painted ? "not_painted" : (requiresExpand ? "requires_expand" : "visible"),
                match_count: 1,
                requires_expand: requiresExpand,
                rect_top: rect.top,
                rect_bottom: rect.bottom,
              };
            }
            """,
            selector,
        )
    except Exception:
        logger.debug("phase 2c exact-resource layout probe failed", exc_info=True)
        return None
    return result if isinstance(result, dict) else None


def _same_committed_render_surface(expected_url: str, committed_url: object) -> bool:
    """Require navigation to remain on the requested route after redirects."""

    if not isinstance(committed_url, str) or not committed_url.strip():
        return False
    try:
        expected = urlsplit(expected_url)
        committed = urlsplit(committed_url)
        expected_query = sorted(
            (key, value)
            for key, value in parse_qsl(expected.query, keep_blank_values=True)
            if key != "_"
        )
        committed_query = sorted(
            (key, value)
            for key, value in parse_qsl(committed.query, keep_blank_values=True)
            if key != "_"
        )
    except (TypeError, ValueError):
        return False
    return (expected.scheme, expected.netloc, expected.path) == (
        committed.scheme,
        committed.netloc,
        committed.path,
    ) and expected_query == committed_query


def _extract_note_html(body: str, note_id_str: str) -> str | None:
    """Compatibility facade for Site-owned GitLab note interpretation."""
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return None
    _found, note_html = note_html_for_id(payload, note_id_str)
    return note_html


def _strip_html(html_blob: str) -> str:
    """Compatibility facade for Site-owned GitLab note text rendering."""

    return rendered_note_text(html_blob)


async def _gitlab_note_ryw_fastpath(
    *,
    page: Any,
    target_url: str,
    site_name: str,
    write_tokens: dict[str, Any] | None,
    timeout_ms: int,
    scoped_extra_http_headers: dict[str, str] | None = None,
    header_scope_url: str | None = None,
    diagnostics: dict[str, Any] | None = None,
    readback_site: Any | None = None,
) -> RenderOutcome | None:
    """Read-your-write fallback for GitLab issue / MR notes.

    When the body-text match misses on an issue or MR page, fetch the
    authoritative ``/discussions.json`` endpoint via the page's request
    context and look for the note_id the editor returned from its POST.
    If the id is in the JSON, the note is observably present on the
    server — downstream agents will see it the moment the Vue layer
    finishes hydrating, regardless of where the DOM-render race left
    ``text_content('body')``.

    Returns a passed ``RenderOutcome`` on match, ``None`` otherwise
    (so the caller falls through to the existing error classification).
    Skip conditions: non-gitlab site, URL not an issue/MR page,
    write_tokens missing note_id, or the JSON fetch fails.
    """
    if site_name != "gitlab" or write_tokens is None:
        logger.info(
            "phase 2c render RYW skip: site=%s write_tokens=%s", site_name, bool(write_tokens)
        )
        return None
    note_id = write_tokens.get("note_id")
    if note_id in (None, ""):
        logger.info("phase 2c render RYW skip: no note_id in write_tokens=%s", write_tokens)
        return None
    lower = target_url.lower()
    if "/-/issues/" not in lower and "/-/merge_requests/" not in lower:
        logger.info("phase 2c render RYW skip: URL not issue/MR: %s", target_url)
        return None
    # Strip an existing /discussions.json suffix so we don't build /discussions.json/discussions.json
    # when the target_url is already the JSON surface.
    base_url = target_url.split("?", 1)[0].rstrip("/")
    if base_url.endswith("/discussions.json"):
        base_url = base_url[: -len("/discussions.json")]
    json_url = f"{base_url}/discussions.json"
    logger.info(
        "phase 2c render RYW firing: note_id=%s json_url=%s (from target=%s)",
        note_id,
        json_url,
        target_url,
    )
    try:
        request_kwargs: dict[str, Any] = {
            "timeout": max(1000, int(timeout_ms)),
            "max_redirects": 0,
        }
        headers = _gitlab_same_origin_request_headers(
            url=json_url,
            scoped_extra_http_headers=scoped_extra_http_headers,
            header_scope_url=header_scope_url,
        )
        if headers:
            request_kwargs["headers"] = headers
        response = await page.request.get(json_url, **request_kwargs)
    except Exception as exc:
        logger.debug(
            "phase 2c render check: RYW fetch of %s raised %s: %s",
            json_url,
            exc.__class__.__name__,
            exc,
        )
        return None
    try:
        status = response.status
    except Exception:
        status = None
    if status is None or status < 200 or status >= 300:
        return None
    try:
        body = await response.text()
    except Exception:
        return None
    note_id_str = str(note_id)
    bound_readback = readback_site or default_catalog().bind(
        site="gitlab",
        origin=_gitlab_origin_from_target(target_url),
    )
    decision = bound_readback.interpret_readback(
        ReadbackObservation(
            kind="resource_identity",
            identity_tokens={"note_id": note_id},
            payload=body,
        )
    )
    matched = decision if isinstance(decision, ReadbackDecision) and decision.verified else None
    logger.info(
        "phase 2c render RYW fetched %s status=%s body_len=%d looking_for=%r match=%s",
        json_url,
        status,
        len(body or ""),
        f"note_id={note_id_str}",
        matched is not None,
    )
    if matched is not None:
        marker = next(
            (
                candidate
                for candidate in (
                    f'"id":{note_id_str}',
                    f'"id": {note_id_str}',
                    f'"id":"{note_id_str}"',
                    f'"id": "{note_id_str}"',
                )
                if candidate in body
            ),
            None,
        )
        pos = body.find(marker) if marker is not None else -1
        snippet = body[max(0, pos - 80) : pos + 200] if pos >= 0 else body[:200]
        return RenderOutcome.passed(
            url=json_url,
            signature=matched.matched_signature or f"note_id={note_id_str}",
            snippet=snippet,
            rendered_body_text=matched.rendered_text,
            diagnostics=diagnostics,
        )
    return None


def _gitlab_same_origin_request_headers(
    *,
    url: str,
    scoped_extra_http_headers: dict[str, str] | None,
    header_scope_url: str | None,
) -> dict[str, str] | None:
    if scoped_extra_http_headers and header_scope_url:
        from warp_taskgen.phases.phase_2_reachability import _same_origin

        if _same_origin(url, header_scope_url):
            return scoped_extra_http_headers
    return None


def _gitlab_origin_from_target(target_url: str) -> str | None:
    try:
        parsed = urlsplit(target_url)
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return f"{parsed.scheme}://{parsed.netloc}"


def _gitlab_issue_description_ryw_urls(target_url: str, write_tokens: dict[str, Any]) -> list[str]:
    origin = _gitlab_origin_from_target(target_url)
    if origin is None:
        return []
    issue_iid = write_tokens.get("issue_iid")
    urls: list[str] = []
    try:
        path = urlsplit(target_url).path
    except ValueError:
        path = ""
    match = re.match(
        r"(?P<issue_base>.*?/-/issues/)(?P<path_iid>\d+)(?:\.json|/discussions\.json)?/?$",
        path,
    )
    if match and issue_iid not in (None, ""):
        urls.append(f"{origin}{match.group('issue_base')}{urlquote(str(issue_iid), safe='')}.json")
    project_id = write_tokens.get("project_id")
    if project_id not in (None, "") and issue_iid not in (None, ""):
        urls.append(
            f"{origin}/api/v4/projects/{urlquote(str(project_id), safe='')}"
            f"/issues/{urlquote(str(issue_iid), safe='')}"
        )
    return list(dict.fromkeys(urls))


def _append_gitlab_issue_description_ryw_diagnostic(
    diagnostics: dict[str, Any] | None,
    entry: dict[str, Any],
) -> None:
    if diagnostics is None:
        return
    attempts = diagnostics.setdefault("gitlab_issue_description_ryw_attempts", [])
    if not isinstance(attempts, list):
        attempts = []
        diagnostics["gitlab_issue_description_ryw_attempts"] = attempts
    if len(attempts) >= 8:
        return
    attempts.append(entry)


def _gitlab_issue_description_snapshot(description: str) -> dict[str, Any]:
    return {
        "description_len": len(description),
        "description_sha256": hashlib.sha256(description.encode("utf-8")).hexdigest()[:16],
        "description_prefix": description[:240],
    }


_WRITE_TOKEN_KEYS: tuple[str, ...] = (
    "note_id",
    "issue_iid",
    "project_id",
    "comment_id",
    "submission_id",
    "review_id",
)


def _write_tokens_from_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    declared = value.get("write_tokens")
    if isinstance(declared, dict):
        try:
            return dict(normalize_identity_tokens(declared))
        except ValueError:
            return {}
    try:
        return dict(normalize_identity_tokens(value))
    except ValueError:
        pass
    write_tokens: dict[str, Any] = {}
    for key in _WRITE_TOKEN_KEYS:
        token_value = value.get(key)
        if token_value not in (None, ""):
            write_tokens[key] = token_value
    return write_tokens


def _editor_call_record_for_selection(
    metadata: dict[str, Any],
    selection: RenderSignatureSelection | None,
) -> dict[str, Any] | None:
    if selection is None or selection.call_index is None:
        return None
    records = metadata.get("editor_call_results")
    if not isinstance(records, list):
        return None
    for record in records:
        if not isinstance(record, dict):
            continue
        if record.get("call_index") == selection.call_index:
            return record
    return None


def _render_check_inputs_from_metadata(
    *,
    metadata: dict[str, Any],
    selection: RenderSignatureSelection | None,
) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    """Bind render verification to the payload-bearing editor call.

    Multi-call adversarial seeds are required for Phase 4 because the
    adversarial seed must include benign setup plus the appended attacker
    write. The aggregate metadata remains useful for reporting, but render
    admission should first use the read surfaces and write identifiers from
    the same call that supplied the signature. If per-call metadata is absent
    (older tests/artifacts), fall back to the aggregate contract.
    """
    aggregate_urls = metadata.get("read_surface_urls")
    urls = (
        [url for url in aggregate_urls if isinstance(url, str) and url.strip()]
        if isinstance(aggregate_urls, list)
        else []
    )
    write_tokens = _write_tokens_from_mapping(metadata)
    diagnostics: dict[str, Any] = {}
    if selection is not None:
        diagnostics.update(
            {
                "payload_call_index": selection.call_index,
                "payload_editor_method": selection.editor_method,
                "payload_source_field": selection.source_field,
            }
        )

    call_record = _editor_call_record_for_selection(metadata, selection)
    if call_record is None:
        if write_tokens:
            diagnostics["write_tokens_source"] = "aggregate_seed_metadata"
            diagnostics["write_token_keys"] = sorted(write_tokens)
        return urls, write_tokens, diagnostics

    call_urls = call_record.get("read_surface_urls")
    if isinstance(call_urls, list):
        selected_urls = [url for url in call_urls if isinstance(url, str) and url.strip()]
        if selected_urls:
            urls = selected_urls
            diagnostics["read_surface_source"] = "payload_editor_call"
    call_write_tokens = call_record.get("write_tokens")
    if isinstance(call_write_tokens, dict):
        selected_write_tokens = _write_tokens_from_mapping(call_write_tokens)
        if selected_write_tokens:
            write_tokens = selected_write_tokens
            diagnostics["write_tokens_source"] = "payload_editor_call"
            diagnostics["write_token_keys"] = sorted(selected_write_tokens)
    if "read_surface_source" not in diagnostics:
        diagnostics["read_surface_source"] = "aggregate_seed_metadata"
    if "write_tokens_source" not in diagnostics and write_tokens:
        diagnostics["write_tokens_source"] = "aggregate_seed_metadata"
        diagnostics["write_token_keys"] = sorted(write_tokens)
    return urls, write_tokens, diagnostics


def _reddit_comment_id(write_tokens: dict[str, Any] | None) -> str | None:
    if not isinstance(write_tokens, dict):
        return None
    value = write_tokens.get("comment_id")
    if value in (None, ""):
        return None
    text = str(value).strip()
    return text or None


async def _reddit_seed_comment_visibility_probe(
    page: Any,
    *,
    comment_id: str,
    normalized_needle: str,
) -> dict[str, Any] | None:
    """Return strict entry-viewport evidence for an exact Postmill comment.

    Generic body-text rendering is not enough for Reddit/Postmill comment
    carriers: a seeded comment can be present in the parent thread DOM while a
    different comment is the one painted/read by the agent. This probe binds the
    payload witness to the editor-returned ``comment_id`` and requires that the
    exact comment is the first visible comment block at page entry.
    """
    if not comment_id or not normalized_needle:
        return None
    try:
        result = await page.evaluate(
            """
            ({ commentId, needle }) => {
              const root = document.body || document.documentElement;
              if (!root) return null;
              const targetId = String(commentId || "");
              const normalizedNeedle = String(needle || "");
              const viewportH = window.innerHeight || 0;
              const viewportW = window.innerWidth || 0;

              function normalize(text) {
                return String(text || "").toLowerCase().replace(/\\s+/g, " ").trim();
              }
              function isVisibleRect(rect) {
                return Boolean(
                  rect &&
                    rect.width > 0 &&
                    rect.height > 0 &&
                    rect.bottom > 0 &&
                    rect.top < viewportH &&
                    rect.right > 0 &&
                    rect.left < viewportW
                );
              }
              function elementCommentId(el) {
                if (!el) return "";
                for (const key of ["data-comment-id", "comment-id", "data-id"]) {
                  const value = el.getAttribute && el.getAttribute(key);
                  if (value) return String(value);
                }
                const id = el.getAttribute && el.getAttribute("id");
                if (id && id.startsWith("comment-")) return id.slice("comment-".length);
                if (id && id.startsWith("comment_")) return id.slice("comment_".length);
                if (id) return id;
                const anchors = el.querySelectorAll ? el.querySelectorAll("a[href*='/comment/']") : [];
                for (const anchor of anchors) {
                  const href = String(anchor.getAttribute("href") || "");
                  const match = href.match(/\\/comment\\/([^/?#]+)/);
                  if (match) return match[1];
                }
                return "";
              }
              function hasCommentPermalink(el, id) {
                if (!el || !el.querySelector) return false;
                if (elementCommentId(el) === id) return true;
                return Boolean(el.querySelector(`a[href*="/comment/${CSS.escape(id)}"]`));
              }
              function candidateRootsFor(id) {
                const nodes = [];
                const selectors = [
                  `[data-comment-id="${CSS.escape(id)}"]`,
                  `[comment-id="${CSS.escape(id)}"]`,
                  `[data-id="${CSS.escape(id)}"]`,
                  `#comment-${CSS.escape(id)}`,
                  `#comment_${CSS.escape(id)}`,
                  `a[href*="/comment/${CSS.escape(id)}"]`,
                ];
                for (const selector of selectors) {
                  try {
                    nodes.push(...document.querySelectorAll(selector));
                  } catch (_) {}
                }
                const roots = [];
                const seen = new Set();
                for (const node of nodes) {
                  let best = null;
                  for (let el = node.nodeType === Node.ELEMENT_NODE ? node : node.parentElement; el && el !== document.documentElement; el = el.parentElement) {
                    const text = normalize(el.innerText || el.textContent || "");
                    if (!text.includes(normalizedNeedle)) continue;
                    if (!hasCommentPermalink(el, id)) continue;
                    best = el;
                    const classes = String(el.className || "");
                    if (/(^|\\s)comment(\\s|$)/i.test(classes) || el.hasAttribute("data-comment-id")) {
                      break;
                    }
                  }
                  if (best && !seen.has(best)) {
                    seen.add(best);
                    roots.push(best);
                  }
                }
                roots.sort((a, b) => {
                  const ar = a.getBoundingClientRect();
                  const br = b.getBoundingClientRect();
                  return (a.innerText || a.textContent || "").length - (b.innerText || b.textContent || "").length ||
                    ar.top - br.top;
                });
                return roots;
              }
              function visibleCommentRoots() {
                const anchors = Array.from(document.querySelectorAll("a[href*='/comment/']"));
                const roots = [];
                const seen = new Set();
                for (const anchor of anchors) {
                  let best = null;
                  for (let el = anchor; el && el !== document.documentElement; el = el.parentElement) {
                    if (!hasCommentPermalink(el, elementCommentId(anchor) || "")) continue;
                    const text = normalize(el.innerText || el.textContent || "");
                    if (!text || text.length < 8) continue;
                    best = el;
                    const classes = String(el.className || "");
                    if (/(^|\\s)comment(\\s|$)/i.test(classes) || el.hasAttribute("data-comment-id")) {
                      break;
                    }
                  }
                  if (!best || seen.has(best)) continue;
                  seen.add(best);
                  const rect = best.getBoundingClientRect();
                  if (isVisibleRect(rect)) {
                    roots.push({ id: elementCommentId(best), top: rect.top, text: normalize(best.innerText || best.textContent || "").slice(0, 160) });
                  }
                }
                roots.sort((a, b) => a.top - b.top);
                return roots;
              }
              function textRangeProbe(container) {
                const walker = document.createTreeWalker(
                  container,
                  NodeFilter.SHOW_TEXT,
                  {
                    acceptNode(node) {
                      const parent = node.parentElement;
                      if (!parent) return NodeFilter.FILTER_REJECT;
                      const tag = parent.tagName;
                      if (tag === "SCRIPT" || tag === "STYLE" || tag === "NOSCRIPT") {
                        return NodeFilter.FILTER_REJECT;
                      }
                      return NodeFilter.FILTER_ACCEPT;
                    },
                  },
                );
                const textNodes = [];
                const charMap = [];
                let corpus = "";
                function appendNormalized(content, nodeIndex) {
                  const lower = String(content || "").toLowerCase();
                  for (let offset = 0; offset < lower.length; offset += 1) {
                    const ch = lower[offset];
                    if (/\\s/.test(ch)) {
                      if (corpus.length === 0 || corpus[corpus.length - 1] === " ") continue;
                      corpus += " ";
                      charMap.push({ nodeIndex, offset, isSpace: true });
                      continue;
                    }
                    corpus += ch;
                    charMap.push({ nodeIndex, offset, isSpace: false });
                  }
                }
                while (walker.nextNode()) {
                  const node = walker.currentNode;
                  const content = node.textContent || "";
                  if (!content) continue;
                  const nodeIndex = textNodes.length;
                  textNodes.push(node);
                  appendNormalized(content, nodeIndex);
                }
                const offset = corpus.indexOf(normalizedNeedle);
                if (offset < 0) return { match_found_in_comment: false };
                const length = normalizedNeedle.length;
                let startInfo = null;
                let endInfo = null;
                let visibleNonspace = 0;
                for (let i = 0; i < length; i += 1) {
                  const info = charMap[offset + i];
                  if (!info) continue;
                  if (!startInfo) startInfo = info;
                  endInfo = info;
                  if (!info.isSpace) visibleNonspace += 1;
                }
                if (!startInfo || !endInfo) return { match_found_in_comment: false };
                const startNode = textNodes[startInfo.nodeIndex];
                const endNode = textNodes[endInfo.nodeIndex];
                const range = document.createRange();
                range.setStart(startNode, startInfo.offset);
                range.setEnd(endNode, Math.min((endInfo.offset || 0) + 1, (endNode.textContent || "").length));
                const rect = range.getBoundingClientRect();
                let requiresExpand = false;
                for (let n = startNode.parentElement; n; n = n.parentElement) {
                  const style = window.getComputedStyle(n);
                  if (style.display === "none" || style.visibility === "hidden") {
                    requiresExpand = true;
                    break;
                  }
                  if (n.tagName === "DETAILS" && !n.open) {
                    requiresExpand = true;
                    break;
                  }
                  if (n.classList && n.classList.contains("comment--collapsed")) {
                    requiresExpand = true;
                    break;
                  }
                  if (n === container) break;
                }
                return {
                  match_found_in_comment: true,
                  visible_at_entry: isVisibleRect(rect),
                  visible_nonspace_chars: isVisibleRect(rect) ? visibleNonspace : 0,
                  rect_top: rect.top,
                  rect_bottom: rect.bottom,
                  viewport_h: viewportH,
                  viewport_w: viewportW,
                  requires_expand: requiresExpand,
                };
              }

              const targetRoots = candidateRootsFor(targetId);
              const firstVisible = visibleCommentRoots()[0] || null;
              if (targetRoots.length === 0) {
                return {
                  ok: false,
                  reason: "seed_comment_block_not_found",
                  comment_id: targetId,
                  first_visible_comment_id: firstVisible ? firstVisible.id : null,
                  first_visible_comment_text: firstVisible ? firstVisible.text : null,
                };
              }
              const probe = textRangeProbe(targetRoots[0]);
              const firstId = firstVisible ? firstVisible.id : null;
              const ok = Boolean(
                probe.match_found_in_comment &&
                  probe.visible_at_entry &&
                  !probe.requires_expand &&
                  firstId === targetId
              );
              return {
                ok,
                reason: ok ? "seed_comment_first_visible_and_painted" :
                  (!probe.match_found_in_comment ? "signature_not_in_seed_comment" :
                    probe.requires_expand ? "seed_comment_requires_expand" :
                    !probe.visible_at_entry ? "seed_comment_not_visible_at_entry" :
                    firstId !== targetId ? "different_first_visible_comment" :
                    "seed_comment_visibility_unknown"),
                comment_id: targetId,
                first_visible_comment_id: firstId,
                first_visible_comment_text: firstVisible ? firstVisible.text : null,
                target_root_count: targetRoots.length,
                ...probe,
              };
            }
            """,
            {"commentId": comment_id, "needle": normalized_needle},
        )
    except Exception:
        logger.debug("phase 2c render check: reddit comment visibility probe failed", exc_info=True)
        return None
    return result if isinstance(result, dict) else None


async def _gitlab_issue_description_ryw_fastpath(
    *,
    page: Any,
    target_url: str,
    site_name: str,
    signature: str,
    write_tokens: dict[str, Any] | None,
    timeout_ms: int,
    scoped_extra_http_headers: dict[str, str] | None = None,
    header_scope_url: str | None = None,
    diagnostics: dict[str, Any] | None = None,
    readback_site: Any | None = None,
) -> RenderOutcome | None:
    """Read-your-write fallback for GitLab created issue descriptions.

    This deliberately fires only from issue detail/read surfaces, never from
    project issue-list pages. A project-list title carrier must still prove the
    title is visible in the listing DOM; this fallback exists for created issue
    descriptions whose detail page/API visibility can race the first render
    probe. Phase 2c still runs the separate reachability graph after this, so a
    listing-to-detail carrier is admitted only if the benign route exposes the
    detail page witnesses.
    """
    if site_name != "gitlab" or write_tokens is None:
        return None
    if write_tokens.get("note_id") not in (None, ""):
        return None
    project_id = write_tokens.get("project_id")
    issue_iid = write_tokens.get("issue_iid")
    if project_id in (None, "") or issue_iid in (None, ""):
        return None
    if not _is_gitlab_issue_detail_surface(target_url):
        return None
    urls = _gitlab_issue_description_ryw_urls(target_url, write_tokens)
    if not urls:
        return None
    logger.info(
        "phase 2c render issue RYW firing: project_id=%s issue_iid=%s urls=%s",
        project_id,
        issue_iid,
        urls,
    )
    for read_url in urls:
        try:
            request_kwargs: dict[str, Any] = {
                "timeout": max(1000, int(timeout_ms)),
                "max_redirects": 0,
            }
            headers = _gitlab_same_origin_request_headers(
                url=read_url,
                scoped_extra_http_headers=scoped_extra_http_headers,
                header_scope_url=header_scope_url,
            )
            if headers:
                request_kwargs["headers"] = headers
            response = await page.request.get(read_url, **request_kwargs)
        except Exception as exc:
            logger.debug(
                "phase 2c render check: issue RYW fetch of %s raised %s: %s",
                read_url,
                exc.__class__.__name__,
                exc,
            )
            _append_gitlab_issue_description_ryw_diagnostic(
                diagnostics,
                {
                    "url": read_url,
                    "result": "request_error",
                    "error": f"{exc.__class__.__name__}: {exc}",
                },
            )
            continue
        try:
            status = response.status
        except Exception:
            status = None
        if status is None or status < 200 or status >= 300:
            _append_gitlab_issue_description_ryw_diagnostic(
                diagnostics,
                {"url": read_url, "status": status, "result": "http_status"},
            )
            continue
        try:
            payload = await response.json()
        except Exception:
            try:
                payload = json.loads(await response.text())
            except Exception:
                _append_gitlab_issue_description_ryw_diagnostic(
                    diagnostics,
                    {"url": read_url, "status": status, "result": "invalid_json"},
                )
                continue
        if not isinstance(payload, dict):
            _append_gitlab_issue_description_ryw_diagnostic(
                diagnostics,
                {"url": read_url, "status": status, "result": "non_object_json"},
            )
            continue
        description = payload.get("description")
        if not isinstance(description, str):
            _append_gitlab_issue_description_ryw_diagnostic(
                diagnostics,
                {"url": read_url, "status": status, "result": "missing_description"},
            )
            continue
        normalized_description = _normalize(description)
        normalized_signature = _normalize(signature)
        bound_readback = readback_site or default_catalog().bind(
            site="gitlab",
            origin=_gitlab_origin_from_target(target_url),
        )
        decision = bound_readback.interpret_readback(
            ReadbackObservation(
                kind="resource_signature",
                identity_tokens={"project_id": project_id, "issue_iid": issue_iid},
                payload={
                    "normalized_description": _normalize(description),
                    "normalized_signature": _normalize(signature),
                    "project_id": payload.get("project_id"),
                    "iid": payload.get("iid"),
                },
                signature=normalized_signature,
            )
        )
        if not isinstance(decision, ReadbackDecision) or not decision.verified:
            _append_gitlab_issue_description_ryw_diagnostic(
                diagnostics,
                {
                    "url": read_url,
                    "status": status,
                    "result": "description_signature_absent",
                    "signature": signature[:120],
                    **_gitlab_issue_description_snapshot(description),
                },
            )
            logger.info(
                "phase 2c render issue RYW fetched %s status=%s description_len=%d match=False",
                read_url,
                status,
                len(description),
            )
            continue
        pos = normalized_description.find(normalized_signature)
        snippet = (
            description[max(0, pos - 40) : pos + len(signature) + 120] if pos >= 0 else description
        )
        logger.info(
            "phase 2c render issue RYW fetched %s status=%s description_len=%d match=True",
            read_url,
            status,
            len(description),
        )
        _append_gitlab_issue_description_ryw_diagnostic(
            diagnostics,
            {
                "url": read_url,
                "status": status,
                "result": "description_signature_present",
                **_gitlab_issue_description_snapshot(description),
            },
        )
        return RenderOutcome.passed(
            url=read_url,
            signature=f"issue_iid={issue_iid}",
            snippet=snippet,
            rendered_body_text=description,
            diagnostics=diagnostics,
        )
    return None


_BODY_POLL_INITIAL_MS = 100
_BODY_POLL_MAX_MS = 2000
# Body-poll deadline (decoupled from selector_timeout_ms). GitLab's
# write-to-visible tail on loaded hosts runs to 5-15 s; the 10 s selector
# timeout catches first-batch note rendering but starves the poll looking
# for a seeded note that arrives in a slow batch 2-3. 20 s gives the
# exponential backoff room to walk its full schedule (100→...→2000 ms)
# before giving up.
_BODY_POLL_TIMEOUT_MS = 20000
# Kept as a compat alias for downstream consumers / tests that referenced
# the old constant. Not used internally after the backoff switch.
_BODY_POLL_INTERVAL_MS = 500


async def _wait_for_body_text(page: Any, needle: str, timeout_ms: int) -> bool:
    """Poll ``page.text_content('body')`` with exponential backoff for *needle*.

    Bug J (2026-04-23) established the poll (previously just a race against
    ``text_content`` returning the SPA shell). This follow-up replaces the
    fixed 500 ms cadence with an exponential backoff that starts at 100 ms
    and caps at 2000 ms. Motivation: GitLab's write-to-visible pipeline
    (Postgres → sidekiq indexer → cache invalidation → action_cable →
    Vue hydration) has a bimodal latency distribution — most writes are
    visible in well under 500 ms, but a long p99 tail caused by sidekiq
    queue depth and cache-warm pauses runs to 5-15 s under the 16-way
    renderer contention Phase 2c imposes. The old 500 ms fixed cadence
    over-polled fast hits (20 polls x 500 ms to catch a 400 ms write) and
    under-covered slow hits (10 s deadline exhausted before the tail).

    The new schedule covers both in one pass:
      100, 200, 400, 800, 1600, 2000, 2000, ...  (until ``timeout_ms``)

    Returns ``True`` on match, ``False`` on timeout. The caller still
    falls through to the full ``_normalize`` comparison on timeout so a
    slow-hydrating signature that arrives after the deadline is still
    recorded as present if ``text_content`` catches it post-wait.
    """
    deadline = time.monotonic() + (timeout_ms / 1000.0)
    needle_norm = _normalize(needle)
    if not needle_norm:
        return False
    interval_ms = _BODY_POLL_INITIAL_MS
    while time.monotonic() < deadline:
        try:
            body = await page.text_content("body") or ""
        except Exception:
            body = ""
        if needle_norm in _normalize(body):
            return True
        try:
            await page.wait_for_timeout(interval_ms)
        except Exception:
            break
        interval_ms = min(interval_ms * 2, _BODY_POLL_MAX_MS)
    return False


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
    parts = urlsplit(url)
    query = f"{parts.query}&" if parts.query else ""
    query = f"{query}{urlencode({'_': int(time.time() * 1000)})}"
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))


def _classify_failure(
    *,
    urls_tried: list[str],
    per_url_errors: dict[str, str],
    signature: str,
    diagnostics: dict[str, Any] | None = None,
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
            diagnostics=diagnostics,
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
        diagnostics=diagnostics,
    )


async def verify_seed_renders(
    *,
    browser: Any,  # playwright.async_api.Browser
    urls: list[str],
    site_name: str,
    site_url: str,
    signature: str | None,
    nav_timeout_ms: int = 30000,
    selector_timeout_ms: int = 10000,
    storage_state_path: str | None = None,
    browser_context_kwargs: dict[str, Any] | None = None,
    write_tokens: dict[str, Any] | None = None,
    diagnostics: dict[str, Any] | None = None,
    readback_site: Any | None = None,
    readback_plan: Any | None = None,
) -> RenderOutcome:
    """Open a fresh context, try each URL until the signature appears.

    Browser handle is owned by the caller (one per Phase 2c run); this
    function creates and tears down a single context (~200ms each, vs
    1.5-3s for a full launch). Caller is responsible for capping
    concurrent contexts at ~8 to avoid Playwright's eviction edge cases.

    ``browser_context_kwargs`` threads the configured benign user's auth into
    Playwright so private/authed-only pages are checked with the same identity
    Phase 4 uses. ``storage_state_path`` is retained for older callers.
    """
    if not urls:
        return RenderOutcome.failed(
            kind="render_unverified",
            detail="editor emitted no read_surface_urls — cannot verify rendering",
            urls_tried=[],
            per_url_errors={},
            diagnostics=diagnostics,
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
            diagnostics=diagnostics,
        )

    needle = _normalize(signature)
    seen: list[str] = []
    errors: dict[str, str] = {}
    context_kwargs: dict[str, Any] = dict(browser_context_kwargs or {})
    from warp_taskgen.phases.phase_2_reachability import _pop_scoped_extra_http_headers

    scoped_extra_http_headers = _pop_scoped_extra_http_headers(context_kwargs)
    if storage_state_path and "storage_state" not in context_kwargs:
        path_obj = Path(storage_state_path)
        if path_obj.exists():
            storage_state, error = playwright_storage_state(path_obj)
            if error is None:
                context_kwargs["storage_state"] = storage_state
            else:
                return RenderOutcome.failed(
                    kind="auth_unusable",
                    detail=f"storage_state {storage_state_path} is unusable: {error}",
                    urls_tried=[],
                    per_url_errors={},
                    diagnostics=diagnostics,
                )
        else:
            return RenderOutcome.failed(
                kind="auth_missing",
                detail=f"storage_state {storage_state_path} not found",
                urls_tried=[],
                per_url_errors={},
                diagnostics=diagnostics,
            )
    reddit_comment_id = _reddit_comment_id(write_tokens)
    strict_reddit_comment_visibility = site_name == "reddit" and reddit_comment_id is not None
    strict_reddit_failure: RenderOutcome | None = None
    context = await browser.new_context(**context_kwargs)
    try:
        page = await context.new_page()
        # Imported lazily to break a potential module-level import cycle
        # with phase_2_reachability (which already imports _with_cache_buster
        # from this module at module load).
        from warp_taskgen.phases.phase_2_reachability import _install_resource_blocker

        await _install_resource_blocker(
            page,
            scoped_extra_http_headers=scoped_extra_http_headers,
            header_scope_url=site_url,
        )
        for raw_url in urls:
            target = _with_cache_buster(_resolve_url(raw_url, site_url))
            seen.append(target)
            try:
                # ``wait_until="commit"`` resolves the moment response
                # headers arrive. Prior ``networkidle`` never settled on
                # SPAs (ActionCable, Gravatar, Sentry); prior
                # ``domcontentloaded`` blocked on deferred JS parse,
                # which under Phase 2c's 64-wide renderer contention
                # tripped the 30 s timeout even when the server returned
                # in <1.4 s. The per-site selector waits below (note list
                # on issue/MR pages, review block on shopping PDPs) are
                # the real readiness signal. If the selector wait fails,
                # the ``text_content`` fallback still runs — the
                # signature check decides.
                await page.goto(target, timeout=nav_timeout_ms, wait_until="commit")
                if site_name == "reddit":
                    try:
                        await page.wait_for_load_state(
                            _REDDIT_DOM_READY_STATE, timeout=selector_timeout_ms
                        )
                    except Exception:
                        pass
                if site_name == "shopping" and "/catalog/product/view/" in target:
                    try:
                        await page.wait_for_selector(
                            _SHOPPING_REVIEW_SELECTOR, timeout=selector_timeout_ms
                        )
                    except Exception:
                        pass
                if site_name == "gitlab" and _is_gitlab_issuable_surface(target):
                    if _is_gitlab_issuable_detail_surface(target):
                        selector = _GITLAB_NOTE_SELECTOR
                    else:
                        selector = _GITLAB_ISSUABLE_LIST_SELECTOR
                    try:
                        await page.wait_for_selector(selector, timeout=selector_timeout_ms)
                    except Exception:
                        pass
                    # GitLab issue/MR detail threads and listing rows are both
                    # populated after the initial response. Poll body text so
                    # Phase 2c does not sample the SPA shell before the seeded
                    # title/note has been inserted into the DOM.
                    await _wait_for_body_text(page, signature, _BODY_POLL_TIMEOUT_MS)
                body_text = await page.text_content("body") or ""
                normalized = _normalize(body_text)
                if needle in normalized:
                    pos = normalized.find(needle)
                    raw_pos = 0
                    if pos >= 0 and pos < len(body_text):
                        raw_pos = pos
                    snippet = body_text[max(0, raw_pos - 40) : raw_pos + len(signature) + 40]
                    layout_probe = await _layout_probe_for_signature(page, needle)
                    supports_site_observer = getattr(
                        readback_site, "supports_readback_observation", None
                    )
                    site_readback_observer = (
                        getattr(readback_site, "observe_readback_html", None)
                        if callable(supports_site_observer) and supports_site_observer()
                        else None
                    )
                    if (
                        readback_plan is not None
                        and getattr(readback_plan, "verification_mode", None) == "seed_resource"
                        and callable(site_readback_observer)
                    ):
                        if not _same_committed_render_surface(target, getattr(page, "url", None)):
                            errors[target] = "site_readback_failed:redirected_read_surface"
                            continue
                        visibility_selector = readback_site.readback_visibility_selector(
                            readback_plan
                        )
                        if isinstance(visibility_selector, ReadbackFailure):
                            errors[target] = (
                                "site_readback_failed:"
                                f"{visibility_selector.reason}:{visibility_selector.detail}"
                            )
                            continue
                        exact_layout_probe = await _exact_selector_layout_probe(
                            page, visibility_selector
                        )
                        if not isinstance(exact_layout_probe, dict) or not exact_layout_probe.get(
                            "ok"
                        ):
                            reason = (
                                exact_layout_probe.get("reason", "probe_failed")
                                if isinstance(exact_layout_probe, dict)
                                else "probe_failed"
                            )
                            errors[target] = f"site_readback_failed:visibility_unproven:{reason}"
                            continue
                        try:
                            html = await page.content()
                        except Exception as exc:
                            errors[target] = (
                                "site_readback_failed:readback_html_unavailable:"
                                f"{exc.__class__.__name__}"
                            )
                            continue
                        observation = site_readback_observer(html, readback_plan)
                        if isinstance(observation, ReadbackFailure):
                            # Sites without the optional observation capability
                            # retain their established render/readback path.
                            if observation.reason != "unsupported_readback_observation":
                                errors[target] = (
                                    "site_readback_failed:"
                                    f"{observation.reason}:{observation.detail}"
                                )
                                continue
                        elif not isinstance(observation, ReadbackObservation):
                            errors[target] = "site_readback_failed:invalid_readback_observation"
                            continue
                        else:
                            decision = readback_site.interpret_readback(observation)
                            if not isinstance(decision, ReadbackDecision) or not decision.verified:
                                reason = (
                                    decision.reason
                                    if isinstance(decision, ReadbackDecision)
                                    else "invalid_readback_decision"
                                )
                                errors[target] = f"site_readback_failed:{reason}"
                                continue
                            readback_diagnostics = dict(diagnostics or {})
                            readback_diagnostics["site_readback"] = {
                                "verified": True,
                                "reason": decision.reason,
                                "visibility": exact_layout_probe,
                            }
                            diagnostics = readback_diagnostics
                    if strict_reddit_comment_visibility:
                        probe = await _reddit_seed_comment_visibility_probe(
                            page,
                            comment_id=reddit_comment_id,
                            normalized_needle=needle,
                        )
                        strict_diagnostics = dict(diagnostics or {})
                        strict_diagnostics["reddit_seed_comment_visibility"] = probe or {
                            "ok": False,
                            "reason": "probe_failed",
                            "comment_id": reddit_comment_id,
                        }
                        bound_readback = readback_site or default_catalog().bind(
                            site="reddit",
                            origin=site_url,
                        )
                        decision = bound_readback.interpret_readback(
                            ReadbackObservation(
                                kind="comment_visibility",
                                identity_tokens={"comment_id": reddit_comment_id},
                                payload=probe,
                                signature=signature,
                            )
                        )
                        if not isinstance(decision, ReadbackDecision) or not decision.verified:
                            reason = (
                                decision.reason
                                if isinstance(decision, ReadbackDecision)
                                else "probe_failed"
                            )
                            errors[target] = f"reddit_seed_comment_visibility_failed:{reason}"
                            strict_reddit_failure = RenderOutcome.failed(
                                kind="reddit_seed_comment_not_visible",
                                detail=(
                                    "reddit comment carrier rendered in page text, but the "
                                    f"seeded comment_id {reddit_comment_id!r} was not proven "
                                    f"as the first visible painted comment ({reason})"
                                ),
                                urls_tried=list(seen),
                                per_url_errors=dict(errors),
                                diagnostics=strict_diagnostics,
                            )
                            continue
                        diagnostics = strict_diagnostics
                    return RenderOutcome.passed(
                        url=target,
                        signature=signature,
                        snippet=snippet,
                        rendered_body_text=body_text,
                        layout_probe=layout_probe,
                        diagnostics=diagnostics,
                    )
                # Read-your-write fallback for GitLab note kinds: the text
                # match missed, but the editor returned the authoritative
                # note_id from its POST response. Fetch the discussions.json
                # surface directly via the page's request context (inherits
                # the live session) and look for that exact id in the JSON
                # body. This bypasses every DOM/render-pipeline race at
                # once — sidekiq indexer delay, page-cache invalidation,
                # GFM token rewriting — because we are reading back the
                # same resource we just wrote using the same API that
                # wrote it. On match, report the RYW hit so downstream
                # diagnostics make the source of verification explicit.
                ryw_hit = await _gitlab_note_ryw_fastpath(
                    page=page,
                    target_url=target,
                    site_name=site_name,
                    write_tokens=write_tokens,
                    timeout_ms=selector_timeout_ms,
                    scoped_extra_http_headers=scoped_extra_http_headers,
                    header_scope_url=site_url,
                    diagnostics=diagnostics,
                    readback_site=readback_site,
                )
                if ryw_hit is not None:
                    return ryw_hit
                issue_ryw_hit = await _gitlab_issue_description_ryw_fastpath(
                    page=page,
                    target_url=target,
                    site_name=site_name,
                    signature=signature,
                    write_tokens=write_tokens,
                    timeout_ms=selector_timeout_ms,
                    scoped_extra_http_headers=scoped_extra_http_headers,
                    header_scope_url=site_url,
                    diagnostics=diagnostics,
                    readback_site=readback_site,
                )
                if issue_ryw_hit is not None:
                    return issue_ryw_hit
                errors[target] = f"signature_absent (body_len={len(body_text)})"
            except Exception as exc:
                msg = f"{exc.__class__.__name__}: {exc}"
                errors[target] = msg
                logger.debug("phase 2c render check error on %s: %s", target, msg)
        if strict_reddit_failure is not None:
            return strict_reddit_failure
        return _classify_failure(
            urls_tried=seen,
            per_url_errors=errors,
            signature=signature,
            diagnostics=diagnostics,
        )
    finally:
        try:
            await context.close()
        except Exception:
            logger.exception("phase 2c render check failed to close context")
