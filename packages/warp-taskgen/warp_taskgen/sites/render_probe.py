"""Site-neutral render-probe primitives shared by Phase 2c probes.

This is a leaf module: it owns the render-check outcome value, the
markdown/whitespace normalizer both Phase 2c probes compare on, and the
single body-text poll. It imports nothing from ``warp_taskgen.sites``
itself and must not be re-exported from ``warp_taskgen.sites.__init__``
— Phase 2c render check and reachability import it directly.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


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


# Sentinel byte for round-tripping escaped delimiters during markdown
# strip. Never legitimately appears in GitLab/Postmill rendered text.
_ESCAPED_STAR = "\x00"
_ESCAPED_UNDERSCORE = "\x01"
_ESCAPED_BACKTICK = "\x02"

# Triple-backtick fence detector. Captures language-tag + inner bytes so
# fenced code bodies round-trip verbatim (GitLab renders them inside
# <pre><code> which preserves content in text_content).
_FENCE_RE = re.compile(r"```[^\n]*\n.*?\n```", re.DOTALL)


def strip_markdown_for_text_match(text: str) -> str:
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


def normalize_for_text_match(text: str | None) -> str:
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

    return re.sub(r"\s+", " ", strip_markdown_for_text_match(text or "")).lower()


_BODY_POLL_INITIAL_MS = 100
_BODY_POLL_MAX_MS = 2000


async def wait_for_body_text(
    page: Any,
    needle: str | Iterable[str],
    timeout_ms: int,
) -> bool:
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

    ``needle`` may be a single string (back-compat) or an iterable of
    strings; with multiple needles the poll waits for *all* of them to
    appear so that a fast-matching primary signature (e.g. a URL that
    is present in the static body) does not short-circuit the wait for
    a later-arriving secondary witness (the hydrated note prose).
    Empty / falsy needles are ignored, so a request with nothing to look
    for is vacuously satisfied and returns ``True`` without polling.

    Returns ``True`` iff every non-empty needle is present in the body
    within ``timeout_ms``, ``False`` on timeout. Render-check callers
    still fall through to the full ``normalize_for_text_match``
    comparison on timeout so a slow-hydrating signature that arrives
    after the deadline is still recorded as present if ``text_content``
    catches it post-wait.
    """
    if isinstance(needle, str):
        needles_raw: tuple[str, ...] = (needle,)
    else:
        needles_raw = tuple(needle)
    needles_norm = tuple(n for n in (normalize_for_text_match(raw) for raw in needles_raw) if n)
    if not needles_norm:
        return True
    deadline = time.monotonic() + (timeout_ms / 1000.0)
    interval_ms = _BODY_POLL_INITIAL_MS
    while time.monotonic() < deadline:
        try:
            body = await page.text_content("body") or ""
        except Exception:
            body = ""
        normalized_body = normalize_for_text_match(body)
        if all(n in normalized_body for n in needles_norm):
            return True
        try:
            await page.wait_for_timeout(interval_ms)
        except Exception:
            break
        interval_ms = min(interval_ms * 2, _BODY_POLL_MAX_MS)
    return False


__all__ = [
    "RenderOutcome",
    "normalize_for_text_match",
    "strip_markdown_for_text_match",
    "wait_for_body_text",
]
