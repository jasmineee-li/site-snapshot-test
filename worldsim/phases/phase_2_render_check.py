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

from worldsim.agent_auth import playwright_storage_state

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


def _trim_signature_candidate(text: str) -> str:
    return text.strip(" \t\r\n-\u2013\u2014:;,.()[]{}<>")


def _stable_render_signature_text(text: str, *, limit: int = 40) -> str | None:
    """Pick rendered-stable text without crossing GitLab rewrite tokens."""
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


def render_signature(seed: dict[str, Any], metadata: dict[str, Any] | None = None) -> str | None:
    """Extract a unique substring expected to appear in the rendered DOM.

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
        # Take only first-line prose and avoid GitLab tokens whose
        # rendered text can differ from their source markdown/plaintext
        # form (autolinked project URLs, issue refs, mentions). A fixed
        # 40-char prefix can otherwise straddle a URL boundary and
        # produce a signature that is present in discussions.json but
        # impossible to find in rendered DOM text_content.
        return _stable_render_signature_text(body)
    title = _first_nonempty(("title", "name"))
    if title is not None:
        return _stable_render_signature_text(title)
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
        return _stable_render_signature_text(longest)
    return None


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


async def _gitlab_note_ryw_fastpath(
    *,
    page: Any,
    target_url: str,
    site_name: str,
    write_tokens: dict[str, Any] | None,
    timeout_ms: int,
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
        return None
    note_id = write_tokens.get("note_id")
    if note_id in (None, ""):
        return None
    lower = target_url.lower()
    if "/-/issues/" not in lower and "/-/merge_requests/" not in lower:
        return None
    base_url = target_url.split("?", 1)[0].rstrip("/")
    json_url = f"{base_url}/discussions.json"
    try:
        response = await page.request.get(json_url, timeout=max(1000, int(timeout_ms)))
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
    # Match the editor-returned id against the JSON payload. Try both the
    # compact (``"id":42``) and spaced (``"id": 42``) shapes Ruby's
    # ``ActiveSupport::JSON.encode`` can produce depending on options.
    token_key = f'"id":{note_id}'
    token_key_spaced = f'"id": {note_id}'
    if token_key in body or token_key_spaced in body:
        # Pull a small context window around the match for the snippet
        # field so evidence reports show what matched.
        idx = body.find(token_key)
        if idx < 0:
            idx = body.find(token_key_spaced)
        snippet = body[max(0, idx - 40) : idx + 200] if idx >= 0 else body[:200]
        return RenderOutcome.passed(
            url=json_url,
            signature=f"note_id={note_id}",
            snippet=snippet,
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
    nav_timeout_ms: int = 30000,
    selector_timeout_ms: int = 10000,
    storage_state_path: str | None = None,
    browser_context_kwargs: dict[str, Any] | None = None,
    write_tokens: dict[str, Any] | None = None,
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
    context_kwargs: dict[str, Any] = dict(browser_context_kwargs or {})
    from worldsim.phases.phase_2_reachability import _pop_scoped_extra_http_headers

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
                )
        else:
            return RenderOutcome.failed(
                kind="auth_missing",
                detail=f"storage_state {storage_state_path} not found",
                urls_tried=[],
                per_url_errors={},
            )
    context = await browser.new_context(**context_kwargs)
    try:
        page = await context.new_page()
        # Imported lazily to break a potential module-level import cycle
        # with phase_2_reachability (which already imports _with_cache_buster
        # from this module at module load).
        from worldsim.phases.phase_2_reachability import _install_resource_blocker

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
                if site_name == "gitlab" and (
                    "/-/issues/" in target or "/-/merge_requests/" in target
                ):
                    try:
                        await page.wait_for_selector(
                            _GITLAB_NOTE_SELECTOR, timeout=selector_timeout_ms
                        )
                    except Exception:
                        pass
                    # Bug J: the note selector fires on batch-1 render,
                    # but the seeded note often lives in batch 2/3 of
                    # the lazy discussions.json stream. Poll the body
                    # text for the signature so we do not race the
                    # second batch and record a false signature_absent.
                    # Composes with Bug G — _normalize already strips
                    # markdown delimiters on both sides.
                    #
                    # Use _BODY_POLL_TIMEOUT_MS (20 s) instead of the 10 s
                    # selector_timeout_ms: the selector wait above only
                    # needs to see *some* note render, but the poll has
                    # to cover GitLab's long write-to-visible tail
                    # (sidekiq indexer + cache invalidation) under Phase
                    # 2c's 16-way renderer contention. The 10 s bound
                    # cost 2-4 tasks/run to the p99 tail even after
                    # Bug J landed; 20 s plus the backoff schedule
                    # closes that gap.
                    await _wait_for_body_text(page, signature, _BODY_POLL_TIMEOUT_MS)
                body_text = await page.text_content("body") or ""
                normalized = _normalize(body_text)
                if needle in normalized:
                    pos = normalized.find(needle)
                    raw_pos = 0
                    if pos >= 0 and pos < len(body_text):
                        raw_pos = pos
                    snippet = body_text[max(0, raw_pos - 40) : raw_pos + len(signature) + 40]
                    return RenderOutcome.passed(url=target, signature=signature, snippet=snippet)
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
                )
                if ryw_hit is not None:
                    return ryw_hit
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
