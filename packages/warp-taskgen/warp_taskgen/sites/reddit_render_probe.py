"""Reddit/Postmill-owned Phase 2c render probe.

Postmill's comment carriers need more than "the payload text appeared
somewhere on the page": the seeded comment must be the exact comment a
Phase 4 agent reads first. This module owns that entry-viewport
evidence and the Postmill DOM-readiness wait.
``phase_2_render_check`` reaches it through a Site-keyed lookup rather
than a ``site_name == "reddit"`` branch.
"""

from __future__ import annotations

import logging
from typing import Any

from warp_taskgen.sites.render_probe import RenderOutcome

logger = logging.getLogger(__name__)

# Postmill/Reddit pages are server-rendered, but ``wait_until="commit"``
# returns before Chromium has parsed enough of the document to include
# late sidebar/profile content. Keep the global commit fast-path for
# GitLab's SPA tail and wait only for Reddit's static DOM readiness.
_REDDIT_DOM_READY_STATE = "domcontentloaded"


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


class RedditRenderProbe:
    """Site-owned render-probe behavior for Reddit/Postmill."""

    site_name = "reddit"

    async def wait_for_render(
        self,
        page: Any,
        *,
        target_url: str,
        signature: str,
        selector_timeout_ms: int,
        body_poll_timeout_ms: int,
    ) -> None:
        try:
            await page.wait_for_load_state(_REDDIT_DOM_READY_STATE, timeout=selector_timeout_ms)
        except Exception:
            pass

    def exact_visibility_comment_id(self, write_tokens: dict[str, Any] | None) -> str | None:
        return _reddit_comment_id(write_tokens)

    async def exact_visibility_probe(
        self,
        page: Any,
        *,
        comment_id: str,
        normalized_needle: str,
    ) -> dict[str, Any] | None:
        return await _reddit_seed_comment_visibility_probe(
            page,
            comment_id=comment_id,
            normalized_needle=normalized_needle,
        )

    async def read_your_write(
        self,
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
        """Postmill is server-rendered; there is no read-your-write fallback."""

        return None


__all__ = ["RedditRenderProbe"]
