// @ts-check
/**
 * Per-character visibility + viewport-bg-color query for the Paint-Verified
 * Payload Oracle.
 *
 * Caller injects ordered attack witnesses via
 * ``__WORLDSIM_WITNESSES_JSON__`` substitution (JSON-encoded at the call
 * site). This query locates the first witness on the page by normalized
 * substring-matching text nodes via TreeWalker — we do NOT depend on any DOM
 * attribute or wrapping element being preserved by the site. This matters
 * because many production sites (Magento, reddit, gitlab) sanitize HTML in
 * UGC fields and rewrite Markdown/links before rendering.
 *
 * Returns an object with:
 *
 *   - ``entries``: one record per character of the matched payload text.
 *     Each record carries ``layoutVisible`` (in-viewport ∧ ancestor
 *     visible ∧ not occluded) and the ``liveRect`` where the glyph is
 *     painted.
 *   - ``backgroundColor``: the page's effective viewport background color,
 *     resolved by walking up the ancestor chain from the matched text
 *     node until a non-transparent ``background-color`` is found, then
 *     falling back to
 *     ``getComputedStyle(document.documentElement).backgroundColor``. Used
 *     host-side by the ink-occupancy oracle.
 *   - ``matchOffset`` / ``matchFound``: diagnostic fields so the host can
 *     distinguish "payload was rendered somewhere else in the DOM"
 *     (matchFound=true) from "payload never reached the DOM"
 *     (matchFound=false).
 *   - ``matchedWitnessId`` / ``matchedWitnessText``: which attack witness
 *     was found. Host-side artifacts persist this for run postmortems.
 *   - ``pageUrl``: current ``window.location.href`` for host-side filtering
 *     of off-surface pages.
 *
 * Range-based per-char rects keep OpenType ligature shaping intact. No
 * DOM mutation of the live payload.
 *
 * @typedef {{x:number,y:number,w:number,h:number}} Rect
 * @typedef {{r:number,g:number,b:number}} Rgb
 * @typedef {{id:string, text:string}} Witness
 * @typedef {{i:number,char:string,isSpace:boolean,layoutVisible:boolean,liveRect:Rect|null}} CharResult
 * @typedef {{id:string,text:string,offset:number}} WitnessMatch
 * @typedef {{entries:CharResult[], backgroundColor:Rgb, matchFound:boolean, matchOffset:number, matchedWitnessId:string|null, matchedWitnessText:string|null, pageUrl:string}} QueryResult
 */

(() => {
  /** @type {Witness[]} */
  const witnesses = __WORLDSIM_WITNESSES_JSON__;

  /** @returns {QueryResult} */
  function emptyResult() {
    return {
      entries: [],
      backgroundColor: { r: 255, g: 255, b: 255 },
      matchFound: false,
      matchOffset: -1,
      matchedWitnessId: null,
      matchedWitnessText: null,
      pageUrl: String(window.location.href || ""),
    };
  }

  if (!Array.isArray(witnesses) || witnesses.length === 0) {
    return emptyResult();
  }

  // Walk every renderable text node and linearize them into one corpus
  // string + a parallel ``charMap`` from global character offset to
  // ``(textNodeIndex, intraNodeOffset)``. This lets us find the payload
  // even when it spans multiple text nodes — which is the common case on
  // sites that auto-linkify URLs (reddit) or apply inline formatting
  // (`<em>`, `<strong>`, `<code>`, `<a>`) inside UGC fields. The pre-fix
  // implementation searched each text node individually and missed every
  // such case. See Finding 4 in
  // ``docs/todo-pvpo-post-ship-review.md``.
  const walker = document.createTreeWalker(
    document.body || document.documentElement,
    NodeFilter.SHOW_TEXT,
    {
      /** @param {Node} node */
      acceptNode(node) {
        const p = /** @type {Element | null} */ (node.parentElement);
        if (!p) return NodeFilter.FILTER_REJECT;
        const tag = p.tagName;
        if (tag === "SCRIPT" || tag === "STYLE" || tag === "NOSCRIPT") {
          return NodeFilter.FILTER_REJECT;
        }
        return NodeFilter.FILTER_ACCEPT;
      },
    },
  );

  /** @type {Text[]} */
  const textNodes = [];
  /** @type {{nodeIndex:number, offset:number}[]} */
  const normalizedCharMap = [];
  let normalizedCorpus = "";
  while (walker.nextNode()) {
    const node = /** @type {Text} */ (walker.currentNode);
    const content = node.textContent || "";
    if (!content) continue;
    const nodeIndex = textNodes.length;
    textNodes.push(node);
    appendNormalized(content, nodeIndex);
  }

  const matches = findWitnessMatches(normalizedCorpus, witnesses);
  if (matches.length === 0) {
    // Payload wasn't rendered anywhere in the DOM. The host treats this as
    // zero coverage — routed to placement-fix.
    return emptyResult();
  }

  let selected = buildMatchResult(matches[0]);
  for (const candidate of matches) {
    const result = buildMatchResult(candidate);
    if (hasVisibleNonSpaceEntry(result.entries)) {
      selected = result;
      break;
    }
  }
  return selected;

  /**
   * Append content to normalizedCorpus while preserving a map from each
   * normalized char back to its source text-node char. Whitespace runs become
   * one ASCII space, matching Python-side witness normalization.
   *
   * @param {string} content
   * @param {number} nodeIndex
   */
  function appendNormalized(content, nodeIndex) {
    for (let o = 0; o < content.length; o++) {
      const ch = content[o];
      if (/\s/.test(ch)) {
        if (
          normalizedCorpus.length === 0 ||
          normalizedCorpus[normalizedCorpus.length - 1] === " "
        ) {
          continue;
        }
        normalizedCorpus += " ";
        normalizedCharMap.push({ nodeIndex, offset: o });
        continue;
      }
      normalizedCorpus += ch;
      normalizedCharMap.push({ nodeIndex, offset: o });
    }
  }

  /**
   * @param {string} haystack
   * @param {Witness[]} candidates
   * @returns {WitnessMatch[]}
   */
  function findWitnessMatches(haystack, candidates) {
    /** @type {WitnessMatch[]} */
    const matches = [];
    for (const candidate of candidates) {
      if (!candidate || typeof candidate.text !== "string") continue;
      const text = normalizeText(candidate.text);
      if (!text) continue;
      let offset = haystack.indexOf(text);
      while (offset >= 0) {
        matches.push({
          id: typeof candidate.id === "string" ? candidate.id : "witness",
          text,
          offset,
        });
        offset = haystack.indexOf(text, offset + Math.max(text.length, 1));
      }
      if (matches.length > 0) return matches;
    }
    return matches;
  }

  /**
   * @param {WitnessMatch} match
   * @returns {QueryResult}
   */
  function buildMatchResult(match) {
    const matchOffset = match.offset;
    const witnessText = match.text;
    const firstSourceInfo = normalizedCharMap[matchOffset];
    const firstSourceNode = textNodes[firstSourceInfo.nodeIndex];

    /** @type {CharResult[]} */
    const entries = [];
    const range = document.createRange();
    const viewportW = window.innerWidth;
    const viewportH = window.innerHeight;

    for (let i = 0; i < witnessText.length; i++) {
      const char = witnessText[i];
      if (/\s/.test(char)) {
        entries.push({
          i,
          char,
          isSpace: true,
          layoutVisible: false,
          liveRect: null,
        });
        continue;
      }

      const sourceInfo = normalizedCharMap[matchOffset + i];
      const sourceNode = textNodes[sourceInfo.nodeIndex];
      range.setStart(sourceNode, sourceInfo.offset);
      range.setEnd(sourceNode, sourceInfo.offset + 1);
      const liveRect = range.getBoundingClientRect();

      const inViewport =
        liveRect.width > 0 &&
        liveRect.height > 0 &&
        liveRect.right > 0 &&
        liveRect.bottom > 0 &&
        liveRect.left < viewportW &&
        liveRect.top < viewportH;

      const ancestor = sourceNode.parentElement;
      const ancestorVisible = ancestor
        ? ancestor.checkVisibility({
            opacityProperty: true,
            visibilityProperty: true,
            contentVisibilityAuto: true,
          })
        : false;

      let notOccluded = false;
      if (inViewport && ancestorVisible && ancestor) {
        const cx = liveRect.left + liveRect.width / 2;
        const cy = liveRect.top + liveRect.height / 2;
        const topEl = document.elementFromPoint(cx, cy);
        notOccluded =
          topEl === ancestor || (topEl !== null && ancestor.contains(topEl));
      }

      const layoutVisible = inViewport && ancestorVisible && notOccluded;

      entries.push({
        i,
        char,
        isSpace: false,
        layoutVisible,
        liveRect: layoutVisible ? serializeRect(liveRect) : null,
      });
    }

    return {
      entries,
      backgroundColor: resolveBackgroundColor(firstSourceNode.parentElement),
      matchFound: true,
      matchOffset,
      matchedWitnessId: match.id,
      matchedWitnessText: witnessText,
      pageUrl: String(window.location.href || ""),
    };
  }

  /**
   * @param {CharResult[]} entries
   * @returns {boolean}
   */
  function hasVisibleNonSpaceEntry(entries) {
    return entries.some((entry) => !entry.isSpace && entry.layoutVisible);
  }

  /**
   * @param {string} str
   * @returns {string}
   */
  function normalizeText(str) {
    return str.replace(/\s+/g, " ").trim();
  }

  /**
   * @param {DOMRect} r
   * @returns {Rect}
   */
  function serializeRect(r) {
    return {
      x: Math.round(r.left),
      y: Math.round(r.top),
      w: Math.round(r.width),
      h: Math.round(r.height),
    };
  }

  /**
   * Resolve the effective background color seen behind the matched text.
   * Walk up from the matched text node's parent; return the first
   * non-transparent ``background-color``. Fall back to documentElement,
   * then opaque white.
   *
   * @param {Element | null} start
   * @returns {Rgb}
   */
  function resolveBackgroundColor(start) {
    /** @type {Element | null} */
    let node = start;
    while (node) {
      const cs = getComputedStyle(node);
      const parsed = parseRgb(cs.backgroundColor);
      if (parsed) return parsed;
      node = node.parentElement;
    }
    const rootBg = parseRgb(
      getComputedStyle(document.documentElement).backgroundColor,
    );
    if (rootBg) return rootBg;
    return { r: 255, g: 255, b: 255 };
  }

  /**
   * Parse ``rgb(r, g, b)`` or ``rgba(r, g, b, a)`` strings. Returns null
   * for ``transparent`` / fully-alpha-zero so the ancestor walk keeps going.
   *
   * @param {string} str
   * @returns {Rgb | null}
   */
  function parseRgb(str) {
    if (!str) return null;
    const match = str.match(
      /rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*([\d.]+)\s*)?\)/,
    );
    if (!match) return null;
    const alpha = match[4] === undefined ? 1 : parseFloat(match[4]);
    if (!isFinite(alpha) || alpha === 0) return null;
    return {
      r: parseInt(match[1], 10),
      g: parseInt(match[2], 10),
      b: parseInt(match[3], 10),
    };
  }
})();
