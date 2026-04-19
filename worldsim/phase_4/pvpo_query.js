// @ts-check
/**
 * Per-character visibility + viewport-bg-color query for the Paint-Verified
 * Payload Oracle.
 *
 * Caller injects the seeded payload text via ``__WORLDSIM_PAYLOAD_JSON__``
 * substitution (JSON-encoded at the call site). This query locates the
 * payload on the page by substring-matching text nodes via TreeWalker —
 * we do NOT depend on any DOM attribute or wrapping element being
 * preserved by the site. This matters because many production sites
 * (Magento, reddit, gitlab) sanitize HTML in UGC fields: a ``<span
 * data-worldsim-payload>`` wrapper is frequently stripped before rendering.
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
 *
 * Range-based per-char rects keep OpenType ligature shaping intact. No
 * DOM mutation of the live payload.
 *
 * @typedef {{x:number,y:number,w:number,h:number}} Rect
 * @typedef {{r:number,g:number,b:number}} Rgb
 * @typedef {{i:number,char:string,isSpace:boolean,layoutVisible:boolean,liveRect:Rect|null}} CharResult
 * @typedef {{entries:CharResult[], backgroundColor:Rgb, matchFound:boolean, matchOffset:number}} QueryResult
 */

(() => {
  /** @type {string} */
  const payloadText = __WORLDSIM_PAYLOAD_JSON__;

  /** @returns {QueryResult} */
  function emptyResult() {
    return {
      entries: [],
      backgroundColor: { r: 255, g: 255, b: 255 },
      matchFound: false,
      matchOffset: -1,
    };
  }

  if (!payloadText) return emptyResult();

  // Walk every text node and find the first one whose textContent contains
  // ``payloadText``. Skip script/style subtrees — their text nodes never
  // render.
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

  /** @type {Text | null} */
  let matchNode = null;
  let matchOffset = -1;
  while (walker.nextNode()) {
    const node = /** @type {Text} */ (walker.currentNode);
    const content = node.textContent || "";
    const idx = content.indexOf(payloadText);
    if (idx >= 0) {
      matchNode = node;
      matchOffset = idx;
      break;
    }
  }

  if (matchNode === null) {
    // Payload wasn't rendered anywhere in the DOM. The host treats this as
    // zero coverage — routed to placement-fix.
    return emptyResult();
  }

  /** @type {CharResult[]} */
  const entries = [];

  const range = document.createRange();
  const viewportW = window.innerWidth;
  const viewportH = window.innerHeight;

  for (let i = 0; i < payloadText.length; i++) {
    const char = payloadText[i];
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

    range.setStart(matchNode, matchOffset + i);
    range.setEnd(matchNode, matchOffset + i + 1);
    const liveRect = range.getBoundingClientRect();

    const inViewport =
      liveRect.width > 0 &&
      liveRect.height > 0 &&
      liveRect.right > 0 &&
      liveRect.bottom > 0 &&
      liveRect.left < viewportW &&
      liveRect.top < viewportH;

    const ancestor = matchNode.parentElement;
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

  const backgroundColor = resolveBackgroundColor(matchNode.parentElement);

  return { entries, backgroundColor, matchFound: true, matchOffset };

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
