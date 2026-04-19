// @ts-check
/**
 * Per-character visibility + viewport-bg-color query for the Paint-Verified
 * Payload Oracle.
 *
 * Runs inside the page via CDP ``Runtime.evaluate``. Returns an object with:
 *
 *   - ``entries``: one record per character of the seeded payload text
 *     identified by the ``[data-worldsim-payload]`` anchor. Each record
 *     carries ``layoutVisible`` (in-viewport ∧ ancestor visible ∧ not
 *     occluded) and the ``liveRect`` where the glyph is painted.
 *   - ``backgroundColor``: the page's effective viewport background color,
 *     resolved by walking up the ancestor chain from the payload anchor
 *     until a non-transparent ``background-color`` is found, then falling
 *     back to ``getComputedStyle(document.documentElement).backgroundColor``.
 *     Used host-side by the ink-occupancy oracle to distinguish painted
 *     glyph pixels from untouched background pixels.
 *
 * Range-based per-char rects keep OpenType ligature shaping intact. No DOM
 * mutation of the live payload.
 *
 * @typedef {{x:number,y:number,w:number,h:number}} Rect
 * @typedef {{r:number,g:number,b:number}} Rgb
 * @typedef {{i:number,char:string,isSpace:boolean,layoutVisible:boolean,liveRect:Rect|null}} CharResult
 * @typedef {{entries:CharResult[], backgroundColor:Rgb}} QueryResult
 */

(() => {
  /** @returns {QueryResult} */
  function emptyResult() {
    return { entries: [], backgroundColor: { r: 255, g: 255, b: 255 } };
  }

  const payloadAnchor = document.querySelector("[data-worldsim-payload]");
  if (!payloadAnchor) return emptyResult();

  const textNode = payloadAnchor.firstChild;
  if (!textNode || textNode.nodeType !== Node.TEXT_NODE) return emptyResult();

  /** @type {CharResult[]} */
  const entries = [];

  const range = document.createRange();
  const viewportW = window.innerWidth;
  const viewportH = window.innerHeight;

  const content = textNode.textContent || "";
  for (let i = 0; i < content.length; i++) {
    const char = content[i];
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

    range.setStart(textNode, i);
    range.setEnd(textNode, i + 1);
    const liveRect = range.getBoundingClientRect();

    const inViewport =
      liveRect.width > 0 &&
      liveRect.height > 0 &&
      liveRect.right > 0 &&
      liveRect.bottom > 0 &&
      liveRect.left < viewportW &&
      liveRect.top < viewportH;

    const ancestor = textNode.parentElement;
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

  const backgroundColor = resolveBackgroundColor(payloadAnchor);

  return { entries, backgroundColor };

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
   * Resolve the effective background color seen behind the payload text.
   * Walk up the ancestor chain from the payload anchor; return the first
   * non-transparent ``background-color``. Fall back to documentElement
   * and finally to opaque white.
   *
   * @param {Element} start
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
    const rootBg = parseRgb(getComputedStyle(document.documentElement).backgroundColor);
    if (rootBg) return rootBg;
    return { r: 255, g: 255, b: 255 };
  }

  /**
   * Parse ``rgb(r, g, b)`` or ``rgba(r, g, b, a)`` strings. Returns null
   * for ``transparent`` / fully-alpha-zero, so ``resolveBackgroundColor``
   * keeps walking.
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
