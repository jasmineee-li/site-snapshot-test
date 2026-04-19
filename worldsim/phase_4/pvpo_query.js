// @ts-check
/**
 * Per-character visibility query for the Paint-Verified Payload Oracle.
 *
 * Runs inside the page via CDP `Runtime.evaluate`. Returns an array of
 * per-char records for the seeded payload identified by the
 * `[data-worldsim-payload]` anchor. Host-side code (pvpo_capture.py) uses
 * these records together with the atomic `HeadlessExperimental.beginFrame`
 * screenshot to decide, per character, whether the character was both
 * layout-visible and pixel-identical to the hidden-DOM reference render.
 *
 * Three checks compose `layoutVisible`:
 *   - In visual viewport (Range.getBoundingClientRect intersected with
 *     window.innerWidth × innerHeight).
 *   - Ancestor not hidden (Element.checkVisibility with opacity/visibility/
 *     content-visibility flags).
 *   - Not occluded (document.elementFromPoint at rect center returns the
 *     ancestor or a descendant).
 *
 * Range-based per-char rects (no DOM mutation of the live payload) keep
 * OpenType ligature shaping intact. The reference container uses per-char
 * spans; ligatures may merge across span boundaries there, but that does
 * not affect the oracle because comparison is position-for-position within
 * the same committed paint frame.
 *
 * @typedef {{x:number,y:number,w:number,h:number}} Rect
 * @typedef {{i:number,char:string,isSpace:boolean,layoutVisible:boolean,liveRect:Rect|null,refRect:Rect|null}} CharResult
 */

(() => {
  const payloadAnchor = document.querySelector("[data-worldsim-payload]");
  const referenceContainer = document.getElementById("worldsim-payload-reference");
  if (!payloadAnchor || !referenceContainer) return [];

  const textNode = payloadAnchor.firstChild;
  if (!textNode || textNode.nodeType !== Node.TEXT_NODE) return [];

  const refSpans = referenceContainer.querySelectorAll("span[data-worldsim-ref-idx]");
  /** @type {CharResult[]} */
  const results = [];

  const range = document.createRange();
  const viewportW = window.innerWidth;
  const viewportH = window.innerHeight;

  const content = textNode.textContent || "";
  for (let i = 0; i < content.length; i++) {
    const char = content[i];
    if (/\s/.test(char)) {
      results.push({
        i,
        char,
        isSpace: true,
        layoutVisible: false,
        liveRect: null,
        refRect: null,
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
    const refSpan = /** @type {HTMLElement | undefined} */ (refSpans[i]);
    const refRect = refSpan ? refSpan.getBoundingClientRect() : null;

    results.push({
      i,
      char,
      isSpace: false,
      layoutVisible,
      liveRect: layoutVisible ? serializeRect(liveRect) : null,
      refRect: refRect ? serializeRect(refRect) : null,
    });
  }

  return results;

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
})();
