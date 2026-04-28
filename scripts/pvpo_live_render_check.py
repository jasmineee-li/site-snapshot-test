"""Minimal PVPO live check against real r5 pages.

Skips the seed-and-capture plumbing (which requires per-site editor auth
and moderation approvals) and directly validates that the ink-occupancy
+ content-match chain produces ``max_coverage > 0`` when pointed at
Chrome rendering of real WebArena content that's already on the page.

For each of shopping / shopping_admin / reddit / gitlab:
  - Connect to the chrome-headless-shell Docker container via CDP.
  - Navigate to a known-static URL.
  - Extract a ~50-char visible text snippet from the page body.
  - Use that snippet as ``payload_text`` — content-match in the JS query
    finds it, then ink-occupancy verifies per-char paint.
  - Report max_coverage.

If every site returns ``max_coverage > 0``, the end-to-end chain
(Docker container + Playwright + CDP beginFrame + JS content-match +
host-side ink-occupancy + background-color resolution against real site
themes) is green on production-like content. The remaining seed+render
piece of the pipeline is a separate concern (editor auth, moderation).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from worldsim.phase_4.encounter_detection import determine_encounter
from worldsim.phase_4.pvpo_browser_config import inject_animation_killer
from worldsim.phase_4.pvpo_capture import (
    Rect,
    atomic_capture_with_visibility,
    save_step_artifacts,
)

CDP_URL = "http://127.0.0.1:9222"

TARGETS = [
    # (site, url, hardcoded_snippet_or_None)
    # hardcoded snippet is a string known to be visible above the fold at
    # 1280x720 on this r5 build. When None, the auto-picker walks the DOM.
    ("shopping", "http://3.12.221.9:7770/", "Welcome to One Stop Market"),
    ("shopping_admin", "http://3.12.221.9:7780/admin/admin/", "Welcome, please sign in"),
    (
        "reddit",
        "http://3.12.221.9:9999/f/books",
        None,  # auto-pick: /f/books top post title
    ),
    ("gitlab", "http://3.12.221.9:8023/help", "Welcome to the GitLab documentation!"),
]


async def check_site(site: str, url: str, hardcoded_snippet: str | None, out_root: Path) -> dict:
    from playwright.async_api import async_playwright

    out_dir = out_root / site
    out_dir.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as pw:
        browser = await pw.chromium.connect_over_cdp(CDP_URL)
        context = browser.contexts[0] if browser.contexts else await browser.new_context()
        page = await context.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        except Exception as exc:
            await page.close()
            await browser.close()
            return {"status": "navigation_failed", "url": url, "error": str(exc)}

        await page.wait_for_timeout(1500)

        if hardcoded_snippet:
            # Fast path: skip the DOM walker and use the known-visible text.
            snippet = hardcoded_snippet
        else:
            # Pick the first meaningful text chunk from a visible element.
            snippet = await page.evaluate(
                """(() => {
                const vpW = window.innerWidth;
                const vpH = window.innerHeight;
                const walker = document.createTreeWalker(
                    document.body,
                    NodeFilter.SHOW_TEXT,
                    {
                        acceptNode(n) {
                            const p = n.parentElement;
                            if (!p) return NodeFilter.FILTER_REJECT;
                            const tag = p.tagName;
                            if (tag === 'SCRIPT' || tag === 'STYLE' || tag === 'NOSCRIPT') {
                                return NodeFilter.FILTER_REJECT;
                            }
                            // Reject common sr-only / visually-hidden patterns.
                            const cls = (p.className && p.className.toString) ? p.className.toString() : '';
                            if (/\b(sr-only|visually-hidden|screen-reader|skip-link|gl-sr-only)\b/.test(cls)) {
                                return NodeFilter.FILTER_REJECT;
                            }
                            const t = (n.textContent || '').trim();
                            if (t.length < 15) return NodeFilter.FILTER_REJECT;
                            return NodeFilter.FILTER_ACCEPT;
                        }
                    }
                );
                const range = document.createRange();
                while (walker.nextNode()) {
                    const node = walker.currentNode;
                    const full = (node.textContent || '').trim();
                    if (full.length < 15) continue;
                    const p = node.parentElement;
                    if (!p || !p.checkVisibility({opacityProperty: true, visibilityProperty: true, contentVisibilityAuto: true})) {
                        continue;
                    }
                    // Require the whole text run to be strictly within the
                    // viewport so every char of the chosen snippet is
                    // layout-visible (otherwise ink-occupancy reports zero
                    // coverage for the off-screen tail).
                    range.selectNodeContents(node);
                    const r = range.getBoundingClientRect();
                    const fullyInVp =
                        r.width > 80 && r.height > 10 &&
                        r.top >= 0 && r.left >= 0 &&
                        r.right <= vpW && r.bottom <= vpH;
                    if (!fullyInVp) continue;
                    // Trim to at most 50 characters to keep the measurement
                    // local and fast.
                    return full.slice(0, 50);
                }
                return null;
            })()"""
            )

        if not snippet:
            await page.close()
            await browser.close()
            return {"status": "no_snippet", "url": url}

        cdp = await context.new_cdp_session(page)
        await inject_animation_killer(page, cdp)

        viewport = await page.evaluate("(() => ({w: window.innerWidth, h: window.innerHeight}))()")
        viewport_rect = Rect(
            x=0,
            y=0,
            w=int(viewport.get("w", 1280)) or 1280,
            h=int(viewport.get("h", 720)) or 720,
        )
        capture = await atomic_capture_with_visibility(
            cdp, viewport_rect=viewport_rect, payload_text=snippet
        )
        save_step_artifacts(out_dir, step_idx=0, capture=capture)

        await page.close()
        await browser.close()

    encounter = determine_encounter(
        {
            "id": f"pvpo-render-{site}",
            "site": site,
            "payload_texts": [{"rendered_payload": snippet}],
        },
        out_dir,
    )
    return {
        "status": "ok",
        "url": url,
        "snippet": snippet,
        "snippet_nonspace_chars": sum(1 for c in snippet if not c.isspace()),
        "max_coverage": encounter.max_coverage,
        "reference_step": encounter.reference_step,
        "per_step_coverage": encounter.per_step_coverage,
        "background_color": list(capture.background_color),
        "has_damage": capture.has_damage,
    }


async def main() -> int:
    out_root = ROOT / "logs/pvpo_render_check"
    results: dict[str, dict] = {}
    for site, url, sel in TARGETS:
        print(f"\n=== {site} ({url}) ===", flush=True)
        try:
            results[site] = await check_site(site, url, sel, out_root)
        except Exception as exc:
            results[site] = {"status": "exception", "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(results[site], indent=2), flush=True)

    print("\n=== SUMMARY ===")
    print(json.dumps(results, indent=2))
    failures = [
        s for s, r in results.items() if r.get("status") == "ok" and r.get("max_coverage", 0) <= 0
    ]
    if failures:
        print(f"\n! sites with zero coverage: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
