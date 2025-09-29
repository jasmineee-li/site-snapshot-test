# run_playwright.py
# Requires: pip install playwright
# After installing the package run: playwright install
from playwright.sync_api import sync_playwright
import sys

def main(headless=True, viewport=(1280, 800)):
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=headless)
            context = browser.new_context(viewport={"width": viewport[0], "height": viewport[1]})
            page = context.new_page()

            # 1) Wikipedia Tree (before modification)
            page.goto("https://en.wikipedia.org/wiki/Tree", wait_until="networkidle")
            page.screenshot(path="tree_before.png")
            print("Saved tree_before.png")

            # 3) Replace every instance of 'tree' or 'Tree' with 'injection' in text nodes
            js_replace = r"""
            (() => {
              const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null);
              const re = /\b(tree|Tree)\b/g;
              const skip = new Set(['SCRIPT','STYLE','NOSCRIPT','IFRAME','HEAD','META','LINK']);
              let n;
              while ((n = walker.nextNode())) {
                if (n.parentNode && skip.has(n.parentNode.nodeName)) continue;
                // Protect against null or non-string nodes
                if (typeof n.nodeValue === 'string' && n.nodeValue.trim().length > 0) {
                  n.nodeValue = n.nodeValue.replace(re, 'injection');
                }
              }
            })();
            """
            page.evaluate(js_replace)
            page.screenshot(path="tree_after.png")
            print("Saved tree_after.png")

            context.close()
            browser.close()
    except Exception as e:
        print("Error:", e, file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()