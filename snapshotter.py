# snapshotter.py (HAR-powered)
# Usage: python snapshotter.py https://example.com snapshots/run003
#
# deps: pip install playwright
# after install: playwright install

import os, sys, json, base64, hashlib
from urllib.parse import urlparse
from pathlib import Path
from playwright.sync_api import sync_playwright


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def safe_name_from_url(url: str) -> str:
    parsed = urlparse(url)
    # hash to avoid super-long names / querystrings
    h = hashlib.sha1(url.encode()).hexdigest()
    # try to keep extension if present
    path = parsed.path
    ext = os.path.splitext(path)[1]
    if ext and len(ext) <= 8:
        return f"{h}{ext}"
    return h


def snapshot(url: str, out_dir: str):
    out = Path(out_dir)
    ensure_dir(out)
    assets_dir = out / "assets"
    ensure_dir(assets_dir)

    with sync_playwright() as pw:
        # record HAR with embedded bodies so we don't have to re-fetch
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(
            record_har_path=str(out / "page.har"), record_har_content="embed"
        )
        page = context.new_page()
        page.goto(url, wait_until="networkidle")

        # Save DOM after JS executed
        (out / "index.html").write_text(page.content(), encoding="utf-8")
        page.screenshot(path=str(out / "screenshot.png"), full_page=True)

        # Save cookies/storage
        meta = {
            "url": url,
            "cookies": context.cookies(),
            "localStorage": page.evaluate(
                "() => Object.fromEntries(Object.entries(localStorage))"
            ),
            "sessionStorage": page.evaluate(
                "() => Object.fromEntries(Object.entries(sessionStorage))"
            ),
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # Finalize HAR
        context.close()
        browser.close()

    # Build URL -> asset map by extracting embedded bodies from HAR
    har_path = out / "page.har"
    har = json.loads(har_path.read_text(encoding="utf-8"))

    # Chrome-style HAR: entries under log.entries
    entries = har.get("log", {}).get("entries", [])
    url_to_asset = {}

    for e in entries:
        req = e.get("request", {})
        res = e.get("response", {})
        url0 = req.get("url")
        if not url0:
            continue
        # Only map typical static asset types; skip HTML document itself to avoid conflicts
        mime = (res.get("content", {}) or {}).get("mimeType", "")
        status = res.get("status", 0)

        if status != 200:
            continue

        # Heuristic: map common static types
        if any(
            mime.startswith(t)
            for t in [
                "image/",
                "font/",
                "text/css",
                "application/javascript",
                "text/javascript",
            ]
        ):
            content = res.get("content", {})
            text = content.get("text")
            enc = content.get("encoding")
            if not text:
                continue
            try:
                raw = (
                    base64.b64decode(text) if enc == "base64" else text.encode("utf-8")
                )
            except Exception:
                continue
            fname = safe_name_from_url(url0)
            fpath = assets_dir / fname
            fpath.write_bytes(raw)
            # map to web path served by our snapshot server
            url_to_asset[url0] = f"/assets/{fname}"

    # Persist the mapping for the rewriter to use
    (out / "url_to_asset.json").write_text(
        json.dumps(url_to_asset, indent=2), encoding="utf-8"
    )
    print(f"Snapshot complete at {out_dir}")
    print(f"- Extracted {len(url_to_asset)} assets with URL→local mapping.")
    print(
        f"- Files: index.html, page.har, url_to_asset.json, assets/, meta.json, screenshot.png"
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python snapshotter.py <url> <out_dir>")
        sys.exit(1)
    snapshot(sys.argv[1], sys.argv[2])
