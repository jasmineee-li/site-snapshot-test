"""Playwright scraper for collecting real website data (HTML, AXTree, screenshots).

Usage:
    python -m eval_awareness_experiments.scraper
    python -m eval_awareness_experiments.scraper --presets github,gitlab
    python -m eval_awareness_experiments.scraper --url https://example.com --name example_home --website-type example
"""

import argparse
import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright

DEFAULT_OUTPUT = Path("eval_awareness_experiments/data/real")
MANIFEST_PATH = Path("eval_awareness_experiments/data/manifest.json")

VIEWPORT = {"width": 1280, "height": 720}

# Predefined URL sets grouped by website type
PRESET_URLS: dict[str, list[tuple[str, str]]] = {
    "github": [
        ("home", "https://github.com"),
        ("repo", "https://github.com/torvalds/linux"),
        ("issues", "https://github.com/torvalds/linux/issues"),
        ("pulls", "https://github.com/torvalds/linux/pulls"),
    ],
    "gitlab": [
        ("home", "https://gitlab.com"),
        ("explore", "https://gitlab.com/explore/projects"),
    ],
    "paypal": [
        ("home", "https://www.paypal.com"),
    ],
    "gmail": [
        ("home", "https://mail.google.com"),
    ],
    "rocketchat": [
        ("home", "https://open.rocket.chat"),
    ],
}


async def scrape_page(
    page,
    output_dir: Path,
    page_name: str,
) -> bool:
    """Capture HTML, axtree, and screenshot for current page state.

    Returns True on success.
    """
    try:
        # HTML
        html_path = output_dir / f"{page_name}.html"
        html = await page.content()
        html_path.write_text(html, encoding="utf-8")

        # Accessibility tree
        axtree_path = output_dir / f"{page_name}_axtree.txt"
        axtree = await page.locator(":root").aria_snapshot()
        axtree_path.write_text(axtree, encoding="utf-8")

        # Screenshot
        screenshot_path = output_dir / f"{page_name}.png"
        await page.screenshot(path=str(screenshot_path), full_page=False)

        return True
    except Exception as e:
        print(f"    Error capturing {page_name}: {e}")
        return False


async def scrape_preset(
    preset_name: str,
    urls: list[tuple[str, str]],
    output_dir: Path,
) -> list[str]:
    """Scrape all URLs for a preset. Returns list of captured page names."""
    preset_dir = output_dir / preset_name
    preset_dir.mkdir(parents=True, exist_ok=True)
    captured = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport=VIEWPORT)

        for i, (page_name, url) in enumerate(urls):
            if i > 0:
                await asyncio.sleep(5)  # space out requests
            try:
                await page.goto(url, wait_until="load", timeout=30000)
                await asyncio.sleep(2)  # let JS settle
                ok = await scrape_page(page, preset_dir, page_name)
                if ok:
                    captured.append(page_name)
                    print(f"  {preset_name}/{page_name} ✓")
                else:
                    print(f"  {preset_name}/{page_name} FAILED")
            except Exception as e:
                print(f"  {preset_name}/{page_name} FAILED: {e}")

        await browser.close()

    return captured


async def scrape_single(
    url: str,
    name: str,
    website_type: str,
    output_dir: Path,
) -> list[str]:
    """Scrape a single URL. Returns list of captured page names."""
    type_dir = output_dir / website_type
    type_dir.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport=VIEWPORT)

        try:
            await page.goto(url, wait_until="load", timeout=30000)
            await asyncio.sleep(2)
            ok = await scrape_page(page, type_dir, name)
            if ok:
                print(f"  {website_type}/{name} ✓")
                await browser.close()
                return [name]
        except Exception as e:
            print(f"  {website_type}/{name} FAILED: {e}")

        await browser.close()
        return []


def update_manifest(
    pages_by_type: dict[str, list[str]],
    output_dir: Path,
    manifest_path: Path,
) -> None:
    """Add real website entries to the manifest."""
    existing = []
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())

    # Remove old real entries
    existing = [e for e in existing if e.get("source") != "real"]

    for website_type, pages in sorted(pages_by_type.items()):
        entry = {
            "id": f"real_{website_type}",
            "source": "real",
            "label": "real",
            "path": str(output_dir / website_type),
            "pages": pages,
            "metadata": {
                "website_type": website_type,
            },
        }
        existing.append(entry)

    manifest_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"\nManifest updated: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Scrape real websites for eval awareness experiments"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output directory",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(MANIFEST_PATH),
        help="Manifest file path",
    )
    parser.add_argument(
        "--presets",
        type=str,
        default=None,
        help="Comma-separated preset names (github,gitlab,paypal,gmail,rocketchat)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Single URL to scrape",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="page",
        help="Page name for single URL scrape",
    )
    parser.add_argument(
        "--website-type",
        type=str,
        default="other",
        help="Website type for single URL scrape",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    pages_by_type: dict[str, list[str]] = {}

    if args.url:
        pages = asyncio.run(scrape_single(args.url, args.name, args.website_type, output_dir))
        if pages:
            pages_by_type[args.website_type] = pages
    else:
        presets = (
            [p.strip() for p in args.presets.split(",")]
            if args.presets
            else list(PRESET_URLS.keys())
        )

        for preset in presets:
            if preset not in PRESET_URLS:
                print(
                    f"  Warning: Unknown preset '{preset}'. Available: {list(PRESET_URLS.keys())}"
                )
                continue

            print(f"\n{preset}:")
            pages = asyncio.run(scrape_preset(preset, PRESET_URLS[preset], output_dir))
            if pages:
                pages_by_type[preset] = pages

    total = sum(len(p) for p in pages_by_type.values())
    if total > 0:
        update_manifest(pages_by_type, output_dir, Path(args.manifest))

    print(f"\nDone. Scraped {total} pages from {len(pages_by_type)} sites.")


if __name__ == "__main__":
    main()
