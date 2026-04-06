"""Playwright scraper for collecting real website data (HTML, AXTree, screenshots).

Usage:
    python -m eval_awareness_experiments.scraper --output data/real/ --urls github,gitlab
    python -m eval_awareness_experiments.scraper --output data/real/ --url https://github.com --name github_home
"""

import argparse
import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright

from eval_awareness_experiments.types import WebsiteSample

# Predefined URL sets for common website types
PRESET_URLS = {
    "github": [
        ("github_home", "https://github.com"),
        ("github_repo", "https://github.com/torvalds/linux"),
        ("github_issues", "https://github.com/torvalds/linux/issues"),
        ("github_pulls", "https://github.com/torvalds/linux/pulls"),
    ],
    "gitlab": [
        ("gitlab_home", "https://gitlab.com"),
        ("gitlab_explore", "https://gitlab.com/explore/projects"),
    ],
    "paypal": [
        ("paypal_home", "https://www.paypal.com"),
    ],
}


async def scrape_website(
    url: str,
    output_dir: Path,
    name: str,
    viewport_width: int = 1280,
    viewport_height: int = 720,
) -> WebsiteSample:
    """Scrape a single website and save HTML, AXTree, and screenshot.

    Args:
        url: The URL to visit.
        output_dir: Directory to save files to.
        name: Base filename for saved files.
        viewport_width: Browser viewport width.
        viewport_height: Browser viewport height.

    Returns:
        WebsiteSample with paths to saved files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(
            viewport={"width": viewport_width, "height": viewport_height}
        )

        await page.goto(url, wait_until="networkidle", timeout=30000)

        # Save HTML
        html_path = output_dir / f"{name}.html"
        html = await page.content()
        html_path.write_text(html, encoding="utf-8")

        # Save accessibility tree
        axtree_path = output_dir / f"{name}_axtree.json"
        axtree = await page.accessibility.snapshot()
        axtree_path.write_text(json.dumps(axtree, indent=2, ensure_ascii=False), encoding="utf-8")

        # Save screenshot
        screenshot_path = output_dir / f"{name}.png"
        await page.screenshot(path=str(screenshot_path), full_page=False)

        await browser.close()

    sample = WebsiteSample(
        id=f"real_{name}",
        source="real",
        website_type=name.split("_")[0],
        html_path=str(html_path),
        axtree_path=str(axtree_path),
        screenshot_path=str(screenshot_path),
        metadata={"url": url, "viewport": f"{viewport_width}x{viewport_height}"},
    )

    print(f"  Scraped {url} -> {output_dir}/{name}.*")
    return sample


async def scrape_presets(presets: list[str], output_dir: Path) -> list[WebsiteSample]:
    """Scrape predefined URL sets."""
    samples = []
    for preset in presets:
        if preset not in PRESET_URLS:
            print(f"  Warning: Unknown preset '{preset}'. Available: {list(PRESET_URLS.keys())}")
            continue

        preset_dir = output_dir / preset
        for name, url in PRESET_URLS[preset]:
            try:
                sample = await scrape_website(url, preset_dir, name)
                samples.append(sample)
            except Exception as e:
                print(f"  Error scraping {url}: {e}")

    return samples


def main():
    parser = argparse.ArgumentParser(description="Scrape real websites for eval awareness experiments")
    parser.add_argument("--output", type=str, default="eval_awareness_experiments/data/real",
                        help="Output directory")
    parser.add_argument("--presets", type=str, default=None,
                        help="Comma-separated preset names (github,gitlab,paypal)")
    parser.add_argument("--url", type=str, default=None,
                        help="Single URL to scrape")
    parser.add_argument("--name", type=str, default=None,
                        help="Name for single URL scrape")
    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.url:
        name = args.name or "page"
        asyncio.run(scrape_website(args.url, output_dir, name))
    elif args.presets:
        presets = [p.strip() for p in args.presets.split(",")]
        samples = asyncio.run(scrape_presets(presets, output_dir))
        print(f"\nScraped {len(samples)} pages total.")
    else:
        # Default: scrape all presets
        presets = list(PRESET_URLS.keys())
        samples = asyncio.run(scrape_presets(presets, output_dir))
        print(f"\nScraped {len(samples)} pages total.")


if __name__ == "__main__":
    main()
