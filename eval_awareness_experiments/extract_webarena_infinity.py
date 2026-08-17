"""Extract website data from webarena-infinity apps.

Starts each app's server, visits each view via Playwright, and captures
HTML, accessibility tree, and screenshot into the data directory.

Usage:
    python -m eval_awareness_experiments.extract_webarena_infinity
    python -m eval_awareness_experiments.extract_webarena_infinity --apps gmail,paypal-my-wallet
"""

import argparse
import asyncio
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

from playwright.async_api import ViewportSize, async_playwright

APPS_DIR = Path("external_benchmarks/webarena-infinity/apps")
DEFAULT_OUTPUT = Path("eval_awareness_experiments/data/webarena-infinity")
MANIFEST_PATH = Path("eval_awareness_experiments/data/manifest.json")

# Annotated with Playwright's TypedDict so the constant satisfies the
# `viewport=` parameter; a bare dict literal infers as `dict[str, int]`.
VIEWPORT: ViewportSize = {"width": 1280, "height": 720}

# Each app maps to a list of (page_name, navigation_action) tuples.
# navigation_action is either:
#   - a string starting with "#" → hash route (navigate via URL)
#   - a string starting with "click:" → CSS selector to click
APP_PAGES: dict[str, list[tuple[str, str]]] = {
    "gmail": [
        ("inbox", "#/inbox"),
        ("starred", "#/starred"),
        ("sent", "#/sent"),
        ("drafts", "#/drafts"),
        ("spam", "#/spam"),
        ("trash", "#/trash"),
        ("important", "#/important"),
        ("allmail", "#/allmail"),
        ("settings", "#/settings"),
    ],
    "gmail-accounts-and-contacts": [
        ("contacts", "click:[data-view='contacts']"),
        ("other-contacts", "click:[data-view='other-contacts']"),
        ("merge-suggestions", "click:[data-view='merge-suggestions']"),
        ("delegates", "click:[data-view='delegates']"),
        ("linked-services", "click:[data-view='linked-services']"),
        ("import-export", "click:[data-view='import-export']"),
        ("settings", "click:[data-view='settings']"),
    ],
    "gitlab-plan-and-track": [
        ("issues", "#/issues"),
        ("boards", "#/boards"),
        ("epics", "#/epics"),
        ("milestones", "#/milestones"),
        ("iterations", "#/iterations"),
        ("labels", "#/labels"),
        ("todos", "#/todos"),
    ],
    "paypal-my-wallet": [
        ("overview", "click:[data-view='overview']"),
        ("transactions", "click:[data-view='transactions']"),
        ("cards", "click:[data-view='cards']"),
        ("banks", "click:[data-view='banks']"),
        ("balance", "click:[data-view='balance']"),
        ("crypto", "click:[data-view='crypto']"),
        ("rewards", "click:[data-view='rewards']"),
        ("gift-cards", "click:[data-view='gift-cards']"),
        ("preferences", "click:[data-view='preferences']"),
    ],
    "elation-clinical-records": [
        ("patients", "#/patients"),
        ("templates", "#/templates"),
        ("categories", "#/categories"),
        ("settings", "#/settings"),
    ],
    "elation-patient-communication": [
        ("home", "#/home"),
        ("inbox", "#/inbox"),
        ("sent", "#/sent"),
        ("drafts", "#/drafts"),
        ("reminders", "#/reminders"),
        ("patients", "#/patients"),
        ("appointments", "#/appointments"),
        ("bulk-letters", "#/bulk-letters"),
        ("settings", "#/settings"),
    ],
    "elation-prescriptions": [
        ("chart", "#/chart"),
        ("med-history", "#/med-history"),
        ("rx-requests", "#/rx-requests"),
        ("settings", "#/settings"),
    ],
    "figma-slides": [
        ("slides", "click:[data-action='setView'][data-view='slides']"),
    ],
    "figma-text-and-typography": [
        ("layers", "#/layers"),
        ("properties", "#/properties"),
        ("styles", "#/styles"),
        ("fonts", "#/fonts"),
        ("settings", "#/settings"),
    ],
    "handshake-career-exploration": [
        ("feed", "#/feed"),
        ("jobs", "#/jobs"),
        ("employers", "#/employers"),
        ("events", "#/events"),
        ("messages", "#/messages"),
        ("career-center", "#/career-center"),
        ("qa", "#/qa"),
        ("profile", "#/profile"),
        ("career-interests", "#/career-interests"),
    ],
    "linear-account-settings": [
        ("profile", "#/profile"),
        ("preferences", "#/preferences"),
        ("notifications", "#/notifications"),
        ("security", "#/security"),
    ],
    "superhuman-general": [
        ("inbox", "#/inbox"),
        ("done", "#/done"),
        ("reminders", "#/reminders"),
        ("sent", "#/sent"),
        ("snippets", "#/snippets"),
        ("opens", "#/opens"),
    ],
    "xero-invoicing": [
        ("dashboard", "#/dashboard"),
        ("invoices", "#/invoices"),
        ("quotes", "#/quotes"),
        ("credit-notes", "#/credit-notes"),
        ("repeating-invoices", "#/repeating-invoices"),
        ("reminders", "#/reminders"),
        ("templates", "#/templates"),
    ],
}


def start_server(app_name: str, port: int) -> subprocess.Popen:
    """Start an app server and wait for it to be ready."""
    app_dir = APPS_DIR / app_name
    proc = subprocess.Popen(
        [sys.executable, "server.py", "--port", str(port)],
        cwd=str(app_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Give the server a moment to start
    time.sleep(1)
    if proc.poll() is not None:
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        raise RuntimeError(f"Server for {app_name} failed to start: {stderr}")
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    """Stop a server process."""
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


async def capture_page(
    page,
    output_dir: Path,
    page_name: str,
) -> dict:
    """Capture HTML, axtree, and screenshot for the current page state."""
    # Wait for any rendering to settle
    await asyncio.sleep(0.5)

    # HTML
    html_path = output_dir / f"{page_name}.html"
    html = await page.content()
    html_path.write_text(html, encoding="utf-8")

    # Accessibility tree (ARIA snapshot)
    axtree_path = output_dir / f"{page_name}_axtree.txt"
    axtree = await page.locator(":root").aria_snapshot()
    axtree_path.write_text(axtree, encoding="utf-8")

    # Screenshot
    screenshot_path = output_dir / f"{page_name}.png"
    await page.screenshot(path=str(screenshot_path), full_page=False)

    return {
        "name": page_name,
        "html": str(html_path),
        "axtree": str(axtree_path),
        "screenshot": str(screenshot_path),
    }


async def extract_app(
    app_name: str,
    port: int,
    output_dir: Path,
) -> dict | None:
    """Extract all pages from one app. Returns a manifest entry."""
    pages_config = APP_PAGES.get(app_name)
    if not pages_config:
        print(f"  No page config for {app_name}, skipping")
        return None

    app_output = output_dir / app_name
    app_output.mkdir(parents=True, exist_ok=True)

    base_url = f"http://localhost:{port}"
    proc = start_server(app_name, port)
    captured_pages = []

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page(viewport=VIEWPORT)

            # Initial load — use "load" instead of "networkidle" since SSE
            # connections keep the network active indefinitely
            await page.goto(base_url, wait_until="load", timeout=15000)
            await asyncio.sleep(2)  # let JS initialize and push seed state

            for page_name, nav_action in pages_config:
                try:
                    if nav_action.startswith("#"):
                        await page.goto(
                            f"{base_url}{nav_action}",
                            wait_until="load",
                            timeout=10000,
                        )
                    elif nav_action.startswith("click:"):
                        selector = nav_action[len("click:") :]
                        await page.click(selector, timeout=5000)

                    # Let the view render
                    await asyncio.sleep(0.5)

                    await capture_page(page, app_output, page_name)
                    captured_pages.append(page_name)
                    print(f"  {app_name}/{page_name} ✓")
                except Exception as e:
                    print(f"  {app_name}/{page_name} FAILED: {e}")

            await browser.close()
    finally:
        stop_server(proc)

    if not captured_pages:
        return None

    return {
        "id": f"webarena-infinity_{app_name}",
        "source": "webarena-infinity",
        "label": "synthetic",
        "path": str(app_output),
        "pages": captured_pages,
        "metadata": {"benchmark": "webarena-infinity", "app": app_name},
    }


async def run(app_names: list[str], output_dir: Path) -> list[dict]:
    """Extract data from all specified apps."""
    entries = []
    base_port = 9100  # avoid clashing with common dev ports

    for i, app_name in enumerate(app_names):
        port = base_port + i
        print(f"\nExtracting {app_name} (port {port})...")
        entry = await extract_app(app_name, port, output_dir)
        if entry:
            entries.append(entry)

    return entries


def update_manifest(entries: list[dict], manifest_path: Path) -> None:
    """Add or update entries in the manifest."""
    existing = []
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())

    # Remove only the entries being replaced (by ID)
    new_ids = {e["id"] for e in entries}
    existing = [e for e in existing if e["id"] not in new_ids]
    existing.extend(entries)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"\nManifest updated: {manifest_path} ({len(existing)} total entries)")


def main():
    parser = argparse.ArgumentParser(description="Extract website data from webarena-infinity apps")
    parser.add_argument(
        "--apps",
        type=str,
        default=",".join(APP_PAGES.keys()),
        help="Comma-separated app names to extract",
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
    args = parser.parse_args()

    app_names = [a.strip() for a in args.apps.split(",")]
    output_dir = Path(args.output)

    # Validate apps exist
    for name in app_names:
        app_dir = APPS_DIR / name
        if not app_dir.exists():
            print(f"Error: App not found: {app_dir}")
            return
        if name not in APP_PAGES:
            print(f"Error: No page config for {name}. Available: {list(APP_PAGES.keys())}")
            return

    entries = asyncio.run(run(app_names, output_dir))
    if entries:
        update_manifest(entries, Path(args.manifest))

    print(
        f"\nDone. Extracted {sum(len(e['pages']) for e in entries)} pages from {len(entries)} apps."
    )


if __name__ == "__main__":
    main()
