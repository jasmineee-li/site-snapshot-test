"""Scrape real authenticated + public websites into the standardized data format.

Produces the same (HTML, screenshot, axtree) tuple as extract_tac.py / scraper.py,
but supports authenticated SaaS via a manual-login-once + reuse-storage_state pattern.

Usage:
    # One-time: log in manually for an auth-required site (headed browser)
    python -m eval_awareness_experiments.scrape_real_sites --login gmail

    # Capture one site (uses saved auth if required)
    python -m eval_awareness_experiments.scrape_real_sites --capture wikipedia
    python -m eval_awareness_experiments.scrape_real_sites --capture gmail

    # Capture all sites (skips auth-required sites missing _auth file with a warning)
    python -m eval_awareness_experiments.scrape_real_sites --capture-all
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright

from eval_awareness_experiments.extract_tac import capture_page

DEFAULT_OUTPUT = Path("eval_awareness_experiments/data/real")
MANIFEST_PATH = Path("eval_awareness_experiments/data/manifest.json")
AUTH_DIR = DEFAULT_OUTPUT / "_auth"

VIEWPORT = {"width": 1280, "height": 720}

# Hand-curated page lists per site. Paths are joined against base_url.
# Add/adjust freely — the shape matches extract_tac.SERVICES.
SITES: dict[str, dict] = {
    # --- auth-required SaaS ---
    "gmail": {
        "requires_auth": True,
        "login_url": "https://accounts.google.com/ServiceLogin?service=mail",
        "base_url": "https://mail.google.com",
        "login_markers": ["accounts.google.com/signin", "ServiceLogin"],
        "pages": [
            ("inbox", "/mail/u/0/#inbox"),
            ("starred", "/mail/u/0/#starred"),
            ("sent", "/mail/u/0/#sent"),
            ("drafts", "/mail/u/0/#drafts"),
            ("settings", "/mail/u/0/#settings/general"),
        ],
    },
    "superhuman": {
        "requires_auth": True,
        "login_url": "https://mail.superhuman.com/",
        "base_url": "https://mail.superhuman.com",
        "login_markers": ["/login", "sign-in"],
        "pages": [
            ("inbox", "/"),
        ],
    },
    "paypal": {
        "requires_auth": True,
        "login_url": "https://www.paypal.com/signin",
        "base_url": "https://www.paypal.com",
        "login_markers": ["/signin", "/connect/"],
        "pages": [
            ("summary", "/myaccount/summary"),
            ("activity", "/myaccount/activities"),
            ("wallet", "/myaccount/money"),
        ],
    },
    "xero": {
        "requires_auth": True,
        "login_url": "https://login.xero.com/",
        "base_url": "https://go.xero.com",
        "login_markers": ["login.xero.com", "/identity/"],
        "pages": [
            ("dashboard", "/Dashboard/"),
            ("bank-accounts", "/Bank/BankAccounts.aspx"),
            ("contacts", "/Contacts/"),
        ],
    },
    "figma": {
        "requires_auth": True,
        "login_url": "https://www.figma.com/login",
        "base_url": "https://www.figma.com",
        "login_markers": ["/login", "/session"],
        "pages": [
            ("files-recent", "/files/recent"),
            ("files-drafts", "/files/drafts"),
            ("community", "/community"),
        ],
    },
    "linear": {
        "requires_auth": True,
        "login_url": "https://linear.app/login",
        "base_url": "https://linear.app",
        "login_markers": ["/login", "/magic-link"],
        # User must navigate to their workspace once during --login; we save the URL they land on
        # as the "home" via storage_state. These paths assume a workspace slug — user may need to edit.
        "pages": [
            ("inbox", "/inbox"),
            ("my-issues", "/my-issues"),
        ],
    },
    "handshake": {
        "requires_auth": True,
        "login_url": "https://app.joinhandshake.com/login",
        "base_url": "https://app.joinhandshake.com",
        "login_markers": ["/login", "/auth/"],
        "pages": [
            ("home", "/stu"),
            ("jobs", "/stu/postings"),
            ("events", "/stu/events"),
        ],
    },
    "elation": {
        "requires_auth": True,
        "login_url": "https://www.elationhealth.com/login/",
        "base_url": "https://app.elationemr.com",
        "login_markers": ["/login", "/accounts/login"],
        "pages": [
            ("dashboard", "/"),
        ],
    },
    # --- public / no auth required ---
    "gitlab": {
        "requires_auth": False,
        "base_url": "https://gitlab.com",
        "pages": [
            ("home", "/"),
            ("explore", "/explore/projects"),
            ("explore-topics", "/explore/projects/topics"),
        ],
    },
    "reddit": {
        "requires_auth": False,
        "base_url": "https://www.reddit.com",
        "pages": [
            ("home", "/"),
            ("popular", "/r/popular/"),
            ("all", "/r/all/"),
        ],
    },
    "craigslist": {
        "requires_auth": False,
        "base_url": "https://www.craigslist.org",
        "pages": [
            ("home", "/"),
            ("sfbay", "https://sfbay.craigslist.org/"),
            ("sfbay-apa", "https://sfbay.craigslist.org/search/apa"),
        ],
    },
    "wikipedia": {
        "requires_auth": False,
        "base_url": "https://en.wikipedia.org",
        "pages": [
            ("main", "/wiki/Main_Page"),
            ("featured", "/wiki/Wikipedia:Featured_articles"),
            ("random-trout", "/wiki/Trout"),
        ],
    },
    "osm": {
        "requires_auth": False,
        "base_url": "https://www.openstreetmap.org",
        "pages": [
            ("home", "/"),
            ("about", "/about"),
            ("export", "/export"),
        ],
    },
}


def _resolve_url(base_url: str, path: str) -> str:
    """Join base_url and path, unless path is already absolute."""
    if path.startswith("http://") or path.startswith("https://"):
        return path
    return f"{base_url}{path}"


def _auth_path(site: str) -> Path:
    return AUTH_DIR / f"{site}.json"


def _is_login_page(current_url: str, markers: list[str]) -> bool:
    return any(marker in current_url for marker in markers)


async def login_flow(site: str, config: dict) -> None:
    """Headed browser: user logs in manually, we save storage_state."""
    AUTH_DIR.mkdir(parents=True, exist_ok=True)
    auth_file = _auth_path(site)

    login_url = config.get("login_url") or config["base_url"]
    print(f"\nLaunching headed browser for {site}...")
    print(f"  Target: {login_url}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(viewport=VIEWPORT)
        page = await context.new_page()
        await page.goto(login_url, wait_until="load", timeout=60000)

        print("\n" + "=" * 60)
        print(f"  Log in to {site} in the browser window.")
        print("  When you're fully logged in (on the app, not the login page),")
        print("  come back here and press Enter to save the session.")
        print("=" * 60)
        await asyncio.get_event_loop().run_in_executor(None, input, "")

        await context.storage_state(path=str(auth_file))
        await browser.close()

    print(f"  Saved auth -> {auth_file}")


async def capture_site(site: str, config: dict, output_dir: Path) -> dict | None:
    """Capture all configured pages for one site. Returns a manifest entry or None."""
    site_dir = output_dir / site
    site_dir.mkdir(parents=True, exist_ok=True)

    requires_auth = config.get("requires_auth", False)
    storage_state = None
    if requires_auth:
        auth_file = _auth_path(site)
        if not auth_file.exists():
            print(f"  {site}: SKIP — auth required but no storage_state at {auth_file}")
            print(f"         Run: python -m eval_awareness_experiments.scrape_real_sites --login {site}")
            return None
        storage_state = str(auth_file)

    captured: list[str] = []
    base_url = config["base_url"]
    login_markers = config.get("login_markers", [])

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport=VIEWPORT,
            storage_state=storage_state,
            ignore_https_errors=True,
        )
        page = await context.new_page()

        for page_name, path in config["pages"]:
            url = _resolve_url(base_url, path)
            try:
                response = await page.goto(url, wait_until="load", timeout=45000)
                await asyncio.sleep(2)  # let JS settle

                status = response.status if response else 0
                current_url = page.url

                # Hard-fail on auth expiry — no silent captures of login pages
                if requires_auth and _is_login_page(current_url, login_markers):
                    raise RuntimeError(
                        f"auth expired for {site}: landed on {current_url}. "
                        f"Re-run --login {site}."
                    )

                if status >= 400:
                    print(f"  {site}/{page_name} FAILED status={status}")
                    continue

                await capture_page(page, site_dir, page_name)
                captured.append(page_name)
                print(f"  {site}/{page_name} ✓")
            except Exception as e:
                print(f"  {site}/{page_name} FAILED: {e}")

        await browser.close()

    if not captured:
        return None

    return {
        "id": f"real_{site}",
        "source": "real",
        "label": "real",
        "path": str(site_dir),
        "pages": captured,
        "metadata": {
            "requires_auth": requires_auth,
            "base_url": base_url,
        },
    }


def update_manifest(entries: list[dict], manifest_path: Path) -> None:
    """Replace existing real_* entries with the new ones, preserve everything else."""
    existing: list[dict] = []
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())

    new_ids = {e["id"] for e in entries}
    # Keep non-real entries AND real entries we didn't just re-capture
    preserved = [
        e for e in existing
        if e.get("source") != "real" or e.get("id") not in new_ids
    ]
    preserved.extend(entries)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(preserved, indent=2), encoding="utf-8")
    print(f"\nManifest updated: {manifest_path} ({len(preserved)} total entries)")


async def run_capture(site_names: list[str], output_dir: Path) -> list[dict]:
    entries: list[dict] = []
    for site in site_names:
        config = SITES.get(site)
        if not config:
            print(f"Unknown site: {site}. Known: {sorted(SITES.keys())}")
            continue
        print(f"\nCapturing {site}...")
        entry = await capture_site(site, config, output_dir)
        if entry:
            entries.append(entry)
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--login", type=str, help="Headed login flow for one site")
    group.add_argument("--capture", type=str, help="Capture one site (comma-separated OK)")
    group.add_argument("--capture-all", action="store_true", help="Capture every configured site")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--manifest", type=str, default=str(MANIFEST_PATH))
    args = parser.parse_args()

    if args.login:
        config = SITES.get(args.login)
        if not config:
            raise SystemExit(f"Unknown site: {args.login}. Known: {sorted(SITES.keys())}")
        if not config.get("requires_auth"):
            raise SystemExit(f"Site '{args.login}' has requires_auth=False; --login not needed.")
        asyncio.run(login_flow(args.login, config))
        return

    if args.capture:
        site_names = [s.strip() for s in args.capture.split(",") if s.strip()]
    else:
        site_names = list(SITES.keys())

    output_dir = Path(args.output)
    entries = asyncio.run(run_capture(site_names, output_dir))
    if entries:
        update_manifest(entries, Path(args.manifest))

    total_pages = sum(len(e["pages"]) for e in entries)
    print(f"\nDone. Captured {total_pages} pages from {len(entries)} sites.")


if __name__ == "__main__":
    main()
