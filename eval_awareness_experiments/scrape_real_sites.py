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

# Override UA so headless Chrome doesn't advertise "HeadlessChrome" — some sites
# (notably figma.com/community) 403 requests with that substring in the UA.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

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
            ("snoozed", "/mail/u/0/#snoozed"),
            ("sent", "/mail/u/0/#sent"),
            ("drafts", "/mail/u/0/#drafts"),
            ("important", "/mail/u/0/#imp"),
            ("chats", "/mail/u/0/#chats"),
            ("all", "/mail/u/0/#all"),
            ("spam", "/mail/u/0/#spam"),
            ("trash", "/mail/u/0/#trash"),
            ("scheduled", "/mail/u/0/#scheduled"),
            ("category-social", "/mail/u/0/#category/social"),
            ("category-updates", "/mail/u/0/#category/updates"),
            ("category-promotions", "/mail/u/0/#category/promotions"),
            ("category-forums", "/mail/u/0/#category/forums"),
            ("settings-general", "/mail/u/0/#settings/general"),
            ("settings-labels", "/mail/u/0/#settings/labels"),
            ("settings-inbox", "/mail/u/0/#settings/inbox"),
            ("settings-accounts", "/mail/u/0/#settings/accounts"),
            ("settings-filters", "/mail/u/0/#settings/filters"),
            ("compose", "/mail/u/0/#inbox?compose=new"),
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
        # Public marketing site only — the logged-in accounting app is too sensitive to scrape.
        "requires_auth": False,
        "base_url": "https://www.xero.com",
        "pages": [
            ("home", "/us/"),
            ("pricing", "/us/pricing-plans/"),
            ("features-accounting", "/us/features/accounting/"),
            ("features-invoicing", "/us/features/online-invoicing/"),
            ("features-payroll", "/us/features/payroll/"),
            ("features-reporting", "/us/features/reporting/"),
            ("features-bank-connections", "/us/features/bank-connections/"),
            ("features-expenses", "/us/features/expenses/"),
            ("features-inventory", "/us/features/inventory/"),
            ("features-projects", "/us/features/projects/"),
            ("app-marketplace", "/us/marketplace/"),
            ("advisors", "/us/advisors/"),
            ("resources", "/us/resources/"),
            ("about", "/us/about/"),
            ("contact", "/us/contact/"),
            ("blog", "/blog/us/"),
            ("security", "/us/security/"),
            ("support", "/us/support/"),
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
            ("files-shared", "/files/shared"),
            ("files-deleted", "/files/trash"),
            ("files-all", "/files"),
            ("community", "/community"),
            ("community-files", "/community/explore"),
            ("community-plugins", "/community/plugins"),
            ("community-widgets", "/community/widgets"),
            ("community-libraries", "/community/libraries"),
            ("settings", "/settings"),
            ("settings-account", "/settings/account"),
            ("settings-notifications", "/settings/notifications"),
            ("settings-security", "/settings/security"),
        ],
    },
    "linear": {
        "requires_auth": True,
        "login_url": "https://linear.app/login",
        "base_url": "https://linear.app",
        "login_markers": ["/login", "/magic-link"],
        # Linear is workspace-scoped. Using zhoukristina9@gmail.com's workspace: kristys-team.
        # Team-scoped pages (e.g. /team/{KEY}/...) need a team key we don't know up front,
        # so we skip those and capture workspace-level + personal + settings pages.
        "pages": [
            ("workspace-home", "/kristys-team/"),
            ("inbox", "/kristys-team/inbox"),
            ("my-issues", "/kristys-team/my-issues"),
            ("my-issues-active", "/kristys-team/my-issues/active"),
            ("my-issues-backlog", "/kristys-team/my-issues/backlog"),
            ("my-issues-created", "/kristys-team/my-issues/created"),
            ("my-issues-subscribed", "/kristys-team/my-issues/subscribed"),
            ("projects", "/kristys-team/projects"),
            ("initiatives", "/kristys-team/initiatives"),
            ("views", "/kristys-team/views"),
            ("labels", "/kristys-team/labels"),
            ("members", "/kristys-team/members"),
            ("roadmap", "/kristys-team/roadmap"),
            ("cycles", "/kristys-team/cycles"),
            ("templates", "/kristys-team/templates"),
            ("settings-workspace", "/kristys-team/settings"),
            ("settings-workspace-members", "/kristys-team/settings/members"),
            ("settings-workspace-teams", "/kristys-team/settings/teams"),
            ("settings-workspace-integrations", "/kristys-team/settings/integrations"),
            ("settings-workspace-api", "/kristys-team/settings/api"),
            ("settings-account", "/settings/account"),
            ("settings-profile", "/settings/profile"),
            ("settings-notifications", "/settings/account/notifications"),
            ("settings-preferences", "/settings/account/preferences"),
            ("settings-security", "/settings/account/security"),
        ],
    },
    "handshake": {
        # Public marketing site only — logged-in student portal skipped.
        "requires_auth": False,
        "base_url": "https://joinhandshake.com",
        "pages": [
            ("home", "/"),
            ("students", "/students/"),
            ("employers", "/employers/"),
            ("career-centers", "/career-centers/"),
            ("pricing-employers", "/employers/pricing/"),
            ("products-students", "/students/features/"),
            ("products-employers", "/employers/features/"),
            ("events", "/events/"),
            ("blog", "/blog/"),
            ("about", "/about/"),
            ("careers", "/careers/"),
            ("press", "/press/"),
            ("contact", "/contact/"),
            ("privacy", "/privacy-policy/"),
            ("terms", "/terms-of-service/"),
        ],
    },
    "elation": {
        # Public marketing site only — real EHR is PHI, too sensitive to scrape.
        "requires_auth": False,
        "base_url": "https://www.elationhealth.com",
        "pages": [
            ("home", "/"),
            ("product", "/product/"),
            ("ehr", "/product/ehr/"),
            ("billing", "/product/billing/"),
            ("patient-portal", "/product/patient-experience/"),
            ("telehealth", "/product/telehealth/"),
            ("integrations", "/integrations/"),
            ("pricing", "/pricing/"),
            ("resources", "/resources/"),
            ("blog", "/blog/"),
            ("customers", "/customers/"),
            ("about", "/about/"),
            ("contact", "/contact-us/"),
            ("security", "/security/"),
        ],
    },
    # --- public / no auth required ---
    "gitlab": {
        "requires_auth": False,
        "base_url": "https://gitlab.com",
        "pages": [
            ("home", "/"),
            ("explore", "/explore/projects"),
            ("explore-trending", "/explore/projects/trending"),
            ("explore-starred", "/explore/projects/starred"),
            ("explore-topics", "/explore/projects/topics"),
            ("explore-snippets", "/explore/snippets"),
            ("sign-in", "/users/sign_in"),
            ("pricing", "/pricing/"),
            ("features", "/features/"),
            ("help", "/help"),
            ("groups", "/explore/groups"),
        ],
    },
    "reddit": {
        "requires_auth": False,
        "base_url": "https://www.reddit.com",
        "pages": [
            ("home", "/"),
            ("popular", "/r/popular/"),
            ("all", "/r/all/"),
            ("news", "/r/news/"),
            ("worldnews", "/r/worldnews/"),
            ("askreddit", "/r/AskReddit/"),
            ("funny", "/r/funny/"),
            ("pics", "/r/pics/"),
            ("science", "/r/science/"),
            ("technology", "/r/technology/"),
            ("programming", "/r/programming/"),
            ("machinelearning", "/r/MachineLearning/"),
            ("flyfishing", "/r/Flyfishing/"),
            ("topics", "/topics/a-1/"),
        ],
    },
    "craigslist": {
        "requires_auth": False,
        "base_url": "https://www.craigslist.org",
        "pages": [
            ("home", "/"),
            ("sites", "/about/sites"),
            ("sfbay", "https://sfbay.craigslist.org/"),
            ("sfbay-apa", "https://sfbay.craigslist.org/search/apa"),
            ("sfbay-jobs", "https://sfbay.craigslist.org/search/jjj"),
            ("sfbay-for-sale", "https://sfbay.craigslist.org/search/sss"),
            ("sfbay-cars", "https://sfbay.craigslist.org/search/cta"),
            ("sfbay-free", "https://sfbay.craigslist.org/search/zip"),
            ("sfbay-services", "https://sfbay.craigslist.org/search/bbb"),
            ("sfbay-community", "https://sfbay.craigslist.org/search/ccc"),
            ("sfbay-events", "https://sfbay.craigslist.org/search/eee"),
            ("nyc-apa", "https://newyork.craigslist.org/search/apa"),
        ],
    },
    "wikipedia": {
        "requires_auth": False,
        "base_url": "https://en.wikipedia.org",
        "pages": [
            ("main", "/wiki/Main_Page"),
            ("featured", "/wiki/Wikipedia:Featured_articles"),
            ("current-events", "/wiki/Portal:Current_events"),
            ("random", "/wiki/Special:Random"),
            ("random-trout", "/wiki/Trout"),
            ("article-python", "/wiki/Python_(programming_language)"),
            ("article-oss", "/wiki/Open-source_software"),
            ("article-ai", "/wiki/Artificial_intelligence"),
            ("contents", "/wiki/Wikipedia:Contents"),
            ("about", "/wiki/Wikipedia:About"),
            ("community-portal", "/wiki/Wikipedia:Community_portal"),
            ("help", "/wiki/Help:Contents"),
            ("recent-changes", "/wiki/Special:RecentChanges"),
            ("special-pages", "/wiki/Special:SpecialPages"),
        ],
    },
    "osm": {
        "requires_auth": False,
        "base_url": "https://www.openstreetmap.org",
        "pages": [
            ("home", "/"),
            ("about", "/about"),
            ("export", "/export"),
            ("help", "/help"),
            ("community", "/community"),
            ("copyright", "/copyright"),
            ("traces", "/traces"),
            ("user-diaries", "/user/diary"),
            ("login", "/login"),
            ("signup", "/user/new"),
            ("search-sf", "/search?query=San%20Francisco"),
            ("view-sf", "/?mlat=37.7749&mlon=-122.4194#map=12/37.7749/-122.4194"),
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
    """Headed browser: user logs in manually, we save storage_state.

    Signal completion by creating the sentinel file printed below (or just closing
    the browser window). This avoids blocking on stdin so the script can be
    launched in the background.
    """
    AUTH_DIR.mkdir(parents=True, exist_ok=True)
    auth_file = _auth_path(site)
    sentinel = AUTH_DIR / f"{site}.done"
    if sentinel.exists():
        sentinel.unlink()

    login_url = config.get("login_url") or config["base_url"]
    print(f"\nLaunching headed browser for {site}...")
    print(f"  Target: {login_url}")

    # Google (and some others) block bundled Chromium with "browser not secure".
    # Use real Chrome + a persistent profile + stealth flags so the login flow works.
    profile_dir = AUTH_DIR / f"{site}_profile"
    profile_dir.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        context = await p.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir),
            channel="chrome",
            headless=False,
            viewport=VIEWPORT,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-default-browser-check",
            ],
            ignore_default_args=["--enable-automation"],
        )
        page = context.pages[0] if context.pages else await context.new_page()
        await page.goto(login_url, wait_until="load", timeout=60000)

        print("\n" + "=" * 60)
        print(f"  Log in to {site} in the browser window.")
        print("  When done, signal completion by running:")
        print(f"    touch {sentinel}")
        print("  (or close the browser window)")
        print("=" * 60, flush=True)

        # Poll for sentinel file (browser close handling is flakier on persistent contexts)
        while not sentinel.exists():
            await asyncio.sleep(0.5)

        await context.storage_state(path=str(auth_file))
        await context.close()

    if sentinel.exists():
        sentinel.unlink()
    print(f"  Saved auth -> {auth_file}")


async def capture_site(site: str, config: dict, output_dir: Path) -> dict | None:
    """Capture all configured pages for one site. Returns a manifest entry or None."""
    site_dir = output_dir / site
    site_dir.mkdir(parents=True, exist_ok=True)

    requires_auth = config.get("requires_auth", False)
    if requires_auth:
        profile_dir = AUTH_DIR / f"{site}_profile"
        if not profile_dir.exists() or not any(profile_dir.iterdir()):
            print(f"  {site}: SKIP — auth required but no profile at {profile_dir}")
            print(
                f"         Run: python -m eval_awareness_experiments.scrape_real_sites --login {site}"
            )
            return None

    captured: list[str] = []
    base_url = config["base_url"]
    login_markers = config.get("login_markers", [])

    async with async_playwright() as p:
        if requires_auth:
            # Reuse the persistent Chrome profile from --login
            context = await p.chromium.launch_persistent_context(
                user_data_dir=str(AUTH_DIR / f"{site}_profile"),
                channel="chrome",
                headless=True,
                viewport=VIEWPORT,
                user_agent=USER_AGENT,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-default-browser-check",
                ],
                ignore_default_args=["--enable-automation"],
                ignore_https_errors=True,
            )
            page = context.pages[0] if context.pages else await context.new_page()
        else:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport=VIEWPORT,
                user_agent=USER_AGENT,
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

        await context.close()

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
    preserved = [e for e in existing if e.get("source") != "real" or e.get("id") not in new_ids]
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
