"""Extract website data from TheAgentCompany Docker services.

Navigates GitLab, RocketChat, and OwnCloud via Playwright, logs in,
and captures HTML, accessibility tree, and screenshot for each page.

Prerequisites:
    - TAC Docker services running (see SETUP.md)
    - Hostname: 127.0.0.1 the-agent-company.com in /etc/hosts

Usage:
    python -m eval_awareness_experiments.extract_tac
    python -m eval_awareness_experiments.extract_tac --services gitlab,rocketchat
"""

import argparse
import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright

DEFAULT_OUTPUT = Path("eval_awareness_experiments/data/tac")
MANIFEST_PATH = Path("eval_awareness_experiments/data/manifest.json")

VIEWPORT = {"width": 1280, "height": 720}

HOST = "the-agent-company.com"

# Service configs: (base_url, login info, pages to capture)
SERVICES: dict[str, dict] = {
    "gitlab": {
        "base_url": f"http://{HOST}:8929",
        "login": {
            "url": f"http://{HOST}:8929/users/sign_in",
            "username": "root",
            "password": "theagentcompany",
            "username_selector": "input[name='user[login]']",
            "password_selector": "input[name='user[password]']",
            "submit_selector": "button:has-text('Sign in')",
        },
        "pages": [
            ("projects", "/"),
            ("project-overview", "/root"),  # user's project list
            ("explore-projects", "/explore"),
            ("issues", "/dashboard/issues?assignee_username=root"),
            ("merge-requests", "/dashboard/merge_requests?assignee_username=root"),
            ("todos", "/dashboard/todos"),
            ("milestones", "/dashboard/milestones"),
            ("snippets", "/dashboard/snippets"),
            ("activity", "/dashboard/activity"),
            ("groups", "/dashboard/groups"),
            # Admin pages are flaky — GitLab often 502s under load.
            # Re-enable if needed with a dedicated run.
            # ("admin-overview", "/-/admin"),
            # ("admin-users", "/-/admin/users"),
            # ("admin-projects", "/-/admin/projects"),
        ],
    },
    "rocketchat": {
        "base_url": f"http://{HOST}:3000",
        "login": {
            "url": f"http://{HOST}:3000",
            "username": "theagentcompany",
            "password": "theagentcompany",
            "username_selector": "input[placeholder='Email or username']",
            "password_selector": "input[placeholder='Password']",
            "submit_selector": "button:has-text('Login')",
        },
        "pages": [
            ("home", "/home"),
            ("general", "/channel/general"),
            ("admin-rooms", "/admin/rooms"),
            ("admin-users", "/admin/users"),
            ("admin-info", "/admin/info"),
            ("directory", "/directory/channels"),
            ("directory-users", "/directory/users"),
        ],
    },
    "owncloud": {
        "base_url": f"http://{HOST}:8092",
        "login": {
            "url": f"http://{HOST}:8092",
            "username": "theagentcompany",
            "password": "theagentcompany",
            "username_selector": "#user",
            "password_selector": "#password",
            "submit_selector": "#submit-form, button[type='submit'], input[type='submit']",
        },
        "pages": [
            ("files", "/index.php/apps/files/"),
            ("shared-with-you", "/index.php/apps/files/?dir=/&view=sharingin"),
            ("shared-with-others", "/index.php/apps/files/?dir=/&view=sharingout"),
            ("shared-by-link", "/index.php/apps/files/?dir=/&view=sharinglinks"),
            ("favorites", "/index.php/apps/files/?dir=/&view=favorites"),
            ("settings", "/index.php/settings/personal"),
            ("admin-settings", "/index.php/settings/admin"),
            ("users", "/index.php/settings/users"),
        ],
    },
    "plane": {
        "base_url": "http://localhost:8091",
        "login": {
            "url": "http://localhost:8091",
            "type": "plane",  # two-step login
            "email": "sarah.johnson@agentcompany.com",
            "password": "sarah.johnson@agentcompany.com",
        },
        "pages": [
            ("home", "/tac/"),
            ("projects", "/tac/projects/"),
            ("issues", "/tac/workspace-views/all-issues/"),
            ("cycles", "/tac/active-cycles/"),
            ("settings-general", "/tac/settings/"),
            ("settings-members", "/tac/settings/members/"),
        ],
        "settle_time": 5,  # Plane's SPA needs more time to load data
    },
}


async def capture_page(page, output_dir: Path, page_name: str) -> dict:
    """Capture HTML, axtree, and screenshot for the current page state."""
    await asyncio.sleep(1)  # let page settle

    # HTML
    html_path = output_dir / f"{page_name}.html"
    html = await page.content()
    html_path.write_text(html, encoding="utf-8")

    # Accessibility tree
    axtree_path = output_dir / f"{page_name}_axtree.txt"
    try:
        axtree = await page.locator(":root").aria_snapshot()
        axtree_path.write_text(axtree, encoding="utf-8")
    except Exception as e:
        print(f"    axtree failed: {e}")
        axtree_path.write_text(f"ERROR: {e}", encoding="utf-8")

    # Screenshot
    screenshot_path = output_dir / f"{page_name}.png"
    await page.screenshot(path=str(screenshot_path), full_page=False)

    return {
        "name": page_name,
        "html": str(html_path),
        "axtree": str(axtree_path),
        "screenshot": str(screenshot_path),
    }


async def login(page, login_config: dict) -> bool:
    """Log in to a service. Returns True on success."""
    url = login_config["url"]
    print(f"  Logging in at {url}...")
    await page.goto(url, wait_until="load", timeout=30000)
    await asyncio.sleep(2)

    try:
        login_type = login_config.get("type", "standard")

        if login_type == "plane":
            # Two-step login: email → continue → password → go to workspace
            await page.fill(
                'input[placeholder="name@example.com"], input[placeholder="name@company.com"]',
                login_config["email"],
            )
            await asyncio.sleep(0.5)
            await page.click('button:has-text("Continue")')
            await asyncio.sleep(2)
            await page.fill('input[placeholder="Enter password"]', login_config["password"])
            await asyncio.sleep(0.5)
            await page.click('button:has-text("Go to workspace")')
        else:
            # Standard login: username + password + submit
            await page.fill(login_config["username_selector"], login_config["username"])
            await page.fill(login_config["password_selector"], login_config["password"])
            await page.click(login_config["submit_selector"], timeout=5000)

        # Wait for navigation after login
        await page.wait_for_load_state("load", timeout=30000)
        await asyncio.sleep(3)  # extra settle time for JS-heavy apps
        print("  Login successful")
        return True
    except Exception as e:
        print(f"  Login failed: {e}")
        return False


async def extract_service(
    service_name: str,
    config: dict,
    output_dir: Path,
) -> dict | None:
    """Extract all pages from one service. Returns a manifest entry."""
    service_output = output_dir / service_name
    service_output.mkdir(parents=True, exist_ok=True)

    captured_pages = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport=VIEWPORT,
            ignore_https_errors=True,
        )
        page = await context.new_page()

        # Log in
        if not await login(page, config["login"]):
            await browser.close()
            return None

        # Capture each page (with retries for flaky services like GitLab)
        base_url = config["base_url"]
        settle_time = config.get("settle_time", 1)
        max_retries = 3
        for page_name, path in config["pages"]:
            url = f"{base_url}{path}"
            for attempt in range(max_retries):
                try:
                    response = await page.goto(url, wait_until="load", timeout=30000)
                    await asyncio.sleep(settle_time)

                    # Check for error pages or login redirects
                    title = await page.title()
                    status = response.status if response else 0
                    current_url = page.url
                    is_login_redirect = "sign_in" in current_url or "login" in current_url
                    is_error = (
                        status >= 400
                        or "500" in title
                        or "502" in title
                        or "Waiting for GitLab" in title
                        or "Something went wrong" in title
                        or is_login_redirect
                    )

                    # Re-login if session expired
                    if is_login_redirect and attempt < max_retries - 1:
                        print(f"  {service_name}/{page_name} session expired, re-logging in...")
                        await login(page, config["login"])

                    if is_error and attempt < max_retries - 1:
                        wait = 10 * (attempt + 1)
                        print(f"  {service_name}/{page_name} got error (status={status}, title='{title}'), retrying in {wait}s...")
                        await asyncio.sleep(wait)
                        continue

                    if is_error:
                        print(f"  {service_name}/{page_name} FAILED after {max_retries} attempts (status={status})")
                        break

                    info = await capture_page(page, service_output, page_name)
                    captured_pages.append(page_name)
                    print(f"  {service_name}/{page_name} ✓")
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"  {service_name}/{page_name} error: {e}, retrying...")
                        await asyncio.sleep(10)
                    else:
                        print(f"  {service_name}/{page_name} FAILED: {e}")

        await browser.close()

    if not captured_pages:
        return None

    return {
        "id": f"tac_{service_name}",
        "source": "tac",
        "label": "synthetic",
        "path": str(service_output),
        "pages": captured_pages,
        "metadata": {"benchmark": "theagentcompany", "service": service_name},
    }


async def run(service_names: list[str], output_dir: Path) -> list[dict]:
    """Extract data from all specified services."""
    entries = []

    for service_name in service_names:
        config = SERVICES.get(service_name)
        if not config:
            print(f"Unknown service: {service_name}. Available: {list(SERVICES.keys())}")
            continue

        print(f"\nExtracting {service_name}...")
        entry = await extract_service(service_name, config, output_dir)
        if entry:
            entries.append(entry)

    return entries


def update_manifest(entries: list[dict], manifest_path: Path) -> None:
    """Add or update entries in the manifest."""
    existing = []
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())

    # Remove old TAC entries
    existing = [e for e in existing if e.get("source") != "tac"]
    existing.extend(entries)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"\nManifest updated: {manifest_path} ({len(existing)} total entries)")


def main():
    parser = argparse.ArgumentParser(
        description="Extract website data from TheAgentCompany Docker services"
    )
    parser.add_argument(
        "--services",
        type=str,
        default=",".join(SERVICES.keys()),
        help="Comma-separated service names to extract",
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

    service_names = [s.strip() for s in args.services.split(",")]
    output_dir = Path(args.output)

    entries = asyncio.run(run(service_names, output_dir))
    if entries:
        update_manifest(entries, Path(args.manifest))

    total_pages = sum(len(e["pages"]) for e in entries)
    print(f"\nDone. Extracted {total_pages} pages from {len(entries)} services.")


if __name__ == "__main__":
    main()
