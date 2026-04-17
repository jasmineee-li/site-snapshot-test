"""One-shot gitlab login → save storage_state.json for r5."""

import asyncio
import json
import pathlib
import sys

from playwright.async_api import async_playwright

HOST = "http://3.12.221.9:8023"
USERNAME = "byteblaze"
PASSWORD = "hello1234"
OUT = pathlib.Path("logs/phase_0d/gitlab/storage_state.json")


async def main() -> int:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context()
        page = await ctx.new_page()
        await page.goto(f"{HOST}/users/sign_in", wait_until="domcontentloaded", timeout=60000)
        await page.fill("input[name='user[login]']", USERNAME)
        await page.fill("input[name='user[password]']", PASSWORD)
        await page.click("button[type='submit'], [data-testid='sign-in-button']")
        try:
            await page.wait_for_url(f"{HOST}/", timeout=30000)
        except Exception:
            await page.wait_for_load_state("networkidle", timeout=15000)
        final = page.url
        if "sign_in" in final:
            body = await page.content()
            print("login failed, still at", final, file=sys.stderr)
            pathlib.Path("/tmp/gitlab_login_debug.html").write_text(body)
            return 2
        state = await ctx.storage_state()
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(state, indent=2))
        print(f"wrote {OUT} ({len(state.get('cookies', []))} cookies) final_url={final}")
        await browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
