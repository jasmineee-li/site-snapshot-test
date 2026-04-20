"""One-shot gitlab login → save storage_state.json for r5.

Accepts GITLAB_HOST / GITLAB_USERNAME / GITLAB_PASSWORD env vars so the
script works across hosts (laptop dev, r5, future deployments). The
defaults reflect WebArena's dummy credentials, which are safe to keep in
the repo. Writes a ``.meta.json`` sidecar so the runtime can apply a TTL
check before trusting the cookies.
"""

import asyncio
import json
import os
import pathlib
import sys
from datetime import UTC, datetime, timezone

from playwright.async_api import async_playwright

HOST = os.environ.get("GITLAB_HOST", "http://127.0.0.1:8023").rstrip("/")
USERNAME = os.environ.get("GITLAB_USERNAME", "byteblaze")
PASSWORD = os.environ.get("GITLAB_PASSWORD", "hello1234")
OUT = pathlib.Path(
    os.environ.get("GITLAB_STORAGE_STATE_PATH", "logs/phase_0d/gitlab/storage_state.json")
)


def _meta_path(artifact: pathlib.Path) -> pathlib.Path:
    return artifact.with_name(artifact.name.replace(".json", ".meta.json"))


async def main() -> int:
    print(f"gitlab login: host={HOST} user={USERNAME} out={OUT}", file=sys.stderr)
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
        _meta_path(OUT).write_text(
            json.dumps(
                {
                    "minted_at": datetime.now(UTC).isoformat(),
                    "mechanism": "form_login",
                    "host": HOST,
                },
                indent=2,
            )
        )
        print(f"wrote {OUT} ({len(state.get('cookies', []))} cookies) final_url={final}")
        await browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
