"""Trajectory observation using Claude computer use with Playwright.

Captures screenshots from real websites for synthetic page generation.
"""

import asyncio
import logging
import shutil
import os
from pathlib import Path
from typing import Any
from datetime import datetime

from anthropic import Anthropic
from playwright.async_api import async_playwright

from agentlab.trajectory_agent.loop import sampling_loop
from agentlab.trajectory_agent.playwright_computer_use.async_api import PlaywrightToolbox
from agentlab.benchmarks.redteam_prompts import get_trajectory_observation_prompt

logger = logging.getLogger(__name__)

MODEL = "claude-opus-4-5-20251101"
BETA_VERSION = "20251124"


class TrajectoryObserver:
    """Observes UI structure on real websites using Claude computer use."""

    def __init__(self, model: str = MODEL):
        self.model = model
        self.anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def observe_trajectory(
        self,
        safe_behavior: str,
        sites: list[str],
        output_dir: Path,
        page_details: list[dict] = None,
        html_library_dir: Path = None,
    ) -> dict[str, Any]:
        """
        Observe UI structure using Claude computer use.

        Args:
            safe_behavior: Safe behavior description
            sites: List of real site URLs to visit
            output_dir: Directory to save screenshots (exp_dir/trajectory/)
            page_details: Page config with domain, real_site, details
            html_library_dir: Optional secondary location for screenshots

        Returns:
            Trajectory data with screenshot paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting trajectory observation for: {safe_behavior[:50]}...")
        logger.info(f"Target sites: {sites}")
        logger.info(f"Output dir: {output_dir}")

        # Run async observation
        result = asyncio.run(
            self._run_observation(
                safe_behavior=safe_behavior,
                sites=sites,
                output_dir=output_dir,
                page_details=page_details,
            )
        )

        # Copy to html_library if specified
        if html_library_dir:
            self._copy_to_library(output_dir, html_library_dir)

        return result

    async def _run_observation(
        self,
        safe_behavior: str,
        sites: list[str],
        output_dir: Path,
        page_details: list[dict] = None,
    ) -> dict[str, Any]:
        """Run the async observation loop."""
        async with async_playwright() as playwright:
            browser = await playwright.firefox.launch(headless=False)
            context = await browser.new_context()
            page = await context.new_page()
            await page.set_viewport_size({"width": 1024, "height": 768})

            # Start on first site
            first_site = sites[0] if sites else "https://www.google.com"
            if not first_site.startswith(("http://", "https://")):
                first_site = f"https://{first_site}"
            await page.goto(first_site)

            # Setup tools with save_dir pointing to output
            playwright_tools = PlaywrightToolbox(
                page,
                use_cursor=True,
                beta_version=BETA_VERSION,
                save_dir=str(output_dir),
            )

            # Build prompt
            prompt = get_trajectory_observation_prompt(
                safe_behavior=safe_behavior,
                sites=sites,
                page_details=page_details,
            )

            # Run sampling loop
            messages = await sampling_loop(
                model=self.model,
                anthropic_client=self.anthropic_client,
                messages=[{"role": "user", "content": prompt}],
                tools=playwright_tools,
                page=page,
                verbose=True,
                only_n_most_recent_images=10,
            )

            await browser.close()

            # Collect screenshot paths
            screenshots = list(output_dir.glob("*.png"))
            logger.info(f"Captured {len(screenshots)} screenshots")

            return {
                "safe_behavior": safe_behavior,
                "sites": sites,
                "screenshots": [str(p) for p in screenshots],
                "output_dir": str(output_dir),
                "timestamp": datetime.now().isoformat(),
                }

    def _copy_to_library(self, output_dir: Path, html_library_dir: Path):
        """Copy screenshots to html-library for reuse."""
        html_library_dir = Path(html_library_dir)
        html_library_dir.mkdir(parents=True, exist_ok=True)

        for png in output_dir.glob("*.png"):
            dest = html_library_dir / png.name
            shutil.copy2(png, dest)
            logger.info(f"Copied {png.name} to {html_library_dir}")
