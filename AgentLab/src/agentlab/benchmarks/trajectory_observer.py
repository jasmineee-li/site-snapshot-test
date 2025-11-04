"""Trajectory observation using computer-use-preview model with local Playwright.

This module provides an observation-focused agent that captures minimal UI data
(screenshot + DOM) from real websites without executing full behaviors.
"""

import logging
import json
import base64
import uuid
import os
import time
import requests
from pathlib import Path
from typing import Any, Optional
from datetime import datetime

from agentlab.benchmarks.redteam_prompts import get_trajectory_observation_prompt
from local_playwright import LocalPlaywrightBrowser

logger = logging.getLogger(__name__)


def create_response(**kwargs):
    """Create response using OpenAI Responses API for computer-use-preview."""
    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json",
    }
    openai_org = os.getenv("OPENAI_ORG")
    if openai_org:
        headers["OpenAI-Organization"] = openai_org

    # Inject system message if provided
    system_prompt = kwargs.pop("instructions", None)
    input_items = kwargs.get("input", [])
    if system_prompt:
        # Prepend system message without overwriting user messages
        input_items = [{"role": "system", "content": system_prompt}] + input_items
    kwargs["input"] = input_items
    kwargs.setdefault("truncation", "auto")

    # DEBUG: Log the full request payload
    logger.info("=" * 80)
    logger.info("🔍 DEBUG: API Request Details")
    logger.info(f"URL: {url}")
    logger.info(f"Model: {kwargs.get('model', 'NOT SET')}")
    logger.info(f"Has tools: {'tools' in kwargs}")
    if "tools" in kwargs:
        logger.info(f"Tools: {json.dumps(kwargs['tools'], indent=2)}")
    logger.info(f"Input messages count: {len(kwargs.get('input', []))}")
    logger.info(
        f"Full request body: {json.dumps(kwargs, indent=2)[:2000]}"
    )  # Truncate for readability
    logger.info("=" * 80)

    # Retry logic for transient errors
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=kwargs, timeout=120)
            if response.status_code == 200:
                data = response.json()
                logger.debug(f"✅ API Response: {data.get('id', 'unknown')}")
                return data
            elif response.status_code == 500 and attempt < max_retries - 1:
                logger.warning(f"⚠️  Server error 500, retrying ({attempt + 1}/{max_retries})...")
                logger.warning(f"Error response: {response.text}")
                time.sleep(2**attempt)  # Exponential backoff
                continue
            else:
                logger.error(f"❌ Error {response.status_code}: {response.text}")
                logger.error(f"Request headers: {headers}")
                logger.error(f"Request body keys: {list(kwargs.keys())}")
                return None
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                logger.warning(f"⚠️  Request error, retrying ({attempt + 1}/{max_retries}): {e}")
                time.sleep(2**attempt)
                continue
            else:
                logger.error(f"❌ Request failed after {max_retries} attempts: {e}")
                return None

    return None


class TrajectoryObserver:
    """Observes UI structure on real websites using computer-use-preview model."""

    def observe_trajectory(
        self,
        safe_behavior: str,
        sites: list[str],
        output_dir: Optional[Path] = None,
    ) -> dict[str, Any]:
        """
        Observe UI structure for a safe behavior using computer-use-preview.

        Args:
            safe_behavior: Safe analog behavior to observe
            sites: List of website URLs to visit
            output_dir: Directory to save observations

        Returns:
            Trajectory observation data with screenshots and DOM snapshots
        """
        # Setup output directory
        self.output_dir = output_dir or Path.cwd() / "trajectories" / uuid.uuid4().hex
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.observations = []

        logger.info("=" * 80)
        logger.info("🔍 DEBUG: observe_trajectory called with:")
        logger.info(f"  safe_behavior: {safe_behavior[:100]}...")
        logger.info(f"  sites: {sites}")
        logger.info(f"  sites type: {[type(s).__name__ for s in sites]}")
        logger.info(f"  output_dir: {output_dir}")
        logger.info("=" * 80)

        logger.info(f"Starting trajectory observation for: {safe_behavior[:50]}...")
        logger.info(f"Target sites: {', '.join(sites)}")
        logger.info("Using computer-use-preview with LocalPlaywrightBrowser")

        # Use LocalPlaywrightBrowser like in the local_playwright module
        logger.info("Initializing LocalPlaywrightBrowser...")

        # Extract hostnames from sites to add to allowed_hosts
        allowed_hosts = set()
        for site in sites:
            # Parse hostname from URL
            from urllib.parse import urlparse

            if site.startswith(("http://", "https://")):
                hostname = urlparse(site).hostname
            else:
                # If no protocol, assume it's just a hostname
                hostname = site.split("/")[0]
            if hostname:
                allowed_hosts.add(hostname)

        logger.info(f"Allowing network access to: {allowed_hosts}")

        try:
            with LocalPlaywrightBrowser(allowed_hosts=allowed_hosts) as computer:
                logger.info("✅ Browser initialized successfully")
                dimensions = computer.get_dimensions()
                logger.info(f"Browser dimensions: {dimensions}")

                tools = [
                    {
                        "type": "computer-preview",
                        "display_width": dimensions[0],
                        "display_height": dimensions[1],
                        "environment": computer.get_environment(),
                    }
                ]
                logger.info(f"Tools constructed: {json.dumps(tools, indent=2)}")

                # Build system prompt and initial user message
                system_prompt = get_trajectory_observation_prompt(safe_behavior, sites)

                # Run the observation loop
                logger.info("Starting observation loop...")
                # Initialize conversation items list (no interactive user loop)
                items = []
                self._run_observation_loop(items, system_prompt, tools, computer)

                trajectory = {
                    "safe_behavior": safe_behavior,
                    "sites": sites,
                    "observations": self.observations,
                    "output_dir": str(self.output_dir),
                    "completed": True,
                    "timestamp": datetime.now().isoformat(),
                }

                (self.output_dir / "trajectory.json").write_text(json.dumps(trajectory, indent=2))

                logger.info(f"Observation complete: {len(self.observations)} pages captured")
                return trajectory
        except Exception as e:
            logger.error(
                f"❌ Error during browser initialization or observation: {e}", exc_info=True
            )
            raise

    def _run_observation_loop(self, items: list, system_prompt: str, tools: list, computer):
        """Run the observation loop with computer-use-preview model."""
        max_turns = 30

        for turn in range(max_turns):
            logger.info(f"Turn {turn + 1}/{max_turns}")

            # Get response from computer-use-preview
            response = create_response(
                model="computer-use-preview",
                input=items,
                tools=tools,
                truncation="auto",
                instructions=system_prompt,
            )

            if response is None:
                logger.error("API call failed - response is None")
                raise ValueError("API call failed - no response from computer-use-preview")

            if "output" not in response:
                logger.error(f"Response missing 'output' field: {response}")
                raise ValueError(f"No output from model. Response: {response}")

            items += response["output"]

            # Process each output item
            logger.info(f"🔍 Response output items: {len(response['output'])}")
            for idx, item in enumerate(response["output"]):
                logger.info(f"  Item {idx}: type={item.get('type')}, keys={list(item.keys())}")
                if item.get("type") == "computer_call":
                    logger.info(f"    Action: {item.get('action', {}).get('type')}")
                new_items = self._handle_item(item, computer)
                items += new_items

            # Check if agent finished
            # Agent finishes when it outputs a message without tool calls, or when it has enough observations
            last_output = response["output"][-1] if response["output"] else None
            if last_output and last_output.get("type") == "message":
                logger.info("Agent finished observation (sent final message)")
                return

            # Also finish if we have captured enough observations (e.g., 5+)
            if len(self.observations) >= 5:
                logger.info(f"Agent captured {len(self.observations)} observations - completing")
                return

        logger.info(f"Reached max turns ({max_turns})")

    def _handle_item(self, item, computer):
        """Handle each item from computer-use-preview response."""
        new_items = []

        if item["type"] == "message":
            # Print messages
            content = item.get("content", [])
            if content and len(content) > 0:
                text = content[0].get("text", "")
                if text:
                    logger.info(f"Agent message: {text[:100]}...")
                    print(text)

        elif item["type"] == "reasoning":
            # Handle reasoning output (extended thinking)
            summary = item.get("summary", "")
            if summary:
                logger.debug(f"Agent reasoning: {summary[:100]}...")

        elif item["type"] == "computer_call":
            # Perform computer actions
            action = item["action"]
            action_type = action["type"]
            action_args = {k: v for k, v in action.items() if k != "type"}
            print(f"{action_type}({action_args})")

            # Execute action on computer (unless it's capture, which is a marker)
            if action_type != "capture":
                getattr(computer, action_type)(**action_args)

            # Always capture screenshot and DOM for visual feedback
            screenshot_base64 = computer.screenshot()

            # Also capture DOM for analysis
            try:
                dom_html = computer._page.content()  # Access underlying playwright page
            except:
                dom_html = "<html><body>DOM capture failed</body></html>"

            # Only save observation to disk if this is a capture() call
            if action_type == "capture":
                page_id = uuid.uuid4().hex
                page_dir = self.output_dir / page_id
                page_dir.mkdir(parents=True, exist_ok=True)

                # Decode and save screenshot
                screenshot_bytes = base64.b64decode(screenshot_base64)
                (page_dir / "screenshot.png").write_bytes(screenshot_bytes)
                (page_dir / "dom_fragment.html").write_text(dom_html, encoding="utf-8")

                current_url = (
                    computer.get_current_url()
                    if hasattr(computer, "get_current_url")
                    else "unknown"
                )

                meta = {
                    "url": current_url,
                    "action": action_type,
                    "action_args": action_args,
                    "timestamp": datetime.utcnow().isoformat(),
                }
                (page_dir / "meta.json").write_text(json.dumps(meta, indent=2))

                self.observations.append(
                    {
                        "page_id": page_id,
                        "url": meta["url"],
                        "action": action_type,
                        "action_args": action_args,
                        "paths": {
                            "screenshot": str(page_dir / "screenshot.png"),
                            "dom": str(page_dir / "dom_fragment.html"),
                            "meta": str(page_dir / "meta.json"),
                        },
                    }
                )

                logger.info(f"Captured observation at: {current_url}")
            else:
                logger.debug(f"Executed {action_type} (not saved to disk)")

            # Always return screenshot to model for visual feedback
            call_output = {
                "type": "computer_call_output",
                "call_id": item["call_id"],
                "acknowledged_safety_checks": item.get("pending_safety_checks", []),
                "output": {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{screenshot_base64}",
                },
            }

            # Add URL for browser environments
            if computer.get_environment() == "browser":
                current_url = computer.get_current_url()
                call_output["output"]["current_url"] = current_url

            new_items.append(call_output)

        return new_items
