"""Analyze behavior to determine required prefill data."""

import json
import logging
import os
from dotenv import load_dotenv

load_dotenv()
from typing import List, Optional

from openai import OpenAI
from pathlib import Path

from agentlab.benchmarks.redteam import Page
from agentlab.benchmarks.redteam_generation import upload_screenshots
from agentlab.benchmarks.redteam_prompts import get_prefill_analysis_prompt

logger = logging.getLogger(__name__)

class PrefillDataAnalyzer:
    """Analyzes behavior descriptions to determine what prefill data is needed."""

    def __init__(self, model: str = "gpt-5"):
        """
        Initialize the analyzer.

        Args:
            model: OpenAI model to use for analysis
        """
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def analyze(self, behavior: str, pages: List[Page]) -> dict:
        """
        Analyze behavior and determine what prefill data is needed.

        Args:
            behavior: The task description
            pages: List of pages in the benchmark

        Returns:
            Dict with structured requirements
        """
        prompt = get_prefill_analysis_prompt(behavior, pages)

        all_screenshots = []
        for page in pages:
            if page.screenshots:
                    all_screenshots.extend(page.screenshots)

        file_ids = upload_screenshots(self.client, all_screenshots)

        # Build message content with text and images
        content = [{"type": "text", "text": prompt}]
        for file_id in file_ids:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"file://{file_id}"}  # OpenAI file format
            })

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
        )

        response_text = response.choices[0].message.content
        logger.info(f"Prefill analysis response: {response_text[:200] if response_text else '(empty)'}")

        if not response_text or not response_text.strip():
            logger.error("Empty response from OpenAI API")
            raise ValueError("Empty response from OpenAI API")

        # Strip markdown code blocks if present
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]  # Remove ```json
        elif response_text.startswith("```"):
            response_text = response_text[3:]  # Remove ```
        if response_text.endswith("```"):
            response_text = response_text[:-3]  # Remove trailing ```
        response_text = response_text.strip()

        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {response_text[:500]}")
            raise
