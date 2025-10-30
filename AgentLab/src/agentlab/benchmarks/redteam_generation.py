"""
Generation modules for redteam benchmark.

This module provides:
- PrefillDataAnalyzer: Analyzes behaviors to determine prefill data requirements
- SyntheticPageGenerator: Generates HTML pages using LLMs
- PageCacheManager: Manages caching of generated pages
- Utility functions for OpenAI Files API uploads
"""

import json
import logging
import hashlib
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
from agentlab.benchmarks.redteam_prompts import (
    get_prefill_analysis_prompt,
    get_page_generation_prompt,
)

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

    def analyze(self, behavior: str, pages: List[Any]) -> dict:
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
    

"""Generate synthetic HTML pages using LLMs.

SPEC-ONLY mode:
- Accepts a prefill spec dict describing required data per page
- Builds a functional UI scaffold and insertion hooks for later data injection
"""


class SyntheticPageGenerator:
    """Generates synthetic HTML pages using LLMs with prefill data embedded."""

    def __init__(self, model: str = "gpt-5"):
        """
        Initialize the generator.

        Args:
            model: OpenAI model to use for generation
        """
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def generate(
        self,
        domain: str,
        context: Dict[str, Any],
        prefill: Optional[dict] = None,
        real_site: str = None,
        screenshots: List[str] | str = None,
    ) -> str:
        """
        Generate synthetic HTML for a domain with prefill data.

        Args:
            domain: e.g., "/gmail", "/calendar"
            context: Additional context (e.g., behavior requirements)
            prefill: Pre-generated spec dict from analyzer
            real_site: Real site name (e.g., "gmail.com")
            screenshots: List of screenshot paths or string for folder path

        Returns:
            HTML string with embedded data
        """
        # Extract spec for this page from prefill
        spec_for_page: Optional[Dict[str, Any]] = None
        if prefill is not None:
            for page_spec in prefill.get("pages", []):
                if page_spec.get("page") == domain:
                    spec_for_page = page_spec
                    break
        logger.info(f"Generating HTML scaffold for {domain} (has_spec={bool(spec_for_page)})...")

        prompt = get_page_generation_prompt(
            domain=domain,
            context=context,
            spec_for_page=spec_for_page,
            real_site=real_site,
            screenshots=screenshots,
        )

        try:
            # Upload screenshots using shared utility
            file_ids = upload_screenshots(self.client, screenshots or [])
            
            input_content = [{"type": "input_text", "text": prompt}]
            for file_id in file_ids:
                input_content.append({"type": "input_image", "file_id": file_id})
            
            response = self.client.responses.create(
                model=self.model,
                input=[{"role": "user", "content": input_content}],
            )
            logger.debug(f"Synthetic generator response: {response.output_text[:200]}...")
            html = self._extract_html(response.output_text)
            validated_html = self._validate_and_sanitize(html)

            return validated_html

        except Exception as e:
            logger.error(f"Error generating HTML for {domain}: {e}")
            raise


    def _extract_html(self, content: str) -> str:
        """Extract HTML from response, removing any markdown formatting."""
        # Remove code fences if present
        content = content.strip()

        # Remove markdown code fences
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first line (```html or ```javascript)
            lines = lines[1:]
            # Remove last line (```)
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)

        return content.strip()

    def _validate_and_sanitize(self, html: str) -> str:
        """Validate and sanitize HTML."""
        # Basic validation
        if not html.startswith("<!DOCTYPE html>") and not html.startswith(
            "<!doctype html>"
        ):
            # Add DOCTYPE if missing
            html = "<!DOCTYPE html>\n" + html

        # Check for required tags
        if "<html" not in html.lower():
            logger.warning("Generated HTML missing <html> tag")

        if "<body" not in html.lower():
            logger.warning("Generated HTML missing <body> tag")

        return html


def upload_file_to_openai(client: OpenAI, file_path: str) -> Optional[str]:
    """
    Upload a file to OpenAI Files API.
    
    Args:
        client: OpenAI client instance
        file_path: Path to the file (relative or absolute)
    
    Returns:
        File ID if successful, None otherwise
    """
    # Handle relative paths from assets/ folder
    if not Path(file_path).is_absolute():
        file_path = Path("assets") / file_path
    
    if not Path(file_path).exists():
        logger.warning(f"Screenshot not found: {file_path}")
        return None
    
    if Path(file_path).is_dir():
        logger.warning(f"Expected file but got directory: {file_path}")
        return None
    
    try:
        with open(file_path, "rb") as file_content:
            result = client.files.create(
                file=file_content,
                purpose="vision",
            )
            logger.info(f"Uploaded {file_path}: {result.id}")
            return result.id
    except Exception as e:
        logger.warning(f"Failed to upload {file_path}: {e}")
        return None


def upload_screenshots(client: OpenAI, screenshots: List[str] | str) -> List[str]:
    """
    Upload multiple screenshots and return their file IDs.
    Screenshots can be individual file paths or folder paths.
    If folder path, uploads all files in the folder (non-recursive).
    
    Args:
        client: OpenAI client instance
        screenshots: List of screenshot paths or string for folder path
    
    Returns:
        List of file IDs (excludes failed uploads)
    """
    def resolve(p: str | Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else Path("assets") / p

    paths: List[Path] = []

    if isinstance(screenshots, str):
        dir_path = resolve(screenshots)
        if dir_path.is_dir():
            paths = [f for f in dir_path.iterdir() if f.is_file() and not f.name.startswith(".")]
        else:
            return []
    else:
        for s in screenshots:
            p = resolve(s)
            if p.is_file():
                paths.append(p)
            elif p.is_dir():
                paths.extend(f for f in p.iterdir() if f.is_file() and not f.name.startswith("."))

    return [fid for fid in (upload_file_to_openai(client, str(p)) for p in paths) if fid]


class PageCacheManager:
    """Manages cached synthetic pages with content-based hashing."""

    def __init__(self, cache_dir: Path | str):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory where cached pages are stored
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def compute_key(
        self,
        domain: str,
        behavior: str,
        prefill: Dict[str, Any],
        seed: int,
    ) -> str:
        """
        Compute deterministic cache key based on content hash.

        Args:
            domain: Page domain
            behavior: Task behavior description
            prefill: Prefill specification dict
            seed: Injection seed for variation

        Returns:
            Cache key (SHA256 hash)
        """
        content = json.dumps(
            {
                "domain": domain,
                "behavior": behavior,
                "prefill": prefill,
                "seed": seed,
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, key: str) -> Optional[str]:
        """
        Get cached page if exists.

        Args:
            key: Cache key from compute_key()

        Returns:
            HTML content if cached, None otherwise
        """
        cache_path = self.cache_dir / f"{key}.html"
        if cache_path.exists():
            logger.info(f"Cache hit: {key}")
            return cache_path.read_text(encoding="utf-8")
        return None

    def put(self, key: str, html: str):
        """
        Cache generated page.

        Args:
            key: Cache key from compute_key()
            html: HTML content to cache
        """
        cache_path = self.cache_dir / f"{key}.html"
        cache_path.write_text(html, encoding="utf-8")
        logger.info(f"Cached page: {cache_path}")

    def clear(self):
        """Clear all cached pages."""
        for path in self.cache_dir.glob("*.html"):
            path.unlink()
        logger.info(f"Cache cleared: {self.cache_dir}")
