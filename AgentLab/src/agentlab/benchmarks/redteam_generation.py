"""
Generation modules for redteam benchmark.

This module provides:
- PrefillDataAnalyzer: Analyzes behaviors to determine prefill data requirements
- SyntheticPageGenerator: Generates HTML pages using LLMs
- Site toolbox utilities: Save and load generated sites for reuse
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from agentlab.benchmarks.llm_utils import OpenRouterLLMClient, strip_code_fences
from agentlab.benchmarks.redteam_prompts import (
    get_prefill_analysis_prompt,
    get_page_generation_prompt,
    get_html_modification_prompt,
)

logger = logging.getLogger(__name__)

TOOLBOX_DIR = Path(__file__).parent.parent.parent / "html-library"


# Backward compatibility alias
LLMClient = OpenRouterLLMClient


class PrefillDataAnalyzer:
    """Analyzes behavior descriptions to determine what prefill data is needed."""

    def __init__(self, model: str = "gpt-5"):
        self.llm = LLMClient(model)

    def analyze(
        self,
        behavior: str,
        pages: List[Any],
        screenshots: Optional[List[str]] = None,
        dom_paths: Optional[List[str]] = None,
    ) -> dict:
        """
        Analyze behavior and determine what prefill data is needed.

        Args:
            behavior: The task description
            pages: List of pages in the benchmark
            screenshots: List of screenshot file paths from trajectory observation
            dom_paths: List of DOM snapshot file paths from trajectory observation

        Returns:
            Dict with structured requirements
        """
        # Read DOM snapshots if provided
        dom_contents = []
        if dom_paths:
            for dom_path in dom_paths:
                try:
                    dom_contents.append(Path(dom_path).read_text(encoding="utf-8"))
                except Exception as e:
                    logger.warning(f"Failed to read DOM from {dom_path}: {e}")

        prompt = get_prefill_analysis_prompt(behavior, pages, screenshots, dom_contents)

        response_text = self.llm.call_with_images(prompt, screenshots or [])

        logger.info(
            f"Prefill analysis response: {response_text[:200] if response_text else '(empty)'}"
        )

        if not response_text or not response_text.strip():
            logger.error(f"Empty response from {self.llm.provider} API")
            raise ValueError(f"Empty response from {self.llm.provider} API")

        response_text = strip_code_fences(response_text)

        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {response_text[:500]}")
            raise


class SyntheticPageGenerator:
    """Generates synthetic HTML pages using LLMs with prefill data embedded."""

    def __init__(self, model: str = "gpt-5"):
        """
        Initialize the generator.

        Args:
            model: Model to use for generation (OpenAI or Anthropic)
        """
        self.llm = LLMClient(model)

    def generate(
        self,
        domain: str,
        context: Dict[str, Any],
        prefill: Optional[dict] = None,
        real_site: str = None,
        screenshots: List[str] | str = None,
        url_mapping: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Generate synthetic HTML for a domain with prefill data.

        Args:
            domain: e.g., "/gmail", "/calendar"
            context: Additional context (e.g., behavior requirements)
            prefill: Pre-generated spec dict from analyzer
            real_site: Real site name (e.g., "gmail.com")
            screenshots: List of screenshot paths from trajectory observation
            url_mapping: Dict mapping page domains to their simulated URLs

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
            url_mapping=url_mapping,
        )

        try:
            html_raw = self.llm.call_with_images(prompt, screenshots or [])
            logger.debug(f"Synthetic generator response: {html_raw[:200]}...")

            html = strip_code_fences(html_raw)
            validated_html = self._validate_and_sanitize(html)

            return validated_html

        except Exception as e:
            logger.error(f"Error generating HTML for {domain}: {e}")
            raise

    def modify_base_site(
        self,
        base_html: str,
        # base_metadata: dict,
        new_prefill_spec: dict,
        domain: str,
        context: Dict[str, Any],
    ) -> str:
        """
        Modify an existing base HTML site to support new prefill requirements.

        Args:
            base_html: Existing HTML content
            base_metadata: Metadata from base site (includes original prefill_spec, behavior, etc.)
            new_prefill_spec: New prefill specification to support
            domain: Page domain (e.g., "/gmail")
            context: Additional context (e.g., new behavior)

        Returns:
            Modified HTML string

        Raises:
            Exception: If modification fails
        """
        logger.info(f"Modifying base site for {domain}...")

        prompt = get_html_modification_prompt(
            base_html=base_html,
            new_prefill_spec=new_prefill_spec,
            domain=domain,
            new_behavior=context.get("doc", ""),
        )

        try:
            html_raw = self.llm.call_with_images(prompt)
            logger.debug(f"Modification response: {html_raw[:200]}...")

            html = strip_code_fences(html_raw)
            validated_html = self._validate_and_sanitize(html)

            return validated_html

        except Exception as e:
            logger.error(f"Error modifying HTML for {domain}: {e}")
            raise

    def _validate_and_sanitize(self, html: str) -> str:
        """Validate and sanitize HTML."""
        # Basic validation
        if not html.startswith("<!DOCTYPE html>") and not html.startswith("<!doctype html>"):
            # Add DOCTYPE if missing
            html = "<!DOCTYPE html>\n" + html

        # Check for required tags
        if "<html" not in html.lower():
            logger.warning("Generated HTML missing <html> tag")

        if "<body" not in html.lower():
            logger.warning("Generated HTML missing <body> tag")

        return html


# ============================================================================
# Site Toolbox Utilities
# ============================================================================


def save_site_to_toolbox(domain: str, html: str, metadata: dict) -> Path:
    """
    Save a generated site to the toolbox for future reuse.

    Args:
        domain: Domain path (e.g., "/gmail", "/calendar")
        html: HTML content
        metadata: Metadata dict containing timestamp, behavior, prefill_spec, etc.

    Returns:
        Path to saved HTML file
    """
    # Normalize domain to folder name (remove leading slash)
    domain_name = domain.lstrip("/").split(".")[0]  # "/gmail" -> "gmail", "gmail.com" -> "gmail"

    # Create domain subfolder
    domain_dir = TOOLBOX_DIR / domain_name
    domain_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp-based filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{domain_name}_{timestamp}"

    html_path = domain_dir / f"{base_name}.html"
    json_path = domain_dir / f"{base_name}.json"

    # Save HTML and metadata
    html_path.write_text(html, encoding="utf-8")
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    logger.info(f"Saved site to toolbox: {html_path}")
    return html_path


def list_toolbox_sites(domain: str) -> List[dict]:
    """
    List all saved sites for a domain.

    Args:
        domain: Domain path (e.g., "/gmail")

    Returns:
        List of metadata dicts, sorted by timestamp (newest first)
    """
    domain_name = domain.lstrip("/").split(".")[0]
    domain_dir = TOOLBOX_DIR / domain_name

    if not domain_dir.exists():
        return []

    sites = []
    for json_file in domain_dir.glob("*.json"):
        try:
            metadata = json.loads(json_file.read_text(encoding="utf-8"))
            metadata["html_path"] = str(json_file.with_suffix(".html"))
            metadata["json_path"] = str(json_file)
            sites.append(metadata)
        except Exception as e:
            logger.warning(f"Failed to read metadata from {json_file}: {e}")

    # Sort by timestamp (newest first)
    sites.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return sites


def load_toolbox_site(html_path: Path | str) -> tuple[str, dict]:
    """
    Load a site from the toolbox.

    Args:
        html_path: Path to HTML file

    Returns:
        Tuple of (html_content, metadata)

    Raises:
        FileNotFoundError: If files don't exist
    """
    html_path = Path(html_path)
    # json_path = html_path.with_suffix(".json")

    if not html_path.exists():
        raise FileNotFoundError(f"HTML file not found: {html_path}")

    # if not json_path.exists():
    #     raise FileNotFoundError(f"Metadata file not found: {json_path}")

    html = html_path.read_text(encoding="utf-8")
    # metadata = json.loads(json_path.read_text(encoding="utf-8"))

    logger.info(f"Loaded site from toolbox: {html_path}")
    return html
