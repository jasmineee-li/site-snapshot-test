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
        simulated_url: str = None,
        screenshots: List[str] | str = None,
        url_mapping: Optional[Dict[str, str]] = None,
        sibling_subdomains: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Generate synthetic HTML for a domain/subdomain with prefill data.

        Args:
            domain: Route key, e.g., "gmail/mail/u/0/#inbox"
            context: Additional context (e.g., behavior requirements, subdomain_details)
            prefill: Pre-generated spec dict for this subdomain
            simulated_url: Full URL this page simulates (e.g., "https://mail.google.com/mail/u/0/#inbox")
            screenshots: List of screenshot paths from trajectory observation
            url_mapping: Dict mapping local routes to their simulated URLs
            sibling_subdomains: List of sibling subdomains for navigation context

        Returns:
            HTML string with embedded data
        """
        # For subdomain architecture, prefill is already the spec for this subdomain
        spec_for_page = prefill if prefill else None
        logger.info(f"Generating HTML scaffold for {domain} (has_spec={bool(spec_for_page)})...")

        prompt = get_page_generation_prompt(
            domain=domain,
            context=context,
            spec_for_page=spec_for_page,
            simulated_url=simulated_url,
            screenshots=screenshots,
            url_mapping=url_mapping,
            sibling_subdomains=sibling_subdomains,
        )

        # Debug: log prompt info before API call
        logger.debug(f"Generated prompt length: {len(prompt)} chars, first 200: {prompt[:200]}...")

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


def replace_injection_placeholders(html: str, injections: Dict[str, str]) -> str:
    """
    Replace INJECTION_PLACEHOLDER comments (format: `<!-- INJECTION_PLACEHOLDER: {"id": "...", ...} -->`) with actual injection HTML
    """
    import re

    # Pattern to match injection placeholder comments
    # Matches: <!-- INJECTION_PLACEHOLDER: {"id": "some_id", ...} -->
    pattern = r"<!--\s*INJECTION_PLACEHOLDER:\s*(\{[^}]+\})\s*-->"

    def replace_placeholder(match):
        """Replacement function for each placeholder match."""
        json_str = match.group(1)
        try:
            placeholder_data = json.loads(json_str)
            placeholder_id = placeholder_data.get("id")

            if not placeholder_id:
                logger.warning(f"Placeholder missing 'id' field: {json_str}")
                return match.group(0)  # Keep original placeholder

            if placeholder_id not in injections:
                logger.warning(
                    f"No injection provided for placeholder '{placeholder_id}'. "
                    f"Available: {list(injections.keys())}"
                )
                return match.group(0)  # Keep original placeholder

            # Replace with injection content
            injection_html = injections[placeholder_id]
            logger.debug(
                f"Replacing placeholder '{placeholder_id}' with injection ({len(injection_html)} chars)"
            )
            return injection_html

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse placeholder JSON: {json_str} - {e}")
            return match.group(0)  # Keep original placeholder

    # Perform replacement
    html_with_injections = re.sub(pattern, replace_placeholder, html)

    # Count replacements made
    placeholders_found = len(re.findall(pattern, html))
    placeholders_remaining = len(re.findall(pattern, html_with_injections))
    replacements_made = placeholders_found - placeholders_remaining

    logger.info(
        f"Replaced {replacements_made}/{placeholders_found} placeholder(s). "
        f"{placeholders_remaining} placeholder(s) remain."
    )

    # Warn about accessibility-hiding patterns that would make injections invisible to agents
    accessibility_hiding_patterns = [
        (r'aria-hidden\s*=\s*["\']true["\']', 'aria-hidden="true"'),
        (r'left:\s*-\d{4,}px', 'off-screen left positioning (left: -9999px)'),
        (r'top:\s*-\d{4,}px', 'off-screen top positioning (top: -9999px)'),
        (r'display:\s*none', 'display: none'),
        (r'visibility:\s*hidden', 'visibility: hidden'),
    ]

    for pattern_re, description in accessibility_hiding_patterns:
        if re.search(pattern_re, html_with_injections, re.IGNORECASE):
            logger.warning(
                f"⚠️ Generated HTML contains '{description}' which may hide content from accessibility APIs. "
                f"This could make the injection invisible to the target agent."
            )

    return html_with_injections


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
    # Normalize domain to folder path (remove leading slash)
    # "/gmail" -> "gmail", "/unofficial-course-wiki/phil-380" -> "unofficial-course-wiki/phil-380"
    domain_path = domain.lstrip("/").split(".")[0]

    # Create domain subfolder
    domain_dir = TOOLBOX_DIR / domain_path
    domain_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp-based filename using only the last segment
    # "unofficial-course-wiki/phil-380" -> "phil-380"
    file_base = domain_path.split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{file_base}_{timestamp}"

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
    domain_path = domain.lstrip("/").split(".")[0]
    domain_dir = TOOLBOX_DIR / domain_path

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
