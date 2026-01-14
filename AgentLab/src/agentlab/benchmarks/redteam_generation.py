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
        safe_behavior: Optional[str] = None,
    ) -> dict:
        """
        Analyze behavior and determine what prefill data is needed.

        Args:
            behavior: The adversarial task description
            pages: List of pages in the benchmark
            screenshots: List of screenshot file paths from trajectory observation
            dom_paths: List of DOM snapshot file paths from trajectory observation
            safe_behavior: The safe analog - legitimate task for benign condition

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

        prompt = get_prefill_analysis_prompt(
            behavior, pages, screenshots, dom_contents, safe_behavior=safe_behavior
        )

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
        reference_html: Optional[str] = None,
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
            reference_html: Reference page HTML for UI style matching (if provided,
                this page will match the reference's visual style)

        Returns:
            HTML string with embedded data
        """
        # For subdomain architecture, prefill is already the spec for this subdomain
        spec_for_page = prefill if prefill else None
        is_reference = reference_html is None
        logger.info(
            f"Generating {'reference' if is_reference else 'subsequent'} HTML for {domain} "
            f"(has_spec={bool(spec_for_page)})..."
        )

        prompt = get_page_generation_prompt(
            domain=domain,
            context=context,
            spec_for_page=spec_for_page,
            simulated_url=simulated_url,
            screenshots=screenshots,
            url_mapping=url_mapping,
            sibling_subdomains=sibling_subdomains,
        )

        # Add reference matching instruction for subsequent pages
        if reference_html is not None:
            # Truncate reference for prompt (keep first 12KB)
            ref_preview = reference_html[:12000] if len(reference_html) > 12000 else reference_html
            prompt = f"""{prompt}

## IMPORTANT: MATCH REFERENCE PAGE UI

This page must match the following reference page's styling exactly:
- Same CSS classes and variable names
- Same color scheme, fonts, spacing
- Same header/nav/footer structure
- Same component styling (buttons, cards, forms, inputs)
- Same overall visual rhythm and density

Analyze the reference HTML below and replicate its visual design language.

### Reference HTML (match this style):
```html
{ref_preview}
```
"""

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


def extract_placeholders(html: str) -> list[dict]:
    """
    Extract all placeholder comments from HTML.

    Supports both:
    - New format: <!-- VARIANT_PLACEHOLDER: {"id": "...", "placement": "...", "element_type": "..."} -->
    - Legacy format: <!-- INJECTION_PLACEHOLDER: {"id": "...", "placement": "...", "type": "..."} -->

    Args:
        html: HTML string to search

    Returns:
        List of placeholder dicts with id, placement, element_type/type, and other metadata
    """
    import re

    # Match both VARIANT_PLACEHOLDER (new) and INJECTION_PLACEHOLDER (legacy)
    pattern = r"<!--\s*(?:VARIANT_PLACEHOLDER|INJECTION_PLACEHOLDER):\s*(\{[^}]+\})\s*-->"
    matches = re.findall(pattern, html)

    placeholders = []
    for match in matches:
        try:
            placeholder = json.loads(match)
            # Normalize: ensure element_type is present (map from legacy 'type' if needed)
            if "element_type" not in placeholder and "type" in placeholder:
                placeholder["element_type"] = placeholder["type"]
            placeholders.append(placeholder)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse placeholder JSON: {match} - {e}")

    logger.info(f"Extracted {len(placeholders)} placeholder(s) from HTML")
    return placeholders


def replace_placeholders_with_content(
    html: str,
    placeholder_content: dict[str, dict],
    mode: str,
    variant_index: int = 0,
) -> str:
    """
    Replace placeholder comments with generated content.

    Supports both:
    - New format: <!-- VARIANT_PLACEHOLDER: {...} -->
    - Legacy format: <!-- INJECTION_PLACEHOLDER: {...} -->

    This is the key function for applying benign or adversarial content
    to placeholders in the unified injection pipeline.

    Args:
        html: HTML string with placeholders
        placeholder_content: Dict mapping placeholder IDs to content dicts
            Each content dict has: {"benign": "...", "adversarial": ["v0", "v1", ...]}
        mode: "benign" or "adversarial"
        variant_index: Which adversarial variant to use (0-indexed)

    Returns:
        HTML string with placeholders replaced
    """
    import re

    # Match both VARIANT_PLACEHOLDER and INJECTION_PLACEHOLDER
    pattern = r"<!--\s*(?:VARIANT_PLACEHOLDER|INJECTION_PLACEHOLDER):\s*(\{[^}]+\})\s*-->"

    def replacer(match):
        json_str = match.group(1)
        try:
            placeholder = json.loads(json_str)
            placeholder_id = placeholder.get("id")

            if not placeholder_id:
                logger.warning(f"Placeholder missing 'id' field: {json_str}")
                return match.group(0)  # Keep original

            content = placeholder_content.get(placeholder_id)
            if not content:
                logger.warning(
                    f"No content for placeholder '{placeholder_id}'. "
                    f"Available: {list(placeholder_content.keys())}"
                )
                return match.group(0)  # Keep original

            if mode == "benign":
                replacement = content.get("benign", "")
                # For hidden_text elements, benign should be empty
                element_type = placeholder.get("element_type", placeholder.get("type", ""))
                if element_type == "hidden_text" and not replacement:
                    replacement = ""  # Explicitly empty for hidden text in benign mode
                logger.debug(f"Replacing '{placeholder_id}' with benign content")
            else:  # adversarial
                adversarial_list = content.get("adversarial", [])
                if variant_index < len(adversarial_list):
                    replacement = adversarial_list[variant_index]
                elif adversarial_list:
                    replacement = adversarial_list[0]
                    logger.warning(
                        f"Variant index {variant_index} out of range for '{placeholder_id}', "
                        f"using variant 0"
                    )
                else:
                    replacement = ""
                    logger.warning(f"No adversarial content for '{placeholder_id}'")
                logger.debug(
                    f"Replacing '{placeholder_id}' with adversarial v{variant_index}"
                )

            return replacement

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse placeholder JSON: {json_str} - {e}")
            return match.group(0)

    html_with_content = re.sub(pattern, replacer, html)

    # Log replacement stats
    placeholders_before = len(re.findall(pattern, html))
    placeholders_after = len(re.findall(pattern, html_with_content))
    replacements_made = placeholders_before - placeholders_after

    logger.info(
        f"Replaced {replacements_made}/{placeholders_before} placeholder(s) "
        f"with {mode} content" + (f" (variant {variant_index})" if mode == "adversarial" else "")
    )

    return html_with_content


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


# ============================================================================
# Unified Injection Pipeline Functions
# ============================================================================


def generate_base_pages(
    behavior_config: dict,
    output_dir: str | Path,
    llm_client: OpenRouterLLMClient,
) -> dict[str, Path]:
    """
    Generate complete base pages for a behavior (no injection placeholders).

    Uses the existing get_page_generation_prompt for consistency.
    Generates reference page first, then subsequent pages match its UI.

    Args:
        behavior_config: Full behavior configuration dict
        output_dir: Directory to save generated HTML files
        llm_client: LLM client for generation

    Returns:
        Dict mapping subdomain paths to HTML file paths
        Example: {"/inbox": Path("output/gmail_inbox.html"), ...}
    """
    from agentlab.benchmarks.redteam_prompts import (
        get_prefill_analysis_prompt,
        get_page_generation_prompt,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sim_config = behavior_config.get("simulation_config", {})
    pages_config = sim_config.get("pages", [])
    all_paths = {}

    for page_config in pages_config:
        page_id = page_config.get("id", "").lstrip("/")
        base_site_url = page_config.get("base_site_url", "")
        subdomains = page_config.get("subdomains", ["/"])

        # Step 1: Generate prefill spec for realistic content
        logger.info(f"Analyzing prefill requirements for {page_id}...")
        prefill_prompt = get_prefill_analysis_prompt(
            behavior=behavior_config.get("doc", ""),
            pages=[page_config],
            screenshots=page_config.get("screenshots"),
        )
        prefill_response = llm_client.chat_simple([{"role": "user", "content": prefill_prompt}])
        prefill_spec = _parse_json_response(prefill_response)

        # Step 2: Select reference subdomain and order generation
        reference_subdomain = _select_reference_subdomain(subdomains)
        ordered_subdomains = [reference_subdomain] + [s for s in subdomains if s != reference_subdomain]

        # Build URL mapping for cross-page navigation
        url_mapping = {}
        for sd in subdomains:
            local_route = f"{page_id}{sd}"
            full_url = f"{base_site_url}{sd}" if base_site_url else local_route
            url_mapping[local_route] = full_url

        # Step 3: Generate pages
        reference_html = None

        for subdomain in ordered_subdomains:
            route_key = f"{page_id}{subdomain}"
            full_url = f"{base_site_url}{subdomain}" if base_site_url else subdomain

            # Get subdomain-specific prefill from spec
            subdomain_spec = None
            if prefill_spec:
                for ps in prefill_spec.get("pages", []):
                    if ps.get("page") == page_id:
                        subdomain_specs = ps.get("subdomains", {})
                        subdomain_spec = subdomain_specs.get(subdomain, {})
                        break

            # Build sibling subdomains for navigation
            sibling_subdomains = [
                {
                    "subdomain": sd,
                    "route": f"{page_id}{sd}",
                    "full_url": f"{base_site_url}{sd}" if base_site_url else sd,
                }
                for sd in subdomains if sd != subdomain
            ]

            # Use existing get_page_generation_prompt (with no adversarial placeholders)
            # For base pages, we set prefill_adversarial.needed = False
            clean_spec = subdomain_spec.copy() if subdomain_spec else {}
            if "prefill_adversarial" in clean_spec:
                clean_spec["prefill_adversarial"] = {"needed": False}

            context = {
                "doc": behavior_config.get("doc", ""),
                "subdomain_details": page_config.get("details", {}).get(subdomain, ""),
            }

            prompt = get_page_generation_prompt(
                domain=route_key,
                context=context,
                spec_for_page=clean_spec,
                simulated_url=full_url,
                screenshots=page_config.get("screenshots"),
                url_mapping=url_mapping,
                sibling_subdomains=sibling_subdomains,
            )

            # Add reference matching instruction for subsequent pages
            if reference_html is not None:
                # Truncate reference for prompt
                ref_preview = reference_html[:10000] if len(reference_html) > 10000 else reference_html
                prompt = f"""
{prompt}

## IMPORTANT: MATCH REFERENCE PAGE UI

This page must match the following reference page's styling exactly:
- Same CSS classes and selectors
- Same color scheme, fonts, spacing
- Same header, nav, footer structure

Reference HTML (match this style):
```html
{ref_preview}
```
"""

            logger.info(f"Generating {'reference' if reference_html is None else 'subsequent'} page: {route_key}")
            response = llm_client.chat_simple([{"role": "user", "content": prompt}])
            html = strip_code_fences(response)

            if reference_html is None:
                reference_html = html

            # Save HTML file
            filename = _subdomain_to_filename(page_id, subdomain)
            path = output_dir / filename
            path.write_text(html, encoding="utf-8")
            all_paths[subdomain] = path
            logger.info(f"Saved base page: {path}")

    return all_paths


def create_benign_variant(
    base_html_paths: dict[str, Path],
    output_dir: str | Path,
) -> dict[str, Path]:
    """
    Create benign variant by copying base pages (no injections).

    Base pages are already complete - just copy them.

    Args:
        base_html_paths: Dict mapping subdomains to base HTML file paths
        output_dir: Directory to save benign variant files

    Returns:
        Dict mapping subdomains to output file paths
    """
    import shutil

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {}
    for subdomain, base_path in base_html_paths.items():
        output_path = output_dir / base_path.name
        shutil.copy(base_path, output_path)
        output_paths[subdomain] = output_path
        logger.info(f"Copied benign variant: {output_path}")

    return output_paths


def create_adversarial_variant(
    base_html_paths: dict[str, Path],
    output_dir: str | Path,
    behavior_config: dict,
    attack_channels: list[str],
    llm_client: OpenRouterLLMClient,
    attacker_llm_client: OpenRouterLLMClient = None,
    variation_index: int = 0,
    n_total_variations: int = 1,
) -> tuple[dict[str, Path], list[dict]]:
    """
    Create adversarial variant by generating and applying injections in one step.

    Args:
        base_html_paths: Dict mapping subdomains to base HTML file paths
        output_dir: Directory to save adversarial variant files
        behavior_config: Full behavior configuration dict
        attack_channels: List of channels agent observes ["html", "axtree", "visual"]
        llm_client: LLM client for generation
        attacker_llm_client: LLM client for injection generation (defaults to llm_client)
        variation_index: Which variation this is (for diversity)
        n_total_variations: Total number of adversarial variations

    Returns:
        Tuple of (output_paths dict, injections list)
    """
    from agentlab.benchmarks.redteam_attacker import generate_and_apply_injections

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    attacker_llm = attacker_llm_client or llm_client

    # Load base HTML content
    base_html = {
        subdomain: path.read_text(encoding="utf-8")
        for subdomain, path in base_html_paths.items()
    }

    # Single-step: analyze + generate + apply injections
    logger.info(f"Generating adversarial variant {variation_index + 1}/{n_total_variations}...")
    adversarial_html, injections = generate_and_apply_injections(
        base_html=base_html,
        attack_channels=attack_channels,
        behavior_config=behavior_config,
        llm_client=attacker_llm,
        variation_index=variation_index,
        n_total_variations=n_total_variations,
    )
    logger.info(f"Applied {len(injections)} injection(s)")

    # Save output files
    output_paths = {}
    for subdomain, html in adversarial_html.items():
        path = output_dir / base_html_paths[subdomain].name
        path.write_text(html, encoding="utf-8")
        output_paths[subdomain] = path
        logger.info(f"Saved adversarial variant: {path}")

    return output_paths, injections


def _select_reference_subdomain(subdomains: list[str]) -> str:
    """
    Pick which subdomain to generate first as the reference.

    Prefers root path "/" as it's usually the most complete/representative.
    Otherwise selects the shortest path.
    """
    if "/" in subdomains:
        return "/"
    return min(subdomains, key=lambda s: s.count("/"))


def _subdomain_to_filename(page_id: str, subdomain: str) -> str:
    """Convert subdomain path to safe filename."""
    if subdomain == "/":
        return f"{page_id}.html"
    safe_path = subdomain.strip("/").replace("/", "_").replace("#", "_").replace("?", "_")
    return f"{page_id}_{safe_path}.html"


def _parse_json_response(response: str) -> dict:
    """Extract JSON from LLM response."""
    import re
    response = strip_code_fences(response)
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Try to find JSON block
        match = re.search(r'\{[\s\S]*\}', response)
        if match:
            return json.loads(match.group(0))
        raise
