"""
Generation modules for redteam benchmark.

This module provides:
- LLMClient: Unified interface for OpenAI and Anthropic APIs
- PrefillDataAnalyzer: Analyzes behaviors to determine prefill data requirements
- SyntheticPageGenerator: Generates HTML pages using LLMs
- Site toolbox utilities: Save and load generated sites for reuse
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Literal
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
import anthropic
from agentlab.benchmarks.redteam_prompts import (
    get_prefill_analysis_prompt,
    get_page_generation_prompt,
    get_html_modification_prompt,
)

logger = logging.getLogger(__name__)

TOOLBOX_DIR = Path(__file__).parent.parent.parent / "assets" / "site-toolbox"

MODEL_CONFIGS = {
    "gpt-5": {
        "provider": "openai",
        "api_key_var": "OPENAI_API_KEY",
    },
    "gpt-4o": {
        "provider": "openai",
        "api_key_var": "OPENAI_API_KEY",
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "api_key_var": "OPENAI_API_KEY",
    },
    "claude-sonnet-4-20250514": {
        "provider": "anthropic",
        "api_key_var": "ANTHROPIC_API_KEY",
    },
    "claude-3-5-sonnet-20241022": {
        "provider": "anthropic",
        "api_key_var": "ANTHROPIC_API_KEY",
    },
    "claude-3-opus-20240229": {
        "provider": "anthropic",
        "api_key_var": "ANTHROPIC_API_KEY",
    },
}


def get_model_provider(model: str) -> Literal["openai", "anthropic"]:
    """Determine which provider to use based on model name."""
    # Check exact match first
    if model in MODEL_CONFIGS:
        return MODEL_CONFIGS[model]["provider"]

    # Fallback: check if model name contains provider identifier
    if "claude" in model.lower():
        return "anthropic"
    elif "gpt" in model.lower() or "o1" in model.lower():
        return "openai"

    # Default to OpenAI for backward compatibility
    logger.warning(f"Unknown model '{model}', defaulting to OpenAI provider")
    return "openai"


def strip_code_fences(text: str) -> str:
    """Strip markdown code fences from text."""
    text = text.strip()

    # Remove opening code fence
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```html"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]

    # Remove closing code fence
    if text.endswith("```"):
        text = text[:-3]

    return text.strip()


def upload_screenshots(
    client: OpenAI | anthropic.Anthropic,
    screenshots: List[str] | str,
    provider: Literal["openai", "anthropic"],
) -> List[str]:
    """
    Upload screenshots to appropriate API and return file IDs.

    Args:
        client: API client instance
        screenshots: List of screenshot paths or folder path
        provider: Which provider to use

    Returns:
        List of file IDs
    """
    if provider == "openai":
        return upload_screenshots_openai(client, screenshots)
    else:
        return upload_screenshots_anthropic(client, screenshots)


class LLMClient:
    """Unified client for OpenAI and Anthropic APIs."""

    def __init__(self, model: str):
        """
        Initialize LLM client with automatic provider detection.

        Args:
            model: Model name (e.g., "gpt-5", "claude-sonnet-4-20250514")
        """
        self.model = model
        self.provider = get_model_provider(model)

        if self.provider == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            self.client = anthropic.Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                default_headers={"anthropic-beta": "files-api-2025-04-14"},
            )

    def call_with_images(
        self,
        prompt: str,
        screenshots: Optional[List[str] | str] = None,
        max_tokens: int = 16000,
    ) -> str:
        """
        Call LLM with prompt and optional images.

        Args:
            prompt: Text prompt
            screenshots: Optional screenshot paths (files or folders)
            max_tokens: Maximum tokens in response

        Returns:
            Response text
        """
        screenshots = screenshots or []

        if self.provider == "openai":
            return self._call_openai(prompt, screenshots)
        else:
            return self._call_anthropic(prompt, screenshots, max_tokens)

    def _call_openai(self, prompt: str, screenshots: List[str] | str) -> str:
        """Call OpenAI Responses API."""
        file_ids = upload_screenshots(self.client, screenshots, "openai")

        input_content = [{"type": "input_text", "text": prompt}]
        for file_id in file_ids:
            input_content.append({"type": "input_image", "file_id": file_id})

        response = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": input_content}],
        )

        return getattr(response, "output_text", "")

    def _call_anthropic(
        self,
        prompt: str,
        screenshots: List[str] | str,
        max_tokens: int,
    ) -> str:
        """Call Anthropic Messages API."""
        file_ids = upload_screenshots(self.client, screenshots, "anthropic")

        content = [{"type": "text", "text": prompt}]
        for file_id in file_ids:
            content.append({"type": "image", "source": {"type": "file", "file_id": file_id}})

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": content}],
        )

        return response.content[0].text


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
            # base_metadata=base_metadata,
            new_prefill_spec=new_prefill_spec,
            domain=domain,
            new_behavior=context.get("behavior", ""),
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
        # Avoid double-prepending assets/ if path already contains it
        if str(file_path).startswith("assets/"):
            file_path = Path(file_path)
        else:
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


def upload_screenshots_openai(client: OpenAI, screenshots: List[str] | str) -> List[str]:
    """
    Upload multiple screenshots to OpenAI and return their file IDs.
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


def upload_file_to_anthropic(client: anthropic.Anthropic, file_path: str) -> Optional[str]:
    """
    Upload a file to Anthropic Files API.

    Args:
        client: Anthropic client instance
        file_path: Path to the file (relative or absolute)

    Returns:
        File ID if successful, None otherwise
    """
    # Handle relative paths from assets/ folder
    if not Path(file_path).is_absolute():
        # Avoid double-prepending assets/ if path already contains it
        if str(file_path).startswith("assets/"):
            file_path = Path(file_path)
        else:
            file_path = Path("assets") / file_path

    if not Path(file_path).exists():
        logger.warning(f"Screenshot not found: {file_path}")
        return None

    if Path(file_path).is_dir():
        logger.warning(f"Expected file but got directory: {file_path}")
        return None

    try:
        with open(file_path, "rb") as f:
            response = client.beta.files.upload(
                file=(Path(file_path).name, f, "image/png"),
            )
            logger.info(f"Uploaded {file_path}: {response.id}")
            return response.id
    except Exception as e:
        logger.warning(f"Failed to upload {file_path}: {e}")
        return None


def upload_screenshots_anthropic(
    client: anthropic.Anthropic, screenshots: List[str] | str
) -> List[str]:
    """
    Upload multiple screenshots to Anthropic and return their file IDs.
    Screenshots can be individual file paths or folder paths.
    If folder path, uploads all files in the folder (non-recursive).

    Args:
        client: Anthropic client instance
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

    return [fid for fid in (upload_file_to_anthropic(client, str(p)) for p in paths) if fid]


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
