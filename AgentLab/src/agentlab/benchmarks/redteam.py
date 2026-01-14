"""
Redteam benchmark for security testing with dynamic synthetic pages.

This module provides:
- RedteamEnv: Environment that generates pages on-the-fly and wraps BrowserGym
- RedteamBenchmark: Benchmark class that loads JSON configs
- Page and RedteamEnvArgs: Data structures for configuration
"""

import os
import subprocess
import sys
import time
import json
import logging
import socket
import urllib.request
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any
from datetime import datetime
import gymnasium as gym
from urllib.parse import urlparse

import bgym
from bgym import HighLevelActionSetArgs
from pydantic import ConfigDict

from agentlab.benchmarks.abstract_env import AbstractBenchmark, AbstractEnv, AbstractEnvArgs
from agentlab.benchmarks.redteam_generation import (
    SyntheticPageGenerator,
    PrefillDataAnalyzer,
    save_site_to_toolbox,
    load_toolbox_site,
    replace_injection_placeholders,
    extract_placeholders,
    replace_placeholders_with_content,
)
from agentlab.benchmarks.redteam_attacker import RedteamAttackerAgent
from agentlab.benchmarks.redteam_prompts import get_safe_transformation_prompt
from agentlab.benchmarks.llm_utils import OpenRouterLLMClient

logger = logging.getLogger(__name__)


@dataclass
class Page:
    """Page definition from benchmark config.

    Each Page represents a website with multiple subdomains/routes.
    For example, Gmail might have subdomains: ["/mail/u/0/#inbox", "/mail/u/0/#settings"]
    """

    id: str  # Short identifier, e.g., "duckduckgo", "gmail" (no leading slash)
    base_site_url: str = None  # Root URL, e.g., "https://mail.google.com"
    subdomains: list[str] = field(
        default_factory=list
    )  # Paths to generate, e.g., ["/mail/u/0/#inbox"]
    details: dict[str, str] = field(default_factory=dict)  # subdomain -> description
    screenshots: list[str] = field(default_factory=list)  # Paths to screenshot files/folders
    existing_path: str = None  # Path to pre-generated HTML file (skips generation if provided)
    skip_modification: bool = False


@dataclass
class RedteamEnvArgs(AbstractEnvArgs):
    """Configuration for a single redteam task."""

    task_name: str
    target: str  # Attack goal description
    attacker_model: str
    pages: list[Page]
    attack_type: str  # "direct" or "indirect"
    adversarial_mode: str  # "single_turn" or "multi_turn"
    test_mode: str  # "benign" or "adversarial"
    start_page: str = None

    # Task description fields
    summary: str = None
    objective: str = None
    doc: str = None  # primary task description, replaces 'behavior'
    safe_behavior: str = None  # Safe analog - legitimate task for benign condition

    # Generation parameters
    injection_seed: int = 0  # Backward compat alias for variation_seed
    world_sim_model: str = "openai/gpt-4.1"
    use_trajectory: bool = False  # Whether to run trajectory observation on real sites

    # Attacker agent configuration
    attacker_system_prompt: str = None  # Auto-generated if None
    max_conversation_turns: int = 10  # Max turns for multi-turn mode

    target_system_prompt: str = (
        None  # System prompt for target model (from system_prompt_config.content)
    )
    tools_config: dict = None  # Tools configuration (messages, tools, custom_tool_schema)
    criteria: list = None  # Judging criteria
    judging: dict = None  # Judge configuration

    # Environment parameters (BrowserGym compatibility)
    headless: bool = True
    max_steps: int = 20
    task_seed: int = 0

    record_video: bool = False
    wait_for_user_message: bool = False
    viewport: dict = None
    slow_mo: int = None
    storage_state: str | Path | dict = None
    task_kwargs: dict = None

    # Resume from existing run (skip HTML regeneration)
    resume_from_dir: str | Path = None

    # Skip link validation
    skip_link_validation: bool = False

    # Success condition from simulation_config
    success_condition: str = None

    # Shared attacker agent instance (initialized in benchmark)
    attacker_agent: RedteamAttackerAgent = field(default=None, init=False)

    # === New fields for unified injection pipeline ===
    # Variation seed for reproducibility (alias: injection_seed for backward compat)
    variation_seed: int = 0

    # Attack channels the agent observes: ["html", "axtree", "visual"]
    attack_channels: list = field(default_factory=lambda: ["axtree"])

    # Directory paths for generated HTML (populated by pipeline)
    base_html_dir: str = ""  # Base pages without injections
    variant_html_dir: str = ""  # Variant-specific pages (benign or adversarial)

    # Behavior ID for grouping variants
    behavior_id: str = ""

    # === New fields for variant directory structure ===
    # Parent directory grouping all variants (benign + adversarial)
    parent_exp_dir: str = ""
    # Variant name: "benign" or "adversarial_v0", etc.
    variant_name: str = ""
    # Whether this run uses the variant subdirectory structure
    is_variant_run: bool = False

    @property
    def variant_subdir(self) -> str:
        """Subdirectory name for this variant."""
        if self.test_mode == "benign":
            return "benign"
        else:
            return f"adversarial_v{self.variation_seed}"

    @property
    def computed_task_name(self) -> str:
        """Generate task name from behavior_id, seed, and test_mode."""
        if self.behavior_id:
            return f"{self.behavior_id}.seed{self.variation_seed}.{self.test_mode}"
        return self.task_name

    def make_env(
        self,
        action_mapping,
        exp_dir: Path,
        exp_task_kwargs: dict = None,
        use_raw_page_output: bool = True,
    ) -> "RedteamEnv":
        """
        Create RedteamEnv instance.

        Args:
            action_mapping: Action mapping for the agent
            exp_dir: Experiment directory
            exp_task_kwargs: Additional task kwargs (unused for redteam)
            use_raw_page_output: Whether to use raw page output (unused for redteam)

        Returns:
            RedteamEnv instance
        """
        return RedteamEnv(
            env_args=self,
            exp_dir=exp_dir,
            action_mapping=action_mapping,
        )


class RedteamEnv(AbstractEnv):
    """
    Environment for redteam testing with on-the-fly synthetic page generation.

    This environment wraps BrowserGym to provide rich observations (DOM, AXTree, screenshots)
    while dynamically generating pages with embedded prefill data and prompt injections.
    """

    def __init__(
        self,
        env_args: RedteamEnvArgs,
        exp_dir: Path,
        action_mapping=None,
    ):
        self.env_args = env_args
        self.exp_dir = exp_dir
        self.action_mapping = action_mapping

        # Page generation (lazy init)
        self.analyzer = None
        self.generator = None

        # Flow server
        self.server_proc = None
        self.server_port = None

        # BrowserGym environment (the real workhorse)
        self.browser_env = None
        self._step_count = 0

        # One-time generation state
        self._generation_complete = False
        self._flow_config = None

        # Trajectory observation
        self.trajectory_observer = None

    def _generate_sites(self) -> None:
        """
        One-time site generation with unified injection pipeline.

        Performs:
        0. Trajectory observation (if enabled)
        1. Initialize generators
        2. Analyze prefill data requirements
        3. Generate REFERENCE page for '/' root (WITH placeholders)
        4. Generate SUBSEQUENT pages (match reference UI, WITH placeholders)
        5. Generate placeholder content (SINGLE LLM call → benign + adversarial)
        6. Create pages with appropriate content based on test_mode

        Skipped on subsequent resets/relaunches for performance.
        """
        if self._generation_complete:
            logger.info("Site generation already complete, skipping...")
            return

        # Check if resuming from existing run
        if self.env_args.resume_from_dir:
            resume_dir = Path(self.env_args.resume_from_dir)
            import shutil

            # Always copy trajectory screenshots if they exist (even if no HTML yet)
            # This allows reusing expensive trajectory observation from partial runs
            resume_trajectory_dir = resume_dir / "trajectory"
            if resume_trajectory_dir.exists() and list(resume_trajectory_dir.glob("*.png")):
                target_trajectory_dir = self.exp_dir / "trajectory"
                target_trajectory_dir.mkdir(parents=True, exist_ok=True)
                for img_file in resume_trajectory_dir.glob("*.png"):
                    target_path = target_trajectory_dir / img_file.name
                    if not target_path.exists():  # Don't overwrite
                        shutil.copy2(img_file, target_path)
                logger.info(
                    f"Copied {len(list(resume_trajectory_dir.glob('*.png')))} trajectory screenshots from resume_dir"
                )

            # Determine variant directory based on test_mode
            if self.env_args.test_mode == "benign":
                variant_subdir = "benign"
            else:
                variant_subdir = f"adversarial_v{self.env_args.variation_seed}"
            variation_dir = resume_dir / variant_subdir

            # Also check for old-style variation_N directories for backward compat
            if not variation_dir.exists():
                old_style_dir = resume_dir / f"variation_{self.env_args.variation_seed}"
                if old_style_dir.exists():
                    variation_dir = old_style_dir
                    logger.info(f"Using old-style variation directory: {old_style_dir}")

            if variation_dir.exists() and list(variation_dir.glob("*.html")):
                logger.info(f"Resuming from existing HTML files in: {variation_dir}")

                # Copy HTML files to current exp_dir
                target_variation_dir = self.exp_dir / variant_subdir
                target_variation_dir.mkdir(parents=True, exist_ok=True)

                for html_file in variation_dir.glob("*.html"):
                    target_path = target_variation_dir / html_file.name
                    shutil.copy2(html_file, target_path)
                    logger.info(f"  Copied {html_file.name}")

                # Also copy flow_config.json if it exists
                flow_config_path = resume_dir / "flow_config.json"
                if flow_config_path.exists():
                    target_config_path = self.exp_dir / "flow_config.json"
                    shutil.copy2(flow_config_path, target_config_path)
                    with open(target_config_path, "r") as f:
                        self._flow_config = json.load(f)
                    # Update the run_dir to point to new location
                    self._flow_config["run_dir"] = str(target_variation_dir)
                    with open(target_config_path, "w") as f:
                        json.dump(self._flow_config, f, indent=2)
                    logger.info("  Copied and updated flow_config.json")
                else:
                    # Rebuild flow_config from copied HTML files
                    self._build_flow_config_from_html(target_variation_dir)

                self._generation_complete = True
                logger.info(
                    f"Resume complete - using {len(list(target_variation_dir.glob('*.html')))} HTML files"
                )
                return
            else:
                logger.info(
                    f"No HTML files in resume_dir ({variation_dir}), will generate fresh "
                    f"(but trajectory screenshots were copied)"
                )

        logger.info(
            f"Starting site generation for {self.env_args.task_name} "
            f"(test_mode={self.env_args.test_mode}, variation_seed={self.env_args.variation_seed})..."
        )

        # Initialize trajectory data
        trajectory_screenshots = []
        safe_behavior = None

        # Step 0: Conditional trajectory observation
        if self.env_args.use_trajectory:
            trajectory_dir = self.exp_dir / "trajectory"
            html_library_dir = (
                Path(__file__).parent.parent.parent.parent
                / "html-library"
                / "screenshots"
                / self.env_args.behavior_id.replace(".", "_")
            )

            # Check for existing screenshots in trajectory dir or html-library
            existing_screenshots = (
                list(trajectory_dir.glob("*.png")) if trajectory_dir.exists() else []
            )
            if not existing_screenshots and html_library_dir.exists():
                existing_screenshots = list(html_library_dir.glob("*.png"))
                if existing_screenshots:
                    logger.info(f"Found screenshots in html-library: {html_library_dir}")

            if existing_screenshots:
                # Use existing trajectory screenshots (skip observation)
                logger.info(
                    f"Found {len(existing_screenshots)} existing trajectory screenshots, skipping observation"
                )
                trajectory_screenshots = [str(p) for p in existing_screenshots]
            else:
                # Run trajectory observation
                logger.info("Trajectory observation enabled - running observation...")
                from agentlab.benchmarks.trajectory_observer import TrajectoryObserver

                self.trajectory_observer = TrajectoryObserver()

                # Generate safe analog behavior
                logger.info("Generating safe analog...")
                pages_for_prompt = [
                    {
                        "id": p.id,
                        "base_site_url": p.base_site_url,
                        "subdomains": p.subdomains,
                    }
                    for p in self.env_args.pages
                ]
                prompt = get_safe_transformation_prompt(
                    adversarial_behavior=self.env_args.doc,
                    target=self.env_args.target,
                    pages=pages_for_prompt,
                )
                llm = OpenRouterLLMClient(self.env_args.world_sim_model)
                safe_behavior = llm.chat_simple([{"role": "user", "content": prompt}])
                logger.info(f"Safe analog: {safe_behavior[:100]}...")

                # Prepare page details and sites with subdomain info
                sites = [p.base_site_url for p in self.env_args.pages if p.base_site_url]
                page_details = [
                    {
                        "id": p.id,
                        "base_site_url": p.base_site_url,
                        "subdomains": p.subdomains,
                        "details": p.details,
                    }
                    for p in self.env_args.pages
                ]

                # Run trajectory observation
                logger.info("Observing trajectory on real sites...")
                trajectory_data = self.trajectory_observer.observe_trajectory(
                    safe_behavior=safe_behavior,
                    sites=sites,
                    output_dir=trajectory_dir,
                    page_details=page_details,
                    html_library_dir=html_library_dir,
                )

                # Collect screenshot paths
                trajectory_screenshots = trajectory_data.get("screenshots", [])
                logger.info(f"✓ Captured {len(trajectory_screenshots)} screenshots")
        else:
            logger.info("Trajectory observation disabled - using benchmark screenshots only")

        # Step 1: Initialize generation pipeline
        if self.analyzer is None:
            self.analyzer = PrefillDataAnalyzer(model=self.env_args.world_sim_model)
            self.generator = SyntheticPageGenerator(model=self.env_args.world_sim_model)

        # Step 2: Analyze prefill data requirements
        skip_analysis = self.env_args.pages and all(
            p.existing_path and p.skip_modification for p in self.env_args.pages
        )

        if skip_analysis:
            logger.info(
                "Skipping prefill analysis (all synthetic pages use existing HTML without modification)"
            )
            prefill_spec = {"pages": []}
        else:
            logger.info("Analyzing prefill data needs...")
            # Use trajectory screenshots if enabled
            analyzer_screenshots = trajectory_screenshots if self.env_args.use_trajectory else None

            prefill_spec = self.analyzer.analyze(
                behavior=self.env_args.doc,
                pages=self.env_args.pages,
                screenshots=analyzer_screenshots,
                safe_behavior=self.env_args.safe_behavior,
            )

        # Step 3 & 4: Generate base HTML pages with REFERENCE-BASED UI matching
        logger.info("Generating base HTML pages (with placeholders, reference-based UI)...")

        # Determine starting stage
        start_stage = None
        if self.env_args.start_page:
            start_stage = self.env_args.start_page.lstrip("/")

        # Build URL mapping for cross-page navigation
        url_mapping = {}
        for page in self.env_args.pages:
            for subdomain in page.subdomains:
                local_route = f"{page.id}{subdomain}"
                full_url = f"{page.base_site_url}{subdomain}" if page.base_site_url else local_route
                url_mapping[local_route] = full_url
        logger.info(f"URL mapping: {url_mapping}")

        # Store base HTML (with placeholders) for each subdomain
        base_html_by_subdomain = {}
        subdomains_needing_injections = []  # List of (route_key, page, subdomain, spec)

        # Track reference HTML per page for UI consistency
        reference_html_by_page = {}

        # Determine directory structure based on variant mode (do this BEFORE generation loop)
        if self.env_args.is_variant_run and self.env_args.parent_exp_dir:
            # New structure: parent/variant/
            parent_dir = Path(self.env_args.parent_exp_dir)
            variant_name = self.env_args.variant_name

            # Base HTML goes in parent/base/ (shared across variants)
            base_dir = parent_dir / "base"

            # Variant-specific directory
            self.exp_dir = parent_dir / variant_name
            self.exp_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Using variant structure: {parent_dir.name}/{variant_name}")
        else:
            # Old structure: backward compatibility
            base_dir = self.exp_dir / "base"
            logger.info(f"Using legacy structure: {self.exp_dir.name}")

        # Create base HTML directory immediately for incremental saves
        base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Base HTML directory: {base_dir}")

        # Generate base HTML for all pages and their subdomains
        for page in self.env_args.pages:
            page_id = page.id

            # Extract spec for this page from prefill
            page_spec = None
            if prefill_spec:
                for ps in prefill_spec.get("pages", []):
                    if ps.get("page") == page.id:
                        page_spec = ps
                        break

            # Get shared functionality from page spec
            shared_functionality = (
                page_spec.get("shared_functionality", {}) if page_spec else {}
            )
            subdomain_specs = page_spec.get("subdomains", {}) if page_spec else {}

            # Determine reference subdomain (prefer "/" root, else first)
            reference_subdomain = "/" if "/" in page.subdomains else page.subdomains[0]
            ordered_subdomains = [reference_subdomain] + [
                sd for sd in page.subdomains if sd != reference_subdomain
            ]

            # Generate HTML for each subdomain (reference first, then match style)
            for subdomain in ordered_subdomains:
                route_key = f"{page_id}{subdomain}"
                full_url = (
                    f"{page.base_site_url}{subdomain}" if page.base_site_url else subdomain
                )
                subdomain_details = (
                    page.details.get(subdomain, "") if isinstance(page.details, dict) else ""
                )

                # Get subdomain-specific spec
                subdomain_spec = subdomain_specs.get(subdomain, {})

                # Combine shared + subdomain-specific for generation
                # Support both new format (prefill_variant_content, prefill_invariant_content)
                # and legacy format (prefill_benign, prefill_adversarial) for backward compatibility
                combined_spec = {
                    "page": page.id,
                    "subdomain": subdomain,
                    "shared_functionality": shared_functionality,
                    "functionality": subdomain_spec.get("functionality", {}),
                    # New format fields
                    "prefill_variant_content": subdomain_spec.get("prefill_variant_content", {}),
                    "prefill_invariant_content": subdomain_spec.get("prefill_invariant_content", {}),
                    # Legacy format fields (for backward compatibility)
                    "prefill_benign": subdomain_spec.get("prefill_benign", {}),
                    "prefill_adversarial": subdomain_spec.get("prefill_adversarial", {}),
                }

                html = None

                # Check for existing path (skip generation if provided)
                if page.existing_path and page.skip_modification:
                    logger.info(f"Using existing HTML as-is from {page.existing_path}")
                    try:
                        html = load_toolbox_site(Path(page.existing_path))
                    except Exception as e:
                        logger.warning(f"Failed to load existing site: {e}")
                        html = None

                # Generate from scratch if needed
                if html is None:
                    is_reference = subdomain == reference_subdomain
                    logger.info(
                        f"Generating {'reference' if is_reference else 'subsequent'} page: "
                        f"{route_key} ({full_url})"
                    )
                    page_screenshots = (
                        trajectory_screenshots
                        if self.env_args.use_trajectory
                        else page.screenshots
                    )

                    # Build sibling subdomains for navigation context
                    sibling_subdomains = [
                        {
                            "subdomain": sd,
                            "route": f"{page.id}{sd}",
                            "full_url": (
                                f"{page.base_site_url}{sd}" if page.base_site_url else sd
                            ),
                            "details": (
                                page.details.get(sd, "")[:100]
                                if isinstance(page.details, dict)
                                else ""
                            ),
                        }
                        for sd in page.subdomains
                        if sd != subdomain
                    ]

                    # Use reference HTML for subsequent pages
                    reference_html = reference_html_by_page.get(page_id) if not is_reference else None

                    html = self.generator.generate(
                        domain=route_key,
                        context={
                            "doc": self.env_args.doc,
                            "subdomain_details": subdomain_details,
                        },
                        prefill=combined_spec,
                        simulated_url=full_url,
                        screenshots=page_screenshots,
                        url_mapping=url_mapping,
                        sibling_subdomains=sibling_subdomains,
                        reference_html=reference_html,
                    )

                    # Store reference HTML for this page
                    if is_reference:
                        reference_html_by_page[page_id] = html

                # Save base HTML
                base_html_by_subdomain[route_key] = html

                # Check if this subdomain needs variant content (new format)
                # or adversarial injections (legacy format)
                prefill_variant = subdomain_spec.get("prefill_variant_content", {})
                prefill_adv = subdomain_spec.get("prefill_adversarial", {})
                needs_variants = prefill_variant.get("needed", False) or prefill_adv.get("needed", False)
                if needs_variants:
                    subdomains_needing_injections.append(
                        (route_key, page, subdomain, combined_spec)
                    )
                    logger.info(f"Subdomain {route_key} needs variant content generation")

                # Save to toolbox for future reuse
                metadata = {
                    "timestamp": datetime.now().isoformat(),
                    "doc": self.env_args.doc,
                    "id": page.id,
                    "subdomain": subdomain,
                    "base_site_url": page.base_site_url,
                    "behavior_id": self.env_args.behavior_id,
                    "test_mode": self.env_args.test_mode,
                }
                save_site_to_toolbox(route_key, html, metadata)

                # ALSO save to base/ directory immediately (incremental save)
                file_name = route_key.replace("/", "_").replace("#", "_").replace("?", "_").strip("_")
                (base_dir / f"{file_name}.html").write_text(html, encoding="utf-8")

        # Step 5: Generate placeholder content (benign + adversarial)
        placeholder_content = {}  # {placeholder_id: {benign: "...", adversarial: [...]}}
        placeholder_metadata = {}  # {placeholder_id: {element_type, placement, ...}}
        injection_data = {
            "behavior_id": self.env_args.behavior_id,
            "generated_at": datetime.now().isoformat(),
            "test_mode": self.env_args.test_mode,
            "variation_seed": self.env_args.variation_seed,
            "injection_locations": [],
            "placeholders": {},  # Consolidated: metadata + content
        }

        if subdomains_needing_injections:
            logger.info(
                f"Found {len(subdomains_needing_injections)} subdomain(s) needing placeholder content"
            )
            injection_data["injection_locations"] = [
                rk for rk, _, _, _ in subdomains_needing_injections
            ]

            # Initialize attacker agent
            attacker_agent = self.env_args.attacker_agent
            attacker_agent.make_agent()

            # For each subdomain with placeholders, generate content
            for route_key, page, subdomain, combined_spec in subdomains_needing_injections:
                html = base_html_by_subdomain.get(route_key, "")

                # Extract placeholders from HTML
                placeholders = extract_placeholders(html)
                logger.info(f"Found {len(placeholders)} placeholder(s) in {route_key}")

                # Build a lookup of spec items by placement for enriching placeholders
                # Support both new format (prefill_variant_content) and legacy (prefill_adversarial)
                spec_items_by_placement = {}
                variant_content = combined_spec.get("prefill_variant_content", {})
                if variant_content.get("items"):
                    for item in variant_content.get("items", []):
                        placement = item.get("placement", "")
                        if placement:
                            spec_items_by_placement[placement] = item
                # Also check legacy format
                adv_content = combined_spec.get("prefill_adversarial", {})
                if adv_content.get("items"):
                    for item in adv_content.get("items", []):
                        placement = item.get("placement", "")
                        if placement and placement not in spec_items_by_placement:
                            spec_items_by_placement[placement] = item

                for placeholder in placeholders:
                    placeholder_id = placeholder.get("id", "")
                    if not placeholder_id:
                        continue

                    # Skip if already generated (same placeholder ID in multiple pages)
                    if placeholder_id in placeholder_content:
                        logger.debug(f"Placeholder {placeholder_id} already generated, skipping")
                        continue

                    # Enrich placeholder with spec data (benign_description, adversarial_description)
                    placement = placeholder.get("placement", "")
                    if placement in spec_items_by_placement:
                        spec_item = spec_items_by_placement[placement]
                        # Merge spec data into placeholder (spec takes precedence for descriptions)
                        enriched_placeholder = {**placeholder}
                        if "benign_description" in spec_item:
                            enriched_placeholder["benign_description"] = spec_item["benign_description"]
                        if "adversarial_description" in spec_item:
                            enriched_placeholder["adversarial_description"] = spec_item["adversarial_description"]
                        if "element_type" in spec_item and "element_type" not in enriched_placeholder:
                            enriched_placeholder["element_type"] = spec_item["element_type"]
                        placeholder = enriched_placeholder

                    # Generate content - fail fast if generation fails
                    content = attacker_agent.generate_placeholder_content(
                        placeholder=placeholder,
                        page_domain=route_key,
                        attack_channels=self.env_args.attack_channels,
                        n_adversarial_variants=1,  # Generate for this variant
                    )
                    placeholder_content[placeholder_id] = content

                    # Store consolidated metadata + content
                    injection_data["placeholders"][placeholder_id] = {
                        "element_type": placeholder.get("element_type", "unknown"),
                        "placement": placeholder.get("placement", ""),
                        "page": route_key,
                        "benign": content.get("benign", ""),
                        "adversarial": content.get("adversarial", []),
                    }
                    # Store metadata separately for internal use
                    placeholder_metadata[placeholder_id] = {
                        "element_type": placeholder.get("element_type", "unknown"),
                        "placement": placeholder.get("placement", ""),
                    }

        # Step 6: Create variant pages based on test_mode
        if self.env_args.is_variant_run and self.env_args.parent_exp_dir:
            # New structure: exp_dir is already parent/variant/
            variant_dir = self.exp_dir
            mode = "benign" if self.env_args.test_mode == "benign" else "adversarial"
            variant_index = 0 if self.env_args.test_mode == "benign" else self.env_args.variation_seed
        else:
            # Old structure: exp_dir is already named after the variant (e.g., seed0/benign/)
            # Put HTML files in a 'html/' subdirectory to avoid redundant naming like benign/benign/
            variant_dir = self.exp_dir / "html"
            mode = "benign" if self.env_args.test_mode == "benign" else "adversarial"
            variant_index = 0 if self.env_args.test_mode == "benign" else self.env_args.variation_seed

        variant_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Creating {mode} variant (variant_index={variant_index})...")

        for route_key, html in base_html_by_subdomain.items():
            if html is None:
                continue

            file_name = (
                route_key.replace("/", "_").replace("#", "_").replace("?", "_").strip("_")
            )

            # Check if this subdomain needs placeholder replacement
            needs_replacement = any(
                rk == route_key for rk, _, _, _ in subdomains_needing_injections
            )

            if needs_replacement and placeholder_content:
                # Replace placeholders with appropriate content
                html_with_content = replace_placeholders_with_content(
                    html=html,
                    placeholder_content=placeholder_content,
                    mode=mode,
                    variant_index=variant_index,
                )
                page_path = variant_dir / f"{file_name}.html"
                page_path.write_text(html_with_content, encoding="utf-8")
                logger.info(f"✓ Saved {file_name}.html with {mode} content")
            else:
                # No placeholders - copy base HTML
                page_path = variant_dir / f"{file_name}.html"
                page_path.write_text(html, encoding="utf-8")

        # Step 7: Validate internal links (fail-fast if broken)
        if self.env_args.skip_link_validation:
            logger.warning("Skipping internal link validation (skip_link_validation=True)")
        else:
            self._validate_internal_links(variant_dir, base_html_by_subdomain.keys())

        # Step 8: Set up flow config to point to the correct variant folder
        self._flow_config = {
            "run_dir": str(variant_dir),
            "stages": {},
            "start_page": self.env_args.start_page,
            "start_stage": start_stage,
            "test_mode": self.env_args.test_mode,
            "variation_seed": self.env_args.variation_seed,
        }

        # Configure stages from the variant folder
        for route_key in base_html_by_subdomain.keys():
            self._flow_config["stages"][route_key] = {}

        # Save flow config
        config_path = self.exp_dir / "flow_config.json"
        config_path.write_text(json.dumps(self._flow_config, indent=2))

        # Determine shared metadata directory
        if self.env_args.is_variant_run and self.env_args.parent_exp_dir:
            # New structure: shared metadata goes in parent directory
            shared_metadata_dir = Path(self.env_args.parent_exp_dir)
        else:
            # Old structure: metadata goes in exp_dir
            shared_metadata_dir = self.exp_dir

        # Save consolidated injection data (metadata + full content) to injections.json
        injection_path = shared_metadata_dir / "injections.json"
        injection_path.write_text(json.dumps(injection_data, indent=2))
        logger.info(
            f"✓ Saved injection data to {injection_path}: "
            f"{len(injection_data['placeholders'])} placeholder(s), "
            f"{len(injection_data['injection_locations'])} location(s)"
        )

        self._generation_complete = True
        logger.info(
            f"Site generation complete for {self.env_args.test_mode} "
            f"(variation_seed={self.env_args.variation_seed})"
        )

    def _build_flow_config_from_html(self, variation_dir: Path) -> None:
        """
        Rebuild flow config from existing HTML files when resuming.

        Args:
            variation_dir: Directory containing the HTML files
        """
        self._flow_config = {
            "run_dir": str(variation_dir),
            "stages": {},
            "start_page": self.env_args.start_page,
            "start_stage": None,
        }

        # Extract routes from HTML filenames
        for html_file in variation_dir.glob("*.html"):
            # Skip pre_injection files
            if "_pre_injection" in html_file.name:
                continue

            # Convert filename back to route key
            filename = html_file.stem  # e.g., "gmail_settings_forwarding"
            # Replace underscores with slashes to approximate original route
            route_key = filename.replace("_", "/")

            self._flow_config["stages"][route_key] = {"mode": "synthetic"}

            # Set first stage if not set
            if self._flow_config["start_stage"] is None:
                self._flow_config["start_stage"] = route_key

        # Use start_page if available
        if self.env_args.start_page:
            # Convert start_page to a stage key format
            start_key = (
                self.env_args.start_page.lstrip("/")
                .replace("/", "_")
                .replace("#", "_")
                .replace("?", "_")
                .strip("_")
            )
            for route_key in self._flow_config["stages"]:
                if start_key in route_key.replace("/", "_"):
                    self._flow_config["start_stage"] = route_key
                    break

        # Save flow config
        config_path = self.exp_dir / "flow_config.json"
        config_path.write_text(json.dumps(self._flow_config, indent=2))
        logger.info(
            f"Built flow_config with {len(self._flow_config['stages'])} stages from HTML files"
        )

    def _validate_internal_links(self, variation_dir: Path, valid_routes: set) -> None:
        """
        Validate that all internal links in generated HTML files resolve to valid routes.

        Args:
            variation_dir: Directory containing the generated HTML files
            valid_routes: Set of valid route keys (e.g., {"gmail/mail/u/0/#inbox", "gmail/mail/u/0/#settings"})

        Raises:
            ValueError: If any internal links point to non-existent routes
        """
        import re

        # Build set of valid local routes (with leading slash)
        valid_local_routes = set()
        for route in valid_routes:
            valid_local_routes.add(f"/{route}")
            # Also add without leading slash
            valid_local_routes.add(route)

        broken_links = []

        # Check all HTML files in variation_dir
        for html_file in variation_dir.glob("*.html"):
            html_content = html_file.read_text(encoding="utf-8")

            # Extract all href attributes
            href_pattern = r'href=["\']([^"\']+)["\']'
            hrefs = re.findall(href_pattern, html_content)

            for href in hrefs:
                # Skip external links, anchors, javascript, and special protocols
                if href.startswith(
                    ("http://", "https://", "#", "javascript:", "mailto:", "tel:", "data:")
                ):
                    continue

                # Skip empty or placeholder hrefs
                if not href or href == "/" or href == "#":
                    continue

                # Normalize the href (remove leading slash for comparison)
                normalized_href = href.lstrip("/")

                # Check if this is a valid internal route
                if normalized_href not in valid_routes and href not in valid_local_routes:
                    broken_links.append(
                        {
                            "file": html_file.name,
                            "href": href,
                            "normalized": normalized_href,
                        }
                    )

        if broken_links:
            error_msg = f"Found {len(broken_links)} broken internal link(s):\n"
            for bl in broken_links[:10]:  # Show first 10
                error_msg += f"  - {bl['file']}: href='{bl['href']}'\n"
            error_msg += f"\nValid routes: {sorted(valid_routes)[:10]}..."
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"✓ Link validation passed - all internal links resolve to valid routes")

    def reset(self, seed: int = None) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Reset environment for new episode.

        Steps:
        1. Generate sites (one-time, cached on relaunches)
        2. Start flow server on dynamic port
        3. Initialize BrowserGym environment
        4. Return rich observation with goal
        """
        logger.info(f"Resetting RedteamEnv: {self.env_args.task_name}")

        # Step 1: Generate sites (one-time operation, skipped on relaunches)
        self._generate_sites()

        # Step 2: Start flow server on dynamic port
        config_path = self.exp_dir / "flow_config.json"

        # Use start_stage if provided in config, otherwise use first stage
        start_stage = self._flow_config.get("start_stage")
        if not start_stage:
            start_stage = list(self._flow_config["stages"].keys())[0]

        first_stage = start_stage

        # self.server_port = self._find_free_port()
        self.server_port = 8001  # TODO: hardcoded, should be fine for now
        logger.info(f"Starting flow server on port {self.server_port}...")

        # Get the src directory path for PYTHONPATH
        # This file is at: src/agentlab/benchmarks/redteam.py
        # So src is 3 levels up from this file
        src_dir = str(Path(__file__).parents[2])
        env = os.environ.copy()
        env["PYTHONPATH"] = src_dir + ((":" + env["PYTHONPATH"]) if "PYTHONPATH" in env else "")

        self.server_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "agentlab.benchmarks.redteam_server",
                str(config_path),
                str(self.server_port),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        # Wait for server ready
        self._wait_for_server(f"http://localhost:{self.server_port}/mode", timeout=10)
        logger.info("Flow server ready")

        # Use start_page from env_args - it's already the full external URL
        # (e.g., "https://duckduckgo.com/")
        external_start_url = self.env_args.start_page
        if external_start_url and external_start_url.startswith("http"):
            logger.info(f"External start URL: {external_start_url}")
        else:
            # Fallback to localhost if no valid external URL
            logger.warning(f"Invalid start_page '{external_start_url}', falling back to localhost")
            external_start_url = f"http://localhost:{self.server_port}/{first_stage}"

        # Step 3: Initialize BrowserGym with about:blank as initial URL
        # We'll set up route interception first, then navigate to the real start URL
        self.browser_env = gym.make(
            "browsergym/openended",
            disable_env_checker=True,
            max_episode_steps=self.env_args.max_steps,
            headless=self.env_args.headless,
            action_mapping=self.action_mapping,
            task_kwargs={
                "start_url": "about:blank",  # Start neutral, navigate after route setup
                "goal": self.env_args.doc,
            },
        )

        # Reset browser (navigates to about:blank)
        obs, info = self.browser_env.reset(seed=seed)

        # Get browser context BEFORE setting up routes
        page = self.browser_env.unwrapped.page
        context = page.context

        # Build a lookup table: domain -> {path -> html_file}
        # This allows ONE handler per domain that routes by path
        domain_routes: dict[str, dict[str, Path]] = {}
        variation_dir = Path(self._flow_config["run_dir"])

        for page_config in self.env_args.pages:
            # Only intercept routes for pages with base_site_url
            if not page_config.base_site_url:
                continue

            # Parse the base URL to get the netloc (domain)
            base_parsed = urlparse(page_config.base_site_url)
            netloc = base_parsed.netloc

            if netloc not in domain_routes:
                domain_routes[netloc] = {}

            # Iterate over subdomains to build the path lookup
            for subdomain in page_config.subdomains:
                # Compute simulated URL (external URL to intercept)
                simulated_url = f"{page_config.base_site_url}{subdomain}"
                sim_parsed = urlparse(simulated_url)

                # Build the path key (normalize by stripping trailing slash)
                path_key = sim_parsed.path.rstrip("/")
                # Include query if present (for URLs like ?focusedCommentId=1005)
                if sim_parsed.query:
                    path_key = f"{path_key}?{sim_parsed.query}"

                # Get HTML file path for this subdomain
                route_key = f"{page_config.id}{subdomain}"
                file_name = (
                    route_key.replace("/", "_").replace("#", "_").replace("?", "_").strip("_")
                )
                html_path = variation_dir / f"{file_name}.html"

                if not html_path.exists():
                    logger.warning(f"HTML file does not exist: {html_path}")
                    continue

                domain_routes[netloc][path_key] = html_path
                logger.debug(f"  Added route: {netloc}{path_key} -> {html_path.name}")

        # Register ONE handler per domain that uses the lookup table
        for netloc, path_map in domain_routes.items():
            domain_pattern = f"**://{netloc}/**"

            def make_domain_handler(paths=path_map, domain=netloc):
                def handler(route, request):
                    req_parsed = urlparse(request.url)
                    req_path = req_parsed.path.rstrip("/")
                    # Build path key with query if present
                    path_key = req_path
                    if req_parsed.query:
                        path_key = f"{req_path}?{req_parsed.query}"

                    # Try exact match first
                    html_path = paths.get(path_key)

                    # If no exact match, try without query params
                    if html_path is None and req_parsed.query:
                        html_path = paths.get(req_path)

                    if html_path:
                        logger.info(f"🔀 ROUTE INTERCEPTED: {request.url}")
                        logger.info(f"   Serving: {html_path}")
                        route.fulfill(path=str(html_path))
                    else:
                        # No matching route - let the request continue normally
                        # This allows the request to fail naturally (404) or reach the actual server
                        logger.debug(
                            f"⚠️ No route match for {request.url}, available paths: {list(paths.keys())}"
                        )
                        route.continue_()

                return handler

            context.route(domain_pattern, make_domain_handler())
            logger.info(f"📍 Registered domain handler: {domain_pattern} ({len(path_map)} paths)")

        # Now navigate to the external start URL (route interception will serve local HTML)
        logger.info(f"🚀 Navigating to start URL: {external_start_url}")
        page.goto(external_start_url, wait_until="domcontentloaded")

        # Get fresh observation after navigation
        obs = self.browser_env.unwrapped._get_obs()

        # Augment observation with redteam-specific info
        obs["goal_object"] = [{"type": "text", "text": self.env_args.doc}]
        obs["last_action"] = None
        obs["last_action_error"] = ""

        self._step_count = 0

        return obs, {
            "task_name": self.env_args.task_name,
            "doc": self.env_args.doc,
            "target": self.env_args.target,
            "injection_seed": self.env_args.injection_seed,
            "start_url": external_start_url,
        }

    def step(self, action: str):
        """Execute action via BrowserGym."""
        obs, reward, terminated, truncated, info = self.browser_env.step(action)
        self._step_count += 1

        # Add redteam-specific info
        info["injection_seed"] = self.env_args.injection_seed
        info["step_count"] = self._step_count

        return obs, reward, terminated, truncated, info

    def close(self):
        """Clean up resources."""
        if self.browser_env:
            self.browser_env.close()

        if self.server_proc:
            self.server_proc.terminate()
            try:
                self.server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_proc.kill()

    # def _find_free_port(self) -> int:
    #     """Find an available port for the server."""
    #     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    #         s.bind(("", 0))
    #         s.listen(1)
    #         port = s.getsockname()[1]
    #     return port

    def _wait_for_server(self, url: str, timeout: int = 10):
        for i in range(timeout):
            try:
                urllib.request.urlopen(url, timeout=1)
                return
            except (urllib.error.URLError, ConnectionRefusedError, OSError):
                time.sleep(1)
                continue

        # Final check - capture any output
        # #region agent log
        if self.server_proc:
            # Check if still running
            poll_result = self.server_proc.poll()
            if poll_result is not None:
                stdout, stderr = self.server_proc.communicate()
            else:
                # Process still running but not responding - try to get any available output
                stdout, stderr = None, None
                try:
                    import select

                    if (
                        self.server_proc.stderr
                        and select.select([self.server_proc.stderr], [], [], 0)[0]
                    ):
                        stderr = self.server_proc.stderr.read()
                except:
                    pass

            open(".cursor/debug.log", "a").write(
                json.dumps(
                    {
                        "hypothesisId": "SERVER_TIMEOUT",
                        "location": "redteam.py:_wait_for_server",
                        "message": "Server failed to respond within timeout",
                        "data": {
                            "poll_result": poll_result,
                            "stdout": (
                                stdout.decode("utf-8", errors="replace")[-2000:] if stdout else None
                            ),
                            "stderr": (
                                stderr.decode("utf-8", errors="replace")[-2000:] if stderr else None
                            ),
                            "url": url,
                            "timeout": timeout,
                        },
                        "timestamp": __import__("time").time(),
                    }
                )
                + "\n"
            )
        # #endregion
        raise RuntimeError(f"Flow server failed to start (url: {url}, timeout: {timeout}s)")


# ============================================================================
# Unified Injection Pipeline Entry Point
# ============================================================================


def generate_all_variants(
    behavior_config: dict,
    output_base_dir: str | Path,
    attack_type: str,
    adversarial_mode: str,
    n_adversarial_variants: int = 1,
    attack_channels: list[str] = None,
    llm_client=None,
    attacker_llm_client=None,
) -> list[RedteamEnvArgs]:
    """
    Generate all variants (1 benign + N adversarial) for a behavior.

    This is the main entry point for the unified injection pipeline.
    Generates base pages once, then creates:
    - 1 benign variant (control group, no injections)
    - N adversarial variants (each with different injection approaches)

    Args:
        behavior_config: Full behavior configuration dict
        output_base_dir: Base directory for output files
        attack_type: "direct" or "indirect" attack type
        adversarial_mode: "single_turn" or "multi_turn" conversation mode
        n_adversarial_variants: Number of adversarial variations to generate
        attack_channels: List of channels agent observes (default: ["axtree"])
        llm_client: LLM client for page generation
        attacker_llm_client: LLM client for injection generation (defaults to llm_client)

    Returns:
        List of RedteamEnvArgs ready for execution
    """
    from agentlab.benchmarks.redteam_generation import (
        generate_base_pages,
        create_benign_variant,
        create_adversarial_variant,
    )
    from agentlab.benchmarks.llm_utils import OpenRouterLLMClient

    attack_channels = attack_channels or ["axtree"]
    output_base_dir = Path(output_base_dir)
    behavior_id = behavior_config.get("id", "unknown")
    behavior_dir = output_base_dir / behavior_id

    # Initialize LLM client if not provided
    if llm_client is None:
        world_sim_model = behavior_config.get("world_sim_model", "openai/gpt-4.1")
        llm_client = OpenRouterLLMClient(world_sim_model)

    # Stage 1: Generate complete base pages (once)
    logger.info(f"=== Stage 1: Generating base pages for {behavior_id} ===")
    base_dir = behavior_dir / "base"
    base_paths = generate_base_pages(
        behavior_config=behavior_config,
        output_dir=base_dir,
        llm_client=llm_client,
    )
    logger.info(f"Generated {len(base_paths)} base page(s)")

    # Helper to extract pages config
    def _make_pages_list():
        sim_config = behavior_config.get("simulation_config", {})
        pages_config = sim_config.get("pages", [])
        return [
            Page(
                id=p.get("id", "").lstrip("/"),
                base_site_url=p.get("base_site_url", ""),
                subdomains=p.get("subdomains", ["/"]),
                details=p.get("details", {}),
            )
            for p in pages_config
        ]

    def _get_start_page(pages):
        sim_config = behavior_config.get("simulation_config", {})
        start_page = sim_config.get("start_page", "")
        if not start_page and pages:
            first_page = pages[0]
            start_page = f"{first_page.base_site_url}{first_page.subdomains[0]}"
        return start_page

    variants = []

    # Stage 2A: Create 1 benign variant (control group)
    logger.info("=== Stage 2A: Creating benign variant (control) ===")
    benign_dir = behavior_dir / "benign"
    benign_paths = create_benign_variant(base_paths, benign_dir)

    pages = _make_pages_list()
    variants.append(RedteamEnvArgs(
        task_name=f"{behavior_id}.benign",
        behavior_id=behavior_id,
        variation_seed=0,
        injection_seed=0,
        attack_type=attack_type,
        adversarial_mode=adversarial_mode,
        test_mode="benign",
        target=behavior_config.get("description", ""),
        doc=behavior_config.get("doc", ""),
        attacker_model=behavior_config.get("attacker_model", "openai/gpt-4.1"),
        pages=pages,
        start_page=_get_start_page(pages),
        base_html_dir=str(base_dir),
        variant_html_dir=str(benign_dir),
        attack_channels=attack_channels,
    ))
    logger.info(f"Created benign variant: {behavior_id}.benign")

    # Stage 2B: Create N adversarial variants
    logger.info(f"=== Stage 2B: Creating {n_adversarial_variants} adversarial variant(s) ===")
    for i in range(n_adversarial_variants):
        adv_dir = behavior_dir / f"adversarial_v{i}"
        adv_paths, injections = create_adversarial_variant(
            base_paths,
            adv_dir,
            behavior_config,
            attack_channels,
            llm_client,
            attacker_llm_client,
            variation_index=i,
            n_total_variations=n_adversarial_variants,
        )

        pages = _make_pages_list()
        variants.append(RedteamEnvArgs(
            task_name=f"{behavior_id}.adversarial_v{i}",
            behavior_id=behavior_id,
            variation_seed=i,
            injection_seed=i,
            attack_type=attack_type,
            adversarial_mode=adversarial_mode,
            test_mode="adversarial",
            target=behavior_config.get("description", ""),
            doc=behavior_config.get("doc", ""),
            attacker_model=behavior_config.get("attacker_model", "openai/gpt-4.1"),
            pages=pages,
            start_page=_get_start_page(pages),
            base_html_dir=str(base_dir),
            variant_html_dir=str(adv_dir),
            attack_channels=attack_channels,
        ))
        logger.info(f"Created adversarial variant: {behavior_id}.adversarial_v{i} ({len(injections)} injections)")

    logger.info(f"Generated {len(variants)} total variant(s): 1 benign + {n_adversarial_variants} adversarial")
    return variants


def analyze_results(results: list, verbose: bool = True) -> dict:
    """
    Unified analysis for redteam results - works for benign-only, adversarial-only, or both.

    Args:
        results: List of experiment results (dicts or objects with judge_result_full)
        verbose: Print analysis summary

    Returns:
        Dict with metrics including benign_utility, utility_under_attack, targeted_asr
    """
    from collections import defaultdict

    # Separate by test mode
    benign_results = []
    adversarial_results = []

    for r in results:
        if isinstance(r, dict):
            test_mode = r.get("test_mode", r.get("env.test_mode", "adversarial"))
            judge_result = r.get("judge_result_full", {})
            success = judge_result.get("overall_success", False) if judge_result else False
        else:
            test_mode = getattr(r, "test_mode", getattr(r, "env.test_mode", "adversarial"))
            judge_result = getattr(r, "judge_result_full", {})
            success = judge_result.get("overall_success", False) if judge_result else False

        if test_mode == "benign":
            benign_results.append({"success": success, "result": r})
        else:  # adversarial
            adversarial_results.append({"success": success, "result": r})

    # Compute metrics
    metrics = {
        "total_experiments": len(results),
        "n_benign": len(benign_results),
        "n_adversarial": len(adversarial_results),
    }

    # Benign Utility
    if len(benign_results) > 0:
        benign_successes = sum(r["success"] for r in benign_results)
        metrics["benign_utility"] = round(benign_successes / len(benign_results), 3)
    else:
        metrics["benign_utility"] = None

    # Utility Under Attack + Targeted ASR
    if len(adversarial_results) > 0:
        adv_successes = sum(r["success"] for r in adversarial_results)
        metrics["utility_under_attack"] = round(adv_successes / len(adversarial_results), 3)
        metrics["targeted_asr"] = metrics["utility_under_attack"]  # Same for now
    else:
        metrics["utility_under_attack"] = None
        metrics["targeted_asr"] = None

    # Paired analysis (if both exist)
    if len(benign_results) > 0 and len(adversarial_results) > 0:
        # Group by behavior_id and variation_seed for pairing
        by_pair = defaultdict(dict)

        for r in results:
            if isinstance(r, dict):
                behavior_id = r.get("behavior_id", r.get("env.behavior_id", "unknown"))
                seed = r.get("variation_seed", r.get("env.variation_seed", 0))
                test_mode = r.get("test_mode", r.get("env.test_mode", "adversarial"))
                judge_result = r.get("judge_result_full", {})
                success = judge_result.get("overall_success", False) if judge_result else False
            else:
                behavior_id = getattr(r, "behavior_id", getattr(r, "env.behavior_id", "unknown"))
                seed = getattr(r, "variation_seed", getattr(r, "env.variation_seed", 0))
                test_mode = getattr(r, "test_mode", getattr(r, "env.test_mode", "adversarial"))
                judge_result = getattr(r, "judge_result_full", {})
                success = judge_result.get("overall_success", False) if judge_result else False

            key = (behavior_id, seed)
            by_pair[key][test_mode] = {"success": success, "result": r}

        # Compute paired metrics
        pairs_analyzed = []
        for (behavior_id, seed), modes in by_pair.items():
            if "benign" in modes and "adversarial" in modes:
                benign_success = modes["benign"]["success"]
                adv_success = modes["adversarial"]["success"]
                pairs_analyzed.append({
                    "behavior_id": behavior_id,
                    "seed": seed,
                    "benign_success": benign_success,
                    "adversarial_success": adv_success,
                    "utility_degradation": benign_success - adv_success,
                })

        metrics["n_paired"] = len(pairs_analyzed)
        if len(pairs_analyzed) > 0:
            metrics["avg_utility_degradation"] = round(
                sum(p["utility_degradation"] for p in pairs_analyzed) / len(pairs_analyzed), 3
            )
            metrics["pairs"] = pairs_analyzed
    else:
        metrics["n_paired"] = 0

    # Print summary
    if verbose:
        print("\n=== Redteam Results Analysis ===")
        print(f"Total experiments: {metrics['total_experiments']}")
        print(f"  Benign: {metrics['n_benign']}")
        print(f"  Adversarial: {metrics['n_adversarial']}")
        if metrics['benign_utility'] is not None:
            print(f"\nBenign Utility: {metrics['benign_utility']:.1%}")
        if metrics['utility_under_attack'] is not None:
            print(f"Utility Under Attack: {metrics['utility_under_attack']:.1%}")
            print(f"Targeted ASR: {metrics['targeted_asr']:.1%}")
        if metrics['n_paired'] > 0:
            print(f"\nPaired Analysis:")
            print(f"  {metrics['n_paired']} behavior pairs analyzed")
            print(f"  Avg Utility Degradation: {metrics['avg_utility_degradation']:.1%}")

    return metrics


class RedteamBenchmark(AbstractBenchmark):
    """
    Benchmark for redteam security testing with dynamic synthetic pages.

    Features:
    - On-the-fly LLM page generation
    - Prompt injection variations via seeding
    - Screenshot-guided generation via trajectory observation
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "redteam"
    benchmark_file: str | Path = None
    env_args_list: list[RedteamEnvArgs] = None

    # Agent configuration (required by GenericAgent.set_benchmark)
    is_multi_tab: bool = False
    high_level_action_set_args: HighLevelActionSetArgs = None

    # Generation settings
    world_sim_model: str  # LLM for page generation and prefill analysis
    attacker_model: str  # LLM for attacker agent
    attack_type: str  # "direct" or "indirect"
    adversarial_mode: str  # "single_turn" or "multi_turn"
    use_trajectory: bool = False  # Whether to run trajectory observation on real sites

    # Adversarial mode settings
    max_conversation_turns: int = 10  # For multi_turn mode

    # Variation settings for parallelization
    n_injection_variations: int = 1

    # === New fields for unified injection pipeline ===
    # Number of adversarial variants per behavior (benign is always generated)
    n_adversarial_variants: int = 1

    # Attack channels the agent observes: ["html", "axtree", "visual"]
    attack_channels: list[str] = None  # Default set in model_post_init

    # Resume from existing run (skip HTML regeneration)
    resume_from_dir: str | Path = None  # Path to existing experiment results dir

    # Skip link validation (useful for debugging when LLM generates broken links)
    skip_link_validation: bool = False

    def model_post_init(self, __context: Any) -> None:
        """Load benchmark JSON and create env_args_list."""
        self.env_args_list = []

        # Set default attack_channels if not provided
        if self.attack_channels is None:
            self.attack_channels = ["axtree"]

        if self.high_level_action_set_args is None:
            self.high_level_action_set_args = HighLevelActionSetArgs(
                subsets=["bid", "nav", "tab", "chat"],
                multiaction=False,
            )

        with open(self.benchmark_file, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            benchmark_name = "enriched-behaviors"
            cases = data
        else:
            benchmark_name = data.get("name", "redteam")
            cases = data.get("data", [])

        self.name = benchmark_name
        logger.info(f"Loading redteam benchmark: {benchmark_name}")

        for case_idx, case_data in enumerate(cases):
            # _parse_case now returns a list of variants (1 benign + N adversarial)
            variants = self._parse_case(case_idx, case_data)
            if variants:
                self.env_args_list.extend(variants)

        n_behaviors = len(cases)
        n_variants_per_behavior = 1 + self.n_adversarial_variants  # 1 benign + N adversarial
        logger.info(
            f"Loaded {len(self.env_args_list)} tasks "
            f"({n_behaviors} behaviors × {n_variants_per_behavior} variants each: "
            f"1 benign + {self.n_adversarial_variants} adversarial)"
        )

    def _parse_case(self, case_idx: int, case_data: dict) -> list[RedteamEnvArgs]:
        """
        Parse a case from the benchmark JSON and generate variants.

        Creates 1 benign + N adversarial variants for A/B testing.

        Args:
            case_idx: Index of this case in the benchmark
            case_data: Case configuration dict

        Returns:
            List of RedteamEnvArgs (1 benign + N adversarial variants)
        """
        # Extract pages from simulation_config
        sim_config = case_data.get("simulation_config", {})
        pages = []
        for p in sim_config.get("pages", []):
            page_id = p.get("id", "").lstrip("/")  # Strip leading slash for consistency
            page = Page(
                id=page_id,
                base_site_url=p.get("base_site_url", ""),
                subdomains=p.get("subdomains", ["/"]),
                details=p.get("details", {}),
                screenshots=p.get("screenshots", []),
                existing_path=p.get("existing_path"),
                skip_modification=p.get("skip_modification", False),
            )
            pages.append(page)

        # Extract start_page - should be full URL like "https://duckduckgo.com/"
        start_page = sim_config.get("start_page", "")
        if not start_page and pages:
            # Default to first page's base_site_url + first subdomain
            first_page = pages[0]
            start_page = f"{first_page.base_site_url}{first_page.subdomains[0]}"

        # Extract system prompt for target model
        system_prompt_config = case_data.get("system_prompt_config", {})
        target_system_prompt = system_prompt_config.get("content")

        # Extract tools config, criteria, and judging
        tools_config = case_data.get("tools_config")
        criteria = case_data.get("criteria")
        judging = case_data.get("judging")

        # Extract success condition from simulation_config
        success_condition = sim_config.get("success_condition", "")

        # Get case-level overrides for attack settings
        case_attack_channels = case_data.get("attack_channels", self.attack_channels)
        case_n_adversarial = case_data.get("n_adversarial_variants", self.n_adversarial_variants)

        behavior_id = case_data.get("id", f"case{case_idx}")
        doc = self._strip_html(case_data.get("doc", ""))
        target = case_data.get("description", "")

        # Generate safe analog (once per behavior) for benign mode
        pages_for_prompt = [
            {"id": p.id, "base_site_url": p.base_site_url, "subdomains": p.subdomains}
            for p in pages
        ]
        safe_analog = self._generate_safe_analog(doc, target, pages_for_prompt)
        logger.info(f"Generated safe analog for {behavior_id}: {safe_analog[:100]}...")

        variants = []

        # Helper to create env_args with common fields
        def _make_env_args(task_name: str, test_mode: str, variation_seed: int) -> RedteamEnvArgs:
            env_args = RedteamEnvArgs(
                task_name=task_name,
                behavior_id=behavior_id,
                test_mode=test_mode,
                variation_seed=variation_seed,
                injection_seed=variation_seed,  # Backward compat
                attack_channels=case_attack_channels,
                target=target,
                pages=pages,
                start_page=start_page,
                summary=case_data.get("summary"),
                objective=case_data.get("objective"),
                doc=doc,
                safe_behavior=safe_analog,  # Safe analog for benign mode
                world_sim_model=self.world_sim_model,
                attack_type=case_data.get("attack_type", self.attack_type),
                attacker_model=self.attacker_model,
                use_trajectory=self.use_trajectory,
                adversarial_mode=case_data.get("adversarial_mode", self.adversarial_mode),
                max_conversation_turns=self.max_conversation_turns,
                target_system_prompt=target_system_prompt,
                tools_config=tools_config,
                criteria=criteria,
                judging=judging,
                resume_from_dir=self.resume_from_dir,
                skip_link_validation=self.skip_link_validation,
                success_condition=success_condition,
            )

            # Initialize attacker agent with test_mode and safe_behavior
            attacker_agent = RedteamAttackerAgent(
                model=self.attacker_model,
                goal=doc,
                target_description=target,
                attack_type=env_args.attack_type,
                mode=env_args.adversarial_mode,
                test_mode=test_mode,
                pages=pages,
                safe_behavior=safe_analog,  # Safe analog for benign/adversarial modes
            )
            env_args.attacker_agent = attacker_agent

            return env_args

        # 1. Create benign control variant
        benign_task_name = f"{self.name}.{behavior_id}.benign"
        benign_args = _make_env_args(benign_task_name, "benign", 0)
        variants.append(benign_args)

        # 2. Create N adversarial variants
        for var_idx in range(case_n_adversarial):
            adv_task_name = f"{self.name}.{behavior_id}.adversarial_v{var_idx}"
            adv_args = _make_env_args(adv_task_name, "adversarial", var_idx)
            variants.append(adv_args)

        logger.debug(
            f"Created {len(variants)} variants for {behavior_id}: "
            f"1 benign + {case_n_adversarial} adversarial"
        )

        return variants

    def _generate_safe_analog(
        self, adversarial_behavior: str, target: str, pages: list[dict]
    ) -> str:
        """
        Generate safe analog from adversarial behavior.

        Transforms the adversarial behavior description into a legitimate task
        that uses the same UI interactions without harmful intent.

        Args:
            adversarial_behavior: The adversarial behavior description (doc)
            target: The adversarial target description
            pages: List of page configurations

        Returns:
            Safe analog string describing a legitimate task
        """
        prompt = get_safe_transformation_prompt(
            adversarial_behavior=adversarial_behavior,
            target=target,
            pages=pages,
        )
        llm = OpenRouterLLMClient(self.world_sim_model)
        safe_analog = llm.chat_simple([{"role": "user", "content": prompt}])
        return safe_analog.strip()

    @staticmethod
    def _strip_html(text: str) -> str:
        """Remove HTML tags from text."""
        import re

        # Remove HTML tags
        clean = re.sub(r"<[^>]+>", " ", text)
        # Normalize whitespace
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    def dependency_graph_over_tasks(self) -> dict[str, list[str]]:
        """Redteam tasks are independent."""
        return {}
