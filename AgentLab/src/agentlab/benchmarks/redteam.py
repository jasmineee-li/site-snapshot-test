"""
Redteam benchmark for security testing with dynamic synthetic pages.

This module provides:
- RedteamEnv: Environment that generates pages on-the-fly and wraps BrowserGym
- RedteamBenchmark: Benchmark class that loads JSON configs
- Page and RedteamEnvArgs: Data structures for configuration
"""

import subprocess
import time
import json
import logging
import socket
import urllib.request
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Literal
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

    domain: str
    base_site_url: str = None  # Root URL, e.g., "https://mail.google.com"
    subdomains: list[str] = field(
        default_factory=list
    )  # Paths to generate, e.g., ["/mail/u/0/#inbox"]
    details: dict[str, str] = field(default_factory=dict)  # subdomain -> description
    mode: Literal["synthetic", "snapshot"] = "synthetic"
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
    start_page: str = None

    # Task description fields
    summary: str = None
    objective: str = None
    doc: str = None  # primary task description, replaces 'behavior'

    # Generation parameters
    injection_seed: int = 0
    world_sim_model: str = "openai/gpt-4.1"
    use_trajectory: bool = False  # Whether to run trajectory observation on real sites

    # Attacker agent configuration
    attack_type: str = "indirect"  # "direct" or "indirect"

    attacker_system_prompt: str = None  # Auto-generated if None
    adversarial_mode: str = "single_turn"  # "single_turn" or "multi_turn"
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

    # Shared attacker agent instance (initialized in benchmark)
    attacker_agent: RedteamAttackerAgent = field(default=None, init=False)

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
        One-time site generation (expensive - only runs on first reset).

        Performs:
        1. Generate safe analog behavior
        2. Record trajectory observations
        3. Analyze prefill data needs (LLM call)
        4. Generate synthetic pages (LLM calls)
        5. Save HTML files and config to disk

        Skipped on subsequent resets/relaunches for performance.
        """
        if self._generation_complete:
            logger.info("Site generation already complete, skipping...")
            return

        # Check if resuming from existing run
        if self.env_args.resume_from_dir:
            resume_dir = Path(self.env_args.resume_from_dir)
            variation_dir = resume_dir / f"variation_{self.env_args.injection_seed}"

            if variation_dir.exists() and list(variation_dir.glob("*.html")):
                logger.info(f"Resuming from existing HTML files in: {variation_dir}")

                # Copy HTML files to current exp_dir
                target_variation_dir = self.exp_dir / f"variation_{self.env_args.injection_seed}"
                target_variation_dir.mkdir(parents=True, exist_ok=True)

                import shutil

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
                logger.warning(
                    f"Resume dir not found or empty: {variation_dir}, generating fresh..."
                )

        logger.info("Starting one-time site generation...")

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
                / self.env_args.task_name.replace(".", "_")
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
                        "domain": p.domain,
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
                        "domain": p.domain,
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
        synthetic_pages = [p for p in self.env_args.pages if p.mode == "synthetic"]
        skip_analysis = synthetic_pages and all(
            p.existing_path and p.skip_modification for p in synthetic_pages
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
            )

        # For now, skip detailed prefill analysis and use an empty spec.
        # This still satisfies the generator interface (expects a dict with 'pages').
        # prefill_spec = {"pages": []}

        # Step 3: Generate base HTML pages (with placeholders for adversarial content)
        logger.info("Generating base HTML pages (with placeholders)...")

        # Determine starting stage (use env_args.start_page if provided, else first page/subdomain)
        start_stage = None
        if self.env_args.start_page:
            start_stage = self.env_args.start_page.lstrip("/")

        # Build URL mapping for cross-page navigation
        # Maps local routes to full simulated URLs
        url_mapping = {}
        for page in self.env_args.pages:
            for subdomain in page.subdomains:
                local_route = f"{page.domain}{subdomain}".replace("//", "/")
                full_url = f"{page.base_site_url}{subdomain}" if page.base_site_url else local_route
                url_mapping[local_route] = full_url
        logger.info(f"URL mapping: {url_mapping}")

        # Store base HTML (with placeholders) for each subdomain
        # Key format: "gmail/mail/u/0/#inbox" (domain + subdomain)
        base_html_by_subdomain = {}
        subdomains_needing_injections = []  # List of (route_key, page, subdomain, spec)

        # Generate base HTML for all pages and their subdomains
        for page in self.env_args.pages:
            domain_name = page.domain.lstrip("/")  # "gmail"

            if page.mode == "synthetic":
                # Extract spec for this page from prefill
                page_spec = None
                if prefill_spec:
                    for ps in prefill_spec.get("pages", []):
                        if ps.get("page") == page.domain:
                            page_spec = ps
                            break

                # Get shared functionality from page spec
                shared_functionality = (
                    page_spec.get("shared_functionality", {}) if page_spec else {}
                )
                subdomain_specs = page_spec.get("subdomains", {}) if page_spec else {}

                # Generate HTML for each subdomain
                for subdomain in page.subdomains:
                    route_key = f"{domain_name}{subdomain}".replace("//", "/")
                    full_url = (
                        f"{page.base_site_url}{subdomain}" if page.base_site_url else subdomain
                    )
                    subdomain_details = (
                        page.details.get(subdomain, "") if isinstance(page.details, dict) else ""
                    )

                    # Get subdomain-specific spec
                    subdomain_spec = subdomain_specs.get(subdomain, {})

                    # Combine shared + subdomain-specific for generation
                    combined_spec = {
                        "page": page.domain,
                        "subdomain": subdomain,
                        "shared_functionality": shared_functionality,
                        "functionality": subdomain_spec.get("functionality", {}),
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
                        logger.info(f"Generating synthetic page for {route_key} ({full_url})")
                        page_screenshots = (
                            trajectory_screenshots
                            if self.env_args.use_trajectory
                            else page.screenshots
                        )

                        # Build sibling subdomains for navigation context
                        sibling_subdomains = [
                            {
                                "subdomain": sd,
                                "route": f"{page.domain}{sd}".replace("//", "/"),
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
                        )

                    # Save base HTML
                    base_html_by_subdomain[route_key] = html

                    # Check if this subdomain needs adversarial injections
                    prefill_adv = subdomain_spec.get("prefill_adversarial", {})
                    if prefill_adv.get("needed", False):
                        subdomains_needing_injections.append(
                            (route_key, page, subdomain, combined_spec)
                        )
                        logger.info(f"Subdomain {route_key} needs adversarial injections")

                    # Save to toolbox for future reuse
                    metadata = {
                        "timestamp": datetime.now().isoformat(),
                        "doc": self.env_args.doc,
                        "domain": page.domain,
                        "subdomain": subdomain,
                        "base_site_url": page.base_site_url,
                        "injection_seed": self.env_args.injection_seed,
                    }
                    save_site_to_toolbox(route_key, html, metadata)

            else:
                # Snapshot mode - use HAR replay or bundle
                domain_clean = page.domain.lstrip("/")
                snapshots_base = Path(__file__).resolve().parents[4] / "snapshots"

                bundle_path = snapshots_base / f"{domain_clean}_bundle"
                if not bundle_path.exists():
                    raise FileNotFoundError(
                        f"Snapshot bundle not found: {bundle_path}\n"
                        f"Please create the bundle first using:\n"
                        f"  python snapshotter.py {page.base_site_url or f'https://{domain_clean}'} {bundle_path}"
                    )
                # For snapshot mode, mark all subdomains as None
                for subdomain in page.subdomains:
                    route_key = f"{domain_name}{subdomain}".replace("//", "/")
                    base_html_by_subdomain[route_key] = None

        # Step 4: Generate injection variations for subdomains that have adversarial content
        if subdomains_needing_injections:
            logger.info(
                f"Found {len(subdomains_needing_injections)} subdomain(s) needing adversarial injections"
            )

            # Use the attacker agent for injection generation
            logger.info(f"Using attacker agent (model: {self.env_args.attacker_model})...")
            attacker_agent = self.env_args.attacker_agent
            attacker_agent.make_agent()

            n_variations_to_generate = 1
            variation_indices = [self.env_args.injection_seed]

            logger.info(f"Generating injection variation(s): {variation_indices}")

            for var_idx in variation_indices:
                logger.info(f"=== Generating variation {var_idx} ===")

                variation_dir = self.exp_dir / f"variation_{var_idx}"
                variation_dir.mkdir(parents=True, exist_ok=True)

                # For each subdomain, either copy base HTML or generate injections
                for route_key, html in base_html_by_subdomain.items():
                    if html is None:
                        continue

                    # Check if this subdomain needs injections
                    needs_injection = any(
                        rk == route_key for rk, _, _, _ in subdomains_needing_injections
                    )

                    # Sanitize route_key for filename: gmail/mail/u/0/#inbox -> gmail_mail_u_0_inbox
                    file_name = (
                        route_key.replace("/", "_").replace("#", "_").replace("?", "_").strip("_")
                    )

                    if needs_injection:
                        # Find the spec for this subdomain
                        spec_for_subdomain = None
                        page_obj = None
                        for rk, p, sd, spec in subdomains_needing_injections:
                            if rk == route_key:
                                page_obj = p
                                spec_for_subdomain = spec
                                break

                        prefill_adv = spec_for_subdomain.get("prefill_adversarial", {})
                        logger.info(f"Generating injections for {route_key}...")

                        pre_injection_path = self.exp_dir / f"{file_name}_pre_injection.html"
                        pre_injection_path.write_text(html, encoding="utf-8")
                        logger.info(f"✓ Saved {file_name}_pre_injection.html (before injection)")

                        try:
                            injections = attacker_agent.generate_injections(
                                prefill_adversarial=prefill_adv,
                                page_domain=route_key,
                                variation_index=var_idx,
                                n_total_variations=n_variations_to_generate,
                            )

                            html_with_injections = replace_injection_placeholders(html, injections)
                            page_path = variation_dir / f"{file_name}.html"
                            page_path.write_text(html_with_injections, encoding="utf-8")
                            logger.info(f"✓ Saved {file_name}.html with injections")

                        except Exception as e:
                            logger.error(f"Failed to generate injections for {route_key}: {e}")
                            page_path = variation_dir / f"{file_name}.html"
                            page_path.write_text(html, encoding="utf-8")
                            logger.warning(
                                f"⚠ Saved {file_name}.html with placeholders (injection failed)"
                            )
                    else:
                        # Benign subdomain - just copy base HTML
                        page_path = variation_dir / f"{file_name}.html"
                        page_path.write_text(html, encoding="utf-8")

                logger.info(f"✓ Variation {var_idx} complete")
        else:
            logger.info("No subdomains need adversarial injections - using base HTML as-is")
            variation_dir = self.exp_dir / "variation_0"
            variation_dir.mkdir(parents=True, exist_ok=True)
            for route_key, html in base_html_by_subdomain.items():
                if html is not None:
                    file_name = (
                        route_key.replace("/", "_").replace("#", "_").replace("?", "_").strip("_")
                    )
                    page_path = variation_dir / f"{file_name}.html"
                    page_path.write_text(html, encoding="utf-8")

        # Step 5: Validate internal links (fail-fast if broken)
        if self.env_args.skip_link_validation:
            logger.warning("Skipping internal link validation (skip_link_validation=True)")
        else:
            self._validate_internal_links(variation_dir, base_html_by_subdomain.keys())

        # Step 6: Set up flow config to point to the correct variation folder
        variation_dir = self.exp_dir / f"variation_{self.env_args.injection_seed}"

        self._flow_config = {
            "run_dir": str(variation_dir),
            "stages": {},
            "start_page": self.env_args.start_page,
            "start_stage": start_stage,
        }

        # Configure stages from the variation folder - each subdomain is a stage
        for route_key in base_html_by_subdomain.keys():
            # Find the page this route belongs to
            page_obj = None
            for p in self.env_args.pages:
                domain_name = p.domain.lstrip("/")
                if route_key.startswith(domain_name):
                    page_obj = p
                    break

            if page_obj and page_obj.mode == "synthetic":
                self._flow_config["stages"][route_key] = {"mode": "synthetic"}
            elif page_obj and page_obj.mode == "snapshot":
                domain_clean = page_obj.domain.lstrip("/")
                snapshots_base = Path(__file__).resolve().parents[4] / "snapshots"
                bundle_path = snapshots_base / f"{domain_clean}_bundle"
                self._flow_config["stages"][route_key] = {
                    "mode": "snapshot",
                    "bundle_dir": str(bundle_path),
                }

        # Save flow config
        config_path = self.exp_dir / "flow_config.json"
        config_path.write_text(json.dumps(self._flow_config, indent=2))

        self._generation_complete = True
        logger.info(
            f"Site generation complete for variation {self.env_args.injection_seed} (cached for relaunches)"
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

        self.server_proc = subprocess.Popen(
            [
                "python",
                "-m",
                "agentlab.benchmarks.redteam_server",
                str(config_path),
                str(self.server_port),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for server ready
        self._wait_for_server(f"http://localhost:{self.server_port}/mode", timeout=10)
        logger.info("Flow server ready")

        # Step 3: Initialize BrowserGym
        self.browser_env = gym.make(
            "browsergym/openended",
            disable_env_checker=True,
            max_episode_steps=self.env_args.max_steps,
            headless=self.env_args.headless,
            action_mapping=self.action_mapping,
            task_kwargs={
                "start_url": f"http://localhost:{self.server_port}/{first_stage}",
                "goal": self.env_args.doc,
            },
        )

        # Reset browser (openended task will automatically navigate to start_url)
        obs, info = self.browser_env.reset(seed=seed)

        # Set up Playwright route interception for simulated URLs

        # Get browser context
        page = self.browser_env.unwrapped.page
        context = page.context

        # Build a lookup table: domain -> {path -> html_file}
        # This allows ONE handler per domain that routes by path
        domain_routes: dict[str, dict[str, Path]] = {}
        variation_dir = Path(self._flow_config["run_dir"])

        for page_config in self.env_args.pages:
            # Only intercept routes for synthetic pages with base_site_url
            if not page_config.base_site_url or page_config.mode != "synthetic":
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
                route_key = f"{page_config.domain}{subdomain}".replace("//", "/")
                stage_name = route_key.lstrip("/")
                file_name = (
                    stage_name.replace("/", "_").replace("#", "_").replace("?", "_").strip("_")
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
            "start_url": f"http://localhost:{self.server_port}/{first_stage}",
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
        """Wait for server to be ready."""
        for i in range(timeout):
            urllib.request.urlopen(url, timeout=1)
            return

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

            open("/Users/jasminexli/grayswan/site-snapshot-test/.cursor/debug.log", "a").write(
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


class RedteamBenchmark(AbstractBenchmark):
    """
    Benchmark for redteam security testing with dynamic synthetic pages.

    Features:
    - On-the-fly LLM page generation
    - Prompt injection variations via seeding
    - Support for both synthetic and snapshot modes
    - Screenshot-guided generation
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
    use_trajectory: bool = False  # Whether to run trajectory observation on real sites

    # Adversarial mode settings
    adversarial_mode: str = "single_turn"  # "single_turn" or "multi_turn"
    max_conversation_turns: int = 10  # For multi_turn mode

    # Variation settings for parallelization
    n_injection_variations: int = 1

    # Resume from existing run (skip HTML regeneration)
    resume_from_dir: str | Path = None  # Path to existing experiment results dir

    # Skip link validation (useful for debugging when LLM generates broken links)
    skip_link_validation: bool = False

    def model_post_init(self, __context: Any) -> None:
        """Load benchmark JSON and create env_args_list."""
        self.env_args_list = []

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
            env_args = self._parse_case(case_idx, case_data)
            if env_args:
                self.env_args_list.append(env_args)

        logger.info(
            f"Loaded {len(self.env_args_list)} tasks "
            f"({len(cases)} cases × {self.n_injection_variations} variations)"
        )

    def _parse_case(self, case_idx: int, case_data: dict) -> RedteamEnvArgs:
        """Parse a case from the benchmark JSON."""
        # Extract pages from simulation_config
        sim_config = case_data.get("simulation_config", {})
        pages = []
        for p in sim_config.get("pages", []):
            page = Page(
                domain=p.get("domain", "/unknown"),
                base_site_url=p.get("base_site_url", ""),
                subdomains=p.get("subdomains", ["/"]),
                details=p.get("details", {}),
                mode=p.get("mode", "synthetic"),
                screenshots=p.get("screenshots", []),
                existing_path=p.get("existing_path"),
                skip_modification=p.get("skip_modification", False),
            )
            pages.append(page)

        # Extract start_page
        start_page = sim_config.get("start_page")
        if not start_page and pages:
            start_page = pages[0].domain

        # Extract system prompt for target model
        system_prompt_config = case_data.get("system_prompt_config", {})
        target_system_prompt = system_prompt_config.get("content")

        # Extract tools config, criteria, and judging
        tools_config = case_data.get("tools_config")
        criteria = case_data.get("criteria")
        judging = case_data.get("judging")

        # Create variations with different injection seeds
        for seed in range(self.n_injection_variations):
            behavior_id = case_data.get("id", f"case{case_idx}")
            task_name = f"{self.name}.{behavior_id}.seed{seed}"

            env_args = RedteamEnvArgs(
                task_name=task_name,
                target=case_data.get("description", ""),
                pages=pages,
                start_page=start_page,
                summary=case_data.get("summary"),
                objective=case_data.get("objective"),
                doc=self._strip_html(case_data.get("doc", "")),
                injection_seed=seed,
                world_sim_model=self.world_sim_model,
                attack_type=case_data.get("attack_type", "indirect"),
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
            )

            # initialize attacker agent
            attacker_agent = RedteamAttackerAgent(
                model=self.attacker_model,
                goal=env_args.doc,
                target_description=env_args.target,
                attack_type=env_args.attack_type,
                mode=env_args.adversarial_mode,
                pages=pages,
            )
            env_args.attacker_agent = attacker_agent

            return env_args

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
