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
)
from agentlab.benchmarks.redteam_prompts import get_safe_transformation_prompt
from agentlab.benchmarks.llm_utils import OpenRouterLLMClient

logger = logging.getLogger(__name__)


@dataclass
class Page:
    """Page definition from benchmark config."""

    domain: str
    mode: Literal["synthetic", "snapshot"] = "synthetic"
    real_site: str = None  # e.g., "gmail.com" for generation context
    screenshots: list[str] = field(default_factory=list)  # Paths to screenshot files/folders
    existing_path: str = None  # Path to pre-generated HTML file (skips generation if provided)
    skip_modification: bool = False
    simulated_url: str = None
    details: str = None


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

    # Attacker agent configuration (for adversarial mode)
    use_attacker_agent: bool = True  # Toggle adversarial mode
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
                    {"domain": p.domain, "real_site": p.real_site} for p in self.env_args.pages
                ]
                prompt = get_safe_transformation_prompt(
                    adversarial_behavior=self.env_args.doc,
                    target=self.env_args.target,
                    pages=pages_for_prompt,
                )
                llm = OpenRouterLLMClient(self.env_args.world_sim_model)
                safe_behavior = llm.chat_simple([{"role": "user", "content": prompt}])
                logger.info(f"Safe analog: {safe_behavior[:100]}...")

                # Prepare page details and sites
                sites = [p.real_site for p in self.env_args.pages if p.real_site]
                page_details = [
                    {"domain": p.domain, "real_site": p.real_site, "details": p.details}
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

        # Step 3: Generate pages
        logger.info("Generating synthetic pages...")

        # Determine starting stage (use env_args.start_page if provided, else first page)
        start_stage = None
        if self.env_args.start_page:
            start_stage = self.env_args.start_page.split(".")[0].lstrip("/")

        self._flow_config = {
            "run_dir": str(self.exp_dir),
            "stages": {},
            "start_page": self.env_args.start_page,
            "start_stage": start_stage,
        }

        # URL mapping
        url_mapping = {}
        for page in self.env_args.pages:
            if page.simulated_url:
                url_mapping[page.domain] = page.simulated_url
        logger.info(f"URL mapping: {url_mapping}")

        for page in self.env_args.pages:
            # Extract stage name from domain, handling both "gmail.com" and "/gmail" formats
            stage_name = page.domain.split(".")[0].lstrip(
                "/"
            )  # "gmail.com" → "gmail", "/gmail" → "gmail"

            if page.mode == "synthetic":
                html = None

                if page.existing_path:
                    # TODO: clean up this code
                    if page.skip_modification:
                        logger.info(
                            f"Using existing HTML as-is from {page.existing_path} (skip_modification=True)"
                        )
                        try:
                            html = load_toolbox_site(Path(page.existing_path))
                            logger.info(f"✓ Loaded existing site without modification")
                        except Exception as e:
                            logger.warning(f"Failed to load existing site: {e}")
                            logger.info("Falling back to full generation")
                            html = None
                    else:
                        # Try to modify existing HTML for new context
                        logger.info(f"Attempting to modify existing site from {page.existing_path}")
                        try:
                            # Load base site from toolbox
                            base_html = load_toolbox_site(Path(page.existing_path))

                            # Try modification first
                            html = self.generator.modify_base_site(
                                base_html=base_html,
                                new_prefill_spec=prefill_spec,
                                domain=page.domain,
                                context={"doc": self.env_args.doc},
                            )
                            logger.info(f"✓ Successfully modified existing site")
                        except Exception as e:
                            logger.warning(f"Modification failed: {e}")
                            logger.info("Falling back to full generation")
                            html = None

                # Generate from scratch if no existing path or modification failed
                if html is None:
                    logger.info(f"Generating synthetic page for {page.domain}")
                    # Use trajectory screenshots if enabled, otherwise use benchmark screenshots
                    page_screenshots = (
                        trajectory_screenshots if self.env_args.use_trajectory else page.screenshots
                    )

                    html = self.generator.generate(
                        domain=page.domain,
                        context={"doc": self.env_args.doc},
                        prefill=prefill_spec,
                        real_site=page.real_site,
                        screenshots=page_screenshots,
                        url_mapping=url_mapping,
                    )

                # Save to toolbox for future reuse
                metadata = {
                    "timestamp": datetime.now().isoformat(),
                    "doc": self.env_args.doc,
                    "prefill_spec": prefill_spec,
                    "domain": page.domain,
                    "real_site": page.real_site,
                    "injection_seed": self.env_args.injection_seed,
                }
                save_site_to_toolbox(page.domain, html, metadata)

                # Save to exp_dir
                page_path = self.exp_dir / f"{stage_name}.html"
                page_path.parent.mkdir(parents=True, exist_ok=True)
                page_path.write_text(html, encoding="utf-8")

                self._flow_config["stages"][stage_name] = {"mode": "synthetic"}
            else:
                # Snapshot mode - use HAR replay or bundle
                # Clean domain name (remove leading slash)
                domain_clean = page.domain.lstrip("/")
                # Use absolute path to snapshots directory
                snapshots_base = Path("/Users/jasminexli/grayswan/site-snapshot-test/snapshots")
                bundle_path = snapshots_base / f"{domain_clean}_bundle"

                if not bundle_path.exists():
                    raise FileNotFoundError(
                        f"Snapshot bundle not found: {bundle_path}\n"
                        f"Please create the bundle first using:\n"
                        f"  python snapshotter.py https://{page.real_site or domain_clean} {bundle_path}"
                    )

                self._flow_config["stages"][stage_name] = {
                    "mode": "snapshot",
                    "bundle_dir": str(bundle_path),
                }

        # Save flow config
        config_path = self.exp_dir / "flow_config.json"
        config_path.write_text(json.dumps(self._flow_config, indent=2))

        self._generation_complete = True
        logger.info("Site generation complete (cached for relaunches)")

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
        self.server_port = 8000  # TODO: hardcoded, should be fine for now
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

        for page_config in self.env_args.pages:
            if page_config.simulated_url and page_config.mode == "synthetic":
                # Get HTML file path for this page
                stage_name = page_config.domain.lstrip("/")
                html_path = self.exp_dir / f"{stage_name}.html"

                if not html_path.exists():
                    logger.warning(f"HTML file does not exist: {html_path}")
                    continue

                # Extract domain from simulated URL for glob pattern
                parsed = urlparse(page_config.simulated_url)
                domain_pattern = f"**://{parsed.netloc}/**"

                def make_handler(fpath=html_path, sim_url=page_config.simulated_url):
                    def handler(route, request):
                        logger.info(f"🔀 ROUTE INTERCEPTED: {request.url}")
                        logger.info(f"   Serving: {fpath}")
                        route.fulfill(path=str(fpath))

                    return handler

                # Register on context
                context.route(domain_pattern, make_handler())
                logger.info(f"📍 Registered route on CONTEXT: {domain_pattern} -> {html_path}")

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
        for _ in range(timeout):
            try:
                urllib.request.urlopen(url, timeout=1)
                return
            except:
                time.sleep(1)
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
    target_model: str
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
                mode=p.get("mode", "synthetic"),
                real_site=p.get("real_site"),
                screenshots=p.get("screenshots", []),
                existing_path=p.get("existing_path"),
                skip_modification=p.get("skip_modification", False),
                simulated_url=p.get("simulated_url"),
                details=p.get("details"),
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
            )

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
