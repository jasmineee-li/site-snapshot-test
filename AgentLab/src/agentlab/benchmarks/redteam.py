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
import gymnasium as gym

import bgym
from bgym import HighLevelActionSetArgs
from pydantic import ConfigDict

from agentlab.benchmarks.abstract_env import AbstractBenchmark, AbstractEnv, AbstractEnvArgs
from agentlab.benchmarks.redteam_attacker import RedteamAttackerAgent

logger = logging.getLogger(__name__)


@dataclass
class Page:
    """Page definition from benchmark config."""

    domain: str
    mode: Literal["synthetic", "snapshot"] = "synthetic"
    real_site: str = None  # e.g., "gmail.com" for generation context
    screenshots: list[str] = field(default_factory=list)  # Paths to screenshot files/folders
    existing_path: str = None  # Path to pre-generated HTML file (skips generation if provided)
    details: str = None


@dataclass
class RedteamEnvArgs(AbstractEnvArgs):
    """Configuration for a single redteam task."""

    task_name: str
    behavior: str  # Task description
    target: str  # Attack goal description
    pages: list[Page]
    tools: list[str]  # e.g., ["navigate", "click", "type", "read_text", "done"]

    # Generation parameters
    injection_seed: int = 0
    world_sim_model: str = "gpt-4o"  # LLM for page generation and prefill analysis
    use_cache: bool = True
    use_trajectory: bool = False  # Whether to run trajectory observation on real sites

    # Attacker agent configuration (for adversarial mode)
    use_attacker_agent: bool = True  # Toggle adversarial mode
    attacker_model: str = "gpt-4o"  # LLM model for attacker agent
    attacker_system_prompt: str = None  # Auto-generated if None
    adversarial_mode: str = "single_turn"  # "single_turn" or "multi_turn"
    max_conversation_turns: int = 10  # Max turns for multi-turn mode
    use_standard_step_format: bool = (
        True  # Use standard StepInfo management instead of custom adversarial loop
    )

    # Environment parameters (BrowserGym compatibility)
    headless: bool = True
    max_steps: int = 15
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


@dataclass
class RedteamAttackerAgentArgs:
    """Agent arguments for the redteam attacker agent."""

    model: str = "gpt-4o"
    goal: str = ""  # Behavior to induce
    target_description: str = ""  # Ideal malicious output
    custom_system_prompt: str = None  # Override auto-generated prompt

    def make_agent(self):
        """Create attacker agent instance."""
        return RedteamAttackerAgent(
            model=self.model,
            goal=self.goal,
            target_description=self.target_description,
            custom_system_prompt=self.custom_system_prompt,
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
        self.cache_manager = None
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

        # Trajectory-enhanced generation
        self.safe_analog_generator = None
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

        # Initialize trajectory data collection
        trajectory_screenshots = []
        trajectory_doms = []
        safe_behavior = None

        # Step 0: Conditional trajectory observation
        if self.env_args.use_trajectory:
            logger.info(
                "Trajectory observation enabled - running safe analog generation and observation..."
            )
            from agentlab.benchmarks.safe_analog_generator import SafeAnalogGenerator
            from agentlab.benchmarks.trajectory_observer import TrajectoryObserver

            self.safe_analog_generator = SafeAnalogGenerator(model=self.env_args.world_sim_model)
            self.trajectory_observer = TrajectoryObserver()

            logger.info("Generating safe analog...")
            safe_behavior = self.safe_analog_generator.generate_safe_analog(
                adversarial_behavior=self.env_args.behavior,
                target=self.env_args.target,
                pages=[{"domain": p.domain, "real_site": p.real_site} for p in self.env_args.pages],
            )
            logger.info(f"Safe analog: {safe_behavior[:100]}...")

            # Run trajectory agent (observation-only)
            logger.info("Observing trajectory on real sites...")
            sites = [p.real_site or p.domain for p in self.env_args.pages]
            sites = [s for s in sites if s]  # Remove None values

            trajectory_data = self.trajectory_observer.observe_trajectory(
                safe_behavior=safe_behavior,
                sites=sites,
                output_dir=self.exp_dir / "trajectory",
            )
            logger.info(f"✓ Trajectory captured: {len(trajectory_data['observations'])} steps")

            # Extract screenshot and DOM paths from trajectory
            for obs in trajectory_data.get("observations", []):
                paths = obs.get("paths", {})
                screenshot_path = paths.get("screenshot")
                dom_path = paths.get("dom")

                if screenshot_path and Path(screenshot_path).exists():
                    trajectory_screenshots.append(screenshot_path)
                if dom_path and Path(dom_path).exists():
                    trajectory_doms.append(dom_path)

            logger.info(
                f"✓ Found {len(trajectory_screenshots)} screenshots, {len(trajectory_doms)} DOM snapshots"
            )
        else:
            logger.info("Trajectory observation disabled - using benchmark screenshots only")

        # Step 1: Initialize generation pipeline
        if self.cache_manager is None:
            from agentlab.benchmarks.redteam_generation import (
                SyntheticPageGenerator,
                PageCacheManager,
                PrefillDataAnalyzer,
            )

            cache_dir = Path.home() / ".agentlab" / "redteam_cache"
            self.cache_manager = PageCacheManager(cache_dir)
            self.analyzer = PrefillDataAnalyzer(model=self.env_args.world_sim_model)
            self.generator = SyntheticPageGenerator(model=self.env_args.world_sim_model)

        # Step 2: Analyze prefill data requirements
        logger.info("Analyzing prefill data needs...")
        # Use trajectory screenshots/DOMs if enabled, otherwise pass None
        analyzer_screenshots = trajectory_screenshots if self.env_args.use_trajectory else None
        analyzer_doms = trajectory_doms if self.env_args.use_trajectory else None

        prefill_spec = self.analyzer.analyze(
            behavior=self.env_args.behavior,
            pages=self.env_args.pages,
            screenshots=analyzer_screenshots,
            dom_paths=analyzer_doms,
        )

        # Save for reproducibility
        spec_path = self.exp_dir / "prefill_spec.json"
        spec_path.write_text(json.dumps(prefill_spec, indent=2), encoding="utf-8")

        # Step 3: Generate pages
        logger.info("Generating synthetic pages...")
        self._flow_config = {
            "run_dir": str(self.exp_dir),
            "stages": {},
        }

        for page in self.env_args.pages:
            # Extract stage name from domain, handling both "gmail.com" and "/gmail" formats
            stage_name = page.domain.split(".")[0].lstrip(
                "/"
            )  # "gmail.com" → "gmail", "/gmail" → "gmail"

            if page.mode == "synthetic":
                # Check if existing HTML path is provided
                if page.existing_path:
                    logger.info(f"Using existing HTML from {page.existing_path}")
                    existing_file = Path(page.existing_path)
                    if not existing_file.exists():
                        raise FileNotFoundError(
                            f"Existing HTML file not found: {page.existing_path}"
                        )
                    html = existing_file.read_text(encoding="utf-8")
                else:
                    # Check cache first
                    cache_key = self.cache_manager.compute_key(
                        domain=page.domain,
                        behavior=self.env_args.behavior,
                        prefill=prefill_spec,
                        seed=self.env_args.injection_seed,
                    )

                    html = self.cache_manager.get(cache_key) if self.env_args.use_cache else None
                    if html is None:
                        # Generate new page with prefill spec and screenshots
                        logger.info(f"Generating synthetic page for {page.domain}")
                        # Use trajectory screenshots if enabled, otherwise use benchmark screenshots
                        page_screenshots = (
                            trajectory_screenshots
                            if self.env_args.use_trajectory
                            else page.screenshots
                        )

                        html = self.generator.generate(
                            domain=page.domain,
                            context={"behavior": self.env_args.behavior},
                            prefill=prefill_spec,
                            real_site=page.real_site,
                            screenshots=page_screenshots,
                        )
                        if self.env_args.use_cache:
                            self.cache_manager.put(cache_key, html)

                # Save to exp_dir
                page_path = self.exp_dir / f"{stage_name}.html"
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
        first_stage = list(self._flow_config["stages"].keys())[0]

        self.server_port = self._find_free_port()
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
                "goal": self.env_args.behavior,
            },
        )

        # Reset browser (openended task will automatically navigate to start_url)
        obs, info = self.browser_env.reset(seed=seed)

        # Augment observation with redteam-specific info
        obs["goal_object"] = [{"type": "text", "text": self.env_args.behavior}]
        obs["last_action"] = None
        obs["last_action_error"] = ""

        self._step_count = 0

        # Return rich observation with task info
        return obs, {
            "task_name": self.env_args.task_name,
            "behavior": self.env_args.behavior,
            "target": self.env_args.target,
            "injection_seed": self.env_args.injection_seed,
            "start_url": f"http://localhost:{self.server_port}/{first_stage}",
            "allowed_tools": self.env_args.tools,
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

    def _find_free_port(self) -> int:
        """Find an available port for the server."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

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

    name: str = "redteam"
    benchmark_file: str | Path = None
    env_args_list: list[RedteamEnvArgs] = None

    # Agent configuration (required by GenericAgent.set_benchmark)
    is_multi_tab: bool = False
    high_level_action_set_args: HighLevelActionSetArgs = None

    # Generation settings
    world_sim_model: str = "gpt-4o"  # LLM for page generation and prefill analysis
    attacker_model: str = "gpt-4o"  # LLM for attacker agent
    use_cache: bool = True
    use_trajectory: bool = False  # Whether to run trajectory observation on real sites

    # Adversarial mode settings
    adversarial_mode: str = "single_turn"  # "single_turn" or "multi_turn"
    max_conversation_turns: int = 10  # For multi_turn mode
    use_standard_step_format: bool = True  # Use standard StepInfo management for better logging

    # Variation settings for parallelization
    n_injection_variations: int = 1

    def model_post_init(self, __context: Any) -> None:
        """Load benchmark JSON and create env_args_list."""
        self.env_args_list = []

        # Initialize action set for agents that need it
        if self.high_level_action_set_args is None:
            self.high_level_action_set_args = HighLevelActionSetArgs(
                subsets=["bid"],  # BrowserGym high-level actions via bIDs
                multiaction=False,
            )

        with open(self.benchmark_file, "r") as f:
            data = json.load(f)

        logger.info(f"Loading redteam benchmark: {data['name']}")

        for case_idx, case_data in enumerate(data["data"]):
            # Parse pages
            pages = []
            for p in case_data["pages"]:
                page = Page(
                    domain=p["domain"],
                    mode=p.get("mode", "synthetic"),
                    real_site=p.get("real_site"),
                    screenshots=p.get("screenshots", []),
                    existing_path=p.get("existing_path"),  # Optional pre-generated HTML
                    details=p.get("details"),  # Optional details for prefill analysis
                )
                pages.append(page)

            # Create variations with different injection seeds
            for seed in range(self.n_injection_variations):
                task_name = f"{self.name}.case{case_idx}.seed{seed}"

                env_args = RedteamEnvArgs(
                    task_name=task_name,
                    behavior=case_data["behavior"],
                    target=case_data.get("target", ""),
                    pages=pages,
                    tools=case_data.get(
                        "tools", ["navigate", "click", "type", "read_text", "done"]
                    ),
                    injection_seed=seed,
                    world_sim_model=self.world_sim_model,
                    attacker_model=self.attacker_model,
                    use_cache=self.use_cache,
                    use_trajectory=self.use_trajectory,
                    adversarial_mode=self.adversarial_mode,
                    max_conversation_turns=self.max_conversation_turns,
                    use_standard_step_format=self.use_standard_step_format,
                )

                self.env_args_list.append(env_args)

        logger.info(
            f"Loaded {len(self.env_args_list)} tasks "
            f"({len(data['data'])} cases × {self.n_injection_variations} variations)"
        )

    def dependency_graph_over_tasks(self) -> dict[str, list[str]]:
        """Redteam tasks are independent."""
        return {}
