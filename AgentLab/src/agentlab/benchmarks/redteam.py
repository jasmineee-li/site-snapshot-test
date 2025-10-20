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

import bgym
from pydantic import ConfigDict

from agentlab.benchmarks.abstract_env import AbstractBenchmark, AbstractEnv, AbstractEnvArgs

logger = logging.getLogger(__name__)


@dataclass
class Page:
    """Page definition from benchmark config."""
    domain: str
    mode: Literal["synthetic", "snapshot"] = "synthetic"
    real_site: str = None  # e.g., "gmail.com" for generation context
    screenshots: list[str] = field(default_factory=list)  # Paths to screenshot files/folders


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
    llm_model: str = "gpt-4o"
    use_cache: bool = True

    # Environment parameters (BrowserGym)
    headless: bool = True
    max_steps: int = 30
    task_seed: int = 0

    def make_env(self, exp_dir: Path, action_mapping=None) -> "RedteamEnv":
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
        self.cache_manager = None
        self.analyzer = None
        self.generator = None

        # Flow server
        self.server_proc = None
        self.server_port = None

        # BrowserGym environment (the real workhorse)
        self.browser_env = None
        self._step_count = 0

    def reset(self, seed: int = None) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Reset and generate synthetic environment.

        Steps:
        1. Initialize generation pipeline
        2. Analyze prefill data needs (LLM call)
        3. Generate synthetic pages (LLM calls)
        4. Start flow server
        5. Initialize BrowserGym environment
        6. Return rich observation with goal
        """
        logger.info(f"Resetting RedteamEnv: {self.env_args.task_name}")

        # Step 1: Initialize generation pipeline (once)
        if self.cache_manager is None:
            from agentlab.benchmarks.redteam_generation import (
                PrefillDataAnalyzer,
                SyntheticPageGenerator,
                PageCacheManager,
            )

            cache_dir = Path.home() / ".agentlab" / "redteam_cache"
            self.cache_manager = PageCacheManager(cache_dir)
            self.analyzer = PrefillDataAnalyzer(model=self.env_args.llm_model)
            self.generator = SyntheticPageGenerator(model=self.env_args.llm_model)

        # Step 2: Analyze prefill data requirements
        logger.info("Analyzing prefill data needs...")
        prefill_spec = self.analyzer.analyze(
            behavior=self.env_args.behavior,
            pages=self.env_args.pages,
        )

        # Save for reproducibility
        spec_path = self.exp_dir / "prefill_spec.json"
        spec_path.write_text(json.dumps(prefill_spec, indent=2), encoding="utf-8")

        # Step 3: Generate pages
        logger.info("Generating synthetic pages...")
        flow_config = {
            "run_dir": str(self.exp_dir),
            "stages": {},
        }

        for page in self.env_args.pages:
            stage_name = page.domain.split(".")[0]  # "gmail.com" → "gmail"

            if page.mode == "synthetic":
                # Check cache first
                cache_key = self.cache_manager.compute_key(
                    domain=page.domain,
                    behavior=self.env_args.behavior,
                    prefill=prefill_spec,
                    seed=self.env_args.injection_seed,
                )

                html = self.cache_manager.get(cache_key)
                if html is None:
                    # Generate new page
                    html = self.generator.generate(
                        domain=page.domain,
                        context={"behavior": self.env_args.behavior},
                        prefill=prefill_spec,
                        real_site=page.real_site,
                        screenshots=page.screenshots or [],
                    )
                    if self.env_args.use_cache:
                        self.cache_manager.put(cache_key, html)

                # Save to exp_dir
                page_path = self.exp_dir / f"{stage_name}.html"
                page_path.write_text(html, encoding="utf-8")

                flow_config["stages"][stage_name] = {"mode": "synthetic"}
            else:
                # Snapshot mode - use HAR replay or bundle
                flow_config["stages"][stage_name] = {
                    "mode": "snapshot",
                    "bundle_dir": f"snapshots/{page.domain}_bundle",
                }

        # Save flow config
        config_path = self.exp_dir / "flow_config.json"
        config_path.write_text(json.dumps(flow_config, indent=2))

        # Step 4: Start flow server on dynamic port
        self.server_port = self._find_free_port()
        logger.info(f"Starting flow server on port {self.server_port}...")

        # Use the serve_flow.py from cua-website-sim (ported to agentlab)
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

        # Step 5: Initialize BrowserGym
        # Use the generic browsergym environment
        self.browser_env = bgym.make_env(
            task_name="browsergym/openended",  # Generic web task
            headless=self.env_args.headless,
            max_steps=self.env_args.max_steps,
            action_mapping=self.action_mapping,
            viewport_size={"width": 1280, "height": 720},
        )

        # Reset browser and navigate to first page
        obs, info = self.browser_env.reset(seed=seed)

        first_stage = list(flow_config["stages"].keys())[0]
        start_url = f"http://localhost:{self.server_port}/{first_stage}"

        # Navigate to start URL
        self.browser_env.step(f'goto("{start_url}")')
        obs = self.browser_env.unwrapped.obs  # Get observation after navigation

        # Inject goal into observation
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
            "start_url": start_url,
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

    # Generation settings
    llm_model: str = "gpt-4o"
    use_cache: bool = True

    # Variation settings for parallelization
    n_injection_variations: int = 1

    def model_post_init(self, __context: Any) -> None:
        """Load benchmark JSON and create env_args_list."""
        self.env_args_list = []

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
                    tools=case_data.get("tools", ["navigate", "click", "type", "read_text", "done"]),
                    injection_seed=seed,
                    llm_model=self.llm_model,
                    use_cache=self.use_cache,
                )

                self.env_args_list.append(env_args)

        logger.info(
            f"Loaded {len(self.env_args_list)} tasks "
            f"({len(data['data'])} cases × {self.n_injection_variations} variations)"
        )

    def dependency_graph_over_tasks(self) -> dict[str, list[str]]:
        """Redteam tasks are independent."""
        return {}