"""RedteamEnv -- app-mode environment class for the redteam benchmark."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import gymnasium as gym

from agentlab.benchmarks.abstract_env import AbstractEnv
from agentlab.benchmarks.redteam.app_artifacts import fetch_normalized_state
from agentlab.benchmarks.redteam.eval_harness import start_app_server, stop_app_server
from agentlab.benchmarks.redteam.runtime_ops import materialize_app_runtime
from agentlab.benchmarks.redteam.utils import resolve_external_app_start_url

logger = logging.getLogger(__name__)

_APP_RUNTIME_PUBLIC_FILES = ("server.py", "index.html")
_APP_RUNTIME_PUBLIC_DIRS = ("css", "js")


class RedteamEnv(AbstractEnv):
    """Environment for redteam testing with on-the-fly synthetic page generation."""

    def __init__(self, env_args, exp_dir: Path, action_mapping=None):
        self.env_args = env_args
        self.exp_dir = exp_dir
        self.action_mapping = action_mapping

        # Server state
        self.server_proc: subprocess.Popen | None = None
        self.server_port: int | None = None

        # BrowserGym environment
        self.browser_env = None
        self._step_count = 0

        self._generation_complete = False

        # App-mode runtime workspace (isolated from canonical pregenerated app_dir)
        self.runtime_app_dir: Path | None = None

    # ------------------------------------------------------------------
    # Site generation
    # ------------------------------------------------------------------

    def _generate_sites(self) -> None:
        """Prepare the isolated app-mode runtime once per environment."""
        if self._generation_complete:
            return

        self.runtime_app_dir = self._prepare_app_runtime()
        self._generation_complete = True

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, seed: int = None) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset environment for a new episode.  ~60 lines, was ~230."""
        logger.info(f"Resetting RedteamEnv: {self.env_args.task_name}")

        self._teardown_current_run()

        try:
            self._generate_sites()
            self._start_server()

            # Initialise BrowserGym (starts at about:blank)
            self.browser_env = gym.make(
                "browsergym/openended",
                disable_env_checker=True,
                max_episode_steps=self.env_args.max_steps,
                headless=self.env_args.headless,
                action_mapping=self.action_mapping,
                task_kwargs={
                    "start_url": "about:blank",
                    "goal": self.env_args.goal_instruction,
                },
            )
            obs, info = self.browser_env.reset(seed=seed)

            # Set up Playwright route interception
            page = self.browser_env.unwrapped.page
            context = page.context

            from agentlab.benchmarks.redteam.routing import setup_route_interception

            setup_route_interception(
                context=context,
                domain_bindings=getattr(self.env_args, "domain_bindings", []) or [],
                server_port=self.server_port,
            )
            external_start_url = self._resolve_app_start_url()

            logger.info(f"Navigating to start URL: {external_start_url}")
            page.goto(external_start_url, wait_until="domcontentloaded")

            # Build observation
            obs = self.browser_env.unwrapped._get_obs()
            obs["goal_object"] = [{"type": "text", "text": self.env_args.goal_instruction}]
            obs["last_action"] = None
            obs["last_action_error"] = ""
            self._step_count = 0

            return obs, {
                "task_name": self.env_args.task_name,
                "doc": self.env_args.goal_instruction,
                "target": self.env_args.target,
                "injection_seed": self.env_args.injection_seed,
                "start_url": external_start_url,
            }
        except Exception:
            self._teardown_current_run()
            raise

    def step(self, action: str):
        """Execute action via BrowserGym."""
        obs, reward, terminated, truncated, info = self.browser_env.step(action)
        self._step_count += 1

        info["injection_seed"] = self.env_args.injection_seed
        info["step_count"] = self._step_count

        return obs, reward, terminated, truncated, info

    def get_browser_state(self) -> dict | None:
        """Query current browser state from the state server."""
        if not self.server_port:
            return None
        return fetch_normalized_state(self.server_port, timeout=5)

    def close(self):
        """Clean up resources."""
        self._teardown_current_run()

    # ------------------------------------------------------------------
    # Server helpers
    # ------------------------------------------------------------------

    def _start_server(self) -> None:
        """Start the per-app server subprocess."""
        app_dir = self.runtime_app_dir or self._prepare_app_runtime()
        self.runtime_app_dir = app_dir
        server_script = app_dir / "server.py"

        if not server_script.exists():
            raise FileNotFoundError(f"App server.py not found: {server_script}")

        server_env = os.environ.copy()
        server_env["SERVER_EVENTS_LOG"] = str(
            (Path(self.exp_dir) / "server_events.log").resolve()
        )

        self.server_proc, self.server_port = start_app_server(
            app_dir,
            port=None,
            timeout=15.0,
            env=server_env,
        )
        logger.info(
            "App server started on port %s (pid=%s)",
            self.server_port,
            self.server_proc.pid,
        )

    def _stop_server_process(self) -> None:
        """Terminate the current server process and wait for it to exit."""
        if not self.server_proc:
            self.server_port = None
            return

        proc = self.server_proc
        self.server_proc = None
        self.server_port = None

        if proc.poll() is not None:
            try:
                proc.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                pass
            return

        stop_app_server(proc)

    def _close_browser_env(self) -> None:
        """Close any active BrowserGym environment and clear the handle."""
        if not self.browser_env:
            return
        browser_env = self.browser_env
        self.browser_env = None
        browser_env.close()

    def _teardown_current_run(self) -> None:
        """Dispose of any prior browser/server state before starting a fresh run."""
        self._close_browser_env()
        self._stop_server_process()

    def _prepare_app_runtime(self) -> Path:
        """Create a per-run app workspace inside ``exp_dir`` for app-mode tasks."""
        app_dir = Path(self.env_args.app_dir)
        runtime_dir = materialize_app_runtime(
            app_dir,
            variant_subdir=self.env_args.variant_subdir,
            runtime_dir=self.exp_dir / "app_runtime",
        )
        logger.info(
            "App mode: prepared isolated runtime app at %s using %s/data.js",
            runtime_dir,
            self.env_args.variant_subdir,
        )
        return runtime_dir

    def _resolve_app_start_url(self) -> str:
        """Resolve the configured app-mode start page onto the simulated external URL."""
        try:
            return resolve_external_app_start_url(
                getattr(self.env_args, "entry_route", "") or getattr(self.env_args, "start_page", "") or "",
                getattr(self.env_args, "pages", []) or [],
            )
        except ValueError as exc:
            raise RuntimeError(f"Invalid app-mode start_page contract: {exc}") from exc
