"""Browser Use agent runner.

Canonical source: ``docs/worldsim-v5-technical-specifcation.md`` "Browser Use Integration".

We use Browser Use as an async Python library (not a subprocess) for running
browser agents against pre-running benchmark environments. Each worker holds
its own ``BrowserSession`` for the lifetime of the worker pool and creates
a new ``Agent`` per task.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Summary of one agent run, extracted from the Browser Use history."""

    elapsed: float
    steps: int
    is_done: bool
    final_result: str | None
    errors: list[str] = field(default_factory=list)


class AgentRunner(Protocol):
    """Protocol every agent implementation in the worker pool must satisfy."""

    async def setup(self, server_url: str) -> None: ...

    async def run(
        self, task: str, server_url: str, task_dir: Path
    ) -> AgentResult: ...

    async def teardown(self) -> None: ...


class BrowserUseAgent:
    """Browser Use-backed :class:`AgentRunner` (spec canonical implementation)."""

    def __init__(
        self,
        llm: Any,
        *,
        use_vision: bool = False,
        max_steps: int = 50,
        timeout: int = 300,
        headless: bool = True,
    ) -> None:
        # ``llm`` is configured by the caller (see worldsim/main.py).
        # To use OpenRouter, pass an OpenRouter-configured ChatOpenAI instance.
        self.llm = llm
        self.use_vision = use_vision
        self.max_steps = max_steps
        self.timeout = timeout
        self.headless = headless
        self._session: Any = None

    async def setup(self, server_url: str) -> None:
        from browser_use import BrowserSession

        self._session = BrowserSession(
            headless=self.headless,
            keep_alive=True,
            args=[
                "--disable-gpu",
                "--disable-extensions",
                "--no-sandbox",
            ],
        )
        await self._session.start()

    async def run(
        self, task: str, server_url: str, task_dir: Path
    ) -> AgentResult:
        from browser_use import Agent

        task_dir = Path(task_dir)
        task_dir.mkdir(parents=True, exist_ok=True)

        agent = Agent(
            task=(
                f"You are interacting with a web application at {server_url}. "
                f"Your task: {task}"
            ),
            llm=self.llm,
            browser_session=self._session,
            use_vision=self.use_vision,
            save_conversation_path=str(task_dir / "conversations"),
            max_steps=self.max_steps,
        )

        t0 = time.time()
        history = await asyncio.wait_for(agent.run(), timeout=self.timeout)
        elapsed = time.time() - t0

        history.save_to_file(task_dir / "history.json")

        screenshots_dir = task_dir / "screenshots"
        for step_idx, path_str in enumerate(history.screenshot_paths()):
            if path_str and Path(path_str).exists():
                screenshots_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path_str, screenshots_dir / f"step_{step_idx}.png")

        return AgentResult(
            elapsed=round(elapsed, 1),
            steps=len(history.history),
            is_done=history.is_done(),
            final_result=history.final_result(),
            errors=history.errors(),
        )

    async def teardown(self) -> None:
        if self._session is not None:
            try:
                await self._session.kill()
            except Exception as e:  # noqa: BLE001
                logger.warning("BrowserSession kill failed: %s", e)
            self._session = None
