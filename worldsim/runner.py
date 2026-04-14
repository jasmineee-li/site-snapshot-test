"""Pluggable runner protocol for Phase 3-4 agent execution.

WorldSim supports two runners:

- **browser_use** (default): the existing BrowserUseAgent + reward function
  pipeline for WebArena Verified operational use.
- **agentlab**: AgentLab GenericAgent + BrowserGym for standardized
  multi-benchmark research comparisons.

Both runners produce the same ``TaskRunResult`` dict shape so Phase 4
analysis, ecological validity gating, and adaptive strategy variation
work identically regardless of which runner executed the agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class RunnerResult:
    """Unified result from any runner backend.

    Fields mirror what Phase 3-4 task dicts already carry. Both the
    Browser Use runner and the AgentLab runner populate the same fields.
    """

    task_id: str
    passed: bool
    message: str
    elapsed: float = 0.0
    steps: int = 0
    is_done: bool = False
    trajectory_dir: str = ""
    reward: float | None = None
    network_trace: list[dict[str, Any]] = field(default_factory=list)
    raw_result: Any = None  # runner-specific, not serialized

    def to_dict(self) -> dict[str, Any]:
        """Convert to the dict shape Phase 3-4 code expects."""
        return {
            "task_id": self.task_id,
            "passed": self.passed,
            "message": self.message,
            "elapsed": self.elapsed,
            "steps": self.steps,
            "is_done": self.is_done,
            "trajectory_dir": self.trajectory_dir,
        }


class TaskRunner(Protocol):
    """Protocol for running a single task through any agent backend.

    Matches the ``task_runner`` callable signature expected by
    ``eval_worker_pool.run_eval`` and ``agent_config.run_tasks_by_site``.
    """

    async def __call__(
        self,
        task: dict[str, Any],
        agent: Any,
        instance: Any,
        task_dir: Path,
    ) -> dict[str, Any]: ...


class RunnerFactory(Protocol):
    """Factory that creates agent instances for the worker pool.

    For Browser Use this returns a ``BrowserUseAgent``. For AgentLab
    this returns a lightweight wrapper that the AgentLab task runner
    knows how to drive.
    """

    def __call__(self) -> Any: ...
