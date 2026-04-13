"""Browser Use runner adapter.

Thin wrapper that re-exports the existing Browser Use agent code through the
pluggable runner interface expected by ``worldsim.runners``.  No logic is
duplicated here; this module simply provides the two entry points the runner
registry resolves:

- ``make_task_runner()`` -- returns the ``run_task`` callable from Phase 3
  (reset, seed, agent run, reward check).
- ``make_agent_factory(model, provider)`` -- returns a zero-arg callable that
  creates a fresh ``BrowserUseAgent``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from worldsim.agent_config import make_agent_factory as _make_agent_factory
from worldsim.browser_use_agent import AgentRunner, BrowserUseAgent
from worldsim.config import BenchmarkInstance
from worldsim.phases.phase_3_benign import run_task as _run_task


def make_task_runner() -> Callable[
    [dict[str, Any], AgentRunner, BenchmarkInstance, Path],
    Any,
]:
    """Return the task runner callable for Browser Use evaluation.

    The returned function has the signature::

        async def run_task(
            task: dict[str, Any],
            agent: AgentRunner,
            instance: BenchmarkInstance,
            task_dir: Path,
        ) -> dict[str, Any]

    It performs environment reset, data seeding, agent execution, and reward
    evaluation, matching the ``TaskRunner`` protocol from ``worldsim.runner``.
    """
    return _run_task


def make_agent_factory(
    model: str = "gemini-3-flash-preview",
    provider: str | None = None,
) -> Callable[[], BrowserUseAgent]:
    """Return a zero-arg factory that creates a fresh ``BrowserUseAgent``.

    Delegates to ``worldsim.agent_config.make_agent_factory``, which lazily
    builds the LLM on first invocation so import errors surface inside the
    worker rather than at factory-creation time.
    """
    return _make_agent_factory(model=model, provider=provider)
