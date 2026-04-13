"""AgentLab runner adapter.

Drives AgentLab's GenericAgent + BrowserGym for standardized multi-benchmark
research comparisons. Implements the same two-function interface as the
Browser Use runner so Phase 3-4 code works identically regardless of backend.

AgentLab and BrowserGym are OPTIONAL dependencies. All imports are guarded
behind try/except so that ``worldsim.runners.agentlab`` can be imported
(e.g., by the registry) without those packages installed. A clear error
surfaces at runtime when a caller actually tries to use the runner.

Key differences from the Browser Use runner:

- AgentLab's ``ExpArgs.run()`` is synchronous, so we wrap it in
  ``asyncio.to_thread()`` to satisfy the async runner protocol.
- BrowserGym task names require a benchmark prefix (e.g., ``webarena.42``).
  The adapter reads ``task["agentlab_task_name"]`` if present, otherwise
  constructs it from the task id.
- Evaluation is handled by BrowserGym's built-in reward signal rather than
  WorldSim's reward functions, so the runner reads ``summary_info.json``
  written by ``ExpArgs`` to extract pass/fail and step count.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from worldsim.agent_config import (
    execution_instance_dict,
    task_reset_endpoints,
)
from worldsim.seeding import apply_data_seed_async

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency gate
# ---------------------------------------------------------------------------

_AGENTLAB_AVAILABLE = False
_IMPORT_ERROR: str | None = None

try:
    from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
    from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags
    from agentlab.experiments.loop import EnvArgs, ExpArgs
    from agentlab.llm.chat_api import LiteLLMModelArgs

    _AGENTLAB_AVAILABLE = True
except ImportError as _exc:
    _IMPORT_ERROR = str(_exc)


def _require_agentlab() -> None:
    """Raise a clear error if agentlab is not installed."""
    if not _AGENTLAB_AVAILABLE:
        raise ImportError(
            "The AgentLab runner requires 'agentlab' and 'browsergym' packages. "
            "Install them with:  pip install agentlab browsergym\n"
            f"Original import error: {_IMPORT_ERROR}"
        )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "gemini/gemini-3-flash"
_DEFAULT_MAX_STEPS = 30
_DEFAULT_BENCHMARK_PREFIX = "webarena"

# Reset endpoint timeout and retry settings, matching phase_3_benign values.
_RESET_TIMEOUT = 300
_RESET_MAX_RETRIES = 2
_RESET_RETRY_DELAY = 10


# ---------------------------------------------------------------------------
# Lightweight agent wrapper
# ---------------------------------------------------------------------------


@dataclass
class AgentLabAgentWrapper:
    """Thin wrapper that the agentlab task runner knows how to drive.

    This is NOT an agent itself; it holds the ``GenericAgentArgs`` that
    ``ExpArgs`` will use to instantiate the real agent inside its run loop.
    The wrapper satisfies the ``RunnerFactory`` protocol (the factory returns
    this object, and the task runner unpacks ``agent_args`` from it).
    """

    agent_args: Any  # GenericAgentArgs when agentlab is installed
    model: str = _DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Environment reset helper
# ---------------------------------------------------------------------------


def _post_reset(endpoint: str) -> None:
    """POST to a benchmark reset endpoint with retries.

    Mirrors the retry logic in ``phase_3_benign._post_reset``.
    """
    last_exc: Exception | None = None
    for attempt in range(1, _RESET_MAX_RETRIES + 1):
        try:
            resp = requests.post(endpoint, timeout=_RESET_TIMEOUT)
            resp.raise_for_status()
            return
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            last_exc = exc
            if attempt < _RESET_MAX_RETRIES:
                logger.warning(
                    "Reset %s attempt %d failed (%s), retrying in %ds",
                    endpoint,
                    attempt,
                    exc,
                    _RESET_RETRY_DELAY,
                )
                time.sleep(_RESET_RETRY_DELAY)
    raise RuntimeError(
        f"Reset endpoint {endpoint} failed after {_RESET_MAX_RETRIES} attempts"
    ) from last_exc


async def _reset_task_environment(task: dict[str, Any]) -> None:
    """Reset every benchmark instance a task may touch."""
    endpoints = task_reset_endpoints(task)
    if not endpoints:
        return
    await asyncio.gather(*[asyncio.to_thread(_post_reset, ep) for ep in endpoints])
    # Brief pause for the environment to stabilize after reset.
    await asyncio.sleep(2)


# ---------------------------------------------------------------------------
# BrowserGym task name resolution
# ---------------------------------------------------------------------------


def _resolve_browsergym_task_name(
    task: dict[str, Any],
    benchmark_prefix: str = _DEFAULT_BENCHMARK_PREFIX,
) -> str:
    """Build the BrowserGym-style task name from a WorldSim task dict.

    Checks ``task["agentlab_task_name"]`` first. Falls back to constructing
    ``"<prefix>.<id>"`` from the task id.
    """
    explicit = task.get("agentlab_task_name")
    if explicit:
        return str(explicit)

    task_id = task.get("id")
    if task_id is None:
        raise ValueError(
            "Task has no 'id' field and no 'agentlab_task_name'; "
            "cannot determine BrowserGym task name"
        )
    return f"{benchmark_prefix}.{task_id}"


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------


def _parse_exp_result(
    task_id: str,
    exp_dir: Path,
    elapsed: float,
) -> dict[str, Any]:
    """Read AgentLab's ``summary_info.json`` and convert to the dict shape
    that Phase 3-4 code expects.

    The summary written by ``ExpArgs.save_summary_info`` contains keys like
    ``n_steps``, ``cum_reward``, ``err_msg``, ``terminated``, ``truncated``.
    """
    summary_path = exp_dir / "summary_info.json"
    if not summary_path.exists():
        return {
            "task_id": task_id,
            "passed": False,
            "message": "AgentLab did not write summary_info.json",
            "elapsed": elapsed,
            "steps": 0,
            "is_done": False,
            "trajectory_dir": str(exp_dir),
        }

    summary = json.loads(summary_path.read_text())

    n_steps = summary.get("n_steps", 0)
    cum_reward = summary.get("cum_reward", 0.0)
    err_msg = summary.get("err_msg")
    terminated = summary.get("terminated", False)
    truncated = summary.get("truncated", False)

    # AgentLab considers a task passed when cumulative reward > 0 and the
    # episode finished without an uncaught error.
    passed = cum_reward > 0 and err_msg is None
    is_done = bool(terminated or truncated)

    if err_msg:
        message = f"AgentLab error: {err_msg}"
    elif passed:
        message = f"passed (reward={cum_reward:.2f}, steps={n_steps})"
    else:
        message = f"failed (reward={cum_reward:.2f}, steps={n_steps})"

    return {
        "task_id": task_id,
        "passed": passed,
        "message": message,
        "elapsed": elapsed,
        "steps": n_steps,
        "is_done": is_done,
        "trajectory_dir": str(exp_dir),
    }


# ---------------------------------------------------------------------------
# Core: synchronous experiment execution (called via asyncio.to_thread)
# ---------------------------------------------------------------------------


def _run_experiment_sync(
    agent_args: Any,
    env_args: Any,
    exp_dir: Path,
) -> None:
    """Prepare and run an AgentLab experiment synchronously.

    This is the function passed to ``asyncio.to_thread`` so the event loop
    is not blocked.
    """
    exp_args = ExpArgs(
        agent_args=agent_args,
        env_args=env_args,
    )
    exp_args.prepare(exp_root=exp_dir.parent)
    # ExpArgs.prepare creates its own subdirectory. Override to use the
    # caller-provided task_dir so trajectories land where WorldSim expects.
    exp_args.exp_dir = exp_dir
    exp_dir.mkdir(parents=True, exist_ok=True)

    exp_args.run()


# ---------------------------------------------------------------------------
# Public API: make_task_runner
# ---------------------------------------------------------------------------


def make_task_runner(
    attack_mode: str = "worldsim",
    benchmark_prefix: str = _DEFAULT_BENCHMARK_PREFIX,
    max_steps: int = _DEFAULT_MAX_STEPS,
) -> Callable[..., Any]:
    """Return an async task runner matching the Phase 3 ``run_task`` signature.

    Args:
        attack_mode: When ``"worldsim"``, adversarial data is seeded into the
            environment before the agent runs. Any other value skips seeding.
        benchmark_prefix: Prefix for BrowserGym task names, e.g., ``"webarena"``.
        max_steps: Maximum number of agent steps per episode.

    Returns:
        An async callable with signature::

            async def run_task(
                task: dict[str, Any],
                agent: AgentLabAgentWrapper,
                instance: BenchmarkInstance,
                task_dir: Path,
            ) -> dict[str, Any]
    """
    _require_agentlab()

    async def run_task(
        task: dict[str, Any],
        agent: Any,
        instance: Any,
        task_dir: Path,
    ) -> dict[str, Any]:
        task_id = task.get("id", "unknown")
        t0 = time.monotonic()

        instance_dict = execution_instance_dict(instance, task)

        # 1. Reset the environment
        await _reset_task_environment(task)

        # 2. Seed adversarial data when in worldsim attack mode
        if attack_mode == "worldsim":
            seed = task.get("data_seed", {})
            if seed.get("mechanism") not in (None, "none"):
                await apply_data_seed_async(seed, instance_dict)

        # 3. Build BrowserGym env args
        browsergym_task = _resolve_browsergym_task_name(task, benchmark_prefix)
        env_args = EnvArgs(
            task_name=browsergym_task,
            max_steps=max_steps,
            headless=True,
        )

        # 4. Extract agent_args from the wrapper
        if isinstance(agent, AgentLabAgentWrapper):
            agent_args = agent.agent_args
        else:
            # Caller passed raw GenericAgentArgs directly
            agent_args = agent

        # 5. Run the experiment (synchronous) in a thread
        try:
            await asyncio.to_thread(
                _run_experiment_sync,
                agent_args,
                env_args,
                task_dir,
            )
        except Exception:
            logger.exception("AgentLab experiment failed for task %s", task_id)

        # 6. Parse results
        elapsed = time.monotonic() - t0
        return _parse_exp_result(task_id, task_dir, elapsed)

    return run_task


# ---------------------------------------------------------------------------
# Public API: make_agent_factory
# ---------------------------------------------------------------------------


def make_agent_factory(
    model: str = _DEFAULT_MODEL,
    provider: str | None = None,
) -> Callable[[], AgentLabAgentWrapper]:
    """Return a zero-arg factory that creates an ``AgentLabAgentWrapper``.

    The wrapper holds a ``GenericAgentArgs`` configured with a
    ``LiteLLMModelArgs`` for the requested model. LiteLLM handles provider
    routing internally, so the *provider* parameter is accepted for API
    compatibility but not used directly.

    Args:
        model: LiteLLM-style model string, e.g., ``"gemini/gemini-3-flash"``
            or ``"openai/gpt-4.1-mini-2025-04-14"``.
        provider: Accepted for interface compatibility with the Browser Use
            runner. Not used by LiteLLM routing.
    """
    _require_agentlab()

    def factory() -> AgentLabAgentWrapper:
        chat_model_args = LiteLLMModelArgs(
            model_name=model,
            max_total_tokens=128_000,
            max_input_tokens=126_000,
            max_new_tokens=4_096,
            temperature=0,
            vision_support=True,
        )

        flags = GenericPromptFlags(
            obs=_default_obs_flags(),
            action=_default_action_flags(),
            use_plan=True,
            use_thinking=True,
            use_memory=True,
            be_cautious=True,
        )

        agent_args = GenericAgentArgs(
            chat_model_args=chat_model_args,
            flags=flags,
        )

        return AgentLabAgentWrapper(agent_args=agent_args, model=model)

    return factory


# ---------------------------------------------------------------------------
# Default flag helpers
# ---------------------------------------------------------------------------


def _default_obs_flags() -> Any:
    """Construct default observation flags for the GenericAgent.

    Imports are deferred because agentlab may not be installed.
    """
    from agentlab.agents import dynamic_prompting as dp

    return dp.ObsFlags(
        use_html=False,
        use_ax_tree=True,
        use_focused_element=True,
        use_error_logs=True,
        use_history=True,
        use_past_error_logs=False,
        use_action_history=True,
        use_think_history=True,
        use_diff=False,
        use_screenshot=True,
        use_som=False,
        extract_visible_tag=True,
        extract_clickable_tag=True,
        extract_coords="False",
        filter_visible_elements_only=False,
    )


def _default_action_flags() -> Any:
    """Construct default action flags for the GenericAgent.

    Imports are deferred because agentlab may not be installed.
    The ``action_set`` field expects a ``HighLevelActionSetArgs`` instance,
    not a plain string.
    """
    from agentlab.agents import dynamic_prompting as dp
    from bgym import HighLevelActionSetArgs

    return dp.ActionFlags(
        action_set=HighLevelActionSetArgs(
            subsets=["bid"],
            multiaction=False,
        ),
        long_description=False,
        individual_examples=True,
    )
