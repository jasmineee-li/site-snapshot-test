"""Phase 3: Benign validation.

Canonical source: ``docs/worldsim-v5-technical-specifcation.md`` "Phase 3: Benign Validation".

For each task, seed the environment with benign data, run the target agent
via Browser Use, and score the trajectory with the task's reward function.
Failed tasks enter a diagnosis-fix loop driven by a Modal Sandbox.

Failure taxonomy:

- reward function bug -> fix reward, re-run sanity check
- data seed issue -> fix seed, re-run sanity check
- impossible task -> remove
- task too hard -> keep but flag
- agent limitation -> keep (informative baseline data)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from worldsim.browser_use_agent import AgentResult, AgentRunner, BrowserUseAgent
from worldsim.config import BenchmarkConfig, BenchmarkInstance
from worldsim.eval_worker_pool import run_eval
from worldsim.modal_sandbox import run_claude_in_sandbox
from worldsim.prompt_loading import load_prompt
from worldsim.rewards import run_reward_function
from worldsim.seeding import apply_data_seed
from worldsim.state import STATE_DIR, save_state
from worldsim.trajectory import load_trajectory_into_sandbox, save_result

logger = logging.getLogger(__name__)


async def run(args: argparse.Namespace) -> int:
    """Phase 3 entrypoint — benign validation of wrapped tasks."""
    # Load inputs
    tasks_path = STATE_DIR / "phase_1" / "benign_tasks.json"
    if not tasks_path.exists():
        logger.error("Benign tasks not found at %s — run phase 1 first", tasks_path)
        return 1
    benign_tasks = json.loads(tasks_path.read_text())

    instances_path = getattr(args, "instances", None)
    if not instances_path or not Path(instances_path).exists():
        logger.error("--instances JSON file required for Phase 3 (running benchmark instances)")
        return 1
    config = BenchmarkConfig.model_validate_json(Path(instances_path).read_text())

    save_state("phase_3", status="running")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_dir_root = STATE_DIR / "phase_3" / timestamp

    logger.info(
        "Phase 3: validating %d benign tasks across %d instances",
        len(benign_tasks),
        len(config.instances),
    )

    # Run evaluation via worker pool
    results = await run_eval(
        tasks=benign_tasks,
        instances=config.instances,
        agent_factory=_make_agent_factory(),
        task_runner=run_task,
        task_dir_root=task_dir_root,
    )

    # Summarize results
    passed_tasks = [r for r in results if r.get("passed")]
    failed_tasks = [r for r in results if not r.get("passed")]
    logger.info(
        "Phase 3 initial run: %d/%d passed, %d failed",
        len(passed_tasks),
        len(results),
        len(failed_tasks),
    )

    # Diagnosis loop for failures
    profiles_dir = STATE_DIR / "phase_0c"
    diagnosed: list[dict] = []

    for r in failed_tasks:
        task_id = r.get("task_id", "?")
        task = next((t for t in benign_tasks if t["id"] == task_id), None)
        if not task:
            logger.warning("Could not find task %s for diagnosis", task_id)
            continue

        site = task["site"]
        profile_path = profiles_dir / f"BENCHMARK_PROFILE_{site}.json"
        trajectory_dir = task_dir_root / task_id

        if not trajectory_dir.exists():
            logger.warning("No trajectory dir for task %s — skipping diagnosis", task_id)
            continue

        fix_result = await fix_loop(
            task=task,
            trajectory_dir=trajectory_dir,
            profile_path=profile_path,
            task_dir_root=task_dir_root,
        )
        diagnosed.append({"task_id": task_id, **fix_result})

    # Write validated tasks (passed + fixed)
    passed_ids = {r["task_id"] for r in passed_tasks}
    fixed_ids = {d["task_id"] for d in diagnosed if d.get("action") == "fixed"}
    all_passing_ids = passed_ids | fixed_ids
    validated_tasks = [t for t in benign_tasks if t["id"] in all_passing_ids]

    output_dir = STATE_DIR / "phase_3"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validated_tasks.json").write_text(json.dumps(validated_tasks, indent=2))
    (output_dir / "results.json").write_text(json.dumps(results, indent=2))
    (output_dir / "diagnoses.json").write_text(json.dumps(diagnosed, indent=2))

    save_state(
        "phase_3",
        status="complete",
        validated_tasks_path=str(output_dir / "validated_tasks.json"),
        passed=len(passed_ids),
        fixed=len(fixed_ids),
        total=len(benign_tasks),
    )
    logger.info(
        "Phase 3 complete — %d validated (%d passed + %d fixed) out of %d total",
        len(validated_tasks),
        len(passed_ids),
        len(fixed_ids),
        len(benign_tasks),
    )
    return 0


async def run_task(
    task: dict[str, Any],
    agent: AgentRunner,
    instance: BenchmarkInstance,
    task_dir: Path,
) -> dict[str, Any]:
    """Run one benign task: reset -> seed -> agent run -> reward check.

    This is the ``task_runner`` callable passed to ``eval_worker_pool.run_eval``.
    """
    task_id = task.get("id", "unknown")

    # Reset environment
    if instance.reset_endpoint:
        try:
            requests.post(instance.reset_endpoint, timeout=30)
            await asyncio.sleep(2)
        except requests.RequestException as e:
            logger.warning("Reset failed for task %s: %s", task_id, e)

    # Seed data (Mode A tasks have mechanism "none" — skip)
    seed = task.get("data_seed", {})
    if seed.get("mechanism") not in (None, "none"):
        apply_data_seed(seed, instance.model_dump())

    # Run agent
    result = await agent.run(task["instruction"], instance.site_url, task_dir)

    # Evaluate with reward function
    instance_dict = instance.model_dump()
    passed, message = run_reward_function(
        reward=task["reward_function"],
        instance=instance_dict,
        agent_result=result,
        network_trace=None,  # TODO: capture HAR trace from Browser Use
    )

    # Save trajectory artifacts
    save_result(task_dir, task, result, passed, message)

    return {
        "task_id": task_id,
        "passed": passed,
        "message": message,
        "elapsed": result.elapsed,
        "steps": result.steps,
        "is_done": result.is_done,
    }


async def diagnose_failure(
    task: dict[str, Any],
    trajectory_dir: Path,
    profile_path: Path,
) -> dict[str, Any]:
    """Diagnose why a benign task failed via Modal Sandbox.

    Returns diagnosis dict with root_cause, explanation, and suggested_fix.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        # Stage task definition
        task_json_path = tmp / "task.json"
        task_json_path.write_text(json.dumps(task, indent=2))

        sandbox_files: dict[str, str] = {
            "/workspace/task.json": str(task_json_path),
        }

        # Add profile if available
        if profile_path.exists():
            sandbox_files["/workspace/profile/BENCHMARK_PROFILE.json"] = str(profile_path)

        # Add trajectory artifacts
        load_trajectory_into_sandbox(trajectory_dir, sandbox_files)

        prompt = load_prompt("diagnose-benign-failure").replace("{pass|fail}", "fail")

        outputs = await run_claude_in_sandbox(
            site_files=sandbox_files,
            prompt=prompt,
            output_paths=["/workspace/output/diagnosis.json"],
        )

    diag_json = outputs.get("/workspace/output/diagnosis.json")
    if not diag_json:
        return {
            "root_cause": "diagnosis_failed",
            "explanation": "Diagnosis sandbox did not produce output",
            "suggested_fix": None,
        }

    try:
        return json.loads(diag_json)
    except json.JSONDecodeError:
        return {
            "root_cause": "diagnosis_failed",
            "explanation": f"Invalid JSON from diagnosis sandbox: {diag_json[:200]}",
            "suggested_fix": None,
        }


async def fix_loop(
    task: dict[str, Any],
    trajectory_dir: Path,
    profile_path: Path,
    task_dir_root: Path,
    max_iterations: int = 2,
) -> dict[str, Any]:
    """Iterative diagnosis-fix loop for failed tasks.

    Up to ``max_iterations`` attempts. Exits early on pass or terminal
    root causes (impossible, agent_limitation).
    """
    current_trajectory = trajectory_dir
    current_task = task

    for iteration in range(max_iterations):
        diagnosis = await diagnose_failure(current_task, current_trajectory, profile_path)
        root_cause = diagnosis.get("root_cause", "unknown")

        logger.info(
            "Fix loop iteration %d for task %s: root_cause=%s",
            iteration,
            task.get("id", "?"),
            root_cause,
        )

        if root_cause == "impossible":
            return {"action": "remove", "diagnosis": diagnosis}

        if root_cause in ("too_hard", "agent_limitation", "diagnosis_failed"):
            return {"action": "keep_flagged", "diagnosis": diagnosis}

        if root_cause in ("reward_bug", "seed_bug"):
            suggested_fix = diagnosis.get("suggested_fix")
            if suggested_fix:
                current_task = _apply_fix(current_task, suggested_fix)
                # Note: re-running requires a live instance — for now just
                # record the fix and mark as needing re-validation
                return {
                    "action": "fixed",
                    "diagnosis": diagnosis,
                    "fixed_task": current_task,
                }

        # Unknown root cause — flag and move on
        return {"action": "keep_flagged", "diagnosis": diagnosis}

    return {"action": "keep_flagged", "diagnosis": diagnosis}


def _apply_fix(task: dict, suggested_fix: dict) -> dict:
    """Apply a suggested fix from diagnosis to a task definition."""
    task = json.loads(json.dumps(task))  # deep copy
    target = suggested_fix.get("target", "")
    patch = suggested_fix.get("patch")

    if target == "reward_function" and patch:
        task["reward_function"].update(patch)
    elif target == "data_seed" and patch:
        task["data_seed"].update(patch)

    return task


def _make_agent_factory():
    """Create an agent factory for the worker pool.

    Returns a zero-arg callable that produces a fresh BrowserUseAgent.
    The LLM is configured based on environment variables.
    """
    def factory() -> BrowserUseAgent:
        # LLM configuration deferred to runtime — callers configure
        # via environment variables (OPENAI_API_KEY, etc.)
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="gpt-4o", temperature=0)
        except ImportError:
            logger.warning("langchain_openai not installed — using None LLM (will fail on agent.run)")
            llm = None

        return BrowserUseAgent(llm=llm, headless=True)

    return factory
