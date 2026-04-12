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
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from worldsim.agent_config import (
    execution_instance_dict,
    make_agent_factory,
    resolve_task_inputs,
    run_tasks_by_site,
    task_reset_endpoints,
)
from worldsim.browser_use_agent import AgentRunner, BrowserUseAgent
from worldsim.config import BenchmarkConfig, BenchmarkInstance
from worldsim.cost_tracker import tracker as cost_tracker
from worldsim.modal_sandbox import run_claude_in_sandbox
from worldsim.prompt_loading import load_prompt
from worldsim.rewards import run_reward_function
from worldsim.seeding import apply_data_seed_async
from worldsim.state import get_state_dir, save_state
from worldsim.trajectory import load_trajectory_into_sandbox, save_result

logger = logging.getLogger(__name__)


async def run(args: argparse.Namespace) -> int:
    """Phase 3 entrypoint — benign validation of wrapped tasks."""
    state_dir = get_state_dir()
    # Load inputs
    tasks_path = state_dir / "phase_1" / "benign_tasks.json"
    if not tasks_path.exists():
        logger.error("Benign tasks not found at %s — run phase 1 first", tasks_path)
        return 1
    benign_tasks = json.loads(tasks_path.read_text())

    instances_path = getattr(args, "instances", None)
    if not instances_path or not Path(instances_path).exists():
        logger.error("--instances JSON file required for Phase 3 (running benchmark instances)")
        return 1
    config = BenchmarkConfig.model_validate_json(Path(instances_path).read_text())

    save_state("phase_3", status="running", instances_path=str(instances_path))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_dir_root = state_dir / "phase_3" / timestamp
    agent_model = getattr(args, "agent_model", None) or "gemini-3.1-pro-preview"
    agent_provider = getattr(args, "agent_provider", None)
    agent_factory = make_agent_factory(model=agent_model, provider=agent_provider)

    logger.info(
        "Phase 3: validating %d benign tasks across %d instances",
        len(benign_tasks),
        len(config.instances),
    )

    # Run evaluation via worker pool, routing tasks only to matching site instances.
    results = await run_tasks_by_site(
        tasks=benign_tasks,
        instances=config.instances,
        agent_factory=agent_factory,
        task_runner=run_task,
        task_dir_root=task_dir_root,
        config_url_placeholders=config.url_placeholders,
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
    profiles_dir = state_dir / "phase_0c"
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
            instances=config.instances,
            agent_factory=agent_factory,
        )
        diagnosed.append({"task_id": task_id, **fix_result})

    # Write validated tasks (passed + fixed)
    passed_ids = {r["task_id"] for r in passed_tasks}
    fixed_ids = {d["task_id"] for d in diagnosed if d.get("action") == "fixed"}
    validated_tasks_by_id = {t["id"]: t for t in benign_tasks if t["id"] in passed_ids}
    for diagnosis in diagnosed:
        if diagnosis.get("action") == "fixed" and diagnosis.get("fixed_task"):
            validated_tasks_by_id[diagnosis["task_id"]] = diagnosis["fixed_task"]
    validated_tasks = [
        validated_tasks_by_id[t["id"]]
        for t in benign_tasks
        if t["id"] in validated_tasks_by_id
    ]

    if not validated_tasks:
        logger.warning(
            "Phase 3: no tasks passed validation — 0/%d benign tasks validated. "
            "This is a valid scientific result but worth investigating.",
            len(benign_tasks),
        )

    output_dir = state_dir / "phase_3"
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
    cost_tracker.log_phase_summary("phase_3")
    cost_tracker.save(state_dir / "cost_report.json")
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

    instance_dict = execution_instance_dict(instance, task)

    # Reset all environments the task depends on.
    await _reset_task_environment(task)

    # Seed data (Mode A tasks have mechanism "none" — skip)
    seed = task.get("data_seed", {})
    if seed.get("mechanism") not in (None, "none"):
        await apply_data_seed_async(seed, instance_dict)

    # Run agent
    instruction, start_urls = resolve_task_inputs(task, instance_dict)
    result = await agent.run(
        instruction,
        instance.site_url,
        task_dir,
        start_urls=start_urls,
    )

    # Evaluate with reward function
    passed, message = run_reward_function(
        reward=task["reward_function"],
        instance=instance_dict,
        agent_result=result,
        network_trace=result.network_trace,
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

    cost_tracker.record(
        "phase_3", outputs.get("_summary"),
        task_id=task.get("id"), site=task.get("site"),
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
    instances: list[BenchmarkInstance],
    agent_factory: Callable[[], BrowserUseAgent],
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
                live_instance = _select_instance_for_task(current_task, instances)
                if live_instance is None:
                    return {
                        "action": "keep_flagged",
                        "diagnosis": diagnosis,
                        "fixed_task": current_task,
                        "rerun": {
                            "passed": False,
                            "message": "No matching live instance available for rerun",
                        },
                    }

                rerun_dir = task_dir_root / f"{task.get('id', 'unknown')}__revalidation_{iteration + 1}"
                rerun_result = await _rerun_live_task(
                    task=current_task,
                    instance=live_instance,
                    agent_factory=agent_factory,
                    task_dir=rerun_dir,
                )
                if rerun_result.get("passed"):
                    return {
                        "action": "fixed",
                        "diagnosis": diagnosis,
                        "fixed_task": current_task,
                        "rerun": rerun_result,
                        "rerun_task_dir": str(rerun_dir),
                    }

                current_trajectory = rerun_dir
                continue

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


def _select_instance_for_task(
    task: dict[str, Any], instances: list[BenchmarkInstance]
) -> BenchmarkInstance | None:
    """Pick a live benchmark instance that matches the task site."""
    site = str(task.get("site", "")).lower()
    for instance in instances:
        if instance.site_name.lower() == site:
            return instance
    return None


async def _rerun_live_task(
    task: dict[str, Any],
    instance: BenchmarkInstance,
    agent_factory: Callable[[], BrowserUseAgent],
    task_dir: Path,
) -> dict[str, Any]:
    """Rerun a patched task against a live instance before validation."""
    agent = agent_factory()
    try:
        try:
            await agent.setup(instance.site_url)
            result = await run_task(task, agent, instance, task_dir)
        except Exception as e:  # noqa: BLE001
            logger.warning("Live rerun failed for task %s: %s", task.get("id", "?"), e)
            return {
                "task_id": task.get("id", "unknown"),
                "passed": False,
                "message": f"rerun failed: {e}",
                "error": repr(e),
                "task_dir": str(task_dir),
            }

        return {**result, "task_dir": str(task_dir)}
    finally:
        await agent.teardown()


async def _reset_task_environment(task: dict[str, Any]) -> None:
    """Reset every benchmark instance a task may touch."""
    endpoints = task_reset_endpoints(task)
    if not endpoints:
        return
    for endpoint in endpoints:
        await asyncio.to_thread(_post_reset, endpoint)
    await asyncio.sleep(2)


def _post_reset(endpoint: str) -> None:
    """Call a benchmark reset endpoint and treat non-2xx as failure."""
    response = requests.post(endpoint, timeout=30)
    response.raise_for_status()
