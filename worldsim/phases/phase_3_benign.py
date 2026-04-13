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
    DEFAULT_MODEL,
    RUNTIME_METADATA_KEY,
    bind_task_to_instance,
    execution_instance_dict,
    make_agent_factory,
    resolve_task_inputs,
    run_tasks_by_site,
    task_reset_endpoints,
)
from worldsim.browser_use_agent import AgentRunner, BrowserUseAgent
from worldsim.config import BenchmarkConfig, BenchmarkInstance
from worldsim.cost_tracker import tracker as cost_tracker
from worldsim.modal_sandbox import preflight_auth_check, run_claude_in_sandbox
from worldsim.prompt_loading import load_prompt
from worldsim.rewards import run_reward_function
from worldsim.seeding import apply_data_seed_async
from worldsim.state import get_state_dir, save_state
from worldsim.task_paths import safe_task_path_component
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

    # Default: filter to only tasks with adversarial counterparts
    full_baseline = getattr(args, "full_baseline", False)
    if not full_baseline:
        adv_tasks_path = state_dir / "phase_2" / "adversarial_tasks.json"
        if adv_tasks_path.exists():
            adv_tasks = json.loads(adv_tasks_path.read_text())
            if not isinstance(adv_tasks, list):
                logger.warning("Phase 3: adversarial_tasks.json is not an array, running all benign tasks")
                adv_tasks = []
            paired_ids = {str(t.get("benign_task_id", "")) for t in adv_tasks if isinstance(t, dict)}
            original_count = len(benign_tasks)
            benign_tasks = [t for t in benign_tasks if str(t.get("id", "")) in paired_ids]
            logger.info(
                "Phase 3: filtered to %d/%d tasks with adversarial counterparts (use --full-baseline for all)",
                len(benign_tasks), original_count,
            )
        else:
            logger.warning("Phase 3: adversarial_tasks.json not found, running all benign tasks")

    instances_path = getattr(args, "instances", None)
    if not instances_path or not Path(instances_path).exists():
        logger.error("--instances JSON file required for Phase 3 (running benchmark instances)")
        return 1
    config = BenchmarkConfig.model_validate_json(Path(instances_path).read_text())

    # On resume, reuse the task_dir_root from the prior run so
    # load_completed_results can find existing result.json files.
    resume = getattr(args, "resume", False)
    prior_state = None
    if resume:
        from worldsim.state import load_state
        prior_state = load_state()

    if (
        prior_state
        and prior_state.get("step") == "phase_3"
        and prior_state.get("task_dir_root")
    ):
        task_dir_root = Path(prior_state["task_dir_root"])
        logger.info("Resume: reusing task_dir_root %s", task_dir_root)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_dir_root = state_dir / "phase_3" / timestamp

    agent_model = getattr(args, "agent_model", None) or DEFAULT_MODEL
    agent_provider = getattr(args, "agent_provider", None)
    # Fail fast if Claude Code auth is missing — diagnosis sandboxes need it.
    try:
        preflight_auth_check()
    except RuntimeError as exc:
        logger.error("Phase 3 auth pre-flight failed:\n%s", exc)
        save_state("phase_3", status="failed", reason="auth_preflight_failed")
        return 1

    agent_factory = make_agent_factory(model=agent_model, provider=agent_provider)
    save_state(
        "phase_3",
        status="running",
        task_dir_root=str(task_dir_root),
        instances_path=str(instances_path),
        agent_model=agent_model,
        agent_provider=agent_provider,
    )

    logger.info(
        "Phase 3: validating %d benign tasks across %d instances",
        len(benign_tasks),
        len(config.instances),
    )

    # Build lookup from raw tasks — run_tasks_by_site calls
    # prepare_tasks_for_execution internally, so no need to call it here.
    prepared_by_id = {
        str(task.get("id", "unknown")): task
        for task in benign_tasks
    }

    # Run evaluation via worker pool, routing tasks only to matching site instances.
    results = await run_tasks_by_site(
        tasks=benign_tasks,
        instances=config.instances,
        agent_factory=agent_factory,
        task_runner=run_task,
        task_dir_root=task_dir_root,
        config_url_placeholders=config.url_placeholders,
        resume=resume,
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

    # Circuit breaker: if >30% of tasks errored (infrastructure problems,
    # not agent failures), skip the expensive diagnosis loop and surface the
    # issue to the operator. Follows AgentLab's retry-loop pattern.
    error_tasks = [r for r in results if r.get("outcome") == "error"]
    if error_tasks and len(error_tasks) > 0.3 * len(results):
        logger.warning(
            "Circuit breaker: %d/%d tasks errored (>30%%), skipping diagnosis. "
            "This likely indicates an infrastructure problem, not task bugs.",
            len(error_tasks), len(results),
        )
        failed_tasks = [r for r in failed_tasks if r.get("outcome") != "error"]

    for r in failed_tasks:
        task_id = str(r.get("task_id", "?"))
        task = prepared_by_id.get(task_id)
        if not task:
            logger.warning("Could not find task %s for diagnosis", task_id)
            continue

        # On resume, skip tasks that were already diagnosed in a prior run.
        diagnosis_file = (
            task_dir_root / safe_task_path_component(task_id) / "diagnosis.json"
        )
        if resume and diagnosis_file.exists():
            try:
                prior_diagnosis = json.loads(diagnosis_file.read_text())
                diagnosed.append({"task_id": task_id, **prior_diagnosis})
                logger.info("Resume: reusing prior diagnosis for task %s", task_id)
                continue
            except (json.JSONDecodeError, OSError):
                pass

        site = task["site"]
        profile_path = profiles_dir / f"BENCHMARK_PROFILE_{site}.json"
        trajectory_dir = Path(
            r.get("trajectory_dir", task_dir_root / safe_task_path_component(task_id))
        )

        if not trajectory_dir.exists():
            logger.warning("No trajectory dir for task %s, skipping diagnosis", task_id)
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

        # Persist diagnosis for resume. Without this, all diagnoses would
        # re-run on crash recovery (Modal sandbox invocations are expensive).
        diagnosis_file.parent.mkdir(parents=True, exist_ok=True)
        diagnosis_file.write_text(json.dumps(fix_result, indent=2))

    # Write validated tasks (passed + fixed)
    passed_ids = {r["task_id"] for r in passed_tasks}
    fixed_ids = {d["task_id"] for d in diagnosed if d.get("action") == "fixed"}
    validated_tasks_by_id = {
        task_id: prepared_by_id[task_id]
        for task_id in passed_ids
        if task_id in prepared_by_id
    }
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

    # Strip internal runtime metadata before persisting to disk.
    clean_tasks = [
        {k: v for k, v in t.items() if k != RUNTIME_METADATA_KEY}
        for t in validated_tasks
    ]

    output_dir = state_dir / "phase_3"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validated_tasks.json").write_text(json.dumps(clean_tasks, indent=2))
    (output_dir / "results.json").write_text(json.dumps(results, indent=2))
    (output_dir / "diagnoses.json").write_text(json.dumps(diagnosed, indent=2))

    save_state(
        "phase_3",
        status="complete",
        task_dir_root=str(task_dir_root),
        validated_tasks_path=str(output_dir / "validated_tasks.json"),
        instances_path=str(instances_path),
        agent_model=agent_model,
        agent_provider=agent_provider,
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

    if result.status != "success" and not _has_scoreable_agent_output(result):
        message = (
            f"agent run {result.status}: "
            + (result.errors[-1] if result.errors else "no additional error details")
        )
        save_result(task_dir, task, result, False, message)
        return {
            "task_id": task_id,
            "passed": False,
            "message": message,
            "elapsed": result.elapsed,
            "steps": result.steps,
            "is_done": result.is_done,
            "trajectory_dir": str(task_dir),
        }

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
        "trajectory_dir": str(task_dir),
    }


def _has_scoreable_agent_output(result: Any) -> bool:
    """Return True when a non-success run still produced usable evaluation artifacts."""
    return bool(
        getattr(result, "steps", 0) > 0
        or getattr(result, "final_result", None)
        or getattr(result, "network_trace", None)
    )


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

        prompt = _render_diagnosis_prompt(task)

        task_id = task.get("id", "unknown")
        outputs = await run_claude_in_sandbox(
            site_files=sandbox_files,
            prompt=prompt,
            output_paths=["/workspace/output/diagnosis.json"],
            label=f"3-diagnose-{task_id}",
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


def _render_diagnosis_prompt(task: dict[str, Any]) -> str:
    """Render the benign diagnosis prompt with real or unknown sanity metadata."""
    sanity_result = _task_sanity_result(task)
    return load_prompt(
        "diagnose-benign-failure",
        validation_command="diagnosis",
    ).replace("{sanity_result}", sanity_result or "unknown")


def _task_sanity_result(task: dict[str, Any]) -> str | None:
    """Extract a normalized sanity result from task metadata when present."""
    candidates = (
        task.get("sanity_result"),
        task.get("sanity_check_result"),
        task.get("sanity_check", {}).get("result") if isinstance(task.get("sanity_check"), dict) else None,
        task.get("sanity", {}).get("result") if isinstance(task.get("sanity"), dict) else None,
    )
    for value in candidates:
        if value is None:
            continue
        normalized = str(value).strip().lower()
        if normalized in {"pass", "fail", "unknown"}:
            return normalized
    return None


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
                candidate_task = _apply_fix(current_task, suggested_fix)
                # If the fix sandbox made no effective changes, the remaining
                # failure is an agent limitation, not a task bug.
                if candidate_task == current_task:
                    logger.info(
                        "Diagnosis made no changes to task %s, treating as agent limitation",
                        task.get("id", "?"),
                    )
                    return {"action": "keep_flagged", "root_cause": "agent_limitation", "diagnosis": diagnosis}
                current_task = candidate_task
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

                rerun_dir = task_dir_root / safe_task_path_component(
                    f"{task.get('id', 'unknown')}__revalidation_{iteration + 1}"
                )
                rerun_result = await _rerun_live_task(
                    task=current_task,
                    instance=live_instance,
                    instances=instances,
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

    if target in {"task_removal", "none"}:
        if patch is not None:
            raise ValueError(f"suggested_fix patch must be null for target {target!r}")
        return task

    if target not in {"reward_function", "data_seed"}:
        raise ValueError(f"unsupported suggested_fix target {target!r}")
    if patch is None:
        return task
    if not isinstance(patch, dict):
        raise ValueError("suggested_fix patch must be an object or null")

    subtree_key = "reward_function" if target == "reward_function" else "data_seed"
    base_subtree = task.get(subtree_key, {})
    if not isinstance(base_subtree, dict):
        raise ValueError(f"task[{subtree_key!r}] must be an object")
    task[subtree_key] = _apply_merge_patch(base_subtree, patch)

    return task


def _apply_merge_patch(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Apply JSON Merge Patch semantics to a task subtree."""
    merged = json.loads(json.dumps(base))
    for key, value in patch.items():
        if value is None:
            merged.pop(key, None)
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _apply_merge_patch(merged[key], value)
        else:
            merged[key] = json.loads(json.dumps(value))
    return merged


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
    instances: list[BenchmarkInstance],
    agent_factory: Callable[[], BrowserUseAgent],
    task_dir: Path,
) -> dict[str, Any]:
    """Rerun a patched task against a live instance before validation."""
    agent = agent_factory()
    bound_task = bind_task_to_instance(task, instance, instances)
    try:
        try:
            await agent.setup(instance.site_url)
            result = await run_task(bound_task, agent, instance, task_dir)
        except Exception as e:  # noqa: BLE001
            logger.warning("Live rerun failed for task %s: %s", task.get("id", "?"), e)
            return {
                "task_id": task.get("id", "unknown"),
                "passed": False,
                "message": f"rerun failed: {e}",
                "error": repr(e),
                "trajectory_dir": str(task_dir),
            }

        return {**result, "trajectory_dir": str(task_dir)}
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
