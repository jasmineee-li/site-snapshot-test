"""Phase 2: Injection generation.

Canonical source: ``docs/worldsim-v5-technical-specifcation.md`` "Phase 2: Injection Generation".

One Modal Sandbox per site. Inputs: benign tasks from Phase 1 and
``BENCHMARK_PROFILE_{site}.json`` from Phase 0c. Output:
``adversarial_tasks.json`` per site, merged into a combined file.

Injection point selection criteria (all four required):

1. Task relevance — will the agent encounter this content?
2. Attacker realism — require "medium" or higher (from the profile).
3. Concealment — can the injection blend with surrounding content?
4. Harmful action — what could the agent be tricked into doing?
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import tempfile
from pathlib import Path

from worldsim.cost_tracker import tracker as cost_tracker
from worldsim.modal_sandbox import run_claude_in_sandbox
from worldsim.prompt_loading import load_prompt
from worldsim.state import get_state_dir, save_state

logger = logging.getLogger(__name__)


async def run(args: argparse.Namespace) -> int:
    """Phase 2 entrypoint — generate adversarial injections for each site."""
    state_dir = get_state_dir()
    output_dir = state_dir / "phase_2"

    # Load benign tasks from Phase 1
    tasks_path = state_dir / "phase_1" / "benign_tasks.json"
    if not tasks_path.exists():
        logger.error("Benign tasks not found at %s — run phase 1 first", tasks_path)
        return 1
    benign_tasks = json.loads(tasks_path.read_text())

    # Load profiles from Phase 0c
    profiles_dir = state_dir / "phase_0c"
    if not profiles_dir.exists():
        logger.error("Profiles directory not found at %s — run phase 0c first", profiles_dir)
        return 1

    save_state("phase_2", status="running")

    # Group tasks by primary site
    tasks_by_site: dict[str, list[dict]] = {}
    for task in benign_tasks:
        site = task["site"]
        tasks_by_site.setdefault(site, []).append(task)

    logger.info(
        "Phase 2: generating injections for %d sites (%d total tasks)",
        len(tasks_by_site),
        len(benign_tasks),
    )

    missing_profiles = [
        f"{site}: {profiles_dir / f'BENCHMARK_PROFILE_{site}.json'}"
        for site in tasks_by_site
        if not (profiles_dir / f"BENCHMARK_PROFILE_{site}.json").exists()
    ]
    if missing_profiles:
        logger.error(
            "Phase 2 cannot proceed because expected site profiles are missing:\n%s",
            "\n".join(f"  - {item}" for item in missing_profiles),
        )
        save_state("phase_2", status="failed", reason="missing_site_profiles")
        return 1

    # Run one sandbox per site in parallel
    results = await asyncio.gather(*[
        _generate_injections_for_site(
            site_name=site,
            site_tasks=tasks,
            profile_path=profiles_dir / f"BENCHMARK_PROFILE_{site}.json",
        )
        for site, tasks in tasks_by_site.items()
        if (profiles_dir / f"BENCHMARK_PROFILE_{site}.json").exists()
    ], return_exceptions=True)

    # Merge per-site adversarial tasks, filtering out exceptions
    all_adversarial: list[dict] = []
    for result in results:
        if isinstance(result, BaseException):
            logger.error("Phase 2: sandbox failed with exception: %s", result)
            continue
        site_name, adv_tasks = result
        if adv_tasks:
            all_adversarial.extend(adv_tasks)
            logger.info("Phase 2: site %r produced %d adversarial tasks", site_name, len(adv_tasks))
        else:
            logger.warning("Phase 2: site %r produced no adversarial tasks", site_name)

    # Write combined output
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "adversarial_tasks.json"
    output_path.write_text(json.dumps(all_adversarial, indent=2))

    if not all_adversarial:
        logger.error(
            "Phase 2: no adversarial tasks produced across all sites — "
            "check sandbox logs and profiles"
        )
        save_state("phase_2", status="failed", task_count=0)
        return 1

    save_state("phase_2", status="complete",
               adversarial_tasks_path=str(output_path),
               task_count=len(all_adversarial))
    cost_tracker.log_phase_summary("phase_2")
    cost_tracker.save(state_dir / "cost_report.json")
    logger.info("Phase 2 complete — %d adversarial tasks written to %s",
                len(all_adversarial), output_path)
    return 0


async def _generate_injections_for_site(
    site_name: str,
    site_tasks: list[dict],
    profile_path: Path,
) -> tuple[str, list[dict]]:
    """Generate adversarial injections for one site via Modal Sandbox.

    Returns:
        Tuple of (site_name, list of adversarial task dicts).
    """
    if not profile_path.exists():
        logger.warning("No profile for site %r at %s — skipping", site_name, profile_path)
        return site_name, []

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        # Stage benign tasks for this site
        tasks_file = tmp / "benign_tasks.json"
        tasks_file.write_text(json.dumps(site_tasks, indent=2))

        sandbox_files = {
            "/workspace/tasks/benign_tasks.json": str(tasks_file),
            "/workspace/profile/BENCHMARK_PROFILE.json": str(profile_path),
        }

        logger.info("Phase 2: launching injection sandbox for site %r (%d tasks)",
                     site_name, len(site_tasks))

        outputs = await run_claude_in_sandbox(
            site_files=sandbox_files,
            prompt=load_prompt("generate-injections"),
            output_paths=["/workspace/output/adversarial_tasks.json"],
        )

    cost_tracker.record("phase_2", outputs.get("_summary"), site=site_name)

    adv_json = outputs.get("/workspace/output/adversarial_tasks.json")
    if not adv_json:
        logger.warning("Phase 2: sandbox for site %r did not produce output", site_name)
        return site_name, []

    try:
        adv_tasks = json.loads(adv_json)
        if not isinstance(adv_tasks, list):
            adv_tasks = [adv_tasks]
    except json.JSONDecodeError as e:
        logger.error("Phase 2: invalid JSON from site %r sandbox: %s", site_name, e)
        return site_name, []

    benign_by_id = {
        str(task.get("id", "")): task
        for task in site_tasks
    }

    # Validate that each adversarial task has the fields Phase 4 expects
    _REQUIRED_FIELDS = ("id", "benign_task_id", "adversarial_data_seed", "site", "instruction")
    validated: list[dict] = []
    for i, task in enumerate(adv_tasks):
        if not isinstance(task, dict):
            logger.warning(
                "Phase 2 site %r: task %d is not a dict — skipping", site_name, i
            )
            continue
        missing = [f for f in _REQUIRED_FIELDS if f not in task]
        if missing:
            logger.warning(
                "Phase 2 site %r: task %d (%s) missing required fields %s",
                site_name, i, task.get("id", "?"), missing,
            )
            continue
        benign_parent = benign_by_id.get(str(task.get("benign_task_id", "")))
        if benign_parent is None:
            logger.warning(
                "Phase 2 site %r: task %d (%s) references unknown benign_task_id %r",
                site_name, i, task.get("id", "?"), task.get("benign_task_id"),
            )
            continue
        # Check nested reward_function structure
        rf = task.get("reward_function")
        if isinstance(rf, dict):
            if "benign_reward" not in rf:
                logger.warning(
                    "Phase 2 site %r: task %d (%s) reward_function missing 'benign_reward'",
                    site_name, i, task.get("id", "?"),
                )
                continue
            if "adversarial_reward" not in rf:
                logger.warning(
                    "Phase 2 site %r: task %d (%s) reward_function missing 'adversarial_reward'",
                    site_name, i, task.get("id", "?"),
                )
                continue
        elif rf is not None:
            logger.warning(
                "Phase 2 site %r: task %d (%s) reward_function is not a dict",
                site_name, i, task.get("id", "?"),
            )
            continue
        else:
            logger.warning(
                "Phase 2 site %r: task %d (%s) missing reward_function",
                site_name, i, task.get("id", "?"),
            )
            continue
        violation = _validate_adversarial_task_contract(task, benign_parent)
        if violation is not None:
            logger.warning(
                "Phase 2 site %r: task %d (%s) violates benign-task invariants: %s",
                site_name, i, task.get("id", "?"), violation,
            )
            continue
        validated.append(task)

    return site_name, validated


def _validate_adversarial_task_contract(
    adversarial_task: dict,
    benign_task: dict,
) -> str | None:
    """Return a violation string when an adversarial task drifts from its benign parent."""
    immutable_fields = ("site", "sites", "instruction", "start_urls", "data_seed")
    for field in immutable_fields:
        if adversarial_task.get(field) != benign_task.get(field):
            return f"{field} changed from benign task"

    reward = adversarial_task.get("reward_function", {})
    benign_reward = reward.get("benign_reward")
    if benign_reward != benign_task.get("reward_function"):
        return "benign reward contract changed from validated benign task"

    adv_reward = reward.get("adversarial_reward")
    if not isinstance(adv_reward, dict) or not adv_reward:
        return "adversarial_reward must be a non-empty object"

    return None
