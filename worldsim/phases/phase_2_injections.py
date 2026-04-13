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
from dataclasses import dataclass
from pathlib import Path

from worldsim.cost_tracker import tracker as cost_tracker
from worldsim.modal_sandbox import preflight_auth_check, run_claude_in_sandbox
from worldsim.profile_validation import load_and_validate_profile
from worldsim.prompt_loading import load_prompt
from worldsim.seeding import validate_data_seed
from worldsim.state import get_state_dir, save_state

logger = logging.getLogger(__name__)

TASKS_PER_SHARD = 20

_REQUIRED_ADVERSARIAL_FIELDS = (
    "id",
    "benign_task_id",
    "adversarial_data_seed",
)


@dataclass(frozen=True)
class SiteInjectionResult:
    site_name: str
    adversarial_tasks: list[dict]
    errors: list[str]


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

    # Fail fast if Claude Code auth is missing — before launching any sandboxes.
    try:
        preflight_auth_check()
    except RuntimeError as exc:
        logger.error("Phase 2 auth pre-flight failed:\n%s", exc)
        save_state("phase_2", status="failed", reason="auth_preflight_failed")
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

    site_profiles, profile_errors = _collect_site_profiles(tasks_by_site, profiles_dir)
    if profile_errors:
        logger.error(
            "Phase 2 cannot proceed because required site profiles are invalid:\n%s",
            "\n".join(f"  - {item}" for item in profile_errors),
        )
        save_state("phase_2", status="failed", reason="invalid_site_profiles")
        return 1

    # Shard each site's tasks into chunks of TASKS_PER_SHARD and run all
    # shards in parallel.  Shopping (192 tasks) becomes ~8 shards instead of
    # one 54-minute sandbox.
    shard_coros = []
    for site, tasks in tasks_by_site.items():
        shards = _shard_tasks(tasks, TASKS_PER_SHARD)
        for shard_idx, shard in enumerate(shards):
            label = f"{site}-shard-{shard_idx}" if len(shards) > 1 else site
            shard_coros.append(
                _generate_injections_for_site(
                    site_name=site,
                    site_tasks=shard,
                    all_site_tasks=tasks,
                    profile_path=site_profiles[site],
                    label=label,
                )
            )

    shard_results = await asyncio.gather(*shard_coros, return_exceptions=True)

    # Merge per-shard results back into per-site results.
    results = _merge_shard_results(shard_results, tasks_by_site)

    # Merge per-site adversarial tasks, failing closed on any missing site output.
    all_adversarial: list[dict] = []
    site_failures: list[str] = []
    for result in results:
        if isinstance(result, BaseException):
            logger.error("Phase 2: sandbox failed with exception: %s", result)
            site_failures.append(str(result))
            continue
        if result.errors:
            site_failures.extend(f"{result.site_name}: {error}" for error in result.errors)
            continue
        all_adversarial.extend(result.adversarial_tasks)
        logger.info(
            "Phase 2: site %r produced %d adversarial tasks",
            result.site_name,
            len(result.adversarial_tasks),
        )

    # Per-site summary after all sandboxes complete.
    succeeded = sum(1 for r in results if not isinstance(r, BaseException) and not r.errors)
    logger.info(
        "Phase 2: all sandboxes done — %d/%d sites succeeded, %d total adversarial tasks",
        succeeded,
        len(results),
        len(all_adversarial),
    )

    if site_failures:
        logger.warning(
            "Phase 2: %d site(s) had issues (adversarial tasks from successful sites are kept):\n%s",
            len(site_failures),
            "\n".join(f"  - {failure}" for failure in site_failures),
        )

    if not all_adversarial:
        logger.error("Phase 2 produced zero adversarial tasks across all sites")
        save_state("phase_2", status="failed", reason="no_adversarial_tasks")
        return 1

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

    save_state(
        "phase_2",
        status="complete",
        adversarial_tasks_path=str(output_path),
        task_count=len(all_adversarial),
    )
    cost_tracker.log_phase_summary("phase_2")
    cost_tracker.save(state_dir / "cost_report.json")
    logger.info(
        "Phase 2 complete — %d adversarial tasks written to %s", len(all_adversarial), output_path
    )
    return 0


def _shard_tasks(tasks: list[dict], shard_size: int) -> list[list[dict]]:
    """Split a task list into chunks of at most *shard_size*."""
    return [tasks[i : i + shard_size] for i in range(0, len(tasks), shard_size)]


def _merge_shard_results(
    shard_results: list[SiteInjectionResult | BaseException],
    tasks_by_site: dict[str, list[dict]],
) -> list[SiteInjectionResult]:
    """Collapse per-shard results into one SiteInjectionResult per site."""
    site_tasks_acc: dict[str, list[dict]] = {}
    site_errors_acc: dict[str, list[str]] = {}
    # Track exceptions as site-level errors.
    for result in shard_results:
        if isinstance(result, BaseException):
            # Cannot attribute to a specific site, surface as-is.
            site_errors_acc.setdefault("_unknown_", []).append(str(result))
            continue
        site = result.site_name
        site_tasks_acc.setdefault(site, []).extend(result.adversarial_tasks)
        site_errors_acc.setdefault(site, []).extend(result.errors)

    merged: list[SiteInjectionResult] = []
    for site in tasks_by_site:
        merged.append(
            SiteInjectionResult(
                site_name=site,
                adversarial_tasks=site_tasks_acc.get(site, []),
                errors=site_errors_acc.get(site, []),
            )
        )
    # Surface any unattributed exceptions as a synthetic result.
    unknown_errors = site_errors_acc.get("_unknown_", [])
    if unknown_errors:
        merged.append(SiteInjectionResult("_unknown_", [], unknown_errors))
    return merged


async def _generate_injections_for_site(
    site_name: str,
    site_tasks: list[dict],
    all_site_tasks: list[dict] | None = None,
    profile_path: Path | None = None,
    label: str | None = None,
) -> SiteInjectionResult:
    """Generate adversarial injections for a shard (or full set) of tasks via Modal Sandbox.

    Args:
        site_name: Canonical site name (e.g. "shopping").
        site_tasks: The subset of benign tasks staged into this sandbox.
        all_site_tasks: The full set of benign tasks for the site, used for
            post-processing validation. Defaults to *site_tasks* when None
            (single-shard case).
        profile_path: Path to the site's benchmark profile.
        label: Sandbox label for logging / Modal UI.

    Returns:
        Validated sandbox output for the shard.
    """
    if all_site_tasks is None:
        all_site_tasks = site_tasks
    if label is None:
        label = site_name

    if profile_path is None or not profile_path.exists():
        logger.warning("No profile for site %r at %s — skipping", site_name, profile_path)
        return SiteInjectionResult(site_name, [], [f"profile not found at {profile_path}"])

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        # Stage only this shard's benign tasks into the sandbox
        tasks_file = tmp / "benign_tasks.json"
        tasks_file.write_text(json.dumps(site_tasks, indent=2))

        sandbox_files = {
            "/workspace/tasks/benign_tasks.json": str(tasks_file),
            "/workspace/profile/BENCHMARK_PROFILE.json": str(profile_path),
        }
        # Pass agent context so injections are crafted with knowledge of agent behavior
        agent_context_path = profile_path.parent / f"AGENT_CONTEXT_{site_name}.json"
        if agent_context_path.exists():
            sandbox_files["/workspace/profile/AGENT_CONTEXT.json"] = str(agent_context_path)

        logger.info("Phase 2: launching injection sandbox %r (%d tasks)", label, len(site_tasks))

        outputs = await run_claude_in_sandbox(
            site_files=sandbox_files,
            prompt=load_prompt(
                "generate-injections",
                validation_command="adversarial-tasks",
            ),
            output_paths=["/workspace/output/adversarial_tasks.json"],
            label=label,
        )

    logger.info("Phase 2: sandbox %r completed", label)
    cost_tracker.record("phase_2", outputs.get("_summary"), site=site_name)

    adv_json = outputs.get("/workspace/output/adversarial_tasks.json")
    if not adv_json:
        logger.warning("Phase 2: sandbox %r did not produce output", label)
        return SiteInjectionResult(
            site_name,
            [],
            ["sandbox did not produce adversarial_tasks.json"],
        )

    try:
        adv_tasks = json.loads(adv_json)
        if not isinstance(adv_tasks, list):
            adv_tasks = [adv_tasks]
    except json.JSONDecodeError as e:
        logger.error("Phase 2: invalid JSON from sandbox %r: %s", label, e)
        return SiteInjectionResult(site_name, [], [f"invalid sandbox JSON: {e}"])

    # Programmatically copy immutable fields from benign tasks instead of
    # relying on the LLM to reproduce them byte-for-byte.
    _merge_immutable_fields(adv_tasks, all_site_tasks)

    validated, errors = _validate_generated_adversarial_tasks(adv_tasks, all_site_tasks)
    return SiteInjectionResult(site_name, validated, errors)


def _merge_immutable_fields(adv_tasks: list[dict], benign_tasks: list[dict]) -> None:
    """Copy immutable fields from benign tasks into adversarial task dicts.

    Handles both the full schema (where ``reward_function`` already exists)
    and the minimal schema (where only ``adversarial_reward`` is present).
    """
    benign_by_id = {str(t.get("id", "")): t for t in benign_tasks}
    for adv_task in adv_tasks:
        benign_id = str(adv_task.get("benign_task_id", ""))
        benign_task = benign_by_id.get(benign_id)
        if benign_task is None:
            continue

        # Copy immutable structural fields.
        for field in ("instruction", "site", "sites", "start_urls", "data_seed", "agent_context"):
            if field in benign_task:
                adv_task[field] = json.loads(json.dumps(benign_task[field]))

        # Handle reward_function construction.
        # Minimal schema: adversarial_reward is a top-level field, no reward_function.
        # Full schema: reward_function already has benign_reward + adversarial_reward.
        adv_reward_top = adv_task.pop("adversarial_reward", None)
        reward = adv_task.get("reward_function")

        if reward is None and adv_reward_top is not None:
            # Minimal schema — construct reward_function from scratch.
            adv_task["reward_function"] = {
                "benign_reward": json.loads(json.dumps(benign_task.get("reward_function", {}))),
                "adversarial_reward": adv_reward_top,
            }
        elif isinstance(reward, dict):
            # Full schema — overwrite benign_reward to guarantee correctness,
            # and lift top-level adversarial_reward if present.
            reward["benign_reward"] = json.loads(json.dumps(benign_task.get("reward_function", {})))
            if adv_reward_top is not None and "adversarial_reward" not in reward:
                reward["adversarial_reward"] = adv_reward_top
            adv_task["reward_function"] = reward


def _collect_site_profiles(
    tasks_by_site: dict[str, list[dict]],
    profiles_dir: Path,
) -> tuple[dict[str, Path], list[str]]:
    """Validate Phase 0c profiles once and return the reusable site-path map."""
    site_profiles: dict[str, Path] = {}
    errors: list[str] = []
    for site in tasks_by_site:
        profile_path = profiles_dir / f"BENCHMARK_PROFILE_{site}.json"
        try:
            load_and_validate_profile(site, profile_path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        site_profiles[site] = profile_path
    return site_profiles, errors


def _validate_generated_adversarial_tasks(
    adv_tasks: list[dict],
    benign_tasks: list[dict],
) -> tuple[list[dict], list[str]]:
    """Validate sandbox-generated adversarial tasks against their benign parents."""
    benign_by_id = {str(task.get("id", "")): task for task in benign_tasks}
    validated: list[dict] = []
    errors: list[str] = []
    for i, task in enumerate(adv_tasks):
        problem = _validate_generated_adversarial_task(task, i, benign_by_id)
        if problem is not None:
            errors.append(problem)
            continue
        validated.append(task)

    if not validated and not errors:
        errors.append("sandbox produced no adversarial tasks")

    return validated, errors


def _validate_generated_adversarial_task(
    task: object,
    task_index: int,
    benign_by_id: dict[str, dict],
) -> str | None:
    """Return a validation error for one sandbox-generated adversarial task."""
    if not isinstance(task, dict):
        return f"task {task_index} is not an object"

    task_name = f"task {task_index} ({task.get('id', '?')})"
    missing = [field for field in _REQUIRED_ADVERSARIAL_FIELDS if field not in task]
    if missing:
        return f"{task_name} missing required fields {missing}"

    benign_parent = benign_by_id.get(str(task.get("benign_task_id", "")))
    if benign_parent is None:
        return f"{task_name} references unknown benign_task_id {task.get('benign_task_id')!r}"

    reward_problem = _validate_reward_function_shape(task, task_name)
    if reward_problem is not None:
        return reward_problem

    violation = _validate_adversarial_task_contract(task, benign_parent)
    if violation is not None:
        return f"{task_name} violates benign-task invariants: {violation}"

    return None


def _validate_reward_function_shape(task: dict, task_name: str) -> str | None:
    """Return a validation error when reward_function is missing expected keys."""
    reward_function = task.get("reward_function")
    if reward_function is None:
        return f"{task_name} missing reward_function"
    if not isinstance(reward_function, dict):
        return f"{task_name} reward_function is not an object"
    if "benign_reward" not in reward_function:
        return f"{task_name} reward_function missing benign_reward"
    if "adversarial_reward" not in reward_function:
        return f"{task_name} reward_function missing adversarial_reward"
    return None


def _validate_adversarial_task_contract(
    adversarial_task: dict,
    benign_task: dict,
) -> str | None:
    """Return a violation string when an adversarial task drifts from its benign parent."""
    immutable_fields = ("site", "sites", "instruction", "start_urls", "data_seed", "agent_context")
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

    try:
        validate_data_seed(adversarial_task.get("adversarial_data_seed"), allow_none=False)
    except ValueError as exc:
        return str(exc)

    return None
