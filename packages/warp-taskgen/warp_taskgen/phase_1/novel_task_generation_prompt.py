"""Phase 1 novel-task generation prompts, corrections, and site agent context."""

from __future__ import annotations

import json
import os
from typing import Any

from warp_taskgen.phase_1.generated_workflows import (
    compile_model_owned_content,
    generation_prompt_addendum,
    owns_model_generated_content,
    restore_compiled_tasks,
)
from warp_taskgen.phase_1.novel_task_site_plan import EligibleSiteProfile
from warp_taskgen.phase_1.novel_task_validation import GeneratedTaskValidationError
from warp_taskgen.phases.phase_1_task_cards import card_benign_reward_shape
from warp_taskgen.prompt_corrections import render_validation_feedback
from warp_taskgen.prompt_loading import load_prompt

CONTRACT_BOUND_ACTION_API_ENV = "WORLDSIM_PHASE1_CONTRACT_BOUND_API"
CONTRACT_BOUND_ACTION_API_REQUIRED_PROFILES = frozenset({"tier2_pure_action_paper"})


def _stamp_new_task_origin(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stamp generated tasks with their Phase 1 source before caching."""
    stamped: list[dict[str, Any]] = []
    for task in tasks:
        item = json.loads(json.dumps(task))
        if isinstance(item, dict):
            item["origin"] = "new_task"
        stamped.append(item)
    return stamped


def _compile_phase1_feature_tasks(
    tasks: list[dict[str, Any]],
    *,
    task_card_plan: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Apply explicitly authored feature generation before per-site caching."""
    return restore_compiled_tasks(tasks, task_card_plan=task_card_plan)


def _compile_phase1_model_owned_features(
    tasks: Any,
    *,
    task_card_plan: dict[str, Any] | None,
) -> Any:
    """Compile strict semantic slots before generic task validation.

    Model output contains only feature-owned semantic facts. Concrete owners
    reconstruct host fields before the ordinary Phase 1 validators see a row.
    """

    return compile_model_owned_content(tasks, task_card_plan=task_card_plan)


def render_generate_benign_tasks_prompt(
    *,
    site_name: str,
    num_tasks: int,
    task_card_plan: dict[str, Any] | None = None,
    _task_number_start: int | None = None,
) -> str:
    """Render the generate-new-tasks prompt without interpreting literal example braces.

    ``_task_number_start`` is private context used by card-sliced generation;
    ordinary callers continue to receive the historical prompt unchanged.
    """
    prompt_name = (
        "generate-benign-action-tasks"
        if _task_card_plan_is_host_action_only(task_card_plan)
        else "generate-benign-tasks"
    )
    prompt = load_prompt(
        prompt_name,
        validation_command=f"benign-tasks --site-name {site_name}",
    )
    rendered = prompt.replace("{site_name}", site_name).replace("{num_tasks}", str(num_tasks))
    if isinstance(task_card_plan, dict):
        addendum = generation_prompt_addendum(
            task_card_plan,
            site_name=site_name,
            _task_number_start=_task_number_start,
        )
        if addendum:
            rendered = f"{rendered}\n\n{addendum}"
    return rendered


def _task_card_plan_is_host_action_only(task_card_plan: dict[str, Any] | None) -> bool:
    """Return whether every active card is plain host-action utility work."""
    if not isinstance(task_card_plan, dict):
        return False
    active_cards = [
        card
        for card in task_card_plan.get("task_cards", [])
        if isinstance(card, dict) and str(card.get("status", "active")) == "active"
    ]
    return bool(active_cards) and all(
        card_benign_reward_shape(card) == "host_action_only"
        and not owns_model_generated_content(card)
        for card in active_cards
    )


def _use_contract_bound_action_api(task_card_plan: dict[str, Any] | None) -> bool:
    """Return whether Phase 1 should use the contract-bound API backend."""
    if not _task_card_plan_is_host_action_only(task_card_plan):
        return False
    profile = ""
    if isinstance(task_card_plan, dict):
        profile = str(task_card_plan.get("task_capability_profile") or "").strip()
    if profile in CONTRACT_BOUND_ACTION_API_REQUIRED_PROFILES:
        return True
    return os.environ.get(CONTRACT_BOUND_ACTION_API_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _render_generate_new_tasks_correction(errors: list[GeneratedTaskValidationError]) -> str:
    return render_validation_feedback(
        artifact_name="benign_tasks.json",
        errors=[error.to_dict() for error in errors],
        summary=(
            f"{len(errors)} validation error(s). Repair the tasks and return the complete "
            "JSON array again."
        ),
        instruction=(
            "Fix only the listed issues, preserve valid task intent where possible, and "
            "return the complete JSON array. Do not include markdown or commentary."
        ),
    )


def _load_site_agent_context(
    site: EligibleSiteProfile,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Load the sibling AGENT_CONTEXT file when present."""
    agent_context_path = site.profile_path.parent / f"AGENT_CONTEXT_{site.site_name}.json"
    if not agent_context_path.exists():
        return None, []
    try:
        data = json.loads(agent_context_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return None, [f"invalid agent context for site {site.site_name!r}: {exc}"]
    if not isinstance(data, dict):
        return None, [
            f"invalid agent context for site {site.site_name!r}: payload must be an object"
        ]
    return data, []


def _attach_agent_context_to_tasks(
    tasks: list[dict[str, Any]],
    agent_context: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Attach site agent context so later phases replay the same prompt contract.

    The embedded context can include benchmark-issued test credentials. Keeping
    it on the task artifact is intentional because later phases may run without
    direct access to the original Phase 0c files and still need the same login
    and response-format contract.
    """
    if agent_context is None:
        return tasks

    attached: list[dict[str, Any]] = []
    for task in tasks:
        hydrated = json.loads(json.dumps(task))
        hydrated["agent_context"] = json.loads(json.dumps(agent_context))
        attached.append(hydrated)
    return attached
