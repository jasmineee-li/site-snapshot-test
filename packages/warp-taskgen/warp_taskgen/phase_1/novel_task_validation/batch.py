"""Batch validation entry points for generated novel tasks."""

from __future__ import annotations

import copy
from typing import Any

from warp_taskgen.phase_1.novel_task_validation.answer_stability import (
    _validate_stable_answer_diversity,
)
from warp_taskgen.phase_1.novel_task_validation.errors import GeneratedTaskValidationError
from warp_taskgen.phase_1.novel_task_validation.route_alignment import (
    _build_start_url_policy,
    _route_contract_index,
)
from warp_taskgen.phase_1.novel_task_validation.single_task import validate_generated_novel_task
from warp_taskgen.phase_1.novel_task_validation.task_card_generation import (
    validate_task_card_generation_distribution,
)
from warp_taskgen.phases.phase_1_task_cards import task_card_index

_DEFAULT_EXPECTED_TASK_COUNT = 30


def validate_generated_novel_tasks(
    raw_tasks: Any,
    *,
    site_name: str,
    profile: dict[str, Any],
    expected_task_count: int | None = _DEFAULT_EXPECTED_TASK_COUNT,
    route_contracts: dict[str, Any] | None = None,
    task_card_plan: dict[str, Any] | None = None,
    host_compiled_evaluator_types: frozenset[str] = frozenset(),
) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate sandbox-generated generate-new-tasks output for one site."""
    validated, detailed_errors = validate_generated_novel_tasks_detailed(
        raw_tasks,
        site_name=site_name,
        profile=profile,
        expected_task_count=expected_task_count,
        route_contracts=route_contracts,
        task_card_plan=task_card_plan,
        host_compiled_evaluator_types=host_compiled_evaluator_types,
    )
    return validated, [error.legacy_render() for error in detailed_errors]


def validate_generated_novel_tasks_detailed(
    raw_tasks: Any,
    *,
    site_name: str,
    profile: dict[str, Any],
    expected_task_count: int | None = _DEFAULT_EXPECTED_TASK_COUNT,
    route_contracts: dict[str, Any] | None = None,
    task_card_plan: dict[str, Any] | None = None,
    host_compiled_evaluator_types: frozenset[str] = frozenset(),
) -> tuple[list[dict[str, Any]], list[GeneratedTaskValidationError]]:
    """Validate sandbox-generated output and return structured errors."""
    if not isinstance(raw_tasks, list):
        return [], [
            GeneratedTaskValidationError(
                code="ROOT_NOT_ARRAY",
                path="$",
                message="sandbox output must be a JSON array",
                expected="JSON array of task objects",
                actual=type(raw_tasks).__name__,
            )
        ]

    allowed_eval_types = {
        capability.get("eval_type", "")
        for capability in profile.get("verification_capabilities", [])
        if capability.get("eval_type")
    }
    route_index = _route_contract_index(route_contracts)
    card_index = task_card_index(task_card_plan)
    start_url_policy = None if route_index is not None else _build_start_url_policy(profile)
    validated: list[dict[str, Any]] = []
    errors: list[GeneratedTaskValidationError] = []
    seen_ids: set[str] = set()

    for index, raw_task in enumerate(raw_tasks):
        task = _normalize_generated_task_for_route(raw_task, route_index)
        problem = validate_generated_novel_task(
            task,
            index=index,
            site_name=site_name,
            allowed_eval_types=allowed_eval_types,
            start_url_policy=start_url_policy,
            route_index=route_index,
            task_card_index=card_index,
            host_compiled_evaluator_types=host_compiled_evaluator_types,
        )
        if problem is not None:
            errors.append(problem)
            continue

        task_id = str(task["id"])
        if task_id in seen_ids:
            errors.append(
                GeneratedTaskValidationError(
                    code="DUPLICATE_TASK_ID",
                    path=f"$[{index}].id",
                    message=f"task id {task_id!r} duplicates a prior id",
                    expected="unique task id",
                    actual=task_id,
                    repair_hint="Renumber generated tasks so every id is unique and consecutive.",
                )
            )
            continue
        seen_ids.add(task_id)
        validated.append(task)

    errors.extend(
        validate_task_card_generation_distribution(
            raw_tasks,
            site_name=site_name,
            task_card_plan=task_card_plan,
        )
    )

    if not validated and not errors:
        errors.append(
            GeneratedTaskValidationError(
                code="NO_TASKS",
                path="$",
                message="sandbox produced no novel tasks",
                expected="non-empty JSON array",
            )
        )
    elif not errors and expected_task_count is not None and len(validated) != expected_task_count:
        errors.append(
            GeneratedTaskValidationError(
                code="WRONG_TASK_COUNT",
                path="$",
                message=f"sandbox produced {len(validated)} novel tasks; expected {expected_task_count}",
                expected=expected_task_count,
                actual=len(validated),
                repair_hint="Return the complete JSON array with exactly the requested number of tasks.",
            )
        )
    elif not errors:
        diversity_problem = _validate_stable_answer_diversity(
            validated,
            route_index,
            task_card_index=card_index,
        )
        if diversity_problem is not None:
            errors.append(diversity_problem)

    return validated, errors


def _normalize_generated_task_for_route(
    task: Any,
    route_index: dict[str, dict[str, Any]] | None,
) -> Any:
    """Canonicalize route-bound editor args before persisting Phase 1 output."""
    if not isinstance(task, dict) or route_index is None:
        return task
    route_id = task.get("route_id")
    if not isinstance(route_id, str):
        return task
    route = route_index.get(route_id)
    if not isinstance(route, dict):
        return task

    editor_arg_templates = route.get("editor_arg_templates")
    if not isinstance(editor_arg_templates, dict):
        return copy.deepcopy(task)

    normalized = copy.deepcopy(task)
    # Benign and adversarial seeds describe the same route-bound fixture. Keep
    # route canonicalization symmetric so later host-compiled checks cannot
    # reject a pair that differs only because one side skipped normalization.
    for seed_key in ("data_seed", "adversarial_data_seed"):
        seed = normalized.get(seed_key)
        if not isinstance(seed, dict) or seed.get("mechanism") != "editor":
            continue
        calls = seed.get("editor_calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict):
                continue
            method = str(call.get("method") or "")
            args = call.get("args")
            if not isinstance(args, dict):
                continue
            template_args = editor_arg_templates.get(method)
            if isinstance(template_args, dict):
                _apply_route_editor_arg_template(args, template_args)
    return normalized


def _apply_route_editor_arg_template(args: dict[str, Any], template_args: dict[str, Any]) -> None:
    for key, template_value in template_args.items():
        if not _is_benign_template_token(template_value):
            continue
        args[key] = template_value
    if "project_path_template" in template_args:
        args.pop("project_id", None)


def _is_benign_template_token(value: Any) -> bool:
    return isinstance(value, str) and value.strip().startswith("{benign_")


__all__ = [
    "_normalize_generated_task_for_route",
    "validate_generated_novel_tasks",
    "validate_generated_novel_tasks_detailed",
]
