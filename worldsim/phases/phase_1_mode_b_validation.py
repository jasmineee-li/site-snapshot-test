"""Phase 1 Mode B validation helpers."""

from __future__ import annotations

import re
from typing import Any

from worldsim.placeholders import extract_placeholders, placeholder_for_site
from worldsim.seeding import validate_data_seed

_NOVEL_TASK_REQUIRED_FIELDS = (
    "id",
    "site",
    "sites",
    "instruction",
    "start_urls",
    "data_seed",
    "reward_function",
)
_ALLOWED_MODE_B_EVALUATORS = {"NetworkEventEvaluator", "AgentResponseEvaluator"}
_DEFAULT_EXPECTED_TASK_COUNT = 30


def validate_generated_novel_tasks(
    raw_tasks: Any,
    *,
    site_name: str,
    profile: dict[str, Any],
    expected_task_count: int | None = _DEFAULT_EXPECTED_TASK_COUNT,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate sandbox-generated Mode B tasks for one site."""
    if not isinstance(raw_tasks, list):
        return [], ["sandbox output must be a JSON array"]

    allowed_eval_types = {
        capability.get("eval_type", "")
        for capability in profile.get("verification_capabilities", [])
        if capability.get("eval_type")
    }
    validated: list[dict[str, Any]] = []
    errors: list[str] = []
    seen_ids: set[str] = set()

    for index, task in enumerate(raw_tasks):
        problem = validate_generated_novel_task(
            task,
            index=index,
            site_name=site_name,
            allowed_eval_types=allowed_eval_types,
        )
        if problem is not None:
            errors.append(problem)
            continue

        task_id = str(task["id"])
        if task_id in seen_ids:
            errors.append(f"task {index} ({task_id}) duplicates a prior id")
            continue
        seen_ids.add(task_id)
        validated.append(task)

    if not validated and not errors:
        errors.append("sandbox produced no novel tasks")
    elif not errors and expected_task_count is not None and len(validated) != expected_task_count:
        errors.append(
            f"sandbox produced {len(validated)} novel tasks; expected {expected_task_count}"
        )

    return validated, errors


def validate_generated_novel_task(
    task: Any,
    *,
    index: int,
    site_name: str,
    allowed_eval_types: set[str],
) -> str | None:
    """Validate one Mode B task against Phase 1 and runtime constraints."""
    prefix = f"task {index}"
    if not isinstance(task, dict):
        return f"{prefix} is not an object"

    task_id = str(task.get("id", "missing"))
    prefix = f"task {index} ({task_id})"
    missing_fields = [field for field in _NOVEL_TASK_REQUIRED_FIELDS if field not in task]
    if missing_fields:
        return f"{prefix} missing required fields: {', '.join(missing_fields)}"

    task_id_pattern = re.compile(rf"^novel_{re.escape(site_name)}_\d+$")
    if not task_id_pattern.match(str(task["id"])):
        return f"{prefix} id must match novel_{site_name}_<n>"

    if task.get("site") != site_name:
        return f"{prefix} site must be {site_name!r}"

    sites = task.get("sites")
    if sites != [site_name]:
        return f"{prefix} sites must equal [{site_name!r}]"

    instruction = task.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        return f"{prefix} instruction must be a non-empty string"

    start_urls = task.get("start_urls")
    if not isinstance(start_urls, list) or not start_urls:
        return f"{prefix} start_urls must be a non-empty list"

    placeholder = placeholder_for_site(site_name)
    if placeholder is None:
        return f"{prefix} site {site_name!r} has no placeholder mapping"
    for url in start_urls:
        if not isinstance(url, str) or not url.strip():
            return f"{prefix} start_urls entries must be non-empty strings"
        tokens = extract_placeholders(url)
        if placeholder not in tokens:
            return f"{prefix} start_urls must use {placeholder}"
        if any(token != placeholder for token in tokens):
            return f"{prefix} start_urls must only use {placeholder}"

    try:
        validate_data_seed(task.get("data_seed"), allow_none=True)
    except ValueError as exc:
        return f"{prefix} invalid data_seed: {exc}"

    reward = task.get("reward_function")
    if not isinstance(reward, dict):
        return f"{prefix} reward_function must be an object"
    if "task_id" in reward:
        return f"{prefix} reward_function must not include task_id"

    eval_configs = reward.get("eval")
    if not isinstance(eval_configs, list) or not eval_configs:
        return f"{prefix} reward_function.eval must be a non-empty list"

    for eval_index, config in enumerate(eval_configs):
        if not isinstance(config, dict):
            return f"{prefix} eval[{eval_index}] must be an object"
        evaluator = config.get("evaluator")
        if evaluator not in _ALLOWED_MODE_B_EVALUATORS:
            return f"{prefix} eval[{eval_index}] uses unsupported evaluator {evaluator!r}"
        if evaluator not in allowed_eval_types:
            return (
                f"{prefix} eval[{eval_index}] uses evaluator {evaluator!r} "
                "not declared in the site profile"
            )
        expected_problem = _validate_eval_expected(
            config,
            evaluator=evaluator,
            prefix=prefix,
            eval_index=eval_index,
        )
        if expected_problem is not None:
            return expected_problem

    return None


def _validate_eval_expected(
    config: dict[str, Any],
    *,
    evaluator: str,
    prefix: str,
    eval_index: int,
) -> str | None:
    """Reject evaluator configs that would degenerate into near-no-op checks."""
    expected = config.get("expected")
    if not isinstance(expected, dict) or not expected:
        return f"{prefix} eval[{eval_index}] must include a non-empty expected object"

    if evaluator == "NetworkEventEvaluator":
        url = expected.get("url")
        if not isinstance(url, str) or not url.strip():
            return f"{prefix} eval[{eval_index}] NetworkEventEvaluator.expected.url must be a non-empty string"
        http_method = expected.get("http_method")
        if http_method is not None and (not isinstance(http_method, str) or not http_method.strip()):
            return f"{prefix} eval[{eval_index}] NetworkEventEvaluator.expected.http_method must be a non-empty string when provided"
        return None

    if evaluator == "AgentResponseEvaluator":
        if not any(key in expected for key in ("task_type", "status", "retrieved_data")):
            return (
                f"{prefix} eval[{eval_index}] AgentResponseEvaluator.expected must include "
                "at least one of task_type, status, or retrieved_data"
            )
        return None

    return None


def sort_novel_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort novel tasks by site, then id, for deterministic merges."""
    return sorted(
        tasks,
        key=lambda task: (str(task.get("site", "")), str(task.get("id", ""))),
    )


def merge_benign_tasks(
    mode_a_tasks: list[dict[str, Any]],
    novel_tasks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge Mode A and Mode B tasks deterministically."""
    return list(mode_a_tasks) + sort_novel_tasks(novel_tasks)


def site_is_mode_b_eligible(profile: dict[str, Any]) -> bool:
    """Return whether a site profile has uncovered injection surfaces."""
    coverage = profile.get("existing_task_coverage", {})
    uncovered = coverage.get("injection_surfaces_without_task_coverage", [])
    return isinstance(uncovered, list) and bool(uncovered)
