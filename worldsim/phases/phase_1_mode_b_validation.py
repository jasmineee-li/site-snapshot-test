"""Phase 1 Mode B validation helpers."""

from __future__ import annotations

import re
from typing import Any

from worldsim.phases.phase_2_exposure_contract import build_exposure_contract
from worldsim.phases.phase_2_target_resolver import derive_benign_target_resource
from worldsim.placeholders import extract_placeholders, placeholder_for_site
from worldsim.seeding import validate_data_seed

_NOVEL_TASK_REQUIRED_FIELDS = (
    "id",
    "origin",
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
    start_url_policy = _build_start_url_policy(profile)
    validated: list[dict[str, Any]] = []
    errors: list[str] = []
    seen_ids: set[str] = set()

    for index, task in enumerate(raw_tasks):
        problem = validate_generated_novel_task(
            task,
            index=index,
            site_name=site_name,
            allowed_eval_types=allowed_eval_types,
            start_url_policy=start_url_policy,
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
    start_url_policy: _StartUrlPolicy | None = None,
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

    if task.get("origin") != "new_task":
        return f"{prefix} origin must be 'new_task'"

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
    if start_url_policy is not None:
        policy_problem = start_url_policy.validate(start_urls, site_name=site_name)
        if policy_problem is not None:
            return f"{prefix} {policy_problem}"

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

    placement_problem = _validate_phase2_placement_target(task, site_name=site_name)
    if placement_problem is not None:
        return f"{prefix} {placement_problem}"

    return None


def _validate_phase2_placement_target(task: dict[str, Any], *, site_name: str) -> str | None:
    """Require generated tasks to target pages Phase 2 can seed and verify.

    Mode B is useful only when the benign task naturally traverses the
    same surface Phase 2 can seed. This calls the same deterministic
    resolver and exposure-contract builder Phase 2 uses, so future
    benchmark support extends by adding placement contracts rather than
    broadening this validator by hand.
    """
    if site_name not in {"gitlab", "reddit"}:
        return None
    placeholder = placeholder_for_site(site_name)
    if placeholder is None:
        return f"site {site_name!r} has no placeholder mapping"
    placeholders = {placeholder: f"https://{site_name}.local"}
    resource = derive_benign_target_resource(task, placeholders)
    contract = build_exposure_contract(
        benign_task_id=str(task.get("id") or ""),
        site=site_name,
        benchmark=str(task.get("benchmark") or "webarena_verified"),
        benign_target_resource=resource,
    )
    eligibility = contract.get("eligibility") if isinstance(contract, dict) else None
    if isinstance(eligibility, dict) and eligibility.get("status") == "eligible":
        return None
    reason = "unknown"
    if isinstance(eligibility, dict):
        reason = str(eligibility.get("reason") or reason)
    elif isinstance(contract, dict):
        reason = str(contract.get("reason") or reason)
    return (
        "start_urls must resolve to an eligible Phase 2 exposure contract; "
        f"resolver kind={resource.get('kind')!r}, reason={reason!r}, "
        f"start_urls={task.get('start_urls')!r}"
    )


class _StartUrlPolicy:
    def __init__(
        self,
        location_pages: list[str],
        location_patterns: list[re.Pattern[str]],
    ) -> None:
        self.location_pages = location_pages
        self.location_patterns = location_patterns

    def validate(self, start_urls: list[str], *, site_name: str) -> str | None:
        paths = [_placeholder_path(url, site_name) for url in start_urls]
        if self.location_patterns and not any(
            pattern.match(path) for path in paths for pattern in self.location_patterns
        ):
            return (
                "start_urls must route through an uncovered injection-surface render page; "
                f"got {start_urls!r}; allowed location_page shapes: {self.location_pages!r}"
            )
        if not self.location_patterns and any(_looks_like_mutation_entry(path) for path in paths):
            return (
                "start_urls must route through rendered content, not a create or edit form; "
                f"got {start_urls!r}"
            )
        return None


def _build_start_url_policy(profile: dict[str, Any]) -> _StartUrlPolicy | None:
    uncovered = profile.get("existing_task_coverage", {}).get(
        "injection_surfaces_without_task_coverage", []
    )
    if not isinstance(uncovered, list) or not uncovered:
        return None
    uncovered_ids = {str(item) for item in uncovered}
    location_pages: list[str] = []
    patterns: list[re.Pattern[str]] = []
    for surface in profile.get("injection_surface", []):
        if not isinstance(surface, dict) or str(surface.get("id", "")) not in uncovered_ids:
            continue
        location = surface.get("location_page")
        if isinstance(location, str) and location.strip():
            location_pages.append(location.strip())
            patterns.append(_location_page_pattern(location))
    return _StartUrlPolicy(location_pages, patterns)


def _location_page_pattern(location_page: str) -> re.Pattern[str]:
    path = re.sub(r"^__[A-Z0-9_]+__", "", location_page.strip())
    escaped = re.escape(path.rstrip("/"))
    escaped = re.sub(r"\\\{[^}]+\\\}", r"[^/]+", escaped)
    return re.compile(rf"^{escaped}/?(?:[?#].*)?$")


def _placeholder_path(url: str, site_name: str) -> str:
    placeholder = placeholder_for_site(site_name) or ""
    if placeholder and url.startswith(placeholder):
        return url[len(placeholder) :] or "/"
    return url


def _looks_like_mutation_entry(path: str) -> bool:
    lowered = path.lower().rstrip("/")
    segments = [segment for segment in lowered.split("/") if segment]
    return any(segment in {"new", "edit", "submit", "create_forum"} for segment in segments)


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
        if http_method is not None and (
            not isinstance(http_method, str) or not http_method.strip()
        ):
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
