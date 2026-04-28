"""Phase 1 generate-new-tasks validation helpers."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
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
_ALLOWED_GENERATE_NEW_TASKS_EVALUATORS = {"NetworkEventEvaluator", "AgentResponseEvaluator"}
_DEFAULT_EXPECTED_TASK_COUNT = 30


@dataclass(frozen=True)
class GeneratedTaskValidationError:
    code: str
    path: str
    message: str
    expected: Any | None = None
    actual: Any | None = None
    repair_hint: str | None = None

    def render(self) -> str:
        parts = [f"{self.code} at {self.path}: {self.message}"]
        if self.expected is not None:
            parts.append(f"expected={self.expected!r}")
        if self.actual is not None:
            parts.append(f"actual={self.actual!r}")
        if self.repair_hint:
            parts.append(f"repair={self.repair_hint}")
        return "; ".join(parts)

    def legacy_render(self) -> str:
        return self.message

    def __contains__(self, text: object) -> bool:
        return isinstance(text, str) and text in self.render()

    def __str__(self) -> str:
        return self.render()

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "path": self.path,
            "message": self.message,
        }
        if self.expected is not None:
            payload["expected"] = self.expected
        if self.actual is not None:
            payload["actual"] = self.actual
        if self.repair_hint:
            payload["repair_hint"] = self.repair_hint
        return payload


def validate_generated_novel_tasks(
    raw_tasks: Any,
    *,
    site_name: str,
    profile: dict[str, Any],
    expected_task_count: int | None = _DEFAULT_EXPECTED_TASK_COUNT,
    route_contracts: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate sandbox-generated generate-new-tasks output for one site."""
    validated, detailed_errors = validate_generated_novel_tasks_detailed(
        raw_tasks,
        site_name=site_name,
        profile=profile,
        expected_task_count=expected_task_count,
        route_contracts=route_contracts,
    )
    return validated, [error.legacy_render() for error in detailed_errors]


def validate_generated_novel_tasks_detailed(
    raw_tasks: Any,
    *,
    site_name: str,
    profile: dict[str, Any],
    expected_task_count: int | None = _DEFAULT_EXPECTED_TASK_COUNT,
    route_contracts: dict[str, Any] | None = None,
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
    start_url_policy = _build_start_url_policy(profile)
    route_index = _route_contract_index(route_contracts)
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

    normalized = copy.deepcopy(task)
    seed = normalized.get("data_seed")
    if not isinstance(seed, dict) or seed.get("mechanism") != "editor":
        return normalized
    calls = seed.get("editor_calls")
    if not isinstance(calls, list):
        return normalized

    editor_arg_templates = route.get("editor_arg_templates")
    if not isinstance(editor_arg_templates, dict):
        return normalized
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


def validate_generated_novel_task(
    task: Any,
    *,
    index: int,
    site_name: str,
    allowed_eval_types: set[str],
    start_url_policy: _StartUrlPolicy | None = None,
    route_index: dict[str, dict[str, Any]] | None = None,
) -> GeneratedTaskValidationError | None:
    """Validate one new_task against Phase 1 and runtime constraints."""
    path = f"$[{index}]"
    if not isinstance(task, dict):
        return GeneratedTaskValidationError(
            code="TASK_NOT_OBJECT",
            path=path,
            message="task entry is not an object",
            expected="object",
            actual=type(task).__name__,
        )

    task_id = str(task.get("id", "missing"))
    missing_fields = [field for field in _NOVEL_TASK_REQUIRED_FIELDS if field not in task]
    if missing_fields:
        return GeneratedTaskValidationError(
            code="MISSING_REQUIRED_FIELD",
            path=path,
            message=f"task {task_id!r} missing required fields: {', '.join(missing_fields)}",
            expected=list(_NOVEL_TASK_REQUIRED_FIELDS),
            actual=sorted(task.keys()),
            repair_hint="Return a complete task object with every required field.",
        )

    task_id_pattern = re.compile(rf"^novel_{re.escape(site_name)}_\d+$")
    if not task_id_pattern.match(str(task["id"])):
        return _field_error(
            index,
            "INVALID_TASK_ID",
            "id",
            f"id must match novel_{site_name}_<n>",
            actual=task["id"],
        )

    if task.get("origin") != "new_task":
        return _field_error(
            index,
            "INVALID_ORIGIN",
            "origin",
            "origin must be 'new_task'",
            expected="new_task",
            actual=task.get("origin"),
        )

    if task.get("site") != site_name:
        return _field_error(
            index,
            "INVALID_SITE",
            "site",
            f"site must be {site_name!r}",
            expected=site_name,
            actual=task.get("site"),
        )

    sites = task.get("sites")
    if sites != [site_name]:
        return _field_error(
            index,
            "INVALID_SITES",
            "sites",
            f"sites must equal [{site_name!r}]",
            expected=[site_name],
            actual=sites,
        )

    instruction = task.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        return _field_error(
            index,
            "INVALID_INSTRUCTION",
            "instruction",
            "instruction must be a non-empty string",
            expected="non-empty string",
            actual=instruction,
        )

    start_urls = task.get("start_urls")
    if not isinstance(start_urls, list) or not start_urls:
        return _field_error(
            index,
            "INVALID_START_URLS",
            "start_urls",
            "start_urls must be a non-empty list",
            expected="non-empty list",
            actual=start_urls,
        )

    placeholder = placeholder_for_site(site_name)
    if placeholder is None:
        return _field_error(
            index,
            "NO_SITE_PLACEHOLDER",
            "start_urls",
            f"site {site_name!r} has no placeholder mapping",
        )
    for url in start_urls:
        if not isinstance(url, str) or not url.strip():
            return _field_error(
                index,
                "INVALID_START_URL",
                "start_urls",
                "start_urls entries must be non-empty strings",
                actual=url,
            )
        tokens = extract_placeholders(url)
        if placeholder not in tokens:
            return _field_error(
                index,
                "INVALID_PLACEHOLDER",
                "start_urls",
                f"start_urls must use {placeholder}",
                expected=placeholder,
                actual=url,
            )
        if any(token != placeholder for token in tokens):
            return _field_error(
                index,
                "INVALID_PLACEHOLDER",
                "start_urls",
                f"start_urls must only use {placeholder}",
                expected=placeholder,
                actual=sorted(tokens),
            )
    if route_index is not None:
        route_problem = _validate_route_contract_alignment(
            task,
            index=index,
            route_index=route_index,
        )
        if route_problem is not None:
            return route_problem
    policy_problem = None
    if start_url_policy is not None:
        policy_problem = start_url_policy.validate(start_urls, site_name=site_name)

    seed = task.get("data_seed") or {}
    if isinstance(seed, dict):
        mechanism = seed.get("mechanism")
        if mechanism not in (None, "none", "editor"):
            return _field_error(
                index,
                "DEPRECATED_SEED_MECHANISM",
                "data_seed.mechanism",
                f"data_seed.mechanism={mechanism!r} not allowed",
                expected=["editor", "none"],
                actual=mechanism,
                repair_hint="Use editor with editor_calls or none; api/form/state_push are deprecated.",
            )

    try:
        validate_data_seed(task.get("data_seed"), allow_none=True)
    except ValueError as exc:
        return _field_error(
            index,
            "INVALID_DATA_SEED",
            "data_seed",
            f"invalid data_seed: {exc}",
            actual=task.get("data_seed"),
        )

    reward = task.get("reward_function")
    if not isinstance(reward, dict):
        return _field_error(
            index,
            "INVALID_REWARD_FUNCTION",
            "reward_function",
            "reward_function must be an object",
            expected="object",
            actual=type(reward).__name__,
        )
    if "task_id" in reward:
        return _field_error(
            index,
            "REWARD_TASK_ID_FORBIDDEN",
            "reward_function.task_id",
            "reward_function must not include task_id",
            actual=reward.get("task_id"),
        )

    eval_configs = reward.get("eval")
    if not isinstance(eval_configs, list) or not eval_configs:
        return _field_error(
            index,
            "INVALID_REWARD_EVAL",
            "reward_function.eval",
            "reward_function.eval must be a non-empty list",
            expected="non-empty list",
            actual=eval_configs,
        )

    for eval_index, config in enumerate(eval_configs):
        if not isinstance(config, dict):
            return _field_error(
                index,
                "INVALID_EVALUATOR_CONFIG",
                f"reward_function.eval[{eval_index}]",
                f"eval[{eval_index}] must be an object",
                expected="object",
                actual=type(config).__name__,
            )
        evaluator = config.get("evaluator")
        if evaluator not in _ALLOWED_GENERATE_NEW_TASKS_EVALUATORS:
            return _field_error(
                index,
                "UNSUPPORTED_EVALUATOR",
                f"reward_function.eval[{eval_index}].evaluator",
                f"eval[{eval_index}] uses unsupported evaluator {evaluator!r}",
                expected=sorted(_ALLOWED_GENERATE_NEW_TASKS_EVALUATORS),
                actual=evaluator,
            )
        if evaluator not in allowed_eval_types:
            return _field_error(
                index,
                "EVALUATOR_NOT_IN_PROFILE",
                f"reward_function.eval[{eval_index}].evaluator",
                f"eval[{eval_index}] uses evaluator {evaluator!r} not declared in the site profile",
                expected=sorted(allowed_eval_types),
                actual=evaluator,
            )
        expected_problem = _validate_eval_expected(
            config,
            evaluator=evaluator,
            prefix=f"task {index} ({task_id})",
            eval_index=eval_index,
        )
        if expected_problem is not None:
            return _field_error(
                index,
                "INVALID_EVALUATOR_EXPECTED",
                f"reward_function.eval[{eval_index}].expected",
                expected_problem,
                actual=config.get("expected"),
            )

    placement_problem = _validate_phase2_placement_target(task, site_name=site_name)
    if placement_problem is not None:
        if policy_problem is not None:
            return _field_error(
                index,
                "START_URL_OUTSIDE_RENDER_PAGE",
                "start_urls",
                policy_problem,
                actual=task.get("start_urls"),
            )
        return _field_error(
            index,
            "INELIGIBLE_START_URL",
            "start_urls",
            placement_problem,
            actual=task.get("start_urls"),
            repair_hint="Choose an eligible route_id from TASK_ROUTE_CONTRACTS.json and align start_urls, data_seed editor method, instruction, and evaluator with that route.",
        )

    return None


def _validate_phase2_placement_target(task: dict[str, Any], *, site_name: str) -> str | None:
    """Require generated tasks to target pages Phase 2 can seed and verify.

    generate-new-tasks is useful only when the benign task naturally traverses the
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


def _field_error(
    task_index: int,
    code: str,
    field_path: str,
    message: str,
    *,
    expected: Any | None = None,
    actual: Any | None = None,
    repair_hint: str | None = None,
) -> GeneratedTaskValidationError:
    return GeneratedTaskValidationError(
        code=code,
        path=f"$[{task_index}].{field_path}",
        message=message,
        expected=expected,
        actual=actual,
        repair_hint=repair_hint,
    )


def _route_contract_index(
    route_contracts: dict[str, Any] | None,
) -> dict[str, dict[str, Any]] | None:
    if route_contracts is None:
        return None
    families = route_contracts.get("route_families")
    if not isinstance(families, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for family in families:
        if not isinstance(family, dict):
            continue
        route_id = family.get("id")
        if isinstance(route_id, str) and route_id.strip():
            out[route_id] = family
    return out or None


def _validate_route_contract_alignment(
    task: dict[str, Any],
    *,
    index: int,
    route_index: dict[str, dict[str, Any]],
) -> GeneratedTaskValidationError | None:
    route_id = task.get("route_id")
    if not isinstance(route_id, str) or not route_id.strip():
        return _field_error(
            index,
            "MISSING_ROUTE_ID",
            "route_id",
            "task must name the route contract it targets",
            expected=sorted(route_index),
            actual=route_id,
            repair_hint="Choose one eligible route id from TASK_ROUTE_CONTRACTS.json.",
        )
    route = route_index.get(route_id)
    if route is None:
        return _field_error(
            index,
            "UNKNOWN_ROUTE_ID",
            "route_id",
            "route_id is not present in TASK_ROUTE_CONTRACTS.json",
            expected=sorted(route_index),
            actual=route_id,
        )
    if route.get("enabled") is False or route.get("eligible") is False:
        return _field_error(
            index,
            "INELIGIBLE_ROUTE_ID",
            "route_id",
            "route_id is not enabled and eligible",
            actual=route_id,
        )

    start_urls = task.get("start_urls")
    patterns = [
        pattern
        for pattern in route.get("allowed_start_url_patterns", [])
        if isinstance(pattern, str) and pattern.strip()
    ]
    if patterns and isinstance(start_urls, list):
        if not any(
            isinstance(url, str) and _matches_route_url_pattern(url, pattern)
            for url in start_urls
            for pattern in patterns
        ):
            return _field_error(
                index,
                "ROUTE_START_URL_MISMATCH",
                "start_urls",
                "start_urls do not match the selected route contract",
                expected=patterns,
                actual=start_urls,
                repair_hint="Use a start URL shape listed on the selected route contract.",
            )

    seed = task.get("data_seed")
    if isinstance(seed, dict) and seed.get("mechanism") == "editor":
        calls = seed.get("editor_calls")
        methods = (
            [call.get("method") for call in calls if isinstance(call, dict)]
            if isinstance(calls, list)
            else []
        )
        allowed = [
            method
            for method in route.get("allowed_editor_methods", [])
            if isinstance(method, str) and method.strip()
        ]
        if allowed and not any(method in allowed for method in methods):
            return _field_error(
                index,
                "ROUTE_EDITOR_METHOD_MISMATCH",
                "data_seed.editor_calls",
                "editor seed methods do not match the selected route contract",
                expected=allowed,
                actual=methods,
                repair_hint="Use an allowed editor method from the selected route contract.",
            )

    requirements = route.get("instruction_requirements")
    if isinstance(requirements, dict):
        text = str(task.get("instruction") or "").casefold()
        include_any = _string_list(requirements.get("include_any"))
        if include_any and not any(token.casefold() in text for token in include_any):
            return _field_error(
                index,
                "ROUTE_INSTRUCTION_TOO_WEAK",
                "instruction",
                "instruction does not force the action required by the selected route",
                expected=include_any,
                actual=task.get("instruction"),
                repair_hint="Rewrite the instruction so the agent must consume the seeded rendered content.",
            )
        surface_terms = _string_list(requirements.get("include_any_surface_term"))
        if surface_terms and not any(token.casefold() in text for token in surface_terms):
            return _field_error(
                index,
                "ROUTE_INSTRUCTION_SURFACE_MISSING",
                "instruction",
                "instruction does not name the content region required by the selected route",
                expected=surface_terms,
                actual=task.get("instruction"),
            )

    return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item.strip()]


def _matches_route_url_pattern(url: str, pattern: str) -> bool:
    escaped = re.escape(pattern.rstrip("/"))
    escaped = escaped.replace(r"\{project_path\}", r".+?")
    escaped = escaped.replace(r"\{file_path\}", r".+?")
    escaped = re.sub(r"\\\{[^}]+\\\}", r"[^/?#]+", escaped)
    return re.match(rf"^{escaped}/?(?:[?#].*)?$", url.rstrip("/")) is not None


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
    existing_task_wraps: list[dict[str, Any]],
    novel_tasks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge existing-task wraps and new-task entries deterministically."""
    return list(existing_task_wraps) + sort_novel_tasks(novel_tasks)


def site_is_generate_new_tasks_eligible(profile: dict[str, Any]) -> bool:
    """Return whether a site profile has uncovered injection surfaces."""
    coverage = profile.get("existing_task_coverage", {})
    uncovered = coverage.get("injection_surfaces_without_task_coverage", [])
    return isinstance(uncovered, list) and bool(uncovered)
