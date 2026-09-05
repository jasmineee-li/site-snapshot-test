"""Route-contract and start-URL validation exports."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from warp_taskgen.phase_1.novel_task_validation.answer_stability import (
    _string_list,
    _validate_answer_stability,
    _validate_binary_status_output_contract,
)
from warp_taskgen.phase_1.novel_task_validation.errors import (
    GeneratedTaskValidationError,
    _field_error,
)
from warp_taskgen.placeholders import placeholder_for_site


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
    if not out and str(route_contracts.get("site") or "").strip().lower() not in {
        "gitlab",
        "reddit",
    }:
        return None
    return out


def _validate_route_contract_alignment(
    task: dict[str, Any],
    *,
    index: int,
    route_index: dict[str, dict[str, Any]],
    skip_answer_response_checks: bool = False,
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
        invalid_urls = [
            url
            for url in start_urls
            if not (
                isinstance(url, str)
                and any(_matches_route_url_pattern(url, pattern) for pattern in patterns)
            )
        ]
        if invalid_urls:
            return _field_error(
                index,
                "ROUTE_START_URL_MISMATCH",
                "start_urls",
                "start_urls do not match the selected route contract",
                expected=patterns,
                actual=invalid_urls,
                repair_hint="Use a start URL shape listed on the selected route contract.",
            )

    example_start_urls = _route_anchor_example_start_urls(route)
    if example_start_urls and isinstance(start_urls, list):
        invalid_inventory_urls = [
            url for url in start_urls if not (isinstance(url, str) and url in example_start_urls)
        ]
        if invalid_inventory_urls:
            return _field_error(
                index,
                "ROUTE_START_URL_NOT_IN_INVENTORY",
                "start_urls",
                "start_urls must use an inventory-backed example from the selected route contract",
                expected=example_start_urls,
                actual=invalid_inventory_urls,
                repair_hint="Use one anchor_examples[].start_url from the selected route contract.",
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
        invalid_methods = [method for method in methods if method not in allowed]
        if allowed and (not methods or invalid_methods):
            return _field_error(
                index,
                "ROUTE_EDITOR_METHOD_MISMATCH",
                "data_seed.editor_calls",
                "editor seed methods do not match the selected route contract",
                expected=allowed,
                actual=invalid_methods or methods,
                repair_hint="Use an allowed editor method from the selected route contract.",
            )

    requirements = route.get("instruction_requirements")
    if isinstance(requirements, dict):
        text = str(task.get("instruction") or "").casefold()
        include_any = _string_list(requirements.get("include_any"))
        include_any_regex = _string_list(requirements.get("include_any_regex"))
        matches_include_any = any(token.casefold() in text for token in include_any)
        matches_include_any_regex = any(
            re.search(pattern, text) is not None for pattern in include_any_regex
        )
        if (include_any or include_any_regex) and not (
            matches_include_any or matches_include_any_regex
        ):
            return _field_error(
                index,
                "ROUTE_INSTRUCTION_TOO_WEAK",
                "instruction",
                "instruction does not force the action required by the selected route",
                expected={"include_any": include_any, "include_any_regex": include_any_regex},
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
        avoid_any = _string_list(requirements.get("avoid_any"))
        avoid_any_regex = _string_list(requirements.get("avoid_any_regex"))
        matched_avoid_terms = [token for token in avoid_any if token.casefold() in text]
        matched_avoid_regex = [
            pattern for pattern in avoid_any_regex if re.search(pattern, text) is not None
        ]
        if matched_avoid_terms or matched_avoid_regex:
            preferred = _string_list(requirements.get("preferred_selector_terms"))
            return _field_error(
                index,
                "ROUTE_INSTRUCTION_ROUTE_DRIFT_RISK",
                "instruction",
                "instruction uses wording that can move the agent away from the verified route",
                expected={
                    "avoid_any": avoid_any,
                    "avoid_any_regex": avoid_any_regex,
                    "preferred_selector_terms": preferred,
                    "reason": requirements.get("route_drift_reason"),
                },
                actual=task.get("instruction"),
                repair_hint=(
                    "Rewrite the instruction to consume the seeded content on the copied "
                    "start URL. Use route-local selector wording such as "
                    f"{', '.join(preferred) if preferred else 'first visible'}."
                ),
            )

    if not skip_answer_response_checks:
        answer_stability_problem = _validate_answer_stability(task, route)
        if answer_stability_problem is not None:
            return _field_error(
                index,
                "ANSWER_TARGET_COLLISION_RISK",
                "reward_function.eval",
                answer_stability_problem,
                actual=task.get("reward_function"),
                repair_hint=(
                    "Keep the instruction forcing the target surface encounter, but make "
                    "AgentResponseEvaluator.expected.retrieved_data an invariant answer "
                    "such as whether the title/body is blank or populated, not the exact "
                    "seeded field selected by latest/newest/first ordering."
                ),
            )

        binary_output_problem = _validate_binary_status_output_contract(task, route)
        if binary_output_problem is not None:
            return _field_error(
                index,
                "BINARY_STATUS_OUTPUT_UNCLEAR",
                "instruction",
                binary_output_problem,
                actual=task.get("instruction"),
                repair_hint=(
                    "For blank/populated tasks, explicitly tell the agent to answer "
                    "with exactly 'blank' or 'populated' as the retrieved value. For "
                    "body/description routes, define blank as operational field-state "
                    "content or use a link/no-link stable answer instead."
                ),
            )

    return None


def _route_anchor_example_start_urls(route: Mapping[str, Any]) -> list[str]:
    if route.get("requires_inventory_backed_start_url") is not True:
        return []
    examples = route.get("anchor_examples")
    if not isinstance(examples, list):
        return []
    return [
        str(example.get("start_url"))
        for example in examples
        if isinstance(example, Mapping)
        and isinstance(example.get("start_url"), str)
        and example.get("start_url")
    ]


def _matches_route_url_pattern(url: str, pattern: str) -> bool:
    escaped = re.escape(pattern.rstrip("/"))
    # GitLab project routes must be namespace-qualified. A one-segment value
    # like `/design/-/issues` is ambiguous with user/group roots and the Phase 2
    # resolver intentionally refuses it, so Phase 1 validation must not accept
    # it as a syntactic `{project_path}` fill.
    escaped = escaped.replace(r"\{project_path\}", r"(?:[^/?#]+/)+[^/?#]+")
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


__all__ = [
    "_StartUrlPolicy",
    "_build_start_url_policy",
    "_location_page_pattern",
    "_looks_like_mutation_entry",
    "_matches_route_url_pattern",
    "_placeholder_path",
    "_route_anchor_example_start_urls",
    "_route_contract_index",
    "_validate_route_contract_alignment",
]
