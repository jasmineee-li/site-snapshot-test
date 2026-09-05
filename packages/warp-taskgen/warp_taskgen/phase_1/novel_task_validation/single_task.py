"""Single-task validation for generated novel tasks."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from typing import Any

from warp_taskgen.phase_1.generated_workflows import validated_host_action_contract
from warp_taskgen.phase_1.novel_task_validation.answer_stability import (
    _validate_agent_response_instruction_deliverable,
)
from warp_taskgen.phase_1.novel_task_validation.errors import (
    GeneratedTaskValidationError,
    _field_error,
)
from warp_taskgen.phase_1.novel_task_validation.placement import (
    _validate_no_literal_sample_editor_values,
    _validate_phase2_placement_target,
)
from warp_taskgen.phase_1.novel_task_validation.rewards import (
    _is_host_action_only_placeholder_evaluator,
    _is_host_compiled_final_state_evaluator,
    _validate_eval_expected,
)
from warp_taskgen.phase_1.novel_task_validation.route_alignment import (
    _StartUrlPolicy,
    _validate_route_contract_alignment,
)
from warp_taskgen.phase_1.novel_task_validation.task_cards import (
    _strip_model_authored_host_metadata,
    _task_uses_host_action_only_card,
    _validate_task_card_alignment,
)
from warp_taskgen.placeholders import extract_placeholders, placeholder_for_site
from warp_taskgen.seeding import validate_data_seed

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


_ALLOWED_GENERATE_NEW_TASKS_EVALUATORS = {
    "NetworkEventEvaluator",
    "AgentResponseEvaluator",
}


def validate_generated_novel_task(
    task: Any,
    *,
    index: int,
    site_name: str,
    allowed_eval_types: set[str],
    start_url_policy: _StartUrlPolicy | None = None,
    route_index: dict[str, dict[str, Any]] | None = None,
    task_card_index: dict[str, dict[str, Any]] | None = None,
    host_compiled_evaluator_types: frozenset[str] = frozenset(),
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
    preserved_benign_action_contract = _preserved_host_compiled_action_contract(
        task,
        task_card_index=task_card_index,
    )
    _strip_model_authored_host_metadata(task)
    if preserved_benign_action_contract is not None:
        # Keep only the contract that the feature compiler already proved. The
        # ordinary model-authored provenance remains stripped below.
        task["task_provenance"] = {
            "benign_action_contract": copy.deepcopy(preserved_benign_action_contract)
        }
    if route_index is not None:
        route_problem = _validate_route_contract_alignment(
            task,
            index=index,
            route_index=route_index,
            skip_answer_response_checks=_task_uses_host_action_only_card(
                task,
                task_card_index,
            ),
        )
        if route_problem is not None:
            return route_problem
    if task_card_index:
        card_problem = _validate_task_card_alignment(
            task,
            index=index,
            site_name=site_name,
            card_index=task_card_index,
            route_index=route_index,
            preserved_benign_action_contract=preserved_benign_action_contract,
        )
        if card_problem is not None:
            return card_problem
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
    sample_literal_problem = _validate_no_literal_sample_editor_values(task)
    if sample_literal_problem is not None:
        return _field_error(
            index,
            "LITERAL_SAMPLE_EDITOR_VALUE",
            sample_literal_problem[0],
            sample_literal_problem[1],
            actual=sample_literal_problem[2],
            repair_hint=(
                "Use route-provided task-scoped sample values or a task-specific value "
                "instead of copying literal prompt examples."
            ),
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
        host_compiled_final_state = _is_host_compiled_final_state_evaluator(task, config)
        host_action_only_placeholder = _is_host_action_only_placeholder_evaluator(
            task,
            task_card_index,
            config,
        )
        if (
            evaluator not in _ALLOWED_GENERATE_NEW_TASKS_EVALUATORS
            and evaluator not in host_compiled_evaluator_types
            and not host_compiled_final_state
            and not host_action_only_placeholder
        ):
            return _field_error(
                index,
                "UNSUPPORTED_EVALUATOR",
                f"reward_function.eval[{eval_index}].evaluator",
                f"eval[{eval_index}] uses unsupported evaluator {evaluator!r}",
                expected=sorted(
                    _ALLOWED_GENERATE_NEW_TASKS_EVALUATORS | host_compiled_evaluator_types
                ),
                actual=evaluator,
            )
        if (
            evaluator not in allowed_eval_types
            and not host_compiled_final_state
            and not host_action_only_placeholder
        ):
            return _field_error(
                index,
                "EVALUATOR_NOT_IN_PROFILE",
                f"reward_function.eval[{eval_index}].evaluator",
                f"eval[{eval_index}] uses evaluator {evaluator!r} not declared in the site profile",
                expected=sorted(allowed_eval_types),
                actual=evaluator,
            )
        if not host_action_only_placeholder:
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

    deliverable_problem = _validate_agent_response_instruction_deliverable(
        task,
        eval_configs=eval_configs,
        route_index=route_index,
        skip_for_host_action_only=_task_uses_host_action_only_card(
            task,
            task_card_index,
        ),
    )
    if deliverable_problem is not None:
        return _field_error(
            index,
            "NAVIGATION_ONLY_INSTRUCTION",
            "instruction",
            deliverable_problem,
            actual=task.get("instruction"),
            repair_hint=(
                "Rewrite AgentResponseEvaluator tasks so the user asks for a "
                "specific answer, title, summary, comparison, or page-derived value."
            ),
        )

    placement_problem = _validate_phase2_placement_target(
        task,
        site_name=site_name,
        route_index=route_index,
    )
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


def _preserved_host_compiled_action_contract(
    task: Mapping[str, Any],
    *,
    task_card_index: dict[str, dict[str, Any]] | None,
) -> Mapping[str, Any] | None:
    """Return a feature contract only after the comparison owner revalidates it.

    A generated row may carry a final-state evaluator before generic Phase 1
    validation.  Preserve its benign-action provenance only for an explicitly
    opted-in compare-and-act card and only when the feature's own canonical
    validator accepts the complete host output.  Raw model rows therefore do
    not gain a profile bypass merely by copying a reward and provenance shape.
    """

    card_id = task.get("task_card_id")
    if not isinstance(card_id, str) or not isinstance(task_card_index, Mapping):
        return None
    task_card = task_card_index.get(card_id)
    if not isinstance(task_card, Mapping):
        return None
    return validated_host_action_contract(task, task_card=task_card)


__all__ = [
    "validate_generated_novel_task",
]
