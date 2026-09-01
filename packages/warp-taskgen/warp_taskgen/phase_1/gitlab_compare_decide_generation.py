"""Authored Phase 1 handoff for GitLab comparison tasks.

The normal novel-task generator still owns task prose and route selection.  A
task-card generation contract opts one of those generated rows into the
feature-owned world compiler; no model-authored world shape is inferred.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from warp_taskgen.phase_1.gitlab_compare_decide import (
    compile_gitlab_compare_decide_task,
    generate_gitlab_compare_decide_world,
)

GITLAB_COMPARE_DECIDE_GENERATION_FAMILY = "gitlab_compare_decide"
_GENERATION_CONTRACT_KEY = "generation_contract"
_GENERATION_CONTRACT_VERSION = 1
_SELECTOR_KEYS = frozenset(
    {
        "labels",
        "project_description_template",
        "project_id",
        "project_name_template",
        "project_path_template",
    }
)
_PRESERVED_TASK_FIELDS = (
    "id",
    "origin",
    "site",
    "sites",
    "instruction",
    "start_urls",
    "route_id",
    "task_card_id",
    "capability_family",
    "benign_task_family_id",
    "task_provenance",
    "agent_context",
    "benign_target_resource",
    "benchmark",
)


def gitlab_compare_decide_generation_contract(
    card: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """Return the explicit authored contract for this feature, if selected."""
    if not isinstance(card, Mapping):
        return None
    raw = card.get(_GENERATION_CONTRACT_KEY)
    if not isinstance(raw, Mapping):
        return None
    if raw.get("family") != GITLAB_COMPARE_DECIDE_GENERATION_FAMILY:
        return None
    return raw


def compile_phase1_gitlab_compare_decide_task(
    task: Mapping[str, Any],
    *,
    task_card: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile one generated Phase 1 row under an authored family contract.

    The contract supplies the bounded world parameters.  The input row keeps
    its normal generated ID, prose, route, and provenance, while the feature
    replaces only its seed/reward/world fields with host-owned values.
    """
    if not isinstance(task, Mapping):
        raise TypeError("GitLab comparison Phase 1 task must be an object")
    contract = gitlab_compare_decide_generation_contract(task_card)
    if contract is None:
        return deepcopy(dict(task))
    if str(task_card.get("site") or task.get("site") or "").strip().lower() != "gitlab":
        raise ValueError("GitLab comparison generation contract requires a GitLab task card")
    version = contract.get("version", _GENERATION_CONTRACT_VERSION)
    if version != _GENERATION_CONTRACT_VERSION:
        raise ValueError(
            "GitLab comparison generation contract has unsupported version "
            f"{version!r}; expected {_GENERATION_CONTRACT_VERSION}"
        )
    decisive_key = _contract_text(contract, "decisive_record_key", "release-blocker")
    dependency = _contract_text(contract, "decision_dependency", "release-4")
    method = _phase1_method(task)
    world = generate_gitlab_compare_decide_world(
        decisive_record_key=decisive_key,
        decision_dependency=dependency,
        method=method,
    )
    resource = task.get("benign_target_resource")
    compiled = compile_gitlab_compare_decide_task(
        world,
        benign_target_resource=resource if isinstance(resource, Mapping) else None,
    )
    for field in _PRESERVED_TASK_FIELDS:
        if field in task:
            compiled[field] = deepcopy(task[field])
    compiled["data_seed"] = _phase1_seed_for_task(compiled["data_seed"], task, method=method)
    compiled["adversarial_data_seed"] = deepcopy(compiled["data_seed"])
    return compiled


def _contract_text(contract: Mapping[str, Any], key: str, default: str) -> str:
    value = contract.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"GitLab comparison generation contract {key!r} must be non-empty")
    return value.strip()


def _phase1_method(task: Mapping[str, Any]) -> str:
    source = task.get("data_seed")
    source_calls = source.get("editor_calls") if isinstance(source, Mapping) else None
    if not isinstance(source_calls, list) or not source_calls:
        raise ValueError(
            "GitLab comparison generation requires a route-selected editor seed call"
        )
    methods = {
        str(call.get("method") or "").strip()
        for call in source_calls
        if isinstance(call, Mapping)
    }
    methods.discard("")
    if len(methods) != 1 or methods.isdisjoint({"create_issue", "create_issue_description"}):
        raise ValueError(
            "GitLab comparison generation requires create_issue or "
            "create_issue_description as its route seed method"
        )
    return next(iter(methods))


def _phase1_seed_for_task(
    seed: Any,
    task: Mapping[str, Any],
    *,
    method: str,
) -> dict[str, Any]:
    """Adapt feature calls to the route method selected by normal generation."""
    if not isinstance(seed, Mapping):
        raise ValueError("GitLab comparison generation requires an editor seed")
    source = task.get("data_seed")
    source_calls = source.get("editor_calls") if isinstance(source, Mapping) else None
    if not isinstance(source_calls, list) or not source_calls:
        raise ValueError(
            "GitLab comparison generation requires a route-selected editor seed call"
        )
    compiled_calls = seed.get("editor_calls")
    if not isinstance(compiled_calls, list) or not compiled_calls:
        raise ValueError("GitLab comparison compiler emitted an invalid editor-call set")
    adapted_calls: list[dict[str, Any]] = []
    for index, generated_call in enumerate(compiled_calls):
        source_call = source_calls[index % len(source_calls)]
        if not isinstance(generated_call, Mapping) or not isinstance(source_call, Mapping):
            raise ValueError("GitLab comparison editor calls must be objects")
        args = generated_call.get("args")
        if not isinstance(args, Mapping):
            raise ValueError("GitLab comparison compiler emitted invalid editor arguments")
        source_args = source_call.get("args")
        adapted_args: dict[str, Any] = {}
        if isinstance(source_args, Mapping):
            for key, value in source_args.items():
                if key not in _SELECTOR_KEYS:
                    continue
                if isinstance(value, str) and value.strip().startswith("{benign_"):
                    generated_value = args.get(key)
                    if isinstance(generated_value, str) and not generated_value.startswith("{"):
                        adapted_args[key] = generated_value
                    continue
                adapted_args[key] = deepcopy(value)
        for selector in _SELECTOR_KEYS:
            generated_value = args.get(selector)
            if selector not in adapted_args and isinstance(generated_value, str) and not generated_value.startswith("{"):
                adapted_args[selector] = generated_value
        if method == "create_issue_description":
            adapted_args["body"] = args["body_template"]
            adapted_args["title"] = args["title_template"]
        else:
            adapted_args.update(deepcopy(dict(args)))
        adapted_calls.append(
            {
                "benchmark": generated_call.get("benchmark", "webarena_verified"),
                "site": "gitlab",
                "method": method,
                "logical_record_key": generated_call["logical_record_key"],
                "resource_kind": generated_call["resource_kind"],
                "args": adapted_args,
            }
        )
    return {"mechanism": "editor", "editor_calls": adapted_calls}


__all__ = [
    "GITLAB_COMPARE_DECIDE_GENERATION_FAMILY",
    "compile_phase1_gitlab_compare_decide_task",
    "gitlab_compare_decide_generation_contract",
]
