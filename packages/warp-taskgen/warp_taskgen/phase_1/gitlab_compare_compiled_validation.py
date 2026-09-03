"""Idempotence checks for host-compiled GitLab comparison rows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from warp_taskgen.phase_1.gitlab_compare_decide import (
    expected_gitlab_compare_decide_response,
    is_feature_default_world,
    select_gitlab_record,
    validate_gitlab_comparison_predicate_sensitivity,
)
from warp_taskgen.phase_1.gitlab_compare_decide_content import (
    GENERATED_COMPARISON_CONTENT_KEY,
    _validate_compare_act_instruction,
    _validate_compare_decide_instruction,
    _world_from_compiled_task,
    validate_compiled_comparison_act,
    validate_compiled_comparison_seed,
)
from warp_taskgen.phase_1.gitlab_compare_decide_reward import (
    build_gitlab_compare_decide_reward,
)

GENERATION_CONTRACT_VERSION = 1
HOST_COMPILED_CONTENT_SOURCES = frozenset({"feature_default", "warp_generated"})


def is_host_compiled_comparison_task(
    task: Mapping[str, Any],
    *,
    act: bool,
    task_card_id: str,
) -> bool:
    """Accept a prior compiler result only after rechecking every derived seam."""

    comparison_contract = task.get("comparison_contract")
    if not isinstance(comparison_contract, Mapping):
        return False
    if comparison_contract.get("workflow_source") != "task_card":
        return False
    if GENERATED_COMPARISON_CONTENT_KEY in task:
        return False
    _validate_host_compiled_task(
        task,
        act=act,
        task_card_id=task_card_id,
    )
    return True


def validate_gitlab_compare_decide_task(
    task: Mapping[str, Any],
    *,
    require_instruction: bool = True,
) -> None:
    """Validate the feature's world, instruction, contract, and answer key."""

    if not isinstance(task, Mapping):
        raise ValueError("GitLab comparison task must be an object")
    world = task.get("world")
    if not isinstance(world, Mapping):
        raise ValueError("GitLab comparison task must include a world")
    resolved = _world_from_compiled_task(task)
    validate_gitlab_comparison_predicate_sensitivity(resolved)
    selected = select_gitlab_record(resolved)
    contract = task.get("comparison_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("GitLab comparison task must include comparison_contract")
    if contract.get("benchmark") not in (None, resolved.benchmark):
        raise ValueError("GitLab comparison contract benchmark disagrees with world")
    if str(contract.get("site") or resolved.site).strip().lower() != resolved.site:
        raise ValueError("GitLab comparison contract site disagrees with world")
    actual_keys = [record.logical_record_key for record in resolved.records]
    if contract.get("expected_logical_record_keys") != actual_keys:
        raise ValueError("GitLab comparison contract logical record keys disagree with world")
    declared_keys = contract.get("record_keys")
    if declared_keys is not None and declared_keys != actual_keys:
        raise ValueError("GitLab comparison contract record_keys disagree with world")
    if contract.get("selected_logical_record_key") != selected.logical_record_key:
        raise ValueError("GitLab comparison contract selected record disagrees with decision rule")
    if _canonical_mapping(contract.get("decision_rule")) != dict(resolved.decision_rule):
        raise ValueError("GitLab comparison contract decision rule disagrees with world")
    if require_instruction:
        _validate_compare_decide_instruction(task.get("instruction"), resolved)
        reward = task.get("reward_function")
        if not isinstance(reward, Mapping):
            raise ValueError("GitLab comparison task must include reward_function")
        evals = reward.get("eval")
        if not isinstance(evals, list) or not evals:
            raise ValueError("GitLab comparison task reward must include an evaluator")
        expected = evals[0].get("expected") if isinstance(evals[0], Mapping) else None
        if expected != expected_gitlab_compare_decide_response(resolved):
            raise ValueError("GitLab comparison expected response disagrees with decision rule")


def _canonical_mapping(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): str(raw).strip() for key, raw in value.items()}


def _validate_host_compiled_task(
    task: Mapping[str, Any],
    *,
    act: bool,
    task_card_id: str,
) -> None:
    comparison_contract = task.get("comparison_contract")
    if not isinstance(comparison_contract, Mapping):
        raise ValueError("GitLab comparison host output requires comparison_contract")
    if comparison_contract.get("generation_contract_version") != GENERATION_CONTRACT_VERSION:
        raise ValueError("GitLab comparison host output has an unsupported generation version")
    if comparison_contract.get("content_source") not in HOST_COMPILED_CONTENT_SOURCES:
        raise ValueError("GitLab comparison host output has an unknown content source")
    if task.get("site") != "gitlab" or task.get("sites") != ["gitlab"]:
        raise ValueError("GitLab comparison host output has an invalid site declaration")
    if task.get("benchmark") not in (None, "webarena_verified"):
        raise ValueError("GitLab comparison host output has an invalid benchmark declaration")
    if task.get("task_card_id") != task_card_id:
        raise ValueError("GitLab comparison host output has the wrong task-card identity")
    world = _world_from_compiled_task(task)
    validate_gitlab_comparison_predicate_sensitivity(world)
    selected = select_gitlab_record(world)
    if world.decisive_record_key != selected.logical_record_key:
        raise ValueError("GitLab comparison world decisive record disagrees with the decision rule")
    if task.get("world") != world.as_mapping():
        raise ValueError("GitLab comparison host output world is not canonical")
    expected_content_source = (
        "feature_default" if is_feature_default_world(world) else "warp_generated"
    )
    if comparison_contract.get("content_source") != expected_content_source:
        raise ValueError(
            "GitLab comparison host output content source does not match the canonical world"
        )
    record_keys = [record.logical_record_key for record in world.records]
    expected_contract = {
        "benchmark": world.benchmark,
        "site": world.site,
        "method": world.method,
        "resource_kind": world.resource_kind,
        "expected_logical_record_keys": record_keys,
        "selected_logical_record_key": selected.logical_record_key,
        "decision_rule": dict(world.decision_rule),
        "record_keys": record_keys,
        "workflow_source": "task_card",
        "content_source": comparison_contract.get("content_source"),
        "generation_contract_version": GENERATION_CONTRACT_VERSION,
        "task_card_id": task_card_id,
    }
    if dict(comparison_contract) != expected_contract:
        raise ValueError("GitLab comparison host output contract is not canonical")
    provenance = task.get("task_provenance")
    if not isinstance(provenance, Mapping) or provenance.get("task_card_id") != task_card_id:
        raise ValueError("GitLab comparison host output has invalid task-card provenance")
    validate_gitlab_compare_decide_task(task, require_instruction=not act)
    if act:
        _validate_compare_act_instruction(task.get("instruction"), world)
        validate_compiled_comparison_act(
            task,
            world,
            expected_task_card_id=task_card_id,
        )
    elif task.get("reward_function") != build_gitlab_compare_decide_reward(world):
        raise ValueError("GitLab compare-and-decide host output reward is not canonical")
    validate_compiled_comparison_seed(task, world)
    if not act and "comparison_act_contract" in task:
        raise ValueError("GitLab compare-and-decide host output must not include an act contract")


__all__ = [
    "GENERATION_CONTRACT_VERSION",
    "HOST_COMPILED_CONTENT_SOURCES",
    "is_host_compiled_comparison_task",
    "validate_gitlab_compare_decide_task",
]
