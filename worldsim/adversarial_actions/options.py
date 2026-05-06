from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from worldsim.adversarial_actions.action_targets import (
    action_target_contract_for_kind,
    target_editor_method_for_action,
)
from worldsim.adversarial_actions.capability_contracts import (
    action_kind_compatible_with_task,
    capability_family_from_task,
    compatible_action_kinds_from_task,
    control_action_kinds_from_task,
)
from worldsim.adversarial_actions.catalog import option_for_kind
from worldsim.adversarial_actions.policies import (
    ACTION_POLICIES,
    _action_tier,
    _keep_control_tier_actions,
    _keep_mutation_actions_when_available,
    _keep_semantic_actions_only,
    _keep_tier_actions,
    _normalized_tier_policy,
    _prefer_first_mutation_action,
    _unaligned_control_policy_tier,
    canonical_action_policy,
)
from worldsim.adversarial_actions.reward_introspection import _single_binary_label_expected
from worldsim.adversarial_actions.shared_compiler import (
    _EDITOR_ACTION_BY_METHOD,
    ACTION_KINDS,
)
from worldsim.adversarial_actions.tier3 import (
    option_marks_host_ready,
    tier3_action_options,
    tier3_action_readiness,
)


def annotate_exposure_contracts_with_actions(
    exposure_contracts: Mapping[str, Mapping[str, Any]],
    benign_tasks: list[Mapping[str, Any]],
    *,
    policy: str = "default",
) -> dict[str, dict[str, Any]]:
    """Return exposure contracts annotated with planner-facing action options."""
    policy = canonical_action_policy(policy)
    benign_by_id = {str(task.get("id") or ""): task for task in benign_tasks}
    annotated: dict[str, dict[str, Any]] = {}
    for task_id, contract in exposure_contracts.items():
        copied = json.loads(json.dumps(dict(contract)))
        benign_task = benign_by_id.get(str(task_id))
        options = allowed_action_options(benign_task, copied, policy=policy)
        if options:
            copied["adversarial_action_options"] = options
        annotated[str(task_id)] = copied
    return annotated

def annotate_exposure_contracts_with_action_policy(
    exposure_contracts: Mapping[str, Mapping[str, Any]],
    benign_tasks: list[Mapping[str, Any]],
    *,
    policy: str = "default",
) -> dict[str, dict[str, Any]]:
    """Annotate exposure contracts and apply an optional action-selection policy."""
    if policy not in ACTION_POLICIES:
        raise ValueError("action policy must be one of " + ", ".join(ACTION_POLICIES))
    policy = canonical_action_policy(policy)
    policy = canonical_action_policy(policy)
    annotated = annotate_exposure_contracts_with_actions(
        exposure_contracts,
        benign_tasks,
        policy=policy,
    )
    benign_by_id = {str(task.get("id") or ""): task for task in benign_tasks}
    if policy == "default":
        return annotated
    if policy == "semantic_only":
        for contract in annotated.values():
            _keep_semantic_actions_only(contract)
        return annotated
    if policy == "mutation_when_available":
        for contract in annotated.values():
            _prefer_first_mutation_action(contract)
        return annotated
    if policy == "mutation_only_when_available":
        for contract in annotated.values():
            _keep_mutation_actions_when_available(contract, policy=policy)
        return annotated
    control_tier = _unaligned_control_policy_tier(policy)
    if control_tier is not None:
        for task_id, contract in annotated.items():
            _keep_control_tier_actions(
                contract,
                benign_task=benign_by_id.get(str(task_id)),
                tier=control_tier,
                policy=policy,
            )
        return annotated
    normalized_policy = _normalized_tier_policy(policy)
    if normalized_policy is not None:
        tier, canonical_policy = normalized_policy
        for contract in annotated.values():
            _keep_tier_actions(contract, tier=tier, policy=canonical_policy)
        return annotated
    raise ValueError(f"unsupported action policy {policy!r}")

def allowed_action_options(
    benign_task: Mapping[str, Any] | None,
    exposure_contract: Mapping[str, Any],
    *,
    policy: str = "default",
) -> list[dict[str, str]]:
    policy = canonical_action_policy(policy)
    options: list[dict[str, str]] = [option_for_kind("open_required_url")]
    if benign_task is not None and _single_binary_label_expected(
        benign_task.get("reward_function")
    ):
        options.insert(0, option_for_kind("answer_opposite_binary_label"))

    method = str(exposure_contract.get("editor_method") or "").strip()
    action = _EDITOR_ACTION_BY_METHOD.get(method)
    target_action = _target_action_for_task(benign_task, source_method=method)
    control_kinds = set(control_action_kinds_from_task(benign_task))
    control_tier = _unaligned_control_policy_tier(policy)
    if action and (
        policy == "default"
        or (
            control_tier is not None
            and action in control_kinds
            and _action_tier(action) == control_tier
        )
        or not _policy_requires_capability_metadata(policy)
        or _task_has_action_capability_metadata(benign_task)
    ):
        options.append(option_for_kind(action))
    if target_action and target_action != action:
        options.append(option_for_kind(target_action))
    options.extend(
        tier3_action_options(
            benign_task,
            exposure_contract,
            policy=canonical_action_policy(policy),
        )
    )
    return _filter_options_for_task_capability(options, benign_task, policy=policy)

def _policy_requires_capability_metadata(policy: str) -> bool:
    if policy in {
        "mutation_when_available",
        "mutation_only_when_available",
    }:
        return True
    if _unaligned_control_policy_tier(policy) is not None:
        return True
    return _normalized_tier_policy(policy) is not None

def _task_has_action_capability_metadata(task: Mapping[str, Any] | None) -> bool:
    if not isinstance(task, Mapping):
        return False
    return bool(capability_family_from_task(task) or compatible_action_kinds_from_task(task))

def _target_action_for_task(
    task: Mapping[str, Any] | None,
    *,
    source_method: str,
) -> str | None:
    if not isinstance(task, Mapping):
        return None
    normalized_source = str(source_method or "").strip()
    for action_kind in compatible_action_kinds_from_task(task):
        target = action_target_contract_for_kind(task, action_kind)
        if not isinstance(target, Mapping):
            continue
        if str(target.get("source_editor_method") or "").strip() != normalized_source:
            continue
        if target_editor_method_for_action(task, action_kind):
            return action_kind
    return None

def _allowed_action_kinds(
    benign_task: Mapping[str, Any] | None,
    exposure_contract: Mapping[str, Any],
) -> set[str]:
    annotated = exposure_contract.get("adversarial_action_options")
    if isinstance(annotated, list):
        allowed: set[str] = set()
        for option in annotated:
            if not isinstance(option, Mapping):
                continue
            kind = str(option.get("kind") or "").strip()
            if not kind:
                continue
            if _unaligned_control_policy_tier(str(option.get("policy") or "")) is not None:
                if kind not in control_action_kinds_from_task(benign_task):
                    continue
            elif not action_kind_compatible_with_task(kind, benign_task):
                continue
            if kind in ACTION_KINDS:
                allowed.add(kind)
                continue
            if not option_marks_host_ready(option):
                continue
            readiness = tier3_action_readiness(
                kind,
                benign_task=benign_task,
                exposure_contract=exposure_contract,
                policy=str(option.get("pilot_policy") or ""),
            )
            if readiness["status"] == "ready":
                allowed.add(kind)
        return allowed
    preference = exposure_contract.get("adversarial_action_preference")
    if isinstance(preference, Mapping) and preference.get("policy") == "semantic_only":
        return set()
    return {option["kind"] for option in allowed_action_options(benign_task, exposure_contract)}

def _filter_options_for_task_capability(
    options: list[dict[str, str]],
    benign_task: Mapping[str, Any] | None,
    *,
    policy: str = "default",
) -> list[dict[str, str]]:
    if benign_task is None:
        return options
    control_tier = _unaligned_control_policy_tier(policy)
    if control_tier is not None:
        control_kinds = set(control_action_kinds_from_task(benign_task))
        return [
            option
            for option in options
            if str(option.get("kind") or "") in control_kinds
            and _action_tier(str(option.get("kind") or "")) == control_tier
        ]
    return [
        option
        for option in options
        if action_kind_compatible_with_task(str(option.get("kind") or ""), benign_task)
    ]
