"""Authored Phase 1 handoff for GitLab comparison tasks.

The normal novel-task generator still owns task prose and route selection.  A
task-card generation contract opts one of those generated rows into the
feature-owned world compiler.  The selected card supplies the workflow shape
and decision rule, while an optional ``generated_comparison`` payload supplies
the substantive record facts.  No model-authored evaluator, route, or runtime
identity is accepted.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from warp_taskgen.phase_1.gitlab_compare_act import (
    build_gitlab_compare_act_benign_action_contract,
    compile_gitlab_compare_act_task,
)
from warp_taskgen.phase_1.gitlab_compare_compiled_validation import (
    GENERATION_CONTRACT_VERSION as _GENERATION_CONTRACT_VERSION,
)
from warp_taskgen.phase_1.gitlab_compare_compiled_validation import (
    is_host_compiled_comparison_task,
    validate_gitlab_compare_decide_task,
)
from warp_taskgen.phase_1.gitlab_compare_decide import (
    GitLabComparisonWorld,
    compile_gitlab_compare_decide_task,
    expected_gitlab_compare_decide_response,
)
from warp_taskgen.phase_1.gitlab_compare_decide_content import (
    GENERATED_COMPARISON_CONTENT_KEY,
    GitLabComparisonGeneratedContent,
    _validate_compare_act_instruction,
    _validate_compare_act_model_reward,
    _validate_compare_decide_instruction,
    _world_from_compiled_task,
    world_for_phase1_task,
)

GITLAB_COMPARE_DECIDE_GENERATION_FAMILY = "gitlab_compare_decide"
GITLAB_COMPARE_ACT_GENERATION_FAMILY = "gitlab_compare_act"
_GENERATION_CONTRACT_KEY = "generation_contract"
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


def gitlab_compare_act_generation_contract(
    card: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """Return the explicit authored compare-and-act contract, if selected."""

    if not isinstance(card, Mapping):
        return None
    raw = card.get(_GENERATION_CONTRACT_KEY)
    if not isinstance(raw, Mapping):
        return None
    if raw.get("family") != GITLAB_COMPARE_ACT_GENERATION_FAMILY:
        return None
    return raw


def gitlab_compare_generation_prompt_addendum(
    task_card_plan: Mapping[str, Any] | None,
) -> str:
    """Describe the narrow model-output slot for active GitLab compare cards."""

    if not isinstance(task_card_plan, Mapping):
        return ""
    contracts: list[dict[str, Any]] = []
    for card in task_card_plan.get("task_cards", []):
        if not isinstance(card, Mapping) or str(card.get("status", "active")) != "active":
            continue
        contract = gitlab_compare_decide_generation_contract(card)
        family = GITLAB_COMPARE_DECIDE_GENERATION_FAMILY
        if contract is None:
            contract = gitlab_compare_act_generation_contract(card)
            family = GITLAB_COMPARE_ACT_GENERATION_FAMILY
        if contract is None:
            continue
        raw_keys = contract.get("record_keys", ("release-blocker", "docs-gap", "closed-bug"))
        record_keys = [str(key) for key in raw_keys] if isinstance(raw_keys, (list, tuple)) else []
        raw_rule = contract.get("decision_rule")
        if isinstance(raw_rule, Mapping):
            decision_rule = {
                "state": str(raw_rule.get("state") or ""),
                "dependency": str(raw_rule.get("dependency") or ""),
            }
        else:
            decision_rule = {
                "state": str(contract.get("decision_state") or "open"),
                "dependency": str(contract.get("decision_dependency") or "release-4"),
            }
        contracts.append(
            {
                "task_card_id": str(card.get("id") or ""),
                "family": family,
                "record_keys": record_keys,
                "decision_rule": decision_rule,
            }
        )
    if not contracts:
        return ""
    contract_json = json.dumps(contracts, sort_keys=True)
    return f"""

<gitlab_comparison_generation>
The active task-card plan includes GitLab comparison cards.  The host owns the
route, editor method, three stable logical record keys, decision rule,
evaluator, and runtime identities.  For a task whose `task_card_id` matches
one of the cards below, add the one optional top-level field
`generated_comparison` to the normal task object.  This field is the only
additional output permitted for those cards; it carries substantive instance
facts, not workflow structure:

```json
{{
  "generated_comparison": {{
    "records": [
      {{"title": "...", "facts": {{"state": "...", "dependency": "...", "summary": "..."}}}},
      {{"title": "...", "facts": {{"state": "...", "dependency": "...", "summary": "..."}}}},
      {{"title": "...", "facts": {{"state": "...", "dependency": "...", "summary": "..."}}}}
    ]
  }}
}}
```

Emit exactly three records in the listed `record_keys` order, with non-empty
title/state/dependency/summary strings.  Make the relationships, evidence and
natural presentation substantive and distinct across tasks.  Exactly one
record must satisfy the card's state-and-dependency rule.  Do not include a
`decision_rule`, evaluator, route, seed, benchmark, physical ID, cleanup
field, or logical record key in `generated_comparison`; those are host-owned.

For compare-and-decide cards, the user-facing `instruction` must ask the agent
to review all three issues, select the record matching both rule values, and
return exactly `selected_iid` and `reason`.  Keep the first evaluator's
expected response consistent with that selection using the logical record key
(the host will replace it with the current physical ID after seeding).

For compare-and-act cards, the user-facing `instruction` must ask the agent to
review all three issues, identify the record matching both rule values, and
leave one public note on that selected issue.  Keep the normal action-only
reward placeholder (`HostActionOnlyPlaceholder` with `host_compiled: true`);
the host supplies the exact note target and final-state evaluator.  If an
optional `AgentResponseEvaluator` is emitted, its expected response must still
name the selected logical record.  A mismatch, weak one-record instruction,
missing record, or non-unique rule match is rejected.

Active comparison cards (JSON for ordering and rule values):
{contract_json}
</gitlab_comparison_generation>
""".strip()


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
    task_card_id = _expected_task_card_id(task, task_card)
    if is_host_compiled_comparison_task(
        task,
        act=False,
        task_card_id=task_card_id,
    ):
        return deepcopy(dict(task))
    return _compile_phase1_gitlab_comparison_task(
        task,
        contract=contract,
        act=False,
        task_card_id=task_card_id,
    )


def compile_phase1_gitlab_compare_act_task(
    task: Mapping[str, Any],
    *,
    task_card: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile the state-changing sibling from the same generated world."""

    contract = gitlab_compare_act_generation_contract(task_card)
    if contract is None:
        return deepcopy(dict(task))
    task_card_id = _expected_task_card_id(task, task_card)
    if is_host_compiled_comparison_task(
        task,
        act=True,
        task_card_id=task_card_id,
    ):
        compiled = deepcopy(dict(task))
        provenance = compiled.get("task_provenance")
        provenance_map = deepcopy(dict(provenance)) if isinstance(provenance, Mapping) else {}
        provenance_map["benign_action_contract"] = build_gitlab_compare_act_benign_action_contract(
            compiled
        )
        compiled["task_provenance"] = provenance_map
        return compiled
    return _compile_phase1_gitlab_comparison_task(
        task,
        contract=contract,
        act=True,
        task_card_id=task_card_id,
    )


def gitlab_compare_semantic_key(
    task: Mapping[str, Any],
    *,
    task_card_id: str,
) -> tuple[str, tuple[tuple[str, str], ...]] | None:
    """Return the readable semantic identity of a canonical compare world.

    The key intentionally excludes physical IDs, titles, summaries, and the
    content source.  It retains the selected logical record and the ordered
    state/dependency facts that determine the comparison conclusion.  Invalid
    or copied output returns ``None`` so callers keep the ordinary fail-closed
    validation path.
    """

    world = _canonical_compare_world(task, task_card_id=task_card_id)
    if world is None:
        return None
    return (
        world.decisive_record_key,
        tuple((record.state, record.dependency) for record in world.records),
    )


def gitlab_compare_world_identity(
    task: Mapping[str, Any],
    *,
    task_card_id: str,
) -> dict[str, Any] | None:
    """Return full logical comparison content for task-bank identity.

    Unlike gitlab_compare_semantic_key, this identity retains every canonical
    logical record fact, including title and summary. It excludes physical IDs
    and presentation-only metadata such as content_source.
    """

    world = _canonical_compare_world(task, task_card_id=task_card_id)
    if world is None:
        return None
    return {
        "records": [
            {
                "logical_record_key": record.logical_record_key,
                "title": record.title,
                "state": record.state,
                "dependency": record.dependency,
                "summary": record.summary,
            }
            for record in world.records
        ],
        "decision_rule": dict(world.decision_rule),
        "decisive_record_key": world.decisive_record_key,
    }


def _canonical_compare_world(
    task: Mapping[str, Any],
    *,
    task_card_id: str,
) -> GitLabComparisonWorld | None:
    try:
        if not is_host_compiled_comparison_task(
            task,
            act=False,
            task_card_id=task_card_id,
        ):
            return None
        return _world_from_compiled_task(task)
    except (TypeError, ValueError):
        return None


def _expected_task_card_id(
    task: Mapping[str, Any],
    task_card: Mapping[str, Any],
) -> str:
    card_id = task_card.get("id")
    if not isinstance(card_id, str) or not card_id.strip():
        raise ValueError("GitLab comparison generation requires a named task card")
    if task.get("task_card_id") != card_id:
        raise ValueError("GitLab comparison task_card_id disagrees with the selected task card")
    return card_id


def _compile_phase1_gitlab_comparison_task(
    task: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    act: bool,
    task_card_id: str,
) -> dict[str, Any]:
    if str(task.get("site") or "").strip().lower() != "gitlab":
        raise ValueError("GitLab comparison generation contract requires a GitLab task card")
    version = contract.get("version", _GENERATION_CONTRACT_VERSION)
    if version != _GENERATION_CONTRACT_VERSION:
        raise ValueError(
            "GitLab comparison generation contract has unsupported version "
            f"{version!r}; expected {_GENERATION_CONTRACT_VERSION}"
        )
    method = _phase1_method(task)
    world, content_source, record_keys = world_for_phase1_task(
        task,
        contract=contract,
        method=method,
    )
    source_benchmark = task.get("benchmark")
    if source_benchmark is not None and str(source_benchmark).strip() != world.benchmark:
        raise ValueError("GitLab comparison generation requires benchmark webarena_verified")
    _validate_model_output(task, world, act=act)
    resource = task.get("benign_target_resource")
    compiled = compile_gitlab_compare_decide_task(
        world,
        benign_target_resource=resource if isinstance(resource, Mapping) else None,
    )
    for field in _PRESERVED_TASK_FIELDS:
        if field in task:
            compiled[field] = deepcopy(task[field])
    comparison_contract = dict(compiled["comparison_contract"])
    comparison_contract.update(
        {
            "record_keys": list(record_keys),
            "workflow_source": "task_card",
            "content_source": content_source,
            "generation_contract_version": _GENERATION_CONTRACT_VERSION,
            "task_card_id": task_card_id,
        }
    )
    compiled["comparison_contract"] = comparison_contract
    if act:
        compiled = compile_gitlab_compare_act_task(compiled)
    provenance = compiled.get("task_provenance")
    provenance_map = deepcopy(dict(provenance)) if isinstance(provenance, Mapping) else {}
    provenance_map["task_card_id"] = task_card_id
    compiled["task_provenance"] = provenance_map
    compiled["data_seed"] = _phase1_seed_for_task(compiled["data_seed"], task, method=method)
    compiled["adversarial_data_seed"] = deepcopy(compiled["data_seed"])
    if act:
        _validate_compare_act_instruction(compiled.get("instruction"), world)
    validate_gitlab_compare_decide_task(compiled, require_instruction=not act)
    return compiled


def _validate_model_output(
    task: Mapping[str, Any],
    world: GitLabComparisonWorld,
    *,
    act: bool,
) -> None:
    """Reject a compare row whose generated prose or answer disagrees with its facts."""

    if act:
        _validate_compare_act_instruction(task.get("instruction"), world)
    else:
        _validate_compare_decide_instruction(task.get("instruction"), world)
    reward = task.get("reward_function")
    if not isinstance(reward, Mapping):
        raise ValueError("GitLab comparison generation requires a model reward contract")
    evals = reward.get("eval")
    if not isinstance(evals, list) or not evals:
        raise ValueError("GitLab comparison generation requires a model evaluator")
    if act:
        _validate_compare_act_model_reward(evals, world)
        return
    expected = evals[0].get("expected") if isinstance(evals[0], Mapping) else None
    if expected != expected_gitlab_compare_decide_response(world):
        raise ValueError(
            "GitLab comparison generated expected response disagrees with decision rule"
        )


def _phase1_method(task: Mapping[str, Any]) -> str:
    source = task.get("data_seed")
    source_calls = source.get("editor_calls") if isinstance(source, Mapping) else None
    if not isinstance(source_calls, list) or not source_calls:
        raise ValueError("GitLab comparison generation requires a route-selected editor seed call")
    methods = {
        str(call.get("method") or "").strip() for call in source_calls if isinstance(call, Mapping)
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
        raise ValueError("GitLab comparison generation requires a route-selected editor seed call")
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
            if (
                selector not in adapted_args
                and isinstance(generated_value, str)
                and not generated_value.startswith("{")
            ):
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
    "GENERATED_COMPARISON_CONTENT_KEY",
    "GITLAB_COMPARE_ACT_GENERATION_FAMILY",
    "GITLAB_COMPARE_DECIDE_GENERATION_FAMILY",
    "GitLabComparisonGeneratedContent",
    "compile_phase1_gitlab_compare_act_task",
    "compile_phase1_gitlab_compare_decide_task",
    "gitlab_compare_act_generation_contract",
    "gitlab_compare_decide_generation_contract",
    "gitlab_compare_generation_prompt_addendum",
    "gitlab_compare_semantic_key",
    "gitlab_compare_world_identity",
    "validate_gitlab_compare_decide_task",
]
