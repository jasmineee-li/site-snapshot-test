"""Feature-local parsing of generated GitLab comparison content."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from warp_taskgen.phase_1.gitlab_compare_act import (
    GITLAB_COMPARE_ACT_ACTION_KIND,
    GITLAB_COMPARE_ACT_ARTIFACT_SCOPE,
    GITLAB_COMPARE_ACT_FAMILY,
    GITLAB_COMPARE_ACT_VERSION,
    build_gitlab_compare_act_benign_action_contract,
    generate_gitlab_compare_act_note,
)
from warp_taskgen.phase_1.gitlab_compare_decide import (
    DEFAULT_DECISION_RULE,
    DEFAULT_RECORD_KEYS,
    GitLabComparisonWorld,
    GitLabIssueRecord,
    expected_gitlab_compare_decide_response,
    generate_gitlab_compare_decide_world,
    select_gitlab_record,
)

GENERATED_COMPARISON_CONTENT_KEY = "generated_comparison"
_HOST_ACTION_ONLY_PLACEHOLDER_EVALUATOR = "HostActionOnlyPlaceholder"
_GENERATED_CONTENT_FIELDS = frozenset({"records"})
_FORBIDDEN_RECORD_FIELDS = frozenset(
    {
        "benchmark",
        "issue_iid",
        "logical_record_key",
        "method",
        "physical_id",
        "resource_kind",
        "site",
        "url",
    }
)


@dataclass(frozen=True)
class GitLabComparisonGeneratedContent:
    """Typed WARP-generated facts for one supplied comparison workflow.

    Logical keys are supplied by the workflow contract and are deliberately
    absent from this payload.  The generated input cannot choose the decision
    rule or answer key.  The decision rule is deliberately not part of this
    payload: it belongs to the authored card.
    """

    records: tuple[GitLabIssueRecord, ...]

    def __post_init__(self) -> None:
        records = tuple(self.records)
        if len(records) != 3:
            raise ValueError("generated GitLab comparison content requires exactly three records")
        keys = [record.logical_record_key for record in records]
        if len(set(keys)) != len(keys):
            raise ValueError("generated GitLab comparison content requires unique record keys")
        object.__setattr__(self, "records", records)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        record_keys: Sequence[str],
    ) -> GitLabComparisonGeneratedContent:
        """Validate one model-produced content payload without side effects."""

        if not isinstance(value, Mapping):
            raise ValueError("generated GitLab comparison content must be an object")
        unexpected = set(value) - _GENERATED_CONTENT_FIELDS
        if unexpected:
            raise ValueError(
                "generated comparison includes host-owned or unsupported field(s): "
                + ", ".join(sorted(str(key) for key in unexpected))
            )
        raw_records = value.get("records")
        if not isinstance(raw_records, list):
            raise ValueError("generated GitLab comparison content requires a records array")
        if len(raw_records) != 3:
            raise ValueError("generated GitLab comparison content requires exactly three records")
        keys = _validate_record_keys(record_keys)
        parsed: list[GitLabIssueRecord] = []
        for index, raw_record in enumerate(raw_records):
            if not isinstance(raw_record, Mapping):
                raise ValueError(f"generated comparison records[{index}] must be an object")
            forbidden = _FORBIDDEN_RECORD_FIELDS.intersection(raw_record)
            if forbidden:
                raise ValueError(
                    "generated comparison records may not include host-owned field(s): "
                    + ", ".join(sorted(forbidden))
                )
            parsed.append(_record_from_generated_mapping(raw_record, key=keys[index]))
        return cls(records=tuple(parsed))

    def as_mapping(self) -> dict[str, Any]:
        return {"records": [record.as_mapping() for record in self.records]}


def world_for_phase1_task(
    task: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    method: str,
) -> tuple[GitLabComparisonWorld, str, tuple[str, ...]]:
    """Build a world from generated facts while keeping rule/route host-owned."""

    record_keys = _contract_record_keys(contract)
    decision_rule = _contract_decision_rule(contract)
    generated = _generated_content_from_task(task, record_keys=record_keys)
    if generated is None:
        if record_keys != DEFAULT_RECORD_KEYS:
            raise ValueError(
                "custom GitLab comparison record keys require generated comparison content"
            )
        decisive_key = _contract_text(contract, "decisive_record_key", record_keys[0])
        if decision_rule["state"] != DEFAULT_DECISION_RULE["state"]:
            raise ValueError(
                "custom GitLab comparison decision states require generated comparison content"
            )
        world = generate_gitlab_compare_decide_world(
            decisive_record_key=decisive_key,
            decision_dependency=decision_rule["dependency"],
            method=method,
        )
        return world, "feature_default", record_keys
    candidate_world = GitLabComparisonWorld(
        records=generated.records,
        decision_rule=decision_rule,
        decisive_record_key=generated.records[0].logical_record_key,
        method=method,
    )
    selected = select_gitlab_record(candidate_world)
    world = GitLabComparisonWorld(
        records=generated.records,
        decision_rule=decision_rule,
        decisive_record_key=selected.logical_record_key,
        method=method,
    )
    return world, "warp_generated", record_keys


def _record_from_world_mapping(value: Any) -> GitLabIssueRecord:
    if not isinstance(value, Mapping):
        raise ValueError("GitLab comparison world records must be objects")
    raw_facts = value.get("facts")
    facts = raw_facts if isinstance(raw_facts, Mapping) else value
    return GitLabIssueRecord(
        str(value.get("logical_record_key") or ""),
        str(value.get("title") or ""),
        str(facts.get("state") or facts.get("status") or ""),
        str(facts.get("dependency") or ""),
        str(facts.get("summary") or facts.get("body") or ""),
    )


def _world_from_compiled_task(task: Mapping[str, Any]) -> GitLabComparisonWorld:
    world = task.get("world")
    if not isinstance(world, Mapping):
        raise ValueError("GitLab comparison task must include a world")
    raw_records = world.get("records")
    record_items = raw_records if isinstance(raw_records, list) else []
    return GitLabComparisonWorld(
        records=tuple(_record_from_world_mapping(record) for record in record_items),
        decision_rule=world.get("decision_rule", DEFAULT_DECISION_RULE),
        decisive_record_key=str(world.get("decisive_record_key") or ""),
        benchmark=str(world.get("benchmark") or "webarena_verified"),
        site=str(world.get("site") or "gitlab"),
        method=str(world.get("method") or "create_issue"),
        resource_kind=str(world.get("resource_kind") or "issue"),
    )


def _validate_compare_decide_instruction(
    instruction: Any,
    world: GitLabComparisonWorld,
) -> None:
    _validate_compare_instruction_scope(instruction, world)
    text = instruction.casefold()
    if "selected_iid" not in text or "reason" not in text:
        raise ValueError("GitLab comparison instruction must request selected_iid and a reason")


def _validate_compare_instruction_scope(
    instruction: Any,
    world: GitLabComparisonWorld,
) -> str:
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("GitLab comparison instruction must be non-empty")
    text = instruction.casefold()
    if "issue" not in text or not re.search(r"\b(?:all\s+three|three|each|every|all)\b", text):
        raise ValueError(
            "GitLab comparison instruction must require reviewing all three issue records"
        )
    if not re.search(r"\b(?:select|choose|identify|determine|which)\w*\b", text):
        raise ValueError("GitLab comparison instruction must ask for a rule-based selection")
    for value in world.decision_rule.values():
        if str(value).casefold() not in text:
            raise ValueError("GitLab comparison instruction must state every decision-rule value")
    return text


def _validate_compare_act_instruction(
    instruction: Any,
    world: GitLabComparisonWorld,
) -> None:
    text = _validate_compare_instruction_scope(instruction, world)
    if not re.search(r"\b(?:leave|add|post|write|create)\w*\b", text) or not re.search(
        r"\b(?:note|comment)\w*\b", text
    ):
        raise ValueError("GitLab compare-act instruction must request a public note or comment")
    if not re.search(r"\b(?:selected|matching|chosen|identified)\b", text):
        raise ValueError("GitLab compare-act instruction must target the selected comparison issue")


def _validate_compare_act_model_reward(
    evals: list[Any],
    world: GitLabComparisonWorld,
) -> None:
    """Accept the action-only placeholder and check optional answer metadata."""

    first = evals[0]
    if not isinstance(first, Mapping):
        raise ValueError("GitLab compare-act generation requires an evaluator object")
    if first.get("evaluator") != _HOST_ACTION_ONLY_PLACEHOLDER_EVALUATOR:
        raise ValueError(
            "GitLab compare-act generation requires the host-action-only reward placeholder"
        )
    if first.get("expected") != {"host_compiled": True}:
        raise ValueError(
            "GitLab compare-act generation requires the host-action-only reward placeholder"
        )
    expected_response = expected_gitlab_compare_decide_response(world)
    for config in evals[1:]:
        if not isinstance(config, Mapping):
            raise ValueError("GitLab compare-act generated evaluator entries must be objects")
        if config.get("evaluator") != "AgentResponseEvaluator":
            continue
        if config.get("expected") != expected_response:
            raise ValueError(
                "GitLab comparison generated expected response disagrees with decision rule"
            )


def validate_compiled_comparison_seed(
    task: Mapping[str, Any],
    world: GitLabComparisonWorld,
) -> None:
    """Re-check host-derived seed calls before accepting an idempotent row."""

    seeds = [task.get("data_seed"), task.get("adversarial_data_seed")]
    if not all(isinstance(seed, Mapping) for seed in seeds):
        raise ValueError("GitLab comparison host output requires editor seeds")
    if any(seed.get("mechanism") != "editor" for seed in seeds if isinstance(seed, Mapping)):
        raise ValueError("GitLab comparison host output requires editor seed mechanisms")
    if seeds[0] != seeds[1]:
        raise ValueError("GitLab comparison host output has divergent benign and adversarial seeds")
    calls = seeds[0].get("editor_calls") if isinstance(seeds[0], Mapping) else None
    if not isinstance(calls, list) or len(calls) != len(world.records):
        raise ValueError("GitLab comparison host output must contain one seed call per record")
    for call, record in zip(calls, world.records, strict=True):
        if not isinstance(call, Mapping):
            raise ValueError("GitLab comparison host output seed calls must be objects")
        if (
            call.get("benchmark") != world.benchmark
            or call.get("site") != world.site
            or call.get("method") != world.method
            or call.get("resource_kind") != world.resource_kind
        ):
            raise ValueError(
                "GitLab comparison host output seed call contract disagrees with world"
            )
        if call.get("logical_record_key") != record.logical_record_key:
            raise ValueError("GitLab comparison host output seed key order disagrees with world")
        args = call.get("args")
        if not isinstance(args, Mapping):
            raise ValueError("GitLab comparison host output seed call has invalid arguments")
        expected_body = _record_body(record)
        if world.method == "create_issue_description":
            title_key, body_key = "title", "body"
        else:
            title_key, body_key = "title_template", "body_template"
        if args.get(title_key) != record.title or args.get(body_key) != expected_body:
            raise ValueError("GitLab comparison host output seed content disagrees with world")


def _record_body(record: GitLabIssueRecord) -> str:
    facts = record.facts
    return (
        f"Logical record: {record.logical_record_key}\n"
        f"State: {facts['state']}\n"
        f"Dependency: {facts['dependency']}\n"
        f"Summary: {facts['summary']}"
    )


def validate_compiled_comparison_act(
    task: Mapping[str, Any],
    world: GitLabComparisonWorld,
    *,
    expected_task_card_id: str | None = None,
) -> None:
    """Re-check the host-owned action contract before an idempotent return."""

    action = task.get("comparison_act_contract")
    selected_key = select_gitlab_record(world).logical_record_key
    note_body = generate_gitlab_compare_act_note(world)
    expected_action = {
        "family": GITLAB_COMPARE_ACT_FAMILY,
        "version": GITLAB_COMPARE_ACT_VERSION,
        "action_kind": GITLAB_COMPARE_ACT_ACTION_KIND,
        "target_binding": "selected_record",
        "target_logical_record_key": selected_key,
        "note_body": note_body,
        "artifact_scope": GITLAB_COMPARE_ACT_ARTIFACT_SCOPE,
        "actor_source": "current_authenticated_user",
    }
    if not isinstance(action, Mapping) or dict(action) != expected_action:
        raise ValueError("GitLab compare-act host output action contract disagrees with world")
    from warp_taskgen.phase_1.gitlab_compare_act_reward import build_gitlab_compare_act_reward

    expected_reward = build_gitlab_compare_act_reward(task)
    if task.get("reward_function") != expected_reward:
        raise ValueError("GitLab compare-act host output reward disagrees with action contract")
    provenance = task.get("task_provenance")
    action_provenance = (
        provenance.get("benign_action_contract") if isinstance(provenance, Mapping) else None
    )
    expected_provenance = build_gitlab_compare_act_benign_action_contract(task)
    if action_provenance is not None:
        if action_provenance != expected_provenance:
            raise ValueError("GitLab compare-act host output provenance disagrees with reward")
    elif (
        expected_task_card_id is None
        or not isinstance(provenance, Mapping)
        or provenance.get("task_card_id") != expected_task_card_id
    ):
        raise ValueError("GitLab compare-act host output provenance disagrees with reward")


def _validate_record_keys(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError("GitLab comparison generation contract requires exactly three record keys")
    keys: list[str] = []
    for index, raw_key in enumerate(value):
        if (
            not isinstance(raw_key, str)
            or raw_key != raw_key.strip()
            or re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", raw_key.strip()) is None
        ):
            raise ValueError(
                f"GitLab comparison generation contract record_keys[{index}] must be a safe identifier"
            )
        keys.append(raw_key.strip())
    if len(set(keys)) != 3:
        raise ValueError("GitLab comparison generation contract record_keys must be unique")
    return tuple(keys)


def _normalize_decision_rule(value: Mapping[str, Any]) -> dict[str, str]:
    rule = {str(key): str(raw).strip() for key, raw in value.items()}
    if set(rule) != set(DEFAULT_DECISION_RULE) or any(not raw for raw in rule.values()):
        raise ValueError(
            "GitLab comparison decision rule must declare non-empty state and dependency"
        )
    return rule


def _record_from_generated_mapping(
    value: Mapping[str, Any],
    *,
    key: str,
) -> GitLabIssueRecord:
    facts = value.get("facts")
    fact_map = facts if isinstance(facts, Mapping) else value
    title = _generated_text(value.get("title"), field="title")
    state = _generated_text(fact_map.get("state", fact_map.get("status")), field="facts.state")
    dependency = _generated_text(fact_map.get("dependency"), field="facts.dependency")
    summary = _generated_text(fact_map.get("summary", fact_map.get("body")), field="facts.summary")
    return GitLabIssueRecord(key, title, state, dependency, summary)


def _generated_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"generated comparison {field} must be a non-empty string")
    return value.strip()


def _contract_record_keys(contract: Mapping[str, Any]) -> tuple[str, ...]:
    raw_keys = contract.get("record_keys", DEFAULT_RECORD_KEYS)
    if not isinstance(raw_keys, Sequence) or isinstance(raw_keys, (str, bytes)):
        raise ValueError("GitLab comparison generation contract record_keys must be an array")
    return _validate_record_keys(raw_keys)


def _contract_decision_rule(contract: Mapping[str, Any]) -> dict[str, str]:
    raw_rule = contract.get("decision_rule")
    if raw_rule is not None:
        if not isinstance(raw_rule, Mapping):
            raise ValueError(
                "GitLab comparison generation contract decision_rule must be an object"
            )
        rule = _normalize_decision_rule(raw_rule)
        legacy_dependency = contract.get("decision_dependency")
        if legacy_dependency is not None and str(legacy_dependency).strip() != rule["dependency"]:
            raise ValueError(
                "GitLab comparison generation contract decision_dependency disagrees with decision_rule"
            )
        return rule
    return _normalize_decision_rule(
        {
            "state": contract.get("decision_state", DEFAULT_DECISION_RULE["state"]),
            "dependency": contract.get("decision_dependency", DEFAULT_DECISION_RULE["dependency"]),
        }
    )


def _generated_content_from_task(
    task: Mapping[str, Any],
    *,
    record_keys: Sequence[str],
) -> GitLabComparisonGeneratedContent | None:
    raw = task.get(GENERATED_COMPARISON_CONTENT_KEY)
    if raw is None:
        return None
    return GitLabComparisonGeneratedContent.from_mapping(raw, record_keys=record_keys)


def _contract_text(contract: Mapping[str, Any], key: str, default: str) -> str:
    value = contract.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"GitLab comparison generation contract {key!r} must be non-empty")
    return value.strip()


__all__ = [
    "GENERATED_COMPARISON_CONTENT_KEY",
    "GitLabComparisonGeneratedContent",
    "validate_compiled_comparison_act",
    "validate_compiled_comparison_seed",
    "world_for_phase1_task",
]
