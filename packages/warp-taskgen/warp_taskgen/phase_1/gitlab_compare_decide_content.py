"""Feature-local parsing of generated GitLab comparison content."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from warp_taskgen.phase_1.gitlab_compare_decide import (
    DEFAULT_DECISION_RULE,
    DEFAULT_RECORD_KEYS,
    GitLabComparisonWorld,
    GitLabIssueRecord,
    generate_gitlab_compare_decide_world,
    select_gitlab_record,
)

GENERATED_COMPARISON_CONTENT_KEY = "generated_comparison"
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
        raise ValueError("GitLab comparison decision rule must declare non-empty state and dependency")
    return rule


def _record_from_generated_mapping(
    value: Mapping[str, Any],
    *,
    key: str,
) -> GitLabIssueRecord:
    facts = value.get("facts")
    fact_map = facts if isinstance(facts, Mapping) else value
    title = _generated_text(value.get("title"), field="title")
    state = _generated_text(
        fact_map.get("state", fact_map.get("status")), field="facts.state"
    )
    dependency = _generated_text(fact_map.get("dependency"), field="facts.dependency")
    summary = _generated_text(
        fact_map.get("summary", fact_map.get("body")), field="facts.summary"
    )
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
            raise ValueError("GitLab comparison generation contract decision_rule must be an object")
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
    "world_for_phase1_task",
]
