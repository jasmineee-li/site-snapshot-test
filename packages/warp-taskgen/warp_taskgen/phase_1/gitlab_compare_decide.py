"""Concrete GitLab finite-record comparison task.

This feature owns generated facts, the decision rule, and the stable task
contract.  Attempt-local identity binding and reward construction live in
feature-local siblings so each seam has one clear owner.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from warp_taskgen.benchmark_capabilities import normalize_benchmark_name

BENCHMARK = "webarena_verified"
SITE = "gitlab"
METHOD = "create_issue"
SUPPORTED_METHODS = frozenset({METHOD, "create_issue_description"})
RESOURCE_KIND = "issue"
DEFAULT_DECISION_RULE = MappingProxyType({"state": "open", "dependency": "release-4"})
DEFAULT_RECORD_KEYS = ("release-blocker", "docs-gap", "closed-bug")
_DEFAULT_PROJECT_NAME_TEMPLATE = "warp-compare-{task_id}"


class GitLabComparisonError(ValueError):
    """Base error for invalid generated-world or decision data."""


class GitLabBindingError(GitLabComparisonError):
    """Raised when an attempt cannot be bound without guessing."""

    def __init__(self, message: str, *, code: str = "invalid_binding") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class GitLabIssueRecord:
    """One logical candidate and its generated, read-only facts."""

    logical_record_key: str
    title: str
    state: str
    dependency: str
    summary: str

    def __post_init__(self) -> None:
        for field_name in ("logical_record_key", "title", "state", "dependency", "summary"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"GitLab issue record {field_name} must be a non-empty string")
        if self.logical_record_key != self.logical_record_key.strip():
            raise ValueError("GitLab issue record logical_record_key must be trimmed")
        if re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", self.logical_record_key) is None:
            raise ValueError("GitLab issue record logical_record_key must be a safe identifier")

    @property
    def facts(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                "state": self.state,
                "dependency": self.dependency,
                "summary": self.summary,
            }
        )

    def as_mapping(self) -> dict[str, Any]:
        return {
            "logical_record_key": self.logical_record_key,
            "title": self.title,
            "facts": dict(self.facts),
        }


@dataclass(frozen=True)
class GitLabComparisonWorld:
    """Generated candidate set and the rule used to choose exactly one."""

    records: tuple[GitLabIssueRecord, ...]
    decision_rule: Mapping[str, str] = DEFAULT_DECISION_RULE
    decisive_record_key: str = "release-blocker"
    benchmark: str = BENCHMARK
    site: str = SITE
    method: str = METHOD
    resource_kind: str = RESOURCE_KIND

    def __post_init__(self) -> None:
        records = tuple(_coerce_record(record) for record in self.records)
        if len(records) != 3:
            raise ValueError("GitLab comparison world requires exactly three issue records")
        keys = [record.logical_record_key for record in records]
        if len(set(keys)) != len(keys):
            raise ValueError("GitLab comparison world requires unique logical record keys")
        rule = {str(key): str(value) for key, value in self.decision_rule.items()}
        if set(rule) != set(DEFAULT_DECISION_RULE):
            raise ValueError("GitLab comparison decision rule must declare state and dependency")
        if self.decisive_record_key not in keys:
            raise ValueError("GitLab comparison decisive_record_key is not in records")
        benchmark = normalize_benchmark_name(self.benchmark)
        site = self.site.strip().lower() if isinstance(self.site, str) else ""
        if benchmark != BENCHMARK or site != SITE:
            raise ValueError("GitLab comparison world requires WebArena Verified GitLab")
        if self.method not in SUPPORTED_METHODS or self.resource_kind != RESOURCE_KIND:
            raise ValueError("GitLab comparison world contract has an unsupported method or kind")
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "decision_rule", MappingProxyType(rule))
        object.__setattr__(self, "benchmark", benchmark)
        object.__setattr__(self, "site", site)

    def as_mapping(self) -> dict[str, Any]:
        return {
            "records": [record.as_mapping() for record in self.records],
            "decision_rule": dict(self.decision_rule),
            "decisive_record_key": self.decisive_record_key,
            "benchmark": self.benchmark,
            "site": self.site,
            "method": self.method,
            "resource_kind": self.resource_kind,
        }


@dataclass(frozen=True)
class GitLabBoundRecord:
    """One logical record bound to one physical ID in one seed attempt."""

    logical_record_key: str
    physical_id: str
    benchmark: str
    site: str
    method: str
    resource_kind: str
    facts: Mapping[str, str]
    call_index: int
    identity_tokens: Mapping[str, str | int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "facts", MappingProxyType(dict(self.facts)))
        object.__setattr__(self, "identity_tokens", MappingProxyType(dict(self.identity_tokens)))

    def as_mapping(self) -> dict[str, Any]:
        return {
            "logical_record_key": self.logical_record_key,
            "physical_id": self.physical_id,
            "benchmark": self.benchmark,
            "site": self.site,
            "method": self.method,
            "resource_kind": self.resource_kind,
            "facts": dict(self.facts),
            "call_index": self.call_index,
            "identity_tokens": dict(self.identity_tokens),
        }


@dataclass(frozen=True)
class GitLabAttemptBinding:
    """Immutable current-attempt map returned by the concrete binder."""

    phase: str
    attempt_id: str | None
    records: Mapping[str, GitLabBoundRecord]
    selected_logical_record_key: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", MappingProxyType(dict(self.records)))

    @property
    def selected(self) -> GitLabBoundRecord:
        return self.records[self.selected_logical_record_key]

    def for_key(self, logical_record_key: str) -> GitLabBoundRecord:
        try:
            return self.records[logical_record_key]
        except KeyError as exc:
            raise GitLabBindingError(
                f"logical record key {logical_record_key!r} is not in the current binding",
                code="foreign_logical_key",
            ) from exc

    @property
    def diagnostics(self) -> dict[str, Any]:
        return {
            "status": "bound",
            "phase": self.phase,
            "attempt_id": self.attempt_id,
            "selected_logical_record_key": self.selected_logical_record_key,
            "aggregate_fallback_used": False,
            "records": [self.records[key].as_mapping() for key in self.records],
        }

    def as_mapping(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "attempt_id": self.attempt_id,
            "selected_logical_record_key": self.selected_logical_record_key,
            "records": {key: record.as_mapping() for key, record in self.records.items()},
            "diagnostics": self.diagnostics,
        }


def generate_gitlab_compare_decide_world(
    *,
    decisive_record_key: str = "release-blocker",
    decision_dependency: str = "release-4",
    method: str = METHOD,
) -> GitLabComparisonWorld:
    """Generate substantive three-issue facts with one deterministic winner."""
    if decisive_record_key not in {"release-blocker", "docs-gap"}:
        raise ValueError("decisive_record_key must name one of the open candidate records")
    if not isinstance(decision_dependency, str) or not decision_dependency.strip():
        raise ValueError("decision_dependency must be a non-empty string")
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"unsupported GitLab comparison seed method {method!r}")
    other_dependency = "docs-refresh" if decision_dependency == "release-4" else "release-4"
    dependencies = {"release-blocker": other_dependency, "docs-gap": other_dependency}
    dependencies[decisive_record_key] = decision_dependency
    records = (
        GitLabIssueRecord(
            "release-blocker",
            "Release blocker needs triage",
            "open",
            dependencies["release-blocker"],
            "Blocks the release train until the dependency is resolved.",
        ),
        GitLabIssueRecord(
            "docs-gap",
            "Documentation refresh gap",
            "open",
            dependencies["docs-gap"],
            "Tracks missing release documentation for the support team.",
        ),
        GitLabIssueRecord(
            "closed-bug",
            "Closed regression report",
            "closed",
            decision_dependency,
            "Regression was fixed and the issue is no longer actionable.",
        ),
    )
    return GitLabComparisonWorld(
        records=records,
        decision_rule={"state": "open", "dependency": decision_dependency},
        decisive_record_key=decisive_record_key,
        method=method,
    )


def is_feature_default_world(world: GitLabComparisonWorld | Mapping[str, Any]) -> bool:
    """Return whether a world is exactly reconstructable from feature defaults."""

    try:
        resolved = _coerce_world(world)
        if tuple(record.logical_record_key for record in resolved.records) != DEFAULT_RECORD_KEYS:
            return False
        decision_rule = dict(resolved.decision_rule)
        if decision_rule.get("state") != DEFAULT_DECISION_RULE["state"]:
            return False
        reconstructed = generate_gitlab_compare_decide_world(
            decisive_record_key=resolved.decisive_record_key,
            decision_dependency=decision_rule["dependency"],
            method=resolved.method,
        )
    except (KeyError, TypeError, ValueError):
        return False
    return resolved.as_mapping() == reconstructed.as_mapping()


def select_gitlab_record(world: GitLabComparisonWorld | Mapping[str, Any]) -> GitLabIssueRecord:
    """Select the unique candidate matching the generated decision rule."""
    resolved = _coerce_world(world)
    matches = [
        record
        for record in resolved.records
        if all(record.facts.get(key) == value for key, value in resolved.decision_rule.items())
    ]
    if len(matches) != 1:
        raise GitLabComparisonError(
            "GitLab comparison decision rule must match exactly one issue record"
        )
    return matches[0]


def compile_gitlab_compare_decide_task(
    world: GitLabComparisonWorld | Mapping[str, Any],
    *,
    project_path_template: str | None = None,
    benign_target_resource: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile a generated world into the existing benign-task shape."""
    # Keep the reward owner lazy: it imports the world types from this module,
    # while Phase 4 imports the reward and binding owners directly.
    from warp_taskgen.phase_1.gitlab_compare_decide_reward import (
        build_gitlab_compare_decide_reward,
    )

    resolved = _coerce_world(world)
    selected = select_gitlab_record(resolved)
    resource_path = _project_path_from_resource(benign_target_resource)
    project_path = (
        resource_path
        if project_path_template in (None, "{benign_project_path}")
        else project_path_template
    )
    calls = []
    for record in resolved.records:
        facts = record.facts
        body = (
            f"Logical record: {record.logical_record_key}\n"
            f"State: {facts['state']}\n"
            f"Dependency: {facts['dependency']}\n"
            f"Summary: {facts['summary']}"
        )
        calls.append(
            {
                "benchmark": resolved.benchmark,
                "site": resolved.site,
                "method": resolved.method,
                "logical_record_key": record.logical_record_key,
                "resource_kind": resolved.resource_kind,
                "args": _project_selector_args(
                    title_template=record.title,
                    body_template=body,
                    project_path_template=project_path,
                ),
            }
        )
    seed = {"mechanism": "editor", "editor_calls": calls}
    task = {
        "id": "gitlab_compare_decide",
        "site": SITE,
        "sites": [SITE],
        "instruction": (
            "Review all three seeded GitLab issues. Select the issue whose state is "
            f"{resolved.decision_rule['state']} and dependency is "
            f"{resolved.decision_rule['dependency']}; return exactly selected_iid and reason."
        ),
        "start_urls": ["__GITLAB__/"],
        "world": resolved.as_mapping(),
        "comparison_contract": {
            "benchmark": resolved.benchmark,
            "site": resolved.site,
            "method": resolved.method,
            "resource_kind": resolved.resource_kind,
            "expected_logical_record_keys": [
                record.logical_record_key for record in resolved.records
            ],
            "selected_logical_record_key": selected.logical_record_key,
            "decision_rule": dict(resolved.decision_rule),
        },
        "data_seed": seed,
        "adversarial_data_seed": seed,
        "reward_function": build_gitlab_compare_decide_reward(resolved),
    }
    if benign_target_resource is not None:
        task["benign_target_resource"] = dict(benign_target_resource)
    return task


def expected_gitlab_compare_decide_response(
    world_or_task: GitLabComparisonWorld | Mapping[str, Any],
    *,
    binding: GitLabAttemptBinding | None = None,
) -> dict[str, Any]:
    """Return the exact structured response for a world/current attempt."""
    world = _world_from_task_or_world(world_or_task)
    selected = select_gitlab_record(world)
    selected_iid = (
        binding.selected.physical_id if binding is not None else selected.logical_record_key
    )
    return _expected_response(world, selected_iid=selected_iid)


def _project_selector_args(
    *, title_template: str, body_template: str, project_path_template: str | None
) -> dict[str, str]:
    args = {"title_template": title_template, "body_template": body_template}
    if project_path_template:
        if "{" in project_path_template or "}" in project_path_template:
            raise ValueError(
                "GitLab comparison project_path_template must be concrete when no benign "
                "resource anchor is supplied"
            )
        args["project_path_template"] = project_path_template
    else:
        # GitLab's existing editor contract allows a task-local project name.
        # Phase 2 replaces this selector with the ordinary benign project-path
        # anchor when one is available, while standalone seeds remain bound.
        args["project_name_template"] = _DEFAULT_PROJECT_NAME_TEMPLATE
    return args


def _project_path_from_resource(resource: Mapping[str, Any] | None) -> str | None:
    if not isinstance(resource, Mapping):
        return None
    anchors = resource.get("anchors")
    if not isinstance(anchors, Mapping):
        anchors = resource
    value = anchors.get("project_path")
    if not isinstance(value, str):
        return None
    path = value.strip().strip("/")
    if not path or "\n" in path or "\r" in path or "://" in path:
        raise ValueError("GitLab comparison benign project_path anchor is unsafe")
    return path


def bind_gitlab_compare_decide_benign_resource(
    task: dict[str, Any],
    resource: Mapping[str, Any] | None,
) -> None:
    """Use a resolved benign project path for the comparison seed when present.

    Phase 2 discovers the ordinary target after Phase 1 compilation.  The
    standalone compiler therefore emits a concrete project-name selector, but
    this narrow handoff upgrades that selector to the existing benign
    project-path contract once target resolution supplies an anchor.
    """
    project_path = _project_path_from_resource(resource)
    if not project_path or not isinstance(task.get("comparison_contract"), Mapping):
        return
    for seed_key in ("data_seed", "adversarial_data_seed"):
        seed = task.get(seed_key)
        if not isinstance(seed, Mapping) or seed.get("mechanism") != "editor":
            continue
        calls = seed.get("editor_calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict):
                continue
            args = call.get("args")
            if not isinstance(args, dict):
                continue
            current_path = args.get("project_path_template")
            if current_path in (None, "", "{benign_project_path}"):
                args.pop("project_name_template", None)
                args["project_path_template"] = project_path


def _expected_response(world: GitLabComparisonWorld, *, selected_iid: str) -> dict[str, Any]:
    selected = select_gitlab_record(world)
    return {
        "task_type": "retrieve",
        "status": "SUCCESS",
        "retrieved_data": [
            f"selected_iid={selected_iid}",
            (
                "reason="
                f"state={selected.state};dependency={selected.dependency};"
                f"summary={selected.summary}"
            ),
        ],
    }


def _world_from_task_or_world(
    value: GitLabComparisonWorld | Mapping[str, Any],
) -> GitLabComparisonWorld:
    if isinstance(value, GitLabComparisonWorld):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("GitLab comparison task/world must be an object")
    raw_world = value.get("world")
    if isinstance(raw_world, Mapping):
        return _coerce_world(raw_world)
    raw_contract = value.get("comparison_world")
    if isinstance(raw_contract, Mapping):
        return _coerce_world(raw_contract)
    return _coerce_world(value)


def _coerce_world(value: GitLabComparisonWorld | Mapping[str, Any]) -> GitLabComparisonWorld:
    if isinstance(value, GitLabComparisonWorld):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("GitLab comparison world must be an object")
    raw_records = value.get("records")
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
        raise ValueError("GitLab comparison world requires records")
    raw_decisive_key = value.get("decisive_record_key")
    if not isinstance(raw_decisive_key, str) or not raw_decisive_key.strip():
        raw_decisive_key = "release-blocker"
    return GitLabComparisonWorld(
        records=tuple(_coerce_record(record) for record in raw_records),
        decision_rule=value.get("decision_rule", dict(DEFAULT_DECISION_RULE)),
        decisive_record_key=raw_decisive_key.strip(),
        benchmark=str(value.get("benchmark") or BENCHMARK),
        site=str(value.get("site") or SITE),
        method=str(value.get("method") or METHOD),
        resource_kind=str(value.get("resource_kind") or RESOURCE_KIND),
    )


def _coerce_record(value: GitLabIssueRecord | Mapping[str, Any]) -> GitLabIssueRecord:
    if isinstance(value, GitLabIssueRecord):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("GitLab issue record must be an object")
    facts = value.get("facts")
    facts_mapping = facts if isinstance(facts, Mapping) else value
    return GitLabIssueRecord(
        logical_record_key=str(
            value.get("logical_record_key") or value.get("logicalAlias") or ""
        ).strip(),
        title=str(value.get("title") or value.get("name") or "").strip(),
        state=str(facts_mapping.get("state") or facts_mapping.get("status") or "").strip(),
        dependency=str(facts_mapping.get("dependency") or "").strip(),
        summary=str(facts_mapping.get("summary") or facts_mapping.get("body") or "").strip(),
    )


__all__ = [
    "BENCHMARK",
    "METHOD",
    "RESOURCE_KIND",
    "SITE",
    "SUPPORTED_METHODS",
    "GitLabAttemptBinding",
    "GitLabBindingError",
    "GitLabBoundRecord",
    "GitLabComparisonError",
    "GitLabComparisonWorld",
    "GitLabIssueRecord",
    "compile_gitlab_compare_decide_task",
    "expected_gitlab_compare_decide_response",
    "generate_gitlab_compare_decide_world",
    "is_feature_default_world",
    "select_gitlab_record",
]
