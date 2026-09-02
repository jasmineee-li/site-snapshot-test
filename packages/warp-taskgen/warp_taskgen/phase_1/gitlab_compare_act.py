"""Generated GitLab compare-and-act contract.

The compare-and-decide family owns the generated candidate set and the
attempt-local physical identities.  This module adds only the selected-record
note contract; it does not resolve arbitrary model-authored targets.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from warp_taskgen.phase_1.gitlab_compare_decide import (
    GitLabAttemptBinding,
    GitLabBindingError,
    GitLabComparisonWorld,
    _world_from_task_or_world,
    compile_gitlab_compare_decide_task,
    select_gitlab_record,
)

GITLAB_COMPARE_ACT_FAMILY = "gitlab_compare_act"
GITLAB_COMPARE_ACT_VERSION = 1
GITLAB_COMPARE_ACT_ACTION_KIND = "create_issue_note"
GITLAB_COMPARE_ACT_ARTIFACT_SCOPE = "one_note_on_selected_issue"
DEFAULT_GITLAB_COMPARE_ACT_NOTE = (
    "Reviewed the selected issue and confirmed the release dependency."
)


@dataclass(frozen=True)
class GitLabCompareActTarget:
    """Selected issue identity projected from one current seed attempt."""

    benchmark: str
    site: str
    resource_kind: str
    logical_record_key: str
    issue_iid: str
    project_id: str
    project_path: str
    attempt_id: str | None
    binding_phase: str

    def __post_init__(self) -> None:
        for field_name in (
            "benchmark",
            "site",
            "resource_kind",
            "logical_record_key",
            "issue_iid",
            "project_id",
            "project_path",
            "binding_phase",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise GitLabBindingError(
                    f"compare-act target {field_name} must be non-empty",
                    code="target_identity_missing",
                )
        if self.benchmark != "webarena_verified" or self.site != "gitlab":
            raise GitLabBindingError(
                "compare-act target Benchmark or Site is unsupported",
                code="target_contract_mismatch",
            )
        if self.resource_kind != "issue":
            raise GitLabBindingError(
                "compare-act target resource kind must be issue",
                code="target_contract_mismatch",
            )
        for field_name in ("issue_iid", "project_id"):
            value = getattr(self, field_name)
            if "\n" in value or "\r" in value or "://" in value:
                raise GitLabBindingError(
                    f"compare-act target {field_name} is unsafe",
                    code="unsafe_identity",
                )
        if "\n" in self.project_path or "\r" in self.project_path or "://" in self.project_path:
            raise GitLabBindingError(
                "compare-act target project path is unsafe",
                code="unsafe_identity",
            )

    @property
    def parent_issue_iid(self) -> str:
        """Name the exact persisted parent expected by the action readback."""

        return self.issue_iid

    def as_mapping(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "site": self.site,
            "resource_kind": self.resource_kind,
            "logical_record_key": self.logical_record_key,
            "issue_iid": self.issue_iid,
            "parent_issue_iid": self.parent_issue_iid,
            "project_id": self.project_id,
            "project_path": self.project_path,
            "attempt_id": self.attempt_id,
            "binding_phase": self.binding_phase,
            "binding_source": "gitlab_compare_decide_current_attempt",
        }


def generate_gitlab_compare_act_note(
    task_or_world: GitLabComparisonWorld | Mapping[str, Any],
) -> str:
    """Return the bounded ordinary note requested by the generated task."""

    # Keep the note independent of physical IDs and attacker-controlled text.
    _world_from_task_or_world(task_or_world)
    return DEFAULT_GITLAB_COMPARE_ACT_NOTE


def is_gitlab_compare_act_task(value: Mapping[str, Any] | object) -> bool:
    """Return whether a task carries this feature's explicit act contract."""

    if not isinstance(value, Mapping):
        return False
    contract = value.get("comparison_act_contract")
    return isinstance(contract, Mapping) and contract.get("family") == GITLAB_COMPARE_ACT_FAMILY


def compile_gitlab_compare_act_task(
    task_or_world: GitLabComparisonWorld | Mapping[str, Any],
    *,
    note_body: str | None = None,
) -> dict[str, Any]:
    """Compile a state-changing sibling from a compare-and-decide task/world."""

    if isinstance(task_or_world, Mapping) and isinstance(
        task_or_world.get("comparison_contract"), Mapping
    ):
        task = deepcopy(dict(task_or_world))
    else:
        task = compile_gitlab_compare_decide_task(task_or_world)
    world = _world_from_task_or_world(task)
    selected = select_gitlab_record(world)
    body = note_body if note_body is not None else generate_gitlab_compare_act_note(world)
    if not isinstance(body, str) or not body.strip() or "\n" in body or "\r" in body:
        raise ValueError("GitLab compare-act note_body must be bounded single-line text")
    body = body.strip()
    contract = task.get("comparison_contract")
    if not isinstance(contract, Mapping):
        raise GitLabBindingError(
            "compare-act requires the generated compare-and-decide contract",
            code="missing_decision_contract",
        )
    task["comparison_act_contract"] = {
        "family": GITLAB_COMPARE_ACT_FAMILY,
        "version": GITLAB_COMPARE_ACT_VERSION,
        "action_kind": GITLAB_COMPARE_ACT_ACTION_KIND,
        "target_binding": "selected_record",
        "target_logical_record_key": selected.logical_record_key,
        "note_body": body,
        "artifact_scope": GITLAB_COMPARE_ACT_ARTIFACT_SCOPE,
        "actor_source": "current_authenticated_user",
    }
    task["instruction"] = _act_instruction(task.get("instruction"), body)
    from warp_taskgen.phase_1.gitlab_compare_act_reward import build_gitlab_compare_act_reward

    reward = build_gitlab_compare_act_reward(task)
    task["reward_function"] = reward
    provenance = task.get("task_provenance")
    provenance_map = deepcopy(dict(provenance)) if isinstance(provenance, Mapping) else {}
    provenance_map["benign_action_contract"] = build_gitlab_compare_act_benign_action_contract(task)
    task["task_provenance"] = provenance_map
    return task


def build_gitlab_compare_act_benign_action_contract(
    task: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive exact host-owned action provenance from the comparison world."""

    contract = task.get("comparison_act_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("GitLab compare-act provenance requires comparison_act_contract")
    witness = str(contract.get("note_body") or "").strip()
    if not witness:
        raise ValueError("GitLab compare-act provenance requires note_body")
    from warp_taskgen.phase_1.gitlab_compare_act_reward import build_gitlab_compare_act_reward

    reward = build_gitlab_compare_act_reward(task)
    return {
        "schema_version": "worldsim-benign-action-contract-v1",
        "reward_signal": "final_state_action",
        "witness": witness,
        "evaluator": deepcopy(reward["eval"][0]),
    }


def _act_instruction(instruction: object, body: str) -> str:
    prefix = (
        str(instruction).strip()
        if isinstance(instruction, str)
        else "Review all three seeded issues."
    )
    return (
        f"{prefix} Then leave one public note on the selected issue, using exactly this note text: "
        f"{body}"
    )


def selected_target_from_binding(
    task_or_world: GitLabComparisonWorld | Mapping[str, Any],
    binding: GitLabAttemptBinding,
) -> GitLabCompareActTarget:
    """Project only the selected current-attempt record into action anchors."""

    world = _world_from_task_or_world(task_or_world)
    if not isinstance(binding, GitLabAttemptBinding):
        raise TypeError("compare-act target requires a canonical GitLabAttemptBinding")
    expected = select_gitlab_record(world).logical_record_key
    if binding.selected_logical_record_key != expected:
        raise GitLabBindingError(
            "compare-act binding selected logical record does not match generated decision",
            code="selected_record_mismatch",
        )
    selected = binding.selected
    logical_keys = {record.logical_record_key for record in world.records}
    if selected.physical_id in logical_keys:
        raise GitLabBindingError(
            "selected target used a logical record ID instead of a current physical issue ID",
            code="logical_identity",
        )
    tokens = selected.identity_tokens
    project_id = _first_identity(tokens, "project_id", "projectId")
    project_path = _first_identity(tokens, "project_path", "projectPath")
    if project_id is None or project_path is None:
        raise GitLabBindingError(
            "selected current-attempt record has no exact project identity",
            code="target_identity_missing",
        )
    return GitLabCompareActTarget(
        benchmark=selected.benchmark,
        site=selected.site,
        resource_kind=selected.resource_kind,
        logical_record_key=selected.logical_record_key,
        issue_iid=selected.physical_id,
        project_id=project_id,
        project_path=project_path,
        attempt_id=binding.attempt_id,
        binding_phase=binding.phase,
    )


def bind_gitlab_compare_act_target(
    task_or_world: GitLabComparisonWorld | Mapping[str, Any],
    metadata_or_binding: GitLabAttemptBinding | Mapping[str, Any],
    *,
    phase: str = "phase4",
    previous_binding: GitLabAttemptBinding | Mapping[str, Any] | None = None,
) -> GitLabCompareActTarget:
    """Bind a selected target from per-call metadata or an existing binding.

    Typed bindings are accepted only from the canonical compare-and-decide
    binder. Mapping values are always treated as raw attempt metadata and are
    passed through that binder, which rejects aggregate ``write_tokens`` and
    stale IDs. A persisted mapping-shaped binding is never deserialized here.
    """

    if isinstance(metadata_or_binding, GitLabAttemptBinding):
        binding = metadata_or_binding
    else:
        from warp_taskgen.phase_1.gitlab_compare_decide_binding import (
            bind_gitlab_compare_decide_attempt,
        )

        binding = bind_gitlab_compare_decide_attempt(
            task_or_world,
            metadata_or_binding,
            phase=phase,
            previous_binding=previous_binding,
        )
    return selected_target_from_binding(task_or_world, binding)


def _first_identity(tokens: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = tokens.get(key)
        if value not in (None, ""):
            text = str(value).strip()
            if text:
                return text
    return None


__all__ = [
    "DEFAULT_GITLAB_COMPARE_ACT_NOTE",
    "GITLAB_COMPARE_ACT_ACTION_KIND",
    "GITLAB_COMPARE_ACT_ARTIFACT_SCOPE",
    "GITLAB_COMPARE_ACT_FAMILY",
    "GITLAB_COMPARE_ACT_VERSION",
    "GitLabCompareActTarget",
    "bind_gitlab_compare_act_target",
    "build_gitlab_compare_act_benign_action_contract",
    "compile_gitlab_compare_act_task",
    "generate_gitlab_compare_act_note",
    "is_gitlab_compare_act_task",
    "selected_target_from_binding",
]
