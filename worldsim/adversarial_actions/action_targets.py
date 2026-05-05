"""Host-owned action target contracts for cross-surface public mutations.

Carrier placement and browser-side action targets are separate contracts. This
module owns the small deterministic contract used when an injected payload is
seeded on one WASP carrier but asks the agent to perform a page-native public
mutation on a related target, such as GitLab issue description -> same-issue
comment.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

ACTION_TARGET_CONTRACT_SCHEMA_VERSION = "worldsim-action-target-contract-v1"

ACTION_TARGET_CONTRACT_FIELDS = frozenset(
    {
        "schema_version",
        "id",
        "site",
        "source_surface_id",
        "source_editor_method",
        "action_kind",
        "target_surface_id",
        "target_editor_method",
        "binding",
        "required_host_evidence",
        "state_probe_kind",
        "model_visible_context_terms",
    }
)

ACTION_TARGET_BINDINGS = frozenset({"same_issue"})
_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_]*(?:[.-][a-z0-9_]+)*$")
_SAFE_DOTTED_ID_RE = re.compile(r"^[a-z][a-z0-9_]*(?:[._-][a-z0-9_]+)*$")
_FORBIDDEN_MODEL_VISIBLE_SUBSTRINGS = (
    "__gitlab__",
    "__reddit__",
    "attacker.invalid",
    "worldsim",
    "fixture",
    "nonce",
    "/api/",
    "graphql",
    "selector",
    "evaluator",
    "reward",
    "cleanup",
)


@dataclass(frozen=True)
class ActionTargetContract:
    """Host-owned target binding for a cross-surface action objective."""

    id: str
    site: str
    source_surface_id: str
    source_editor_method: str
    action_kind: str
    target_surface_id: str
    target_editor_method: str
    binding: str
    state_probe_kind: str
    required_host_evidence: tuple[str, ...] = ()
    model_visible_context_terms: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": ACTION_TARGET_CONTRACT_SCHEMA_VERSION,
            "id": self.id,
            "site": self.site,
            "source_surface_id": self.source_surface_id,
            "source_editor_method": self.source_editor_method,
            "action_kind": self.action_kind,
            "target_surface_id": self.target_surface_id,
            "target_editor_method": self.target_editor_method,
            "binding": self.binding,
            "state_probe_kind": self.state_probe_kind,
        }
        if self.required_host_evidence:
            result["required_host_evidence"] = list(self.required_host_evidence)
        if self.model_visible_context_terms:
            result["model_visible_context_terms"] = list(self.model_visible_context_terms)
        return result

    def to_model_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "action_kind": self.action_kind,
            "binding": self.binding,
            "model_visible_context_terms": list(self.model_visible_context_terms),
        }


def action_target_contract_from_mapping(value: Mapping[str, Any]) -> ActionTargetContract:
    problem = validate_action_target_contract(value)
    if problem is not None:
        raise ValueError(problem)
    return ActionTargetContract(
        id=str(value["id"]).strip(),
        site=str(value["site"]).strip(),
        source_surface_id=str(value["source_surface_id"]).strip(),
        source_editor_method=str(value["source_editor_method"]).strip(),
        action_kind=str(value["action_kind"]).strip(),
        target_surface_id=str(value["target_surface_id"]).strip(),
        target_editor_method=str(value["target_editor_method"]).strip(),
        binding=str(value["binding"]).strip(),
        state_probe_kind=str(value["state_probe_kind"]).strip(),
        required_host_evidence=_string_tuple(value.get("required_host_evidence")),
        model_visible_context_terms=_string_tuple(value.get("model_visible_context_terms")),
    )


def validate_action_target_contract(value: Any) -> str | None:
    """Return a fail-closed validation problem for an action target contract."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        return "action_target_contract must be an object"
    unknown = sorted(set(value) - ACTION_TARGET_CONTRACT_FIELDS)
    if unknown:
        return f"action_target_contract contains unsupported field(s): {unknown}"
    schema_version = value.get("schema_version")
    if (
        schema_version is not None
        and schema_version != ACTION_TARGET_CONTRACT_SCHEMA_VERSION
    ):
        return (
            "action_target_contract.schema_version must be "
            f"{ACTION_TARGET_CONTRACT_SCHEMA_VERSION!r}"
        )
    for key in ("id", "site", "action_kind", "binding", "state_probe_kind"):
        problem = _identifier_field_problem(value, key)
        if problem is not None:
            return problem
    for key in (
        "source_surface_id",
        "source_editor_method",
        "target_surface_id",
        "target_editor_method",
    ):
        problem = _safe_id_field_problem(value, key)
        if problem is not None:
            return problem
    binding = str(value.get("binding") or "").strip()
    if binding not in ACTION_TARGET_BINDINGS:
        return (
            "action_target_contract.binding must be one of "
            f"{sorted(ACTION_TARGET_BINDINGS)}"
        )
    required_problem = _string_array_problem(value, "required_host_evidence")
    if required_problem is not None:
        return required_problem
    required_evidence = _string_tuple(value.get("required_host_evidence"))
    if not required_evidence:
        return "action_target_contract.required_host_evidence must be a non-empty array"
    for item in required_evidence:
        if _IDENTIFIER_RE.fullmatch(item) is None:
            return (
                "action_target_contract.required_host_evidence must contain "
                "lowercase identifiers"
            )
    terms_problem = _string_array_problem(value, "model_visible_context_terms")
    if terms_problem is not None:
        return terms_problem
    for term in _string_tuple(value.get("model_visible_context_terms")):
        lowered = term.casefold()
        for forbidden in _FORBIDDEN_MODEL_VISIBLE_SUBSTRINGS:
            if forbidden in lowered:
                return (
                    "action_target_contract.model_visible_context_terms contains "
                    f"forbidden prompt-visible text {forbidden!r}"
                )
    return None


def validate_action_target_contracts_field(
    value: Any,
    *,
    field_name: str = "action_target_contracts",
) -> str | None:
    """Return a validation problem for a plural action target contract field."""

    if value is None:
        return None
    if isinstance(value, Mapping):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        return f"{field_name} must be an object or array of objects"
    for index, item in enumerate(values):
        if not isinstance(item, Mapping):
            return f"{field_name}[{index}] must be an object"
        problem = validate_action_target_contract(item)
        if problem is not None:
            return f"{field_name}[{index}].{problem}"
    return None


def action_target_contracts_from_card(
    card: Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any], ...]:
    return _contracts_from_mapping(card)


def action_target_contracts_from_task(
    task: Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(task, Mapping):
        return ()
    provenance = task.get("task_provenance")
    if isinstance(provenance, Mapping):
        contracts = _contracts_from_mapping(provenance)
        if contracts:
            return contracts
    return _contracts_from_mapping(task)


def action_target_contract_for_kind(
    task: Mapping[str, Any] | None,
    action_kind: str,
) -> Mapping[str, Any] | None:
    normalized = str(action_kind or "").strip()
    for contract in action_target_contracts_from_task(task):
        if str(contract.get("action_kind") or "").strip() == normalized:
            return contract
    return None


def target_editor_method_for_action(
    task: Mapping[str, Any] | None,
    action_kind: str,
) -> str | None:
    contract = action_target_contract_for_kind(task, action_kind)
    if not isinstance(contract, Mapping):
        return None
    method = contract.get("target_editor_method")
    return str(method).strip() if isinstance(method, str) and method.strip() else None


def _contracts_from_mapping(value: Mapping[str, Any] | None) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Mapping):
        return ()
    raw = value.get("action_target_contracts")
    if isinstance(raw, Mapping):
        raw_values = [raw]
    elif isinstance(raw, list):
        raw_values = raw
    else:
        single = value.get("action_target_contract")
        raw_values = [single] if isinstance(single, Mapping) else []
    contracts: list[Mapping[str, Any]] = []
    for item in raw_values:
        if not isinstance(item, Mapping):
            continue
        problem = validate_action_target_contract(item)
        if problem is None:
            contracts.append(item)
    return tuple(contracts)


def _identifier_field_problem(value: Mapping[str, Any], key: str) -> str | None:
    raw = value.get(key)
    if not isinstance(raw, str) or _IDENTIFIER_RE.fullmatch(raw.strip()) is None:
        return f"action_target_contract.{key} must be a lowercase identifier"
    return None


def _safe_id_field_problem(value: Mapping[str, Any], key: str) -> str | None:
    raw = value.get(key)
    if not isinstance(raw, str) or _SAFE_DOTTED_ID_RE.fullmatch(raw.strip()) is None:
        return f"action_target_contract.{key} must be a safe identifier"
    return None


def _string_array_problem(value: Mapping[str, Any], key: str) -> str | None:
    if key not in value:
        return None
    raw = value.get(key)
    if not isinstance(raw, list) or not all(
        isinstance(item, str) and item.strip() for item in raw
    ):
        return f"action_target_contract.{key} must be an array of strings"
    return None


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str) and value.strip():
        return (value.strip(),)
    if isinstance(value, list):
        return tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
    return ()
