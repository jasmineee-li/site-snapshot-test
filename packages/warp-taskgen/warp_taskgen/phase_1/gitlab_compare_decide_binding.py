"""Strict current-attempt binding for the GitLab comparison feature.

The comparison compiler owns the stable world and task shape.  This module
owns the attempt-local join from per-call seed evidence to those logical
records; aggregate seed metadata is intentionally not an input to the join.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from warp_taskgen.benchmark_capabilities import normalize_benchmark_name
from warp_taskgen.phase_1.gitlab_compare_decide import (
    GitLabAttemptBinding,
    GitLabBindingError,
    GitLabBoundRecord,
    GitLabComparisonWorld,
    _world_from_task_or_world,
    select_gitlab_record,
)


def bind_gitlab_compare_decide_attempt(
    task_or_world: GitLabComparisonWorld | Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    phase: str,
    previous_binding: GitLabAttemptBinding | Mapping[str, Any] | None = None,
) -> GitLabAttemptBinding:
    """Strictly bind per-call seed results to logical GitLab records.

    The aggregate ``write_tokens`` field is intentionally ignored.  Every
    candidate must have its own result, identity, and contract metadata.
    """
    world = _world_from_task_or_world(task_or_world)
    declared_indices: dict[str, int] | None = None
    if isinstance(task_or_world, Mapping) and not isinstance(task_or_world, GitLabComparisonWorld):
        _validate_comparison_contract(task_or_world, world)
        declared_indices = _validate_seed_declarations(task_or_world, world)
    phase_name = phase.strip().lower() if isinstance(phase, str) else ""
    if phase_name not in {"phase2c", "phase4"}:
        raise GitLabBindingError("phase must be phase2c or phase4", code="invalid_phase")
    if not isinstance(metadata, Mapping):
        raise GitLabBindingError("attempt metadata must be an object", code="invalid_metadata")
    raw_rows = metadata.get("editor_call_results")
    if not isinstance(raw_rows, list):
        raise GitLabBindingError(
            "strict binding requires per-call editor_call_results; aggregate fallback is forbidden",
            code="missing_editor_call_results",
        )
    expected = {record.logical_record_key: record for record in world.records}
    rows_by_key: dict[str, GitLabBoundRecord] = {}
    seen_indices: set[int] = set()
    seen_physical_ids: set[str] = set()
    for row_number, raw_row in enumerate(raw_rows):
        if not isinstance(raw_row, Mapping):
            raise GitLabBindingError(
                f"call-result row {row_number + 1} is missing", code="missing_result"
            )
        nested = raw_row.get("result")
        if "result" in raw_row:
            if not isinstance(nested, Mapping):
                raise GitLabBindingError(
                    f"call result row {row_number + 1} is missing", code="missing_result"
                )
            evidence: Mapping[str, Any] = nested
        else:
            evidence = raw_row
        logical_key = _first_text(
            raw_row,
            "logical_record_key",
            "logicalRecordKey",
            "logicalAlias",
            "logical_alias",
        )
        if logical_key is None:
            logical_key = _first_text(
                evidence,
                "logical_record_key",
                "logicalRecordKey",
                "logicalAlias",
                "logical_alias",
            )
        if logical_key is None:
            raise GitLabBindingError("call-result row has no logical record key", code="missing_key")
        logical_key = logical_key.strip()
        if logical_key in rows_by_key:
            raise GitLabBindingError(
                f"duplicate logical record key {logical_key!r}", code="duplicate_key"
            )
        if logical_key not in expected:
            raise GitLabBindingError(
                f"foreign logical record key {logical_key!r}", code="foreign_logical_key"
            )
        raw_index = raw_row.get("call_index", evidence.get("call_index"))
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            raise GitLabBindingError(
                f"call result for {logical_key} has no valid call_index", code="missing_call_index"
            )
        if declared_indices is not None and declared_indices.get(logical_key) != raw_index:
            raise GitLabBindingError(
                f"call_index for {logical_key} does not match its seed declaration",
                code="call_index_mismatch",
            )
        if raw_index in seen_indices:
            raise GitLabBindingError(
                f"call_index {raw_index} appears more than once", code="duplicate_call_index"
            )
        seen_indices.add(raw_index)
        benchmark = _first_text(evidence, "benchmark", "benchmark_name")
        if benchmark is None or _safe_normalize_benchmark(benchmark) != world.benchmark:
            raise GitLabBindingError(
                f"call result for {logical_key} has mismatched benchmark", code="benchmark_mismatch"
            )
        site = _first_text(evidence, "site")
        if site is None or site.strip().lower() != world.site:
            raise GitLabBindingError(
                f"call result for {logical_key} has mismatched site", code="site_mismatch"
            )
        method = _first_text(evidence, "method", "editor_method")
        if method is not None and "." in method:
            method = method.rsplit(".", 1)[-1]
        if method is None or method.strip() != world.method:
            raise GitLabBindingError(
                f"call result for {logical_key} has mismatched method", code="method_mismatch"
            )
        resource_kind = _resource_kind(evidence)
        if resource_kind != world.resource_kind:
            raise GitLabBindingError(
                f"call result for {logical_key} has mismatched resource kind",
                code="resource_kind_mismatch",
            )
        physical_id = _physical_identity(evidence)
        if physical_id is None:
            raise GitLabBindingError(
                f"call result for {logical_key} has no safe physical identity", code="missing_identity"
            )
        if physical_id in seen_physical_ids:
            raise GitLabBindingError(
                f"physical ID {physical_id!r} is bound to more than one logical record",
                code="duplicate_identity",
            )
        seen_physical_ids.add(physical_id)
        identity_tokens = _identity_evidence(evidence)
        _assert_identity_matches_physical_id(evidence, identity_tokens, physical_id, logical_key)
        expected_record = expected[logical_key]
        reported_facts = evidence.get("facts")
        if reported_facts is not None and _canonical_facts(reported_facts) != dict(expected_record.facts):
            raise GitLabBindingError(
                f"facts for {logical_key} do not match generated world", code="facts_mismatch"
            )
        rows_by_key[logical_key] = GitLabBoundRecord(
            logical_record_key=logical_key,
            physical_id=physical_id,
            benchmark=world.benchmark,
            site=world.site,
            method=world.method,
            resource_kind=world.resource_kind,
            facts=expected_record.facts,
            call_index=raw_index,
            identity_tokens=identity_tokens,
        )
    if seen_indices != set(range(len(raw_rows))):
        raise GitLabBindingError("call results have missing call_index values", code="missing_call_index")
    missing = [key for key in expected if key not in rows_by_key]
    if missing:
        raise GitLabBindingError(
            "missing logical record key(s): " + ", ".join(missing), code="missing_key"
        )
    previous_ids = _binding_physical_ids(previous_binding)
    reused = sorted(seen_physical_ids & previous_ids)
    if reused:
        raise GitLabBindingError(
            f"stale physical ID {reused[0]!r} was reused from the previous attempt",
            code="stale_identity",
        )
    attempt_id = metadata.get("attempt_id") or metadata.get("attemptId")
    if attempt_id in (None, ""):
        attempt_id_value = None
    elif isinstance(attempt_id, str) and attempt_id.strip():
        attempt_id_value = attempt_id.strip()
    else:
        raise GitLabBindingError("attempt_id must be a non-empty string", code="invalid_attempt_id")
    previous_attempt_id = _binding_attempt_id(previous_binding)
    if attempt_id_value is not None and previous_attempt_id == attempt_id_value:
        raise GitLabBindingError(
            f"attempt_id {attempt_id_value!r} is stale from the previous attempt",
            code="stale_attempt",
        )
    selected = select_gitlab_record(world)
    return GitLabAttemptBinding(
        phase=phase_name,
        attempt_id=attempt_id_value,
        records=rows_by_key,
        selected_logical_record_key=selected.logical_record_key,
    )


def _first_text(value: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        raw = value.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return None


def _safe_normalize_benchmark(value: str) -> str | None:
    try:
        return normalize_benchmark_name(value)
    except ValueError:
        return None


def _resource_kind(evidence: Mapping[str, Any]) -> str | None:
    raw = evidence.get("resource_kind") or evidence.get("resourceKind")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    for key in ("created_resource", "createdResource"):
        resource = evidence.get(key)
        if isinstance(resource, Mapping):
            kind = resource.get("kind")
            if isinstance(kind, str) and kind.strip():
                return kind.strip()
    resources = evidence.get("created_resources") or evidence.get("createdResources")
    if isinstance(resources, list) and resources:
        kinds = {
            str(item.get("kind")).strip()
            for item in resources
            if isinstance(item, Mapping) and item.get("kind") not in (None, "")
        }
        if len(kinds) == 1:
            return next(iter(kinds))
    return None


def _physical_identity(evidence: Mapping[str, Any]) -> str | None:
    candidates: list[Any] = [
        evidence.get("physical_id"),
        evidence.get("physicalId"),
        evidence.get("physicalID"),
        evidence.get("resource_id"),
        evidence.get("issue_iid"),
        evidence.get("iid"),
    ]
    tokens = evidence.get("write_tokens")
    if isinstance(tokens, Mapping):
        candidates.extend((tokens.get("issue_iid"), tokens.get("resource_id"), tokens.get("id")))
    for key in ("created_resource", "createdResource"):
        resource = evidence.get(key)
        if isinstance(resource, Mapping):
            candidates.extend((resource.get("iid"), resource.get("id")))
    resources = evidence.get("created_resources") or evidence.get("createdResources")
    if isinstance(resources, list):
        for resource in resources:
            if isinstance(resource, Mapping):
                candidates.extend((resource.get("iid"), resource.get("id")))
    for raw in candidates:
        if isinstance(raw, bool) or raw in (None, ""):
            continue
        if not isinstance(raw, (str, int)):
            continue
        value = str(raw).strip()
        if not value or "\n" in value or "\r" in value or "://" in value:
            raise GitLabBindingError("physical identity contains unsafe URL/text", code="unsafe_identity")
        return value
    return None


def _identity_evidence(evidence: Mapping[str, Any]) -> Mapping[str, str | int]:
    from warp_taskgen.seeding.site_contracts import normalize_identity_tokens

    raw = evidence.get("identity_tokens")
    if raw is None:
        raw = evidence.get("write_tokens")
    if raw is None:
        raw = {}
    try:
        normalized = normalize_identity_tokens(raw)
    except ValueError as exc:
        raise GitLabBindingError(
            "call result identity evidence is unsafe", code="unsafe_identity"
        ) from exc
    return normalized


def _assert_identity_matches_physical_id(
    evidence: Mapping[str, Any],
    identity_tokens: Mapping[str, str | int],
    physical_id: str,
    logical_key: str,
) -> None:
    """Reject a row whose explicit issue identity disagrees with its ID."""
    candidates: list[Any] = [
        identity_tokens.get("issue_iid"),
        identity_tokens.get("resource_id"),
        evidence.get("issue_iid"),
        evidence.get("iid"),
        evidence.get("resource_id"),
    ]
    tokens = evidence.get("write_tokens")
    if isinstance(tokens, Mapping):
        candidates.extend((tokens.get("issue_iid"), tokens.get("resource_id")))
    for key in ("created_resource", "createdResource"):
        resource = evidence.get(key)
        if isinstance(resource, Mapping):
            # GitLab's global ``id`` differs from the issue IID used as the
            # browser-visible physical identity.  Only an explicit IID is a
            # comparable identity claim here.
            candidates.append(resource.get("iid"))
    for raw in candidates:
        if raw in (None, "") or isinstance(raw, bool):
            continue
        if str(raw).strip() != physical_id:
            raise GitLabBindingError(
                f"identity evidence for {logical_key} does not match physical ID {physical_id!r}",
                code="identity_mismatch",
            )


def _canonical_facts(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): str(raw).strip() for key, raw in value.items() if raw not in (None, "")}


def _validate_seed_declarations(
    task: Mapping[str, Any], world: GitLabComparisonWorld
) -> dict[str, int]:
    raw_seed = task.get("adversarial_data_seed") or task.get("data_seed")
    calls = raw_seed.get("editor_calls") if isinstance(raw_seed, Mapping) else None
    if not isinstance(calls, list):
        raise GitLabBindingError(
            "strict binding requires the comparison seed's editor_calls declarations",
            code="missing_seed_declarations",
        )
    expected_keys = {record.logical_record_key for record in world.records}
    indices: dict[str, int] = {}
    for index, call in enumerate(calls):
        if not isinstance(call, Mapping):
            raise GitLabBindingError(
                f"seed declaration {index + 1} is missing", code="missing_seed_declaration"
            )
        raw_key = call.get("logical_record_key")
        if not isinstance(raw_key, str) or not raw_key.strip():
            raise GitLabBindingError(
                f"seed declaration {index + 1} has no logical record key",
                code="missing_seed_key",
            )
        key = raw_key.strip()
        if key in indices:
            raise GitLabBindingError(f"duplicate seed logical record key {key!r}", code="duplicate_key")
        if key not in expected_keys:
            raise GitLabBindingError(f"foreign seed logical record key {key!r}", code="foreign_logical_key")
        benchmark = _first_text(call, "benchmark", "benchmark_name")
        site = _first_text(call, "site")
        method = _first_text(call, "method")
        resource_kind = _first_text(call, "resource_kind", "resourceKind")
        if (
            benchmark is None
            or _safe_normalize_benchmark(benchmark) != world.benchmark
            or site is None
            or site.lower() != world.site
            or method is None
            or method != world.method
            or (resource_kind is not None and resource_kind != world.resource_kind)
        ):
            raise GitLabBindingError(
                f"seed declaration for {key} has a mismatched contract",
                code="seed_contract_mismatch",
            )
        indices[key] = index
    if set(indices) != expected_keys:
        missing = sorted(expected_keys - set(indices))
        raise GitLabBindingError(
            "missing seed logical record key(s): " + ", ".join(missing), code="missing_seed_key"
        )
    return indices


def _validate_comparison_contract(task: Mapping[str, Any], world: GitLabComparisonWorld) -> None:
    contract = task.get("comparison_contract")
    if not isinstance(contract, Mapping):
        return
    for field_name, expected in (
        ("benchmark", world.benchmark),
        ("site", world.site),
        ("method", world.method),
        ("resource_kind", world.resource_kind),
    ):
        raw = contract.get(field_name)
        if raw is not None and str(raw).strip().lower() != str(expected).lower():
            raise GitLabBindingError(
                f"comparison contract has mismatched {field_name}", code="contract_mismatch"
            )
    expected_keys = contract.get("expected_logical_record_keys")
    if expected_keys is not None:
        if not isinstance(expected_keys, list) or set(expected_keys) != {
            record.logical_record_key for record in world.records
        }:
            raise GitLabBindingError(
                "comparison contract has mismatched logical record keys",
                code="contract_mismatch",
            )
    selected_key = contract.get("selected_logical_record_key")
    if selected_key is not None and selected_key != select_gitlab_record(world).logical_record_key:
        raise GitLabBindingError(
            "comparison contract has mismatched selected logical record key",
            code="contract_mismatch",
        )
    rule = contract.get("decision_rule")
    if rule is not None and _canonical_facts(rule) != dict(world.decision_rule):
        raise GitLabBindingError(
            "comparison contract has mismatched decision rule", code="contract_mismatch"
        )


def _binding_physical_ids(binding: GitLabAttemptBinding | Mapping[str, Any] | None) -> set[str]:
    if binding is None:
        return set()
    records: object = binding.records if isinstance(binding, GitLabAttemptBinding) else binding.get("records")
    if isinstance(records, Mapping):
        ids: set[str] = set()
        for record in records.values():
            if isinstance(record, GitLabBoundRecord):
                ids.add(record.physical_id)
            elif isinstance(record, Mapping):
                raw = record.get("physical_id") or record.get("physicalId")
                if raw not in (None, ""):
                    ids.add(str(raw))
        return ids
    if isinstance(records, list):
        return {
            str(raw.get("physical_id") or raw.get("physicalId"))
            for raw in records
            if isinstance(raw, Mapping)
            and (raw.get("physical_id") or raw.get("physicalId")) not in (None, "")
        }
    return set()


def _binding_attempt_id(binding: GitLabAttemptBinding | Mapping[str, Any] | None) -> str | None:
    if isinstance(binding, GitLabAttemptBinding):
        return binding.attempt_id
    if isinstance(binding, Mapping):
        raw = binding.get("attempt_id") or binding.get("attemptId")
        return str(raw).strip() if isinstance(raw, str) and raw.strip() else None
    return None


__all__ = ["bind_gitlab_compare_decide_attempt"]
