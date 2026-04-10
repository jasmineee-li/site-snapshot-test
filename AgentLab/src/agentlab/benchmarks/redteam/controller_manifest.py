"""Behavior spec normalization, contract construction, and manifest I/O."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from agentlab.benchmarks.redteam.app_artifacts import (
    APP_MANIFEST_CONTRACT_VERSION,
    GENERATION_STATUS_IN_PROGRESS,
    behavior_contract_path,
    behavior_contract_compatibility_error,
    behavior_contract_surface_fingerprint,
    behavior_variant_name_error,
    build_attack_metadata,
    compute_docs_snapshot,
    load_behavior_contract,
    validate_path_component,
)
from agentlab.benchmarks.redteam.behavior_ids import resolve_behavior_id
from agentlab.benchmarks.redteam.config import parse_case_fields
from agentlab.benchmarks.redteam.controller_state import generation_phase_status_template
from agentlab.benchmarks.redteam.execution import execution_backend_metadata
from agentlab.benchmarks.redteam.git_ops import ControllerWorkspace
from agentlab.benchmarks.redteam.phase_ids import (
    PHASE_1A,
    PHASE_1B,
    PHASE_1C,
    PHASE_2A,
    PHASE_2B,
    PHASE_3A,
    PHASE_3B,
    PHASE_4A,
    PHASE_4B,
    PHASE_5,
)
from agentlab.benchmarks.redteam.utils import (
    normalize_route_reference,
    sha256_file,
    utc_timestamp as _timestamp,
)


# ---------------------------------------------------------------------------
# Behavior-spec normalization helpers
# ---------------------------------------------------------------------------


def _resolved_app_id(spec: dict[str, Any]) -> str:
    app_id = str(spec.get("app_id") or "").strip()
    if app_id:
        return validate_path_component(app_id, "app_id")
    return validate_path_component(resolve_behavior_id(spec), "behavior_id")


def _normalized_behavior_spec_ids(spec: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(spec)

    app_id = str(normalized.get("app_id") or "").strip()
    if app_id:
        normalized["app_id"] = validate_path_component(app_id, "app_id")

    mapped_behaviors = normalized.get("mapped_behaviors")
    if isinstance(mapped_behaviors, list) and mapped_behaviors:
        normalized_mapped_behaviors: list[dict[str, Any]] = []
        for item in mapped_behaviors:
            if not isinstance(item, dict):
                continue
            normalized_item = dict(item)
            behavior_id = validate_path_component(
                resolve_behavior_id(normalized_item),
                "behavior_id",
            )
            normalized_item["id"] = behavior_id
            if "behavior_id" in normalized_item:
                normalized_item["behavior_id"] = behavior_id
            normalized_mapped_behaviors.append(normalized_item)
        normalized["mapped_behaviors"] = normalized_mapped_behaviors
    else:
        behavior_id = validate_path_component(resolve_behavior_id(normalized), "behavior_id")
        normalized["id"] = behavior_id
        if "behavior_id" in normalized:
            normalized["behavior_id"] = behavior_id

    return normalized


def _mapped_behavior_specs(spec: dict[str, Any]) -> list[dict[str, Any]]:
    mapped = spec.get("mapped_behaviors") or [spec]
    return [dict(item) for item in mapped if isinstance(item, dict)]


# ---------------------------------------------------------------------------
# Seed metadata
# ---------------------------------------------------------------------------


def _update_manifest_seed_metadata(
    manifest: dict[str, Any],
    *,
    app_dir: Path,
) -> str | None:
    benign_seed = app_dir / "benign" / "data.js"
    if not benign_seed.exists():
        return "benign/data.js missing; cannot compute shared seed metadata"
    shared_seed_hash = f"sha256:{sha256_file(benign_seed)}"
    previous_hash = str(manifest.get("shared_seed_hash") or "").strip()
    previous_version = manifest.get("shared_seed_version")
    normalized_previous_version = previous_version if isinstance(previous_version, int) and previous_version > 0 else 1
    if previous_hash and previous_hash not in {"", "sha256:pending"} and previous_hash != shared_seed_hash:
        manifest["shared_seed_version"] = normalized_previous_version + 1
    else:
        manifest["shared_seed_version"] = normalized_previous_version
    manifest["shared_seed_hash"] = shared_seed_hash
    return None


# ---------------------------------------------------------------------------
# Route / domain helpers
# ---------------------------------------------------------------------------


def _default_primary_spa_domain_bindings(app_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    bindings: list[dict[str, Any]] = []
    seen_domains: set[str] = set()
    for page in app_manifest.get("pages") or []:
        if not isinstance(page, dict):
            continue
        base_site_url = str(page.get("base_site_url") or "").strip()
        if "://" not in base_site_url:
            continue
        domain = base_site_url.split("://", 1)[1].split("/", 1)[0].strip()
        if not domain or domain in seen_domains:
            continue
        seen_domains.add(domain)
        bindings.append({"domain": domain, "mode": "primary_spa"})
    return bindings


def _normalized_route_list(routes: list[Any]) -> list[str]:
    normalized_routes: list[str] = []
    for route in routes:
        normalized = normalize_route_reference(str(route or ""))
        if normalized and normalized not in normalized_routes:
            normalized_routes.append(normalized)
    return normalized_routes


def _require_safe_behavior(behavior_spec: dict[str, Any]) -> str:
    behavior_id = resolve_behavior_id(behavior_spec)
    safe_behavior = str(behavior_spec.get("safe_behavior") or "").strip()
    if safe_behavior:
        return safe_behavior
    if any(
        key in behavior_spec
        for key in ("entry_route", "allowed_routes", "domain_bindings", "mock_endpoints")
    ):
        raise ValueError(f"{behavior_id}: missing non-empty safe_behavior")
    return parse_case_fields(behavior_spec)["doc"]


def _compatibility_evidence(
    payload: dict[str, Any],
    *,
    app_manifest: dict[str, Any],
    extra: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    fingerprint_error = error
    try:
        fingerprint = behavior_contract_surface_fingerprint(payload)
    except (TypeError, ValueError) as exc:
        fingerprint = ""
        if fingerprint_error is None:
            fingerprint_error = str(exc)

    evidence = {
        "checked_at": _timestamp(),
        "checked_against_seed_version": app_manifest.get("shared_seed_version"),
        "checked_against_seed_hash": app_manifest.get("shared_seed_hash"),
        "checked_entry_route": payload.get("entry_route"),
        "checked_behavior_fingerprint": fingerprint,
        "checked_safe_behavior": payload.get("safe_behavior"),
        "checked_success_condition": payload.get("success_condition"),
    }
    if isinstance(extra, dict) and extra:
        evidence.update(extra)
    if fingerprint_error:
        evidence["error"] = fingerprint_error
    return evidence


def _load_task_definitions(tasks_file: Path) -> list[dict[str, Any]] | None:
    try:
        payload = json.loads(tasks_file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict) and item.get("id")]


def _compatibility_seed_ref_evidence(
    payload: dict[str, Any],
    *,
    app_dir: Path | None,
) -> tuple[dict[str, Any], str | None]:
    if app_dir is None:
        return {}, f"behaviors/{payload['behavior_id']}.json missing benign/data.js for compatibility check"
    benign_path = app_dir / "benign" / "data.js"
    if not benign_path.exists():
        return {}, f"behaviors/{payload['behavior_id']}.json missing benign/data.js for compatibility check"
    try:
        from agentlab.benchmarks.redteam.app_artifacts import _resolved_seed_refs, read_variant_data_js

        benign_data = read_variant_data_js(benign_path)
        resolved_seed_refs, seed_ref_error = _resolved_seed_refs(
            list(payload.get("seed_refs") or []),
            behavior_id=payload["behavior_id"],
            benign_data=benign_data,
        )
    except Exception as exc:
        return {}, f"behaviors/{payload['behavior_id']}.json could not read benign/data.js ({exc})"

    evidence = {
        "seed_refs_resolved": seed_ref_error is None,
    }
    if resolved_seed_refs:
        evidence["resolved_seed_refs"] = resolved_seed_refs
    return evidence, seed_ref_error


def _compatibility_task_evidence(
    payload: dict[str, Any],
    *,
    app_manifest: dict[str, Any],
    app_dir: Path | None,
) -> tuple[dict[str, Any], str | None]:
    if app_dir is None:
        return {}, (
            f"behaviors/{payload['behavior_id']}.json missing real-tasks.json for compatibility check"
        )
    tasks = _load_task_definitions(app_dir / "real-tasks.json")
    if tasks is None:
        return {}, (
            f"behaviors/{payload['behavior_id']}.json missing real-tasks.json for compatibility check"
        )
    if not tasks:
        return {}, (
            f"behaviors/{payload['behavior_id']}.json real-tasks.json is unreadable or does not define "
            "task objects"
        )

    matching_tasks = [
        task
        for task in tasks
        if isinstance(task.get("behavior_compatibility"), dict)
        and str((task.get("behavior_compatibility") or {}).get("behavior_id") or "").strip()
        == payload["behavior_id"]
    ]
    if not matching_tasks:
        return {}, (
            f"behaviors/{payload['behavior_id']}.json missing a real-task compatibility anchor tagged "
            f"for behavior_id {payload['behavior_id']!r}"
        )
    if len(matching_tasks) > 1:
        return {}, (
            f"behaviors/{payload['behavior_id']}.json has multiple real-task compatibility anchors "
            f"for behavior_id {payload['behavior_id']!r}"
        )

    task = matching_tasks[0]
    success_condition = str(task.get("success_condition") or payload.get("safe_behavior") or "").strip()
    compatibility_task = {
        "suite": "real-tasks",
        "behavior_id": payload["behavior_id"],
        "task_id": str(task.get("id") or "").strip(),
        "instruction": str(task.get("instruction") or "").strip(),
        "success_condition": success_condition,
        "passed": False,
    }
    evidence = {"compatibility_task": compatibility_task}
    if compatibility_task["instruction"] != str(payload.get("safe_behavior") or "").strip():
        return evidence, (
            f"behaviors/{payload['behavior_id']}.json current real-tasks compatibility anchor does not "
            "match safe_behavior"
        )
    if success_condition != str(payload.get("success_condition") or "").strip():
        return evidence, (
            f"behaviors/{payload['behavior_id']}.json current real-tasks compatibility anchor does not "
            "match success_condition"
        )

    results = (((app_manifest.get("functional_tests") or {}).get("real_evaluation") or {}).get("results")) or []
    if not results:
        return evidence, (
            f"behaviors/{payload['behavior_id']}.json missing declared real-task results for compatibility check"
        )
    matching_result = next(
        (
            item
            for item in results
            if str(item.get("task_id") or "").strip() == compatibility_task["task_id"]
        ),
        None,
    )
    if matching_result is None:
        return evidence, (
            f"behaviors/{payload['behavior_id']}.json compatibility task {compatibility_task['task_id']!r} "
            "was not executed in benign real-task results"
        )
    compatibility_task["passed"] = bool(matching_result.get("passed"))
    if compatibility_task["passed"] is not True:
        return evidence, (
            f"behaviors/{payload['behavior_id']}.json compatibility task {compatibility_task['task_id']!r} "
            "did not pass benign real-task evaluation"
        )
    return evidence, None


def _compatibility_context_matches(
    *,
    payload: dict[str, Any],
    app_manifest: dict[str, Any],
    existing_contract: dict[str, Any],
) -> bool:
    evidence = existing_contract.get("compatibility_evidence")
    if not isinstance(evidence, dict):
        return False
    return (
        evidence.get("checked_against_seed_version") == app_manifest.get("shared_seed_version")
        and str(evidence.get("checked_against_seed_hash") or "").strip()
        == str(app_manifest.get("shared_seed_hash") or "").strip()
        and str(evidence.get("checked_safe_behavior") or "").strip()
        == str(payload.get("safe_behavior") or "").strip()
        and str(evidence.get("checked_success_condition") or "").strip()
        == str(payload.get("success_condition") or "").strip()
        and normalize_route_reference(str(evidence.get("checked_entry_route") or ""))
        == normalize_route_reference(str(payload.get("entry_route") or ""))
        and behavior_contract_surface_fingerprint(existing_contract)
        == behavior_contract_surface_fingerprint(payload)
    )


def _preserved_pending_compatibility_state(
    *,
    payload: dict[str, Any],
    app_manifest: dict[str, Any],
    existing_contract: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]] | None:
    if not existing_contract:
        return None
    existing_status = str(existing_contract.get("compatibility_status") or "").strip()
    existing_evidence = existing_contract.get("compatibility_evidence")
    if existing_status not in {"passed", "failed", "stale"} or not isinstance(existing_evidence, dict):
        return None
    if _compatibility_context_matches(
        payload=payload,
        app_manifest=app_manifest,
        existing_contract=existing_contract,
    ):
        return existing_status, deepcopy(existing_evidence)
    return "stale", _compatibility_evidence(
        payload,
        app_manifest=app_manifest,
        error="Compatibility evidence is stale and must be revalidated against the current benign seed.",
    )


def _is_pending_compatibility_error(error: str | None) -> bool:
    if not error:
        return False
    return any(
        marker in error
        for marker in (
            "missing benign/data.js for compatibility check",
            "could not read benign/data.js",
            "missing real-tasks.json for compatibility check",
            "missing declared real-task results for compatibility check",
            "missing compatibility_evidence.compatibility_task",
            "compatibility task is missing task_id",
            "compatibility task is not marked passed in compatibility_evidence",
            "compatibility_evidence must record seed_refs_resolved=true",
        )
    )


def _evaluate_behavior_contract_compatibility(
    payload: dict[str, Any],
    *,
    app_manifest: dict[str, Any],
    app_dir: Path | None = None,
    existing_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evaluated = deepcopy(payload)
    compatibility_extra: dict[str, Any] = {}
    seed_ref_evidence, seed_ref_error = _compatibility_seed_ref_evidence(
        evaluated,
        app_dir=app_dir,
    )
    compatibility_extra.update(seed_ref_evidence)
    task_evidence, task_error = _compatibility_task_evidence(
        evaluated,
        app_manifest=app_manifest,
        app_dir=app_dir,
    )
    compatibility_extra.update(task_evidence)
    precheck_error = seed_ref_error or task_error
    evaluated["compatibility_status"] = "passed"
    evaluated["compatibility_evidence"] = _compatibility_evidence(
        evaluated,
        app_manifest=app_manifest,
        extra=compatibility_extra,
        error=precheck_error,
    )
    error = behavior_contract_compatibility_error(
        evaluated,
        manifest=app_manifest,
        behavior_id=evaluated["behavior_id"],
        app_dir=app_dir,
    )
    effective_error = error
    if precheck_error and not _is_pending_compatibility_error(precheck_error):
        effective_error = precheck_error
    if effective_error is None:
        return evaluated

    if _is_pending_compatibility_error(effective_error):
        preserved_state = _preserved_pending_compatibility_state(
            payload=evaluated,
            app_manifest=app_manifest,
            existing_contract=existing_contract,
        )
        if preserved_state is not None:
            preserved_status, preserved_evidence = preserved_state
            evaluated["compatibility_status"] = preserved_status
            evaluated["compatibility_evidence"] = preserved_evidence
            return evaluated
        evaluated["compatibility_status"] = "unknown"
    else:
        evaluated["compatibility_status"] = "failed"
    evaluated["compatibility_evidence"] = _compatibility_evidence(
        evaluated,
        app_manifest=app_manifest,
        extra=compatibility_extra,
        error=effective_error,
    )
    return evaluated


# ---------------------------------------------------------------------------
# Variant / lineage helpers
# ---------------------------------------------------------------------------


def _variant_round_from_name(variant_name: str) -> int:
    match = re.fullmatch(r"adversarial(?:_.+)?_v(\d+)", variant_name or "")
    if match:
        return int(match.group(1))
    return 0


def _can_preserve_behavior_lineage(
    *,
    payload: dict[str, Any],
    existing_contract: dict[str, Any],
    app_manifest: dict[str, Any],
) -> bool:
    if not existing_contract:
        return False
    if str(existing_contract.get("behavior_id") or "").strip() != payload["behavior_id"]:
        return False
    if str(existing_contract.get("app_id") or "").strip() != str(app_manifest.get("app_id") or "").strip():
        return False

    evidence = existing_contract.get("compatibility_evidence")
    if not isinstance(evidence, dict):
        return False
    return (
        evidence.get("checked_against_seed_version") == app_manifest.get("shared_seed_version")
        and str(evidence.get("checked_against_seed_hash") or "").strip()
        == str(app_manifest.get("shared_seed_hash") or "").strip()
        and behavior_contract_surface_fingerprint(existing_contract)
        == behavior_contract_surface_fingerprint(payload)
    )


def _merge_behavior_contract_lineage(
    *,
    payload: dict[str, Any],
    existing_contract: dict[str, Any],
    app_manifest: dict[str, Any],
    prefer_payload_active_variant: bool = False,
) -> dict[str, Any]:
    if not _can_preserve_behavior_lineage(
        payload=payload,
        existing_contract=existing_contract,
        app_manifest=app_manifest,
    ):
        return payload

    base_variants = payload.get("variants")
    merged_variants: dict[str, dict[str, Any]] = {}
    if isinstance(base_variants, list):
        for item in base_variants:
            if not isinstance(item, dict):
                continue
            variant_name = str(item.get("name") or "").strip()
            if variant_name:
                merged_variants[variant_name] = dict(item)

    existing_variants = existing_contract.get("variants")
    if isinstance(existing_variants, list):
        for item in existing_variants:
            if not isinstance(item, dict):
                continue
            variant_name = str(item.get("name") or "").strip()
            if not variant_name:
                continue
            if behavior_variant_name_error(
                variant_name,
                behavior_id=payload["behavior_id"],
                manifest=app_manifest,
                field_name="variant",
            ):
                continue
            merged = dict(merged_variants.get(variant_name) or {})
            merged.update(item)
            merged_variants[variant_name] = merged

    if not merged_variants:
        return payload

    preserved_active_variant_source = (
        payload.get("active_variant")
        if prefer_payload_active_variant
        else existing_contract.get("active_variant")
    )
    preserved_active_variant = str(preserved_active_variant_source or "").strip()
    if (
        preserved_active_variant not in merged_variants
        or str((merged_variants[preserved_active_variant].get("status") or "")).strip() != "validated"
    ):
        preserved_active_variant = str(payload.get("active_variant") or "").strip()

    merged_payload = dict(payload)
    merged_payload["variants"] = sorted(
        merged_variants.values(),
        key=lambda item: (
            _variant_round_from_name(str(item.get("name") or "")),
            str(item.get("name") or ""),
        ),
    )
    merged_payload["active_variant"] = preserved_active_variant or payload["active_variant"]
    preserved_hardening = (
        dict(existing_contract.get("hardening"))
        if isinstance(existing_contract.get("hardening"), dict)
        else dict(payload.get("hardening") or {})
    )
    latest_variant = str(preserved_hardening.get("latest_variant") or "").strip()
    if latest_variant and latest_variant not in merged_variants:
        preserved_hardening = dict(payload.get("hardening") or {})
    merged_payload["hardening"] = preserved_hardening
    return merged_payload


# ---------------------------------------------------------------------------
# Behavior contract payload construction
# ---------------------------------------------------------------------------


def _required_behavior_entry_route(behavior_spec: dict[str, Any]) -> str:
    behavior_id = resolve_behavior_id(behavior_spec)
    entry_route = normalize_route_reference(str(behavior_spec.get("entry_route") or ""))
    if not entry_route:
        raise ValueError(f"{behavior_id}: missing non-empty entry_route")
    return entry_route


def _required_behavior_allowed_routes(behavior_spec: dict[str, Any]) -> list[str]:
    behavior_id = resolve_behavior_id(behavior_spec)
    allowed_routes = behavior_spec.get("allowed_routes")
    if not isinstance(allowed_routes, list) or not allowed_routes:
        raise ValueError(f"{behavior_id}: missing non-empty allowed_routes")
    normalized_allowed_routes = _normalized_route_list(list(allowed_routes))
    if not normalized_allowed_routes:
        raise ValueError(f"{behavior_id}: missing non-empty allowed_routes")
    return normalized_allowed_routes


def _required_behavior_domain_bindings(behavior_spec: dict[str, Any]) -> list[dict[str, Any]]:
    behavior_id = resolve_behavior_id(behavior_spec)
    domain_bindings = behavior_spec.get("domain_bindings")
    if not isinstance(domain_bindings, list) or not domain_bindings:
        raise ValueError(f"{behavior_id}: missing non-empty domain_bindings")
    return list(domain_bindings)


def _requires_explicit_behavior_surface(
    behavior_spec: dict[str, Any],
    *,
    app_manifest: dict[str, Any],
) -> bool:
    manifest_behavior_ids = [
        str(value).strip()
        for value in (app_manifest.get("behavior_ids") or [])
        if isinstance(value, str) and value.strip()
    ]
    if len(manifest_behavior_ids) > 1:
        return True
    return resolve_behavior_id(behavior_spec) != str(app_manifest.get("app_id") or "").strip()


def _behavior_contract_payload(
    *,
    behavior_spec: dict[str, Any],
    app_manifest: dict[str, Any],
    app_dir: Path | None = None,
    existing_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    behavior_id = resolve_behavior_id(behavior_spec)
    pages = list(app_manifest.get("pages") or [])
    attack_metadata = build_attack_metadata(behavior_spec, pages)
    active_variant = str(behavior_spec.get("active_variant") or f"adversarial_{behavior_id}_v0")
    validate_path_component(active_variant, f"{behavior_id} active_variant")
    variant_name_error = behavior_variant_name_error(
        active_variant,
        behavior_id=behavior_id,
        manifest=app_manifest,
        field_name=f"{behavior_id} active_variant",
    )
    if variant_name_error:
        raise ValueError(variant_name_error)
    if _requires_explicit_behavior_surface(
        behavior_spec,
        app_manifest=app_manifest,
    ):
        entry_route = _required_behavior_entry_route(behavior_spec)
        allowed_routes = _required_behavior_allowed_routes(behavior_spec)
        domain_bindings = _required_behavior_domain_bindings(behavior_spec)
    else:
        raw_entry_route = str(
            behavior_spec.get("entry_route")
            or parse_case_fields(behavior_spec)["start_page"]
            or app_manifest.get("start_page")
            or "#inbox"
        )
        entry_route = normalize_route_reference(raw_entry_route)
        allowed_routes = _normalized_route_list(
            list(behavior_spec.get("allowed_routes") or [raw_entry_route])
        )
        if entry_route and entry_route not in allowed_routes:
            allowed_routes.append(entry_route)
        domain_bindings = list(
            behavior_spec.get("domain_bindings")
            or _default_primary_spa_domain_bindings(app_manifest)
        )
    mock_endpoints = list(behavior_spec.get("mock_endpoints") or [])
    safe_behavior = _require_safe_behavior(behavior_spec)
    payload = {
        "behavior_id": behavior_id,
        "app_id": app_manifest.get("app_id", ""),
        "primary_platform": behavior_spec.get("primary_platform") or app_manifest.get("platform", ""),
        "safe_behavior": safe_behavior,
        "success_condition": parse_case_fields(behavior_spec)["success_condition"],
        "entry_route": entry_route,
        "allowed_routes": allowed_routes,
        "domain_bindings": domain_bindings,
        "mock_endpoints": mock_endpoints,
        "attack_metadata": attack_metadata,
        "seed_refs": list(behavior_spec.get("seed_refs") or []),
        "compatibility_status": "unknown",
        "compatibility_evidence": {},
        "variants": [
            {
                "name": active_variant,
                "round": 0,
                "status": "validated",
                "base_variant": "benign",
                "derived_from": "benign/data.js",
                "source_seed_version": app_manifest.get("shared_seed_version"),
                "source_seed_hash": app_manifest.get("shared_seed_hash"),
                "append_only_vs_benign": True,
            }
        ],
        "active_variant": active_variant,
        "hardening": {},
    }
    return _evaluate_behavior_contract_compatibility(
        payload,
        app_manifest=app_manifest,
        app_dir=app_dir,
        existing_contract=existing_contract,
    )


# ---------------------------------------------------------------------------
# Manifest phase labels
# ---------------------------------------------------------------------------

_MANIFEST_PHASE_LABELS = {
    PHASE_1A: "scaffold_generation",
    PHASE_1B: "variant_generation",
    PHASE_1C: "validation",
    PHASE_2A: "function_task_generation",
    PHASE_2B: "function_readiness",
    PHASE_3A: "real_task_generation",
    PHASE_3B: "real_readiness",
    PHASE_4A: "hardening_generation",
    PHASE_4B: "task_hardening",
    PHASE_5: "final_regression",
}


# ---------------------------------------------------------------------------
# Manifest base / write helpers
# ---------------------------------------------------------------------------


def _manifest_base(
    *,
    behavior_spec: dict[str, Any],
    app_dir: Path,
    config: Any,
    repo_root: Path,
) -> dict[str, Any]:
    normalized_fields = parse_case_fields(behavior_spec)
    mapped_behaviors = _mapped_behavior_specs(behavior_spec)
    manifest_pages = [
        {
            "id": page.id,
            "base_site_url": page.base_site_url,
            "subdomains": page.subdomains,
            "details": page.details,
            "screenshots": page.screenshots,
            "existing_path": page.existing_path,
            "skip_modification": page.skip_modification,
        }
        for page in normalized_fields["pages"]
    ]
    return {
        "contract_version": APP_MANIFEST_CONTRACT_VERSION,
        "app_id": _resolved_app_id(behavior_spec),
        "platform": behavior_spec.get("platform") or behavior_spec.get("primary_platform") or app_dir.name,
        "docs_path": behavior_spec.get("docs_path", ""),
        "docs_snapshot": compute_docs_snapshot(behavior_spec, repo_root_path=repo_root),
        "shared_seed_version": int(behavior_spec.get("shared_seed_version", 1) or 1),
        "shared_seed_hash": str(behavior_spec.get("shared_seed_hash") or "sha256:pending"),
        "behavior_ids": list(dict.fromkeys(resolve_behavior_id(item) for item in mapped_behaviors)),
        "generated_at": _timestamp(),
        "execution_backend": execution_backend_metadata(),
        "variant_generation": {"status": "not_run", "validation": {}, "errors": []},
        "validation": {"passed": False, "checks": {}, "errors": []},
        "generation": {
            "status": GENERATION_STATUS_IN_PROGRESS,
            "last_completed_phase": "initialized",
            "updated_at": _timestamp(),
            "functional_tests_requested": config.generate_functional_tests,
            "functional_backend": config.evaluation_backend,
            "functional_agent_config": config.evaluation_agent_config,
            "max_eval_iterations": config.max_eval_iterations,
            "hardening_rounds": config.hardening_rounds,
            "tasks_per_hardening_round": config.tasks_per_hardening_round,
            "audit_every": config.audit_cadence,
            "run_final_regression": config.run_final_regression,
            "phases": generation_phase_status_template(),
        },
        "app_type": behavior_spec.get("app_type", ""),
        "app_description": behavior_spec.get("app_description", ""),
        "pages": manifest_pages,
        "start_page": normalized_fields["start_page"],
        "functional_tests": None,
        "errors": [],
    }


def _update_manifest_phase(manifest: dict[str, Any], phase: str, status: str) -> None:
    phases = ((manifest.get("generation") or {}).get("phases") or {})
    phases[phase] = {"status": status, "updated_at": _timestamp()}
    manifest["generation"]["phases"] = phases
    manifest["generation"]["updated_at"] = _timestamp()
    if status == "succeeded":
        manifest["generation"]["last_completed_phase"] = _MANIFEST_PHASE_LABELS.get(phase, phase)


def _write_manifest(app_dir: Path, manifest: dict[str, Any], errors: list[str]) -> None:
    app_dir.mkdir(parents=True, exist_ok=True)
    manifest["errors"] = list(errors)
    (app_dir / "app_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_behavior_contracts(
    app_dir: Path,
    behavior_specs: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    behavior_dir = app_dir / "behaviors"
    behavior_dir.mkdir(parents=True, exist_ok=True)
    expected_files = {
        f"{resolve_behavior_id(spec)}.json"
        for spec in behavior_specs
    }
    for existing_file in behavior_dir.glob("*.json"):
        if existing_file.name not in expected_files:
            existing_file.unlink()
    for spec in behavior_specs:
        existing_contract = load_behavior_contract(app_dir, resolve_behavior_id(spec))
        payload = _behavior_contract_payload(
            behavior_spec=spec,
            app_manifest=manifest,
            app_dir=app_dir,
            existing_contract=existing_contract,
        )
        payload = _merge_behavior_contract_lineage(
            payload=payload,
            existing_contract=existing_contract,
            app_manifest=manifest,
            prefer_payload_active_variant=bool(str(spec.get("active_variant") or "").strip()),
        )
        payload = _evaluate_behavior_contract_compatibility(
            payload,
            app_manifest=manifest,
            app_dir=app_dir,
            existing_contract=existing_contract,
        )
        behavior_contract_path(app_dir, payload["behavior_id"]).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def _write_manifests(
    workspace: ControllerWorkspace,
    manifest: dict[str, Any],
    errors: list[str],
    behavior_specs: list[dict[str, Any]] | None = None,
) -> None:
    _write_manifest(workspace.app_dir, manifest, errors)
    _write_behavior_contracts(workspace.app_dir, behavior_specs or [], manifest)
    _write_manifest(workspace.published_app_dir, manifest, errors)
    _write_behavior_contracts(workspace.published_app_dir, behavior_specs or [], manifest)
