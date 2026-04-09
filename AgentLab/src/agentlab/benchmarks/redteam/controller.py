"""Top-level controller for redteam app generation."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentlab.benchmarks.redteam.app_artifacts import (
    APP_MANIFEST_CONTRACT_VERSION,
    GENERATION_STATUS_FAILED,
    GENERATION_STATUS_IN_PROGRESS,
    GENERATION_STATUS_SUCCEEDED,
    benchmark_ready_app_dir_error,
    behavior_contract_surface_fingerprint,
    behavior_contract_path,
    behavior_variant_name_error,
    build_attack_metadata,
    compute_docs_snapshot,
    docs_snapshot_mismatch_error,
    functional_quality_gate_passed,
    functional_tests_complete,
    load_app_manifest,
    load_behavior_contract,
    real_task_quality_gate_passed,
    validate_path_component,
)
from agentlab.benchmarks.redteam.audit_runner import run_audit_step
from agentlab.benchmarks.redteam.behavior_ids import resolve_behavior_id
from agentlab.benchmarks.redteam.config import parse_case_fields
from agentlab.benchmarks.redteam.controller_state import (
    controller_events_path,
    controller_state_path,
    generation_phase_status_template,
)
from agentlab.benchmarks.redteam.execution import execution_backend_metadata
from agentlab.benchmarks.redteam.git_ops import (
    ControllerWorkspace,
    GitControllerError,
    checkpoint_scope,
    current_head,
    ensure_controller_workspace,
    normalize_output_dir,
    publish_controller_workspace,
    reset_hard,
)
from agentlab.benchmarks.redteam.phase_ids import (
    CANONICAL_PHASES,
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
    PHASE_COMPLETED,
    normalize_phase_id,
)
from agentlab.benchmarks.redteam.runtime_ops import materialize_app_runtime
from agentlab.benchmarks.redteam.utils import normalize_route_reference


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _update_manifest_seed_metadata(
    manifest: dict[str, Any],
    *,
    app_dir: Path,
) -> str | None:
    benign_seed = app_dir / "benign" / "data.js"
    if not benign_seed.exists():
        return "benign/data.js missing; cannot compute shared seed metadata"
    shared_seed_hash = f"sha256:{_sha256_file(benign_seed)}"
    previous_hash = str(manifest.get("shared_seed_hash") or "").strip()
    previous_version = manifest.get("shared_seed_version")
    normalized_previous_version = previous_version if isinstance(previous_version, int) and previous_version > 0 else 1
    if previous_hash and previous_hash not in {"", "sha256:pending"} and previous_hash != shared_seed_hash:
        manifest["shared_seed_version"] = normalized_previous_version + 1
    else:
        manifest["shared_seed_version"] = normalized_previous_version
    manifest["shared_seed_hash"] = shared_seed_hash
    return None


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


def _behavior_contract_payload(
    *,
    behavior_spec: dict[str, Any],
    app_manifest: dict[str, Any],
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
    raw_entry_route = str(
        behavior_spec.get("entry_route")
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
        "compatibility_status": "passed",
        "compatibility_evidence": {
            "checked_against_seed_version": app_manifest.get("shared_seed_version"),
            "checked_against_seed_hash": app_manifest.get("shared_seed_hash"),
            "checked_entry_route": entry_route,
        },
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
    payload["compatibility_evidence"]["checked_behavior_fingerprint"] = (
        behavior_contract_surface_fingerprint(payload)
    )
    return payload


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

_PHASE_ORDER = [
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
]


@dataclass
class ControllerConfig:
    design_guides_dir: str | None = None
    template_dir: str | None = None
    evaluation_backend: str = "agent-browser"
    evaluation_agent_config: str | None = None
    workers: int = 8
    repetitions: int = 3
    max_eval_iterations: int = 3
    hardening_rounds: int = 3
    tasks_per_hardening_round: int = 20
    audit_cadence: int = 0
    generate_functional_tests: bool = True
    run_final_regression: bool = True
    max_total_controller_iterations: int | None = None

    def requested_repetitions(self) -> int:
        return self.repetitions

    def effective_repetitions(self) -> int:
        return 1 if self.evaluation_backend == "agent-browser" else self.repetitions


@dataclass
class ControllerState:
    behavior_id: str
    current_phase: str
    branch: str
    worktree_path: str
    owned_paths: list[str]
    phase_statuses: dict[str, dict[str, Any]]
    base_commit: str = ""
    phase_checkpoint_commits: dict[str, str] = field(default_factory=dict)
    last_good_commit: str = ""
    diagnostic_commit: str = ""
    current_iteration: int = 0
    stop_reason: str = ""
    phase_attempt_counters: dict[str, int] = field(default_factory=dict)
    evaluation_config: dict[str, Any] = field(default_factory=dict)
    authoring_backend: dict[str, Any] = field(default_factory=dict)
    artifact_contract_version: int = APP_MANIFEST_CONTRACT_VERSION
    total_eval_audit_iterations: int = 0
    total_hardening_rounds: int = 0
    total_accepted_repairs: int = 0
    updated_at: str = field(default_factory=_timestamp)


@dataclass
class PhaseOutcome:
    phase_id: str
    status: str
    checkpoint_commit: str | None = None
    stop_reason: str = ""
    primary_artifacts: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class ControllerResult:
    app_dir: str
    behavior_id: str
    manifest: dict[str, Any]
    controller_state_path: str
    events_path: str
    phase_outcomes: list[PhaseOutcome]
    errors: list[str] = field(default_factory=list)


def _write_json(path: str | Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_controller_state(logs_dir: str | Path) -> dict[str, Any]:
    path = controller_state_path(logs_dir)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def append_controller_event(logs_dir: str | Path, payload: dict[str, Any]) -> None:
    path = controller_events_path(logs_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {"timestamp": _timestamp(), **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def write_controller_state(logs_dir: str | Path, state: ControllerState) -> None:
    state.updated_at = _timestamp()
    _write_json(controller_state_path(logs_dir), asdict(state))


def _manifest_base(
    *,
    behavior_spec: dict[str, Any],
    app_dir: Path,
    config: ControllerConfig,
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
        payload = _behavior_contract_payload(
            behavior_spec=spec,
            app_manifest=manifest,
        )
        existing_contract = load_behavior_contract(app_dir, payload["behavior_id"])
        payload = _merge_behavior_contract_lineage(
            payload=payload,
            existing_contract=existing_contract,
            app_manifest=manifest,
            prefer_payload_active_variant=bool(str(spec.get("active_variant") or "").strip()),
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


def _publish_workspace(workspace: ControllerWorkspace) -> None:
    if workspace.app_dir.exists():
        publish_controller_workspace(workspace)


_HELPER_PHASES = {PHASE_2B, PHASE_3B, PHASE_4B, PHASE_5}


def _numeric_pass_rate(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _real_readiness_gate(result: dict[str, Any], *, threshold: float) -> tuple[bool, float | None]:
    pass_rate = _numeric_pass_rate(result.get("pass_rate"))
    return (
        bool(result.get("ran"))
        and result.get("error") in (None, "")
        and pass_rate is not None
        and pass_rate >= threshold,
        pass_rate,
    )


def _checkpoint_phase(
    workspace: ControllerWorkspace,
    state: ControllerState,
    phase: str,
    *,
    message_suffix: str,
    allow_empty: bool = False,
    record_as_good: bool = True,
) -> str | None:
    commit = checkpoint_scope(
        workspace.worktree_path,
        scope_path=workspace.app_dir,
        message=f"controller({workspace.app_dir.name}): {message_suffix}",
    )
    if commit and record_as_good:
        state.last_good_commit = commit
        state.phase_checkpoint_commits[phase] = commit
    elif allow_empty:
        commit = current_head(workspace.worktree_path)
        if record_as_good:
            state.last_good_commit = commit
            state.phase_checkpoint_commits[phase] = commit
    return commit


def _effective_repetitions(config: ControllerConfig) -> int:
    return config.effective_repetitions()


def _coerce_controller_state(
    payload: dict[str, Any],
    *,
    workspace: ControllerWorkspace,
    config: ControllerConfig,
) -> ControllerState:
    if not payload:
        return _build_initial_state(workspace, config)

    if "phase_statuses" in payload:
        allowed_fields = {field.name for field in ControllerState.__dataclass_fields__.values()}
        sanitized_payload = {key: value for key, value in payload.items() if key in allowed_fields}
        sanitized_payload["current_phase"] = normalize_phase_id(sanitized_payload.get("current_phase"))
        sanitized_payload["phase_checkpoint_commits"] = {
            normalize_phase_id(key): str(value)
            for key, value in dict(sanitized_payload.get("phase_checkpoint_commits") or {}).items()
            if str(value)
        }
        return ControllerState(**sanitized_payload)

    raise ValueError(
        "Unsupported legacy controller state artifact. Regenerate or rerun with the current app pipeline."
    )



def _mark_remaining_phases_skipped(
    manifest: dict[str, Any],
    state: ControllerState,
    phase_order: list[str],
    from_phase: str,
) -> None:
    for remaining_phase in phase_order[phase_order.index(from_phase) + 1 :]:
        _update_manifest_phase(manifest, remaining_phase, "skipped")
        state.phase_statuses[remaining_phase] = {"status": "skipped", "updated_at": _timestamp()}


def _update_state_for_phase(
    state: ControllerState,
    *,
    phase: str,
    status: str,
    iteration: int | None = None,
    stop_reason: str = "",
) -> None:
    state.current_phase = phase
    state.phase_statuses.setdefault(phase, {"status": "pending", "updated_at": None})
    state.phase_statuses[phase] = {"status": status, "updated_at": _timestamp()}
    if iteration is not None:
        state.current_iteration = iteration
        state.phase_attempt_counters[phase] = iteration
    if stop_reason:
        state.stop_reason = stop_reason


def _build_initial_state(workspace: ControllerWorkspace, config: ControllerConfig) -> ControllerState:
    base_commit = current_head(workspace.worktree_path)
    return ControllerState(
        behavior_id=workspace.raw_behavior_id,
        current_phase=PHASE_1A,
        branch=workspace.branch,
        worktree_path=str(workspace.worktree_path),
        owned_paths=[str(workspace.published_app_dir), str(workspace.logs_dir)],
        phase_statuses=generation_phase_status_template(),
        base_commit=base_commit,
        last_good_commit=base_commit,
        evaluation_config={
            "requested_workers": config.workers,
            "requested_repetitions": config.requested_repetitions(),
            "effective_workers": config.workers,
            "effective_repetitions": _effective_repetitions(config),
            "max_eval_iterations": config.max_eval_iterations,
            "hardening_rounds": config.hardening_rounds,
            "tasks_per_hardening_round": config.tasks_per_hardening_round,
            "audit_cadence": config.audit_cadence,
            "max_total_controller_iterations": config.max_total_controller_iterations
            or (2 * config.max_eval_iterations + config.hardening_rounds),
        },
        authoring_backend=execution_backend_metadata(),
    )


def _phase_rewind_commit(state: ControllerState, target_phase: str) -> str:
    target_phase = normalize_phase_id(target_phase)
    if target_phase == PHASE_1A:
        commit = state.base_commit
    else:
        try:
            target_index = _PHASE_ORDER.index(target_phase)
        except ValueError as exc:
            raise RuntimeError(f"Unknown rerun phase: {target_phase}") from exc
        commit = ""
        for phase in reversed(_PHASE_ORDER[:target_index]):
            commit = str((state.phase_checkpoint_commits or {}).get(phase) or "")
            if commit:
                break
    if not commit:
        raise RuntimeError(
            f"Cannot rerun from {target_phase}: missing checkpoint boundary for this controller state"
        )
    return commit


def _enforce_budget(state: ControllerState) -> None:
    budget = int((state.evaluation_config or {}).get("max_total_controller_iterations") or 0)
    used = int(state.total_eval_audit_iterations + state.total_hardening_rounds)
    if budget > 0 and used > budget:
        raise RuntimeError("global_budget_exhausted")


def _prepare_workspace_for_resume(
    workspace: ControllerWorkspace,
    state: ControllerState,
    *,
    rerun_from_phase: str | None = None,
) -> bool:
    last_good_commit = str(state.last_good_commit or "")
    current_phase = normalize_phase_id(state.current_phase)
    phase_status = str((state.phase_statuses.get(current_phase) or {}).get("status") or "")
    preserving_in_progress_helper_state = (
        rerun_from_phase is None
        and current_phase in _HELPER_PHASES
        and phase_status == "in_progress"
    )
    if rerun_from_phase is not None:
        reset_hard(workspace.worktree_path, _phase_rewind_commit(state, rerun_from_phase))
        return True
    if current_phase == PHASE_COMPLETED:
        return False
    if last_good_commit and not preserving_in_progress_helper_state:
        reset_hard(workspace.worktree_path, last_good_commit)
        return True
    return False


def _run_core_generation(
    *,
    behavior_spec: dict[str, Any],
    workspace: ControllerWorkspace,
    config: ControllerConfig,
    resume: bool,
    rerun_from_phase: str | None = None,
) -> ControllerResult:
    from agentlab.benchmarks.redteam import app_pipeline
    from agentlab.benchmarks.redteam.eval_harness import write_functional_results
    from agentlab.benchmarks.redteam.variant_ops import generate_variants_result
    from agentlab.benchmarks.redteam.validation import validate_runtime

    app_dir = workspace.app_dir
    published_app_dir = workspace.published_app_dir
    logs_dir = workspace.logs_dir
    mapped_behavior_specs = _mapped_behavior_specs(behavior_spec)
    app_dir.parent.mkdir(parents=True, exist_ok=True)
    app_dir.mkdir(parents=True, exist_ok=True)
    published_app_dir.parent.mkdir(parents=True, exist_ok=True)

    existing_manifest = load_app_manifest(published_app_dir) if published_app_dir.exists() else {}
    manifest = existing_manifest or _manifest_base(
        behavior_spec=behavior_spec,
        app_dir=published_app_dir,
        config=config,
        repo_root=workspace.repo_root,
    )
    docs_snapshot_error = docs_snapshot_mismatch_error(
        manifest=manifest,
        behavior_spec=behavior_spec,
        repo_root_path=workspace.repo_root,
    )
    if docs_snapshot_error:
        errors = [docs_snapshot_error]
        manifest["generation"]["status"] = GENERATION_STATUS_FAILED
        state = _coerce_controller_state(
            load_controller_state(logs_dir),
            workspace=workspace,
            config=config,
        )
        state.stop_reason = "docs_snapshot_mismatch"
        write_controller_state(logs_dir, state)
        _write_manifests(workspace, manifest, errors, mapped_behavior_specs)
        return ControllerResult(
            app_dir=str(published_app_dir),
            behavior_id=workspace.raw_behavior_id,
            manifest=manifest,
            controller_state_path=str(controller_state_path(logs_dir)),
            events_path=str(controller_events_path(logs_dir)),
            phase_outcomes=[],
            errors=errors,
        )
    if str(manifest.get("shared_seed_hash") or "").strip() == "sha256:pending":
        _update_manifest_seed_metadata(manifest, app_dir=published_app_dir)
    phase_outcomes: list[PhaseOutcome] = []
    prior_errors = list(existing_manifest.get("errors") or [])
    errors: list[str] = []

    state = _coerce_controller_state(
        load_controller_state(logs_dir),
        workspace=workspace,
        config=config,
    )
    if prior_errors and not (
        resume and rerun_from_phase is None and normalize_phase_id(state.current_phase) == PHASE_COMPLETED
    ):
        append_controller_event(
            logs_dir,
            {
                "phase": normalize_phase_id(state.current_phase),
                "action": "discard_prior_manifest_errors",
                "errors": prior_errors,
            },
        )
    if resume:
        if rerun_from_phase is None and normalize_phase_id(state.current_phase) == PHASE_COMPLETED:
            return ControllerResult(
                app_dir=str(published_app_dir),
                behavior_id=workspace.raw_behavior_id,
                manifest=manifest,
                controller_state_path=str(controller_state_path(logs_dir)),
                events_path=str(controller_events_path(logs_dir)),
                phase_outcomes=[],
                errors=list(existing_manifest.get("errors") or []),
            )
        try:
            resume_pipeline_state = app_pipeline.load_pipeline_state(
                app_dir,
                logs_dir=logs_dir,
                strict=True,
            )
        except RuntimeError as exc:
            backend_error = str(exc)
        else:
            backend_error = app_pipeline._resume_backend_error(
                requested_backend=config.evaluation_backend,
                manifest=manifest,
                pipeline_state=resume_pipeline_state,
            )
        if backend_error:
            errors.append(backend_error)
            manifest["generation"]["status"] = GENERATION_STATUS_FAILED
            state.stop_reason = "generation_failed"
            write_controller_state(logs_dir, state)
            _write_manifests(workspace, manifest, errors, mapped_behavior_specs)
            return ControllerResult(
                app_dir=str(published_app_dir),
                behavior_id=workspace.raw_behavior_id,
                manifest=manifest,
                controller_state_path=str(controller_state_path(logs_dir)),
                events_path=str(controller_events_path(logs_dir)),
                phase_outcomes=[],
                errors=errors,
            )
        reset_applied = _prepare_workspace_for_resume(
            workspace,
            state,
            rerun_from_phase=rerun_from_phase,
        )
        if reset_applied:
            _publish_workspace(workspace)
            manifest = (
                load_app_manifest(app_dir)
                or load_app_manifest(published_app_dir)
                or _manifest_base(
                    behavior_spec=behavior_spec,
                    app_dir=published_app_dir,
                    config=config,
                    repo_root=workspace.repo_root,
                )
            )
    if rerun_from_phase:
        state.current_phase = normalize_phase_id(rerun_from_phase)
        for phase in _PHASE_ORDER[_PHASE_ORDER.index(state.current_phase) :]:
            state.phase_statuses[phase] = {"status": "pending", "updated_at": None}
            state.phase_checkpoint_commits.pop(phase, None)

    write_controller_state(logs_dir, state)
    _write_manifests(workspace, manifest, errors, mapped_behavior_specs)

    phase_order = _PHASE_ORDER
    start_index = phase_order.index(normalize_phase_id(state.current_phase)) if normalize_phase_id(state.current_phase) in phase_order else 0

    try:
        for phase in phase_order[start_index:]:
            if not config.generate_functional_tests and phase not in {PHASE_1A, PHASE_1B, PHASE_1C}:
                _update_manifest_phase(manifest, phase, "skipped")
                state.phase_statuses[phase] = {"status": "skipped", "updated_at": _timestamp()}
                _checkpoint_phase(
                    workspace,
                    state,
                    phase,
                    message_suffix=f"skip {phase}",
                    allow_empty=True,
                )
                _publish_workspace(workspace)
                _write_manifests(workspace, manifest, errors, mapped_behavior_specs)
                write_controller_state(logs_dir, state)
                continue
            if config.hardening_rounds <= 0 and phase in {PHASE_4A, PHASE_4B}:
                _update_manifest_phase(manifest, phase, "skipped")
                state.phase_statuses[phase] = {"status": "skipped", "updated_at": _timestamp()}
                _checkpoint_phase(
                    workspace,
                    state,
                    phase,
                    message_suffix=f"skip {phase}",
                    allow_empty=True,
                )
                _publish_workspace(workspace)
                _write_manifests(workspace, manifest, errors, mapped_behavior_specs)
                write_controller_state(logs_dir, state)
                continue
            if not config.run_final_regression and phase == PHASE_5:
                _update_manifest_phase(manifest, phase, "skipped")
                state.phase_statuses[phase] = {"status": "skipped", "updated_at": _timestamp()}
                _checkpoint_phase(
                    workspace,
                    state,
                    phase,
                    message_suffix=f"skip {phase}",
                    allow_empty=True,
                )
                _publish_workspace(workspace)
                _write_manifests(workspace, manifest, errors, mapped_behavior_specs)
                write_controller_state(logs_dir, state)
                continue
            _enforce_budget(state)
            _update_state_for_phase(state, phase=phase, status="in_progress")
            write_controller_state(logs_dir, state)
            append_controller_event(logs_dir, {"phase": phase, "action": "start"})

            if phase == PHASE_1A:
                result = app_pipeline.generate_app_scaffold(
                    behavior_spec=behavior_spec,
                    app_dir=app_dir,
                    design_guides_dir=config.design_guides_dir,
                    repo_root_path=workspace.repo_root,
                    template_dir=config.template_dir,
                    timeout=600,
                )
                if result["errors"]:
                    errors.extend(result["errors"])
                status = "succeeded" if result["generated"] else "failed"
                _update_manifest_phase(manifest, phase, status)
                commit = (
                    _checkpoint_phase(workspace, state, phase, message_suffix=f"complete {phase}")
                    if status == "succeeded"
                    else None
                )
                phase_outcomes.append(PhaseOutcome(phase_id=phase, status=status, checkpoint_commit=commit))
            elif phase == PHASE_1B:
                variant_result = generate_variants_result(
                    app_dir=app_dir,
                    behavior_spec=behavior_spec,
                    design_guides_dir=config.design_guides_dir,
                )
                manifest["behavior_ids"] = [resolve_behavior_id(item) for item in mapped_behavior_specs]
                manifest["variant_generation"] = {
                    "status": variant_result.status,
                    "validation": variant_result.validation,
                    "errors": list(variant_result.errors),
                }
                seed_error = None
                if variant_result.errors:
                    errors.extend(variant_result.errors)
                else:
                    seed_error = _update_manifest_seed_metadata(manifest, app_dir=app_dir)
                    if seed_error:
                        errors.append(seed_error)
                status = "succeeded" if variant_result.status == "validated" and not seed_error else "failed"
                _update_manifest_phase(manifest, phase, status)
                commit = (
                    _checkpoint_phase(workspace, state, phase, message_suffix=f"complete {phase}")
                    if status == "succeeded"
                    else None
                )
                phase_outcomes.append(PhaseOutcome(phase_id=phase, status=status, checkpoint_commit=commit))
            elif phase == PHASE_1C:
                validation = validate_runtime(app_dir)
                manifest["validation"] = validation
                if validation.get("errors"):
                    errors.extend(validation["errors"])
                status = "succeeded" if validation.get("passed") else "failed"
                _update_manifest_phase(manifest, phase, status)
                commit = (
                    _checkpoint_phase(
                        workspace,
                        state,
                        phase,
                        message_suffix=f"complete {phase}",
                        allow_empty=True,
                    )
                    if status == "succeeded"
                    else None
                )
                phase_outcomes.append(
                    PhaseOutcome(
                        phase_id=phase,
                        status=status,
                        checkpoint_commit=commit,
                        primary_artifacts=[str(app_dir / "app_manifest.json")],
                    )
                )
            elif phase == PHASE_2A:
                result = app_pipeline.generate_task_suite(
                    app_dir,
                    behavior_id=workspace.raw_behavior_id,
                    suite="function",
                    template_dir=config.template_dir,
                )
                if result["errors"]:
                    errors.extend(result["errors"])
                if not isinstance(manifest.get("functional_tests"), dict):
                    manifest["functional_tests"] = {}
                manifest["functional_tests"].setdefault("suite_generation", {})
                manifest["functional_tests"]["suite_generation"]["function"] = result
                manifest["functional_tests"]["function_sanity_passed"] = bool(result.get("sanity_passed"))
                status = "succeeded" if result.get("sanity_passed") else "failed"
                _update_manifest_phase(manifest, phase, status)
                commit = (
                    _checkpoint_phase(workspace, state, phase, message_suffix=f"complete {phase}")
                    if status == "succeeded"
                    else None
                )
                phase_outcomes.append(PhaseOutcome(phase_id=phase, status=status, checkpoint_commit=commit))
            elif phase == PHASE_2B:
                runtime_dir = materialize_app_runtime(
                    app_dir,
                    variant_subdir="benign",
                    runtime_dir=app_dir / "results" / PHASE_2B / "runtime_benign",
                )
                result = app_pipeline.run_eval_audit_loop(
                    app_dir=app_dir,
                    suite="function",
                    backend=config.evaluation_backend,
                    agent_config=config.evaluation_agent_config,
                    max_iterations=config.max_eval_iterations,
                    threshold=app_pipeline.DEFAULT_FUNCTIONAL_THRESHOLD,
                    update_state=True,
                    logs_dir=logs_dir,
                    workers=config.workers,
                    repetitions=_effective_repetitions(config),
                    runtime_app_dir=runtime_dir,
                    runtime_variant_subdir="benign",
                )
                state.total_eval_audit_iterations += len(result.get("iterations") or [])
                if not isinstance(manifest.get("functional_tests"), dict):
                    manifest["functional_tests"] = {}
                manifest["functional_tests"]["function_evaluation"] = result
                manifest["functional_tests"]["quality_gate"] = {
                    "threshold": app_pipeline.DEFAULT_FUNCTIONAL_THRESHOLD,
                    "passed": isinstance(result.get("pass_rate"), (int, float))
                    and result["pass_rate"] >= app_pipeline.DEFAULT_FUNCTIONAL_THRESHOLD,
                    "pass_rate": result.get("pass_rate"),
                }
                write_functional_results(
                    app_dir,
                    function_results=list(result.get("results") or []),
                    backend=config.evaluation_backend,
                )
                audit_prompt = ""
                iterations = result.get("iterations") or []
                if iterations:
                    last_iter = iterations[-1]
                    audit_prompt = str(last_iter.get("repair_prompt_path") or "")
                    if last_iter.get("audit_summary_path"):
                        run_audit_step(
                            app_dir=app_dir,
                            phase=phase,
                            suite="function",
                            iteration=int(last_iter.get("iteration") or len(iterations)),
                            results_dir=last_iter.get("results_dir") or "",
                            backend=config.evaluation_backend,
                            agent_config=config.evaluation_agent_config,
                        )
                status = "succeeded" if result.get("ran") and not result.get("error") else "failed"
                _update_manifest_phase(manifest, phase, status)
                commit = (
                    _checkpoint_phase(workspace, state, phase, message_suffix=f"complete {phase}")
                    if status == "succeeded"
                    else None
                )
                phase_outcomes.append(PhaseOutcome(phase_id=phase, status=status, checkpoint_commit=commit, diagnostics={"repair_prompt_path": audit_prompt}))
            elif phase == PHASE_3A:
                result = app_pipeline.generate_task_suite(
                    app_dir,
                    behavior_id=workspace.raw_behavior_id,
                    suite="real",
                    template_dir=config.template_dir,
                )
                if result["errors"]:
                    errors.extend(result["errors"])
                if not isinstance(manifest.get("functional_tests"), dict):
                    manifest["functional_tests"] = {}
                manifest["functional_tests"].setdefault("suite_generation", {})
                manifest["functional_tests"]["suite_generation"]["real"] = result
                manifest["functional_tests"]["real_sanity_passed"] = bool(result.get("sanity_passed"))
                status = "succeeded" if result.get("sanity_passed") else "failed"
                _update_manifest_phase(manifest, phase, status)
                commit = (
                    _checkpoint_phase(workspace, state, phase, message_suffix=f"complete {phase}")
                    if status == "succeeded"
                    else None
                )
                phase_outcomes.append(PhaseOutcome(phase_id=phase, status=status, checkpoint_commit=commit))
            elif phase == PHASE_3B:
                runtime_dir = materialize_app_runtime(
                    app_dir,
                    variant_subdir="benign",
                    runtime_dir=app_dir / "results" / PHASE_3B / "runtime_benign",
                )
                result = app_pipeline.run_eval_audit_loop(
                    app_dir=app_dir,
                    suite="real",
                    backend=config.evaluation_backend,
                    agent_config=config.evaluation_agent_config,
                    max_iterations=config.max_eval_iterations,
                    threshold=app_pipeline.DEFAULT_REAL_TASK_THRESHOLD,
                    update_state=True,
                    logs_dir=logs_dir,
                    workers=config.workers,
                    repetitions=_effective_repetitions(config),
                    runtime_app_dir=runtime_dir,
                    runtime_variant_subdir="benign",
                )
                state.total_eval_audit_iterations += len(result.get("iterations") or [])
                if not isinstance(manifest.get("functional_tests"), dict):
                    manifest["functional_tests"] = {}
                manifest["functional_tests"]["real_evaluation"] = result
                write_functional_results(
                    app_dir,
                    real_results=list(result.get("results") or []),
                    backend=config.evaluation_backend,
                )
                real_gate_passed, real_pass_rate = _real_readiness_gate(
                    result,
                    threshold=app_pipeline.DEFAULT_REAL_TASK_THRESHOLD,
                )
                manifest["functional_tests"]["real_quality_gate"] = {
                    "threshold": app_pipeline.DEFAULT_REAL_TASK_THRESHOLD,
                    "passed": real_gate_passed,
                    "pass_rate": real_pass_rate,
                }
                if real_gate_passed:
                    baseline = app_pipeline._ensure_readiness_baseline(
                        app_dir,
                        backend=config.evaluation_backend,
                    )
                    app_pipeline._freeze_real_task_baseline(
                        app_dir,
                        baseline_results=list(result.get("results", [])),
                    )
                    manifest["functional_tests"]["readiness_baseline"] = baseline
                else:
                    if real_pass_rate is None:
                        errors.append("Real-task benign readiness gate did not produce a numeric pass rate.")
                    else:
                        errors.append(
                            "Real-task benign readiness gate failed: "
                            f"{real_pass_rate:.3f} < {app_pipeline.DEFAULT_REAL_TASK_THRESHOLD:.3f}."
                        )
                status = "succeeded" if real_gate_passed else "failed"
                _update_manifest_phase(manifest, phase, status)
                commit = (
                    _checkpoint_phase(workspace, state, phase, message_suffix=f"complete {phase}")
                    if status == "succeeded"
                    else None
                )
                phase_outcomes.append(PhaseOutcome(phase_id=phase, status=status, checkpoint_commit=commit))
            elif phase == PHASE_4A:
                if not isinstance(manifest.get("functional_tests"), dict):
                    manifest["functional_tests"] = {}
                baseline = app_pipeline._load_backend_readiness_baseline(
                    app_dir,
                    backend=config.evaluation_backend,
                )
                if not baseline:
                    real_evaluation = manifest["functional_tests"].get("real_evaluation") or {}
                    real_results = list(real_evaluation.get("results") or [])
                    if real_results:
                        baseline = app_pipeline._ensure_readiness_baseline(
                            app_dir,
                            backend=config.evaluation_backend,
                        )
                        app_pipeline._freeze_real_task_baseline(
                            app_dir,
                            baseline_results=real_results,
                        )
                if baseline:
                    manifest["functional_tests"]["readiness_baseline"] = baseline
                has_hardening_baseline = bool(app_pipeline._load_real_task_baseline_snapshot(app_dir))
                status = "succeeded" if has_hardening_baseline else "failed"
                if not has_hardening_baseline:
                    errors.append("Hardening baseline snapshot missing before phase 4B.")
                _update_manifest_phase(manifest, phase, status)
                commit = (
                    _checkpoint_phase(
                        workspace,
                        state,
                        phase,
                        message_suffix=f"complete {phase}",
                        allow_empty=(status == "skipped"),
                    )
                    if status != "failed"
                    else None
                )
                phase_outcomes.append(PhaseOutcome(phase_id=phase, status=status, checkpoint_commit=commit))
            elif phase == PHASE_4B:
                hardening = app_pipeline.run_hardening_rounds(
                    app_dir=app_dir,
                    behavior_id=workspace.raw_behavior_id,
                    template_dir=config.template_dir,
                    backend=config.evaluation_backend,
                    agent_config=config.evaluation_agent_config,
                    hardening_rounds=config.hardening_rounds,
                    tasks_per_hardening_round=config.tasks_per_hardening_round,
                    audit_every=config.audit_cadence,
                    update_state=True,
                    logs_dir=logs_dir,
                    phase_name=PHASE_4B,
                )
                state.total_hardening_rounds += len(hardening.get("rounds") or [])
                if not isinstance(manifest.get("functional_tests"), dict):
                    manifest["functional_tests"] = {}
                manifest["functional_tests"]["task_hardening"] = hardening
                status = "succeeded" if hardening.get("ran") and not hardening.get("error") else ("skipped" if config.hardening_rounds <= 0 else "failed")
                _update_manifest_phase(manifest, phase, status)
                commit = (
                    _checkpoint_phase(workspace, state, phase, message_suffix=f"complete {phase}", allow_empty=True)
                    if status != "failed"
                    else None
                )
                phase_outcomes.append(PhaseOutcome(phase_id=phase, status=status, checkpoint_commit=commit))
            elif phase == PHASE_5:
                baseline_results = {
                    "function": list(((manifest.get("functional_tests") or {}).get("function_evaluation") or {}).get("results", [])),
                    "real": list(((manifest.get("functional_tests") or {}).get("real_evaluation") or {}).get("results", [])),
                }
                regression = app_pipeline.run_final_regression_eval(
                    app_dir=app_dir,
                    behavior_id=workspace.raw_behavior_id,
                    backend=config.evaluation_backend,
                    agent_config=config.evaluation_agent_config,
                    baseline_results=baseline_results,
                    update_state=True,
                    logs_dir=logs_dir,
                )
                if not isinstance(manifest.get("functional_tests"), dict):
                    manifest["functional_tests"] = {}
                manifest["functional_tests"]["final_regression"] = regression
                status = "succeeded" if regression.get("passed") else "failed"
                _update_manifest_phase(manifest, phase, status)
                commit = _checkpoint_phase(
                    workspace,
                    state,
                    phase,
                    message_suffix=f"complete {phase}" if regression.get("passed") else f"diagnostic {phase}",
                    allow_empty=True,
                    record_as_good=bool(regression.get("passed")),
                )
                if regression.get("passed"):
                    state.last_good_commit = commit or state.last_good_commit
                else:
                    state.diagnostic_commit = commit or ""
                phase_outcomes.append(PhaseOutcome(phase_id=phase, status=status, checkpoint_commit=commit))

            if phase_outcomes[-1].status == "failed":
                _mark_remaining_phases_skipped(manifest, state, phase_order, phase)
            _update_state_for_phase(
                state,
                phase=phase,
                status=phase_outcomes[-1].status,
                stop_reason=phase_outcomes[-1].stop_reason,
            )
            _publish_workspace(workspace)
            _write_manifests(workspace, manifest, errors, mapped_behavior_specs)
            write_controller_state(logs_dir, state)
            append_controller_event(
                logs_dir,
                {
                    "phase": phase,
                    "action": "complete",
                    "commit_hash": phase_outcomes[-1].checkpoint_commit,
                    "stop_reason": phase_outcomes[-1].stop_reason,
                },
            )
            if phase_outcomes[-1].status == "failed":
                break
    finally:
        app_pipeline.close_authoring_session(app_dir)

    if not config.generate_functional_tests:
        errors.append(
            "Functional readiness generation was skipped; this app directory is not benchmark-admissible."
        )

    generation_ok = (
        not errors
        and manifest.get("validation", {}).get("passed") is True
        and (manifest.get("variant_generation") or {}).get("status") == "validated"
        and (
            not config.generate_functional_tests
            or (
                functional_tests_complete(manifest)
                and functional_quality_gate_passed(manifest)
                and real_task_quality_gate_passed(manifest)
                and (
                    config.hardening_rounds <= 0
                    or not bool(((manifest.get("functional_tests") or {}).get("task_hardening") or {}).get("error"))
                )
                and (
                    not config.run_final_regression
                    or bool(((manifest.get("functional_tests") or {}).get("final_regression") or {}).get("passed"))
                )
            )
        )
    )
    manifest["generation"]["status"] = GENERATION_STATUS_SUCCEEDED if generation_ok else GENERATION_STATUS_FAILED
    manifest["generation"]["last_completed_phase"] = manifest["generation"].get("last_completed_phase", "initialized")
    state.current_phase = PHASE_COMPLETED
    state.stop_reason = "generation_succeeded" if generation_ok else "generation_failed"
    _publish_workspace(workspace)
    write_controller_state(logs_dir, state)
    _write_manifests(workspace, manifest, errors, mapped_behavior_specs)
    append_controller_event(
        logs_dir,
        {
            "phase": PHASE_COMPLETED,
            "action": "finish",
            "commit_hash": state.last_good_commit,
            "stop_reason": state.stop_reason,
        },
    )
    return ControllerResult(
        app_dir=str(published_app_dir),
        behavior_id=workspace.raw_behavior_id,
        manifest=manifest,
        controller_state_path=str(controller_state_path(logs_dir)),
        events_path=str(controller_events_path(logs_dir)),
        phase_outcomes=phase_outcomes,
        errors=errors,
    )


def _resolve_workspace_and_validate_output(
    *,
    behavior_spec: dict[str, Any],
    output_dir: str | Path,
) -> ControllerWorkspace:
    behavior_id = _resolved_app_id(behavior_spec)
    normalized_output_dir = normalize_output_dir(output_dir)
    return ensure_controller_workspace(
        repo_root=normalized_output_dir,
        output_dir=normalized_output_dir,
        behavior_id=behavior_id,
    )


def run_behavior(
    behavior_spec: dict[str, Any],
    output_dir: str | Path,
    config: ControllerConfig,
) -> ControllerResult:
    normalized_behavior_spec = _normalized_behavior_spec_ids(behavior_spec)
    workspace = _resolve_workspace_and_validate_output(
        behavior_spec=normalized_behavior_spec,
        output_dir=output_dir,
    )
    return _run_core_generation(
        behavior_spec=normalized_behavior_spec,
        workspace=workspace,
        config=config,
        resume=False,
    )


def resume_behavior(
    behavior_spec: dict[str, Any],
    output_dir: str | Path,
    config: ControllerConfig,
) -> ControllerResult:
    normalized_behavior_spec = _normalized_behavior_spec_ids(behavior_spec)
    workspace = _resolve_workspace_and_validate_output(
        behavior_spec=normalized_behavior_spec,
        output_dir=output_dir,
    )
    return _run_core_generation(
        behavior_spec=normalized_behavior_spec,
        workspace=workspace,
        config=config,
        resume=True,
    )


def rerun_behavior_from_phase(
    behavior_spec: dict[str, Any],
    output_dir: str | Path,
    config: ControllerConfig,
    phase: str,
) -> ControllerResult:
    normalized_behavior_spec = _normalized_behavior_spec_ids(behavior_spec)
    workspace = _resolve_workspace_and_validate_output(
        behavior_spec=normalized_behavior_spec,
        output_dir=output_dir,
    )
    return _run_core_generation(
        behavior_spec=normalized_behavior_spec,
        workspace=workspace,
        config=config,
        resume=True,
        rerun_from_phase=phase,
    )
