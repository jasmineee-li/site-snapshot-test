"""Top-level controller for redteam app generation."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentlab.benchmarks.redteam.app_artifacts import (
    APP_MANIFEST_CONTRACT_VERSION,
    GENERATION_STATUS_FAILED,
    GENERATION_STATUS_SUCCEEDED,
    _required_app_assets_error,
    _runtime_trust_boundary_error,
    behavior_contract_compatibility_error,
    benchmark_ready_app_dir_error,
    docs_snapshot_mismatch_error,
    functional_quality_gate_passed,
    functional_tests_complete,
    load_app_manifest,
    load_behavior_contract,
    real_task_quality_gate_passed,
)
from agentlab.benchmarks.redteam.behavior_ids import resolve_behavior_id
from agentlab.benchmarks.redteam.controller_state import (
    controller_events_path,
    controller_state_path,
)
from agentlab.benchmarks.redteam.git_ops import (
    ControllerWorkspace,
    current_head,  # noqa: F401 -- re-exported for test compatibility
    ensure_controller_workspace,
    normalize_output_dir,
    publish_controller_workspace,  # noqa: F401 -- re-exported for test compatibility
)
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
    PHASE_COMPLETED,
    normalize_phase_id,
)
from agentlab.benchmarks.redteam.runtime_ops import materialize_app_runtime
from agentlab.benchmarks.redteam.progress import (
    LoggingProgressReporter,
    NullProgressReporter,
    PhaseProgress,
    ProgressReporter,
)
from agentlab.benchmarks.redteam.utils import (
    sha256_bytes,
    sha256_file,
    utc_timestamp as _timestamp,
)

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Re-exports from controller_manifest
# ---------------------------------------------------------------------------
from agentlab.benchmarks.redteam.controller_manifest import (  # noqa: F401
    _resolved_app_id,
    _normalized_behavior_spec_ids,
    _mapped_behavior_specs,
    _update_manifest_seed_metadata,
    _default_primary_spa_domain_bindings,
    _normalized_route_list,
    _require_safe_behavior,
    _variant_round_from_name,
    _can_preserve_behavior_lineage,
    _merge_behavior_contract_lineage,
    _behavior_contract_payload,
    _MANIFEST_PHASE_LABELS,
    _manifest_base,
    _update_manifest_phase,
    _write_manifest,
    _write_behavior_contracts,
    _write_manifests,
)

# ---------------------------------------------------------------------------
# Re-exports from controller_ops
# ---------------------------------------------------------------------------
from agentlab.benchmarks.redteam.controller_ops import (  # noqa: F401
    load_controller_state,
    append_controller_event,
    write_controller_state,
    _publish_workspace,
    _HELPER_PHASES,
    _numeric_pass_rate,
    _real_readiness_gate,
    _checkpoint_phase,
    _effective_repetitions,
    _coerce_controller_state,
    _mark_remaining_phases_skipped,
    _update_state_for_phase,
    _build_initial_state,
    _phase_rewind_commit,
    _enforce_budget,
    _prepare_workspace_for_resume,
)

# ---------------------------------------------------------------------------
# Kept aliases (used by _run_core_generation indirectly via phase dispatch)
# ---------------------------------------------------------------------------
_sha256_bytes = sha256_bytes
_sha256_file = sha256_file

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

_AUTHORING_PHASES = frozenset(
    {
        PHASE_1A,
        PHASE_1B,
        PHASE_2A,
        PHASE_2B,
        PHASE_3A,
        PHASE_3B,
        PHASE_4B,
    }
)


def _phase_will_run(*, config: ControllerConfig, phase: str) -> bool:
    if not config.generate_functional_tests and phase not in {PHASE_1A, PHASE_1B, PHASE_1C}:
        return False
    if config.hardening_rounds <= 0 and phase in {PHASE_4A, PHASE_4B}:
        return False
    if not config.run_final_regression and phase == PHASE_5:
        return False
    return True


def _requires_pipeline_config_validation(
    *,
    config: ControllerConfig,
    current_phase: str,
) -> bool:
    if os.getenv("AGENTLAB_REDTEAM_SKIP_CONFIG_VALIDATION"):
        return False

    start_phase = normalize_phase_id(current_phase)
    if start_phase not in _PHASE_ORDER:
        return False

    start_index = _PHASE_ORDER.index(start_phase)
    return any(
        phase in _AUTHORING_PHASES and _phase_will_run(config=config, phase=phase)
        for phase in _PHASE_ORDER[start_index:]
    )


def _validate_pipeline_config_or_raise() -> None:
    from agentlab.benchmarks.redteam.pipeline_config import PipelineConfig

    pipeline_cfg = PipelineConfig()
    config_errors = pipeline_cfg.validate(check_cli=True)
    fatal_errors = [error for error in config_errors if error.severity == "fatal"]
    if not fatal_errors:
        return

    _logger.error("Configuration validation failed:")
    for error in fatal_errors:
        _logger.error("  %s", error.to_log_line())
    raise RuntimeError(
        f"Pipeline configuration invalid: {fatal_errors[0].message}. "
        f"Recovery: {fatal_errors[0].recovery}"
    )


def _progress_status_for_phase_outcome(status: str) -> str:
    if status == "succeeded":
        return "complete"
    if status == "skipped":
        return "skipped"
    return "failed"


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


def _completed_resume_requires_benchmark_readiness(
    manifest: dict[str, Any],
    *,
    config: ControllerConfig,
) -> bool:
    generation = manifest.get("generation") or {}
    return config.generate_functional_tests or generation.get("functional_tests_requested") is True


def _completed_resume_repair_error(
    *,
    published_app_dir: Path,
    manifest: dict[str, Any],
    mapped_behavior_specs: list[dict[str, Any]],
    config: ControllerConfig,
) -> str | None:
    generation = manifest.get("generation") or {}
    if generation.get("status") != GENERATION_STATUS_SUCCEEDED:
        return "app manifest generation status is not succeeded"
    if (manifest.get("variant_generation") or {}).get("status") != "validated":
        return "app manifest variant generation status is not validated"
    if (manifest.get("validation") or {}).get("passed") is not True:
        return "app manifest validation did not pass"

    require_benchmark_readiness = _completed_resume_requires_benchmark_readiness(
        manifest,
        config=config,
    )
    if require_benchmark_readiness:
        app_level_error = benchmark_ready_app_dir_error(
            published_app_dir,
            manifest=manifest,
            requested_functional_backend=config.evaluation_backend,
        )
        if app_level_error:
            return app_level_error
    else:
        runtime_trust_error = _runtime_trust_boundary_error(published_app_dir)
        if runtime_trust_error:
            return runtime_trust_error
        required_assets_error = _required_app_assets_error(published_app_dir)
        if required_assets_error:
            return required_assets_error

    for mapped_behavior_spec in mapped_behavior_specs:
        behavior_id = resolve_behavior_id(mapped_behavior_spec)
        try:
            behavior_contract = load_behavior_contract(published_app_dir, behavior_id)
        except (OSError, ValueError) as exc:
            return f"{behavior_id}: unreadable behavior contract ({exc})"

        compatibility_error = behavior_contract_compatibility_error(
            behavior_contract,
            manifest=manifest,
            behavior_id=behavior_id,
        )
        if compatibility_error:
            return f"{behavior_id}: {compatibility_error}"

        if require_benchmark_readiness:
            behavior_error = benchmark_ready_app_dir_error(
                published_app_dir,
                manifest=manifest,
                behavior_contract=behavior_contract,
                behavior_id=behavior_id,
                requested_functional_backend=config.evaluation_backend,
            )
            if behavior_error:
                return f"{behavior_id}: {behavior_error}"
            continue

        runtime_trust_error = _runtime_trust_boundary_error(
            published_app_dir,
            behavior_contract=behavior_contract,
        )
        if runtime_trust_error:
            return f"{behavior_id}: {runtime_trust_error}"
        required_assets_error = _required_app_assets_error(
            published_app_dir,
            behavior_contract=behavior_contract,
        )
        if required_assets_error:
            return f"{behavior_id}: {required_assets_error}"

    return None


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
    if resume or rerun_from_phase is not None:
        manifest = existing_manifest or _manifest_base(
            behavior_spec=behavior_spec,
            app_dir=published_app_dir,
            config=config,
            repo_root=workspace.repo_root,
        )
    else:
        manifest = _manifest_base(
            behavior_spec=behavior_spec,
            app_dir=published_app_dir,
            config=config,
            repo_root=workspace.repo_root,
        )
    docs_snapshot_error = docs_snapshot_mismatch_error(
        manifest=manifest,
        behavior_spec=behavior_spec,
        repo_root_path=workspace.repo_root,
        fail_closed_on_missing=bool(resume and rerun_from_phase is None),
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
        if not resume:
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
    skip_resume_backend_validation = False
    if resume:
        if rerun_from_phase is None and normalize_phase_id(state.current_phase) == PHASE_COMPLETED:
            repair_error = _completed_resume_repair_error(
                published_app_dir=published_app_dir,
                manifest=manifest,
                mapped_behavior_specs=mapped_behavior_specs,
                config=config,
            )
            if repair_error is None:
                return ControllerResult(
                    app_dir=str(published_app_dir),
                    behavior_id=workspace.raw_behavior_id,
                    manifest=manifest,
                    controller_state_path=str(controller_state_path(logs_dir)),
                    events_path=str(controller_events_path(logs_dir)),
                    phase_outcomes=[],
                    errors=list(existing_manifest.get("errors") or []),
                )
            rerun_from_phase = PHASE_1A
            skip_resume_backend_validation = True
            append_controller_event(
                logs_dir,
                {
                    "phase": PHASE_COMPLETED,
                    "action": "repair_completed_app",
                    "reason": repair_error,
                    "rerun_from_phase": rerun_from_phase,
                },
            )
        backend_error = None
        if not skip_resume_backend_validation:
            try:
                resume_pipeline_state = app_pipeline.load_pipeline_state(
                    app_dir,
                    logs_dir=logs_dir,
                    strict=True,
                )
            except RuntimeError as exc:
                backend_error = str(exc)
            else:
                backend_error = app_pipeline.resume_backend_error(
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

    if _requires_pipeline_config_validation(
        config=config,
        current_phase=state.current_phase,
    ):
        _validate_pipeline_config_or_raise()

    write_controller_state(logs_dir, state)
    _write_manifests(workspace, manifest, errors, mapped_behavior_specs)

    phase_order = _PHASE_ORDER
    start_index = phase_order.index(normalize_phase_id(state.current_phase)) if normalize_phase_id(state.current_phase) in phase_order else 0
    reporter = LoggingProgressReporter(behavior_id=workspace.raw_behavior_id)

    try:
        for phase in phase_order[start_index:]:
            phase_start = time.monotonic()
            reporter.report(PhaseProgress(phase_id=phase, status="starting"))
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
                from agentlab.benchmarks.redteam.audit_runner import run_audit_step

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
                    baseline = app_pipeline.ensure_readiness_baseline(
                        app_dir,
                        backend=config.evaluation_backend,
                    )
                    app_pipeline.freeze_real_task_baseline(
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
                baseline = app_pipeline.load_backend_readiness_baseline(
                    app_dir,
                    backend=config.evaluation_backend,
                )
                if not baseline:
                    real_evaluation = manifest["functional_tests"].get("real_evaluation") or {}
                    real_results = list(real_evaluation.get("results") or [])
                    if real_results:
                        baseline = app_pipeline.ensure_readiness_baseline(
                            app_dir,
                            backend=config.evaluation_backend,
                        )
                        app_pipeline.freeze_real_task_baseline(
                            app_dir,
                            baseline_results=real_results,
                        )
                if baseline:
                    manifest["functional_tests"]["readiness_baseline"] = baseline
                has_hardening_baseline = bool(app_pipeline.load_real_task_baseline_snapshot(app_dir))
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

            elapsed = time.monotonic() - phase_start
            reporter.report(PhaseProgress(
                phase_id=phase,
                status=_progress_status_for_phase_outcome(phase_outcomes[-1].status),
                elapsed_seconds=elapsed,
                detail=phase_outcomes[-1].stop_reason or "",
            ))
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
