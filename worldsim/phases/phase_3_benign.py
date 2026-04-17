"""Phase 3: Benign validation.

Canonical source: ``docs/worldsim-v5-technical-specifcation.md`` "Phase 3: Benign Validation".

For each task, seed the environment with benign data, run the target agent
via Browser Use, and score the trajectory with the task's reward function.
Failed tasks enter a diagnosis-fix loop driven by a Modal Sandbox.

Failure taxonomy:

- reward function bug -> fix reward, re-run sanity check
- data seed issue -> fix seed, re-run sanity check
- impossible task -> remove
- task too hard -> keep but flag
- agent limitation -> keep (informative baseline data)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import tempfile
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from worldsim.agent_config import (
    DEFAULT_MODEL,
    RUNTIME_METADATA_KEY,
    bind_task_to_instance,
    cap_tasks_per_site,
    execution_instance_dict,
    make_agent_factory,
    resolve_task_inputs,
    run_tasks_by_site,
    task_reset_endpoints,
)
from worldsim.agent_prompt import build_agent_prompt
from worldsim.atomic_io import write_json_atomic
from worldsim.auth_tokens import acquire_tokens_for_instances
from worldsim.browser_use_agent import AgentRunner, BrowserUseAgent, resolve_instance_agent_auth
from worldsim.config import BenchmarkConfig, BenchmarkInstance, has_configured_agent_auth
from worldsim.cost_tracker import tracker as cost_tracker
from worldsim.fix_validation import validate_fix_patch
from worldsim.modal_sandbox import preflight_auth_check, run_claude_in_sandbox
from worldsim.phase_3_triage import TRIAGE_CACHE_VERSION, triage_failures
from worldsim.profile_validation import load_and_validate_profile, profile_requires_agent_auth
from worldsim.prompt_loading import load_prompt
from worldsim.resume_metadata import (
    RESULT_FINGERPRINT_KEY,
    fingerprint_payload,
    instances_identity,
)
from worldsim.rewards import run_reward_function
from worldsim.seeding import apply_data_seed_async, collect_seed_runtime_errors
from worldsim.site_lock import task_lock
from worldsim.state import get_state_dir, save_state
from worldsim.storage_state_preflight import (
    apply_skip_auth_for_host_bound_storage_states,
    inspect_storage_state_preflight,
)
from worldsim.task_paths import safe_task_path_component
from worldsim.task_reset_cache import (
    TaskResetCache,
    callable_accepts_keyword,
    result_likely_mutated_state,
)
from worldsim.trajectory import load_trajectory_into_sandbox, save_result

logger = logging.getLogger(__name__)
_CHECKPOINT_FINGERPRINT_KEY = "_source_fingerprint"


def _resume_fingerprint_task(task: dict[str, Any]) -> dict[str, Any]:
    """Strip execution-local routing metadata from result fingerprinting."""
    normalized = json.loads(json.dumps(task))
    normalized.pop(RUNTIME_METADATA_KEY, None)
    return normalized


def _phase_3_state_metadata(
    *,
    task_dir_root: Path,
    instances_path: Path | str,
    agent_model: str,
    sandbox_model: str,
    agent_provider: str | None,
    full_baseline: bool,
    max_tasks_per_site: int | None,
    sites: str | None,
    benchmark_root: Path | None,
    allow_unknown_auth: bool,
    skip_host_bound_storage_state_auth: bool,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "task_dir_root": str(task_dir_root),
        "instances_path": str(instances_path),
        "agent_model": agent_model,
        "sandbox_model": sandbox_model,
        "agent_provider": agent_provider,
        "full_baseline": full_baseline,
        "max_tasks_per_site": max_tasks_per_site,
        "allow_unknown_auth": allow_unknown_auth,
        "skip_host_bound_storage_state_auth": skip_host_bound_storage_state_auth,
    }
    if sites is not None:
        metadata["sites"] = sites
    if benchmark_root is not None:
        metadata["benchmark_path"] = str(benchmark_root)
    return metadata


def _filter_tasks_by_sites(
    tasks: list[dict[str, Any]],
    sites_filter_raw: str | None,
    *,
    phase_label: str,
) -> list[dict[str, Any]]:
    if not sites_filter_raw:
        return tasks
    sites_filter = {site.strip() for site in sites_filter_raw.split(",") if site.strip()}
    known_sites = {str(task.get("site", "")).strip() for task in tasks if task.get("site")}
    unknown = sites_filter - known_sites
    if unknown:
        raise ValueError(
            f"{phase_label}: --sites includes unknown site(s): {sorted(unknown)}. "
            f"Known sites: {sorted(known_sites)}"
        )
    filtered = [task for task in tasks if str(task.get("site", "")).strip() in sites_filter]
    logger.info("%s: --sites filter active, running only %s", phase_label, sorted(sites_filter))
    return filtered


def _write_json_atomic(
    path: Path,
    payload: dict[str, Any],
    *,
    failpoint_base: str | None = None,
) -> None:
    write_json_atomic(path, payload, failpoint_base=failpoint_base)


def _triage_metadata_fields(triage: dict[str, Any]) -> dict[str, Any]:
    return {
        "triage_decision": triage.get("decision"),
        "triage_likely_root_cause": triage.get("likely_root_cause"),
        "triage_confidence": triage.get("confidence"),
        "triage_reason": triage.get("reason"),
        "triage_source": triage.get("source"),
        "triage_escalate": triage.get("escalate"),
        "triage_cache_version": TRIAGE_CACHE_VERSION,
    }


def _persist_triage_metadata(
    *,
    result: dict[str, Any],
    triage: dict[str, Any],
) -> None:
    trajectory_dir_raw = str(result.get("trajectory_dir", "")).strip()
    if not trajectory_dir_raw:
        return
    trajectory_dir = Path(trajectory_dir_raw)
    result_path = trajectory_dir / "result.json"
    if not result_path.exists():
        return
    try:
        payload = json.loads(result_path.read_text())
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to load %s for triage metadata update", result_path)
        return
    if not isinstance(payload, dict):
        logger.warning("Skipping triage metadata update for non-object %s", result_path)
        return
    metadata_fields = _triage_metadata_fields(triage)
    if all(payload.get(key) == value for key, value in metadata_fields.items()):
        return
    payload.update(metadata_fields)
    _write_json_atomic(
        result_path,
        payload,
        failpoint_base="phase_3.outputs.triage_result",
    )


def _fingerprint_payload(*parts: Any) -> str:
    return fingerprint_payload(*parts)


def _phase_3_eval_context(
    *,
    instances: list[BenchmarkInstance],
    config_url_placeholders: dict[str, str] | None,
    agent_model: str,
    agent_provider: str | None,
    benchmark_root: Path | None,
) -> dict[str, Any]:
    return {
        "phase": "phase_3_initial_result",
        "instances": instances_identity(instances),
        "config_url_placeholders": config_url_placeholders,
        "agent_model": agent_model,
        "agent_provider": agent_provider,
        "benchmark_root": str(benchmark_root) if benchmark_root is not None else None,
    }


def _phase_3_result_fingerprint(
    task: dict[str, Any],
    *,
    eval_context: dict[str, Any],
    site_profile: dict[str, Any] | None,
) -> str:
    return _fingerprint_payload(_resume_fingerprint_task(task), eval_context, site_profile)


def _phase_3_diagnosis_fingerprint(
    task: dict[str, Any],
    result: dict[str, Any],
    *,
    instances: list[BenchmarkInstance],
    config_url_placeholders: dict[str, str] | None,
    benchmark_root: Path | None,
    sandbox_model: str,
    site_profile: dict[str, Any] | None,
) -> str:
    return _fingerprint_payload(
        task,
        result,
        {
            "phase": "phase_3_diagnosis",
            "instances": instances_identity(instances),
            "config_url_placeholders": config_url_placeholders,
            "benchmark_root": str(benchmark_root) if benchmark_root is not None else None,
            "sandbox_model": sandbox_model,
            "site_profile": site_profile,
        },
    )


async def run(args: argparse.Namespace) -> int:
    """Phase 3 entrypoint — benign validation of wrapped tasks."""
    state_dir = get_state_dir()
    # Load inputs
    tasks_path = state_dir / "phase_1" / "benign_tasks.json"
    if not tasks_path.exists():
        logger.error("Benign tasks not found at %s — run phase 1 first", tasks_path)
        return 1
    benign_tasks = json.loads(tasks_path.read_text())
    sites_filter_raw = getattr(args, "sites", None)
    try:
        benign_tasks = _filter_tasks_by_sites(benign_tasks, sites_filter_raw, phase_label="Phase 3")
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    # Default: filter to only tasks with adversarial counterparts
    full_baseline = getattr(args, "full_baseline", False)
    if not full_baseline:
        adv_tasks_path = state_dir / "phase_2" / "adversarial_tasks.json"
        if adv_tasks_path.exists():
            adv_tasks = json.loads(adv_tasks_path.read_text())
            if not isinstance(adv_tasks, list):
                logger.warning(
                    "Phase 3: adversarial_tasks.json is not an array, running all benign tasks"
                )
                adv_tasks = []
            paired_ids = {
                str(t.get("benign_task_id", "")) for t in adv_tasks if isinstance(t, dict)
            }
            original_count = len(benign_tasks)
            benign_tasks = [t for t in benign_tasks if str(t.get("id", "")) in paired_ids]
            logger.info(
                "Phase 3: filtered to %d/%d tasks with adversarial counterparts (use --full-baseline for all)",
                len(benign_tasks),
                original_count,
            )
        else:
            logger.warning("Phase 3: adversarial_tasks.json not found, running all benign tasks")

    # Per-site cap for smoke testing (applied after adversarial-paired filtering)
    max_tasks_per_site = getattr(args, "max_tasks_per_site", None)
    if max_tasks_per_site is not None:
        pre_cap = len(benign_tasks)
        benign_tasks = cap_tasks_per_site(benign_tasks, max_tasks_per_site)
        logger.info(
            "Phase 3: capped to %d/%d tasks (max %d per site)",
            len(benign_tasks),
            pre_cap,
            max_tasks_per_site,
        )

    instances_path = getattr(args, "instances", None)
    if not instances_path or not Path(instances_path).exists():
        logger.error("--instances JSON file required for Phase 3 (running benchmark instances)")
        return 1
    config = BenchmarkConfig.model_validate_json(Path(instances_path).read_text())

    # On resume, reuse the task_dir_root from the prior run so
    # load_completed_results can find existing result.json files.
    resume = getattr(args, "resume", False)
    prior_state = None
    if resume:
        from worldsim.state import load_state

        prior_state = load_state()

    if prior_state and prior_state.get("step") == "phase_3" and prior_state.get("task_dir_root"):
        task_dir_root = Path(prior_state["task_dir_root"])
        logger.info("Resume: reusing task_dir_root %s", task_dir_root)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_dir_root = state_dir / "phase_3" / timestamp

    agent_model = getattr(args, "agent_model", None) or DEFAULT_MODEL
    sandbox_model = getattr(args, "sandbox_model", None) or "claude-sonnet-4-6"
    agent_provider = getattr(args, "agent_provider", None)
    benchmark_root = getattr(args, "benchmark", None)
    allow_unknown_auth = bool(getattr(args, "allow_unknown_auth", False))
    skip_host_bound_storage_state_auth = bool(
        getattr(args, "skip_host_bound_storage_state_auth", False)
    )
    state_metadata = _phase_3_state_metadata(
        task_dir_root=task_dir_root,
        instances_path=instances_path,
        agent_model=agent_model,
        sandbox_model=sandbox_model,
        agent_provider=agent_provider,
        full_baseline=full_baseline,
        max_tasks_per_site=max_tasks_per_site,
        sites=sites_filter_raw,
        benchmark_root=benchmark_root,
        allow_unknown_auth=allow_unknown_auth,
        skip_host_bound_storage_state_auth=skip_host_bound_storage_state_auth,
    )
    preflight = inspect_storage_state_preflight(
        config.instances,
        benchmark_root=benchmark_root,
    )
    preflight_errors = list(preflight.errors)
    host_bound_mismatches = list(preflight.mismatches)
    if preflight_errors:
        error_lines = [
            f"site {error.site_name!r}: {error.message} (declared path {error.declared_path!r})"
            for error in preflight_errors
        ]
        logger.error(
            "Phase 3 storage-state pre-flight failed:\n%s",
            "\n".join(f"  - {line}" for line in error_lines),
        )
        save_state(
            "phase_3",
            status="failed",
            reason="storage_state_preflight_error",
            storage_state_preflight_errors=error_lines,
            **state_metadata,
        )
        return 1
    if host_bound_mismatches:
        mismatch_lines = [
            (
                f"site {mismatch.site_name!r}: storage_state {mismatch.artifact_path} "
                f"records hosts {list(mismatch.recorded_hosts)!r}, but live instances use "
                f"{list(mismatch.instance_hosts)!r}"
            )
            for mismatch in host_bound_mismatches
        ]
        if skip_host_bound_storage_state_auth:
            logger.warning(
                "Phase 3 found host-bound storage_state artifacts and will skip agent auth for "
                "those sites because --skip-host-bound-storage-state-auth was set:\n%s",
                "\n".join(f"  - {line}" for line in mismatch_lines),
            )
            config = apply_skip_auth_for_host_bound_storage_states(config, host_bound_mismatches)
        else:
            logger.error(
                "Phase 3 storage-state pre-flight failed:\n%s\nRe-run Phase 0d against the "
                "current instances host, or pass --skip-host-bound-storage-state-auth to "
                "proceed without browser auth for those sites.",
                "\n".join(f"  - {line}" for line in mismatch_lines),
            )
            save_state(
                "phase_3",
                status="failed",
                reason="host_bound_storage_state",
                host_bound_storage_state_errors=mismatch_lines,
                **state_metadata,
            )
            return 1
    # Acquire fresh bearer tokens for instances that use runtime generation.
    token_errors = acquire_tokens_for_instances(config.instances)
    if token_errors:
        logger.error(
            "Phase 3 token acquisition failed:\n%s",
            "\n".join(f"  - {error}" for error in token_errors),
        )
        save_state(
            "phase_3",
            status="failed",
            reason="token_acquisition_failed",
            token_errors=token_errors,
            **state_metadata,
        )
        return 1
    seed_runtime_errors = collect_seed_runtime_errors(
        benign_tasks,
        config.instances,
        seed_field="data_seed",
    )
    if seed_runtime_errors:
        logger.error(
            "Phase 3 seed pre-flight failed:\n%s",
            "\n".join(f"  - {error}" for error in seed_runtime_errors),
        )
        save_state(
            "phase_3",
            status="failed",
            reason="seed_runtime_config_error",
            seed_runtime_errors=seed_runtime_errors,
            **state_metadata,
        )
        return 1
    # Fail fast if Claude Code auth is missing — diagnosis sandboxes need it.
    try:
        preflight_auth_check()
    except RuntimeError as exc:
        logger.error("Phase 3 auth pre-flight failed:\n%s", exc)
        save_state("phase_3", status="failed", reason="auth_preflight_failed", **state_metadata)
        return 1

    agent_factory = make_agent_factory(model=agent_model, provider=agent_provider)
    save_state("phase_3", status="running", **state_metadata)

    logger.info(
        "Phase 3: validating %d benign tasks across %d instances",
        len(benign_tasks),
        len(config.instances),
    )
    profiles_dir = state_dir / "phase_0c"
    site_profiles = _load_site_profiles(benign_tasks, profiles_dir)
    agent_auth_errors = _collect_agent_auth_runtime_errors(config.instances, site_profiles)
    if agent_auth_errors:
        logger.error(
            "Phase 3 agent-auth pre-flight failed:\n%s",
            "\n".join(f"  - {error}" for error in agent_auth_errors),
        )
        save_state(
            "phase_3",
            status="failed",
            reason="agent_runtime_config_error",
            agent_runtime_errors=agent_auth_errors,
            **state_metadata,
        )
        return 1
    eval_context = _phase_3_eval_context(
        instances=config.instances,
        config_url_placeholders=config.url_placeholders,
        agent_model=agent_model,
        agent_provider=agent_provider,
        benchmark_root=benchmark_root,
    )

    # Build lookup from raw tasks — run_tasks_by_site calls
    # prepare_tasks_for_execution internally, so no need to call it here.
    prepared_by_id = {str(task.get("id", "unknown")): task for task in benign_tasks}
    reset_cache = TaskResetCache()

    # Thread the benchmark codebase root through so BrowserUseAgent can resolve
    # relative auth_mechanism.storage_state.path values. See worldsim/browser_use_agent.py:_resolve_auth.

    async def _bound_run_task(task, agent, instance, task_dir):
        run_kwargs: dict[str, Any] = {
            "benchmark_root": benchmark_root,
            "site_profile": site_profiles.get(str(task.get("site", ""))),
            "resume_fingerprint": _phase_3_result_fingerprint(
                task,
                eval_context=eval_context,
                site_profile=site_profiles.get(str(task.get("site", ""))),
            ),
        }
        if callable_accepts_keyword(run_task, "reset_cache"):
            run_kwargs["reset_cache"] = reset_cache
        return await run_task(
            task,
            agent,
            instance,
            task_dir,
            **run_kwargs,
        )

    # Run evaluation via worker pool, routing tasks only to matching site instances.
    results = await run_tasks_by_site(
        tasks=benign_tasks,
        instances=config.instances,
        agent_factory=agent_factory,
        task_runner=_bound_run_task,
        task_dir_root=task_dir_root,
        config_url_placeholders=config.url_placeholders,
        resume=resume,
        resume_fingerprint_builder=lambda task: _phase_3_result_fingerprint(
            task,
            eval_context=eval_context,
            site_profile=site_profiles.get(str(task.get("site", ""))),
        ),
    )

    # Summarize results
    passed_tasks = [r for r in results if r.get("passed")]
    failed_tasks = [r for r in results if not r.get("passed")]
    logger.info(
        "Phase 3 initial run: %d/%d passed, %d failed",
        len(passed_tasks),
        len(results),
        len(failed_tasks),
    )

    # Diagnosis loop for failures
    diagnosed: list[dict] = []
    triage_records: list[dict[str, Any]] = []

    # Circuit breaker: if >30% of tasks errored (infrastructure problems,
    # not agent failures), skip the expensive diagnosis loop and surface the
    # issue to the operator. Follows AgentLab's retry-loop pattern.
    error_tasks = [r for r in results if r.get("outcome") == "error"]
    if error_tasks and len(error_tasks) > 0.3 * len(results):
        logger.warning(
            "Circuit breaker: %d/%d tasks errored (>30%%), skipping diagnosis. "
            "This likely indicates an infrastructure problem, not task bugs.",
            len(error_tasks),
            len(results),
        )
        save_state(
            "phase_3",
            status="failed",
            reason="infra_error_circuit_breaker",
            **state_metadata,
            error_tasks=[str(r.get("task_id", "?")) for r in error_tasks],
            total=len(results),
        )
        return 1

    if failed_tasks:
        triage_records = await triage_failures(
            failed_results=failed_tasks,
            prepared_by_id=prepared_by_id,
        )
        triage_by_id = {str(record.get("task_id", "")): record for record in triage_records}
        for result in failed_tasks:
            task_id = str(result.get("task_id", ""))
            triage = triage_by_id.get(task_id)
            if triage is None:
                continue
            result.update(_triage_metadata_fields(triage))
            _persist_triage_metadata(result=result, triage=triage)
    else:
        triage_by_id = {}

    output_dir = state_dir / "phase_3"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        output_dir / "triage.json",
        triage_records,
        failpoint_base="phase_3.outputs.triage",
    )

    diagnosable_failures = [
        r for r in failed_tasks if r.get("outcome") != "error" and bool(r.get("triage_escalate"))
    ]
    triaged_agent_limitations = [
        r for r in failed_tasks if r.get("triage_decision") == "agent_limitation"
    ]
    triaged_infra = [r for r in failed_tasks if r.get("triage_decision") == "infra_error"]
    if triaged_infra:
        logger.warning(
            "Skipping diagnosis for %d infrastructure-error tasks after triage",
            len(triaged_infra),
        )
    if triaged_agent_limitations:
        logger.info(
            "Skipping diagnosis for %d obvious agent-limitation failures after triage",
            len(triaged_agent_limitations),
        )

    raw_diagnoses = await asyncio.gather(
        *[
            _diagnose_one_task(
                r=r,
                prepared_by_id=prepared_by_id,
                profiles_dir=profiles_dir,
                task_dir_root=task_dir_root,
                instances=config.instances,
                config_url_placeholders=config.url_placeholders,
                agent_factory=agent_factory,
                resume=resume,
                benchmark_root=benchmark_root,
                sandbox_model=sandbox_model,
                site_profile=site_profiles.get(
                    str(prepared_by_id.get(str(r.get("task_id", "")), {}).get("site", ""))
                ),
            )
            for r in diagnosable_failures
        ],
        return_exceptions=True,
    )

    diagnosis_failures: list[tuple[str, BaseException]] = []
    for i, result in enumerate(raw_diagnoses):
        if isinstance(result, BaseException):
            task_id = str(diagnosable_failures[i].get("task_id", "?"))
            logger.error(
                "Diagnosis failed for task %s: %s",
                task_id,
                result,
            )
            diagnosis_failures.append((task_id, result))
            continue
        if result is not None:
            diagnosed.append(result)

    if diagnosis_failures:
        save_state(
            "phase_3",
            status="failed",
            reason="diagnosis_exception",
            failed_tasks=[task_id for task_id, _ in diagnosis_failures],
            **state_metadata,
        )
        return 1

    # Write validated tasks (passed + fixed)
    passed_ids = {r["task_id"] for r in passed_tasks}
    fixed_ids = {d["task_id"] for d in diagnosed if d.get("action") == "fixed"}
    validated_tasks_by_id = {
        task_id: prepared_by_id[task_id] for task_id in passed_ids if task_id in prepared_by_id
    }
    for diagnosis in diagnosed:
        if diagnosis.get("action") == "fixed" and diagnosis.get("fixed_task"):
            validated_tasks_by_id[diagnosis["task_id"]] = diagnosis["fixed_task"]
    validated_tasks = [
        validated_tasks_by_id[t["id"]] for t in benign_tasks if t["id"] in validated_tasks_by_id
    ]

    if not validated_tasks:
        logger.warning(
            "Phase 3: no tasks passed validation — 0/%d benign tasks validated. "
            "This is a valid scientific result but worth investigating.",
            len(benign_tasks),
        )

    # Strip internal runtime metadata before persisting to disk.
    clean_tasks = [
        {k: v for k, v in t.items() if k != RUNTIME_METADATA_KEY} for t in validated_tasks
    ]

    write_json_atomic(
        output_dir / "validated_tasks.json",
        clean_tasks,
        failpoint_base="phase_3.outputs.validated_tasks",
    )
    write_json_atomic(
        output_dir / "results.json",
        results,
        failpoint_base="phase_3.outputs.results",
    )
    write_json_atomic(
        output_dir / "diagnoses.json",
        diagnosed,
        failpoint_base="phase_3.outputs.diagnoses",
    )

    save_state(
        "phase_3",
        status="complete",
        **state_metadata,
        validated_tasks_path=str(output_dir / "validated_tasks.json"),
        passed=len(passed_ids),
        fixed=len(fixed_ids),
        triaged_agent_limitations=len(triaged_agent_limitations),
        triaged_infra=len(triaged_infra),
        triaged_escalated=len(diagnosable_failures),
        total=len(benign_tasks),
    )
    cost_tracker.log_phase_summary("phase_3")
    cost_tracker.save(state_dir / "cost_report.json")
    logger.info(
        "Phase 3 complete — %d validated (%d passed + %d fixed) out of %d total",
        len(validated_tasks),
        len(passed_ids),
        len(fixed_ids),
        len(benign_tasks),
    )
    return 0


async def _diagnose_one_task(
    r: dict[str, Any],
    prepared_by_id: dict[str, dict[str, Any]],
    profiles_dir: Path,
    task_dir_root: Path,
    instances: list[BenchmarkInstance],
    agent_factory: Callable[[], BrowserUseAgent],
    resume: bool,
    config_url_placeholders: dict[str, str] | None = None,
    benchmark_root: Path | None = None,
    sandbox_model: str = "claude-sonnet-4-6",
    site_profile: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Run diagnosis-fix loop for a single failed task. Returns diagnosis dict or None."""
    task_id = str(r.get("task_id", "?"))
    task = prepared_by_id.get(task_id)
    if not task:
        logger.warning("Could not find task %s for diagnosis", task_id)
        return None

    # On resume, skip tasks that were already diagnosed in a prior run.
    diagnosis_file = task_dir_root / safe_task_path_component(task_id) / "diagnosis.json"
    source_fingerprint = _phase_3_diagnosis_fingerprint(
        task,
        r,
        instances=instances,
        config_url_placeholders=config_url_placeholders,
        benchmark_root=benchmark_root,
        sandbox_model=sandbox_model,
        site_profile=site_profile,
    )
    if resume and diagnosis_file.exists():
        try:
            prior_diagnosis = json.loads(diagnosis_file.read_text())
            if (
                isinstance(prior_diagnosis, dict)
                and prior_diagnosis.get(_CHECKPOINT_FINGERPRINT_KEY) == source_fingerprint
            ):
                logger.info("Resume: reusing prior diagnosis for task %s", task_id)
                return {
                    "task_id": task_id,
                    **{
                        key: value
                        for key, value in prior_diagnosis.items()
                        if key != _CHECKPOINT_FINGERPRINT_KEY
                    },
                }
        except (json.JSONDecodeError, OSError):
            pass

    site = task["site"]
    profile_path = profiles_dir / f"BENCHMARK_PROFILE_{site}.json"
    trajectory_dir = Path(
        r.get("trajectory_dir", task_dir_root / safe_task_path_component(task_id))
    )

    if not trajectory_dir.exists():
        logger.warning("No trajectory dir for task %s, skipping diagnosis", task_id)
        return None

    fix_result = await fix_loop(
        task=task,
        trajectory_dir=trajectory_dir,
        profile_path=profile_path,
        task_dir_root=task_dir_root,
        instances=instances,
        agent_factory=agent_factory,
        benchmark_root=benchmark_root,
        sandbox_model=sandbox_model,
        site_profile=site_profile,
    )

    # Persist diagnosis for resume. Without this, all diagnoses would
    # re-run on crash recovery (Modal sandbox invocations are expensive).
    _write_json_atomic(
        diagnosis_file,
        {
            **fix_result,
            _CHECKPOINT_FINGERPRINT_KEY: source_fingerprint,
        },
        failpoint_base="phase_3.diagnosis.checkpoint",
    )

    return {"task_id": task_id, **fix_result}


async def run_task(
    task: dict[str, Any],
    agent: AgentRunner,
    instance: BenchmarkInstance,
    task_dir: Path,
    *,
    benchmark_root: Path | None = None,
    site_profile: dict[str, Any] | None = None,
    reset_cache: TaskResetCache | None = None,
    resume_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Run one benign task: reset -> seed -> agent run -> reward check.

    This is the ``task_runner`` callable passed to ``eval_worker_pool.run_eval``.

    ``benchmark_root`` is forwarded to ``BrowserUseAgent.run`` so
    ``auth_mechanism.storage_state.path`` values that are declared relative
    (the common case for benchmarks that ship ``.auth/*.json`` artifacts next
    to their sources) resolve against the correct codebase root. Absolute
    paths bypass the root.
    """
    task_id = task.get("id", "unknown")

    instance_dict = execution_instance_dict(instance, task)
    if isinstance(site_profile, dict):
        instance_dict["site_profile"] = json.loads(json.dumps(site_profile))
    instance_dict["seed_task"] = json.loads(json.dumps(task))

    should_reset = True
    if reset_cache is not None:
        should_reset = reset_cache.should_reset(task)
    if should_reset:
        await _reset_task_environment(task)
        if reset_cache is not None:
            reset_cache.mark_clean(task)

    task_likely_mutated = False
    try:
        # Seed data (Mode A tasks have mechanism "none" — skip)
        seed = task.get("data_seed", {})
        if seed.get("mechanism") not in (None, "none"):
            task_likely_mutated = True
            await apply_data_seed_async(seed, instance_dict)

        # Run agent
        instance_agent_auth = resolve_instance_agent_auth(instance_dict)
        instruction, start_urls = resolve_task_inputs(task, instance_dict)
        site_prompt = _build_agent_prompt(
            _agent_context_with_instance_auth(task.get("agent_context"), instance_agent_auth),
            instruction,
            start_urls,
            task=task,
        )
        run_kwargs: dict[str, Any] = {"start_urls": start_urls}
        if site_prompt is not None:
            run_kwargs["site_prompt"] = site_prompt
        # Auth from instances.json — single source of truth. No fallback to
        # Phase 0c LLM-generated auth. If agent_auth is not configured for a site,
        # the task runs without auth (fail-fast over silent degradation).
        if instance_agent_auth is not None:
            run_kwargs["auth_mechanism"] = instance_agent_auth
            if benchmark_root is not None:
                run_kwargs["benchmark_root"] = benchmark_root
            site_value = task.get("site")
            if isinstance(site_value, str) and site_value:
                run_kwargs["task_site"] = site_value
        result = await agent.run(
            instruction,
            instance.site_url,
            task_dir,
            **run_kwargs,
        )
    except Exception:
        if reset_cache is not None:
            reset_cache.mark_dirty(task)
        raise

    task_likely_mutated = task_likely_mutated or result_likely_mutated_state(task, result)
    if reset_cache is not None:
        if task_likely_mutated:
            reset_cache.mark_dirty(task)
        else:
            reset_cache.mark_clean(task)

    if result.status != "success" and not _has_scoreable_agent_output(result):
        message = f"agent run {result.status}: " + (
            result.errors[-1] if result.errors else "no additional error details"
        )
        extra: dict[str, Any] = {}
        if resume_fingerprint is not None:
            extra[RESULT_FINGERPRINT_KEY] = resume_fingerprint
        save_result(task_dir, task, result, False, message, **extra)
        return {
            "task_id": task_id,
            "passed": False,
            "outcome": "error",
            "message": message,
            "elapsed": result.elapsed,
            "steps": result.steps,
            "is_done": result.is_done,
            "trajectory_dir": str(task_dir),
        }

    # Evaluate with reward function (offloaded to thread to avoid blocking
    # the event loop when the subprocess evaluator path is used).
    passed, message = await asyncio.to_thread(
        run_reward_function,
        reward=task["reward_function"],
        instance=instance_dict,
        agent_result=result,
        network_trace=result.network_trace,
    )

    # Save trajectory artifacts
    extra: dict[str, Any] = {}
    if resume_fingerprint is not None:
        extra[RESULT_FINGERPRINT_KEY] = resume_fingerprint
    save_result(task_dir, task, result, passed, message, **extra)

    return {
        "task_id": task_id,
        "passed": passed,
        "message": message,
        "elapsed": result.elapsed,
        "steps": result.steps,
        "is_done": result.is_done,
        "trajectory_dir": str(task_dir),
    }


def _has_scoreable_agent_output(result: Any) -> bool:
    """Return True when a non-success run still produced usable evaluation artifacts."""
    return bool(
        getattr(result, "steps", 0) > 0
        or getattr(result, "final_result", None)
        or getattr(result, "network_trace", None)
    )


def _agent_context_with_instance_auth(
    agent_context: Any,
    instance_agent_auth: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(agent_context, dict):
        agent_context = {}
    merged = json.loads(json.dumps(agent_context))
    if not isinstance(instance_agent_auth, dict):
        return merged or None
    if str(instance_agent_auth.get("type", "")).strip() == "none":
        return merged or None
    if "authentication" not in merged:
        authentication = instance_agent_auth.get("authentication")
        if isinstance(authentication, dict):
            merged["authentication"] = json.loads(json.dumps(authentication))
        else:
            merged["authentication"] = {
                "pre_authenticated": True,
                "credentials": None,
                "description": "Pre-authenticated via deployment config.",
            }
    return merged or None


async def diagnose_failure(
    task: dict[str, Any],
    trajectory_dir: Path,
    profile_path: Path,
    *,
    rejection_feedback: str | None = None,
    sandbox_model: str = "claude-sonnet-4-6",
) -> dict[str, Any]:
    """Diagnose why a benign task failed via Modal Sandbox.

    Returns diagnosis dict with root_cause, explanation, and suggested_fix.
    If ``rejection_feedback`` is provided, it is appended to the diagnosis
    prompt as a retry signal explaining why the previous patch was rejected.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        # Stage task definition
        task_json_path = tmp / "task.json"
        task_json_path.write_text(json.dumps(task, indent=2))

        sandbox_files: dict[str, str] = {
            "/workspace/task.json": str(task_json_path),
        }

        # Add profile if available
        if profile_path.exists():
            sandbox_files["/workspace/profile/BENCHMARK_PROFILE.json"] = str(profile_path)

        # Add trajectory artifacts
        load_trajectory_into_sandbox(trajectory_dir, sandbox_files)

        prompt = _render_diagnosis_prompt(task)
        if rejection_feedback:
            prompt = (
                prompt
                + "\n\n## Previous patch rejected\n\n"
                + rejection_feedback
                + "\n\nIf you cannot propose a safe patch, set `root_cause` to "
                "`agent_limitation` or `too_hard` and `suggested_fix.target` to `none`.\n"
            )

        task_id = task.get("id", "unknown")
        outputs = await run_claude_in_sandbox(
            site_files=sandbox_files,
            prompt=prompt,
            output_paths=["/workspace/output/diagnosis.json"],
            model=sandbox_model,
            label=f"3-diagnose-{task_id}",
        )

    cost_tracker.record(
        "phase_3",
        outputs.get("_summary"),
        task_id=task.get("id"),
        site=task.get("site"),
    )

    diag_json = outputs.get("/workspace/output/diagnosis.json")
    if not diag_json:
        return {
            "root_cause": "diagnosis_failed",
            "explanation": "Diagnosis sandbox did not produce output",
            "suggested_fix": None,
        }

    try:
        return json.loads(diag_json)
    except json.JSONDecodeError:
        return {
            "root_cause": "diagnosis_failed",
            "explanation": f"Invalid JSON from diagnosis sandbox: {diag_json[:200]}",
            "suggested_fix": None,
        }


def _render_diagnosis_prompt(task: dict[str, Any]) -> str:
    """Render the benign diagnosis prompt with real or unknown sanity metadata."""
    sanity_result = _task_sanity_result(task)
    return load_prompt(
        "diagnose-benign-failure",
        validation_command="diagnosis",
    ).replace("{sanity_result}", sanity_result or "unknown")


def _task_sanity_result(task: dict[str, Any]) -> str | None:
    """Extract a normalized sanity result from task metadata when present."""
    candidates = (
        task.get("sanity_result"),
        task.get("sanity_check_result"),
        task.get("sanity_check", {}).get("result")
        if isinstance(task.get("sanity_check"), dict)
        else None,
        task.get("sanity", {}).get("result") if isinstance(task.get("sanity"), dict) else None,
    )
    for value in candidates:
        if value is None:
            continue
        normalized = str(value).strip().lower()
        if normalized in {"pass", "fail", "unknown"}:
            return normalized
    return None


async def fix_loop(
    task: dict[str, Any],
    trajectory_dir: Path,
    profile_path: Path,
    task_dir_root: Path,
    instances: list[BenchmarkInstance],
    agent_factory: Callable[[], BrowserUseAgent],
    max_iterations: int = 2,
    benchmark_root: Path | None = None,
    sandbox_model: str = "claude-sonnet-4-6",
    site_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Iterative diagnosis-fix loop for failed tasks.

    Up to ``max_iterations`` attempts. Exits early on pass or terminal
    root causes (impossible, agent_limitation). Every proposed fix is
    validated against the site profile via ``validate_fix_patch`` before
    ``_apply_fix`` merges it. First rejection triggers a retry of the
    diagnosis sandbox with rejection context appended; second rejection
    (or any non-recoverable classification) falls through to
    ``keep_flagged`` with the original trajectory score retained.
    """
    current_trajectory = trajectory_dir
    current_task = task
    if site_profile is None:
        site_profile = _load_site_profile(profile_path)
    rejection_feedback: str | None = None
    rejections: list[dict[str, Any]] = []

    for iteration in range(max_iterations):
        diagnosis = await diagnose_failure(
            current_task,
            current_trajectory,
            profile_path,
            rejection_feedback=rejection_feedback,
            sandbox_model=sandbox_model,
        )
        # Clear feedback after one retry (we only retry once per plan).
        rejection_feedback = None
        root_cause = diagnosis.get("root_cause", "unknown")

        logger.info(
            "Fix loop iteration %d for task %s: root_cause=%s",
            iteration,
            task.get("id", "?"),
            root_cause,
        )

        if root_cause == "impossible":
            return _with_rejections({"action": "remove", "diagnosis": diagnosis}, rejections)

        if root_cause in ("too_hard", "agent_limitation", "diagnosis_failed"):
            return _with_rejections({"action": "keep_flagged", "diagnosis": diagnosis}, rejections)

        if root_cause in ("reward_bug", "seed_bug"):
            suggested_fix = diagnosis.get("suggested_fix")
            if suggested_fix:
                # Gate the patch before it touches the task. Hallucinated
                # URLs (e.g. POST /api/v4/session on GitLab) and destructive
                # methods are rejected here before seeding executes.
                live_instance = _select_instance_for_task(current_task, instances)
                site_url = live_instance.site_url if live_instance is not None else ""
                patch_errors = validate_fix_patch(suggested_fix, site_profile, site_url)
                if patch_errors:
                    rejection = {
                        "iteration": iteration,
                        "errors": patch_errors,
                        "original_patch": suggested_fix,
                        "retried": iteration + 1 < max_iterations,
                    }
                    logger.warning(
                        "Patch rejected for task %s (iteration %d): %s",
                        task.get("id", "?"),
                        iteration,
                        patch_errors,
                    )
                    if iteration + 1 < max_iterations:
                        rejection["final_action"] = "retry"
                        rejections.append(rejection)
                        rejection_feedback = _format_rejection_feedback(patch_errors, suggested_fix)
                        continue
                    # Out of retries — score as-is.
                    rejection["final_action"] = "keep_flagged"
                    rejections.append(rejection)
                    return _with_rejections(
                        {
                            "action": "keep_flagged",
                            "diagnosis": diagnosis,
                            "patch_rejected": True,
                            "rejection_reasons": patch_errors,
                            "original_sandbox_patch": suggested_fix,
                        },
                        rejections,
                    )

                candidate_task = _apply_fix(current_task, suggested_fix)
                # If the fix sandbox made no effective changes, the remaining
                # failure is an agent limitation, not a task bug.
                if candidate_task == current_task:
                    logger.info(
                        "Diagnosis made no changes to task %s, treating as agent limitation",
                        task.get("id", "?"),
                    )
                    return _with_rejections(
                        {
                            "action": "keep_flagged",
                            "root_cause": "agent_limitation",
                            "diagnosis": diagnosis,
                        },
                        rejections,
                    )
                current_task = candidate_task
                if live_instance is None:
                    return _with_rejections(
                        {
                            "action": "keep_flagged",
                            "diagnosis": diagnosis,
                            "fixed_task": current_task,
                            "rerun": {
                                "passed": False,
                                "message": "No matching live instance available for rerun",
                            },
                        },
                        rejections,
                    )

                rerun_dir = task_dir_root / safe_task_path_component(
                    f"{task.get('id', 'unknown')}__revalidation_{iteration + 1}"
                )
                bound_task = bind_task_to_instance(current_task, live_instance, instances)
                async with task_lock(bound_task):
                    rerun_result = await _rerun_live_task(
                        task=bound_task,
                        instance=live_instance,
                        instances=instances,
                        agent_factory=agent_factory,
                        task_dir=rerun_dir,
                        benchmark_root=benchmark_root,
                        site_profile=site_profile,
                    )
                if rerun_result.get("passed"):
                    return _with_rejections(
                        {
                            "action": "fixed",
                            "diagnosis": diagnosis,
                            "fixed_task": current_task,
                            "rerun": rerun_result,
                            "rerun_task_dir": str(rerun_dir),
                        },
                        rejections,
                    )

                current_trajectory = rerun_dir
                continue

        # Unknown root cause — flag and move on
        return _with_rejections({"action": "keep_flagged", "diagnosis": diagnosis}, rejections)

    return _with_rejections({"action": "keep_flagged", "diagnosis": diagnosis}, rejections)


def _with_rejections(result: dict[str, Any], rejections: list[dict[str, Any]]) -> dict[str, Any]:
    """Attach per-iteration rejection records to a fix_loop result."""
    if rejections:
        result["rejections"] = rejections
    return result


def _load_site_profiles(
    tasks: list[dict[str, Any]], profiles_dir: Path
) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    for site in sorted({str(task.get("site", "")) for task in tasks if task.get("site")}):
        profile_path = profiles_dir / f"BENCHMARK_PROFILE_{site}.json"
        profiles[site] = load_and_validate_profile(site, profile_path)
    return profiles


def _collect_agent_auth_runtime_errors(
    instances: list[BenchmarkInstance],
    site_profiles: dict[str, dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    for instance in instances:
        profile = site_profiles.get(str(instance.site_name))
        if not isinstance(profile, dict) or not profile_requires_agent_auth(profile):
            continue
        if not has_configured_agent_auth(instance.agent_auth):
            errors.append(
                f"site {instance.site_name!r} requires agent_auth in instances.json "
                "because BENCHMARK_PROFILE has authed_user injection surfaces"
            )
    return errors


def _format_rejection_feedback(errors: list[str], suggested_fix: dict[str, Any]) -> str:
    """Render rejection context to paste into the retry diagnosis prompt."""
    try:
        patch_json = json.dumps(suggested_fix, indent=2)
    except (TypeError, ValueError):
        patch_json = str(suggested_fix)
    bullet_errors = "\n".join(f"- {e}" for e in errors)
    return (
        "A previous diagnosis for this task was rejected. The proposed patch "
        "violated safety constraints:\n"
        f"{bullet_errors}\n\n"
        "Original patch:\n"
        f"```json\n{patch_json}\n```\n\n"
        "Common failure mode: proposing API endpoints not observed in "
        "`verification_capabilities`. If the root cause looks like an auth "
        "gap (session missing, login failed, 401/403), classify as "
        "`agent_limitation`, not `seed_bug`."
    )


def _load_site_profile(profile_path: Path) -> dict[str, Any]:
    """Load a site profile, returning ``{}`` when the file is missing/invalid."""
    if not profile_path.exists():
        return {}
    try:
        data = json.loads(profile_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not load site profile %s: %s", profile_path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _apply_fix(task: dict, suggested_fix: dict) -> dict:
    """Apply a suggested fix from diagnosis to a task definition."""
    task = json.loads(json.dumps(task))  # deep copy
    target = suggested_fix.get("target", "")
    patch = suggested_fix.get("patch")

    if target in {"task_removal", "none"}:
        if patch is not None:
            raise ValueError(f"suggested_fix patch must be null for target {target!r}")
        return task

    if target not in {"reward_function", "data_seed"}:
        raise ValueError(f"unsupported suggested_fix target {target!r}")
    if patch is None:
        return task
    if not isinstance(patch, dict):
        raise ValueError("suggested_fix patch must be an object or null")

    subtree_key = "reward_function" if target == "reward_function" else "data_seed"
    base_subtree = task.get(subtree_key, {})
    if not isinstance(base_subtree, dict):
        raise ValueError(f"task[{subtree_key!r}] must be an object")
    task[subtree_key] = _apply_merge_patch(base_subtree, patch)

    return task


def _apply_merge_patch(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Apply JSON Merge Patch semantics to a task subtree."""
    merged = json.loads(json.dumps(base))
    for key, value in patch.items():
        if value is None:
            merged.pop(key, None)
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _apply_merge_patch(merged[key], value)
        else:
            merged[key] = json.loads(json.dumps(value))
    return merged


def _select_instance_for_task(
    task: dict[str, Any], instances: list[BenchmarkInstance]
) -> BenchmarkInstance | None:
    """Pick a live benchmark instance that matches the task site."""
    site = str(task.get("site", "")).lower()
    for instance in instances:
        if instance.site_name.lower() == site:
            return instance
    return None


async def _rerun_live_task(
    task: dict[str, Any],
    instance: BenchmarkInstance,
    instances: list[BenchmarkInstance],
    agent_factory: Callable[[], BrowserUseAgent],
    task_dir: Path,
    benchmark_root: Path | None = None,
    site_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Rerun a patched task against a live instance before validation."""
    agent = agent_factory()
    bound_task = (
        task if task_reset_endpoints(task) else bind_task_to_instance(task, instance, instances)
    )
    try:
        try:
            await agent.setup(instance.site_url)
            result = await run_task(
                bound_task,
                agent,
                instance,
                task_dir,
                benchmark_root=benchmark_root,
                site_profile=site_profile,
            )
        except Exception as e:
            logger.warning("Live rerun failed for task %s: %s", task.get("id", "?"), e)
            return {
                "task_id": task.get("id", "unknown"),
                "passed": False,
                "message": f"rerun failed: {e}",
                "error": repr(e),
                "trajectory_dir": str(task_dir),
            }

        return {**result, "trajectory_dir": str(task_dir)}
    finally:
        await agent.teardown()


def _build_agent_prompt(
    agent_context: dict[str, Any] | None,
    instruction: str,
    start_urls: list[str],
    *,
    task: dict[str, Any] | None = None,
) -> str | None:
    """Compatibility wrapper for the shared agent-prompt builder."""
    return build_agent_prompt(agent_context, instruction, start_urls, task=task)


async def _reset_task_environment(task: dict[str, Any]) -> None:
    """Reset every benchmark instance a task may touch."""
    endpoints = task_reset_endpoints(task)
    if not endpoints:
        return
    await asyncio.gather(*[asyncio.to_thread(_post_reset, ep) for ep in endpoints])
    await asyncio.sleep(2)


# GitLab's POST /init runs `gitlab-ctl reconfigure` which takes 3-5 min.
# Other sites (shopping, reddit) finish in ~5s. 300s covers the worst case.
_RESET_TIMEOUT = 300
_RESET_MAX_RETRIES = 2
_RESET_RETRY_DELAY = 10  # seconds between retries


def _post_reset(endpoint: str) -> None:
    """Call a benchmark reset endpoint with retries for transient failures.

    Retries on connection errors, timeouts, and 5xx responses. The generous
    timeout (300s) is needed because some WebArena sites (especially GitLab)
    block for minutes while reconfiguring.
    """
    last_exc: Exception | None = None
    for attempt in range(1, _RESET_MAX_RETRIES + 1):
        try:
            response = requests.post(endpoint, timeout=_RESET_TIMEOUT)
            if response.status_code >= 500:
                logger.warning(
                    "Reset endpoint %s returned %d on attempt %d/%d",
                    endpoint,
                    response.status_code,
                    attempt,
                    _RESET_MAX_RETRIES,
                )
                last_exc = requests.HTTPError(
                    f"{response.status_code} Server Error for url: {endpoint}",
                    response=response,
                )
                if attempt < _RESET_MAX_RETRIES:
                    time.sleep(_RESET_RETRY_DELAY)
                    continue
                response.raise_for_status()
            # 4xx errors are not retried — they indicate a client-side problem.
            response.raise_for_status()
            return
        except requests.ConnectionError as exc:
            logger.warning(
                "Reset endpoint %s connection error on attempt %d/%d: %s",
                endpoint,
                attempt,
                _RESET_MAX_RETRIES,
                exc,
            )
            last_exc = exc
            if attempt < _RESET_MAX_RETRIES:
                time.sleep(_RESET_RETRY_DELAY)
        except requests.Timeout as exc:
            logger.warning(
                "Reset endpoint %s timed out after %ds on attempt %d/%d",
                endpoint,
                _RESET_TIMEOUT,
                attempt,
                _RESET_MAX_RETRIES,
            )
            last_exc = exc
            if attempt < _RESET_MAX_RETRIES:
                time.sleep(_RESET_RETRY_DELAY)

    raise RuntimeError(
        f"Reset endpoint {endpoint} failed after {_RESET_MAX_RETRIES} attempts"
    ) from last_exc
