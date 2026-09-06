"""Phase 1 generate-new-tasks runner: per-site sandbox generation and the batch run."""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from warp_taskgen.cost_tracker import tracker as cost_tracker
from warp_taskgen.modal_sandbox import (
    preflight_sandbox_environment,
    run_claude_in_sandbox,
    sandbox_paid_call_started,
    upload_to_volume,
)
from warp_taskgen.phase_1.contract_bound_action_api import (
    generate_contract_bound_action_tasks_api,
)
from warp_taskgen.phase_1.generated_workflows import (
    host_compiled_evaluator_types as feature_host_compiled_evaluator_types,
)
from warp_taskgen.phase_1.novel_task_cache import (
    SiteGenerateNewTasksResult,
    _load_all_cached_site_results,
    _site_cache_metadata_path,
    _write_site_cache_metadata,
    compute_generate_new_tasks_shared_inputs_fingerprint,
    compute_site_cache_fingerprint,
    load_cached_novel_tasks,
)
from warp_taskgen.phase_1.novel_task_generation_prompt import (
    _attach_agent_context_to_tasks,
    _compile_phase1_feature_tasks,
    _compile_phase1_model_owned_features,
    _load_site_agent_context,
    _render_generate_new_tasks_correction,
    _stamp_new_task_origin,
    _use_contract_bound_action_api,
    render_generate_benign_tasks_prompt,
)
from warp_taskgen.phase_1.novel_task_site_plan import (
    DEFAULT_NOVEL_TASKS_PER_SITE,
    EligibleSiteProfile,
    _action_counts_for_site,
    _fail_if_action_counts_unavailable,
    _fail_if_requested_sites_ineligible,
    _fail_if_task_card_plan_missing_sites,
    _normalize_site_filter,
    _site_requested_count,
    load_generate_new_tasks_eligible_sites,
)
from warp_taskgen.phase_1.novel_task_validation import (
    GeneratedTaskValidationError,
    sort_novel_tasks,
    validate_generated_novel_tasks_detailed,
)
from warp_taskgen.phase_1.task_card_batch_generation import (
    CardSliceResult,
    TaskCardGenerationSlice,
    collect_card_slices,
    rekey_sandbox_task_ids,
    task_card_generation_slices,
)
from warp_taskgen.phases.phase_1_route_contracts import build_task_route_contracts
from warp_taskgen.phases.phase_1_task_cards import (
    task_card_generation_counts,
    task_card_plan_for_site,
)
from warp_taskgen.state import get_state_dir

logger = logging.getLogger(__name__)

GENERATE_NEW_TASKS_FIX_MAX_ITERATIONS = 2
NOVEL_TASK_OUTPUT_PATH = "/workspace/output/benign_tasks.json"
GENERATE_NEW_TASKS_RESUME_METADATA_PATH = "generate_new_tasks_resume_metadata.json"


def _read_only_volume(volume: Any) -> Any:
    """Return a read-only mount when the object supports it."""
    read_only = getattr(volume, "read_only", None)
    return read_only() if callable(read_only) else volume


async def run_generate_new_tasks(
    *,
    manifest: dict[str, Any],
    benchmark_root: Path,
    output_dir: Path,
    sandbox_model: str = "claude-sonnet-4-6",
    site_filter: Iterable[str] | None = None,
    novel_tasks_per_site: int = DEFAULT_NOVEL_TASKS_PER_SITE,
    task_card_plan: dict[str, Any] | None = None,
    action_counts: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Generate novel tasks for eligible sites."""
    state_dir = get_state_dir()
    profiles_dir = state_dir / "phase_0c"
    if not profiles_dir.exists():
        raise FileNotFoundError(
            f"Profiles directory not found at {profiles_dir} — run phase 0c first"
        )

    manifest_eval_types = manifest.get("evaluation", {}).get("eval_types", [])
    site_filter_set = _normalize_site_filter(site_filter)
    eligible_sites = load_generate_new_tasks_eligible_sites(
        profiles_dir=profiles_dir,
        manifest_eval_types=manifest_eval_types,
        site_filter=site_filter_set,
    )
    _fail_if_requested_sites_ineligible(site_filter=site_filter_set, eligible_sites=eligible_sites)
    if not eligible_sites:
        logger.info("Phase 1 (generate-new-tasks): no eligible sites found, nothing to generate")
        return []
    _fail_if_task_card_plan_missing_sites(
        task_card_plan=task_card_plan,
        eligible_sites=eligible_sites,
    )

    shared_inputs_fingerprint = compute_generate_new_tasks_shared_inputs_fingerprint(
        benchmark_root=benchmark_root,
        manifest=manifest,
        sandbox_model=sandbox_model,
        task_card_plan=task_card_plan,
        action_counts=action_counts,
    )
    cached_results = _load_all_cached_site_results(
        eligible_sites=eligible_sites,
        output_dir=output_dir,
        shared_inputs_fingerprint=shared_inputs_fingerprint,
        novel_tasks_per_site=novel_tasks_per_site,
        task_card_plan=task_card_plan,
        action_counts=action_counts,
    )
    if cached_results is not None:
        logger.info(
            "Phase 1 (generate-new-tasks): reusing cached per-site novel tasks for %d eligible sites",
            len(cached_results),
        )
        all_cached_tasks: list[dict[str, Any]] = []
        for result in cached_results:
            all_cached_tasks.extend(result.benign_tasks)
        return sort_novel_tasks(all_cached_tasks)

    logger.info(
        "Phase 1 (generate-new-tasks): generating novel tasks for %d eligible sites",
        len(eligible_sites),
    )
    site_plans = {
        site.site_name: task_card_plan_for_site(task_card_plan, site.site_name)
        for site in eligible_sites
    }
    _fail_if_action_counts_unavailable(
        site_plans=site_plans,
        action_counts=action_counts,
    )
    requested_counts = {
        site.site_name: _site_requested_count(
            site_plans[site.site_name],
            novel_tasks_per_site=novel_tasks_per_site,
            action_counts=_action_counts_for_site(
                site_plans[site.site_name],
                action_counts,
            ),
        )
        for site in eligible_sites
    }
    sites_needing_paid_work = [
        site for site in eligible_sites if requested_counts[site.site_name] > 0
    ]
    uses_sandbox = any(
        not _use_contract_bound_action_api(site_plans[site.site_name])
        for site in sites_needing_paid_work
    )
    # A malformed prior report must never be replaced by an empty in-memory
    # tracker after another paid Phase 1 dispatch. Cache hits return above, so
    # this gate has no effect on runs that do not need a paid call.
    cost_report_path = state_dir / "cost_report.json"
    if sites_needing_paid_work:
        cost_tracker.ensure_phase1_paid_dispatch_allowed(cost_report_path)
        # The CLI normally loads this report before dispatch. Keep direct Python
        # callers on the same preservation path before any immediate save.
        cost_tracker.load(cost_report_path)
    benchmark_volume = None
    if uses_sandbox:
        # Fail fast if sandbox auth or image setup is missing before we pay for volume upload.
        await preflight_sandbox_environment()
        benchmark_volume = await upload_to_volume(Path(benchmark_root).resolve())

    results = await asyncio.gather(
        *[
            generate_new_tasks_for_site(
                site=site,
                benchmark_volume=benchmark_volume,
                output_dir=output_dir,
                cache_fingerprint=compute_site_cache_fingerprint(
                    shared_inputs_fingerprint=shared_inputs_fingerprint,
                    site=site,
                    novel_tasks_per_site=novel_tasks_per_site,
                    task_card_plan=task_card_plan,
                    action_counts=action_counts,
                ),
                sandbox_model=sandbox_model,
                novel_tasks_per_site=novel_tasks_per_site,
                action_counts=_action_counts_for_site(
                    site_plans[site.site_name],
                    action_counts,
                ),
                task_card_plan=site_plans[site.site_name],
            )
            for site in eligible_sites
        ],
        return_exceptions=True,
    )

    all_novel_tasks: list[dict[str, Any]] = []
    failures: list[str] = []
    for result in results:
        if isinstance(result, BaseException):
            failures.append(str(result))
            continue
        if result.errors:
            failures.extend(f"{result.site_name}: {error}" for error in result.errors)
            continue
        logger.info(
            "Phase 1 (generate-new-tasks): site %r produced %d novel tasks",
            result.site_name,
            len(result.benign_tasks),
        )
        all_novel_tasks.extend(result.benign_tasks)

    if failures:
        raise RuntimeError(
            "Phase 1 (generate-new-tasks) failed because one or more sites did not produce valid novel tasks:\n"
            + "\n".join(f"  - {failure}" for failure in failures)
        )

    return sort_novel_tasks(all_novel_tasks)


async def generate_new_tasks_for_site(
    *,
    site: EligibleSiteProfile,
    benchmark_volume: Any | None,
    output_dir: Path,
    cache_fingerprint: str,
    sandbox_model: str = "claude-sonnet-4-6",
    novel_tasks_per_site: int = DEFAULT_NOVEL_TASKS_PER_SITE,
    action_counts: dict[str, int] | None = None,
    task_card_plan: dict[str, Any] | None = None,
    _allow_task_card_slicing: bool = True,
    _write_site_cache: bool = True,
    _task_number_start: int | None = None,
) -> SiteGenerateNewTasksResult:
    """Generate and validate novel tasks for one site."""
    intermediate_path = output_dir / f"novel_tasks_{site.site_name}.json"
    agent_context, agent_context_errors = _load_site_agent_context(site)
    if agent_context_errors:
        return SiteGenerateNewTasksResult(site.site_name, [], agent_context_errors)
    route_contracts = build_task_route_contracts(site_name=site.site_name, profile=site.profile)
    route_contracts_path = output_dir / f"TASK_ROUTE_CONTRACTS_{site.site_name}.json"
    route_contracts_path.write_text(json.dumps(route_contracts, indent=2))
    task_card_plan_path = output_dir / f"TASK_CARD_PLAN_{site.site_name}.json"
    if task_card_plan is not None:
        task_card_plan_path.write_text(json.dumps(task_card_plan, indent=2, sort_keys=True))
    expected_task_count = _site_requested_count(
        task_card_plan,
        novel_tasks_per_site=novel_tasks_per_site,
        action_counts=action_counts,
    )
    if expected_task_count <= 0:
        logger.info(
            "Phase 1 (generate-new-tasks): skipping site %r because explicit action counts request zero rows",
            site.site_name,
        )
        return SiteGenerateNewTasksResult(site.site_name, [], [])
    feature_evaluator_types = feature_host_compiled_evaluator_types(task_card_plan)
    if site.site_name in {"gitlab", "reddit"} and not route_contracts.get("route_families"):
        logger.info(
            "Phase 1 (generate-new-tasks): skipping site %r because no eligible route families remain after core-surface filtering",
            site.site_name,
        )
        return SiteGenerateNewTasksResult(site.site_name, [], [])
    cached_result = load_cached_novel_tasks(
        intermediate_path=intermediate_path,
        site_name=site.site_name,
        profile=site.profile,
        cache_fingerprint=cache_fingerprint,
        expected_agent_context=agent_context,
        expected_task_count=expected_task_count,
        route_contracts=route_contracts,
        task_card_plan=task_card_plan,
        host_compiled_evaluator_types=feature_evaluator_types,
    )
    if cached_result is not None:
        return cached_result

    card_slices = task_card_generation_slices(
        task_card_plan,
        site_name=site.site_name,
    )
    if _allow_task_card_slicing and card_slices:
        return await _generate_new_tasks_for_site_card_slices(
            site=site,
            benchmark_volume=benchmark_volume,
            output_dir=output_dir,
            cache_fingerprint=cache_fingerprint,
            sandbox_model=sandbox_model,
            novel_tasks_per_site=novel_tasks_per_site,
            action_counts=action_counts,
            task_card_plan=task_card_plan,
            card_slices=card_slices,
            route_contracts=route_contracts,
            feature_evaluator_types=feature_evaluator_types,
            write_site_cache=_write_site_cache,
        )

    if _use_contract_bound_action_api(task_card_plan):
        cost_report_path = _phase1_cost_report_path(output_dir)
        cost_tracker.ensure_phase1_paid_dispatch_allowed(cost_report_path)
        logger.info(
            "Phase 1 (generate-new-tasks): launching contract-bound API backend for site %r",
            site.site_name,
        )
        try:
            api_kwargs: dict[str, Any] = {
                "site_name": site.site_name,
                "task_card_plan": task_card_plan or {},
                "route_contracts": route_contracts,
                "profile": site.profile,
                "requested_count": expected_task_count,
                "action_counts": action_counts,
                "sandbox_model": sandbox_model,
            }
            if _task_number_start is not None:
                api_kwargs["task_number_start"] = _task_number_start
            api_kwargs["cost_report_path"] = cost_report_path
            generated_tasks = await generate_contract_bound_action_tasks_api(
                **api_kwargs,
            )
        except ValueError as exc:
            return SiteGenerateNewTasksResult(site.site_name, [], [str(exc)])
        validated_tasks, detailed_errors = validate_generated_novel_tasks_detailed(
            generated_tasks,
            site_name=site.site_name,
            profile=site.profile,
            expected_task_count=expected_task_count,
            route_contracts=route_contracts,
            task_card_plan=task_card_plan,
            host_compiled_evaluator_types=feature_evaluator_types,
        )
        if detailed_errors:
            return SiteGenerateNewTasksResult(
                site.site_name,
                [],
                [error.render() for error in detailed_errors],
            )
        try:
            compiled_tasks = _compile_phase1_feature_tasks(
                validated_tasks,
                task_card_plan=task_card_plan,
            )
        except ValueError as exc:
            return SiteGenerateNewTasksResult(site.site_name, [], [str(exc)])
        validated_tasks, detailed_errors = validate_generated_novel_tasks_detailed(
            compiled_tasks,
            site_name=site.site_name,
            profile=site.profile,
            expected_task_count=expected_task_count,
            route_contracts=route_contracts,
            task_card_plan=task_card_plan,
            host_compiled_evaluator_types=feature_evaluator_types,
        )
        if detailed_errors:
            return SiteGenerateNewTasksResult(
                site.site_name,
                [],
                [error.render() for error in detailed_errors],
            )
        sorted_tasks = sort_novel_tasks(
            _attach_agent_context_to_tasks(validated_tasks, agent_context)
        )
        if _write_site_cache:
            intermediate_path.write_text(json.dumps(sorted_tasks, indent=2))
            _write_site_cache_metadata(
                _site_cache_metadata_path(intermediate_path),
                fingerprint=cache_fingerprint,
                site_name=site.site_name,
            )
        logger.info(
            "Phase 1 (generate-new-tasks): site %r contract-bound API completed",
            site.site_name,
        )
        return SiteGenerateNewTasksResult(site.site_name, sorted_tasks, [])

    if benchmark_volume is None:
        return SiteGenerateNewTasksResult(
            site.site_name,
            [],
            ["sandbox generation backend requires a benchmark volume"],
        )

    cost_report_path = _phase1_cost_report_path(output_dir)
    cost_tracker.ensure_phase1_paid_dispatch_allowed(cost_report_path)

    logger.info(
        "Phase 1 (generate-new-tasks): launching novel-task sandbox for site %r", site.site_name
    )
    base_prompt = render_generate_benign_tasks_prompt(
        site_name=site.site_name,
        num_tasks=expected_task_count,
        task_card_plan=task_card_plan,
        _task_number_start=_task_number_start,
    )
    prompt = base_prompt
    last_errors: list[str] = []

    for attempt in range(1 + GENERATE_NEW_TASKS_FIX_MAX_ITERATIONS):
        site_files: dict[str, str] = {
            "/workspace/profile/BENCHMARK_PROFILE.json": str(site.profile_path),
        }
        site_files["/workspace/profile/TASK_ROUTE_CONTRACTS.json"] = str(route_contracts_path)
        if task_card_plan is not None:
            site_files["/workspace/profile/TASK_CARD_PLAN.json"] = str(task_card_plan_path)
        # Pass agent context to sandbox so generated tasks align with response format
        agent_context_path = site.profile_path.parent / f"AGENT_CONTEXT_{site.site_name}.json"
        if agent_context_path.exists():
            site_files["/workspace/profile/AGENT_CONTEXT.json"] = str(agent_context_path)

        try:
            outputs = await run_claude_in_sandbox(
                site_files=site_files,
                prompt=prompt,
                output_paths=[NOVEL_TASK_OUTPUT_PATH],
                model=sandbox_model,
                volumes={"/workspace/benchmark": _read_only_volume(benchmark_volume)},
                label=f"1b-{site.site_name}",
            )
        except Exception as exc:
            # This boundary is the only place that can observe a sandbox paid
            # call failure. Setup/preflight failures occur before the SDK
            # process boundary and must not be counted as paid observations.
            if sandbox_paid_call_started(exc):
                cost_tracker.record_and_save(
                    "phase_1",
                    None,
                    cost_report_path,
                    site=site.site_name,
                )
            raise
        if not isinstance(outputs, Mapping):
            cost_tracker.record_and_save(
                "phase_1",
                None,
                cost_report_path,
                site=site.site_name,
            )
            raise TypeError("sandbox paid response must be a mapping")
        # Persist before reading or validating any generated payload. A later
        # validation failure must not erase this returned paid response.
        cost_tracker.record_and_save(
            "phase_1",
            outputs.get("_summary"),
            cost_report_path,
            site=site.site_name,
        )

        payload = outputs.get(NOVEL_TASK_OUTPUT_PATH)
        if not payload:
            return SiteGenerateNewTasksResult(
                site.site_name,
                [],
                ["sandbox did not produce benign_tasks.json"],
            )

        try:
            raw_tasks = json.loads(payload)
        except json.JSONDecodeError as exc:
            return SiteGenerateNewTasksResult(site.site_name, [], [f"invalid sandbox JSON: {exc}"])

        generated_tasks = (
            _stamp_new_task_origin(raw_tasks) if isinstance(raw_tasks, list) else raw_tasks
        )
        if isinstance(generated_tasks, list) and _task_number_start is not None:
            try:
                generated_tasks = rekey_sandbox_task_ids(
                    generated_tasks,
                    site_name=site.site_name,
                    task_number_start=_task_number_start,
                )
            except ValueError as exc:
                return SiteGenerateNewTasksResult(site.site_name, [], [str(exc)])
        try:
            generated_tasks = _compile_phase1_model_owned_features(
                generated_tasks,
                task_card_plan=task_card_plan,
            )
        except (TypeError, ValueError) as exc:
            detailed_errors = [
                GeneratedTaskValidationError(
                    code="FEATURE_GENERATION_CONTRACT_INVALID",
                    path="$",
                    message=str(exc),
                )
            ]
            validated_tasks = []
        else:
            validated_tasks, detailed_errors = validate_generated_novel_tasks_detailed(
                generated_tasks,
                site_name=site.site_name,
                profile=site.profile,
                expected_task_count=expected_task_count,
                route_contracts=route_contracts,
                task_card_plan=task_card_plan,
                host_compiled_evaluator_types=feature_evaluator_types,
            )
        if not detailed_errors:
            try:
                compiled_tasks = _compile_phase1_feature_tasks(
                    validated_tasks,
                    task_card_plan=task_card_plan,
                )
            except ValueError as exc:
                detailed_errors = [
                    GeneratedTaskValidationError(
                        code="FEATURE_GENERATION_CONTRACT_INVALID",
                        path="$",
                        message=str(exc),
                    )
                ]
            else:
                validated_tasks, detailed_errors = validate_generated_novel_tasks_detailed(
                    compiled_tasks,
                    site_name=site.site_name,
                    profile=site.profile,
                    expected_task_count=expected_task_count,
                    route_contracts=route_contracts,
                    task_card_plan=task_card_plan,
                    host_compiled_evaluator_types=feature_evaluator_types,
                )
        if not detailed_errors:
            sorted_tasks = sort_novel_tasks(
                _attach_agent_context_to_tasks(validated_tasks, agent_context)
            )
            if _write_site_cache:
                intermediate_path.write_text(json.dumps(sorted_tasks, indent=2))
                _write_site_cache_metadata(
                    _site_cache_metadata_path(intermediate_path),
                    fingerprint=cache_fingerprint,
                    site_name=site.site_name,
                )
            logger.info("Phase 1 (generate-new-tasks): site %r sandbox completed", site.site_name)
            return SiteGenerateNewTasksResult(site.site_name, sorted_tasks, [])

        last_errors = [error.render() for error in detailed_errors]
        if attempt < GENERATE_NEW_TASKS_FIX_MAX_ITERATIONS:
            logger.warning(
                "Phase 1 (generate-new-tasks): site %r output failed validation, retrying (%d/%d): %s",
                site.site_name,
                attempt + 1,
                GENERATE_NEW_TASKS_FIX_MAX_ITERATIONS,
                "; ".join(last_errors),
            )
            correction = _render_generate_new_tasks_correction(detailed_errors)
            prompt = base_prompt + correction

    logger.info("Phase 1 (generate-new-tasks): site %r sandbox completed", site.site_name)
    return SiteGenerateNewTasksResult(
        site.site_name,
        [],
        last_errors or ["sandbox produced no novel tasks"],
    )


def _phase1_cost_report_path(output_dir: Path) -> Path:
    """Resolve the state-root report for normal and temporary slice outputs."""

    resolved = output_dir.resolve()
    for parent in (resolved, *resolved.parents):
        if parent.name == "phase_1":
            return parent.parent / "cost_report.json"
    return get_state_dir() / "cost_report.json"


async def _generate_new_tasks_for_site_card_slices(
    *,
    site: EligibleSiteProfile,
    benchmark_volume: Any | None,
    output_dir: Path,
    cache_fingerprint: str,
    sandbox_model: str,
    novel_tasks_per_site: int,
    action_counts: dict[str, int] | None,
    task_card_plan: dict[str, Any],
    card_slices: tuple[TaskCardGenerationSlice, ...],
    route_contracts: dict[str, Any],
    feature_evaluator_types: frozenset[str],
    write_site_cache: bool,
) -> SiteGenerateNewTasksResult:
    """Generate explicit card slices, then promote one validated site batch."""

    async def generate_slice(
        card_slice: TaskCardGenerationSlice,
        index: int,
    ) -> CardSliceResult:
        # Child calls need isolated prompt/route files, but their temporary
        # outputs are never promoted as caches or manifests.
        with tempfile.TemporaryDirectory(
            prefix=f".phase1-card-{index + 1}-",
            dir=output_dir,
        ) as temporary_dir:
            result = await generate_new_tasks_for_site(
                site=site,
                benchmark_volume=benchmark_volume,
                output_dir=Path(temporary_dir),
                cache_fingerprint=cache_fingerprint,
                sandbox_model=sandbox_model,
                novel_tasks_per_site=novel_tasks_per_site,
                action_counts=(
                    None
                    if task_card_generation_counts(card_slice.task_card_plan) is not None
                    else _action_counts_for_site(card_slice.task_card_plan, action_counts)
                ),
                task_card_plan=card_slice.task_card_plan,
                _allow_task_card_slicing=False,
                _write_site_cache=False,
                _task_number_start=card_slice.task_number_start,
            )
            return CardSliceResult(result.benign_tasks, result.errors)

    expected_task_count = _site_requested_count(
        task_card_plan,
        novel_tasks_per_site=novel_tasks_per_site,
        action_counts=action_counts,
    )
    batch = await collect_card_slices(
        card_slices=card_slices,
        generate_slice=generate_slice,
        expected_task_count=expected_task_count,
        site_name=site.site_name,
        profile=site.profile,
        route_contracts=route_contracts,
        task_card_plan=task_card_plan,
        host_compiled_evaluator_types=feature_evaluator_types,
    )
    if batch.errors:
        return SiteGenerateNewTasksResult(site.site_name, [], batch.errors)
    if write_site_cache:
        intermediate_path = output_dir / f"novel_tasks_{site.site_name}.json"
        intermediate_path.write_text(json.dumps(batch.benign_tasks, indent=2))
        _write_site_cache_metadata(
            _site_cache_metadata_path(intermediate_path),
            fingerprint=cache_fingerprint,
            site_name=site.site_name,
        )
    return SiteGenerateNewTasksResult(site.site_name, batch.benign_tasks, [])
