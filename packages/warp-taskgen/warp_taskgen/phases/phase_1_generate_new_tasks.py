"""Phase 1 generate-new-tasks helpers: eligible-site discovery and novel-task generation."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from warp_taskgen.cost_tracker import tracker as cost_tracker
from warp_taskgen.modal_sandbox import (
    preflight_sandbox_environment,
    run_claude_in_sandbox,
    upload_to_volume,
)
from warp_taskgen.phase_1.gitlab_compare_decide_generation import (
    compile_phase1_gitlab_compare_act_task,
    compile_phase1_gitlab_compare_decide_task,
    gitlab_compare_act_generation_contract,
    gitlab_compare_decide_generation_contract,
)
from warp_taskgen.phase_1.novel_task_validation import (
    GeneratedTaskValidationError,
    sort_novel_tasks,
    validate_generated_novel_tasks,
    validate_generated_novel_tasks_detailed,
)
from warp_taskgen.phases.phase_1_contract_bound_action_api import (
    contract_bound_tool_schema_digest,
    generate_contract_bound_action_tasks_api,
)
from warp_taskgen.phases.phase_1_route_contracts import (
    build_task_route_contracts,
    route_contracts_digest,
)
from warp_taskgen.phases.phase_1_task_cards import (
    card_action_kinds,
    card_benign_reward_shape,
    task_card_plan_digest,
    task_card_plan_for_site,
)
from warp_taskgen.placeholders import normalize_site_name
from warp_taskgen.profile_validation import load_and_validate_profile
from warp_taskgen.prompt_corrections import render_validation_feedback
from warp_taskgen.prompt_loading import load_prompt
from warp_taskgen.state import get_state_dir

logger = logging.getLogger(__name__)

DEFAULT_NOVEL_TASKS_PER_SITE = 30
GENERATE_NEW_TASKS_FIX_MAX_ITERATIONS = 2
NOVEL_TASK_OUTPUT_PATH = "/workspace/output/benign_tasks.json"
GENERATE_NEW_TASKS_RESUME_METADATA_PATH = "generate_new_tasks_resume_metadata.json"
SITE_CACHE_METADATA_SUFFIX = ".metadata.json"
GENERATE_NEW_TASKS_CACHE_SCHEMA_VERSION = 7
CONTRACT_BOUND_ACTION_API_ENV = "WORLDSIM_PHASE1_CONTRACT_BOUND_API"
CONTRACT_BOUND_ACTION_API_REQUIRED_PROFILES = frozenset({"tier2_pure_action_paper"})


def _read_only_volume(volume: Any) -> Any:
    """Return a read-only mount when the object supports it."""
    read_only = getattr(volume, "read_only", None)
    return read_only() if callable(read_only) else volume


@dataclass(frozen=True)
class EligibleSiteProfile:
    site_name: str
    profile_path: Path
    profile: dict[str, Any]


@dataclass(frozen=True)
class SiteGenerateNewTasksResult:
    site_name: str
    benign_tasks: list[dict[str, Any]]
    errors: list[str]


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
    uses_sandbox = any(
        not _use_contract_bound_action_api(site_plans[site.site_name]) for site in eligible_sites
    )
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


def load_generate_new_tasks_eligible_sites(
    *,
    profiles_dir: Path,
    manifest_eval_types: list[str],
    site_filter: Iterable[str] | None = None,
) -> list[EligibleSiteProfile]:
    """Return profiles with Phase 4-admissible carrier route families."""
    eligible: list[EligibleSiteProfile] = []
    site_filter_set = _normalize_site_filter(site_filter)

    for profile_path in sorted(profiles_dir.glob("BENCHMARK_PROFILE_*.json")):
        site_name = profile_path.stem.removeprefix("BENCHMARK_PROFILE_")
        if site_filter_set is not None and site_name not in site_filter_set:
            logger.info(
                "Phase 1 (generate-new-tasks): skipping site %r due to --sites filter", site_name
            )
            continue
        profile = load_and_validate_profile(
            site_name,
            profile_path,
            manifest_eval_types=manifest_eval_types,
        )
        route_contracts = build_task_route_contracts(site_name=site_name, profile=profile)
        if route_contracts.get("route_families"):
            eligible.append(
                EligibleSiteProfile(
                    site_name=site_name,
                    profile_path=profile_path,
                    profile=profile,
                )
            )
        else:
            logger.info(
                "Phase 1 (generate-new-tasks): skipping site %r (no Phase 4-admissible carrier route families)",
                site_name,
            )

    return eligible


def _normalize_site_filter(site_filter: Iterable[str] | None) -> set[str] | None:
    if site_filter is None:
        return None
    normalized = {
        normalize_site_name(str(site).strip()) for site in site_filter if str(site).strip()
    }
    return normalized or None


def _fail_if_requested_sites_ineligible(
    *,
    site_filter: Iterable[str] | None,
    eligible_sites: list[EligibleSiteProfile],
) -> None:
    requested = _normalize_site_filter(site_filter)
    if requested is None:
        return
    eligible = {site.site_name for site in eligible_sites}
    missing = sorted(requested - eligible)
    if not missing:
        return
    raise RuntimeError(
        "Phase 1 (generate-new-tasks) cannot satisfy the requested site filter because "
        "the following selected site(s) have no Phase 4-admissible carrier route "
        f"families: {', '.join(missing)}. This usually means Phase 0c did not produce "
        "live inventory for inventory-backed routes, or the site has no strict-WASP "
        "carrier surface under the current route contracts. Fix the profile/inventory "
        "source and rerun Phase 0c rather than silently generating fewer sites."
    )


def _fail_if_task_card_plan_missing_sites(
    *,
    task_card_plan: dict[str, Any] | None,
    eligible_sites: list[EligibleSiteProfile],
) -> None:
    """Reject card-guided generation that would mix with legacy site generation."""
    if task_card_plan is None:
        return
    missing = [
        site.site_name
        for site in eligible_sites
        if task_card_plan_for_site(task_card_plan, site.site_name) is None
    ]
    if not missing:
        return
    active_sites = sorted(
        {
            str(card.get("site")).strip()
            for card in task_card_plan.get("task_cards", [])
            if isinstance(card, dict)
            and str(card.get("status", "active")) == "active"
            and str(card.get("site") or "").strip()
        }
    )
    raise RuntimeError(
        "Phase 1 task-card-guided generation cannot silently fall back to legacy "
        "generation for site(s) without active task cards: "
        + ", ".join(sorted(missing))
        + ". Requested/eligible generated sites must be a subset of the task-card "
        f"plan sites: {', '.join(active_sites) or '<none>'}."
    )


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
    )
    if cached_result is not None:
        return cached_result

    if _use_contract_bound_action_api(task_card_plan):
        logger.info(
            "Phase 1 (generate-new-tasks): launching contract-bound API backend for site %r",
            site.site_name,
        )
        try:
            generated_tasks = await generate_contract_bound_action_tasks_api(
                site_name=site.site_name,
                task_card_plan=task_card_plan or {},
                route_contracts=route_contracts,
                profile=site.profile,
                requested_count=expected_task_count,
                action_counts=action_counts,
                sandbox_model=sandbox_model,
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

    logger.info(
        "Phase 1 (generate-new-tasks): launching novel-task sandbox for site %r", site.site_name
    )
    base_prompt = render_generate_benign_tasks_prompt(
        site_name=site.site_name,
        num_tasks=expected_task_count,
        task_card_plan=task_card_plan,
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

        outputs = await run_claude_in_sandbox(
            site_files=site_files,
            prompt=prompt,
            output_paths=[NOVEL_TASK_OUTPUT_PATH],
            model=sandbox_model,
            volumes={"/workspace/benchmark": _read_only_volume(benchmark_volume)},
            label=f"1b-{site.site_name}",
        )
        cost_tracker.record("phase_1", outputs.get("_summary"), site=site.site_name)

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
        validated_tasks, detailed_errors = validate_generated_novel_tasks_detailed(
            generated_tasks,
            site_name=site.site_name,
            profile=site.profile,
            expected_task_count=expected_task_count,
            route_contracts=route_contracts,
            task_card_plan=task_card_plan,
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
                )
        if not detailed_errors:
            sorted_tasks = sort_novel_tasks(
                _attach_agent_context_to_tasks(validated_tasks, agent_context)
            )
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


def _stamp_new_task_origin(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stamp generated tasks with their Phase 1 source before caching."""
    stamped: list[dict[str, Any]] = []
    for task in tasks:
        item = json.loads(json.dumps(task))
        if isinstance(item, dict):
            item["origin"] = "new_task"
        stamped.append(item)
    return stamped


def _compile_phase1_feature_tasks(
    tasks: list[dict[str, Any]],
    *,
    task_card_plan: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Apply explicitly authored feature generation before per-site caching."""
    if not isinstance(task_card_plan, Mapping):
        return tasks
    compiled: list[dict[str, Any]] = []
    cards = {
        str(card.get("id")): card
        for card in task_card_plan.get("task_cards", [])
        if isinstance(card, Mapping) and isinstance(card.get("id"), str)
    }
    for task in tasks:
        if not isinstance(task, dict):
            compiled.append(task)
            continue
        card = cards.get(str(task.get("task_card_id") or ""))
        if not isinstance(card, Mapping):
            compiled.append(task)
        elif gitlab_compare_act_generation_contract(card) is not None:
            compiled.append(compile_phase1_gitlab_compare_act_task(task, task_card=card))
        elif gitlab_compare_decide_generation_contract(card) is not None:
            compiled.append(compile_phase1_gitlab_compare_decide_task(task, task_card=card))
        else:
            compiled.append(task)
    return compiled


def load_cached_novel_tasks(
    *,
    intermediate_path: Path,
    site_name: str,
    profile: dict[str, Any],
    cache_fingerprint: str,
    expected_agent_context: dict[str, Any] | None = None,
    expected_task_count: int = DEFAULT_NOVEL_TASKS_PER_SITE,
    route_contracts: dict[str, Any] | None = None,
    task_card_plan: dict[str, Any] | None = None,
) -> SiteGenerateNewTasksResult | None:
    """Return a validated cached per-site result when available."""
    if not intermediate_path.exists():
        return None

    metadata = _load_site_cache_metadata(_site_cache_metadata_path(intermediate_path))
    if metadata.get("fingerprint") != cache_fingerprint:
        logger.warning(
            "Phase 1 (generate-new-tasks): ignoring cached tasks for site %r because cache metadata does not match current inputs",
            site_name,
        )
        return None

    try:
        cached_tasks = json.loads(intermediate_path.read_text())
    except json.JSONDecodeError as exc:
        logger.warning(
            "Phase 1 (generate-new-tasks): ignoring invalid cached tasks for site %r at %s: %s",
            site_name,
            intermediate_path,
            exc,
        )
        return None

    validated_cached, errors = validate_generated_novel_tasks(
        cached_tasks,
        site_name=site_name,
        profile=profile,
        expected_task_count=expected_task_count,
        route_contracts=route_contracts,
        task_card_plan=task_card_plan,
    )
    if errors:
        logger.warning(
            "Phase 1 (generate-new-tasks): ignoring invalid cached tasks for site %r: %s",
            site_name,
            "; ".join(errors),
        )
        return None
    if expected_agent_context is not None and any(
        task.get("agent_context") != expected_agent_context for task in validated_cached
    ):
        logger.warning(
            "Phase 1 (generate-new-tasks): ignoring cached tasks for site %r because embedded agent context is missing or stale",
            site_name,
        )
        return None

    logger.info(
        "Phase 1 (generate-new-tasks): reusing %d cached novel tasks for site %r",
        len(validated_cached),
        site_name,
    )
    return SiteGenerateNewTasksResult(site_name, validated_cached, [])


def load_existing_novel_tasks(output_path: Path) -> list[dict[str, Any]] | None:
    """Return existing novel tasks from a merged output file, if present."""
    if not output_path.exists():
        return None

    try:
        merged = json.loads(output_path.read_text())
    except json.JSONDecodeError:
        logger.warning(
            "Phase 1 (generate-new-tasks): ignoring invalid merged output at %s", output_path
        )
        return None

    if not isinstance(merged, list):
        return None

    novel_tasks = [
        task
        for task in merged
        if isinstance(task, dict) and str(task.get("id", "")).startswith("novel_")
    ]
    if not novel_tasks:
        return None
    return sort_novel_tasks(novel_tasks)


def validate_existing_novel_tasks(
    novel_tasks: list[dict[str, Any]],
    *,
    eligible_sites: list[EligibleSiteProfile],
    expected_task_count: int = DEFAULT_NOVEL_TASKS_PER_SITE,
    task_card_plan: dict[str, Any] | None = None,
    action_counts: dict[str, int] | None = None,
) -> list[str]:
    """Validate merged-output novel tasks against the current eligible-site set."""
    tasks_by_site: dict[str, list[dict[str, Any]]] = {}
    for task in novel_tasks:
        site_name = str(task.get("site", ""))
        tasks_by_site.setdefault(site_name, []).append(task)

    eligible_by_site = {site.site_name: site for site in eligible_sites}
    errors: list[str] = []

    unexpected_sites = sorted(set(tasks_by_site) - set(eligible_by_site))
    if unexpected_sites:
        errors.append(
            "merged output contains novel tasks for unexpected sites: "
            + ", ".join(unexpected_sites)
        )

    for site_name, site in eligible_by_site.items():
        site_task_card_plan = task_card_plan_for_site(task_card_plan, site_name)
        site_action_counts = _action_counts_for_site(site_task_card_plan, action_counts)
        site_expected_count = _site_requested_count(
            site_task_card_plan,
            novel_tasks_per_site=expected_task_count,
            action_counts=site_action_counts,
        )
        if site_expected_count <= 0:
            if tasks_by_site.get(site_name):
                errors.append(
                    f"merged output contains novel tasks for site {site_name!r} with requested count 0"
                )
            continue
        site_tasks = tasks_by_site.get(site_name)
        if site_tasks is None:
            errors.append(f"merged output is missing novel tasks for eligible site {site_name!r}")
            continue
        _, site_errors = validate_generated_novel_tasks(
            site_tasks,
            site_name=site_name,
            profile=site.profile,
            expected_task_count=site_expected_count,
            route_contracts=build_task_route_contracts(
                site_name=site.site_name,
                profile=site.profile,
            ),
            task_card_plan=site_task_card_plan,
        )
        errors.extend(site_errors)

    return errors


def _load_all_cached_site_results(
    *,
    eligible_sites: list[EligibleSiteProfile],
    output_dir: Path,
    shared_inputs_fingerprint: str,
    novel_tasks_per_site: int,
    task_card_plan: dict[str, Any] | None = None,
    action_counts: dict[str, int] | None = None,
) -> list[SiteGenerateNewTasksResult] | None:
    """Return cached per-site results when every eligible site cache validates."""
    cached_results: list[SiteGenerateNewTasksResult] = []
    for site in eligible_sites:
        agent_context, agent_context_errors = _load_site_agent_context(site)
        if agent_context_errors:
            return None
        site_task_card_plan = task_card_plan_for_site(task_card_plan, site.site_name)
        site_expected_count = _site_requested_count(
            site_task_card_plan,
            novel_tasks_per_site=novel_tasks_per_site,
            action_counts=_action_counts_for_site(site_task_card_plan, action_counts),
        )
        if site_expected_count <= 0:
            continue
        cached_result = load_cached_novel_tasks(
            intermediate_path=output_dir / f"novel_tasks_{site.site_name}.json",
            site_name=site.site_name,
            profile=site.profile,
            cache_fingerprint=compute_site_cache_fingerprint(
                shared_inputs_fingerprint=shared_inputs_fingerprint,
                site=site,
                novel_tasks_per_site=novel_tasks_per_site,
                task_card_plan=task_card_plan,
                action_counts=action_counts,
            ),
            expected_agent_context=agent_context,
            expected_task_count=site_expected_count,
            route_contracts=build_task_route_contracts(
                site_name=site.site_name,
                profile=site.profile,
            ),
            task_card_plan=site_task_card_plan,
        )
        if cached_result is None:
            return None
        cached_results.append(cached_result)
    return cached_results


def render_generate_benign_tasks_prompt(
    *,
    site_name: str,
    num_tasks: int,
    task_card_plan: dict[str, Any] | None = None,
) -> str:
    """Render the generate-new-tasks prompt without interpreting literal example braces."""
    prompt_name = (
        "generate-benign-action-tasks"
        if _task_card_plan_is_host_action_only(task_card_plan)
        else "generate-benign-tasks"
    )
    prompt = load_prompt(
        prompt_name,
        validation_command=f"benign-tasks --site-name {site_name}",
    )
    return prompt.replace("{site_name}", site_name).replace("{num_tasks}", str(num_tasks))


def _task_card_plan_is_host_action_only(task_card_plan: dict[str, Any] | None) -> bool:
    """Return whether every active task card is action-only utility."""
    if not isinstance(task_card_plan, dict):
        return False
    active_cards = [
        card
        for card in task_card_plan.get("task_cards", [])
        if isinstance(card, dict) and str(card.get("status", "active")) == "active"
    ]
    return bool(active_cards) and all(
        card_benign_reward_shape(card) == "host_action_only" for card in active_cards
    )


def _action_counts_for_site(
    task_card_plan: dict[str, Any] | None,
    action_counts: dict[str, int] | None,
) -> dict[str, int] | None:
    if action_counts is None:
        return None
    if not isinstance(task_card_plan, dict):
        return {}
    available: set[str] = set()
    for card in task_card_plan.get("task_cards", []):
        if not isinstance(card, dict) or str(card.get("status", "active")) != "active":
            continue
        available.update(card_action_kinds(card))
    return {kind: count for kind, count in action_counts.items() if kind in available}


def _fail_if_action_counts_unavailable(
    *,
    site_plans: dict[str, dict[str, Any] | None],
    action_counts: dict[str, int] | None,
) -> None:
    if action_counts is None:
        return
    available: set[str] = set()
    for plan in site_plans.values():
        if not isinstance(plan, dict):
            continue
        for card in plan.get("task_cards", []):
            if not isinstance(card, dict) or str(card.get("status", "active")) != "active":
                continue
            available.update(card_action_kinds(card))
    unavailable = sorted(
        kind for kind, count in action_counts.items() if count > 0 and kind not in available
    )
    if unavailable:
        raise ValueError(
            "requested action kind(s) unavailable for selected sites/task-card plan: "
            + ", ".join(unavailable)
        )


def _site_requested_count(
    task_card_plan: dict[str, Any] | None,
    *,
    novel_tasks_per_site: int,
    action_counts: dict[str, int] | None,
) -> int:
    if action_counts is None:
        return novel_tasks_per_site
    return sum(_action_counts_for_site(task_card_plan, action_counts).values())


def _normalize_action_counts(action_counts: dict[str, int] | None) -> dict[str, int] | None:
    if action_counts is None:
        return None
    return {kind: int(action_counts[kind]) for kind in sorted(action_counts)}


def _use_contract_bound_action_api(task_card_plan: dict[str, Any] | None) -> bool:
    """Return whether Phase 1 should use the contract-bound API backend."""
    if not _task_card_plan_is_host_action_only(task_card_plan):
        return False
    profile = ""
    if isinstance(task_card_plan, dict):
        profile = str(task_card_plan.get("task_capability_profile") or "").strip()
    if profile in CONTRACT_BOUND_ACTION_API_REQUIRED_PROFILES:
        return True
    return os.environ.get(CONTRACT_BOUND_ACTION_API_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _render_generate_new_tasks_correction(errors: list[GeneratedTaskValidationError]) -> str:
    return render_validation_feedback(
        artifact_name="benign_tasks.json",
        errors=[error.to_dict() for error in errors],
        summary=(
            f"{len(errors)} validation error(s). Repair the tasks and return the complete "
            "JSON array again."
        ),
        instruction=(
            "Fix only the listed issues, preserve valid task intent where possible, and "
            "return the complete JSON array. Do not include markdown or commentary."
        ),
    )


def compute_generate_new_tasks_shared_inputs_fingerprint(
    *,
    benchmark_root: Path,
    manifest: dict[str, Any],
    sandbox_model: str = "claude-sonnet-4-6",
    task_card_plan: dict[str, Any] | None = None,
    action_counts: dict[str, int] | None = None,
) -> str:
    """Return a content-based digest for shared generate-new-tasks generation inputs."""
    payload = {
        "benchmark_tree_digest": _directory_tree_digest(benchmark_root),
        "manifest": manifest,
        "prompt": load_prompt(
            "generate-benign-tasks",
            validation_command="benign-tasks --site-name {site_name}",
        ),
        "host_action_prompt": load_prompt(
            "generate-benign-action-tasks",
            validation_command="benign-tasks --site-name {site_name}",
        ),
        "contract_bound_action_tool_schema": contract_bound_tool_schema_digest(),
        "contract_bound_action_backend_env": os.environ.get(CONTRACT_BOUND_ACTION_API_ENV, ""),
        "sandbox_model": sandbox_model,
        "task_card_plan_digest": task_card_plan_digest(task_card_plan),
        "action_counts": _normalize_action_counts(action_counts),
    }
    return _stable_json_digest(payload)


def compute_site_cache_fingerprint(
    *,
    shared_inputs_fingerprint: str,
    site: EligibleSiteProfile,
    novel_tasks_per_site: int = DEFAULT_NOVEL_TASKS_PER_SITE,
    task_card_plan: dict[str, Any] | None = None,
    action_counts: dict[str, int] | None = None,
) -> str:
    """Return a content-based digest for one site's cached novel-task output."""
    agent_context_path = site.profile_path.parent / f"AGENT_CONTEXT_{site.site_name}.json"
    agent_context_digest = None
    if agent_context_path.exists():
        agent_context_digest = sha256(agent_context_path.read_bytes()).hexdigest()

    payload = {
        "cache_schema_version": GENERATE_NEW_TASKS_CACHE_SCHEMA_VERSION,
        "shared_inputs_fingerprint": shared_inputs_fingerprint,
        "site_name": site.site_name,
        "profile": site.profile,
        "route_contracts": route_contracts_digest(
            build_task_route_contracts(site_name=site.site_name, profile=site.profile)
        ),
        "agent_context_digest": agent_context_digest,
        "task_count": novel_tasks_per_site,
        "action_counts": _normalize_action_counts(
            _action_counts_for_site(
                task_card_plan_for_site(task_card_plan, site.site_name),
                action_counts,
            )
        ),
        "task_card_plan_digest": task_card_plan_digest(
            task_card_plan_for_site(task_card_plan, site.site_name)
        ),
    }
    return _stable_json_digest(payload)


def _load_site_agent_context(
    site: EligibleSiteProfile,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Load the sibling AGENT_CONTEXT file when present."""
    agent_context_path = site.profile_path.parent / f"AGENT_CONTEXT_{site.site_name}.json"
    if not agent_context_path.exists():
        return None, []
    try:
        data = json.loads(agent_context_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return None, [f"invalid agent context for site {site.site_name!r}: {exc}"]
    if not isinstance(data, dict):
        return None, [
            f"invalid agent context for site {site.site_name!r}: payload must be an object"
        ]
    return data, []


def _attach_agent_context_to_tasks(
    tasks: list[dict[str, Any]],
    agent_context: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Attach site agent context so later phases replay the same prompt contract.

    The embedded context can include benchmark-issued test credentials. Keeping
    it on the task artifact is intentional because later phases may run without
    direct access to the original Phase 0c files and still need the same login
    and response-format contract.
    """
    if agent_context is None:
        return tasks

    attached: list[dict[str, Any]] = []
    for task in tasks:
        hydrated = json.loads(json.dumps(task))
        hydrated["agent_context"] = json.loads(json.dumps(agent_context))
        attached.append(hydrated)
    return attached


def compute_generate_new_tasks_resume_fingerprint(
    *,
    shared_inputs_fingerprint: str,
    eligible_sites: list[EligibleSiteProfile],
    novel_tasks_per_site: int = DEFAULT_NOVEL_TASKS_PER_SITE,
    task_card_plan: dict[str, Any] | None = None,
    action_counts: dict[str, int] | None = None,
) -> str:
    """Return a deterministic digest for merged-output resume reuse."""
    payload = {
        "shared_inputs_fingerprint": shared_inputs_fingerprint,
        "site_cache_fingerprints": [
            {
                "site_name": site.site_name,
                "fingerprint": compute_site_cache_fingerprint(
                    shared_inputs_fingerprint=shared_inputs_fingerprint,
                    site=site,
                    novel_tasks_per_site=novel_tasks_per_site,
                    task_card_plan=task_card_plan,
                    action_counts=action_counts,
                ),
            }
            for site in sorted(eligible_sites, key=lambda item: item.site_name)
        ],
    }
    return _stable_json_digest(payload)


def _site_cache_metadata_path(intermediate_path: Path) -> Path:
    return intermediate_path.with_suffix(intermediate_path.suffix + SITE_CACHE_METADATA_SUFFIX)


def _write_site_cache_metadata(metadata_path: Path, *, fingerprint: str, site_name: str) -> None:
    metadata_path.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "site_name": site_name,
            },
            indent=2,
        )
    )


def _load_site_cache_metadata(metadata_path: Path) -> dict[str, Any]:
    if not metadata_path.exists():
        return {}
    try:
        payload = json.loads(metadata_path.read_text())
    except json.JSONDecodeError:
        logger.warning(
            "Phase 1 (generate-new-tasks): ignoring invalid site-cache metadata at %s",
            metadata_path,
        )
        return {}
    return payload if isinstance(payload, dict) else {}


def _stable_json_digest(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _directory_tree_digest(root: Path) -> str:
    hasher = sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        rel_path = path.relative_to(root).as_posix().encode("utf-8")
        hasher.update(rel_path)
        hasher.update(b"\0")
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
        hasher.update(b"\0")
    return hasher.hexdigest()
