"""Phase 1 generate-new-tasks helpers: eligible-site discovery and novel-task generation."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from worldsim.cost_tracker import tracker as cost_tracker
from worldsim.modal_sandbox import (
    preflight_sandbox_environment,
    run_claude_in_sandbox,
    upload_to_volume,
)
from worldsim.phases.phase_1_generate_new_tasks_validation import (
    GeneratedTaskValidationError,
    sort_novel_tasks,
    validate_generated_novel_tasks,
    validate_generated_novel_tasks_detailed,
)
from worldsim.phases.phase_1_route_contracts import (
    build_task_route_contracts,
    route_contracts_digest,
)
from worldsim.phases.phase_1_task_cards import (
    task_card_plan_digest,
    task_card_plan_for_site,
)
from worldsim.placeholders import normalize_site_name
from worldsim.profile_validation import load_and_validate_profile
from worldsim.prompt_corrections import render_validation_feedback
from worldsim.prompt_loading import load_prompt
from worldsim.state import get_state_dir

logger = logging.getLogger(__name__)

DEFAULT_NOVEL_TASKS_PER_SITE = 30
GENERATE_NEW_TASKS_FIX_MAX_ITERATIONS = 2
NOVEL_TASK_OUTPUT_PATH = "/workspace/output/benign_tasks.json"
GENERATE_NEW_TASKS_RESUME_METADATA_PATH = "generate_new_tasks_resume_metadata.json"
SITE_CACHE_METADATA_SUFFIX = ".metadata.json"
GENERATE_NEW_TASKS_CACHE_SCHEMA_VERSION = 6


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

    shared_inputs_fingerprint = compute_generate_new_tasks_shared_inputs_fingerprint(
        benchmark_root=benchmark_root,
        manifest=manifest,
        sandbox_model=sandbox_model,
        task_card_plan=task_card_plan,
    )
    cached_results = _load_all_cached_site_results(
        eligible_sites=eligible_sites,
        output_dir=output_dir,
        shared_inputs_fingerprint=shared_inputs_fingerprint,
        novel_tasks_per_site=novel_tasks_per_site,
        task_card_plan=task_card_plan,
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

    # Fail fast if sandbox auth or image setup is missing before we pay for volume upload.
    await preflight_sandbox_environment()

    logger.info(
        "Phase 1 (generate-new-tasks): generating novel tasks for %d eligible sites",
        len(eligible_sites),
    )
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
                ),
                sandbox_model=sandbox_model,
                novel_tasks_per_site=novel_tasks_per_site,
                task_card_plan=task_card_plan_for_site(task_card_plan, site.site_name),
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


async def generate_new_tasks_for_site(
    *,
    site: EligibleSiteProfile,
    benchmark_volume: Any,
    output_dir: Path,
    cache_fingerprint: str,
    sandbox_model: str = "claude-sonnet-4-6",
    novel_tasks_per_site: int = DEFAULT_NOVEL_TASKS_PER_SITE,
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
        expected_task_count=novel_tasks_per_site,
        route_contracts=route_contracts,
        task_card_plan=task_card_plan,
    )
    if cached_result is not None:
        return cached_result

    logger.info(
        "Phase 1 (generate-new-tasks): launching novel-task sandbox for site %r", site.site_name
    )
    base_prompt = render_generate_benign_tasks_prompt(
        site_name=site.site_name,
        num_tasks=novel_tasks_per_site,
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
            expected_task_count=novel_tasks_per_site,
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
        site_tasks = tasks_by_site.get(site_name)
        if site_tasks is None:
            errors.append(f"merged output is missing novel tasks for eligible site {site_name!r}")
            continue
        _, site_errors = validate_generated_novel_tasks(
            site_tasks,
            site_name=site_name,
            profile=site.profile,
            expected_task_count=expected_task_count,
            route_contracts=build_task_route_contracts(
                site_name=site.site_name,
                profile=site.profile,
            ),
            task_card_plan=task_card_plan_for_site(task_card_plan, site_name),
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
) -> list[SiteGenerateNewTasksResult] | None:
    """Return cached per-site results when every eligible site cache validates."""
    cached_results: list[SiteGenerateNewTasksResult] = []
    for site in eligible_sites:
        agent_context, agent_context_errors = _load_site_agent_context(site)
        if agent_context_errors:
            return None
        cached_result = load_cached_novel_tasks(
            intermediate_path=output_dir / f"novel_tasks_{site.site_name}.json",
            site_name=site.site_name,
            profile=site.profile,
            cache_fingerprint=compute_site_cache_fingerprint(
                shared_inputs_fingerprint=shared_inputs_fingerprint,
                site=site,
                novel_tasks_per_site=novel_tasks_per_site,
                task_card_plan=task_card_plan,
            ),
            expected_agent_context=agent_context,
            expected_task_count=novel_tasks_per_site,
            route_contracts=build_task_route_contracts(
                site_name=site.site_name,
                profile=site.profile,
            ),
            task_card_plan=task_card_plan_for_site(task_card_plan, site.site_name),
        )
        if cached_result is None:
            return None
        cached_results.append(cached_result)
    return cached_results


def render_generate_benign_tasks_prompt(*, site_name: str, num_tasks: int) -> str:
    """Render the generate-new-tasks prompt without interpreting literal example braces."""
    prompt = load_prompt(
        "generate-benign-tasks",
        validation_command=f"benign-tasks --site-name {site_name}",
    )
    return prompt.replace("{site_name}", site_name).replace("{num_tasks}", str(num_tasks))


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
) -> str:
    """Return a content-based digest for shared generate-new-tasks generation inputs."""
    payload = {
        "benchmark_tree_digest": _directory_tree_digest(benchmark_root),
        "manifest": manifest,
        "prompt": load_prompt(
            "generate-benign-tasks",
            validation_command="benign-tasks --site-name {site_name}",
        ),
        "sandbox_model": sandbox_model,
        "task_card_plan_digest": task_card_plan_digest(task_card_plan),
    }
    return _stable_json_digest(payload)


def compute_site_cache_fingerprint(
    *,
    shared_inputs_fingerprint: str,
    site: EligibleSiteProfile,
    novel_tasks_per_site: int = DEFAULT_NOVEL_TASKS_PER_SITE,
    task_card_plan: dict[str, Any] | None = None,
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
