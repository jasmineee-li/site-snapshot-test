"""Phase 1 novel-task cache: cached site results, cache validation, and resume fingerprints."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from warp_taskgen.phase_1.contract_bound_action_api import (
    contract_bound_prompt_inputs,
    contract_bound_tool_schema_digest,
)
from warp_taskgen.phase_1.generated_workflows import (
    generation_prompt_fingerprint_inputs,
    owns_model_generated_content,
)
from warp_taskgen.phase_1.generated_workflows import (
    host_compiled_evaluator_types as feature_host_compiled_evaluator_types,
)
from warp_taskgen.phase_1.novel_task_generation_prompt import (
    CONTRACT_BOUND_ACTION_API_ENV,
    _load_site_agent_context,
    _use_contract_bound_action_api,
)
from warp_taskgen.phase_1.novel_task_site_plan import (
    DEFAULT_NOVEL_TASKS_PER_SITE,
    EligibleSiteProfile,
    _action_counts_for_site,
    _site_requested_count,
)
from warp_taskgen.phase_1.novel_task_validation import (
    sort_novel_tasks,
    validate_generated_novel_tasks,
)
from warp_taskgen.phase_1.task_card_batch_generation import task_card_generation_slices
from warp_taskgen.phases.phase_1_route_contracts import (
    build_task_route_contracts,
    route_contracts_digest,
)
from warp_taskgen.phases.phase_1_task_cards import (
    task_card_plan_digest,
    task_card_plan_for_site,
)
from warp_taskgen.prompt_loading import load_prompt

logger = logging.getLogger(__name__)

SITE_CACHE_METADATA_SUFFIX = ".metadata.json"
GENERATE_NEW_TASKS_CACHE_SCHEMA_VERSION = 8


@dataclass(frozen=True)
class SiteGenerateNewTasksResult:
    site_name: str
    benign_tasks: list[dict[str, Any]]
    errors: list[str]


@dataclass(frozen=True)
class SiteCacheInspection:
    """Read-only explanation of one Phase 1 site-cache decision."""

    status: str
    reason_code: str
    result: SiteGenerateNewTasksResult | None = None


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
    host_compiled_evaluator_types: frozenset[str] = frozenset(),
) -> SiteGenerateNewTasksResult | None:
    """Return a validated cached per-site result when available."""
    inspection = inspect_cached_novel_tasks(
        intermediate_path=intermediate_path,
        site_name=site_name,
        profile=profile,
        cache_fingerprint=cache_fingerprint,
        expected_agent_context=expected_agent_context,
        expected_task_count=expected_task_count,
        route_contracts=route_contracts,
        task_card_plan=task_card_plan,
        host_compiled_evaluator_types=host_compiled_evaluator_types,
    )
    if inspection.result is not None:
        logger.info(
            "Phase 1 (generate-new-tasks): reusing %d cached novel tasks for site %r",
            len(inspection.result.benign_tasks),
            site_name,
        )
        return inspection.result
    if inspection.status != "missing":
        logger.warning(
            "Phase 1 (generate-new-tasks): ignoring cached tasks for site %r (%s)",
            site_name,
            inspection.reason_code,
        )
    return None


def inspect_cached_novel_tasks(
    *,
    intermediate_path: Path,
    site_name: str,
    profile: dict[str, Any],
    cache_fingerprint: str,
    expected_agent_context: dict[str, Any] | None = None,
    expected_task_count: int = DEFAULT_NOVEL_TASKS_PER_SITE,
    route_contracts: dict[str, Any] | None = None,
    task_card_plan: dict[str, Any] | None = None,
    host_compiled_evaluator_types: frozenset[str] = frozenset(),
) -> SiteCacheInspection:
    """Inspect a site cache with the same checks used before runtime reuse."""
    if not intermediate_path.exists():
        return SiteCacheInspection("missing", "cache_artifact_missing")

    metadata_path = _site_cache_metadata_path(intermediate_path)
    if not metadata_path.exists():
        return SiteCacheInspection("stale", "cache_metadata_missing")
    try:
        metadata = json.loads(metadata_path.read_text())
    except json.JSONDecodeError:
        return SiteCacheInspection("invalid", "cache_metadata_invalid")
    if not isinstance(metadata, dict):
        return SiteCacheInspection("invalid", "cache_metadata_invalid")
    if metadata.get("fingerprint") != cache_fingerprint:
        return SiteCacheInspection("stale", "cache_fingerprint_mismatch")

    try:
        cached_tasks = json.loads(intermediate_path.read_text())
    except json.JSONDecodeError:
        return SiteCacheInspection("invalid", "cache_artifact_invalid_json")

    validated_cached, errors = validate_generated_novel_tasks(
        cached_tasks,
        site_name=site_name,
        profile=profile,
        expected_task_count=expected_task_count,
        route_contracts=route_contracts,
        task_card_plan=task_card_plan,
        host_compiled_evaluator_types=host_compiled_evaluator_types,
    )
    if errors:
        return SiteCacheInspection("invalid", "cache_task_validation_failed")
    if expected_agent_context is not None and any(
        task.get("agent_context") != expected_agent_context for task in validated_cached
    ):
        return SiteCacheInspection("stale", "embedded_agent_context_mismatch")

    return SiteCacheInspection(
        "reusable",
        "cache_valid",
        SiteGenerateNewTasksResult(site_name, validated_cached, []),
    )


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
            host_compiled_evaluator_types=feature_host_compiled_evaluator_types(
                site_task_card_plan
            ),
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
            host_compiled_evaluator_types=feature_host_compiled_evaluator_types(
                site_task_card_plan
            ),
        )
        if cached_result is None:
            return None
        cached_results.append(cached_result)
    return cached_results


def _normalize_action_counts(action_counts: dict[str, int] | None) -> dict[str, int] | None:
    if action_counts is None:
        return None
    return {kind: int(action_counts[kind]) for kind in sorted(action_counts)}


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
        **generation_prompt_fingerprint_inputs(task_card_plan),
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

    site_task_card_plan = task_card_plan_for_site(task_card_plan, site.site_name)
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
                site_task_card_plan,
                action_counts,
            )
        ),
        "task_card_plan_digest": task_card_plan_digest(site_task_card_plan),
    }
    site_uses_contract_bound_backend = _use_contract_bound_action_api(site_task_card_plan) or any(
        _use_contract_bound_action_api(card_slice.task_card_plan)
        for card_slice in task_card_generation_slices(
            site_task_card_plan,
            site_name=site.site_name,
        )
    )
    if site_uses_contract_bound_backend:
        # Keep optional prompt inputs local to sites that actually consume the
        # contract-bound backend; model-only Site caches retain their identity.
        prompt_inputs = contract_bound_prompt_inputs()
        if prompt_inputs:
            payload["contract_bound_prompt_inputs"] = prompt_inputs
    if _uses_sliced_model_prompt(site_task_card_plan, site_name=site.site_name):
        # The sliced prompt carries a site-global ordinal range and substantive
        # variation cues.  Keep the global cache schema and unaffected cache
        # payloads stable; only these model-owned slices need invalidation.
        payload["sliced_model_prompt_context_version"] = 1
    return _stable_json_digest(payload)


def _uses_sliced_model_prompt(
    task_card_plan: Mapping[str, Any] | None,
    *,
    site_name: str,
) -> bool:
    """Return whether this site plan emits a model-owned sliced prompt."""

    return any(
        owns_model_generated_content(card)
        for card_slice in task_card_generation_slices(task_card_plan, site_name=site_name)
        for card in card_slice.task_card_plan.get("task_cards", [])
        if isinstance(card, Mapping)
    )


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
