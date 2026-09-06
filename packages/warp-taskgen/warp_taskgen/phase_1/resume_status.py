"""Read-only Phase 1 generation reuse inspection for operator status."""

from __future__ import annotations

import json
import os
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from warp_taskgen.phase_1.generated_workflows import (
    host_compiled_evaluator_types as feature_host_compiled_evaluator_types,
)
from warp_taskgen.phase_1.novel_task_cache import (
    compute_generate_new_tasks_resume_fingerprint,
    compute_generate_new_tasks_shared_inputs_fingerprint,
    compute_site_cache_fingerprint,
    inspect_cached_novel_tasks,
    load_existing_novel_tasks,
    validate_existing_novel_tasks,
)
from warp_taskgen.phase_1.novel_task_generation_prompt import (
    CONTRACT_BOUND_ACTION_API_ENV,
    _load_site_agent_context,
)
from warp_taskgen.phase_1.novel_task_site_plan import (
    DEFAULT_NOVEL_TASKS_PER_SITE,
    EligibleSiteProfile,
    _action_counts_for_site,
    _fail_if_action_counts_unavailable,
    _fail_if_requested_sites_ineligible,
    _fail_if_task_card_plan_missing_sites,
    _site_requested_count,
    load_generate_new_tasks_eligible_sites,
)
from warp_taskgen.phase_1.novel_task_validation import sort_novel_tasks
from warp_taskgen.phases.phase_1_route_contracts import build_task_route_contracts
from warp_taskgen.phases.phase_1_task_cards import (
    load_or_compile_task_card_plan,
    task_card_plan_for_site,
)
from warp_taskgen.phases.phase_1_tasks import (
    _parse_phase_1_action_counts,
    _parse_sites_filter,
)


def inspect_phase1_generation_resume(
    run_root: Path,
    state: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Explain exact Phase 1 site-cache reuse without executing pipeline work."""
    if (
        state.get("step") != "phase_1"
        or state.get("status") not in {"running", "failed"}
        or not state.get("generate_novel")
    ):
        return None

    projection = _base_projection(run_root)
    try:
        manifest = _load_manifest(run_root, state)
        benchmark_root = _benchmark_root(state, manifest)
        site_filter = _parse_sites_filter(state.get("sites"))
        task_card_plan = load_or_compile_task_card_plan(
            path=_optional_path(state.get("task_card_plan_path")),
            task_capability_profile=_optional_text(state.get("task_capability_profile")),
            sites=site_filter,
        )
        action_counts = _resolve_action_counts(state)
        evaluation = manifest.get("evaluation")
        if not isinstance(evaluation, Mapping):
            raise ValueError("manifest evaluation must be an object")
        eval_types = evaluation.get("eval_types", [])
        if not isinstance(eval_types, list):
            raise ValueError("manifest evaluation.eval_types must be a list")
        eligible_sites = load_generate_new_tasks_eligible_sites(
            profiles_dir=run_root / "phase_0c",
            manifest_eval_types=eval_types,
            site_filter=site_filter,
        )
        if not eligible_sites:
            raise ValueError("no Phase 1 generation-eligible sites")
        _fail_if_requested_sites_ineligible(
            site_filter=site_filter,
            eligible_sites=eligible_sites,
        )
        _fail_if_task_card_plan_missing_sites(
            task_card_plan=task_card_plan,
            eligible_sites=eligible_sites,
        )
        site_plans = {
            site.site_name: task_card_plan_for_site(task_card_plan, site.site_name)
            for site in eligible_sites
        }
        _fail_if_action_counts_unavailable(
            site_plans=site_plans,
            action_counts=action_counts,
        )
        sandbox_model = str(state.get("sandbox_model") or "claude-sonnet-4-6")
        novel_tasks_per_site = int(
            state.get("novel_tasks_per_site") or DEFAULT_NOVEL_TASKS_PER_SITE
        )
        shared_fingerprint = compute_generate_new_tasks_shared_inputs_fingerprint(
            benchmark_root=benchmark_root,
            manifest=manifest,
            sandbox_model=sandbox_model,
            task_card_plan=task_card_plan,
            action_counts=action_counts,
        )
        merged_output, merged_novel_tasks = _inspect_merged_output(
            run_root=run_root,
            eligible_sites=eligible_sites,
            shared_fingerprint=shared_fingerprint,
            novel_tasks_per_site=novel_tasks_per_site,
            task_card_plan=task_card_plan,
            action_counts=action_counts,
        )
    except (
        AttributeError,
        FileNotFoundError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        projection.update(status="unavailable", reason_code=_safe_reason(exc))
        return projection

    site_rows: list[dict[str, Any]] = []
    reusable_site_tasks: list[dict[str, Any]] = []
    all_requested_site_caches_reusable = True
    for site in eligible_sites:
        expected_count = 0
        try:
            site_plan = site_plans[site.site_name]
            site_action_counts = _action_counts_for_site(site_plan, action_counts)
            expected_count = _site_requested_count(
                site_plan,
                novel_tasks_per_site=novel_tasks_per_site,
                action_counts=site_action_counts,
            )
            agent_context, context_errors = _load_site_agent_context(site)
            if context_errors:
                raise ValueError(context_errors[0])
            inspection = inspect_cached_novel_tasks(
                intermediate_path=run_root / "phase_1" / f"novel_tasks_{site.site_name}.json",
                site_name=site.site_name,
                profile=site.profile,
                cache_fingerprint=compute_site_cache_fingerprint(
                    shared_inputs_fingerprint=shared_fingerprint,
                    site=site,
                    novel_tasks_per_site=novel_tasks_per_site,
                    task_card_plan=task_card_plan,
                    action_counts=action_counts,
                ),
                expected_agent_context=agent_context,
                expected_task_count=expected_count,
                route_contracts=build_task_route_contracts(
                    site_name=site.site_name,
                    profile=site.profile,
                ),
                task_card_plan=site_plan,
                host_compiled_evaluator_types=feature_host_compiled_evaluator_types(site_plan),
            )
            reusable_count = expected_count if inspection.status == "reusable" else 0
            if expected_count > 0:
                if inspection.result is None:
                    all_requested_site_caches_reusable = False
                else:
                    reusable_site_tasks.extend(inspection.result.benign_tasks)
            site_rows.append(
                {
                    "site": site.site_name,
                    "cache_status": inspection.status,
                    "reason_code": inspection.reason_code,
                    "requested_tasks": expected_count,
                    "reusable_tasks": reusable_count,
                    "remaining_tasks": expected_count - reusable_count,
                }
            )
        except (
            AttributeError,
            KeyError,
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            if expected_count > 0:
                all_requested_site_caches_reusable = False
            site_rows.append(
                {
                    "site": site.site_name,
                    "cache_status": "unavailable",
                    "reason_code": _safe_reason(exc),
                    "requested_tasks": expected_count,
                    "reusable_tasks": 0,
                    "remaining_tasks": expected_count,
                }
            )

    requested = sum(int(row["requested_tasks"]) for row in site_rows)
    site_cache_reusable = sum(int(row["reusable_tasks"]) for row in site_rows)
    if (
        merged_output["reason_code"]
        in {
            "merged_resume_metadata_missing",
            "merged_resume_metadata_invalid",
            "merged_resume_fingerprint_mismatch",
        }
        and merged_novel_tasks is not None
        and all_requested_site_caches_reusable
        and sort_novel_tasks(reusable_site_tasks) == sort_novel_tasks(merged_novel_tasks)
    ):
        merged_output = {
            "status": "reusable",
            "reason_code": "merged_matches_current_site_caches",
        }
    if merged_output["status"] == "reusable":
        reusable = requested
        reuse_source = "merged_output"
    elif merged_output["status"] == "unavailable":
        reusable = 0
        reuse_source = "none"
    else:
        reusable = site_cache_reusable
        reuse_source = "site_caches" if reusable else "none"
    projection.update(
        status="inspected",
        requested_tasks=requested,
        reusable_tasks=reusable,
        remaining_tasks=requested - reusable,
        reuse_source=reuse_source,
        sites=site_rows,
        merged_output=merged_output,
        merged_output_present=(run_root / "phase_1" / "benign_tasks.json").exists(),
        resume_metadata_present=(
            run_root / "phase_1" / "generate_new_tasks_resume_metadata.json"
        ).exists(),
    )
    if merged_output["status"] == "unavailable":
        projection["resume_blocker"] = merged_output["reason_code"]
    return projection


def _inspect_merged_output(
    *,
    run_root: Path,
    eligible_sites: list[EligibleSiteProfile],
    shared_fingerprint: str,
    novel_tasks_per_site: int,
    task_card_plan: dict[str, Any] | None,
    action_counts: dict[str, int] | None,
) -> tuple[dict[str, str], list[dict[str, Any]] | None]:
    output_path = run_root / "phase_1" / "benign_tasks.json"
    if not output_path.exists():
        return {"status": "missing", "reason_code": "merged_output_missing"}, None

    try:
        novel_tasks = load_existing_novel_tasks(output_path)
    except UnicodeDecodeError:
        return {"status": "unavailable", "reason_code": "merged_output_unreadable"}, None
    except OSError:
        return {"status": "unavailable", "reason_code": "merged_output_unreadable"}, None
    if novel_tasks is None:
        return {"status": "invalid", "reason_code": "merged_output_invalid"}, None
    validation_errors = validate_existing_novel_tasks(
        novel_tasks,
        eligible_sites=eligible_sites,
        expected_task_count=novel_tasks_per_site,
        task_card_plan=task_card_plan,
        action_counts=action_counts,
    )
    if validation_errors:
        return {"status": "invalid", "reason_code": "merged_task_validation_failed"}, None

    metadata_path = run_root / "phase_1" / "generate_new_tasks_resume_metadata.json"
    if not metadata_path.exists():
        return (
            {"status": "stale", "reason_code": "merged_resume_metadata_missing"},
            novel_tasks,
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return (
            {"status": "invalid", "reason_code": "merged_resume_metadata_invalid"},
            novel_tasks,
        )
    except UnicodeDecodeError:
        return (
            {"status": "unavailable", "reason_code": "merged_resume_metadata_unreadable"},
            novel_tasks,
        )
    except OSError:
        return (
            {"status": "unavailable", "reason_code": "merged_resume_metadata_unreadable"},
            novel_tasks,
        )
    if not isinstance(metadata, dict):
        return (
            {"status": "invalid", "reason_code": "merged_resume_metadata_invalid"},
            novel_tasks,
        )

    expected_fingerprint = compute_generate_new_tasks_resume_fingerprint(
        shared_inputs_fingerprint=shared_fingerprint,
        eligible_sites=eligible_sites,
        novel_tasks_per_site=novel_tasks_per_site,
        task_card_plan=task_card_plan,
        action_counts=action_counts,
    )
    if metadata.get("fingerprint") != expected_fingerprint:
        return (
            {"status": "stale", "reason_code": "merged_resume_fingerprint_mismatch"},
            novel_tasks,
        )
    return (
        {"status": "reusable", "reason_code": "merged_resume_fingerprint_matches"},
        novel_tasks,
    )


def _base_projection(run_root: Path) -> dict[str, Any]:
    env_value = os.environ.get(CONTRACT_BOUND_ACTION_API_ENV)
    return {
        "status": "unavailable",
        "authority": "advisory",
        "requested_tasks": 0,
        "reusable_tasks": 0,
        "remaining_tasks": 0,
        "sites": [],
        "environment_binding": {
            "name": CONTRACT_BOUND_ACTION_API_ENV,
            "affects_cache_identity": True,
            "persisted_in_run_definition": False,
            "current": "set" if env_value is not None else "unset",
            "normalized_value": "enabled"
            if str(env_value or "").strip().lower() in {"1", "true", "yes", "on"}
            else "disabled",
        },
        "resume_command": (
            f"WARP_TASKGEN_STATE_DIR={shlex.quote(str(run_root))} uv run warp-taskgen resume"
        ),
        "resume_caveat": "Confirm the prior process or Remote Job is stopped before resuming.",
        "effects": {"writes": False, "model_calls": False, "network": False},
    }


def _load_manifest(run_root: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = _optional_path(state.get("manifest_path"))
    if manifest_path is None:
        manifest_path = run_root / "phase_0a" / "BENCHMARK_MANIFEST.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Phase 1 manifest must be an object")
    return payload


def _benchmark_root(state: Mapping[str, Any], manifest: Mapping[str, Any]) -> Path:
    raw_path = state.get("benchmark_path") or manifest.get("benchmark_codebase")
    path = _optional_path(raw_path)
    if path is None or not path.is_dir():
        raise FileNotFoundError("Phase 1 benchmark root is unavailable")
    return path


def _optional_path(value: Any) -> Path | None:
    return Path(value) if isinstance(value, str) and value else None


def _optional_text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _resolve_action_counts(state: Mapping[str, Any]) -> dict[str, int] | None:
    raw_counts = state.get("phase_1_action_counts")
    if raw_counts is None:
        raw_counts = state.get("action_counts")
    if not isinstance(raw_counts, Mapping):
        return _parse_phase_1_action_counts(raw_counts)

    counts: dict[str, int] = {}
    for raw_kind, raw_count in raw_counts.items():
        if not isinstance(raw_kind, str) or not raw_kind.strip():
            raise ValueError("persisted Phase 1 action kind must be a non-empty string")
        if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count < 0:
            raise ValueError("persisted Phase 1 action counts must be non-negative integers")
        counts[raw_kind] = raw_count
    if not counts or sum(counts.values()) <= 0:
        raise ValueError("persisted Phase 1 action counts must request at least one row")
    return counts


def _safe_reason(exc: BaseException) -> str:
    if isinstance(exc, FileNotFoundError):
        return "required_input_missing"
    if isinstance(exc, json.JSONDecodeError):
        return "required_json_invalid"
    if isinstance(exc, OSError):
        return "required_input_unreadable"
    return "required_input_invalid"
