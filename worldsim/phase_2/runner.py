"""Phase 2 runner behavior."""
# ruff: noqa: F821,E402

from __future__ import annotations

from worldsim.phase_2._context import install_context

install_context(globals())


@dataclass
class SiteInjectionResult:
    site_name: str
    adversarial_tasks: list[dict]
    errors: list[str]


async def run(args: argparse.Namespace) -> int:
    """Phase 2 entrypoint — generate adversarial injections for each site."""
    state_dir = get_state_dir()
    output_dir = state_dir / "phase_2"
    sandbox_model = getattr(args, "sandbox_model", None) or "claude-sonnet-4-6"
    text_fill_model = getattr(args, "phase_2_text_model", None) or DEFAULT_TEXT_FILL_MODEL
    texts_per_plan = getattr(args, "phase_2b_texts_per_plan", None) or DEFAULT_TEXTS_PER_PLAN
    text_fill_concurrency = (
        getattr(args, "phase_2_text_fill_concurrency", None) or DEFAULT_TEXT_FILL_CONCURRENCY
    )
    phase_2a_action_policy = getattr(args, "phase_2a_action_policy", None)
    max_tasks_per_site = getattr(args, "max_tasks_per_site", None)
    sites_filter_raw = getattr(args, "sites", None)
    state_metadata: dict[str, Any] = {
        "sandbox_model": sandbox_model,
        "max_tasks_per_site": max_tasks_per_site,
        "sites": sites_filter_raw,
        "phase_2b_texts_per_plan": texts_per_plan,
        "phase_2_text_fill_concurrency": text_fill_concurrency,
        "phase_2_text_model": text_fill_model,
        "phase_2a_resolution_signature": _phase_2a_resolution_signature(args),
        "phase_2a_action_policy": phase_2a_action_policy,
        "exposure_contract_signature": exposure_contract_signature(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    plans_path = output_dir / "adversarial_plans.json"
    diagnostics_path = output_dir / "text_fill_diagnostics.json"
    output_path = output_dir / "adversarial_tasks.json"

    # Phase 2c-only short-circuit: re-verify an existing adversarial dataset
    # against a live dev instance without re-running 2a planning or 2b text
    # fill. This is the `phase 2c` CLI alias (and `phase 2 --feasibility-only`).
    if getattr(args, "feasibility_only", False):
        if not output_path.exists():
            logger.error(
                "Phase 2c --feasibility-only requires an existing %s; run phase 2 first",
                output_path,
            )
            return 1
        prior_state = load_state() or {}
        return await _run_feasibility_stage(
            args=args,
            output_path=output_path,
            output_dir=output_dir,
            state_metadata={
                **state_metadata,
                "feasibility_only": True,
            },
            prior_phase_2_status=prior_state.get("status"),
        )

    # Load benign tasks from Phase 1
    tasks_path = state_dir / "phase_1" / "benign_tasks.json"
    if not tasks_path.exists():
        logger.error("Benign tasks not found at %s — run phase 1 first", tasks_path)
        return 1
    benign_tasks = json.loads(tasks_path.read_text())
    try:
        benchmark_name = _infer_task_records_benchmark(
            benign_tasks,
            label="Phase 1 benign tasks",
        )
        capabilities = get_benchmark_capabilities(benchmark_name)
        if not capabilities.phase_2_supported:
            raise ValueError(f"benchmark {benchmark_name!r} does not support WorldSim v5 Phase 2")
    except ValueError as exc:
        logger.error("Phase 2 benchmark gate failed: %s", exc)
        save_state(
            "phase_2",
            status="failed",
            reason="unsupported_benchmark",
            benchmark_error=str(exc),
            **state_metadata,
        )
        return 1
    state_metadata["benchmark_name"] = benchmark_name

    # Load profiles from Phase 0c
    profiles_dir = state_dir / "phase_0c"
    if not profiles_dir.exists():
        logger.error("Profiles directory not found at %s — run phase 0c first", profiles_dir)
        return 1

    # Optional per-site cap (same deterministic seeded sampler Phase 3/4 use,
    # so the same N tasks pair across phases).
    if max_tasks_per_site is not None:
        from worldsim.agent_config import cap_tasks_per_site

        before = len(benign_tasks)
        benign_tasks = cap_tasks_per_site(benign_tasks, max_tasks_per_site)
        logger.info(
            "Phase 2: capped at %d tasks/site via seeded sampler (%d -> %d tasks)",
            max_tasks_per_site,
            before,
            len(benign_tasks),
        )

    # Group tasks by primary site
    tasks_by_site: dict[str, list[dict]] = {}
    for task in benign_tasks:
        site = task["site"]
        tasks_by_site.setdefault(site, []).append(task)

    # Optional per-site filter. When set, only the listed sites run; other
    # sites' entries in adversarial_tasks.json are preserved via merge below.
    sites_filter: set[str] | None = None
    if sites_filter_raw:
        sites_filter = {s.strip() for s in sites_filter_raw.split(",") if s.strip()}
        unknown = sites_filter - set(tasks_by_site.keys())
        if unknown:
            logger.error(
                "Phase 2: --sites includes unknown site(s): %s. Known sites: %s",
                sorted(unknown),
                sorted(tasks_by_site.keys()),
            )
            return 1
        tasks_by_site = {s: ts for s, ts in tasks_by_site.items() if s in sites_filter}
        logger.info("Phase 2: --sites filter active, running only %s", sorted(tasks_by_site.keys()))

    logger.info(
        "Phase 2: generating injections for %d sites (%d total tasks, phase_2a_runtime=api)",
        len(tasks_by_site),
        sum(len(ts) for ts in tasks_by_site.values()),
    )

    site_profiles, profile_errors = _collect_site_profiles(tasks_by_site, profiles_dir)
    if profile_errors:
        logger.error(
            "Phase 2 cannot proceed because required site profiles are invalid:\n%s",
            "\n".join(f"  - {item}" for item in profile_errors),
        )
        save_state(
            "phase_2",
            status="failed",
            reason="invalid_site_profiles",
            **state_metadata,
        )
        return 1
    site_profile_payloads = {
        site: json.loads(path.read_text()) for site, path in site_profiles.items()
    }
    benign_by_id = {str(task.get("id", "")): task for task in benign_tasks}
    expected_benign_task_ids = {
        str(task.get("id", "")) for tasks in tasks_by_site.values() for task in tasks
    }

    prior_state = load_state() or {}
    site_failures = list(prior_state.get("generation_failures") or [])
    reusable_plans = _load_reusable_phase_2_plans(
        prior_state=prior_state,
        plans_path=plans_path,
        sites_filter=sites_filter,
        expected_benign_task_ids=expected_benign_task_ids,
        benign_by_id=benign_by_id,
        site_profiles=site_profile_payloads,
        current_sandbox_model=sandbox_model,
        current_phase_2a_resolution_signature=state_metadata["phase_2a_resolution_signature"],
    )
    reusable_final_tasks = None
    if reusable_plans is None and sites_filter is None:
        reusable_final_tasks = _load_reusable_phase_2_tasks(
            prior_state=prior_state,
            output_path=output_path,
            sites_filter=sites_filter,
            expected_task_ids=None,
            expected_benign_task_ids=expected_benign_task_ids,
            texts_per_plan=texts_per_plan,
            benign_by_id=benign_by_id,
            site_profiles=site_profile_payloads,
            current_sandbox_model=sandbox_model,
            current_text_model=text_fill_model,
            current_phase_2a_resolution_signature=state_metadata["phase_2a_resolution_signature"],
        )
    if reusable_plans is None and reusable_final_tasks is None:
        save_state("phase_2", status="running", phase_2_stage="planning", **state_metadata)

        # Resolve the per-site live-instance map once before the shard
        # loop so every shard of a given site sees the same instance
        # descriptor. None means the legacy L1/L2-only path (either
        # --no-l3-l4 was set, --feasibility-instances is absent, or the
        # wrapper file had no instances). See `_load_phase_2a_instance_by_site`.
        instance_by_site = _load_phase_2a_instance_by_site(args)
        _warm_phase_2a_instance_tokens(instance_by_site)

        # Shard each site's tasks into chunks of TASKS_PER_SHARD and launch
        # bounded host-side API calls. Shopping (192 tasks) becomes ~8 shorter
        # strategy calls instead of one huge request.
        shard_coros = []
        shard_limiter = asyncio.Semaphore(DEFAULT_PHASE_2A_SHARD_CONCURRENCY)
        for site, tasks in tasks_by_site.items():
            shards = _shard_tasks(tasks, TASKS_PER_SHARD)
            per_site_instance = instance_by_site.get(site) if instance_by_site is not None else None
            for shard_idx, shard in enumerate(shards):
                label = f"{site}-shard-{shard_idx}" if len(shards) > 1 else site
                shard_coros.append(
                    _run_shard_with_limit(
                        shard_limiter,
                        site_name=site,
                        site_tasks=shard,
                        all_site_tasks=tasks,
                        profile_path=site_profiles[site],
                        label=label,
                        sandbox_model=sandbox_model,
                        instance=per_site_instance,
                        benchmark=benchmark_name,
                        action_policy=phase_2a_action_policy,
                    )
                )
        shard_results = await asyncio.gather(*shard_coros, return_exceptions=True)

        # Merge per-shard results back into per-site results.
        results = _merge_shard_results(shard_results, tasks_by_site)

        all_plans: list[dict] = []
        for result in results:
            if isinstance(result, BaseException):
                logger.error("Phase 2: sandbox failed with exception: %s", result)
                site_failures.append(str(result))
                continue
            if result.errors:
                site_failures.extend(f"{result.site_name}: {error}" for error in result.errors)
            # Fail-open: include whatever valid tasks succeeded even if sibling shards failed.
            if result.adversarial_tasks:
                all_plans.extend(result.adversarial_tasks)
                logger.info(
                    "Phase 2: site %r produced %d validated plans (%d shard error(s))",
                    result.site_name,
                    len(result.adversarial_tasks),
                    len(result.errors),
                )

        succeeded = sum(1 for r in results if not isinstance(r, BaseException) and not r.errors)
        logger.info(
            "Phase 2: planning sandboxes done — %d/%d sites succeeded, %d total plans",
            succeeded,
            len(results),
            len(all_plans),
        )
        if site_failures:
            logger.warning(
                "Phase 2: %d planning shard(s) failed — continuing with partial plans:\n%s",
                len(site_failures),
                "\n".join(f"  - {failure}" for failure in site_failures),
            )

        if not all_plans:
            logger.error("Phase 2 planning produced zero adversarial plans across all sites")
            save_state(
                "phase_2",
                status="failed",
                reason="no_adversarial_plans",
                generation_failures=site_failures,
                phase_2_stage="planning",
                **state_metadata,
            )
            return 1

        # Fold in any validated shards persisted to disk that the current
        # in-memory aggregation missed — e.g. when one shard re-ran in
        # isolation after a prior run, prior sidecars would otherwise be
        # silently dropped. Scope to the sites actually in this run's
        # input (tasks_by_site keys) so we don't resurrect quarantined
        # out-of-scope sites.
        active_sites = set(tasks_by_site.keys())
        if sites_filter is not None:
            active_sites &= sites_filter
        all_plans, recovered_ids = _recover_orphaned_shards(
            output_dir / "shards",
            all_plans,
            allowed_sites=active_sites,
            benign_by_id=benign_by_id,
            site_profiles=site_profile_payloads,
        )
        if recovered_ids:
            logger.warning(
                "Phase 2 aggregation: recovered %d orphan shard task(s) from disk: %s",
                len(recovered_ids),
                ", ".join(recovered_ids[:10]) + (" …" if len(recovered_ids) > 10 else ""),
            )

        merged_plans = _merge_preserving_unfiltered_sites(
            plans_path,
            all_plans,
            sites_filter=sites_filter,
        )
        write_json_atomic(
            plans_path,
            merged_plans,
            failpoint_base="phase_2.output.adversarial_plans",
        )
        reusable_plans = merged_plans
    else:
        if reusable_final_tasks is not None:
            logger.info(
                "Phase 2: reusing %d saved adversarial task(s) from %s",
                len(reusable_final_tasks),
                output_path,
            )
        else:
            logger.info(
                "Phase 2: reusing %d saved adversarial plan(s) from %s",
                len(reusable_plans),
                plans_path,
            )

    text_fill_diagnostics = _load_text_fill_diagnostics(diagnostics_path)
    if reusable_final_tasks is None:
        candidate_plans = [
            plan
            for plan in reusable_plans
            if sites_filter is None or str(plan.get("site", "")) in sites_filter
        ]
        reusable_final_tasks = _load_reusable_phase_2_tasks(
            prior_state=prior_state,
            output_path=output_path,
            sites_filter=sites_filter,
            expected_task_ids={str(plan.get("id", "")) for plan in candidate_plans},
            expected_benign_task_ids={
                str(plan.get("benign_task_id", "")) for plan in candidate_plans
            },
            texts_per_plan=texts_per_plan,
            benign_by_id=benign_by_id,
            site_profiles=site_profile_payloads,
            current_sandbox_model=sandbox_model,
            current_text_model=text_fill_model,
            current_phase_2a_resolution_signature=state_metadata["phase_2a_resolution_signature"],
        )
        if reusable_final_tasks is None:
            save_state(
                "phase_2",
                status="running",
                phase_2_stage="text_fill",
                generation_failures=site_failures,
                **state_metadata,
            )

            prefilled_tasks = [task for task in candidate_plans if "seed_template" not in task]
            plans_to_fill = [task for task in candidate_plans if "seed_template" in task]
            if plans_to_fill:
                filled_tasks, text_fill_diagnostics = await fill_texts_for_tasks(
                    plans_to_fill,
                    texts_per_plan=texts_per_plan,
                    concurrency=text_fill_concurrency,
                    model=text_fill_model,
                )
            else:
                filled_tasks, text_fill_diagnostics = ([], [])
            filled_tasks = prefilled_tasks + filled_tasks
            write_json_atomic(diagnostics_path, text_fill_diagnostics)
        else:
            logger.info(
                "Phase 2: reusing %d saved adversarial task(s) from %s",
                len(reusable_final_tasks),
                output_path,
            )
            filled_tasks = reusable_final_tasks
    else:
        filled_tasks = reusable_final_tasks

    if not filled_tasks:
        logger.error("Phase 2 text fill produced zero adversarial tasks")
        save_state(
            "phase_2",
            status="failed",
            reason="no_text_filled_tasks",
            generation_failures=site_failures,
            text_fill_failures=text_fill_diagnostics,
            phase_2_stage="text_fill",
            **state_metadata,
        )
        return 1

    merged_output = _merge_preserving_unfiltered_sites(
        output_path,
        filled_tasks,
        sites_filter=sites_filter,
    )
    if reusable_final_tasks is None or output_path.read_text() != json.dumps(
        merged_output, indent=2
    ):
        write_json_atomic(
            output_path,
            merged_output,
            failpoint_base="phase_2.output.adversarial_tasks",
        )

    text_fill_failures = [
        diag
        for diag in text_fill_diagnostics
        if diag.get("status") not in {"ok", "reused_existing"}
    ]
    status = "partial_complete" if site_failures or text_fill_failures else "complete"
    save_state(
        "phase_2",
        status="running",
        phase_2_stage="feasibility",
        adversarial_tasks_path=str(output_path),
        task_count=len(merged_output),
        generation_failures=site_failures,
        text_fill_failures=text_fill_failures,
        partial=bool(site_failures or text_fill_failures),
        **state_metadata,
    )

    feasibility_rc = await _run_feasibility_stage(
        args=args,
        output_path=output_path,
        output_dir=output_dir,
        state_metadata=state_metadata,
        prior_phase_2_status=status,
    )
    if feasibility_rc != 0:
        return feasibility_rc

    # Final "complete" marker: every sub-stage (2a planning, 2b text fill,
    # 2c feasibility) has succeeded. `phase_2_stage="complete"` is what
    # downstream tooling looks at to know Phase 2 is done.
    save_state(
        "phase_2",
        status=status,
        phase_2_stage="complete",
        adversarial_tasks_path=str(output_path),
        task_count=len(merged_output),
        generation_failures=site_failures,
        text_fill_failures=text_fill_failures,
        partial=bool(site_failures or text_fill_failures),
        **state_metadata,
    )

    cost_tracker.log_phase_summary("phase_2")
    cost_tracker.save(state_dir / "cost_report.json")
    logger.info(
        "Phase 2 %s — %d adversarial tasks written to %s",
        status,
        len(merged_output),
        output_path,
    )
    return 0


# Import sibling domains after defining run(), then link module globals so the
# mechanically split functions preserve the old single-module lookup semantics.
import sys as _sys

from worldsim.phase_2 import eligibility as _eligibility
from worldsim.phase_2 import generation as _generation
from worldsim.phase_2 import option_a as _option_a
from worldsim.phase_2 import plan_validation as _plan_validation
from worldsim.phase_2 import reuse as _reuse
from worldsim.phase_2 import shards as _shards
from worldsim.phase_2 import target_inputs as _target_inputs
from worldsim.phase_2 import target_stage as _target_stage
from worldsim.phase_2._context import link_modules as _link_modules
from worldsim.phase_2.phase_2c import stage as _phase_2c_stage

_link_modules(
    [
        _sys.modules[__name__],
        _eligibility,
        _generation,
        _option_a,
        _plan_validation,
        _reuse,
        _shards,
        _target_inputs,
        _target_stage,
        _phase_2c_stage,
    ]
)

__all__ = [name for name in globals() if not name.startswith("__")]
