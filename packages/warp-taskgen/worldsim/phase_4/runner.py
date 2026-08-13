"""Phase 4 runner behavior."""
# ruff: noqa: F821,E402

from __future__ import annotations

from worldsim.phase_4 import execution as _execution
from worldsim.phase_4._context import install_context
from worldsim.phase_4.admission import (
    _collect_agent_auth_runtime_errors,
    _load_admitted_phase_4_tasks,
    _load_site_profiles,
)
from worldsim.phase_4.postprocess_progress import (
    Phase4ProgressState,
    completed_task_ids_from_task_dir_root,
    record_postprocess_result,
    record_postprocess_start,
    record_variant_progress,
    write_phase_4_progress,
)
from worldsim.phase_4.preflight import (
    BaseStateProbeResult,
    _preflight_host_messages_api,
    _probe_seed_base_state_for_task_targets,
)
from worldsim.phase_4.results import _write_phase_4_results
from worldsim.run_control import pause_aware_map, pause_requested

install_context(globals())


async def run(args: argparse.Namespace) -> int:
    """Phase 4 entrypoint — adversarial evaluation with adaptive strategy variation."""
    state_dir = get_state_dir()
    resume = getattr(args, "resume", False)
    prior_state = None
    if resume:
        from worldsim.state import load_state

        prior_state = load_state()

    if prior_state and prior_state.get("step") == "phase_4" and prior_state.get("task_dir_root"):
        task_dir_root = Path(prior_state["task_dir_root"])
        logger.info("Resume: reusing task_dir_root %s", task_dir_root)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_dir_root = state_dir / "phase_4" / timestamp

    agent_model = getattr(args, "agent_model", None) or DEFAULT_MODEL
    agent_runner = getattr(args, "runner", RUNNER_BROWSER_USE) or RUNNER_BROWSER_USE
    sandbox_model = getattr(args, "sandbox_model", None) or "claude-sonnet-4-6"
    agent_provider = getattr(args, "agent_provider", None)
    agent_service_tier = getattr(args, "agent_service_tier", None)
    agent_llm_timeout = getattr(args, "agent_llm_timeout", None)
    agent_step_timeout = getattr(args, "agent_step_timeout", None)
    agent_task_timeout = getattr(args, "agent_task_timeout", None)

    benchmark_root = getattr(args, "benchmark", None)
    allow_unknown_auth = bool(getattr(args, "allow_unknown_auth", False))
    skip_host_bound_storage_state_auth = bool(
        getattr(args, "skip_host_bound_storage_state_auth", False)
    )
    max_tasks_per_site = getattr(args, "max_tasks_per_site", None)
    sites_filter_raw = getattr(args, "sites", None)
    adversarial_action_kind_filter_raw = getattr(args, "adversarial_action_kind", None)
    task_id_filter_raw = getattr(args, "phase_4_task_id", None)
    instances_path = getattr(args, "instances", None)
    phase_4_max_workers = getattr(args, "phase_4_max_workers", None)
    phase_4_variant_budget = getattr(args, "phase_4_variant_budget", None)
    phase_4_variant_system = _normalize_phase_4_variant_system(
        getattr(args, "phase_4_variant_system", None)
    )
    phase_4_eval_awareness_max_iterations = _normalize_eval_awareness_max_iterations(
        getattr(args, "phase_4_eval_awareness_max_iterations", None)
    )
    skip_intermediate_asr = bool(getattr(args, "skip_intermediate_asr", False))
    intermediate_asr_max_steps_per_task = getattr(args, "intermediate_asr_max_steps_per_task", None)

    _sweep_orphan_inflight_sentinels(task_dir_root)

    state_metadata = _phase_4_state_metadata(
        task_dir_root=task_dir_root,
        instances_path=instances_path or "",
        agent_model=agent_model,
        agent_runner=agent_runner,
        sandbox_model=sandbox_model,
        agent_provider=agent_provider,
        agent_service_tier=agent_service_tier,
        agent_llm_timeout=agent_llm_timeout,
        agent_step_timeout=agent_step_timeout,
        agent_task_timeout=agent_task_timeout,
        max_tasks_per_site=max_tasks_per_site,
        sites=sites_filter_raw,
        adversarial_action_kind=adversarial_action_kind_filter_raw,
        task_id_filter=task_id_filter_raw,
        benchmark_root=benchmark_root,
        allow_unknown_auth=allow_unknown_auth,
        skip_host_bound_storage_state_auth=skip_host_bound_storage_state_auth,
        phase_4_max_workers=phase_4_max_workers,
        phase_4_variant_budget=phase_4_variant_budget,
        phase_4_variant_system=phase_4_variant_system,
        phase_4_eval_awareness_max_iterations=phase_4_eval_awareness_max_iterations,
        skip_intermediate_asr=skip_intermediate_asr,
        intermediate_asr_max_steps_per_task=intermediate_asr_max_steps_per_task,
    )

    admission = _load_admitted_phase_4_tasks(
        state_dir=state_dir,
        sites_filter_raw=sites_filter_raw,
        adversarial_action_kind_filter_raw=adversarial_action_kind_filter_raw,
        max_tasks_per_site=max_tasks_per_site,
        state_metadata=state_metadata,
        task_id_filter_raw=task_id_filter_raw,
    )
    if admission["return_code"] is not None:
        return int(admission["return_code"])
    tasks = admission["tasks"]
    active_sites = admission["active_sites"]

    # Load benchmark config
    if not instances_path or not Path(instances_path).exists():
        logger.error("--instances JSON file required for Phase 4")
        return 1
    config = load_benchmark_config(instances_path)
    active_instances = [
        instance
        for instance in config.instances
        if not active_sites or normalize_site_name(instance.site_name) in active_sites
    ]
    if benchmark_root is None:
        benchmark_root = config.benchmark_codebase
    try:
        run_benchmark = infer_benchmark_name(
            [
                config.benchmark_name,
                *(task.get("benchmark") for task in tasks),
                *(task.get("benchmark_name") for task in tasks),
                *(task.get("benchmark_adapter") for task in tasks),
            ]
        )
    except ValueError as exc:
        logger.error("Phase 4 benchmark metadata gate failed: %s", exc)
        save_state(
            "phase_4",
            status="failed",
            reason="unsupported_benchmark",
            error=str(exc),
            **state_metadata,
        )
        return 1
    benchmark = run_benchmark or config.benchmark_name
    try:
        capabilities = get_benchmark_capabilities(benchmark).require("phase_4_execution")
    except ValueError:
        message = f"benchmark {benchmark!r} does not support WARP Taskgen Phase 4"
        logger.error("Phase 4 benchmark metadata gate failed: %s", message)
        save_state(
            "phase_4",
            status="failed",
            reason="unsupported_benchmark",
            error=message,
            **state_metadata,
        )
        return 1
    if agent_runner not in capabilities.supported_runners:
        message = (
            f"runner {agent_runner!r} is not supported for benchmark "
            f"{capabilities.canonical_name!r}; supported={capabilities.supported_runners}"
        )
        logger.error("Phase 4 benchmark runner gate failed: %s", message)
        save_state(
            "phase_4",
            status="failed",
            reason="unsupported_runner",
            error=message,
            **state_metadata,
        )
        return 1
    if agent_runner == RUNNER_AGENTLAB:
        errors = _agentlab_phase4_preflight_errors()
        if errors:
            message = "AgentLab Phase 4 runner preflight failed: " + "; ".join(errors)
            logger.error("Phase 4 runner gate failed: %s", message)
            save_state(
                "phase_4",
                status="failed",
                reason="runner_not_worldsim_v5_ready",
                error=message,
                **state_metadata,
            )
            return 1
    # page-surface-stable PVPO observes the normal runner-owned browser.
    # Historical configs may still contain pvpo_cdp_url values, but the
    # canonical backend no longer requires a dedicated PVPO browser endpoint.
    from worldsim.storage_state_preflight import ensure_storage_state

    healed_any = False
    storage_state_resolution_errors: list[StorageStatePreflightError] = []
    for instance in active_instances:
        auth = instance.agent_auth if isinstance(instance.agent_auth, dict) else None
        if not isinstance(auth, dict) or auth.get("type") != "storage_state":
            continue
        storage_state = auth.get("storage_state")
        declared_path = (
            str(storage_state.get("path") or "") if isinstance(storage_state, dict) else ""
        )
        try:
            healed_path = await ensure_storage_state(
                instance,
                benchmark_root=benchmark_root,
                benchmark_name=config.benchmark_name,
            )
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "auto-mint storage_state raised for %s: %s",
                instance.site_name,
                exc,
            )
            storage_state_resolution_errors.append(
                StorageStatePreflightError(
                    site_name=instance.site_name,
                    declared_path=declared_path,
                    message=str(exc),
                )
            )
            continue
        if healed_path is not None:
            storage_state = auth.get("storage_state")
            if isinstance(storage_state, dict):
                previous_path = storage_state.get("path")
                storage_state["path"] = str(healed_path)
                healed_any = healed_any or previous_path != str(healed_path)
            logger.info(
                "resolved storage_state for %s at %s",
                instance.site_name,
                healed_path,
            )

    preflight = inspect_storage_state_preflight(
        active_instances,
        benchmark_root=benchmark_root,
    )
    preflight_errors = [*storage_state_resolution_errors, *list(preflight.errors)]
    host_bound_mismatches = list(preflight.mismatches)
    # Auto-heal: if preflight discovered resolution/load errors after the
    # general freshness pass, retry errored sites once and re-run preflight.
    # WebArena Verified opts in by default (dummy creds in repo); other
    # benchmarks require WORLDSIM_AUTO_MINT_STORAGE_STATE=1.
    if preflight_errors:
        errored_sites = {error.site_name for error in preflight_errors}
        for instance in active_instances:
            if instance.site_name not in errored_sites:
                continue
            try:
                healed_path = await ensure_storage_state(
                    instance,
                    benchmark_root=benchmark_root,
                    benchmark_name=config.benchmark_name,
                )
            except Exception as exc:  # pragma: no cover — defensive
                logger.warning(
                    "auto-mint storage_state raised for %s: %s",
                    instance.site_name,
                    exc,
                )
                continue
            if healed_path is not None:
                auth = instance.agent_auth if isinstance(instance.agent_auth, dict) else None
                storage_state = auth.get("storage_state") if isinstance(auth, dict) else None
                if isinstance(storage_state, dict):
                    storage_state["path"] = str(healed_path)
                storage_state_resolution_errors = [
                    error
                    for error in storage_state_resolution_errors
                    if error.site_name != instance.site_name
                ]
                logger.info(
                    "auto-healed storage_state for %s at %s",
                    instance.site_name,
                    healed_path,
                )
                healed_any = True
    if healed_any:
        preflight = inspect_storage_state_preflight(
            active_instances,
            benchmark_root=benchmark_root,
        )
        preflight_errors = [*storage_state_resolution_errors, *list(preflight.errors)]
        host_bound_mismatches = list(preflight.mismatches)
    if preflight_errors:
        error_lines = [
            f"site {error.site_name!r}: {error.message} (declared path {error.declared_path!r})"
            for error in preflight_errors
        ]
        logger.error(
            "Phase 4 storage-state pre-flight failed:\n%s",
            "\n".join(f"  - {line}" for line in error_lines),
        )
        save_state(
            "phase_4",
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
                "Phase 4 found host-bound storage_state artifacts and will skip agent auth for "
                "those sites because --skip-host-bound-storage-state-auth was set:\n%s",
                "\n".join(f"  - {line}" for line in mismatch_lines),
            )
            config = apply_skip_auth_for_host_bound_storage_states(config, host_bound_mismatches)
            active_instances = [
                instance
                for instance in config.instances
                if not active_sites or normalize_site_name(instance.site_name) in active_sites
            ]
        else:
            logger.error(
                "Phase 4 storage-state pre-flight failed:\n%s\nRe-run Phase 0d against the "
                "current instances host, or pass --skip-host-bound-storage-state-auth to "
                "proceed without browser auth for those sites.",
                "\n".join(f"  - {line}" for line in mismatch_lines),
            )
            save_state(
                "phase_4",
                status="failed",
                reason="host_bound_storage_state",
                host_bound_storage_state_errors=mismatch_lines,
                **state_metadata,
            )
            return 1
    # Magento base_url + pending-review probes were removed 2026-04-21 with
    # the WASP-aligned scoping decision (see
    # docs/handoffs/wasp-aligned-scoping-decision.md). The pipeline no longer
    # targets Magento; both probes are dead infrastructure.
    # Acquire fresh bearer tokens for instances that use runtime generation.
    token_errors = acquire_tokens_for_instances(active_instances)
    if token_errors:
        logger.error(
            "Phase 4 token acquisition failed:\n%s",
            "\n".join(f"  - {error}" for error in token_errors),
        )
        save_state(
            "phase_4",
            status="failed",
            reason="token_acquisition_failed",
            token_errors=token_errors,
            **state_metadata,
        )
        return 1
    seed_runtime_errors = collect_seed_runtime_errors(
        tasks,
        active_instances,
        seed_field="adversarial_data_seed",
    )
    if seed_runtime_errors:
        logger.error(
            "Phase 4 seed pre-flight failed:\n%s",
            "\n".join(f"  - {error}" for error in seed_runtime_errors),
        )
        save_state(
            "phase_4",
            status="failed",
            reason="seed_runtime_config_error",
            seed_runtime_errors=seed_runtime_errors,
            **state_metadata,
        )
        return 1
    # Fail fast if Claude Code auth is missing — judge/variant sandboxes need it.
    try:
        preflight_auth_check()
    except RuntimeError as exc:
        logger.error("Phase 4 auth pre-flight failed:\n%s", exc)
        save_state("phase_4", status="failed", reason="auth_preflight_failed", **state_metadata)
        return 1

    profiles_dir = state_dir / "phase_0c"
    site_profiles = _load_site_profiles(tasks, profiles_dir)
    seed_probe_cache: dict[tuple[str, str, str, str], BaseStateProbeResult] = {}
    agent_auth_errors = _collect_agent_auth_runtime_errors(active_instances, site_profiles)
    if agent_auth_errors:
        logger.error(
            "Phase 4 agent-auth pre-flight failed:\n%s",
            "\n".join(f"  - {error}" for error in agent_auth_errors),
        )
        save_state(
            "phase_4",
            status="failed",
            reason="agent_runtime_config_error",
            agent_runtime_errors=agent_auth_errors,
            **state_metadata,
        )
        return 1

    preflight_ok, preflight_err = await _preflight_host_messages_api(sandbox_model=sandbox_model)
    if not preflight_ok:
        logger.error("Phase 4 preflight against Anthropic Messages API failed: %s", preflight_err)
        save_state(
            "phase_4",
            status="failed",
            reason="host_api_preflight_failed",
            host_api_preflight_error=preflight_err,
            **state_metadata,
        )
        return 1

    logger.info(
        "Phase 4: evaluating %d adversarial tasks across %d instances",
        len(tasks),
        len(active_instances),
    )
    infrastructure_errors = _probe_seed_base_state_for_task_targets(
        tasks,
        active_instances,
        cache=seed_probe_cache,
    )
    if infrastructure_errors:
        logger.error(
            "Phase 4 seed base-state probe failed:\n%s",
            "\n".join(f"  - {error}" for error in infrastructure_errors),
        )
        save_state(
            "phase_4",
            status="failed",
            reason="infrastructure_failed",
            infrastructure_errors=infrastructure_errors,
            **state_metadata,
        )
        return 1
    agent_factory = make_agent_factory(
        model=agent_model,
        provider=agent_provider,
        service_tier=agent_service_tier,
        llm_timeout=agent_llm_timeout,
        step_timeout=agent_step_timeout,
        task_timeout=agent_task_timeout,
        runner=agent_runner,
    )
    if phase_4_max_workers is not None:
        logger.info("Phase 4 browser-agent worker concurrency cap: %d", phase_4_max_workers)
    browser_worker_semaphore = (
        asyncio.Semaphore(phase_4_max_workers) if phase_4_max_workers is not None else None
    )
    reset_cache = TaskResetCache()
    save_state(
        "phase_4",
        status="running",
        pause_stage="initial_evaluation",
        **state_metadata,
    )
    completed_initial_task_ids = completed_task_ids_from_task_dir_root(task_dir_root)

    def _write_progress_safely(stage: str, *, status: str = "running", **kwargs: Any) -> None:
        try:
            write_phase_4_progress(
                state_dir,
                status=status,
                stage=stage,
                task_dir_root=task_dir_root,
                total_tasks=len(tasks),
                phase_4_max_workers=phase_4_max_workers,
                **kwargs,
            )
        except Exception as exc:
            logger.warning("Could not write Phase 4 progress heartbeat: %s", exc)

    progress_lock = asyncio.Lock()
    started_initial_task_ids: set[str] = set()
    active_initial_task_ids: set[str] = set()
    failed_initial_task_ids: set[str] = set()

    def _initial_progress_extra() -> dict[str, Any]:
        active_ids = sorted(active_initial_task_ids)
        return {
            "initial_started_tasks": len(started_initial_task_ids),
            "running_initial_browser_tasks": len(active_ids),
            "running_initial_browser_task_ids": active_ids[:12],
            "active_initial_tasks": len(active_ids),
            "active_initial_task_ids": active_ids[:12],
            "failed_initial_tasks": len(failed_initial_task_ids),
        }

    _write_progress_safely(
        "initial_evaluation",
        completed_initial_tasks=len(completed_initial_task_ids),
        extra=_initial_progress_extra(),
    )

    async def _record_initial_result(result: dict[str, Any]) -> None:
        task_id = result.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            return
        async with progress_lock:
            completed_initial_task_ids.add(task_id.strip())
            active_initial_task_ids.discard(task_id.strip())
            _write_progress_safely(
                "initial_evaluation",
                completed_initial_tasks=len(completed_initial_task_ids),
                extra=_initial_progress_extra(),
            )

    # Thread the benchmark codebase root through so BrowserUseAgent can validate
    # absolute auth_mechanism.storage_state.path values for containment. Relative
    # paths anchor to the WorldSim state dir (where Phase 0d writes), not to
    # benchmark_root.

    async def _bound_run_adversarial_task(task, agent, instance, task_dir):
        task_id = str(task.get("id", "unknown")).strip() or "unknown"
        async with progress_lock:
            started_initial_task_ids.add(task_id)
            active_initial_task_ids.add(task_id)
            _write_progress_safely(
                "initial_evaluation",
                completed_initial_tasks=len(completed_initial_task_ids),
                extra=_initial_progress_extra(),
            )
        run_kwargs: dict[str, Any] = {
            "benchmark_root": benchmark_root,
            "sandbox_model": sandbox_model,
            "all_instances": config.instances,
            "site_profile": site_profiles.get(str(task.get("site", ""))),
            "resume_fingerprint": _phase_4_result_fingerprint(
                task,
                eval_context=_phase_4_eval_context_for_task(
                    task,
                    instances=config.instances,
                    config_url_placeholders=config.url_placeholders,
                    agent_model=agent_model,
                    agent_runner=agent_runner,
                    agent_provider=agent_provider,
                    agent_llm_timeout=agent_llm_timeout,
                    agent_step_timeout=agent_step_timeout,
                    agent_task_timeout=agent_task_timeout,
                    sandbox_model=sandbox_model,
                    benchmark_root=benchmark_root,
                ),
                site_profile=site_profiles.get(str(task.get("site", ""))),
            ),
        }
        if callable_accepts_keyword(_execution.run_adversarial_task, "reset_cache"):
            run_kwargs["reset_cache"] = reset_cache
        if callable_accepts_keyword(_execution.run_adversarial_task, "seed_probe_cache"):
            run_kwargs["seed_probe_cache"] = seed_probe_cache
        try:
            return await _execution.run_adversarial_task(
                task,
                agent,
                instance,
                task_dir,
                **run_kwargs,
            )
        except Exception:
            async with progress_lock:
                active_initial_task_ids.discard(task_id)
                failed_initial_task_ids.add(task_id)
                _write_progress_safely(
                    "initial_evaluation",
                    completed_initial_tasks=len(completed_initial_task_ids),
                    extra=_initial_progress_extra(),
                )
            raise

    # Initial adversarial run — run_tasks_by_site calls
    # prepare_tasks_for_execution internally, so no need to call it here.
    results = await run_tasks_by_site(
        tasks=tasks,
        instances=config.instances,
        agent_factory=agent_factory,
        task_runner=_bound_run_adversarial_task,
        task_dir_root=task_dir_root,
        config_url_placeholders=config.url_placeholders,
        resume=resume,
        max_workers=phase_4_max_workers,
        result_callback=_record_initial_result,
        resume_fingerprint_builder=lambda task: _phase_4_result_fingerprint(
            task,
            eval_context=_phase_4_eval_context_for_task(
                task,
                instances=config.instances,
                config_url_placeholders=config.url_placeholders,
                agent_model=agent_model,
                agent_runner=agent_runner,
                agent_provider=agent_provider,
                agent_llm_timeout=agent_llm_timeout,
                agent_step_timeout=agent_step_timeout,
                agent_task_timeout=agent_task_timeout,
                sandbox_model=sandbox_model,
                benchmark_root=benchmark_root,
            ),
            site_profile=site_profiles.get(str(task.get("site", ""))),
        ),
        pause_check=lambda: pause_requested(state_dir),
    )

    task_by_id = {str(task.get("id", "unknown")): task for task in tasks}

    save_state(
        "phase_4",
        status="running",
        pause_stage="postprocessing",
        **state_metadata,
    )
    progress_state = Phase4ProgressState(
        state_dir=state_dir,
        task_dir_root=task_dir_root,
        total_tasks=len(tasks),
        completed_initial_tasks=len(results),
        phase_4_max_workers=phase_4_max_workers,
        phase_4_variant_budget=phase_4_variant_budget,
        phase_4_variant_system=phase_4_variant_system,
        phase_4_eval_awareness_max_iterations=phase_4_eval_awareness_max_iterations,
    )
    _write_progress_safely(
        "postprocessing",
        completed_initial_tasks=len(results),
        extra=_initial_progress_extra(),
    )
    agent_execution_fingerprint = {
        "agent_model": agent_model,
        "agent_runner": agent_runner,
        "agent_provider": agent_provider,
        "agent_service_tier": agent_service_tier,
        "agent_llm_timeout": agent_llm_timeout,
        "agent_step_timeout": agent_step_timeout,
        "agent_task_timeout": agent_task_timeout,
    }

    async def _postprocess_one_task_with_progress(result: dict[str, Any]) -> dict[str, Any]:
        task_id = str(result.get("task_id", "unknown"))
        try:
            await record_postprocess_start(progress_state, task_id)
        except Exception as exc:
            logger.warning("Could not write Phase 4 postprocess-start heartbeat: %s", exc)

        async def _record_progress(event: str, data: Mapping[str, Any]) -> None:
            try:
                await record_variant_progress(progress_state, task_id, event, data)
            except Exception as exc:
                logger.warning("Could not write Phase 4 variant heartbeat: %s", exc)

        try:
            processed = await _postprocess_one_task(
                result=result,
                task_by_id=task_by_id,
                config=config,
                profiles_dir=profiles_dir,
                agent_factory=agent_factory,
                task_dir_root=task_dir_root,
                resume=resume,
                benchmark_root=benchmark_root,
                sandbox_model=sandbox_model,
                site_profile=site_profiles.get(
                    str(task_by_id.get(str(result.get("task_id", "")), {}).get("site", ""))
                ),
                variant_budget_preset=phase_4_variant_budget,
                variant_system=phase_4_variant_system,
                eval_awareness_max_iterations=phase_4_eval_awareness_max_iterations,
                agent_execution=agent_execution_fingerprint,
                progress_callback=_record_progress,
                browser_worker_semaphore=browser_worker_semaphore,
            )
        except Exception:
            try:
                await record_postprocess_result(progress_state, task_id, failed=True)
            except Exception as exc:
                logger.warning("Could not write Phase 4 postprocess-failure heartbeat: %s", exc)
            raise
        try:
            await record_postprocess_result(progress_state, task_id)
        except Exception as exc:
            logger.warning("Could not write Phase 4 postprocess-complete heartbeat: %s", exc)
        return processed

    raw_postprocessed = await pause_aware_map(
        results,
        _postprocess_one_task_with_progress,
        concurrency=phase_4_max_workers or max(1, len(results)),
        state_dir=state_dir,
    )

    final_results: list[dict] = []
    postprocess_failures: list[tuple[str, BaseException]] = []
    for i, processed in enumerate(raw_postprocessed):
        if isinstance(processed, BaseException):
            task_id = str(results[i].get("task_id", "unknown"))
            logger.error("Post-processing failed for task %s: %s", task_id, processed)
            postprocess_failures.append((task_id, processed))
            continue
        final_results.append(processed)

    if postprocess_failures:
        _write_progress_safely(
            "postprocess_exception",
            status="failed",
            completed_initial_tasks=len(results),
            postprocessed_tasks=len(final_results),
            postprocess_attempted_tasks=len(final_results) + len(postprocess_failures),
            postprocess_failed_tasks=len(postprocess_failures),
            extra=_initial_progress_extra(),
        )
        save_state(
            "phase_4",
            status="failed",
            reason="postprocess_exception",
            failed_tasks=[task_id for task_id, _ in postprocess_failures],
            **state_metadata,
        )
        return 1

    save_state(
        "phase_4",
        status="running",
        pause_stage="finalizing",
        **state_metadata,
    )
    intermediate_asr_summary: dict[str, Any] | None = None
    try:
        from worldsim.phase_4.intermediate_asr import (
            evaluate_intermediate_asr,
            task_lookup_from_tasks,
        )

        intermediate_asr_summary = await evaluate_intermediate_asr(
            phase4_dir=state_dir / "phase_4",
            results=final_results,
            task_lookup=task_lookup_from_tasks(tasks),
            sandbox_model=sandbox_model,
            max_steps_per_task=intermediate_asr_max_steps_per_task,
            enabled=not skip_intermediate_asr,
        )
    except Exception as exc:
        logger.warning("Phase 4 intermediate ASR post-hoc evaluation failed: %s", exc)
        intermediate_asr_summary = {
            "status": "failed",
            "failure_class": "intermediate_asr_exception",
            "diagnosis": str(exc),
        }

    return _write_phase_4_results(
        state_dir=state_dir,
        state_metadata=state_metadata,
        final_results=final_results,
        tasks=tasks,
        intermediate_asr_summary=intermediate_asr_summary,
    )


def _agentlab_phase4_preflight_errors() -> list[str]:
    repo_root = Path(__file__).resolve().parents[2]
    errors: list[str] = []
    sidecar = repo_root / "packages" / "worldsim-agentlab-runner"
    if not sidecar.is_dir():
        errors.append(f"missing AgentLab sidecar package at {sidecar}")
    lockfile = sidecar / "uv.lock"
    if not lockfile.is_file():
        errors.append(f"missing AgentLab sidecar lockfile at {lockfile}")
    return errors


import sys as _sys

from worldsim.phase_4 import eval_awareness_iterator as _eval_awareness_iterator
from worldsim.phase_4 import placement_loop as _placement_loop
from worldsim.phase_4 import postprocess as _postprocess
from worldsim.phase_4 import resume as _resume
from worldsim.phase_4 import strategy_variation as _strategy_variation
from worldsim.phase_4 import variant_eval as _variant_eval
from worldsim.phase_4._context import link_modules as _link_modules

_link_modules(
    [
        _sys.modules[__name__],
        # Execution is imported explicitly above; it is intentionally not
        # linked into the runner's compatibility namespace.
        _eval_awareness_iterator,
        _placement_loop,
        _postprocess,
        _resume,
        _strategy_variation,
        _variant_eval,
    ]
)

__all__ = [name for name in globals() if not name.startswith("__")]
