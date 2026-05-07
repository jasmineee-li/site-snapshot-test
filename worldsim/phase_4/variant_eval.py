"""Phase 4 variant evaluation helpers."""
# ruff: noqa: F821

from __future__ import annotations

from worldsim.phase_4._context import install_context

install_context(globals())

def _merge_variant_task(
    original_task: dict[str, Any],
    candidate: Any,
) -> dict[str, Any]:
    """Preserve immutable benign fields while accepting seed-only variant diffs."""
    if not isinstance(candidate, dict):
        logger.warning("Variant payload was not an object; keeping original task")
        return original_task

    merged = json.loads(json.dumps(original_task))
    candidate_seed = candidate.get("adversarial_data_seed")
    if not isinstance(candidate_seed, dict):
        logger.warning("Variant payload omitted adversarial_data_seed; keeping original task")
        return merged

    try:
        validate_data_seed(candidate_seed, allow_none=False)
    except ValueError as exc:
        logger.warning("Variant produced invalid adversarial_data_seed: %s", exc)
        return merged
    seed_error = self_contained_adversarial_seed_error(merged.get("data_seed"), candidate_seed)
    if seed_error is not None:
        logger.warning("Variant produced invalid adversarial_data_seed: %s", seed_error)
        return merged
    contract_error = _seed_preserves_exposure_contract_error(merged, candidate_seed)
    if contract_error is not None:
        logger.warning("Variant produced invalid adversarial_data_seed: %s", contract_error)
        return merged

    immutable_fields = (
        "id",
        "benign_task_id",
        "site",
        "sites",
        "instruction",
        "start_urls",
        "data_seed",
        "agent_context",
        "reward_function",
        "intent_template_id",
        "revision",
    )
    for field in immutable_fields:
        if field in candidate and candidate[field] != original_task.get(field):
            logger.warning(
                "Variant attempted to mutate immutable field %r; keeping original value", field
            )

    merged["adversarial_data_seed"] = candidate_seed
    candidate_payload_text = candidate.get("payload_text")
    if not _synchronize_variant_payload_texts(
        original_task,
        merged,
        candidate_seed,
        candidate_payload_text if isinstance(candidate_payload_text, dict) else None,
    ):
        logger.warning(
            "Variant produced seed/payload_text mismatch for task %s; keeping original task",
            original_task.get("id", "unknown"),
        )
        return json.loads(json.dumps(original_task))
    for field in ("applied_strategy", "placement_fix", "ecological_validity_fix"):
        if field in candidate:
            merged[field] = candidate[field]
    return merged

def _rebase_adversarial_task(
    adversarial_task: dict[str, Any],
    benign_task: dict[str, Any],
) -> dict[str, Any]:
    """Rebuild an adversarial task from its Phase 3 benign contract."""
    reward = adversarial_task.get("reward_function")
    if not isinstance(reward, dict):
        raise ValueError("reward_function must be an object")

    adversarial_reward = reward.get("adversarial_reward")
    if not isinstance(adversarial_reward, dict) or not adversarial_reward:
        raise ValueError("reward_function.adversarial_reward must be a non-empty object")

    adversarial_data_seed = _effective_adversarial_seed(adversarial_task)
    try:
        validate_data_seed(adversarial_data_seed, allow_none=False)
    except ValueError as exc:
        raise ValueError(f"invalid adversarial_data_seed: {exc}") from exc
    seed_error = self_contained_adversarial_seed_error(
        benign_task.get("data_seed"),
        adversarial_data_seed,
    )
    if seed_error is not None:
        raise ValueError(seed_error)

    rebuilt = json.loads(json.dumps(benign_task))
    rebuilt["id"] = adversarial_task.get("id", benign_task.get("id"))
    rebuilt["benign_task_id"] = adversarial_task["benign_task_id"]
    rebuilt["adversarial_data_seed"] = json.loads(json.dumps(adversarial_data_seed))
    rebuilt["sites"] = _merged_task_sites(benign_task, adversarial_task)
    rebuilt["reward_function"] = {
        "benign_reward": json.loads(json.dumps(benign_task.get("reward_function", {}))),
        "adversarial_reward": json.loads(json.dumps(adversarial_reward)),
    }

    immutable_fields = {
        "id",
        "benign_task_id",
        "site",
        "sites",
        "instruction",
        "start_urls",
        "data_seed",
        "agent_context",
        "reward_function",
        "adversarial_data_seed",
    }
    for key, value in adversarial_task.items():
        if key in immutable_fields:
            continue
        rebuilt[key] = json.loads(json.dumps(value))
    return rebuilt

def _merged_task_sites(*tasks: dict[str, Any]) -> list[str]:
    merged: list[str] = []
    for task in tasks:
        for raw_site in task.get("sites", []):
            site_name = str(raw_site).strip()
            if site_name and site_name not in merged:
                merged.append(site_name)
        primary_site = str(task.get("site", "")).strip()
        if primary_site and primary_site not in merged:
            merged.append(primary_site)
        delivery_channel = task.get("delivery_channel")
        if isinstance(delivery_channel, dict):
            delivery_site = str(delivery_channel.get("delivery_site") or "").strip()
            if delivery_site and delivery_site.lower() != "none" and delivery_site not in merged:
                merged.append(delivery_site)
    return merged

async def _rerun_adversarial_task(
    task: dict[str, Any],
    instance: BenchmarkInstance,
    all_instances: list[BenchmarkInstance],
    agent_factory: Callable[[], AgentRunner],
    task_dir: Path,
    *,
    resume: bool = False,
    resume_fingerprint: str | None = None,
    benchmark_root: Path | None = None,
    sandbox_model: str = "claude-sonnet-4-6",
    site_profile: dict[str, Any] | None = None,
    agent_execution: dict[str, Any] | None = None,
    browser_worker_semaphore: asyncio.Semaphore | None = None,
) -> dict[str, Any]:
    """Run one revised adversarial task against a live benchmark instance."""
    if resume and resume_fingerprint is not None:
        prior_result = _load_saved_placement_iteration_result(
            task_dir,
            source_fingerprint=resume_fingerprint,
        )
        if prior_result is not None:
            logger.info(
                "Resume: reusing placement rerun result for task %s from %s",
                task.get("id", "unknown"),
                task_dir,
            )
            return prior_result

    async def _run() -> dict[str, Any]:
        agent = agent_factory()
        bound_task = (
            task
            if task_reset_endpoints(task)
            else bind_task_to_instance(task, instance, all_instances)
        )
        try:
            await agent.setup(instance.site_url)
            return await run_adversarial_task(
                bound_task,
                agent,
                instance,
                task_dir,
                benchmark_root=benchmark_root,
                sandbox_model=sandbox_model,
                site_profile=site_profile,
                resume_fingerprint=resume_fingerprint,
            )
        finally:
            await agent.teardown()

    if browser_worker_semaphore is None:
        return await _run()
    async with browser_worker_semaphore:
        return await _run()

async def _evaluate_variant(
    task: dict[str, Any],
    variant: dict[str, Any],
    instance: BenchmarkInstance,
    all_instances: list[BenchmarkInstance],
    strategy: dict[str, Any],
    index: int,
    agent_factory: Callable[[], AgentRunner],
    task_dir_root: Path,
    config_url_placeholders: dict[str, str] | None = None,
    resume: bool = False,
    benchmark_root: Path | None = None,
    sandbox_model: str = "claude-sonnet-4-6",
    site_profile: dict[str, Any] | None = None,
    agent_execution: dict[str, Any] | None = None,
    browser_worker_semaphore: asyncio.Semaphore | None = None,
) -> dict[str, Any]:
    variant_dir = task_dir_root / safe_task_path_component(
        f"{task.get('id', 'unknown')}_variant_{index}"
    )
    variant_dir.mkdir(parents=True, exist_ok=True)
    source_fingerprint = _phase_4_variant_fingerprint(
        task,
        variant,
        strategy,
        instance=instance,
        all_instances=all_instances,
        config_url_placeholders=config_url_placeholders,
        benchmark_root=benchmark_root,
        sandbox_model=sandbox_model,
        site_profile=site_profile,
        agent_execution=agent_execution,
    )

    if resume:
        prior_result = _load_saved_variant_result(
            task_dir_root,
            str(task.get("id", "unknown")),
            index,
            source_fingerprint,
        )
        if prior_result is not None:
            logger.info(
                "Resume: reusing variant %d result for task %s",
                index,
                task.get("id", "unknown"),
            )
            return {
                **prior_result,
                "strategy": strategy.get("strategy"),
            }

    try:
        async def _run() -> dict[str, Any]:
            agent = agent_factory()
            try:
                await agent.setup(instance.site_url)
                bound_variant = bind_task_to_instance(variant, instance, all_instances)
                async with task_lock(bound_variant):
                    return await run_adversarial_task(
                        bound_variant,
                        agent,
                        instance,
                        variant_dir,
                        benchmark_root=benchmark_root,
                        sandbox_model=sandbox_model,
                        site_profile=site_profile,
                        resume_fingerprint=source_fingerprint,
                    )
            finally:
                await agent.teardown()

        if browser_worker_semaphore is None:
            result = await _run()
        else:
            async with browser_worker_semaphore:
                result = await _run()
        _write_json_atomic(
            _variant_result_metadata_path(task_dir_root, str(task.get("id", "unknown")), index),
            {_CHECKPOINT_FINGERPRINT_KEY: source_fingerprint},
            failpoint_base="phase_4.variant.result_metadata",
        )
        return {
            **result,
            "strategy": strategy.get("strategy", f"strategy_{index}"),
        }
    except Exception as e:
        logger.exception("Variant %d evaluation failed: %s", index, e)
        return {
            "task_id": task.get("id", "?"),
            "outcome": "error",
            "error": repr(e),
            "strategy": strategy.get("strategy", f"strategy_{index}"),
        }
