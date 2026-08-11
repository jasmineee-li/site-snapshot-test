"""Phase 2 shards behavior."""
# ruff: noqa: F821

from __future__ import annotations

from worldsim.phase_2._context import install_context

install_context(globals())


def _shard_tasks(tasks: list[dict], shard_size: int) -> list[list[dict]]:
    """Split a task list into chunks of at most *shard_size*."""
    return [tasks[i : i + shard_size] for i in range(0, len(tasks), shard_size)]


async def _run_shard_with_limit(
    limiter: asyncio.Semaphore,
    **kwargs: Any,
) -> SiteInjectionResult:
    """Apply bounded concurrency around one Phase 2a API shard."""
    async with limiter:
        return await _generate_injections_for_site(**kwargs)


def _merge_shard_results(
    shard_results: list[SiteInjectionResult | BaseException],
    tasks_by_site: dict[str, list[dict]],
) -> list[SiteInjectionResult]:
    """Collapse per-shard results into one SiteInjectionResult per site."""
    site_tasks_acc: dict[str, list[dict]] = {}
    site_errors_acc: dict[str, list[str]] = {}
    # Track exceptions as site-level errors.
    for result in shard_results:
        if isinstance(result, BaseException):
            # Cannot attribute to a specific site, surface as-is.
            site_errors_acc.setdefault("_unknown_", []).append(str(result))
            continue
        site = result.site_name
        site_tasks_acc.setdefault(site, []).extend(result.adversarial_tasks)
        site_errors_acc.setdefault(site, []).extend(result.errors)

    merged: list[SiteInjectionResult] = []
    for site in tasks_by_site:
        merged.append(
            SiteInjectionResult(
                site_name=site,
                adversarial_tasks=site_tasks_acc.get(site, []),
                errors=site_errors_acc.get(site, []),
            )
        )
    # Surface any unattributed exceptions as a synthetic result.
    unknown_errors = site_errors_acc.get("_unknown_", [])
    if unknown_errors:
        merged.append(SiteInjectionResult("_unknown_", [], unknown_errors))
    return merged


def _recover_orphaned_shards(
    shards_dir: Path,
    in_memory_plans: list[dict[str, Any]],
    *,
    allowed_sites: set[str],
    benign_by_id: dict[str, dict[str, Any]] | None = None,
    site_profiles: dict[str, dict[str, Any]] | None = None,
    required_checkpoint_definition: RunDefinition | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Fold disk-persisted shard tasks that the in-memory aggregation missed.

    Phase 2a writes each validated shard to ``shards_dir`` at line 1725.
    If a shard re-runs in isolation (or the orchestrator crashes mid-run
    and resumes), only the latest shard's ``SiteInjectionResult`` lives
    in memory — earlier sidecars are valid, enriched, and on disk, but
    otherwise silently dropped.

    Scan shard JSON files (including one-shard ``<site>.json`` files), ignore ids already in
    ``in_memory_plans``, filter to ``allowed_sites``, and append the
    surviving tasks. On cross-shard id collision, newest-mtime wins.
    Returns ``(merged_plans, recovered_ids)``.
    """
    if not shards_dir.is_dir():
        return list(in_memory_plans), []
    in_memory_ids = {str(plan.get("id") or "") for plan in in_memory_plans if plan.get("id")}
    best_by_id: dict[str, tuple[float, dict[str, Any]]] = {}
    for shard_path in sorted(shards_dir.glob("*.json")):
        if shard_path.name.endswith(".manifest.json"):
            continue
        try:
            data = json.loads(shard_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Phase 2 orphan recovery: skipping %s (%s)", shard_path.name, exc)
            continue
        if not isinstance(data, list):
            continue
        if required_checkpoint_definition is not None and not planning_shard_checkpoint_matches(
            shard_path,
            data,
            definition=required_checkpoint_definition,
        ):
            logger.warning(
                "Phase 2 orphan recovery: skipping unbound planning checkpoint %s",
                shard_path.name,
            )
            continue
        mtime = shard_path.stat().st_mtime
        for task in data:
            if not isinstance(task, dict):
                continue
            task_id = str(task.get("id") or "")
            if not task_id or task_id in in_memory_ids:
                continue
            site = _effective_task_site(task)
            if site not in allowed_sites:
                continue
            prior = best_by_id.get(task_id)
            if prior is None or mtime > prior[0]:
                best_by_id[task_id] = (mtime, task)
    if not best_by_id:
        return list(in_memory_plans), []
    orphans, dropped_count = _validate_recovered_planning_tasks(
        [task for _, task in best_by_id.values()],
        benign_by_id=benign_by_id,
        site_profiles=site_profiles,
    )
    if not orphans:
        if dropped_count:
            logger.info(
                "Phase 2 aggregation: dropped %d orphan shard task(s) failing live validators",
                dropped_count,
            )
        return list(in_memory_plans), []
    if dropped_count:
        logger.info(
            "Phase 2 aggregation: kept %d orphan shard task(s); dropped %d failing live validators",
            len(orphans),
            dropped_count,
        )
    _reconstruct_orphan_start_urls(orphans)
    merged = list(in_memory_plans) + orphans
    _normalize_l4_benign_task_ids_in_place(merged)
    recovered_ids = sorted(str(task.get("id") or "") for task in orphans)
    return merged, recovered_ids


def _load_reusable_planning_shard(
    shard_path: Path,
    *,
    expected_site: str,
    expected_input_task_ids: list[str],
    definition: RunDefinition,
    benign_by_id: dict[str, dict[str, Any]],
    site_profiles: dict[str, dict[str, Any]],
) -> list[dict[str, Any]] | None:
    """Load one exact paused shard before admitting another API call."""

    try:
        data = json.loads(shard_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not isinstance(data, list)
        or not data
        or not planning_shard_checkpoint_matches(
            shard_path,
            data,
            definition=definition,
            expected_input_task_ids=expected_input_task_ids,
        )
    ):
        return None
    if any(_effective_task_site(task) != expected_site for task in data):
        return None
    reusable, dropped_count = _validate_recovered_planning_tasks(
        data,
        benign_by_id=benign_by_id,
        site_profiles=site_profiles,
    )
    if dropped_count or len(reusable) != len(data):
        return None
    _reconstruct_orphan_start_urls(reusable)
    _normalize_l4_benign_task_ids_in_place(reusable)
    return reusable


def _validate_recovered_planning_tasks(
    candidates: list[dict[str, Any]],
    *,
    benign_by_id: dict[str, dict[str, Any]] | None,
    site_profiles: dict[str, dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], int]:
    """Apply the existing Phase 2 contract and placement validators."""

    # Re-run the live Phase 2a validator chain on every candidate
    # orphan from an Option A site. Stale shards can pre-date the
    # api/form/state_push sunset (commit ff8381d5) and carry
    # `seed_template.mechanism="api"` with `api_calls` instead of
    # `editor_calls`, or carry contract violations like
    # `editor_calls[].site` mismatching the task site. Mirror the live
    # `_validate_generated_adversarial_task` order: contract first, then
    # placement. Skip contract validation only when the caller did not
    # supply benign/site-profile context (legacy callers in tests).
    orphans: list[dict[str, Any]] = []
    dropped_count = 0
    for task in candidates:
        if _is_option_a_site(task):
            task_name = f"orphan {task.get('id') or '<unknown>'}"
            if benign_by_id is not None and site_profiles is not None:
                benign_parent = benign_by_id.get(str(task.get("benign_task_id", "")))
                site_profile = site_profiles.get(_effective_task_site(task))
                if benign_parent is None or site_profile is None:
                    logger.warning(
                        "[phase_2] skip-on-reject: %s (contract): current benign parent or site profile is missing",
                        task_name,
                    )
                    dropped_count += 1
                    continue
                contract_error = _validate_adversarial_task_contract(
                    task, benign_parent, site_profile
                )
                if contract_error is not None:
                    logger.warning(
                        "[phase_2] skip-on-reject: %s (contract): %s",
                        task_name,
                        contract_error,
                    )
                    dropped_count += 1
                    continue
                stale_contract_reason = _stale_reusable_exposure_contract_reason(task)
                if stale_contract_reason is not None:
                    logger.warning(
                        "[phase_2] skip-on-reject: %s (stale exposure_contract): %s",
                        task_name,
                        stale_contract_reason,
                    )
                    dropped_count += 1
                    continue
            placement_error = _validate_option_a_placement(task, task_name)
            if placement_error is not None:
                logger.warning(
                    "[phase_2] skip-on-reject: %s (Option A placement): %s",
                    task_name,
                    placement_error,
                )
                dropped_count += 1
                continue
        orphans.append(task)
    return orphans, dropped_count
