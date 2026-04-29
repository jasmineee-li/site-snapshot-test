"""Phase 4 admission behavior."""
# ruff: noqa: F821

from __future__ import annotations

from worldsim.phase_4._context import install_context

install_context(globals())

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
        if not has_effective_agent_auth(instance.agent_auth):
            errors.append(
                f"site {instance.site_name!r} requires agent_auth in instances.json "
                "because BENCHMARK_PROFILE has authed_user injection surfaces"
            )
            continue
        auth = instance.agent_auth if isinstance(instance.agent_auth, dict) else {}
        auth_type = str(auth.get("type") or "").strip()
        if auth_type == "http_headers":
            try:
                resolve_agent_auth_headers(auth)
            except RuntimeError as exc:
                errors.append(
                    f"site {instance.site_name!r} has invalid http_headers agent_auth: {exc}"
                )
                continue
            parsed = urlparse(str(instance.site_url or ""))
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                errors.append(
                    f"site {instance.site_name!r} has invalid site_url for http_headers "
                    "agent_auth scoping"
                )
        elif auth_type == "http_basic":
            parsed = urlparse(str(instance.site_url or ""))
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                errors.append(
                    f"site {instance.site_name!r} has invalid site_url for http_basic "
                    "agent_auth scoping"
                )
    return errors



def _load_admitted_phase_4_tasks(
    *,
    state_dir: Path,
    sites_filter_raw: str | None,
    max_tasks_per_site: int | None,
    state_metadata: dict[str, Any],
) -> dict[str, Any]:
    def _admission_failure() -> dict[str, Any]:
        return {"return_code": 1, "tasks": [], "active_sites": set()}

    # Load adversarial tasks from Phase 2
    adv_tasks_path = state_dir / "phase_2" / "adversarial_tasks.json"
    if not adv_tasks_path.exists():
        logger.error("Adversarial tasks not found at %s — run phase 2 first", adv_tasks_path)
        return _admission_failure()
    adversarial_tasks = json.loads(adv_tasks_path.read_text())
    try:
        adversarial_tasks = _filter_tasks_by_sites(
            adversarial_tasks,
            sites_filter_raw,
            phase_label="Phase 4",
        )
    except ValueError as exc:
        logger.error("%s", exc)
        return _admission_failure()

    contracts_path = state_dir / "phase_3" / "contracts.json"
    if not contracts_path.exists():
        logger.error("Phase 3 contracts.json not found at %s — run phase 3 first", contracts_path)
        return _admission_failure()
    contract_entries = json.loads(contracts_path.read_text())
    if not isinstance(contract_entries, list):
        logger.error(
            "Phase 3 contracts.json at %s must be a JSON array, got %s",
            contracts_path,
            type(contract_entries).__name__,
        )
        save_state(
            "phase_4",
            status="failed",
            reason="malformed_contracts",
            **state_metadata,
        )
        return _admission_failure()
    contract_errors: list[str] = []
    valid_contracts_by_id: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(contract_entries):
        if not isinstance(entry, dict):
            contract_errors.append(f"entry {index}: not a JSON object")
            continue
        entry_id = entry.get("id")
        if not isinstance(entry_id, str) or not entry_id.strip():
            contract_errors.append(f"entry {index}: missing or empty id")
            continue
        status = entry.get("validity_status")
        if status not in ("valid", "invalid"):
            contract_errors.append(
                f"entry {index} ({entry_id}): validity_status must be 'valid' or 'invalid', "
                f"got {status!r}"
            )
            continue
        if status == "valid" and not isinstance(entry.get("task"), dict):
            contract_errors.append(
                f"entry {index} ({entry_id}): valid contract missing task object"
            )
            continue
        if status == "valid":
            valid_contracts_by_id[str(entry_id)] = entry
    if contract_errors:
        logger.error(
            "Phase 3 contracts.json at %s is malformed:\n%s",
            contracts_path,
            "\n".join(f"  - {msg}" for msg in contract_errors),
        )
        save_state(
            "phase_4",
            status="failed",
            reason="malformed_contracts",
            malformed_contracts=contract_errors,
            **state_metadata,
        )
        return _admission_failure()
    tasks: list[dict[str, Any]] = []
    rebase_errors: list[str] = []
    skipped_invalid = 0
    skipped_orphan = 0
    skipped_infeasible = 0
    skipped_unverified = 0
    skipped_missing_exposure = 0
    sites_filter_set = (
        {site.strip() for site in sites_filter_raw.split(",") if site.strip()}
        if isinstance(sites_filter_raw, str) and sites_filter_raw.strip()
        else None
    )
    exhausted_contract_ids: set[str] = set()
    for entry in contract_entries:
        if not isinstance(entry, dict) or entry.get("adversarially_exhausted") is not True:
            continue
        entry_id = str(entry.get("id", "")).strip()
        task = entry.get("task")
        task_site = str(task.get("site", "")).strip() if isinstance(task, dict) else ""
        if entry_id and (sites_filter_set is None or task_site in sites_filter_set):
            exhausted_contract_ids.add(entry_id)
    grace_warning_emitted = False
    strict_feasibility = _strict_feasibility_enabled()
    admitted_by_origin: dict[str, int] = {"existing_task": 0, "new_task": 0}
    for adversarial_task in adversarial_tasks:
        feasibility = adversarial_task.get("feasibility")
        feasibility_status = feasibility.get("status") if isinstance(feasibility, dict) else None
        if feasibility_status == "infeasible":
            skipped_infeasible += 1
            continue
        if feasibility_status != "verified":
            if strict_feasibility:
                skipped_unverified += 1
                continue
            if not grace_warning_emitted:
                logger.warning(
                    "Phase 4: admitting tasks without feasibility.status='verified' "
                    "(grace mode). Set STRICT_FEASIBILITY_ADMISSION=True or "
                    "WORLDSIM_STRICT_FEASIBILITY=true to enforce."
                )
                grace_warning_emitted = True
        exposure_error = _exposure_admission_error(adversarial_task)
        if exposure_error is not None:
            logger.debug(
                "Phase 4: skipping task %s due to exposure admission failure: %s",
                adversarial_task.get("id", "?"),
                exposure_error,
            )
            skipped_missing_exposure += 1
            continue
        benign_task_id = str(adversarial_task.get("benign_task_id", "")).strip()
        if not benign_task_id:
            rebase_errors.append(f"{adversarial_task.get('id', '?')}: missing benign_task_id")
            continue
        entry = valid_contracts_by_id.get(benign_task_id)
        if entry is None:
            if any(
                str(candidate.get("id", "")) == benign_task_id for candidate in contract_entries
            ):
                skipped_invalid += 1
            else:
                skipped_orphan += 1
            continue
        try:
            rebuilt = _rebase_adversarial_task(adversarial_task, entry["task"])
        except (KeyError, TypeError, ValueError) as exc:
            rebase_errors.append(f"{adversarial_task.get('id', '?')}: {exc}")
            continue
        origin = _normalize_task_origin(entry.get("origin"), task=entry.get("task"))
        rebuilt["origin"] = origin
        admitted_by_origin[origin] = admitted_by_origin.get(origin, 0) + 1
        tasks.append(rebuilt)
    logger.info(
        "Phase 4: admitted %d/%d adversarial tasks (existing_task=%d, new_task=%d); "
        "skipped %d with invalid benign contract, %d with unknown benign_task_id, "
        "%d infeasible, %d unverified, %d without eligible exposure (strict=%s)",
        len(tasks),
        len(adversarial_tasks),
        admitted_by_origin.get("existing_task", 0),
        admitted_by_origin.get("new_task", 0),
        skipped_invalid,
        skipped_orphan,
        skipped_infeasible,
        skipped_unverified,
        skipped_missing_exposure,
        strict_feasibility,
    )
    if rebase_errors:
        logger.error(
            "Phase 4 found malformed adversarial tasks after Phase 3 validation:\n%s",
            "\n".join(f"  - {error}" for error in rebase_errors),
        )
        save_state(
            "phase_4",
            status="failed",
            reason="malformed_adversarial_tasks",
            rebase_errors=rebase_errors,
            **state_metadata,
        )
        return _admission_failure()

    if not tasks:
        if exhausted_contract_ids:
            logger.error(
                "No adversarial tasks to evaluate because Phase 3 marked %d benign "
                "contract(s) adversarially_exhausted",
                len(exhausted_contract_ids),
            )
            save_state(
                "phase_4",
                status="failed",
                reason="dataset_exhausted",
                adversarially_exhausted_contract_count=len(exhausted_contract_ids),
                adversarially_exhausted_contract_ids=sorted(exhausted_contract_ids),
                skipped_infeasible=skipped_infeasible,
                skipped_unverified=skipped_unverified,
                skipped_missing_exposure=skipped_missing_exposure,
                **state_metadata,
            )
            return _admission_failure()
        logger.error("No tasks to evaluate")
        save_state(
            "phase_4",
            status="failed",
            reason="no_validated_adversarial_tasks",
            skipped_infeasible=skipped_infeasible,
            skipped_unverified=skipped_unverified,
            skipped_missing_exposure=skipped_missing_exposure,
            **state_metadata,
        )
        return _admission_failure()

    # Per-site cap for smoke testing (applied after validated-task filtering)
    if max_tasks_per_site is not None:
        pre_cap = len(tasks)
        tasks = cap_tasks_per_site(tasks, max_tasks_per_site)
        post_cap_by_origin: dict[str, int] = {"existing_task": 0, "new_task": 0}
        for task in tasks:
            origin = _normalize_task_origin(task.get("origin"), task=task)
            post_cap_by_origin[origin] = post_cap_by_origin.get(origin, 0) + 1
        logger.info(
            "Phase 4: capped to %d/%d tasks (max %d per site; post-cap existing_task=%d, new_task=%d)",
            len(tasks),
            pre_cap,
            max_tasks_per_site,
            post_cap_by_origin.get("existing_task", 0),
            post_cap_by_origin.get("new_task", 0),
        )

    active_sites = {site for task in tasks for site in _task_reachable_sites(task)}
    return {"return_code": None, "tasks": tasks, "active_sites": active_sites}

def _strict_feasibility_enabled() -> bool:
    import os as _os

    override = _os.environ.get("WORLDSIM_STRICT_FEASIBILITY")
    if override is None or not override.strip():
        return STRICT_FEASIBILITY_ADMISSION
    return override.strip().lower() in {"1", "true", "yes", "on"}

