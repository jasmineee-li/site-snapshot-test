"""Phase 4 resume behavior."""
# ruff: noqa: F821

from __future__ import annotations

from worldsim.phase_4._context import install_context

install_context(globals())


def _sweep_orphan_inflight_sentinels(task_dir_root: Path) -> int:
    """Delete legacy ``.aer_inflight`` sentinel files left on disk by old runs.

    Pre-cutover code wrote an ``.aer_inflight`` resume-hint file at PVPO
    gate entry and unlinked it on clean exit. Nothing consumed the sentinel
    for a routing decision (resume is driven by the ``processed_result.json``
    fingerprint check in ``_postprocess_one_task``), so the sentinel was
    removed in a follow-up sweep. This helper stays so re-runs of older
    trajectories don't leave empty marker files lying around.

    Returns the count of sentinel files removed (useful for tests and
    log-level triage).
    """
    if not task_dir_root.exists():
        return 0
    orphans = list(task_dir_root.rglob(_LEGACY_AER_INFLIGHT_SENTINEL))
    for orphan in orphans:
        try:
            orphan.unlink()
        except OSError:
            pass
    if orphans:
        logger.warning(
            "Phase 4: swept %d legacy %s sentinel(s)",
            len(orphans),
            _LEGACY_AER_INFLIGHT_SENTINEL,
        )
    return len(orphans)


def _resume_fingerprint_task(task: dict[str, Any]) -> dict[str, Any]:
    """Strip execution-local worker binding from initial-result fingerprinting."""
    normalized = json.loads(json.dumps(task))
    normalized.pop(RUNTIME_METADATA_KEY, None)
    return normalized


def _resume_fingerprint_result(result: dict[str, Any]) -> dict[str, Any]:
    """Project a result dict to the fields that round-trip through ``result.json``."""
    return {k: result[k] for k in _FINGERPRINT_RESULT_KEYS if k in result}


def _phase_4_state_metadata(
    *,
    task_dir_root: Path,
    instances_path: Path | str,
    agent_model: str,
    sandbox_model: str,
    agent_provider: str | None,
    agent_service_tier: str | None,
    max_tasks_per_site: int | None,
    sites: str | None,
    benchmark_root: Path | None,
    allow_unknown_auth: bool,
    skip_host_bound_storage_state_auth: bool,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "task_dir_root": str(task_dir_root),
        "instances_path": str(instances_path),
        "agent_model": agent_model,
        "sandbox_model": sandbox_model,
        "agent_provider": agent_provider,
        "agent_service_tier": agent_service_tier,
        "max_tasks_per_site": max_tasks_per_site,
        "allow_unknown_auth": allow_unknown_auth,
        "skip_host_bound_storage_state_auth": skip_host_bound_storage_state_auth,
    }
    if sites is not None:
        metadata["sites"] = sites
    if benchmark_root is not None:
        metadata["benchmark_path"] = str(benchmark_root)
    return metadata


def _write_json_atomic(
    path: Path,
    payload: dict[str, Any],
    *,
    failpoint_base: str | None = None,
) -> None:
    write_json_atomic(path, payload, failpoint_base=failpoint_base)


def _fingerprint_payload(*parts: Any) -> str:
    return fingerprint_payload(*parts)


def _phase_4_eval_context(
    *,
    instances: list[BenchmarkInstance],
    config_url_placeholders: dict[str, str] | None,
    agent_model: str,
    agent_provider: str | None,
    sandbox_model: str,
    benchmark_root: Path | None,
) -> dict[str, Any]:
    return {
        "phase": "phase_4_initial_result",
        "resume_version": _PHASE_4_RESUME_VERSION,
        "instances": instances_identity(instances),
        "config_url_placeholders": config_url_placeholders,
        "agent_model": agent_model,
        "agent_provider": agent_provider,
        "sandbox_model": sandbox_model,
        "benchmark_root": str(benchmark_root) if benchmark_root is not None else None,
    }


def _task_reachable_sites(task: dict[str, Any]) -> list[str]:
    sites = normalize_task_sites(task)
    for candidate in (
        _delivery_site_name(task.get("delivery_channel")),
        _seed_target_site(task),
    ):
        normalized = normalize_site_name(candidate)
        if normalized and normalized not in sites:
            sites.append(normalized)
    return sites


def _task_reachable_instances(
    task: dict[str, Any],
    instances: list[BenchmarkInstance],
) -> list[BenchmarkInstance]:
    reachable_sites = set(_task_reachable_sites(task))
    if not reachable_sites:
        return list(instances)
    return [
        instance
        for instance in instances
        if normalize_site_name(instance.site_name) in reachable_sites
    ]


def _task_reachable_placeholders(
    task: dict[str, Any],
    config_url_placeholders: dict[str, str] | None,
) -> dict[str, str] | None:
    if not config_url_placeholders:
        return config_url_placeholders
    allowed = {
        placeholder
        for site in _task_reachable_sites(task)
        if (placeholder := placeholder_for_site(site))
    }
    return {token: value for token, value in config_url_placeholders.items() if token in allowed}


def _phase_4_eval_context_for_task(
    task: dict[str, Any],
    *,
    instances: list[BenchmarkInstance],
    config_url_placeholders: dict[str, str] | None,
    agent_model: str,
    agent_provider: str,
    sandbox_model: str,
    benchmark_root: Path | None,
) -> dict[str, Any]:
    return _phase_4_eval_context(
        instances=_task_reachable_instances(task, instances),
        config_url_placeholders=_task_reachable_placeholders(task, config_url_placeholders),
        agent_model=agent_model,
        agent_provider=agent_provider,
        sandbox_model=sandbox_model,
        benchmark_root=benchmark_root,
    )


def _phase_4_result_fingerprint(
    task: dict[str, Any],
    *,
    eval_context: dict[str, Any],
    site_profile: dict[str, Any] | None,
) -> str:
    return _fingerprint_payload(_resume_fingerprint_task(task), eval_context, site_profile)


def _seed_target_site(task: dict[str, Any]) -> str:
    delivery_channel = task.get("delivery_channel")
    delivery_site = _delivery_site_name(delivery_channel)
    return delivery_site or str(task.get("site", "")).strip()


def _seed_target_benchmark(task: dict[str, Any], *metadata_sources: Mapping[str, Any]) -> str:
    values: list[Any] = [
        task.get("benchmark"),
        task.get("benchmark_name"),
        task.get("benchmark_adapter"),
    ]
    for source in metadata_sources:
        if not isinstance(source, Mapping):
            continue
        values.extend(
            (
                source.get("benchmark"),
                source.get("benchmark_name"),
                source.get("benchmark_adapter"),
            )
        )
    seed = task.get("adversarial_data_seed")
    if isinstance(seed, dict):
        for call in seed.get("editor_calls", []):
            if not isinstance(call, dict):
                continue
            values.extend(
                (
                    call.get("benchmark"),
                    call.get("benchmark_name"),
                    call.get("benchmark_adapter"),
                )
            )
    try:
        benchmark = infer_benchmark_name(values)
    except ValueError as exc:
        raise ValueError(f"invalid adversarial seed benchmark metadata: {exc}") from exc
    if benchmark is None:
        raise ValueError("adversarial seed is missing benchmark metadata")
    return benchmark


def _seed_target_sites(tasks: list[dict[str, Any]]) -> list[str]:
    sites: set[str] = set()
    for task in tasks:
        if not isinstance(task, dict):
            continue
        seed = task.get("adversarial_data_seed")
        if not _seed_uses_editor_calls(seed):
            continue
        site_name = _seed_target_site(task)
        if site_name:
            sites.add(site_name)
    return sorted(sites)


def _seed_uses_editor_calls(seed: Any) -> bool:
    if not isinstance(seed, dict):
        return False
    editor_calls = seed.get("editor_calls")
    if not isinstance(editor_calls, list):
        return False
    return any(isinstance(call, dict) for call in editor_calls)


def _seed_has_actions(seed: Any) -> bool:
    return _seed_contract_has_actions(seed)


def _seed_requires_reset(seed: Any) -> bool:
    return _seed_contract_requires_reset(seed)


def _phase_4_postprocess_fingerprint(
    task: dict[str, Any],
    result: dict[str, Any],
    *,
    primary_instances: list[BenchmarkInstance],
    all_instances: list[BenchmarkInstance],
    config_url_placeholders: dict[str, str] | None,
    benchmark_root: Path | None,
    sandbox_model: str,
    site_profile: dict[str, Any] | None,
) -> str:
    return _fingerprint_payload(
        task,
        _resume_fingerprint_result(result),
        {
            "phase": "phase_4_postprocess",
            "resume_version": _PHASE_4_RESUME_VERSION,
            "primary_instances": instances_identity(primary_instances),
            "all_instances": instances_identity(_task_reachable_instances(task, all_instances)),
            "config_url_placeholders": _task_reachable_placeholders(task, config_url_placeholders),
            "benchmark_root": str(benchmark_root) if benchmark_root is not None else None,
            "sandbox_model": sandbox_model,
            "site_profile": site_profile,
        },
    )


def _phase_4_variant_fingerprint(
    task: dict[str, Any],
    variant: dict[str, Any],
    strategy: dict[str, Any],
    *,
    instance: BenchmarkInstance,
    all_instances: list[BenchmarkInstance],
    config_url_placeholders: dict[str, str] | None,
    benchmark_root: Path | None,
    sandbox_model: str,
    site_profile: dict[str, Any] | None,
) -> str:
    return _fingerprint_payload(
        task,
        variant,
        strategy,
        {
            "phase": "phase_4_variant",
            "resume_version": _PHASE_4_RESUME_VERSION,
            "instance": instance_identity(instance),
            "all_instances": instances_identity(_task_reachable_instances(task, all_instances)),
            "config_url_placeholders": _task_reachable_placeholders(task, config_url_placeholders),
            "benchmark_root": str(benchmark_root) if benchmark_root is not None else None,
            "sandbox_model": sandbox_model,
            "site_profile": site_profile,
        },
    )


def _placement_iteration_result_fingerprint(
    task: dict[str, Any],
    *,
    base_source_fingerprint: str,
    iteration: int,
) -> str:
    return _fingerprint_payload(
        _resume_fingerprint_task(task),
        {
            "phase": "phase_4_placement_iteration",
            "resume_version": _PHASE_4_RESUME_VERSION,
            "base_source_fingerprint": base_source_fingerprint,
            "iteration": iteration,
        },
    )


def _strategy_variation_checkpoint_path(task_dir_root: Path, task_id: str) -> Path:
    return task_dir_root / safe_task_path_component(task_id) / "strategy_variation_checkpoint.json"


def _placement_fix_checkpoint_path(task_dir_root: Path, task_id: str) -> Path:
    return task_dir_root / safe_task_path_component(task_id) / _PLACEMENT_FIX_CHECKPOINT


def _variant_result_metadata_path(task_dir_root: Path, task_id: str, index: int) -> Path:
    variant_dir = task_dir_root / safe_task_path_component(f"{task_id}_variant_{index}")
    return variant_dir / _VARIANT_RESULT_METADATA


def _load_json_dict(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _has_phase_4_resume_artifacts(payload: dict[str, Any], *, trajectory_dir: Path) -> bool:
    outcome = payload.get("outcome")
    if outcome is None or outcome in {"seed_preflight_mismatch", "error", "complied"}:
        return True
    history_path = trajectory_dir / "history.json"
    if not history_path.exists():
        return False
    try:
        history_payload = json.loads(history_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return isinstance(history_payload, (dict, list))


def _normalize_saved_adversarial_result(
    payload: dict[str, Any],
    *,
    trajectory_dir: Path,
) -> dict[str, Any]:
    """Project a saved ``result.json`` sentinel back to the live runtime shape."""
    outcome = payload.get("outcome")
    normalized: dict[str, Any] = {
        "task_id": payload.get("task_id"),
        "trajectory_dir": str(trajectory_dir),
    }
    for key in ("outcome", "encounter", "elapsed", "steps"):
        if key in payload:
            normalized[key] = payload.get(key)
    for key in _FINGERPRINT_RESULT_KEYS:
        if key in {
            "task_id",
            "outcome",
            "encounter",
            "elapsed",
            "steps",
            "benign_passed",
            "adversarial_passed",
            "trajectory_dir",
            "error",
        }:
            continue
        if key in payload:
            normalized[key] = payload.get(key)
    if outcome == "error":
        error = payload.get("error") or payload.get("message")
        if error is not None:
            normalized["error"] = error
        if "passed" in payload:
            normalized["benign_passed"] = False
            normalized["adversarial_passed"] = False
    else:
        error = payload.get("error")
        if error is not None:
            normalized["error"] = error
        if "passed" in payload:
            normalized["benign_passed"] = bool(payload.get("passed"))
            normalized["adversarial_passed"] = outcome == "complied"
    return normalized


def _load_saved_variant_result(
    task_dir_root: Path,
    task_id: str,
    index: int,
    source_fingerprint: str,
) -> dict[str, Any] | None:
    variant_dir = task_dir_root / safe_task_path_component(f"{task_id}_variant_{index}")
    result_file = variant_dir / "result.json"
    if not result_file.exists():
        return None
    metadata = _load_json_dict(_variant_result_metadata_path(task_dir_root, task_id, index))
    payload = _load_json_dict(result_file)
    if payload is None:
        return None
    metadata_fingerprint = (
        metadata.get(_CHECKPOINT_FINGERPRINT_KEY) if isinstance(metadata, dict) else None
    )
    payload_fingerprint = payload.get(RESULT_FINGERPRINT_KEY)
    if metadata_fingerprint != source_fingerprint and payload_fingerprint != source_fingerprint:
        return None
    if not _has_phase_4_resume_artifacts(payload, trajectory_dir=variant_dir):
        return None
    return _normalize_saved_adversarial_result(payload, trajectory_dir=variant_dir)


def _variant_changes_seed(original_task: dict[str, Any], variant_task: dict[str, Any]) -> bool:
    return json.dumps(
        original_task.get("adversarial_data_seed"),
        sort_keys=True,
    ) != json.dumps(
        variant_task.get("adversarial_data_seed"),
        sort_keys=True,
    )


def _write_placement_fix_checkpoint(
    checkpoint_path: Path,
    *,
    source_fingerprint: str,
    payload: dict[str, Any],
) -> None:
    _write_json_atomic(
        checkpoint_path,
        {
            _CHECKPOINT_FINGERPRINT_KEY: source_fingerprint,
            **payload,
        },
        failpoint_base="phase_4.placement_fix.checkpoint",
    )


def _load_saved_placement_iteration_result(
    task_dir: Path,
    *,
    source_fingerprint: str,
) -> dict[str, Any] | None:
    payload = _load_json_dict(task_dir / "result.json")
    if payload is None:
        return None
    if payload.get(RESULT_FINGERPRINT_KEY) != source_fingerprint:
        return None
    if not _has_phase_4_resume_artifacts(payload, trajectory_dir=task_dir):
        return None
    return _normalize_saved_adversarial_result(payload, trajectory_dir=task_dir)


def _variant_generation_record_for_result(
    *,
    index: int,
    strategy: dict[str, Any],
    variant: dict[str, Any] | None = None,
    error: str | None = None,
    status: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "index": index,
        "strategy": strategy,
    }
    if variant is not None:
        record["variant"] = variant
    if error is not None:
        record["error"] = error
    if status is not None:
        record["status"] = status
    if reason is not None:
        record["reason"] = reason
    return record


def _rebuild_variant_generation_progress(
    task: dict[str, Any],
    checkpoint: dict[str, Any] | None,
    *,
    selected_strategies: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], set[int]]:
    records = checkpoint.get(_VARIANT_GENERATION_RECORDS_KEY) if checkpoint else None
    if not isinstance(records, list):
        variant_candidates = checkpoint.get("variant_candidates") if checkpoint else None
        variant_generation_errors = (
            checkpoint.get("variant_generation_errors") if checkpoint else None
        )
        if not isinstance(variant_candidates, list):
            variant_candidates = []
        if not isinstance(variant_generation_errors, list):
            variant_generation_errors = []
        return variant_candidates, variant_generation_errors, [], set()

    variant_candidates: list[dict[str, Any]] = []
    variant_generation_errors: list[dict[str, Any]] = []
    normalized_records: list[dict[str, Any]] = []
    completed_indexes: set[int] = set()
    for raw_record in records:
        if not isinstance(raw_record, dict):
            continue
        index = raw_record.get("index")
        if not isinstance(index, int) or not 0 <= index < len(selected_strategies):
            continue
        if index in completed_indexes:
            continue
        strategy = raw_record.get("strategy")
        if not isinstance(strategy, dict):
            strategy = selected_strategies[index]
        record = {
            "index": index,
            "strategy": strategy,
        }
        variant = raw_record.get("variant")
        if isinstance(variant, dict):
            record["variant"] = variant
            if _variant_changes_seed(task, variant):
                variant_candidates.append({"variant": variant, "strategy": strategy})
        else:
            error = raw_record.get("error")
            status = raw_record.get("status")
            reason = raw_record.get("reason", "")
            if isinstance(error, str):
                record["error"] = error
                variant_generation_errors.append(
                    {
                        "strategy": strategy.get("strategy", f"strategy_{index}"),
                        "error": error,
                    }
                )
            elif isinstance(status, str):
                record["status"] = status
                if isinstance(reason, str):
                    record["reason"] = reason
                if status in {"inapplicable", "skipped", "failed"}:
                    variant_generation_errors.append(
                        {
                            "strategy": strategy.get("strategy", f"strategy_{index}"),
                            "status": status,
                            "reason": reason if isinstance(reason, str) else "",
                        }
                    )
        normalized_records.append(record)
        completed_indexes.add(index)
    normalized_records.sort(key=lambda record: int(record["index"]))
    return variant_candidates, variant_generation_errors, normalized_records, completed_indexes


def _tasks_equivalent(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Return True when two task payloads are materially identical."""
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)
