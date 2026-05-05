"""Phase 4 preflight behavior."""
# ruff: noqa: F821

from __future__ import annotations

from worldsim.phase_4._context import install_context

install_context(globals())

async def _preflight_host_messages_api(*, sandbox_model: str) -> tuple[bool, str | None]:
    """Probe the host-side Anthropic Messages API after local validation passes.

    Keep this after task admission / storage-state / runtime-config checks so
    deterministic local validation errors surface without requiring Claude
    credentials. The actual browser-eval path still fails closed before any
    agent work starts.
    """
    if os.environ.get("WORLDSIM_PHASE_4_SKIP_PREFLIGHT", "").strip() in ("1", "true", "True"):
        logger.info("Phase 4 preflight SKIPPED (WORLDSIM_PHASE_4_SKIP_PREFLIGHT set)")
        return (True, None)

    from worldsim.phase_4.anthropic_client import preflight_check

    ok, err = await preflight_check(sandbox_model=sandbox_model)
    if ok:
        logger.info("Phase 4 preflight OK (model=%s)", sandbox_model)
    return (ok, err)

@dataclass(frozen=True)
class SeedPreflightMismatch:
    call_index: int
    site: str
    resource_type: str
    kind: str
    detail: str

    @property
    def message(self) -> str:
        return self.detail

@dataclass(frozen=True)
class PreflightReport:
    ok: bool
    mismatches: tuple[SeedPreflightMismatch, ...]

@dataclass(frozen=True)
class BaseStateProbeResult:
    ok: bool
    mismatch: SeedPreflightMismatch | None = None

def _serialize_preflight_mismatch_records(
    mismatches: tuple[SeedPreflightMismatch, ...],
) -> list[dict[str, Any]]:
    return [
        {
            "call_index": mismatch.call_index,
            "site": mismatch.site,
            "resource_type": mismatch.resource_type,
            "kind": mismatch.kind,
            "detail": mismatch.detail,
        }
        for mismatch in mismatches
    ]

def _pvpo_endpoint_preflight_errors(
    instances: list[BenchmarkInstance],
    *,
    active_sites: set[str] | None = None,
) -> list[str]:
    """Validate per-instance PVPO endpoint assignment for Phase 4."""
    relevant_instances = [
        instance
        for instance in instances
        if active_sites is None or normalize_site_name(instance.site_name) in active_sites
    ]
    if not relevant_instances:
        return []

    errors: list[str] = []
    seen_urls: dict[str, str] = {}
    for instance in relevant_instances:
        label = instance.replica_name or f"{instance.site_name}[{instance.replica_index}]"
        raw_url = instance.pvpo_cdp_url
        try:
            normalized_url = validate_pvpo_cdp_url(
                raw_url,
                field_name=f"BenchmarkInstance(site={label}).pvpo_cdp_url",
                allow_empty=True,
            )
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if normalized_url is None:
            continue
        endpoint_identity = canonical_pvpo_endpoint_identity(normalized_url)
        prior = seen_urls.get(endpoint_identity)
        if prior is not None:
            errors.append(
                f"duplicate pvpo_cdp_url {normalized_url!r} for instances {prior!r} and {label!r}; "
                "Phase 4 requires one dedicated PVPO browser endpoint per worker"
            )
        else:
            seen_urls[endpoint_identity] = label
    return errors

def _save_seed_preflight_result(
    *,
    task_dir: Path,
    task: dict[str, Any],
    payload: dict[str, Any],
    resume_fingerprint: str | None,
) -> None:
    """Persist deterministic preflight failures so resume can reuse them."""
    extra: dict[str, Any] = {}
    if resume_fingerprint is not None:
        extra[RESULT_FINGERPRINT_KEY] = resume_fingerprint
    save_result(
        task_dir,
        task,
        AgentResult(
            elapsed=float(payload.get("elapsed", 0.0) or 0.0),
            steps=int(payload.get("steps", 0) or 0),
            is_done=False,
            final_result=None,
            status="error",
            errors=[str(payload.get("error", ""))] if payload.get("error") else [],
            network_trace=[],
        ),
        False,
        str(payload.get("error", "")),
        outcome="seed_preflight_mismatch",
        benign_passed=False,
        adversarial_passed=False,
        seed_preflight_mismatches=payload.get("seed_preflight_mismatches", []),
        seed_preflight_mismatch_records=payload.get("seed_preflight_mismatch_records", []),
        trajectory_dir=str(task_dir),
        **extra,
    )

async def preflight_adversarial_seed(
    adv_seed: dict[str, Any],
    instance: dict[str, Any],
    *,
    benchmark: str = "webarena_verified",
    base_state_cache: dict[tuple[str, str, str, str], BaseStateProbeResult] | None = None,
) -> PreflightReport:
    mismatches: list[SeedPreflightMismatch] = []
    try:
        editor_errors = await asyncio.to_thread(preflight_editor_seed_calls, adv_seed, instance)
    except Exception as exc:
        editor_errors = [
            {
                "call_index": -1,
                "site": str(instance.get("site_name", "")).strip() or "unknown",
                "kind": "editor_error",
                "detail": str(exc),
                "method": "unknown",
            }
        ]
    mismatches.extend(_preflight_mismatch_from_editor_error(error) for error in editor_errors)
    task = instance.get("seed_task")
    if isinstance(task, dict):
        delivery_channel = task.get("delivery_channel")
        if isinstance(delivery_channel, dict) and isinstance(
            delivery_channel.get("path_template"), str
        ):
            from worldsim.phase_2 import runner as phase_2_contracts

            try:
                contract_error = phase_2_contracts._validate_finalized_http_seed_contract(
                    adv_seed,
                    delivery_channel,
                    sites=task.get("sites"),
                )
            except Exception as exc:
                mismatches.append(
                    SeedPreflightMismatch(
                        call_index=-1,
                        site=str(instance.get("site_name", "")).strip() or "unknown",
                        resource_type="contract",
                        kind="contract_error",
                        detail=str(exc),
                    )
                )
            else:
                if contract_error is not None:
                    mismatches.append(
                        SeedPreflightMismatch(
                            call_index=-1,
                            site=str(instance.get("site_name", "")).strip() or "unknown",
                            resource_type="contract",
                            kind="contract_error",
                            detail=contract_error,
                        )
                    )
    if mismatches:
        return PreflightReport(ok=False, mismatches=tuple(mismatches))
    if _seed_uses_editor_calls(adv_seed):
        base_state = _probe_seed_base_state(instance, benchmark=benchmark, cache=base_state_cache)
        if not base_state.ok and base_state.mismatch is not None:
            return PreflightReport(ok=False, mismatches=(base_state.mismatch,))
    return PreflightReport(ok=True, mismatches=())

def _preflight_mismatch_from_editor_error(error: dict[str, Any]) -> SeedPreflightMismatch:
    return SeedPreflightMismatch(
        call_index=int(error.get("call_index", -1)),
        site=str(error.get("site", "unknown")).strip() or "unknown",
        resource_type=str(error.get("method", "unknown")).strip() or "unknown",
        kind=str(error.get("kind", "editor_error")).strip() or "editor_error",
        detail=str(error.get("detail", "editor preflight failed")),
    )

def _probe_seed_base_state_for_task_targets(
    tasks: list[dict[str, Any]],
    instances: list[BenchmarkInstance],
    *,
    cache: dict[tuple[str, str, str, str], BaseStateProbeResult] | None = None,
) -> list[str]:
    errors: list[str] = []
    seen_cache_keys: set[tuple[str, str, str, str]] = set()
    for task in tasks:
        if not isinstance(task, dict):
            continue
        seed = task.get("adversarial_data_seed")
        if not _seed_uses_editor_calls(seed):
            continue
        seed_site = _seed_target_site(task)
        if not seed_site:
            continue
        try:
            instance = select_task_site_instance(task, seed_site, instances)
        except ValueError:
            errors.append(
                f"base-state probe could not find configured instance for site {seed_site!r}"
            )
            continue
        instance_dict = instance.model_dump()
        try:
            seed_benchmark = _seed_target_benchmark(task, instance_dict)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        cache_key = _probe_seed_cache_key(instance_dict, benchmark=seed_benchmark)
        if cache_key in seen_cache_keys:
            continue
        seen_cache_keys.add(cache_key)
        result = _probe_seed_base_state(instance_dict, benchmark=seed_benchmark, cache=cache)
        if not result.ok and result.mismatch is not None:
            errors.append(result.mismatch.message)
    return errors

def _probe_seed_base_state(
    instance: dict[str, Any],
    *,
    benchmark: str = "webarena_verified",
    cache: dict[tuple[str, str, str, str], BaseStateProbeResult] | None = None,
) -> BaseStateProbeResult:
    site_name, site_url, cache_key = _probe_seed_cache_parts(instance, benchmark=benchmark)
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    if not site_name or not site_url:
        result = BaseStateProbeResult(
            ok=False,
            mismatch=SeedPreflightMismatch(
                call_index=-1,
                site=site_name or "unknown",
                resource_type="base_state",
                kind="base_state_missing",
                detail="instance is missing site_name or site_url for base-state probe",
            ),
        )
        if cache is not None:
            cache[cache_key] = result
        return result
    try:
        editor_cls = EDITOR_REGISTRY.get((benchmark, site_name))
        if editor_cls is None:
            result = BaseStateProbeResult(ok=True)
            if cache is not None:
                cache[cache_key] = result
            return result
        editor_cls.probe_base_state(instance)
    except EditorError as exc:
        result = BaseStateProbeResult(
            ok=False,
            mismatch=SeedPreflightMismatch(
                call_index=-1,
                site=site_name,
                resource_type="base_state",
                kind=exc.kind,
                detail=exc.detail,
            ),
        )
    except Exception as exc:
        result = BaseStateProbeResult(
            ok=False,
            mismatch=SeedPreflightMismatch(
                call_index=-1,
                site=site_name,
                resource_type="base_state",
                kind="base_state_missing",
                detail=str(exc),
            ),
        )
    else:
        result = BaseStateProbeResult(ok=True)
    if cache is not None:
        cache[cache_key] = result
    return result

def _probe_seed_cache_parts(
    instance: dict[str, Any],
    *,
    benchmark: str = "webarena_verified",
) -> tuple[str, str, tuple[str, str, str, str]]:
    site_name = str(instance.get("site_name", "")).strip().lower()
    site_url = str(instance.get("site_url", "")).rstrip("/")
    return site_name, site_url, _probe_seed_cache_key(instance, benchmark=benchmark)

def _probe_seed_cache_key(
    instance: dict[str, Any],
    *,
    benchmark: str = "webarena_verified",
) -> tuple[str, str, str, str]:
    site_name = str(instance.get("site_name", "")).strip().lower()
    site_url = str(instance.get("site_url", "")).rstrip("/")
    auth_fingerprint = _fingerprint_payload(
        instance.get("replica_index"),
        instance.get("replica_name"),
        instance.get("auth"),
        instance.get("api_auth"),
        instance.get("agent_auth"),
    )
    return (benchmark, site_name, site_url, auth_fingerprint)
