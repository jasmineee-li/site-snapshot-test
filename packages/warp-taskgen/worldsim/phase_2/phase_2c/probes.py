"""Phase 2c render and reachability probes."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from worldsim.benchmark_capabilities import infer_benchmark_name
from worldsim.phase_2.phase_2c import auth_preflight as _auth_preflight
from worldsim.phase_2.phase_2c.exposure import (
    _first_rendered_payload,
    _phase4_exposure_inadmissible_reason,
    _reachability_resource_for_task,
    _required_url_token,
    _selected_rendered_payload,
)
from worldsim.phases.phase_2_reachability import (
    ReachabilityOutcome,
    derive_second_witness,
    verify_reachable,
)
from worldsim.phases.phase_2_render_check import (
    RenderOutcome,
    _render_check_inputs_from_metadata,
    render_signature,
    render_signature_selection,
    verify_seed_renders,
)

logger = logging.getLogger(__name__)

# Phase 2c render verification is a hard gate by default. The env-var opt-out
# is intended for unit tests that mock the seed flow and don't need a browser,
# and for development hosts without Playwright installed. Production runs MUST
# leave this unset.
_SKIP_RENDER_CHECK_ENV = "WORLDSIM_PHASE_2C_SKIP_RENDER_CHECK"

# Global browser-probe cap. Per-replica caps bulkhead against the
# *backend*; this one bulkheads against the *client*. GitLab ships
# deferred JS that fights for CPU; 64-wide renderers on r5.4xlarge
# starved the scheduler enough to trip the 30 s ``domcontentloaded``
# timeout even on a healthy replica returning in <1.4 s. Capping the
# number of concurrent Chromium processes globally gives each renderer
# ~2 vCPU headroom and eliminates the nav-failed tail.
_BROWSER_PROBE_CAP = 8

# RenderOutcome.kind value the render-check sets when the seed wrote OK
# but the signature was not visible in any read-surface URL within the
# body-poll window. Hoisted as a constant so the retry-once gate stays
# grep-friendly and tests can pin the kind without importing
# verify_seed_renders' classifier internals.
_RENDER_UNVERIFIED_KIND = "render_unverified"

# Single retry breather for the render-check on its first miss. Targets
# GitLab's slow write-to-visible tail (sidekiq + page-cache invalidation)
# under Phase 2c's 16-way renderer contention.
_RENDER_UNVERIFIED_RETRY_DELAY_S = 3.0

# Launch flags applied to every per-task Chromium. ``--disable-dev-shm-usage``
# moves shared memory from ``/dev/shm`` (64 MiB Docker default) to ``/tmp``;
# ``--disable-gpu`` drops the GPU process per renderer under headless;
# ``--no-sandbox`` is acceptable because Phase 2c probes internal WASP replicas.
_PROBE_LAUNCH_ARGS: tuple[str, ...] = (
    "--disable-dev-shm-usage",
    "--disable-gpu",
    "--no-sandbox",
)

_PATCHABLE_GLOBAL_NAMES = (
    "ReachabilityOutcome",
    "RenderOutcome",
    "derive_second_witness",
    "infer_benchmark_name",
    "render_signature",
    "render_signature_selection",
    "verify_reachable",
    "verify_seed_renders",
    "_auth_probe_failure_kind",
    "_first_rendered_payload",
    "_instance_benchmark_or_none",
    "_phase4_exposure_inadmissible_reason",
    "_reachability_resource_for_task",
    "_render_check_inputs_from_metadata",
    "_required_url_token",
    "_resolve_benign_browser_context_auth",
    "_selected_rendered_payload",
    "_BROWSER_PROBE_CAP",
    "_PROBE_LAUNCH_ARGS",
    "_RENDER_UNVERIFIED_KIND",
    "_RENDER_UNVERIFIED_RETRY_DELAY_S",
    "_SKIP_RENDER_CHECK_ENV",
    "logger",
)


def _patchable_globals() -> dict[str, Any]:
    return {name: globals()[name] for name in _PATCHABLE_GLOBAL_NAMES}


def _restore_patchable_globals(values: dict[str, Any]) -> None:
    for name, value in values.items():
        globals()[name] = value


async def _ensure_playwright_chromium_ready(async_playwright_factory: Any) -> None:
    """Fail once, before worker fan-out, when the browser bundle is missing."""

    pw_handle: Any = None
    try:
        pw_handle = await async_playwright_factory().start()
        chromium = getattr(pw_handle, "chromium", None)
        executable_raw = str(getattr(chromium, "executable_path", "") or "")
        executable = Path(executable_raw) if executable_raw else None
        if executable is None or not executable.is_file():
            missing = executable_raw or "<unknown>"
            raise RuntimeError(
                "phase 2c render verification requires the Playwright Chromium "
                "browser bundle. Missing executable: "
                f"{missing}. Run: uv run python -m playwright install chromium. "
                "On fresh Linux hosts, also run: sudo $(command -v uv) run "
                "python -m playwright install-deps chromium."
            )
    finally:
        if pw_handle is not None:
            try:
                await pw_handle.stop()
            except Exception:
                logger.debug("phase 2c: failed to stop playwright readiness handle", exc_info=True)


def _instance_benchmark_or_none(instance: dict[str, Any]) -> str | None:
    values = [
        instance.get("benchmark"),
        instance.get("benchmark_name"),
        instance.get("benchmark_adapter"),
    ]
    try:
        return infer_benchmark_name(values)
    except ValueError:
        return None


def _resolve_benign_browser_context_auth(
    instance: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    return _auth_preflight._resolve_benign_browser_context_auth(instance)


def _auth_probe_failure_kind(reason: str) -> str:
    return _auth_preflight._auth_probe_failure_kind(reason)


def _agent_auth_type(instance: dict[str, Any]) -> str:
    return _auth_preflight._agent_auth_type(instance)


async def _run_reachability_check(
    *,
    browser: Any,
    render_semaphore: asyncio.Semaphore | None,
    task: dict[str, Any],
    seed: dict[str, Any],
    metadata: dict[str, Any],
    instance: dict[str, Any],
    render_outcome: RenderOutcome,
) -> ReachabilityOutcome:
    """Run the Option A reachability probe guarded by the same semaphore."""

    reachability_outcome_cls = ReachabilityOutcome
    derive_second_witness_fn = derive_second_witness
    render_signature_fn = render_signature
    verify_reachable_fn = verify_reachable
    auth_probe_failure_kind_fn = _auth_probe_failure_kind
    first_rendered_payload_fn = _first_rendered_payload
    phase4_exposure_inadmissible_reason_fn = _phase4_exposure_inadmissible_reason
    reachability_resource_for_task_fn = _reachability_resource_for_task
    required_url_token_fn = _required_url_token
    resolve_benign_browser_context_auth_fn = _resolve_benign_browser_context_auth
    selected_rendered_payload_fn = _selected_rendered_payload
    logger_obj = logger

    phase4_exposure_error = phase4_exposure_inadmissible_reason_fn(
        task.get("exposure_contract")
    )
    if phase4_exposure_error is not None:
        return reachability_outcome_cls.unreachable(
            kind=f"phase4_exposure_{phase4_exposure_error}",
            detail=(
                "exposure contract is seedable but not admissible for current "
                f"Phase 4 encounter semantics: {phase4_exposure_error}"
            ),
            url=str(instance.get("site_url") or ""),
        )

    benign_target_resource = reachability_resource_for_task_fn(task, metadata=metadata)
    site_url = str(instance.get("site_url") or "").strip()
    url_token = required_url_token_fn(task)
    payload_source = (
        selected_rendered_payload_fn(task)
        or first_rendered_payload_fn(seed)
        or (render_outcome.rendered_body_text if render_outcome is not None else None)
    )
    if url_token is not None:
        signature = url_token
        stable_signature = render_signature_fn(seed, metadata)
        normalized_url = url_token.casefold()
        normalized_stable = stable_signature.casefold() if stable_signature else ""
        if (
            stable_signature
            and normalized_stable != normalized_url
            and normalized_stable not in normalized_url
            and normalized_url not in normalized_stable
        ):
            second_witness = stable_signature
        else:
            second_witness = derive_second_witness_fn(payload_source, signature)
    else:
        signature = render_signature_fn(seed, metadata)
        second_witness = derive_second_witness_fn(payload_source, signature)
    browser_context_kwargs, auth_error = resolve_benign_browser_context_auth_fn(instance)
    if auth_error is not None:
        return reachability_outcome_cls.unreachable(
            kind=auth_probe_failure_kind_fn(auth_error),
            detail=auth_error,
            url=site_url,
            witnesses_missing=tuple(
                witness
                for witness in (signature, second_witness)
                if isinstance(witness, str) and witness
            ),
        )

    async def _do() -> ReachabilityOutcome:
        try:
            outcome = await verify_reachable_fn(
                browser=browser,
                benign_target_resource=benign_target_resource,
                instance_site_url=site_url,
                signature=signature,
                second_witness=second_witness,
                browser_context_kwargs=browser_context_kwargs,
            )
            return outcome
        except Exception as exc:  # pragma: no cover - defensive
            logger_obj.exception("phase 2c reachability probe crashed")
            return reachability_outcome_cls.unreachable(
                kind="probe_raised",
                detail=f"{exc.__class__.__name__}: {exc}",
                url=site_url,
            )

    if render_semaphore is None:
        return await _do()
    async with render_semaphore:
        return await _do()


async def _run_render_check(
    *,
    browser: Any,
    render_semaphore: asyncio.Semaphore | None,
    seed: dict[str, Any],
    metadata: dict[str, Any],
    instance: dict[str, Any],
) -> RenderOutcome:
    render_outcome_cls = RenderOutcome
    render_check_inputs_from_metadata_fn = _render_check_inputs_from_metadata
    render_signature_fn = render_signature
    render_signature_selection_fn = render_signature_selection
    verify_seed_renders_fn = verify_seed_renders
    auth_probe_failure_kind_fn = _auth_probe_failure_kind
    resolve_benign_browser_context_auth_fn = _resolve_benign_browser_context_auth
    logger_obj = logger

    selection = render_signature_selection_fn(seed, metadata)
    if isinstance(metadata, dict):
        urls, write_tokens, render_diagnostics = render_check_inputs_from_metadata_fn(
            metadata=metadata,
            selection=selection,
        )
    else:
        urls = []
        write_tokens = {}
        render_diagnostics = {}
    site_name = str(instance.get("site_name", "")).strip().lower()
    site_url = str(instance.get("site_url", "")).rstrip("/")
    signature = (
        selection.signature if selection is not None else render_signature_fn(seed, metadata)
    )

    browser_context_kwargs, auth_error = resolve_benign_browser_context_auth_fn(instance)
    if auth_error is not None:
        return render_outcome_cls.failed(
            kind=auth_probe_failure_kind_fn(auth_error),
            detail=auth_error,
            urls_tried=[],
            per_url_errors={},
        )

    async def _do() -> RenderOutcome:
        try:
            return await verify_seed_renders_fn(
                browser=browser,
                urls=urls,
                site_name=site_name,
                site_url=site_url,
                signature=signature,
                browser_context_kwargs=browser_context_kwargs,
                write_tokens=write_tokens or None,
                diagnostics=render_diagnostics or None,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger_obj.exception("phase 2c render check crashed")
            return render_outcome_cls.failed(
                kind="render_check_error",
                detail=f"render check raised {exc.__class__.__name__}: {exc}",
                urls_tried=urls,
                per_url_errors={},
            )

    if render_semaphore is None:
        return await _do()
    async with render_semaphore:
        return await _do()


__all__ = [
    "_BROWSER_PROBE_CAP",
    "_PROBE_LAUNCH_ARGS",
    "_RENDER_UNVERIFIED_KIND",
    "_RENDER_UNVERIFIED_RETRY_DELAY_S",
    "_SKIP_RENDER_CHECK_ENV",
    "ReachabilityOutcome",
    "RenderOutcome",
    "_agent_auth_type",
    "_auth_probe_failure_kind",
    "_ensure_playwright_chromium_ready",
    "_instance_benchmark_or_none",
    "_patchable_globals",
    "_resolve_benign_browser_context_auth",
    "_restore_patchable_globals",
    "_run_reachability_check",
    "_run_render_check",
]
