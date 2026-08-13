"""Phase 2c source-data preflight filtering."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from warp_taskgen.benchmark_capabilities import infer_benchmark_name
from warp_taskgen.phase_2.phase_2c import auth_preflight as _auth_preflight
from warp_taskgen.phase_2.phase_2c.policy import FeasibilityPolicyCatalog

logger = logging.getLogger(__name__)

_PREFLIGHT_AUTH_REFRESH_LOCKS: dict[str, asyncio.Lock] = {}
_agent_auth_type = _auth_preflight._agent_auth_type
_preflight_request_context_options = _auth_preflight._preflight_request_context_options
_PATCHABLE_GLOBAL_NAMES = (
    "_PREFLIGHT_AUTH_REFRESH_LOCKS",
    "_agent_auth_type",
    "_preflight_request_context_options",
    "infer_benchmark_name",
    "logger",
)


def _patchable_globals() -> dict[str, Any]:
    return {name: globals()[name] for name in _PATCHABLE_GLOBAL_NAMES}


def _restore_patchable_globals(values: dict[str, Any]) -> None:
    for name, value in values.items():
        globals()[name] = value


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


async def _run_preflight_and_filter_raw(
    raw: list[dict[str, Any]],
    *,
    instances_by_site: dict[str, list[dict[str, Any]]],
    benchmark_root: Path | None = None,
    feasibility_policy_catalog: FeasibilityPolicyCatalog | None = None,
) -> list[dict[str, Any]]:
    """Probe every task and mutate ``raw`` in place to drop quarantined tasks.

    Uses a lazy Playwright ``APIRequestContext`` per (site, auth-context)
    pair — shares Playwright's TLS/proxy setup with the render_check path
    without importing httpx. Returns the list of dropped records with
    ``source_data_issue`` metadata attached; the caller routes this list
    to the ``adversarial_tasks.dropped_source_data.json`` sidecar.

    Skips silently if Playwright is not importable (development without
    Playwright installed); mirrors the render_check skip-envvar shape.
    """
    from warp_taskgen.phases.phase_2c_preflight import (
        auth_self_test_path,
        preflight_benign_targets,
        self_test_preflight_auth,
    )

    try:
        from playwright.async_api import async_playwright
    except ImportError:
        logger.info(
            "phase 2c preflight: playwright not importable; skipping source-"
            "data quarantine for this run."
        )
        return []

    pw_handle = await async_playwright().start()
    contexts_created: list[Any] = []

    async def _factory(context_options: dict[str, Any] | None) -> Any:
        kwargs: dict[str, Any] = {}
        if isinstance(context_options, dict):
            for key in ("storage_state", "extra_http_headers", "http_credentials"):
                value = context_options.get(key)
                if value:
                    kwargs[key] = value
        ctx = await pw_handle.request.new_context(**kwargs)
        contexts_created.append(ctx)
        return ctx

    async def _self_test_context_options(
        *,
        site: str,
        instance: dict[str, Any],
        context_options: dict[str, Any],
    ) -> Any:
        benchmark = _instance_benchmark_or_none(instance)
        if benchmark is None:
            return None
        ctx = await _factory(context_options)
        try:
            kwargs: dict[str, Any] = {
                "request_context": ctx,
                "site": site,
                "site_url": str(instance.get("site_url") or ""),
                "benchmark": benchmark,
            }
            if feasibility_policy_catalog is not None:
                kwargs["feasibility_policy_catalog"] = feasibility_policy_catalog
            return await self_test_preflight_auth(
                **kwargs,
            )
        finally:
            try:
                await ctx.dispose()
            except Exception:
                logger.debug(
                    "phase 2c preflight: auth self-test context dispose failed",
                    exc_info=True,
                )

    async def _ensure_live_storage_state_options(
        *,
        site: str,
        instance: dict[str, Any],
        context_options: dict[str, Any],
        skip_reason: str | None,
    ) -> tuple[dict[str, Any], str | None]:
        benchmark = _instance_benchmark_or_none(instance)
        if benchmark is None:
            return context_options, skip_reason
        auth_path_kwargs: dict[str, Any] = {"benchmark": benchmark}
        if feasibility_policy_catalog is not None:
            auth_path_kwargs["feasibility_policy_catalog"] = feasibility_policy_catalog
        if (
            auth_self_test_path(site, **auth_path_kwargs) is None
            or _agent_auth_type(instance) != "storage_state"
        ):
            return context_options, skip_reason

        if skip_reason is None:
            classification = await _self_test_context_options(
                site=site,
                instance=instance,
                context_options=context_options,
            )
            if classification is None or classification.kind == "reachable":
                return context_options, None
            if classification.kind not in {"login_redirect", "auth_missing"}:
                raise RuntimeError(
                    "phase 2c preflight: auth self-test for "
                    f"{site} at {instance.get('site_url')!r} was inconclusive: "
                    f"{classification.kind} ({classification.detail})"
                )
            logger.warning(
                "phase 2c preflight: storage_state auth for %s at %s is stale: %s; "
                "reacquiring via Phase 0d",
                site,
                instance.get("site_url"),
                classification.detail,
            )
        else:
            logger.warning(
                "phase 2c preflight: storage_state auth for %s at %s is unusable: %s; "
                "reacquiring via Phase 0d",
                site,
                instance.get("site_url"),
                skip_reason,
            )

        lock_key = f"{site}:{instance.get('site_url')}"
        lock = _PREFLIGHT_AUTH_REFRESH_LOCKS.setdefault(lock_key, asyncio.Lock())
        async with lock:
            from warp_taskgen.phases.phase_0d_auth_bootstrap import (
                AuthBootstrapError,
                reacquire_storage_state,
            )

            try:
                refreshed_path = await reacquire_storage_state(
                    site_name=site,
                    instance=instance,
                    benchmark_root=benchmark_root,
                )
            except AuthBootstrapError as exc:
                reason = (
                    "storage_state refresh failed; source-data quarantine skipped "
                    f"for this instance ({exc})"
                )
                logger.warning(
                    "phase 2c preflight: %s at %s: %s",
                    site,
                    instance.get("site_url"),
                    reason,
                )
                return {}, reason
            instance["storage_state_path"] = str(refreshed_path)
            agent_auth = instance.get("agent_auth")
            if isinstance(agent_auth, dict):
                updated_auth = dict(agent_auth)
                storage_state = updated_auth.get("storage_state")
                if isinstance(storage_state, dict):
                    updated_storage_state = dict(storage_state)
                else:
                    updated_storage_state = {}
                updated_storage_state["path"] = str(refreshed_path)
                updated_auth["storage_state"] = updated_storage_state
                instance["agent_auth"] = updated_auth

        refreshed_options, refreshed_skip = _preflight_request_context_options(
            instance,
            benchmark_root=benchmark_root,
        )
        if refreshed_skip is not None:
            reason = (
                "reacquired storage_state is unusable; source-data quarantine skipped "
                f"for this instance ({refreshed_skip})"
            )
            logger.warning(
                "phase 2c preflight: %s at %s: %s",
                site,
                instance.get("site_url"),
                reason,
            )
            return {}, reason
        refreshed_classification = await _self_test_context_options(
            site=site,
            instance=instance,
            context_options=refreshed_options,
        )
        if refreshed_classification is None or refreshed_classification.kind == "reachable":
            logger.info(
                "phase 2c preflight: refreshed storage_state auth for %s at %s",
                site,
                instance.get("site_url"),
            )
            return refreshed_options, None
        if refreshed_classification.kind in {"login_redirect", "auth_missing"}:
            reason = (
                "storage_state refresh did not authenticate; source-data quarantine "
                f"skipped for this instance ({refreshed_classification.kind}: "
                f"{refreshed_classification.detail})"
            )
            logger.warning(
                "phase 2c preflight: %s at %s: %s",
                site,
                instance.get("site_url"),
                reason,
            )
            return {}, reason
        raise RuntimeError(
            "phase 2c preflight: storage_state refresh for "
            f"{site} at {instance.get('site_url')!r} did not authenticate: "
            f"{refreshed_classification.kind} ({refreshed_classification.detail})"
        )

    try:
        preflight_instances_by_site: dict[str, list[dict[str, Any]]] = {}
        for site, site_instances in instances_by_site.items():
            resolved_instances: list[dict[str, Any]] = []
            for instance in site_instances:
                resolved = dict(instance)
                context_options, skip_reason = _preflight_request_context_options(
                    instance,
                    benchmark_root=benchmark_root,
                )
                context_options, skip_reason = await _ensure_live_storage_state_options(
                    site=site,
                    instance=resolved,
                    context_options=context_options,
                    skip_reason=skip_reason,
                )
                if resolved.get("storage_state_path"):
                    instance["storage_state_path"] = resolved["storage_state_path"]
                if isinstance(resolved.get("agent_auth"), dict):
                    instance["agent_auth"] = resolved["agent_auth"]
                if skip_reason is not None:
                    resolved["preflight_auth_skip_reason"] = skip_reason
                else:
                    resolved["preflight_request_context"] = context_options
                resolved_instances.append(resolved)
            preflight_instances_by_site[site] = resolved_instances
        preflight_kwargs: dict[str, Any] = {
            "instances_by_site": preflight_instances_by_site,
            "request_context_factory": _factory,
        }
        if feasibility_policy_catalog is not None:
            preflight_kwargs["feasibility_policy_catalog"] = feasibility_policy_catalog
        keep, dropped = await preflight_benign_targets(raw, **preflight_kwargs)
    finally:
        for ctx in contexts_created:
            try:
                await ctx.dispose()
            except Exception:
                logger.debug(
                    "phase 2c preflight: request context dispose failed",
                    exc_info=True,
                )
        try:
            await pw_handle.stop()
        except Exception:
            logger.debug("phase 2c preflight: playwright stop failed", exc_info=True)

    if dropped:
        kept_ids = {id(t) for t in keep}
        raw[:] = [t for t in raw if id(t) in kept_ids]
    return dropped
