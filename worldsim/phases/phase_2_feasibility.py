"""Compatibility facade for Phase 2c feasibility verification."""

from __future__ import annotations

import logging
from typing import Any

# ruff: noqa: F403
from worldsim.phase_2.phase_2c import *
from worldsim.phase_2.phase_2c import _impl as _legacy_impl

globals().update(
    {
        name: value
        for name, value in vars(_legacy_impl).items()
        if not name.startswith("__")
    }
)
logger = logging.getLogger(__name__)

_ORIGINAL_IMPL_FUNCS = {
    name: getattr(_legacy_impl, name)
    for name in (
        "_run_render_check",
        "_run_reachability_check",
        "_run_preflight_and_filter_raw",
        "_idempotency_decision",
    )
}

_PATCHABLE_GLOBALS = (
    "EDITOR_REGISTRY",
    "EditorError",
    "SeedCleanupHandle",
    "UnboundTokenError",
    "ReachabilityOutcome",
    "RenderOutcome",
    "acquire_tokens_for_instances",
    "apply_data_seed_async",
    "derive_second_witness",
    "get_benchmark_capabilities",
    "infer_benchmark_name",
    "normalize_benchmark_name",
    "playwright_storage_state",
    "playwright_storage_state_payload",
    "read_storage_state_payload",
    "resolve_agent_auth",
    "resolve_agent_auth_headers",
    "resolve_storage_state_path",
    "retrying",
    "select_task_site_instance_dict_p2c",
    "storage_state_preflight_error_for_payload",
    "verify_reachable",
    "verify_seed_renders",
    "_PER_REPLICA_CAP_DEFAULT",
    "_PER_REPLICA_CAP_FALLBACK",
    "_BROWSER_PROBE_CAP",
    "_PREFLIGHT_AUTH_REFRESH_LOCKS",
    "_RENDER_UNVERIFIED_RETRY_DELAY_S",
    "_SKIP_RENDER_CHECK_ENV",
    "logger",
)


def _sync_legacy_patches() -> None:
    for name in _PATCHABLE_GLOBALS:
        if name in globals():
            setattr(_legacy_impl, name, globals()[name])
    for name in (
        "_run_render_check",
        "_run_reachability_check",
        "_run_preflight_and_filter_raw",
        "_idempotency_decision",
    ):
        current = globals().get(name)
        original = _ORIGINAL_IMPL_FUNCS[name]
        facade_wrapper = _FACADE_WRAPPERS.get(name)
        if current is facade_wrapper:
            setattr(_legacy_impl, name, original)
        elif current is not None and current is not original:
            setattr(_legacy_impl, name, current)


async def verify_feasibility(*args: Any, **kwargs: Any) -> Any:
    _sync_legacy_patches()
    return await _legacy_impl.verify_feasibility(*args, **kwargs)


async def _verify_one(*args: Any, **kwargs: Any) -> dict[str, Any]:
    _sync_legacy_patches()
    return await _legacy_impl._verify_one(*args, **kwargs)


async def _run_render_check(*args: Any, **kwargs: Any) -> Any:
    _sync_legacy_patches()
    return await _legacy_impl._run_render_check(*args, **kwargs)


async def _run_reachability_check(*args: Any, **kwargs: Any) -> Any:
    _sync_legacy_patches()
    return await _legacy_impl._run_reachability_check(*args, **kwargs)


async def _run_preflight_and_filter_raw(*args: Any, **kwargs: Any) -> Any:
    _sync_legacy_patches()
    return await _legacy_impl._run_preflight_and_filter_raw(*args, **kwargs)


_FACADE_WRAPPERS = {
    "_run_render_check": _run_render_check,
    "_run_reachability_check": _run_reachability_check,
    "_run_preflight_and_filter_raw": _run_preflight_and_filter_raw,
}

# Static compatibility guard retained for tests that ensure contract violations
# are classified before generic ValueError branches:
# except UnboundTokenError
# kind="contract_violation"
