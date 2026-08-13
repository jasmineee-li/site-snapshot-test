"""Conservative per-instance reset caching for Phase 3/4 task execution."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

from warp_taskgen.agent_config import RUNTIME_METADATA_KEY

_MUTATING_HTTP_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def callable_accepts_keyword(callable_obj: Callable[..., Any], keyword: str) -> bool:
    """Return True when ``callable_obj`` can accept ``keyword=...``.

    This keeps orchestration code compatible with monkeypatched test doubles
    and other wrapper callables that may lag behind the production signature.
    """
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False

    params = sig.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return True

    param = params.get(keyword)
    return param is not None and param.kind is not inspect.Parameter.POSITIONAL_ONLY


def _binding_keys_from_instance_payload(payload: dict[str, Any]) -> tuple[str, ...]:
    keys: set[str] = set()
    reset_endpoint = payload.get("reset_endpoint")
    site_url = payload.get("site_url")
    if isinstance(reset_endpoint, str) and reset_endpoint.strip():
        keys.add(f"reset:{reset_endpoint.strip()}")
    elif isinstance(site_url, str) and site_url.strip():
        keys.add(f"site:{site_url.strip()}")
    return tuple(sorted(keys))


def _task_binding_keys(
    task: dict[str, Any],
    *,
    extra_bindings: list[dict[str, Any]] | None = None,
) -> tuple[str, ...]:
    runtime = task.get(RUNTIME_METADATA_KEY, {})
    keys: set[str] = set()

    bound_instances = runtime.get("bound_instances")
    if isinstance(bound_instances, dict):
        for payload in bound_instances.values():
            if not isinstance(payload, dict):
                continue
            keys.update(_binding_keys_from_instance_payload(payload))

    if not keys:
        reset_endpoints = runtime.get("reset_endpoints")
        if isinstance(reset_endpoints, list):
            for endpoint in reset_endpoints:
                if isinstance(endpoint, str) and endpoint.strip():
                    keys.add(f"reset:{endpoint.strip()}")

    for binding in extra_bindings or []:
        if isinstance(binding, dict):
            keys.update(_binding_keys_from_instance_payload(binding))

    return tuple(sorted(keys))


def _task_seeded(task: dict[str, Any]) -> bool:
    for field_name in ("data_seed", "adversarial_data_seed"):
        seed = task.get(field_name)
        if not isinstance(seed, dict):
            continue
        mechanism = str(seed.get("mechanism", "")).strip().lower()
        if mechanism and mechanism != "none":
            return True
    return False


def _task_hosts(task: dict[str, Any]) -> set[str]:
    runtime = task.get(RUNTIME_METADATA_KEY, {})
    hosts: set[str] = set()

    bound_instances = runtime.get("bound_instances")
    if isinstance(bound_instances, dict):
        for payload in bound_instances.values():
            if not isinstance(payload, dict):
                continue
            site_url = payload.get("site_url")
            if not isinstance(site_url, str):
                continue
            host = urlsplit(site_url).hostname
            if host:
                hosts.add(host.lower())
    return hosts


def result_likely_mutated_state(task: dict[str, Any], agent_result: Any) -> bool:
    """Return True when a task likely mutated benchmark state."""
    if _task_seeded(task):
        return True

    network_trace = getattr(agent_result, "network_trace", None)
    if not isinstance(network_trace, list):
        return False

    bound_hosts = _task_hosts(task)
    for entry in network_trace:
        if not isinstance(entry, dict):
            continue
        method = str(entry.get("method", "")).strip().upper()
        if method not in _MUTATING_HTTP_METHODS:
            continue
        url = entry.get("url")
        if not isinstance(url, str) or not url.strip():
            return True
        host = urlsplit(url).hostname
        if not bound_hosts or (host and host.lower() in bound_hosts):
            return True
    return False


class TaskResetCache:
    """Track whether bound instances need a reset before the next task."""

    def __init__(self) -> None:
        self._needs_reset: dict[str, bool] = {}

    def should_reset(
        self,
        task: dict[str, Any],
        *,
        extra_bindings: list[dict[str, Any]] | None = None,
    ) -> bool:
        keys = _task_binding_keys(task, extra_bindings=extra_bindings)
        if not keys:
            return True
        return any(self._needs_reset.get(key, True) for key in keys)

    def mark_clean(
        self,
        task: dict[str, Any],
        *,
        extra_bindings: list[dict[str, Any]] | None = None,
    ) -> None:
        for key in _task_binding_keys(task, extra_bindings=extra_bindings):
            self._needs_reset[key] = False

    def mark_dirty(
        self,
        task: dict[str, Any],
        *,
        extra_bindings: list[dict[str, Any]] | None = None,
    ) -> None:
        for key in _task_binding_keys(task, extra_bindings=extra_bindings):
            self._needs_reset[key] = True
