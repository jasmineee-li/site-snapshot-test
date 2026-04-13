"""Benchmark adapter registry.

Maps benchmark names to adapter modules that know how to load tasks,
configure evaluators, and set up baseline attacks for each benchmark.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Lazy registry: values are (module_path, class_name) tuples.
_ADAPTER_REGISTRY: dict[str, tuple[str, str]] = {
    "webarena-verified": ("worldsim.adapters.webarena_verified", "WebArenaVerifiedAdapter"),
    "WebArena Verified": ("worldsim.adapters.webarena_verified", "WebArenaVerifiedAdapter"),
    "doomarena": ("worldsim.adapters.doomarena", "DoomArenaAdapter"),
    "stwebagentbench": ("worldsim.adapters.stwebagentbench", "STWebAgentBenchAdapter"),
    "wasp": ("worldsim.adapters.wasp", "WASPAdapter"),
    "safearena": ("worldsim.adapters.safearena", "SafeArenaAdapter"),
    "safearena-harm": ("worldsim.adapters.safearena", "SafeArenaAdapter"),
    "safearena-safe": ("worldsim.adapters.safearena", "SafeArenaAdapter"),
}


def get_adapter(benchmark_name: str) -> Any:
    """Instantiate and return the adapter for *benchmark_name*.

    Raises ``ValueError`` for unknown benchmarks and ``ImportError``
    if the adapter module cannot be loaded.
    """
    if benchmark_name not in _ADAPTER_REGISTRY:
        raise ValueError(
            f"Unknown benchmark {benchmark_name!r}. "
            f"Available: {sorted(set(n for n in _ADAPTER_REGISTRY))}"
        )
    import importlib

    module_path, class_name = _ADAPTER_REGISTRY[benchmark_name]
    module = importlib.import_module(module_path)
    adapter_cls = getattr(module, class_name)
    return adapter_cls(benchmark_name=benchmark_name)


def available_benchmarks() -> list[str]:
    """Return deduplicated list of registered benchmark names."""
    return sorted(set(_ADAPTER_REGISTRY))


def browsergym_task_name_for(adapter: Any, task: dict[str, Any]) -> str:
    """Resolve the BrowserGym task name for a wrapped task.

    Adapters can implement ``get_browsergym_task_name(task)`` directly or
    expose a simple ``browsergym_prefix`` property plus a stable task id.
    """
    explicit = task.get("agentlab_task_name")
    if explicit:
        return str(explicit)

    resolver = getattr(adapter, "get_browsergym_task_name", None)
    if callable(resolver):
        return str(resolver(task))

    prefix = getattr(adapter, "browsergym_prefix", None)
    task_id = task.get("id", task.get("task_id"))
    if prefix and task_id not in (None, ""):
        return f"{prefix}.{task_id}"

    adapter_name = type(adapter).__name__
    raise ValueError(
        f"{adapter_name} does not define BrowserGym task-name resolution and "
        "task does not provide agentlab_task_name"
    )
