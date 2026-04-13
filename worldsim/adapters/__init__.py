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
