"""Runner registry.

Maps runner names to factory functions that produce (task_runner, agent_factory)
pairs compatible with ``agent_config.run_tasks_by_site``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Lazy registry: values are import paths, resolved on first access.
_RUNNER_REGISTRY: dict[str, str] = {
    "browser_use": "worldsim.runners.browser_use",
    "agentlab": "worldsim.runners.agentlab",
}


def get_runner_module(name: str) -> Any:
    """Import and return the runner module for *name*.

    Raises ``ValueError`` if the name is not registered and ``ImportError``
    if the module cannot be loaded (e.g., missing optional dependencies).
    """
    if name not in _RUNNER_REGISTRY:
        raise ValueError(
            f"Unknown runner {name!r}. Available: {sorted(_RUNNER_REGISTRY)}"
        )
    import importlib

    module_path = _RUNNER_REGISTRY[name]
    try:
        return importlib.import_module(module_path)
    except ImportError as exc:
        if name == "agentlab":
            raise ImportError(
                f"AgentLab runner requires optional dependencies. "
                f"Install with: pip install worldsim[comparison]\n"
                f"Original error: {exc}"
            ) from exc
        raise


def available_runners() -> list[str]:
    """Return the list of registered runner names."""
    return sorted(_RUNNER_REGISTRY)
