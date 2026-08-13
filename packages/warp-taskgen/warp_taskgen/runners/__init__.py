"""Browser-agent runner registry."""

from __future__ import annotations

import importlib
from typing import Any

from warp_taskgen.agent_runtime import RUNNER_AGENTLAB, RUNNER_BROWSER_USE

_RUNNER_REGISTRY: dict[str, str] = {
    RUNNER_BROWSER_USE: "warp_taskgen.runners.browser_use",
    RUNNER_AGENTLAB: "warp_taskgen.runners.agentlab",
}


def normalize_runner_name(name: object) -> str:
    return str(name or RUNNER_BROWSER_USE).strip().lower().replace("-", "_")


def available_runners() -> list[str]:
    return sorted(_RUNNER_REGISTRY)


def get_runner_module(name: object) -> Any:
    runner = normalize_runner_name(name)
    if runner not in _RUNNER_REGISTRY:
        raise ValueError(f"unknown runner {name!r}; available={available_runners()}")
    try:
        return importlib.import_module(_RUNNER_REGISTRY[runner])
    except ImportError as exc:
        if runner == RUNNER_AGENTLAB:
            raise ImportError(
                "AgentLab runner requires optional comparison dependencies. "
                "Use an isolated environment for AgentLab/BrowserGym because "
                "current AgentLab releases depend on openai<2 while Browser Use "
                "0.12.6 depends on openai==2.16.0."
            ) from exc
        raise
