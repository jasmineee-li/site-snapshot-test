"""WebArena Verified benchmark adapter.

Placeholder adapter for the WebArena Verified benchmark. When using the
Browser Use runner, evaluation flows through the existing rewards.py and
placeholders.py code. When using the AgentLab runner, BrowserGym handles
evaluation natively (WebArena is a default BrowserGym benchmark).

This adapter provides the placeholder-to-site mapping and task name
resolution needed by the runner dispatch layer.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_PLACEHOLDER_TO_SITE: dict[str, str] = {
    "__SHOPPING__": "shopping",
    "__SHOPPING_ADMIN__": "shopping_admin",
    "__GITLAB__": "gitlab",
    "__REDDIT__": "reddit",
    "__WIKIPEDIA__": "wikipedia",
    "__MAP__": "map",
}


class WebArenaVerifiedAdapter:
    """Adapter for WebArena Verified benchmark.

    For the Browser Use runner, the existing rewards.py evaluator handles
    evaluation. For the AgentLab runner, BrowserGym's native WebArena
    support provides evaluation through the gym reward signal.
    """

    def __init__(self, benchmark_name: str) -> None:
        self.benchmark_name = benchmark_name

    @property
    def placeholder_to_site(self) -> dict[str, str]:
        return dict(_PLACEHOLDER_TO_SITE)

    @property
    def browsergym_prefix(self) -> str:
        """BrowserGym task name prefix for this benchmark."""
        return "webarena"

    def get_browsergym_task_name(self, task: dict[str, Any]) -> str:
        """Return the BrowserGym task name for a WorldSim task dict."""
        task_id = task.get("id")
        if task_id is None:
            raise ValueError("Task has no 'id' field")
        return f"webarena.{task_id}"
