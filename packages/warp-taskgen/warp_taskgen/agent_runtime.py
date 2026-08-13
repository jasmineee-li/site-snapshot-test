"""Browser-agent runtime contracts.

These contracts are intentionally independent of Browser Use. Browser Use is
the default implementation today, but Phase 4 and the worker pool should depend
on the lifecycle/result surface rather than on a specific harness module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

RUNNER_BROWSER_USE = "browser_use"
RUNNER_AGENTLAB = "agentlab"


@dataclass(frozen=True)
class AgentRunRequest:
    """Canonical request shape for future runner-neutral execution paths.

    The current Phase 4 call path still expands these fields into keyword
    arguments for compatibility with ``BrowserUseAgent.run``. New runtimes
    should converge on this object instead of growing more runner-specific
    kwargs at the orchestration layer.
    """

    task: str
    server_url: str
    task_dir: Path
    start_urls: list[str] = field(default_factory=list)
    site_prompt: str | None = None
    auth_mechanism: dict[str, Any] | None = None
    benchmark_root: Path | None = None
    task_site: str | None = None
    payload_text: str | None = None
    payload_witnesses: list[str | dict[str, Any]] = field(default_factory=list)
    pvpo_cdp_url: str | None = None
    instance_id: str | None = None
    url_origin_rewrites: dict[str, str] | None = None


@dataclass
class AgentResult:
    """Summary of one browser-agent run."""

    elapsed: float
    steps: int
    is_done: bool
    final_result: str | None
    status: str = "success"
    errors: list[str] = field(default_factory=list)
    network_trace: list[dict[str, Any]] = field(default_factory=list)


class AgentRunner(Protocol):
    """Lifecycle every browser-agent implementation in the worker pool satisfies."""

    async def setup(self, server_url: str) -> None: ...

    async def run(
        self,
        task: str,
        server_url: str,
        task_dir: Path,
        *,
        start_urls: list[str] | None = None,
        site_prompt: str | None = None,
        auth_mechanism: dict[str, Any] | None = None,
        benchmark_root: Path | None = None,
        task_site: str | None = None,
        payload_text: str | None = None,
        payload_witnesses: list[str | dict[str, Any]] | None = None,
        pvpo_cdp_url: str | None = None,
        instance_id: str | None = None,
        url_origin_rewrites: dict[str, str] | None = None,
    ) -> AgentResult: ...

    async def teardown(self) -> None: ...
