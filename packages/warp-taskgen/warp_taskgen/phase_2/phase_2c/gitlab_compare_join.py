"""Phase 2c join for the concrete GitLab comparison task."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from warp_taskgen.phase_1.gitlab_compare_decide import GitLabAttemptBinding
from warp_taskgen.phase_1.gitlab_compare_decide_binding import (
    bind_gitlab_compare_decide_attempt,
)


def join_gitlab_compare_decide_seed(
    task: Mapping[str, Any],
    metadata: dict[str, Any],
    *,
    phase: str = "phase2c",
) -> GitLabAttemptBinding | None:
    """Bind and retain comparison evidence when a task declares that contract.

    Ordinary tasks return ``None`` and retain their existing Phase 2c path.
    Comparison tasks receive only the strict per-call binding; aggregate
    metadata is never promoted as an action or scoring anchor.
    """
    contract = task.get("comparison_contract")
    if not isinstance(contract, Mapping):
        return None
    if str(task.get("site") or contract.get("site") or "").strip().lower() != "gitlab":
        return None
    binding = bind_gitlab_compare_decide_attempt(task, metadata, phase=phase)
    metadata["gitlab_compare_decide_binding"] = binding.as_mapping()
    return binding


__all__ = ["join_gitlab_compare_decide_seed"]
