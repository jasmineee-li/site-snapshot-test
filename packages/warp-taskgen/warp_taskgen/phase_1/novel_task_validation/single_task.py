"""Single-task validation export backed by the parity implementation."""

from __future__ import annotations

from warp_taskgen.phase_1.novel_task_validation._impl import validate_generated_novel_task

__all__ = ["validate_generated_novel_task"]
