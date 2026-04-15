"""Trajectory artifact loader for Phase 3/4 diagnosis sandboxes.

Canonical source: ``docs/worldsim-v5-technical-specifcation.md`` "Trajectory Artifacts and
Diagnosis Sandboxes".

Per-task trajectory layout::

    task_e1/
        history.json        # action sequence + reasoning (10-100KB)
        result.json         # pass/fail + reward message (<1KB)
        screenshots/        # one PNG per agent step
        conversations/      # raw model conversation logs

All files are added to the sandbox file map, but Claude Code decides what
to read. The action log is always read first; screenshots are read selectively
if the text log is ambiguous. This avoids context explosion on long
trajectories.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from worldsim.atomic_io import write_json_atomic
from worldsim.browser_use_agent import AgentResult


def save_result(
    task_dir: Path,
    task: dict[str, Any],
    result: AgentResult,
    passed: bool,
    message: str,
    **extra: Any,
) -> None:
    """Write ``result.json`` to a task trajectory directory.

    This is the trajectory artifact that diagnosis and judge sandboxes expect
    to find at ``/workspace/trajectory/result.json``.

    Extra keyword arguments (e.g. ``outcome``, ``ecologically_valid``) are
    merged into the result dict so downstream resume can reconstruct the
    full classification without re-running evaluation.

    When ``_resume_fingerprint`` is present, resume logic only reuses the
    sentinel for matching execution context.
    """
    task_dir = Path(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "task_id": task.get("id", "unknown"),
        "passed": passed,
        "message": message,
        "status": result.status,
        "elapsed": result.elapsed,
        "steps": result.steps,
        "is_done": result.is_done,
        "final_result": result.final_result,
        "errors": result.errors,
        **extra,
    }
    # Atomic write: tmpfile + os.replace prevents truncated result.json on crash,
    # which would cause load_completed_results to misidentify the task as complete.
    target = task_dir / "result.json"
    write_json_atomic(
        target,
        data,
        failpoint_base="trajectory.save_result",
    )


def load_trajectory_into_sandbox(
    trajectory_dir: Path,
    sandbox_files: dict[str, str],
) -> None:
    """Add every file in a trajectory directory to a sandbox file map in-place.

    Args:
        trajectory_dir: Local directory containing the trajectory artifacts.
        sandbox_files: Mutable dict mapping sandbox-remote paths to local paths,
            as consumed by :func:`worldsim.modal_sandbox.run_claude_in_sandbox`.
            This function mutates the dict in place.
    """
    trajectory_dir = Path(trajectory_dir)
    for f in trajectory_dir.rglob("*"):
        if f.is_file():
            rel = f.relative_to(trajectory_dir)
            # Intentionally stage all trajectory artifacts, including network.har.
            # The live writer in browser_use_agent redacts trace entries before
            # persisting both network_trace.json and network.har, so this broad
            # staging is not a current-data leak in the normal pipeline path.
            sandbox_files[f"/workspace/trajectory/{rel}"] = str(f)
