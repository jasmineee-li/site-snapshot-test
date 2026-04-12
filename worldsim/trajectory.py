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

from pathlib import Path


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
            sandbox_files[f"/workspace/trajectory/{rel}"] = str(f)
