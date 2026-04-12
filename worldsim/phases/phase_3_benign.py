"""Phase 3: Benign validation.

Canonical source: ``docs/worldsim-v5-technical-specifcation.md`` "Phase 3: Benign Validation".

For each task, seed the environment with benign data, run the target agent
via Browser Use, and score the trajectory with the task's reward function.
Failed tasks enter a diagnosis-fix loop driven by a Modal Sandbox.

Failure taxonomy:

- reward function bug -> fix reward, re-run sanity check
- data seed issue -> fix seed, re-run sanity check
- impossible task -> remove
- task too hard -> keep but flag
- agent limitation -> keep (informative baseline data)

Scheduled for commit 12. Blocked end-to-end on the user standing up at
least one WebArena instance.
"""

from __future__ import annotations

import argparse


async def run(args: argparse.Namespace) -> int:
    """Phase 3 entrypoint. Not yet implemented."""
    raise NotImplementedError(
        "Phase 3 (benign validation) is scheduled for commit 12. "
        "Depends on a pre-running WebArena instance — see v5 spec 'Prerequisites'."
    )


# TODO(commit 12):
# async def run_task(task, agent, instance, task_dir): ...
# async def diagnose_failure(task, trajectory_dir, profile_path) -> dict: ...
# async def fix_loop(task, failure_diagnosis) -> dict: ...
