"""Phase 1: Task generation.

Canonical source: ``docs/worldsim-v5-technical-specifcation.md`` "Phase 1: Task Generation".

Two modes:

- **Mode A — Wrap existing benchmark tasks.** Reads task definitions from
  the benchmark codebase (paths discovered in Phase 0a's
  ``task_definition_paths``) and produces benign task bundles.
- **Mode B — Generate new tasks on existing environments.** Stretch goal per
  the v5 spec; implemented only after Phases 3–4 are validated end-to-end.
  One Modal Sandbox per site, with ``BENCHMARK_PROFILE_{site}.json`` as context.

Scheduled for commit 10 (Mode A). Mode B is deferred.
"""

from __future__ import annotations

import argparse


async def run(args: argparse.Namespace) -> int:
    """Phase 1 entrypoint. Not yet implemented."""
    raise NotImplementedError(
        "Phase 1 Mode A is scheduled for commit 10. "
        "Phase 1 Mode B (new task generation) is a stretch goal per the v5 spec."
    )
