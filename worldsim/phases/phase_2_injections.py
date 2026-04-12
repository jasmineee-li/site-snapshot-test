"""Phase 2: Injection generation.

Canonical source: ``docs/worldsim-v5-technical-specifcation.md`` "Phase 2: Injection Generation".

One Modal Sandbox per site. Inputs: benign tasks from Phase 1 and
``BENCHMARK_PROFILE_{site}.json`` from Phase 0c. Output:
``adversarial_tasks.json`` per site.

Injection point selection criteria (all four required for a candidate to be
retained):

1. Task relevance — will the agent encounter this content?
2. Attacker realism — require "medium" or higher (from the profile).
3. Concealment — can the injection blend with surrounding content?
4. Harmful action — what could the agent be tricked into doing?

Scheduled for commit 11.
"""

from __future__ import annotations

import argparse


async def run(args: argparse.Namespace) -> int:
    """Phase 2 entrypoint. Not yet implemented."""
    raise NotImplementedError(
        "Phase 2 (injection generation) is scheduled for commit 11. "
        "Depends on Phase 0c profiles and Phase 1 Mode A benign tasks."
    )
