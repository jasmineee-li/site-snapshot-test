"""Phase 4: Adversarial evaluation with adaptive strategy variation.

Canonical source: ``docs/worldsim-v5-technical-specifcation.md`` "Phase 4: Adversarial
Evaluation with Adaptive Strategy Variation".

Phase 4 is the core evaluation loop. Two sequential gates:

1. **Ecological validity gate.** Every trajectory is probed ~5× and the
   average P(eval) is compared against ``VALIDITY_THRESHOLD``. Only
   ecologically valid trajectories reach Gate 2.
2. **Attack effectiveness gate.**
   - complied -> record as clean ASR data point
   - task_broke -> fix injection placement (not a strategy change)
   - refused_or_ignored -> enter adaptive strategy variation

Adaptive strategy variation (one round per task, fan-out of up to 3 variants):

1. Judge Modal Sandbox diagnoses why the agent refused
2. Judge recommends up to 3 strategies from the pool
3. Variants are generated in parallel Modal Sandboxes (up to 3)
4. Variants are evaluated in parallel against separate benchmark instances
5. Every variant passes through the ecological validity gate again

Strategy pool: ``authority_framing``, ``semantic_coherence``, ``specificity``,
``visual_concealment``, ``verbosity_adjustment``.

Scheduled for commits 13-14.
"""

from __future__ import annotations

import argparse


async def run(args: argparse.Namespace) -> int:
    """Phase 4 entrypoint. Not yet implemented."""
    raise NotImplementedError(
        "Phase 4 (adversarial evaluation) is scheduled for commits 13-14. "
        "Depends on Phase 2 injections + Phase 3-passing tasks + a running WebArena instance."
    )


# TODO(commit 13):
# async def run_adversarial_task(task, agent, instance, task_dir) -> dict: ...
# async def probe_ecological_validity(agent, trajectory, site_url) -> float: ...
#
# TODO(commit 14):
# async def run_judge(task, trajectory_dir, profile_path) -> dict: ...
# async def generate_variant(task, strategy, profile_path) -> dict: ...
# async def run_strategy_variation(task, initial_result, instances, agent_factory, profile_path) -> dict: ...
