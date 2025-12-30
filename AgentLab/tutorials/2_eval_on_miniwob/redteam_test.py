"""
Test script for redteam benchmark with adversarial agent support.

This script demonstrates how to run a redteam benchmark evaluation with AgentLab,
where an attacker agent attempts to induce harmful behavior in a target agent.

"""

from pathlib import Path
import sys
import logging

# Ensure local AgentLab package is importable
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from agentlab.benchmarks.redteam import RedteamBenchmark
from agentlab.experiments.study import Study
from agentlab.agents.generic_agent.agent_configs import (
    AGENT_4o_VISION,
    AGENT_GPT5_MINI,
    AGENT_LLAMA3_70B,
    AGENT_4o_MINI,
    AGENT_37_SONNET,
)

# target_agent_config = AGENT_GPT5_MINI
target_agent_config = AGENT_4o_MINI

benchmark = RedteamBenchmark(
    benchmark_file="/Users/jasminexli/grayswan/site-snapshot-test/AgentLab/tests/new-data/1228_behaviors_single.json",
    world_sim_model="anthropic/claude-opus-4.5",
    attacker_model="x-ai/grok-4",
    adversarial_mode="single_turn",
    n_injection_variations=1,
    use_trajectory=True,
    # Resume from existing run (uncomment to reuse generated HTML files):
    resume_from_dir="/Users/jasminexli/grayswan/site-snapshot-test/results/2025-12-29_10-44-04_genericagent-gpt-4o-mini-2024-07-18-on-enriched-behaviors/2025-12-29_10-44-04_GenericAgent-gpt-4o-mini-2024-07-18_on_enriched-behaviors.modify-cicd-workflow.seed0_0",
    skip_link_validation=True,
)
print("created redteambenchmark")
# Create study - attacker agent is automatically configured from benchmark
study = Study(
    agent_args=[target_agent_config],
    benchmark=benchmark,
    comment="Redteam adversarial test - Attacker vs target",
)

if __name__ == "__main__":
    # Run sequentially for debugging
    # Each experiment will have attacker + target conversation
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("STARTING REDTEAM TEST")
    logger.info(f"Benchmark: {benchmark.name}")
    logger.info(f"Number of tasks: {len(benchmark.env_args_list)}")
    logger.info(f"World sim model: {benchmark.world_sim_model}")
    logger.info(f"Attacker model: {benchmark.attacker_model}")
    # logger.info(f"Target model: {benchmark.target_model}")
    logger.info("=" * 80)

    try:
        logger.info("About to call study.run()...")
        study.run(
            n_jobs=1,
            parallel_backend="sequential",
        )
        logger.info("study.run() completed successfully!")
    except Exception as e:
        logger.error(f"Error during study.run(): {e}", exc_info=True)
        raise
