"""
Test script for redteam benchmark with adversarial agent support.

This script demonstrates how to run a redteam benchmark evaluation with AgentLab,
where an attacker agent attempts to induce harmful behavior in a target agent.

The benchmark automatically configures the attacker agent based on the behavior
and target specified in the benchmark JSON file.
"""

from pathlib import Path
import sys
import logging

# Ensure local AgentLab package is importable
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from agentlab.benchmarks.redteam import RedteamBenchmark
from agentlab.experiments.study import Study
from agentlab.agents.generic_agent.agent_configs import AGENT_4o_VISION, AGENT_GPT5_MINI

# Target agent being evaluated (will be attacked by redteam agent)
# You can customize obs flags here if needed
target_agent_config = AGENT_GPT5_MINI

# Create benchmark with attacker agent enabled (default)
benchmark = RedteamBenchmark(
    benchmark_file="/Users/jasminexli/grayswan/site-snapshot-test/AgentLab/tests/v5.json",
    llm_model="gpt-5",
    world_sim_model="gpt-5",
    n_injection_variations=1,
    use_trajectory=False,
)
print("created redteambenchmark")
# Create study - attacker agent is automatically configured from benchmark
study = Study(
    agent_args=[target_agent_config],
    benchmark=benchmark,
    comment="Redteam adversarial test - Attacker vs GPT-4o-vision target",
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
