"""
Test script for redteam benchmark.

This script demonstrates how to run a redteam benchmark evaluation with AgentLab.
"""

from pathlib import Path
import sys

# Ensure local AgentLab package is importable
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from agentlab.benchmarks.redteam import RedteamBenchmark
from agentlab.experiments.study import Study
from agentlab.agents.generic_agent.agent_configs import AGENT_4o_VISION, AGENT_GPT5_MINI

# Configure agent for redteam testing  
# You can customize obs flags here if needed
agent_config = AGENT_4o_VISION

# Create benchmark
benchmark = RedteamBenchmark(
    benchmark_file="/Users/jasminexli/grayswan/site-snapshot-test/cua-website-sim/benchmarks/v1.json",
    llm_model="gpt-4o",
    n_injection_variations=1,  # Just one variation for testing
)

# Create study
study = Study(
    agent_args=[agent_config],
    benchmark=benchmark,
    comment="Redteam test - GPT-4o agent",
)

if __name__ == "__main__":
    # Run sequentially for debugging
    study.run(
        n_jobs=1,
        parallel_backend="sequential",
    )