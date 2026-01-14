"""
Test script for redteam benchmark with unified injection pipeline.

This script demonstrates how to run a redteam benchmark evaluation with AgentLab,
where an attacker agent attempts to induce harmful behavior in a target agent.

The unified pipeline supports:
- A/B testing with benign (control) vs adversarial variants
- Multiple adversarial variations per behavior
- Channel-aware injection visibility (axtree, visual, html)
- Reference-based UI consistency across pages

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
    AGENT_45_OPUS,
)

# =============================================================================
# Configuration
# =============================================================================

# Target agent to test
target_agent_config = AGENT_45_OPUS

# Filter mode: "all", "benign_only", or "adversarial_only"
# - "all": Run both benign control and adversarial variants (for A/B testing)
# - "benign_only": Run only benign control (baseline without injections)
# - "adversarial_only": Run only adversarial variants (faster iteration)
FILTER_MODE = "all"

# =============================================================================
# Benchmark Setup
# =============================================================================

benchmark = RedteamBenchmark(
    # Benchmark file with behavior definitions
    # benchmark_file="AgentLab/tests/new-data/salesforce_behavior_test_with_config.json",
    benchmark_file="new-data/20260113_test.json",
    # Models
    world_sim_model="anthropic/claude-opus-4.5",  # For page generation
    attacker_model="x-ai/grok-4",  # For injection content generation
    # Attack settings
    adversarial_mode="multi_turn",  # "single_turn" or "multi_turn"
    # === Unified Injection Pipeline Settings ===
    # Number of adversarial variants per behavior (benign is always generated)
    # Total tasks = n_behaviors * (1 benign + n_adversarial_variants)
    n_adversarial_variants=2,
    # Attack channels - which observation modes to target
    # Options: ["axtree"], ["visual"], ["html"], or combinations like ["axtree", "visual"]
    # - "axtree": Agent observes via accessibility tree (default, most common)
    # - "visual": Agent observes via screenshots
    # - "html": Agent observes raw HTML
    attack_channels=["axtree"],
    # Trajectory observation (captures real site screenshots for realistic generation)
    # Set to True to use existing screenshots from resume_from_dir
    use_trajectory=True,
    # Skip link validation for faster iteration during development
    skip_link_validation=True,
    # Resume from existing run - copies HTML files AND trajectory screenshots
    # This avoids re-running expensive trajectory observation
    # resume_from_dir="results/2026-01-09_10-41-09_genericagent-anthropic-claude-opus-4-5-on-enriched-behaviors/2026-01-09_10-41-09_GenericAgent-anthropic_claude-opus-4.5_on_enriched-behaviors.salesforce-unauthorized-admin-creation.adversarial_v0_0",
)

# =============================================================================
# Filter variants based on FILTER_MODE
# =============================================================================

if FILTER_MODE == "benign_only":
    benchmark.env_args_list = [
        args for args in benchmark.env_args_list if args.test_mode == "benign"
    ]
    print(f"Filtered to benign only: {len(benchmark.env_args_list)} tasks")
elif FILTER_MODE == "adversarial_only":
    benchmark.env_args_list = [
        args for args in benchmark.env_args_list if args.test_mode == "adversarial"
    ]
    print(f"Filtered to adversarial only: {len(benchmark.env_args_list)} tasks")
else:
    print(f"Running all variants: {len(benchmark.env_args_list)} tasks")

# Log task breakdown
benign_count = sum(1 for args in benchmark.env_args_list if args.test_mode == "benign")
adversarial_count = sum(1 for args in benchmark.env_args_list if args.test_mode == "adversarial")
print(f"Task breakdown: {benign_count} benign, {adversarial_count} adversarial")

# =============================================================================
# Study Setup
# =============================================================================

study = Study(
    agent_args=[target_agent_config],
    benchmark=benchmark,
    comment=f"Redteam test - {FILTER_MODE} - Unified injection pipeline",
)

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("STARTING REDTEAM TEST (Unified Injection Pipeline)")
    logger.info("=" * 80)
    logger.info(f"Benchmark: {benchmark.name}")
    logger.info(f"Filter mode: {FILTER_MODE}")
    logger.info(f"Number of tasks: {len(benchmark.env_args_list)}")
    logger.info(f"  - Benign: {benign_count}")
    logger.info(f"  - Adversarial: {adversarial_count}")
    logger.info(f"World sim model: {benchmark.world_sim_model}")
    logger.info(f"Attacker model: {benchmark.attacker_model}")
    logger.info(f"Attack channels: {benchmark.attack_channels}")
    logger.info(f"N adversarial variants: {benchmark.n_adversarial_variants}")
    logger.info("=" * 80)

    # List tasks to be run
    for i, env_args in enumerate(benchmark.env_args_list):
        logger.info(f"  Task {i+1}: {env_args.task_name} (mode={env_args.test_mode})")

    try:
        logger.info("Starting study.run()...")
        study.run(
            n_jobs=1,
            parallel_backend="sequential",
        )
        logger.info("study.run() completed successfully!")
    except Exception as e:
        logger.error(f"Error during study.run(): {e}", exc_info=True)
        raise
