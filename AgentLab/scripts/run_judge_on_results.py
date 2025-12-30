#!/usr/bin/env python3
"""
Post-process existing redteam study results with judges.

Usage:
    python scripts/run_judge_on_results.py <results_dir> <behavior_config_json>

Example:
    python scripts/run_judge_on_results.py \
        results/2025-12-30_10-25-33_genericagent-gpt-4o-mini-2024-07-18-on-enriched-behaviors \
        tests/new-data/1228_behaviors_single.json
"""

import sys
import json
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from agentlab.benchmarks.redteam_judge import RedteamJudge
from agentlab.experiments.exp_utils import RESULTS_DIR

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_experiment_data(exp_dir: Path) -> dict:
    """Load agent actions and conversation from experiment directory."""

    # Try to load summary_info.json
    summary_path = exp_dir / "summary_info.json"
    if summary_path.exists():
        with open(summary_path, "r") as f:
            summary = json.load(f)

        # Extract conversation and actions
        steps = summary.get("steps", [])

        conversation = []
        agent_actions = []

        for step in steps:
            # Add agent observation/thought
            if "observation" in step:
                conversation.append({"role": "observation", "content": str(step["observation"])})

            # Add agent action
            if "action" in step:
                action_data = step["action"]
                agent_actions.append(
                    {
                        "action": action_data,
                        "step": step.get("step_idx", 0),
                    }
                )
                conversation.append({"role": "agent", "content": f"Action: {action_data}"})

        return {
            "conversation": conversation,
            "agent_actions": agent_actions,
        }

    # Fallback: try to reconstruct from log files
    logger.warning(f"summary_info.json not found in {exp_dir}, trying log.json")

    log_path = exp_dir.parent / "log.json"
    if log_path.exists():
        with open(log_path, "r") as f:
            log_data = json.load(f)

        # Extract from log format
        conversation = log_data.get("conversation", [])
        agent_actions = log_data.get("actions", [])

        return {
            "conversation": conversation,
            "agent_actions": agent_actions,
        }

    raise FileNotFoundError(f"Could not find experiment data in {exp_dir}")


def load_behavior_config(behavior_file: Path, task_id: str) -> dict:
    """Load behavior configuration from JSON file."""

    with open(behavior_file, "r") as f:
        behaviors = json.load(f)

    # Find matching behavior
    for behavior in behaviors:
        if behavior.get("task_id") == task_id:
            return behavior

    raise ValueError(f"Behavior {task_id} not found in {behavior_file}")


def run_judge_on_experiment(exp_dir: Path, behavior_config: dict) -> dict:
    """Run judge on a single experiment."""

    logger.info(f"\nProcessing experiment: {exp_dir.name}")

    # Extract task_id from directory name
    # Format: YYYY-MM-DD_HH-MM-SS_AgentName_on_benchmark.task-id.seed0_0
    parts = exp_dir.name.split(".")
    if len(parts) < 3:  # Need at least timestamp, benchmark, task-id, seed
        logger.warning(f"Could not extract task_id from {exp_dir.name}")
        return None

    # task_id is the second-to-last part (before .seed0_0)
    task_id = parts[
        -2
    ]  # e.g., "modify-cicd-workflow" from "...enriched-behaviors.modify-cicd-workflow.seed0_0"

    logger.info(f"Task ID: {task_id}")

    # Load experiment data
    try:
        exp_data = load_experiment_data(exp_dir)
    except FileNotFoundError as e:
        logger.error(f"Could not load experiment data: {e}")
        return None

    # Find variation directory
    variation_dir = exp_dir / "variation_0"
    if not variation_dir.exists():
        logger.error(f"Variation directory not found: {variation_dir}")
        return None

    # Initialize judge
    judging_config = behavior_config.get("judging", {})
    if not judging_config:
        logger.warning(f"No judging config found for {task_id}")
        return None

    behavior_context = {
        "doc": behavior_config.get("doc", ""),
        "criteria": behavior_config.get("criteria", []),
        "short_description": behavior_config.get("short_description", ""),
        "task_id": task_id,
    }

    judge = RedteamJudge(judging_config, behavior_context)

    # Run evaluation
    result = judge.evaluate(
        agent_actions=exp_data["agent_actions"],
        variation_dir=variation_dir,
        conversation=exp_data["conversation"],
    )

    # Convert to dict
    result_dict = judge.to_dict(result)

    # Save judge results
    output_path = exp_dir / "judge_results.json"
    with open(output_path, "w") as f:
        json.dump(result_dict, f, indent=2)

    logger.info(f"✓ Saved judge results to {output_path}")
    logger.info(f"  Overall success: {result.overall_success}")
    logger.info(f"  Overall score: {result.overall_score:.2f}")

    for ir in result.individual_results:
        logger.info(f"  - {ir.judge_id}: success={ir.success}, score={ir.score:.2f}")

    return result_dict


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    behavior_file = Path(sys.argv[2])

    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        sys.exit(1)

    if not behavior_file.exists():
        logger.error(f"Behavior config file not found: {behavior_file}")
        sys.exit(1)

    # Load all behaviors
    with open(behavior_file, "r") as f:
        behaviors = json.load(f)

    # Create behavior lookup by task_id (try both "id" and "task_id" fields)
    behavior_lookup = {}
    for b in behaviors:
        task_id = b.get("task_id") or b.get("id")
        if task_id:
            behavior_lookup[task_id] = b

    # Find all experiment directories
    exp_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("20")]

    logger.info(f"Found {len(exp_dirs)} experiment(s) in {results_dir}")

    results = []
    for exp_dir in exp_dirs:
        # Extract task_id from directory name
        parts = exp_dir.name.split(".")
        if len(parts) >= 3:  # Need at least timestamp, benchmark, task-id, seed
            # task_id is the second-to-last part
            task_id = parts[-2]

            # Get behavior config - try both with "id" and "task_id" keys
            behavior_config = behavior_lookup.get(task_id)
            if not behavior_config:
                # Try looking up by "id" field instead
                for b in behaviors:
                    if b.get("id") == task_id or b.get("task_id") == task_id:
                        behavior_config = b
                        break

            if not behavior_config:
                logger.warning(f"No behavior config found for task_id: {task_id}")
                continue

            # Run judge
            result = run_judge_on_experiment(exp_dir, behavior_config)
            if result:
                results.append(
                    {
                        "exp_dir": str(exp_dir),
                        "task_id": task_id,
                        "result": result,
                    }
                )

    # Save summary
    summary_path = results_dir / "judge_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✓ Processed {len(results)} experiment(s)")
    logger.info(f"✓ Summary saved to {summary_path}")

    # Print summary statistics
    if results:
        successes = sum(1 for r in results if r["result"]["overall_success"])
        avg_score = sum(r["result"]["overall_score"] for r in results) / len(results)

        logger.info(f"\nSummary:")
        logger.info(f"  Total experiments: {len(results)}")
        logger.info(f"  Successful attacks: {successes} ({successes/len(results)*100:.1f}%)")
        logger.info(f"  Average score: {avg_score:.2f}")
    else:
        logger.warning("\nNo experiments were successfully processed.")
        logger.warning("Check that:")
        logger.warning("  1. Experiment directories contain summary_info.json or log.json")
        logger.warning("  2. Task IDs in directory names match behavior config 'id' field")
        logger.warning("  3. Behavior configs have 'judging' sections")


if __name__ == "__main__":
    main()
