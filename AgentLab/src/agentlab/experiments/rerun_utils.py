"""
Utilities for flexible experiment re-running.

Supports:
- Re-judging without re-running agent
- Reusing HTML from previous runs
- Running with different attacker models
- Multiple seeds/variations
"""

import json
import gzip
import pickle
import logging
from pathlib import Path
from typing import Optional, List, Literal
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RerunConfig:
    """Configuration for selective re-running."""

    # What to rerun
    mode: Literal["full", "judge_only", "attacker_only", "html_only"] = "full"

    # Source for HTML reuse
    resume_html_from: Optional[str] = None  # Path to previous results dir

    # Attacker configuration
    attacker_model: Optional[str] = None  # Override attacker model
    attacker_temperature: float = 0.7

    # Seed configuration
    seeds: List[int] = None  # List of seeds to run (default: [0])
    variations_per_seed: int = 2  # Number of adversarial variations per seed

    # Filtering
    behavior_ids: Optional[List[str]] = None  # Only run these behaviors
    test_modes: Optional[List[str]] = None  # Only run these modes (benign, adversarial)
    skip_successful: bool = False  # Skip experiments that already succeeded

    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [0]


def load_experiment_result(exp_dir: Path) -> dict:
    """Load experiment result from directory."""
    exp_dir = Path(exp_dir)

    result = {
        "exp_dir": str(exp_dir),
        "exp_args": None,
        "judge_result": None,
        "summary_info": None,
        "steps": [],
    }

    # Load exp_args
    exp_args_path = exp_dir / "exp_args.pkl"
    if exp_args_path.exists():
        with open(exp_args_path, "rb") as f:
            result["exp_args"] = pickle.load(f)

    # Load judge result
    judge_path = exp_dir / "judge_result.json"
    if judge_path.exists():
        with open(judge_path) as f:
            result["judge_result"] = json.load(f)

    # Load summary info
    summary_path = exp_dir / "summary_info.json"
    if summary_path.exists():
        with open(summary_path) as f:
            result["summary_info"] = json.load(f)

    # Load steps
    step_files = sorted(exp_dir.glob("step_*.pkl.gz"))
    for step_file in step_files:
        with gzip.open(step_file, "rb") as f:
            result["steps"].append(pickle.load(f))

    return result


def rejudge_experiment(exp_dir: Path, judge_config: Optional[dict] = None) -> dict:
    """
    Re-run judging on an existing experiment without re-running the agent.

    Args:
        exp_dir: Path to experiment directory
        judge_config: Optional new judge configuration to use

    Returns:
        New judge result
    """
    from agentlab.benchmarks.redteam_judge import JudgeRunner

    exp_dir = Path(exp_dir)
    result = load_experiment_result(exp_dir)

    if not result["exp_args"]:
        raise ValueError(f"No exp_args.pkl found in {exp_dir}")

    if not result["steps"]:
        raise ValueError(f"No step files found in {exp_dir}")

    exp_args = result["exp_args"]
    env_args = exp_args.env_args

    # Override judge config if provided
    if judge_config:
        env_args.judging = judge_config

    # Build episode info from steps
    episode_info = result["steps"]

    # Create judge runner and evaluate
    runner = JudgeRunner(env_args)
    judge_result = runner.evaluate(exp_dir, episode_info)

    # Save new result
    runner.save_result(exp_dir, judge_result)

    logger.info(f"Re-judged {exp_dir}: success={judge_result.get('overall_success')}")
    return judge_result


def find_experiments_to_rerun(
    study_dir: Path,
    config: RerunConfig,
) -> List[Path]:
    """
    Find experiment directories that need re-running based on config.

    Args:
        study_dir: Path to study results directory
        config: RerunConfig specifying what to rerun

    Returns:
        List of experiment directory paths
    """
    study_dir = Path(study_dir)
    experiments = []

    # Find all experiment directories
    for behavior_dir in study_dir.iterdir():
        if not behavior_dir.is_dir():
            continue
        if behavior_dir.name.startswith("."):
            continue

        # Check behavior filter
        behavior_id = behavior_dir.name.split("_")[-1] if "_" in behavior_dir.name else behavior_dir.name

        if config.behavior_ids and behavior_id not in config.behavior_ids:
            continue

        # Find variant directories
        for variant_dir in behavior_dir.iterdir():
            if not variant_dir.is_dir():
                continue
            if not (variant_dir / "exp_args.pkl").exists():
                continue

            # Check test mode filter
            variant_name = variant_dir.name
            if config.test_modes:
                if "benign" in variant_name and "benign" not in config.test_modes:
                    continue
                if "adversarial" in variant_name and "adversarial" not in config.test_modes:
                    continue

            # Check if already successful
            if config.skip_successful:
                judge_path = variant_dir / "judge_result.json"
                if judge_path.exists():
                    with open(judge_path) as f:
                        judge = json.load(f)
                    if judge.get("overall_success"):
                        continue

            experiments.append(variant_dir)

    return experiments


def rejudge_study(
    study_dir: Path,
    config: RerunConfig,
    new_judge_config: Optional[dict] = None,
) -> dict:
    """
    Re-judge all experiments in a study.

    Args:
        study_dir: Path to study results directory
        config: RerunConfig for filtering
        new_judge_config: Optional new judge configuration

    Returns:
        Summary of re-judging results
    """
    experiments = find_experiments_to_rerun(study_dir, config)

    results = {
        "total": len(experiments),
        "success": 0,
        "failed": 0,
        "errors": [],
    }

    for exp_dir in experiments:
        try:
            judge_result = rejudge_experiment(exp_dir, new_judge_config)
            if judge_result.get("overall_success"):
                results["success"] += 1
            else:
                results["failed"] += 1
        except Exception as e:
            results["errors"].append({"exp_dir": str(exp_dir), "error": str(e)})
            logger.error(f"Error re-judging {exp_dir}: {e}")

    return results


def copy_html_for_rerun(
    source_study_dir: Path,
    target_study_dir: Path,
    behavior_ids: Optional[List[str]] = None,
) -> int:
    """
    Copy HTML files from a previous run to a new run directory.

    Args:
        source_study_dir: Path to source study with HTML files
        target_study_dir: Path to target study directory
        behavior_ids: Optional list of behavior IDs to copy (None = all)

    Returns:
        Number of HTML files copied
    """
    import shutil

    source_dir = Path(source_study_dir)
    target_dir = Path(target_study_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    copied = 0

    for behavior_dir in source_dir.iterdir():
        if not behavior_dir.is_dir():
            continue

        behavior_id = behavior_dir.name.split("_")[-1]
        if behavior_ids and behavior_id not in behavior_ids:
            continue

        for variant_dir in behavior_dir.iterdir():
            if not variant_dir.is_dir():
                continue

            # Copy HTML files
            for html_file in variant_dir.glob("*.html"):
                target_variant = target_dir / behavior_dir.name / variant_dir.name
                target_variant.mkdir(parents=True, exist_ok=True)
                shutil.copy2(html_file, target_variant / html_file.name)
                copied += 1

            # Copy injections.json if exists
            injections = variant_dir.parent / "injections.json"
            if injections.exists():
                target_behavior = target_dir / behavior_dir.name
                target_behavior.mkdir(parents=True, exist_ok=True)
                shutil.copy2(injections, target_behavior / "injections.json")

    logger.info(f"Copied {copied} HTML files from {source_dir} to {target_dir}")
    return copied


def get_study_summary(study_dir: Path) -> dict:
    """Get summary of a study's experiments and their status."""
    study_dir = Path(study_dir)

    summary = {
        "total_experiments": 0,
        "completed": 0,
        "failed": 0,
        "no_judge": 0,
        "success": 0,
        "by_behavior": {},
        "by_mode": {"benign": 0, "adversarial": 0},
    }

    for behavior_dir in study_dir.iterdir():
        if not behavior_dir.is_dir() or behavior_dir.name.startswith("."):
            continue

        behavior_id = behavior_dir.name.split("_")[-1]
        summary["by_behavior"][behavior_id] = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "no_steps": 0,
        }

        for variant_dir in behavior_dir.iterdir():
            if not variant_dir.is_dir():
                continue
            if not (variant_dir / "exp_args.pkl").exists():
                continue

            summary["total_experiments"] += 1
            behavior_stats = summary["by_behavior"][behavior_id]
            behavior_stats["total"] += 1

            # Track by mode
            if "benign" in variant_dir.name:
                summary["by_mode"]["benign"] += 1
            else:
                summary["by_mode"]["adversarial"] += 1

            # Check status
            summary_path = variant_dir / "summary_info.json"
            judge_path = variant_dir / "judge_result.json"

            if summary_path.exists():
                with open(summary_path) as f:
                    info = json.load(f)
                if info.get("n_steps", 0) == 0:
                    summary["failed"] += 1
                    behavior_stats["no_steps"] += 1
                else:
                    summary["completed"] += 1
            else:
                summary["failed"] += 1

            if judge_path.exists():
                with open(judge_path) as f:
                    judge = json.load(f)
                if judge.get("overall_success"):
                    summary["success"] += 1
                    behavior_stats["success"] += 1
                elif judge.get("message") == "No judges configured":
                    summary["no_judge"] += 1
                else:
                    behavior_stats["failed"] += 1
            else:
                summary["no_judge"] += 1

    return summary


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment rerun utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Summary command
    summary_parser = subparsers.add_parser("summary", help="Get study summary")
    summary_parser.add_argument("study_dir", help="Path to study directory")

    # Rejudge command
    rejudge_parser = subparsers.add_parser("rejudge", help="Re-judge experiments")
    rejudge_parser.add_argument("study_dir", help="Path to study directory")
    rejudge_parser.add_argument("--behaviors", nargs="+", help="Only these behaviors")
    rejudge_parser.add_argument("--modes", nargs="+", choices=["benign", "adversarial"])
    rejudge_parser.add_argument("--skip-successful", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.command == "summary":
        summary = get_study_summary(Path(args.study_dir))
        print(json.dumps(summary, indent=2))

    elif args.command == "rejudge":
        config = RerunConfig(
            mode="judge_only",
            behavior_ids=args.behaviors,
            test_modes=args.modes,
            skip_successful=args.skip_successful,
        )
        results = rejudge_study(Path(args.study_dir), config)
        print(json.dumps(results, indent=2))
