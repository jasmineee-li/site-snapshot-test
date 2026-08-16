#!/usr/bin/env python3
"""
Flexible experiment re-running for redteam benchmarks.

Usage Examples:

1. Re-judge only (no agent re-run):
   python rerun_experiments.py rejudge results/2026-01-17_study/

2. Rerun with same HTML but new attacker:
   python rerun_experiments.py rerun \
       --resume-html results/2026-01-17_study/ \
       --attacker-model anthropic/claude-3.5-sonnet \
       --behaviors empty-box-listing slack-phishing

3. Run with multiple seeds:
   python rerun_experiments.py rerun \
       --seeds 0 1 2 \
       --variations 3

4. Get study summary:
   python rerun_experiments.py summary results/2026-01-17_study/

5. Find and rerun failed experiments:
   python rerun_experiments.py rerun-failed results/2026-01-17_study/
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add AgentLab to path
sys.path.insert(0, str(Path(__file__).parent / "AgentLab" / "src"))

from agentlab.experiments.rerun_utils import (
    RerunConfig,
    copy_html_for_rerun,
    get_study_summary,
    rejudge_study,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def cmd_summary(args):
    """Show summary of a study."""
    summary = get_study_summary(Path(args.study_dir))

    print("\n" + "=" * 60)
    print("STUDY SUMMARY")
    print("=" * 60)
    print(f"Total experiments: {summary['total_experiments']}")
    print(f"Completed: {summary['completed']}")
    print(f"Failed/No steps: {summary['failed']}")
    print(f"No judge configured: {summary['no_judge']}")
    print(f"Successful attacks: {summary['success']}")
    print("\nBy mode:")
    print(f"  Benign: {summary['by_mode']['benign']}")
    print(f"  Adversarial: {summary['by_mode']['adversarial']}")
    print("\nBy behavior:")
    for bid, stats in summary["by_behavior"].items():
        status = "✅" if stats["success"] > 0 else "❌"
        print(f"  {status} {bid}: {stats['success']}/{stats['total']} success")


def cmd_rejudge(args):
    """Re-judge experiments without re-running agent."""
    config = RerunConfig(
        mode="judge_only",
        behavior_ids=args.behaviors,
        test_modes=args.modes,
        skip_successful=args.skip_successful,
    )

    print(f"\nRe-judging experiments in {args.study_dir}...")
    results = rejudge_study(Path(args.study_dir), config)

    print("\n" + "=" * 60)
    print("RE-JUDGING RESULTS")
    print("=" * 60)
    print(f"Total re-judged: {results['total']}")
    print(f"Now successful: {results['success']}")
    print(f"Still failed: {results['failed']}")
    if results["errors"]:
        print(f"Errors: {len(results['errors'])}")
        for err in results["errors"][:5]:
            print(f"  - {err['exp_dir']}: {err['error']}")


def cmd_rerun(args):
    """Rerun experiments with flexible configuration."""
    from agentlab.agents.generic_agent.agent_configs import AGENT_SONNET_3_7
    from agentlab.benchmarks.redteam import RedteamBenchmark
    from agentlab.experiments.study import Study

    # Create benchmark with configuration
    benchmark_kwargs = {
        "benchmark_file": args.benchmark or "new-data/behaviors_enriched.json",
        "n_adversarial_variations": args.variations,
    }

    if args.attacker_model:
        benchmark_kwargs["attacker_model"] = args.attacker_model

    if args.behaviors:
        benchmark_kwargs["behavior_ids"] = args.behaviors

    if args.resume_html:
        benchmark_kwargs["resume_from_dir"] = args.resume_html
        print(f"Reusing HTML from: {args.resume_html}")

    benchmark = RedteamBenchmark(**benchmark_kwargs)

    # Configure agent
    agent = AGENT_SONNET_3_7

    # Create study
    study_name = f"rerun_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    study = Study(agent, benchmark, study_name=study_name)

    # Handle multiple seeds
    if args.seeds and len(args.seeds) > 1:
        print(f"Running with seeds: {args.seeds}")
        # TODO: Implement multi-seed support
        # For now, just use first seed
        print("Note: Multi-seed support coming soon. Using first seed only.")

    # Run
    print(f"\nStarting study: {study_name}")
    print(f"Behaviors: {args.behaviors or 'all'}")
    print(f"Variations per behavior: {args.variations}")
    print(f"Attacker model: {args.attacker_model or 'default'}")

    if args.dry_run:
        print("\n[DRY RUN] Would run the following experiments:")
        for exp_args in study.exp_args_list[:10]:
            print(f"  - {exp_args.env_args.task_name}")
        if len(study.exp_args_list) > 10:
            print(f"  ... and {len(study.exp_args_list) - 10} more")
        return

    study.run(n_jobs=args.n_jobs)
    print(f"\nStudy complete. Results in: {study.dir}")


def cmd_rerun_failed(args):
    """Rerun only failed experiments from a previous study."""
    from agentlab.experiments.study import Study

    study = Study.load(args.study_dir)
    incomplete = study.find_incomplete(include_errors=True)

    print(f"\nFound {len(incomplete)} incomplete/failed experiments:")
    for exp in incomplete[:10]:
        print(f"  - {exp.env_args.task_name}")
    if len(incomplete) > 10:
        print(f"  ... and {len(incomplete) - 10} more")

    if args.dry_run:
        print("\n[DRY RUN] Would rerun the above experiments")
        return

    print("\nRelaunching...")
    study.run(n_relaunch=args.max_relaunch)


def cmd_copy_html(args):
    """Copy HTML files from one study to another for reuse."""
    copied = copy_html_for_rerun(
        Path(args.source),
        Path(args.target),
        behavior_ids=args.behaviors,
    )
    print(f"Copied {copied} HTML files")


def main():
    parser = argparse.ArgumentParser(
        description="Flexible experiment re-running for redteam benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Summary command
    p_summary = subparsers.add_parser("summary", help="Show study summary")
    p_summary.add_argument("study_dir", help="Path to study directory")

    # Rejudge command
    p_rejudge = subparsers.add_parser("rejudge", help="Re-judge without re-running agent")
    p_rejudge.add_argument("study_dir", help="Path to study directory")
    p_rejudge.add_argument("--behaviors", "-b", nargs="+", help="Only these behavior IDs")
    p_rejudge.add_argument("--modes", "-m", nargs="+", choices=["benign", "adversarial"])
    p_rejudge.add_argument("--skip-successful", "-s", action="store_true")

    # Rerun command
    p_rerun = subparsers.add_parser("rerun", help="Rerun experiments with new config")
    p_rerun.add_argument("--benchmark", help="Path to behaviors JSON file")
    p_rerun.add_argument("--behaviors", "-b", nargs="+", help="Only these behavior IDs")
    p_rerun.add_argument("--resume-html", help="Reuse HTML from this study directory")
    p_rerun.add_argument("--attacker-model", help="Override attacker model")
    p_rerun.add_argument("--seeds", type=int, nargs="+", default=[0], help="Seeds to run")
    p_rerun.add_argument(
        "--variations", type=int, default=2, help="Adversarial variations per behavior"
    )
    p_rerun.add_argument("--n-jobs", type=int, default=4, help="Parallel jobs")
    p_rerun.add_argument("--dry-run", action="store_true", help="Show what would run")

    # Rerun-failed command
    p_failed = subparsers.add_parser("rerun-failed", help="Rerun failed experiments")
    p_failed.add_argument("study_dir", help="Path to study directory")
    p_failed.add_argument("--max-relaunch", type=int, default=3, help="Max relaunch attempts")
    p_failed.add_argument("--dry-run", action="store_true")

    # Copy HTML command
    p_copy = subparsers.add_parser("copy-html", help="Copy HTML for reuse")
    p_copy.add_argument("source", help="Source study directory")
    p_copy.add_argument("target", help="Target directory")
    p_copy.add_argument("--behaviors", "-b", nargs="+", help="Only these behavior IDs")

    args = parser.parse_args()

    if args.command == "summary":
        cmd_summary(args)
    elif args.command == "rejudge":
        cmd_rejudge(args)
    elif args.command == "rerun":
        cmd_rerun(args)
    elif args.command == "rerun-failed":
        cmd_rerun_failed(args)
    elif args.command == "copy-html":
        cmd_copy_html(args)


if __name__ == "__main__":
    main()
