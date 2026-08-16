"""End-to-end overnight pipeline: DoomArena tasks → trajectory awareness judges.

For each site, runs DoomArena popup attack tasks with glm-5:thinking, then
runs trajectory awareness judges on the resulting transcripts.

Usage:
    python -m eval_awareness_experiments.run_doomarena_pipeline \
        --sites reddit shopping shopping_admin gitlab \
        --tasks-per-site 20 \
        --judge-model anthropic/claude-opus-4.6:thinking

    # Dry run (just print what would happen):
    python -m eval_awareness_experiments.run_doomarena_pipeline --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent

DOOMARENA_RUNNER = "eval_awareness_experiments.run_doomarena_glm5_popup"

DEFAULT_SITES = ["reddit", "shopping", "shopping_admin", "gitlab"]
ALL_ONLINE_SITES = ["reddit", "shopping", "shopping_admin", "gitlab"]


def run_doomarena_site(
    site: str,
    max_tasks: int,
    agent_model: str,
    max_steps: int,
    report_port: int,
) -> Path | None:
    """Run DoomArena tasks for a single site. Returns the study dir or None on failure."""
    logger.info(f"=== Running DoomArena on {site} ({max_tasks} tasks, model={agent_model}) ===")

    cmd = [
        sys.executable, "-m", DOOMARENA_RUNNER,
        "--single-site", site,
        "--max-tasks", str(max_tasks),
        "--online-sites", *ALL_ONLINE_SITES,
        "--model-name", agent_model,
        "--max-steps", str(max_steps),
        "--report-port", str(report_port),
        "--shuffle", "42",
    ]

    env = os.environ.copy()
    env.setdefault("DOOMARENA_WEBARENA_BASE_URL", "http://localhost")

    logger.info(f"  cmd: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=str(REPO_ROOT))

    # Print stdout/stderr for debugging
    if result.stdout:
        for line in result.stdout.strip().split("\n")[-20:]:
            logger.info(f"  [stdout] {line}")
    if result.stderr:
        for line in result.stderr.strip().split("\n")[-10:]:
            logger.warning(f"  [stderr] {line}")

    if result.returncode != 0:
        logger.error(f"  DoomArena run failed for {site} (exit {result.returncode})")
        return None

    # Find the study dir that was just created (most recent in results/browsergym/)
    results_dir = REPO_ROOT / "results" / "browsergym"
    study_dirs = sorted(results_dir.glob("study_*"), key=lambda p: p.stat().st_mtime)
    if not study_dirs:
        logger.error("  No study directories found after run")
        return None

    study_dir = study_dirs[-1]
    logger.info(f"  Study dir: {study_dir}")

    # Verify results
    attack_csv = study_dir / "attack_results_v2.csv"
    if attack_csv.exists():
        logger.info(f"  Attack results: {attack_csv.read_text().strip().split(chr(10))[-1][:200]}")
    else:
        logger.warning("  No attack_results_v2.csv found")

    return study_dir


def discover_trajectory_dirs(study_dir: Path) -> list[dict]:
    """Find all task directories with step pickles in a DoomArena study dir.

    Returns list of dicts with keys: task_dir, task_id, has_summary, n_steps,
    attack_success, task_reward.
    """
    trajectories = []
    for sub in sorted(study_dir.rglob("summary_info.json")):
        task_dir = sub.parent
        steps = list(task_dir.glob("step_*.pkl.gz"))
        task_name = task_dir.name
        # Extract webarena task id from dir name (e.g. "..._on_webarena.27_0" → "webarena.27")
        parts = task_name.rsplit("_on_", 1)
        task_id = parts[-1].rsplit("_", 1)[0] if len(parts) > 1 else task_name

        # Read attack success from attack_summary_info.json
        attack_success = None
        attack_info_path = task_dir / "attack_summary_info.json"
        if attack_info_path.exists():
            try:
                attack_info = json.loads(attack_info_path.read_text())
                attack_success = attack_info.get("attack_successful")
            except (json.JSONDecodeError, OSError):
                pass

        # Read task reward from summary_info.json
        task_reward = None
        try:
            summary = json.loads(sub.read_text())
            task_reward = summary.get("cum_reward")
        except (json.JSONDecodeError, OSError):
            pass

        trajectories.append({
            "task_dir": str(task_dir),
            "task_id": task_id,
            "n_steps": len(steps),
            "has_summary": True,
            "attack_success": attack_success,
            "task_reward": task_reward,
        })
    return trajectories


async def run_trajectory_judges(
    trajectory_dirs: list[dict],
    site: str,
    judge_model: str,
    judge_names: list[str],
    output_dir: Path,
) -> None:
    """Run trajectory awareness judges on discovered DoomArena trajectories."""
    from eval_awareness_experiments.experiments.trajectory_awareness import (
        TrajectoryAwarenessExperiment,
    )
    from eval_awareness_experiments.llm import LLM
    from eval_awareness_experiments.types import WebsiteSample

    if not trajectory_dirs:
        logger.warning(f"  No trajectories to judge for {site}")
        return

    # Build WebsiteSample objects from the discovered trajectory dirs
    samples = []
    for traj in trajectory_dirs:
        sample = WebsiteSample(
            id=f"doomarena_{site}_{traj['task_id']}",
            source="doomarena",
            website_type=site,
            metadata={
                "trajectory_dir": traj["task_dir"],
                "benchmark": "doomarena",
                "agent": "glm-5",
                "task_id": traj["task_id"],
                "is_trajectory": True,
                "attack_success": traj.get("attack_success"),
                "task_reward": traj.get("task_reward"),
            },
        )
        samples.append(sample)

    logger.info(f"  Running judges on {len(samples)} trajectories for {site}")
    logger.info(f"  Judges: {judge_names}")
    logger.info(f"  Judge model: {judge_model}")

    model = LLM(judge_model)
    exp = TrajectoryAwarenessExperiment(
        model=model,
        output_dir=output_dir / site,
        judge_names=judge_names,
    )
    results = await exp.run(samples, ["trajectory"])
    logger.info(f"  Judge results: {len(results)} entries saved to {output_dir / site}")


def main():
    parser = argparse.ArgumentParser(description="End-to-end DoomArena pipeline: run tasks + judge transcripts")
    parser.add_argument("--sites", nargs="+", default=DEFAULT_SITES)
    parser.add_argument("--tasks-per-site", type=int, default=20)
    parser.add_argument("--agent-model", default="z-ai/glm-5:thinking")
    parser.add_argument("--judge-model", default="anthropic/claude-opus-4.6:thinking")
    parser.add_argument("--judges", nargs="+", default=["test_awareness_steering", "verbalized_awareness"])
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--output-dir", default="eval_awareness_experiments/results/doomarena_trajectory_awareness")
    parser.add_argument("--skip-doomarena", action="store_true", help="Skip DoomArena runs, only run judges on existing study dirs")
    parser.add_argument("--study-dirs", nargs="+",
                        help="Explicit study dirs to judge (with --skip-doomarena). "
                             "Format: 'site:path' (e.g. 'reddit:results/browsergym/study_...') "
                             "or just 'path' (site auto-detected from attack_results_v2.csv)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pipeline manifest: tracks what we ran
    pipeline_manifest = {
        "started": datetime.utcnow().isoformat(),
        "config": vars(args),
        "sites": {},
    }

    # Use different report ports per site to avoid conflicts if we ever parallelize
    port_map = {"reddit": 1234, "shopping": 1235, "shopping_admin": 1236, "gitlab": 1237}

    if args.dry_run:
        print(f"DRY RUN: would run {len(args.sites)} sites × {args.tasks_per_site} tasks")
        for site in args.sites:
            print(f"  {site}: {args.tasks_per_site} tasks, port {port_map.get(site, 1234)}")
        print(f"Then judge with: {args.judges} via {args.judge_model}")
        return

    study_dirs_by_site: dict[str, Path] = {}

    if args.skip_doomarena:
        # Use provided study dirs
        if args.study_dirs:
            for sd in args.study_dirs:
                # Parse "site:path" or just "path" (auto-detect site)
                if ":" in sd and not sd.startswith("/"):
                    site_name, path_str = sd.split(":", 1)
                    study_dirs_by_site[site_name] = Path(path_str)
                else:
                    p = Path(sd)
                    # Auto-detect site from attack_results_v2.csv
                    csv_path = p / "attack_results_v2.csv"
                    detected = False
                    if csv_path.exists():
                        csv_text = csv_path.read_text()
                        for site in args.sites:
                            if site in csv_text:
                                study_dirs_by_site[site] = p
                                logger.info(f"  Auto-detected site '{site}' from {p.name}")
                                detected = True
                                break
                    if not detected:
                        logger.warning(f"  Can't infer site from {sd}, skipping")
        else:
            # Find most recent study dirs by scanning attack CSVs
            results_dir = REPO_ROOT / "results" / "browsergym"
            for study_dir in sorted(results_dir.glob("study_*"), key=lambda p: p.stat().st_mtime, reverse=True):
                for site in args.sites:
                    if site not in study_dirs_by_site:
                        csv_path = study_dir / "attack_results_v2.csv"
                        if csv_path.exists() and site in csv_path.read_text():
                            study_dirs_by_site[site] = study_dir
                            logger.info(f"  Auto-detected site '{site}' from {study_dir.name}")
    else:
        # Step 1: Run DoomArena for each site
        for site in args.sites:
            port = port_map.get(site, 1234)
            study_dir = run_doomarena_site(
                site=site,
                max_tasks=args.tasks_per_site,
                agent_model=args.agent_model,
                max_steps=args.max_steps,
                report_port=port,
            )
            if study_dir:
                study_dirs_by_site[site] = study_dir
                pipeline_manifest["sites"][site] = {
                    "study_dir": str(study_dir),
                    "status": "doomarena_complete",
                }
            else:
                pipeline_manifest["sites"][site] = {"status": "doomarena_failed"}

    # Step 2: Discover trajectories and run judges
    for site, study_dir in study_dirs_by_site.items():
        logger.info(f"\n=== Judging trajectories for {site} from {study_dir} ===")
        trajectories = discover_trajectory_dirs(study_dir)
        logger.info(f"  Found {len(trajectories)} trajectories ({sum(t['n_steps'] for t in trajectories)} total steps)")

        pipeline_manifest["sites"].setdefault(site, {})
        pipeline_manifest["sites"][site]["n_trajectories"] = len(trajectories)
        pipeline_manifest["sites"][site]["trajectory_dirs"] = [t["task_dir"] for t in trajectories]

        asyncio.run(run_trajectory_judges(
            trajectory_dirs=trajectories,
            site=site,
            judge_model=args.judge_model,
            judge_names=args.judges,
            output_dir=output_dir,
        ))

        pipeline_manifest["sites"][site]["status"] = "judges_complete"

    # Save pipeline manifest
    pipeline_manifest["finished"] = datetime.utcnow().isoformat()
    manifest_path = output_dir / "pipeline_manifest.json"
    manifest_path.write_text(json.dumps(pipeline_manifest, indent=2))
    logger.info(f"\nPipeline manifest saved to {manifest_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
