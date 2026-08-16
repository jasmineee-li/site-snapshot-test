"""Import TAC experiment data (screenshots + trajectories) into eval_awareness format.

Reads from external_benchmarks/tac-experiments/evaluation/1.0.0/ which contains
agent runs from OpenHands, TTE, and MUSE against TheAgentCompany services
(GitLab, RocketChat, OwnCloud, Plane).

Usage:
    # Default: 5 trajectories per run, top 3 runs by screenshot coverage
    python -m eval_awareness_experiments.import_tac_experiments

    # Specific runs
    python -m eval_awareness_experiments.import_tac_experiments --runs 20241217_OpenHands-0.14.2-gpt-4o-2024-08-06

    # More trajectories
    python -m eval_awareness_experiments.import_tac_experiments --max-per-run 10
"""

import argparse
import gzip
import json
import os
import random
import shutil
from pathlib import Path

import yaml

TAC_EXPERIMENTS = Path("external_benchmarks/tac-experiments/evaluation/1.0.0")
TAC_TASKS = Path("external_benchmarks/TheAgentCompany/workspaces/tasks")
DEFAULT_OUTPUT = Path("eval_awareness_experiments/data/tac")
MANIFEST_PATH = Path("eval_awareness_experiments/data/manifest.json")

# Runs with best screenshot coverage and diverse agents
DEFAULT_RUNS = [
    "20241217_OpenHands-0.14.2-gpt-4o-2024-08-06",
    "20251110_TTE-MatrixAgent-Deepseek-V3.2",
    "20250510_OWL-RolePlay-gpt-4o-o3-mini",
]


def get_task_services(task_id: str) -> list[str]:
    """Get the services a task depends on from dependencies.yml."""
    # Strip -image suffix if present
    clean_id = task_id.removesuffix("-image")
    dep_file = TAC_TASKS / clean_id / "dependencies.yml"
    if dep_file.exists():
        with open(dep_file) as f:
            deps = yaml.safe_load(f)
        return deps if deps else []
    return []


def get_task_instruction(task_id: str) -> str:
    """Get the task instruction from task.md."""
    clean_id = task_id.removesuffix("-image")
    task_file = TAC_TASKS / clean_id / "task.md"
    if task_file.exists():
        return task_file.read_text().strip()
    return ""


def detect_run_agent(run_name: str) -> str:
    """Infer agent type from run directory name."""
    name_lower = run_name.lower()
    if "openhands" in name_lower:
        return "openhands"
    if "tte" in name_lower:
        return "tte"
    if "muse" in name_lower:
        return "muse"
    if "owl" in name_lower:
        return "owl"
    return "unknown"


def detect_traj_format(run_name: str) -> str:
    """Detect trajectory file extension for a run."""
    traj_dir = TAC_EXPERIMENTS / run_name / "trajectories"
    if not traj_dir.exists():
        return ""
    for f in os.listdir(traj_dir):
        if f.endswith(".json.gz"):
            return ".json.gz"
        if f.endswith(".json"):
            return ".json"
        if f.endswith(".txt"):
            return ".txt"
    return ""


def load_trajectory(traj_path: Path) -> dict | str:
    """Load a trajectory file in any format."""
    name = traj_path.name
    if name.endswith(".json.gz"):
        with gzip.open(traj_path, "rt", encoding="utf-8") as f:
            return json.load(f)
    elif name.endswith(".json"):
        with open(traj_path, encoding="utf-8") as f:
            return json.load(f)
    elif name.endswith(".txt"):
        return traj_path.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Unknown trajectory format: {name}")


def extract_task_id(filename: str) -> str:
    """Extract task_id from trajectory filename like traj_{task_id}.json.gz."""
    name = filename
    for ext in [".json.gz", ".json", ".txt"]:
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    if name.startswith("traj_"):
        name = name[len("traj_") :]
    return name


def import_run(
    run_name: str,
    output_dir: Path,
    max_tasks: int,
    seed: int,
) -> list[dict]:
    """Import screenshots and trajectories from a single run.

    Returns list of trajectory metadata dicts.
    """
    run_dir = TAC_EXPERIMENTS / run_name
    screenshots_dir = run_dir / "screenshots"
    traj_dir = run_dir / "trajectories"
    agent = detect_run_agent(run_name)
    traj_ext = detect_traj_format(run_name)

    if not traj_dir.exists():
        print("  No trajectories dir, skipping")
        return []

    # List all tasks with trajectories
    traj_files = sorted(os.listdir(traj_dir))
    task_ids = [extract_task_id(f) for f in traj_files]

    # Prefer tasks that have screenshots
    has_screenshots = set()
    if screenshots_dir.exists():
        has_screenshots = set(os.listdir(screenshots_dir))

    tasks_with_screenshots = [t for t in task_ids if t in has_screenshots]
    tasks_without = [t for t in task_ids if t not in has_screenshots]

    rng = random.Random(seed)
    rng.shuffle(tasks_with_screenshots)
    rng.shuffle(tasks_without)

    # Prioritize tasks with screenshots
    selected = (tasks_with_screenshots + tasks_without)[:max_tasks]

    trajectories = []
    for task_id in selected:
        task_output = output_dir / run_name / task_id
        task_output.mkdir(parents=True, exist_ok=True)

        # Copy trajectory
        traj_file = f"traj_{task_id}{traj_ext}"
        traj_src = traj_dir / traj_file
        if traj_src.exists():
            traj_data = load_trajectory(traj_src)
            dst = task_output / "trajectory.json"
            if isinstance(traj_data, str):
                # MUSE text format — wrap it
                traj_data = {"format": "muse_text", "content": traj_data}
            with open(dst, "w", encoding="utf-8") as f:
                json.dump(traj_data, f)

        # Copy screenshots
        task_screenshots = screenshots_dir / task_id if screenshots_dir.exists() else None
        if task_screenshots and task_screenshots.is_dir():
            screenshots_dst = task_output / "screenshots"
            screenshots_dst.mkdir(exist_ok=True)
            for png in sorted(task_screenshots.glob("*.png")):
                shutil.copy2(png, screenshots_dst / png.name)

        services = get_task_services(task_id)
        instruction = get_task_instruction(task_id)

        traj_meta = {
            "task_id": task_id,
            "agent": agent,
            "run": run_name,
            "services": services,
            "instruction": instruction[:200] if instruction else "",
            "has_screenshots": task_id in has_screenshots,
            "dir": str(task_output),
        }
        trajectories.append(traj_meta)
        status = "+" if task_id in has_screenshots else " "
        print(f"  [{status}] {task_id}")

    return trajectories


def update_manifest(
    trajectories_by_run: dict[str, list[dict]],
    manifest_path: Path,
) -> None:
    """Add TAC entries to the manifest."""
    existing = []
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())

    # Remove old tac entries
    existing = [e for e in existing if e.get("source") != "tac"]

    # Group trajectories by service combination for manifest entries
    # Each unique service set gets one manifest entry
    service_groups: dict[str, list[dict]] = {}
    for _run_name, trajs in trajectories_by_run.items():
        for t in trajs:
            key = ",".join(sorted(t["services"])) if t["services"] else "none"
            service_groups.setdefault(key, []).append(t)

    for service_key, trajs in service_groups.items():
        services = service_key.split(",") if service_key != "none" else []
        entry = {
            "id": f"tac_{service_key}",
            "source": "tac",
            "label": "synthetic",
            "path": str(DEFAULT_OUTPUT),
            "pages": [],  # Will be populated when Docker extraction is done
            "trajectories": trajs,
            "metadata": {
                "benchmark": "theagentcompany",
                "services": services,
            },
        }
        existing.append(entry)

    manifest_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"\nManifest updated: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Import TAC experiment screenshots and trajectories"
    )
    parser.add_argument(
        "--runs",
        type=str,
        default=",".join(DEFAULT_RUNS),
        help="Comma-separated run names to import",
    )
    parser.add_argument(
        "--max-per-run",
        type=int,
        default=5,
        help="Max tasks to import per run",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output directory",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(MANIFEST_PATH),
        help="Manifest file path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List available runs and exit",
    )
    args = parser.parse_args()

    if args.list_runs:
        for run in sorted(os.listdir(TAC_EXPERIMENTS)):
            run_dir = TAC_EXPERIMENTS / run
            if not run_dir.is_dir():
                continue
            agent = detect_run_agent(run)
            traj_count = (
                len(os.listdir(run_dir / "trajectories"))
                if (run_dir / "trajectories").exists()
                else 0
            )
            screenshot_count = (
                len(os.listdir(run_dir / "screenshots"))
                if (run_dir / "screenshots").exists()
                else 0
            )
            print(f"{run}  agent={agent}  trajs={traj_count}  screenshots={screenshot_count}")
        return

    runs = [r.strip() for r in args.runs.split(",")]
    output_dir = Path(args.output)

    # Validate runs exist
    for run in runs:
        if not (TAC_EXPERIMENTS / run).exists():
            print(f"Error: Run not found: {TAC_EXPERIMENTS / run}")
            return

    trajectories_by_run: dict[str, list[dict]] = {}
    total = 0

    for run in runs:
        print(f"\n{run} ({detect_run_agent(run)}):")
        trajs = import_run(run, output_dir, args.max_per_run, args.seed)
        trajectories_by_run[run] = trajs
        total += len(trajs)
        print(f"  Imported {len(trajs)} tasks")

    if total > 0:
        update_manifest(trajectories_by_run, Path(args.manifest))

    print(f"\nDone. Imported {total} tasks across {len(runs)} runs.")


if __name__ == "__main__":
    main()
