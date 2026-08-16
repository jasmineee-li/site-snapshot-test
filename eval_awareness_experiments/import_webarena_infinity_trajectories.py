"""Import agent trajectories from webarena-infinity HuggingFace dataset.

Downloads history.json, result.json, and screenshots for selected apps/agents,
then saves them into the data directory alongside the extracted page data.

Usage:
    # Default: 5 trajectories per app, all available agents
    python -m eval_awareness_experiments.import_webarena_infinity_trajectories

    # More trajectories, specific apps
    python -m eval_awareness_experiments.import_webarena_infinity_trajectories --max-per-app 10 --apps gmail,paypal-my-wallet

    # Specific agent
    python -m eval_awareness_experiments.import_webarena_infinity_trajectories --agents kimi
"""

import argparse
import json
import random
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_tree

REPO_ID = "webarena-x/webarena-infinity-trajectories"
REPO_TYPE = "dataset"

DEFAULT_APPS = [
    "gmail",
    "gmail-accounts-and-contacts",
    "gitlab-plan-and-track",
    "paypal-my-wallet",
]
DEFAULT_AGENTS = ["gemini", "kimi", "qwen"]
DEFAULT_OUTPUT = Path("eval_awareness_experiments/data/webarena-infinity")
MANIFEST_PATH = Path("eval_awareness_experiments/data/manifest.json")


def list_available_tasks(agent: str, app: str) -> list[str]:
    """List task IDs available for a given agent/app combo."""
    try:
        entries = list(list_repo_tree(
            REPO_ID, repo_type=REPO_TYPE,
            path_in_repo=f"data/{agent}/{app}",
        ))
        return [e.path.split("/")[-1] for e in entries]
    except Exception:
        return []


def download_trajectory(
    agent: str, app: str, task_id: str, output_dir: Path
) -> dict | None:
    """Download a single trajectory. Returns trajectory metadata or None on failure."""
    traj_dir = output_dir / f"{agent}_{task_id}"
    traj_dir.mkdir(parents=True, exist_ok=True)

    remote_prefix = f"data/{agent}/{app}/{task_id}"

    # Download history.json and result.json
    downloaded = {}
    for filename in ["history.json", "result.json"]:
        try:
            local_path = hf_hub_download(
                REPO_ID, f"{remote_prefix}/{filename}",
                repo_type=REPO_TYPE, local_dir=str(output_dir / ".hf_cache"),
            )
            # Copy to our structure
            dst = traj_dir / filename
            dst.write_bytes(Path(local_path).read_bytes())
            downloaded[filename] = str(dst)
        except Exception as e:
            print(f"    Failed to download {filename}: {e}")
            return None

    # Download screenshots
    try:
        screenshot_entries = list(list_repo_tree(
            REPO_ID, repo_type=REPO_TYPE,
            path_in_repo=f"{remote_prefix}/screenshots",
        ))
        screenshots_dir = traj_dir / "screenshots"
        screenshots_dir.mkdir(exist_ok=True)
        for entry in screenshot_entries:
            if hasattr(entry, "rfilename") and entry.rfilename.endswith(".png"):
                local_path = hf_hub_download(
                    REPO_ID, entry.rfilename,
                    repo_type=REPO_TYPE, local_dir=str(output_dir / ".hf_cache"),
                )
                dst = screenshots_dir / Path(entry.rfilename).name
                dst.write_bytes(Path(local_path).read_bytes())
    except Exception:
        pass  # Screenshots are optional

    # Read result.json for metadata
    result = json.loads((traj_dir / "result.json").read_text())

    return {
        "agent": agent,
        "task_id": task_id,
        "instruction": result.get("instruction", ""),
        "passed": result.get("passed"),
        "steps": result.get("steps"),
        "difficulty": result.get("difficulty", ""),
        "dir": str(traj_dir),
    }


def import_trajectories(
    apps: list[str],
    agents: list[str],
    output_dir: Path,
    max_per_app: int,
    seed: int,
) -> dict[str, list[dict]]:
    """Import trajectories for specified apps. Returns {app: [trajectory_metadata]}."""
    rng = random.Random(seed)
    all_trajectories: dict[str, list[dict]] = {}

    for app in apps:
        print(f"\n{app}:")
        app_output = output_dir / app / "trajectories"

        # Collect all available (agent, task_id) pairs
        candidates: list[tuple[str, str]] = []
        for agent in agents:
            task_ids = list_available_tasks(agent, app)
            if not task_ids:
                print(f"  {agent}: not available")
                continue
            print(f"  {agent}: {len(task_ids)} tasks available")
            for tid in task_ids:
                candidates.append((agent, tid))

        if not candidates:
            print("  No trajectories found, skipping")
            continue

        # Sample
        rng.shuffle(candidates)
        selected = candidates[:max_per_app]

        trajectories = []
        for agent, task_id in selected:
            print(f"  Downloading {agent}/{task_id}...", end=" ")
            traj = download_trajectory(agent, app, task_id, app_output)
            if traj:
                trajectories.append(traj)
                print("ok")
            else:
                print("FAILED")

        all_trajectories[app] = trajectories
        print(f"  Imported {len(trajectories)}/{max_per_app} trajectories")

    return all_trajectories


def update_manifest(
    trajectories_by_app: dict[str, list[dict]],
    manifest_path: Path,
) -> None:
    """Update manifest entries with trajectory information."""
    existing = []
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())

    for entry in existing:
        app = entry.get("metadata", {}).get("app")
        if app in trajectories_by_app:
            entry["trajectories"] = trajectories_by_app[app]

    manifest_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"\nManifest updated: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Import webarena-infinity agent trajectories from HuggingFace"
    )
    parser.add_argument(
        "--apps", type=str, default=",".join(DEFAULT_APPS),
        help="Comma-separated app names",
    )
    parser.add_argument(
        "--agents", type=str, default=",".join(DEFAULT_AGENTS),
        help="Comma-separated agent names (gemini, kimi, qwen)",
    )
    parser.add_argument(
        "--max-per-app", type=int, default=5,
        help="Max trajectories to import per app",
    )
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_OUTPUT),
        help="Output directory",
    )
    parser.add_argument(
        "--manifest", type=str, default=str(MANIFEST_PATH),
        help="Manifest file path",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling",
    )
    args = parser.parse_args()

    apps = [a.strip() for a in args.apps.split(",")]
    agents = [a.strip() for a in args.agents.split(",")]
    output_dir = Path(args.output)

    trajectories_by_app = import_trajectories(
        apps, agents, output_dir, args.max_per_app, args.seed,
    )

    total = sum(len(t) for t in trajectories_by_app.values())
    if total > 0:
        update_manifest(trajectories_by_app, Path(args.manifest))
    print(f"\nDone. Imported {total} trajectories across {len(trajectories_by_app)} apps.")


if __name__ == "__main__":
    main()
