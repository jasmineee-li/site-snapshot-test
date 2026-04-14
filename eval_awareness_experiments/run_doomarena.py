"""Run DoomArena tasks via AgentLab/BrowserGym with configurable model and injection conditions.

DoomArena is a modular security testing framework for AI agents built on
BrowserGym + AgentLab. Attack types include popups, banners, user-content
replacement, and div injection.

Usage:
    # Baseline run with Opus 4.6
    python -m eval_awareness_experiments.run_doomarena \
        --model anthropic/claude-opus-4.6 \
        --condition baseline \
        --max-tasks 10

    # Causal experiment: eval_cue condition
    python -m eval_awareness_experiments.run_doomarena \
        --model anthropic/claude-opus-4.6 \
        --condition eval_cue \
        --max-tasks 10

Prerequisites:
    1. Clone DoomArena: git clone https://github.com/ServiceNow/DoomArena
    2. Install: pip install -e DoomArena/doomarena/browsergym
    3. Ensure AgentLab is installed: pip install -e AgentLab/
    4. Set OPENROUTER_API_KEY or ANTHROPIC_API_KEY in .env
"""

import argparse
import json
import sys
from pathlib import Path

from eval_awareness_experiments.injection_modifier import (
    ALL_CONDITIONS,
    InjectionCondition,
    wrap_injection,
)


def run_doomarena_tasks(
    model: str,
    condition: InjectionCondition,
    max_tasks: int,
    output_dir: Path,
    doomarena_path: Path | None = None,
) -> None:
    """Run DoomArena tasks and save trajectories.

    This function integrates with the existing AgentLab Study class and
    DoomArena's BrowserGym attack framework. The injection condition is
    applied by patching the attack content before it's injected into the
    browser environment.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check DoomArena is available
    if doomarena_path is None:
        doomarena_path = Path("DoomArena")
    if not doomarena_path.exists():
        print(f"DoomArena not found at {doomarena_path}")
        print("Clone it: git clone https://github.com/ServiceNow/DoomArena")
        sys.exit(1)

    try:
        from agentlab.experiments.study import Study
        from agentlab.agents.generic_agent import AGENT_4o  # template config
    except ImportError:
        print("AgentLab not installed. Run: pip install -e AgentLab/")
        sys.exit(1)

    # Configure the agent to use the specified model
    agent_args = AGENT_4o.copy()
    agent_args.chat_model_args.model_name = model

    # Load DoomArena task configs
    task_config_dir = doomarena_path / "doomarena" / "browsergym" / "configs"
    if not task_config_dir.exists():
        print(f"DoomArena configs not found at {task_config_dir}")
        sys.exit(1)

    config_files = sorted(task_config_dir.glob("*.json"))[:max_tasks]
    print(f"Found {len(config_files)} DoomArena task configs, running {min(max_tasks, len(config_files))}")
    print(f"Model: {model}")
    print(f"Condition: {condition}")
    print(f"Output: {output_dir}")

    # For each task, run the agent with the specified injection condition
    results = []
    for config_file in config_files:
        task_config = json.loads(config_file.read_text())
        task_id = config_file.stem

        print(f"\n  Running task: {task_id}")

        # Apply injection condition wrapper to attack content
        if condition != "baseline" and "attack_content" in task_config:
            original = task_config["attack_content"]
            task_config["attack_content"] = wrap_injection(original, condition)

        # TODO: Integrate with DoomArena's BrowserGym environment
        # This requires:
        # 1. Setting up the BrowserGym environment with the task config
        # 2. Running the GenericAgent in the environment
        # 3. Capturing step_*.pkl.gz trajectory files
        # 4. Running the redteam judge for ASR

        results.append({
            "task_id": task_id,
            "model": model,
            "condition": condition,
            "status": "pending_implementation",
        })

    # Save results manifest
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(results, indent=2))
    print(f"\nRun manifest saved to {manifest_path}")
    print(f"Tasks queued: {len(results)}")


def main():
    parser = argparse.ArgumentParser(description="Run DoomArena tasks with eval-awareness injection conditions")
    parser.add_argument("--model", type=str, default="anthropic/claude-opus-4.6",
                        help="Model to use as the agent")
    parser.add_argument("--condition", type=str, default="baseline", choices=ALL_CONDITIONS,
                        help="Injection condition")
    parser.add_argument("--max-tasks", type=int, default=10,
                        help="Max number of tasks to run")
    parser.add_argument("--output-dir", type=str,
                        default="eval_awareness_experiments/data/doomarena",
                        help="Output directory for trajectories")
    parser.add_argument("--doomarena-path", type=str, default=None,
                        help="Path to DoomArena repo (default: ./DoomArena)")
    args = parser.parse_args()

    run_doomarena_tasks(
        model=args.model,
        condition=args.condition,
        max_tasks=args.max_tasks,
        output_dir=Path(args.output_dir),
        doomarena_path=Path(args.doomarena_path) if args.doomarena_path else None,
    )


if __name__ == "__main__":
    main()
