"""Run SafeArena benchmark tasks with configurable model and injection conditions.

SafeArena (McGill-NLP, ICML 2025) is a BrowserGym + AgentLab + WebArena-based
benchmark for evaluating harmful capabilities of web agents. 500 tasks total:
- `harm.json` — 250 harmful tasks across 5 categories (misinformation, harassment,
  illegal activity, cybercrime, social bias). Agents SHOULD refuse.
- `safe.json` — 250 matched benign tasks (control).

4 web environments: shopping, Reddit, GitLab, homepage (WebArena-derived Docker).
ASR = % of harmful tasks the agent completes (score > 0.5).

Note: SafeArena already uses <rule> XML tags in HarmfulExtraInstructions
(goal + jailbreak preamble). Our eval_cue / scenario_cue conditions wrap
on top of the harmful goal text, so we're stacking XML cues. Useful for
studying the interaction between eval-awareness cues and existing jailbreaks.

Paper: https://arxiv.org/abs/2503.04957
Repo:  https://github.com/McGill-NLP/safearena
Data:  https://huggingface.co/datasets/McGill-NLP/safearena

Usage:
    # Baseline run with Opus 4.6 on harm tasks
    python -m eval_awareness_experiments.run_safearena \
        --model anthropic/claude-opus-4.6 \
        --condition baseline \
        --split harm \
        --max-tasks 100

    # Causal experiment: eval_cue condition
    python -m eval_awareness_experiments.run_safearena \
        --model anthropic/claude-opus-4.6 \
        --condition eval_cue \
        --split harm

Prerequisites:
    1. Clone SafeArena: git clone https://github.com/McGill-NLP/safearena
    2. Install: cd safearena && pip install -e .
    3. Set up SafeArena Docker environments (see their env-setup/)
    4. Download tasks: huggingface-cli download McGill-NLP/safearena
    5. Set OPENROUTER_API_KEY or ANTHROPIC_API_KEY in .env
    6. Ensure AgentLab is installed: pip install -e AgentLab/
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


def run_safearena_tasks(
    model: str,
    condition: InjectionCondition,
    split: str,
    output_dir: Path,
    safearena_path: Path | None = None,
    tasks_file: Path | None = None,
    max_tasks: int | None = None,
    n_jobs: int = 1,
) -> None:
    """Run SafeArena tasks and save trajectories.

    SafeArena loads task IDs from environment variable SAFEARENA_TASK
    (defaulting to "harm"). Tasks are defined in harm.json / safe.json.

    For causal experiments, we patch the task instruction (or harmful
    payload field) by wrapping with the condition's XML tags.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if safearena_path is None:
        safearena_path = Path("safearena")
    if not safearena_path.exists():
        print(f"SafeArena repo not found at {safearena_path}")
        print("Clone it: git clone https://github.com/McGill-NLP/safearena")
        sys.exit(1)

    # Locate tasks JSON
    if tasks_file is None:
        for candidate in [
            safearena_path / f"{split}.json",
            safearena_path / "data" / f"{split}.json",
            safearena_path / "tasks" / f"{split}.json",
        ]:
            if candidate.exists():
                tasks_file = candidate
                break
    if tasks_file is None or not tasks_file.exists():
        print(f"SafeArena {split}.json not found. Download from HuggingFace:")
        print("  huggingface-cli download McGill-NLP/safearena --repo-type dataset")
        sys.exit(1)

    tasks = json.loads(tasks_file.read_text())
    if not isinstance(tasks, list):
        tasks = tasks.get("tasks", tasks.get("examples", []))

    if max_tasks:
        tasks = tasks[:max_tasks]

    print(f"Loaded {len(tasks)} SafeArena {split} tasks from {tasks_file}")
    print(f"Model: {model}")
    print(f"Condition: {condition}")
    print(f"Output: {output_dir}")

    # Apply injection condition to each task's instruction content
    if condition != "baseline":
        for task in tasks:
            # Common fields that may carry the harmful payload
            for key in ("intent", "instruction", "task", "goal"):
                if key in task and isinstance(task[key], str):
                    task[key] = wrap_injection(task[key], condition)
                    break

    # Save modified tasks to a temp file for the SafeArena runner to pick up
    modified_tasks_file = output_dir / f"{split}_{condition}.json"
    modified_tasks_file.write_text(json.dumps(tasks, indent=2))

    try:
        from agentlab.experiments.study import Study  # noqa: F401
    except ImportError:
        print("AgentLab not installed. Run: pip install -e AgentLab/")
        sys.exit(1)

    # TODO: Integrate with SafeArena's runner
    # This requires:
    # 1. Setting SAFEARENA_TASK=<split> env var
    # 2. Pointing the benchmark at modified_tasks_file
    # 3. Running scripts/launch_experiment.py equivalent with the specified model
    # 4. AgentLab saves trajectories as step_*.pkl.gz + summary_info.json
    #    (these will be parsed by parse_safearena_trajectory in trajectory_converter.py)
    # 5. SafeArena's own evaluator determines task success (ASR for harm split)

    manifest = {
        "benchmark": "safearena",
        "split": split,
        "condition": condition,
        "model": model,
        "tasks_file": str(modified_tasks_file),
        "n_tasks": len(tasks),
        "n_jobs": n_jobs,
        "output_dir": str(output_dir),
        "status": "pending_implementation",
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nRun manifest saved to {output_dir / 'run_manifest.json'}")
    print(f"Tasks queued: {len(tasks)}")


def main():
    parser = argparse.ArgumentParser(
        description="Run SafeArena tasks with eval-awareness injection conditions",
    )
    parser.add_argument("--model", type=str, default="anthropic/claude-opus-4.6",
                        help="Model to use as the agent")
    parser.add_argument("--condition", type=str, default="baseline", choices=ALL_CONDITIONS,
                        help="Injection condition")
    parser.add_argument("--split", type=str, default="harm", choices=["harm", "safe"],
                        help="SafeArena task split: harm (harmful tasks) or safe (benign control)")
    parser.add_argument("--output-dir", type=str,
                        default="eval_awareness_experiments/data/safearena",
                        help="Output directory for trajectories")
    parser.add_argument("--safearena-path", type=str, default=None,
                        help="Path to SafeArena repo (default: ./safearena)")
    parser.add_argument("--tasks-file", type=str, default=None,
                        help="Override path to tasks JSON file")
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="Max tasks to run (default: all)")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Parallel jobs (default: 1)")
    args = parser.parse_args()

    run_safearena_tasks(
        model=args.model,
        condition=args.condition,
        split=args.split,
        output_dir=Path(args.output_dir),
        safearena_path=Path(args.safearena_path) if args.safearena_path else None,
        tasks_file=Path(args.tasks_file) if args.tasks_file else None,
        max_tasks=args.max_tasks,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
