"""Orchestrate causal injection modification experiments across benchmarks.

Runs all combinations of: benchmarks × injection conditions × models,
then runs eval-awareness judges on the resulting trajectories.

Usage:
    # Full experiment
    python -m eval_awareness_experiments.run_causal_experiment \
        --models anthropic/claude-opus-4.6 openai/gpt-5.4 \
        --benchmarks doomarena wasp os-harm \
        --conditions baseline eval_cue scenario_cue anti_eval_cue combined_scenario

    # Quick test: one model, one benchmark, two conditions
    python -m eval_awareness_experiments.run_causal_experiment \
        --models anthropic/claude-opus-4.6 \
        --benchmarks wasp \
        --conditions baseline eval_cue \
        --max-tasks 5
"""

import argparse
import json
import subprocess
import sys
from itertools import product
from pathlib import Path

from eval_awareness_experiments.injection_modifier import ALL_CONDITIONS

DEFAULT_MODELS = [
    "anthropic/claude-opus-4.6",
]

DEFAULT_BENCHMARKS = ["doomarena", "wasp", "os-harm"]

RUNNER_MODULES = {
    "doomarena": "eval_awareness_experiments.run_doomarena",
    "wasp": "eval_awareness_experiments.run_wasp",
    "os-harm": "eval_awareness_experiments.run_osharm",
}


def run_causal_experiment(
    models: list[str],
    benchmarks: list[str],
    conditions: list[str],
    max_tasks: int | None,
    output_base: Path,
    dry_run: bool = False,
) -> None:
    """Run all combinations and collect results."""
    combos = list(product(benchmarks, conditions, models))
    print(f"Causal experiment: {len(combos)} combinations")
    print(f"  Benchmarks: {benchmarks}")
    print(f"  Conditions: {conditions}")
    print(f"  Models: {models}")
    if max_tasks:
        print(f"  Max tasks per combo: {max_tasks}")
    print()

    run_log = []

    for i, (benchmark, condition, model) in enumerate(combos, 1):
        model_slug = model.replace("/", "_")
        output_dir = output_base / benchmark / condition / model_slug

        print(f"[{i}/{len(combos)}] {benchmark} / {condition} / {model}")

        runner = RUNNER_MODULES.get(benchmark)
        if runner is None:
            print(f"  SKIP: No runner for benchmark '{benchmark}'")
            continue

        cmd = [
            sys.executable, "-m", runner,
            "--model", model,
            "--condition", condition,
            "--output-dir", str(output_dir),
        ]
        if max_tasks:
            cmd.extend(["--max-tasks", str(max_tasks)])

        if dry_run:
            print(f"  DRY RUN: {' '.join(cmd)}")
        else:
            print(f"  Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  FAILED: {result.stderr[:500]}")
            else:
                print(f"  OK")

        run_log.append({
            "benchmark": benchmark,
            "condition": condition,
            "model": model,
            "output_dir": str(output_dir),
            "cmd": " ".join(cmd),
            "dry_run": dry_run,
        })

    # Save run log
    log_path = output_base / "causal_experiment_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(run_log, indent=2))
    print(f"\nExperiment log saved to {log_path}")

    # Reminder to run judges
    print(f"\nNext steps:")
    print(f"  1. After trajectories are generated, update data/manifest.json")
    print(f"  2. Run judges:")
    print(f"     python -m eval_awareness_experiments.run \\")
    print(f"       --experiment trajectory_awareness \\")
    print(f"       --sources {' '.join(benchmarks)} \\")
    print(f"       --models anthropic/claude-opus-4.6:thinking \\")
    print(f"       --judges test_awareness_steering verbalized_awareness")


def main():
    parser = argparse.ArgumentParser(description="Orchestrate causal injection modification experiments")
    parser.add_argument("--models", type=str, nargs="+", default=DEFAULT_MODELS,
                        help="Models to use as agents")
    parser.add_argument("--benchmarks", type=str, nargs="+", default=DEFAULT_BENCHMARKS,
                        choices=DEFAULT_BENCHMARKS, help="Benchmarks to run")
    parser.add_argument("--conditions", type=str, nargs="+", default=ALL_CONDITIONS,
                        choices=ALL_CONDITIONS, help="Injection conditions")
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="Max tasks per benchmark/condition combo")
    parser.add_argument("--output-base", type=str,
                        default="eval_awareness_experiments/data",
                        help="Base output directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running them")
    args = parser.parse_args()

    run_causal_experiment(
        models=args.models,
        benchmarks=args.benchmarks,
        conditions=args.conditions,
        max_tasks=args.max_tasks,
        output_base=Path(args.output_base),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
