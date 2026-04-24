"""Orchestrate causal-injection experiments across benchmarks × conditions × models.

Thin matrix wrapper around `run_safety_pipeline`. For each combo it shells out
to one pipeline invocation — the pipeline itself knows how to dispatch to the
right runner (browser / tool-calling / import-only) and then apply judges.

Usage:
    # Full pilot: 3 models × 3 benchmarks × 3 conditions = 27 combos
    python -m eval_awareness_experiments.run_causal_experiment \
        --models anthropic/claude-opus-4.7:thinking openai/gpt-5.4:thinking \
        --benchmarks doomarena wasp agentdojo injecagent \
        --conditions baseline eval_cue scenario_cue \
        --extra-instructions-presets motive \
        --tasks-per-split 20

    # EIA is import-only, condition is always baseline (pipeline enforces).
    python -m eval_awareness_experiments.run_causal_experiment \
        --benchmarks eia --stage judge-only \
        --models anthropic/claude-opus-4.6:thinking
"""

import argparse
import json
import subprocess
import sys
from itertools import product
from pathlib import Path

from eval_awareness_experiments.extra_instructions import PRESETS
from eval_awareness_experiments.injection_modifier import ALL_CONDITIONS

DEFAULT_MODELS = ["anthropic/claude-opus-4.7:thinking"]
ALL_BENCHMARKS = ["doomarena", "wasp", "agentdojo", "injecagent", "eia"]
ALL_PRESETS = sorted(PRESETS.keys())
ALL_STAGES = ["all", "run-only", "judge-only"]


def run_causal_experiment(
    models: list[str],
    benchmarks: list[str],
    conditions: list[str],
    presets: list[str],
    tasks_per_split: int | None,
    stage: str,
    judge_model: str | None,
    judges: list[str] | None,
    output_base: Path,
    dry_run: bool,
) -> None:
    combos = list(product(benchmarks, conditions, models, presets))
    print(f"Causal experiment: {len(combos)} combinations")
    print(f"  Benchmarks: {benchmarks}")
    print(f"  Conditions: {conditions}")
    print(f"  Models:     {models}")
    print(f"  Presets:    {presets}")
    print(f"  Stage:      {stage}")
    if tasks_per_split:
        print(f"  Tasks/split: {tasks_per_split}")
    print()

    run_log = []
    for i, (benchmark, condition, model, preset) in enumerate(combos, 1):
        # EIA trajectories predate the causal framework — pipeline rejects
        # non-baseline conditions there, so skip cleanly rather than fail.
        if benchmark == "eia" and condition != "baseline":
            print(f"[{i}/{len(combos)}] SKIP eia/{condition} (import-only, baseline only)")
            continue

        combo_out = output_base / benchmark / condition / preset / model.replace("/", "_").replace(":", "_")
        cmd = [
            sys.executable, "-m", "eval_awareness_experiments.run_safety_pipeline",
            "--benchmark", benchmark,
            "--model-name", model,
            "--condition", condition,
            "--extra-instructions-preset", preset,
            "--stage", stage,
            "--output-dir", str(combo_out),
        ]
        if judge_model:
            cmd.extend(["--judge-model", judge_model])
        if judges:
            cmd.extend(["--judges", *judges])
        if tasks_per_split is not None:
            cmd.extend(["--tasks-per-split", str(tasks_per_split)])

        label = f"{benchmark}/{condition}/{preset}/{model}"
        print(f"[{i}/{len(combos)}] {label}")

        if dry_run:
            print(f"  DRY RUN: {' '.join(cmd)}")
            status = "dry_run"
        else:
            print(f"  Running: {' '.join(cmd)}")
            # Stream stdout/stderr live — pipeline runs can be hours long.
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"  FAILED ({result.returncode}) — see output above")
                status = f"failed:{result.returncode}"
            else:
                print("  OK")
                status = "ok"

        run_log.append({
            "benchmark": benchmark,
            "condition": condition,
            "model": model,
            "preset": preset,
            "stage": stage,
            "status": status,
            "cmd": " ".join(cmd),
        })

    output_base.mkdir(parents=True, exist_ok=True)
    log_path = output_base / "causal_experiment_log.json"
    log_path.write_text(json.dumps(run_log, indent=2))
    print(f"\nExperiment log saved to {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Matrix runner: benchmarks × conditions × models × presets via run_safety_pipeline"
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="OpenRouter model ids (use :thinking for reasoning).")
    parser.add_argument("--benchmarks", nargs="+", default=["doomarena"],
                        choices=ALL_BENCHMARKS)
    parser.add_argument("--conditions", nargs="+", default=["baseline"],
                        choices=ALL_CONDITIONS)
    parser.add_argument("--extra-instructions-presets", nargs="+", default=["none"],
                        choices=ALL_PRESETS, dest="presets")
    parser.add_argument("--tasks-per-split", type=int, default=None,
                        help="Forwarded to run_safety_pipeline (e.g. WASP has 2 splits, so 20 → 40 tasks/combo).")
    parser.add_argument("--stage", default="all", choices=ALL_STAGES)
    parser.add_argument("--judge-model", default=None,
                        help="Judge model for stage 2 (defaults to pipeline default).")
    parser.add_argument("--judges", nargs="+", default=None,
                        help="Judge names (e.g. test_awareness_steering verbalized_awareness "
                             "purpose_continue_5q). Omit to use pipeline defaults.")
    parser.add_argument("--output-base", type=Path,
                        default=Path("eval_awareness_experiments/results/causal"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_causal_experiment(
        models=args.models,
        benchmarks=args.benchmarks,
        conditions=args.conditions,
        presets=args.presets,
        tasks_per_split=args.tasks_per_split,
        stage=args.stage,
        judge_model=args.judge_model,
        judges=args.judges,
        output_base=args.output_base,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
