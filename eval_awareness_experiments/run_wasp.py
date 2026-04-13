"""Run WASP benchmark tasks with configurable model and injection conditions.

WASP (Web Agent Security against Prompt Injection) is built on VisualWebArena
with GitLab and Reddit clone environments. Injection formats include goal
hijacking (URL/plaintext) and generic (URL/plaintext).

Usage:
    # Baseline run with Opus 4.6
    python -m eval_awareness_experiments.run_wasp \
        --model anthropic/claude-opus-4.6 \
        --condition baseline

    # Causal experiment: scenario_cue condition
    python -m eval_awareness_experiments.run_wasp \
        --model anthropic/claude-opus-4.6 \
        --condition scenario_cue

Prerequisites:
    1. Clone WASP: git clone https://github.com/facebookresearch/wasp
    2. Set up VisualWebArena Docker environments (GitLab + Reddit)
    3. Set OPENROUTER_API_KEY or ANTHROPIC_API_KEY in .env
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


def run_wasp_tasks(
    model: str,
    condition: InjectionCondition,
    output_dir: Path,
    wasp_path: Path | None = None,
    output_format: str = "claude",
    max_tasks: int | None = None,
) -> None:
    """Run WASP prompt injection tasks and save trajectories.

    WASP injection content lives in constants.py PromptInjectionFormat.MESSAGE
    templates. For causal experiments, we wrap the instantiated injection text
    with the condition's XML tags before it's placed in the environment.

    Modification point: _prepare_injection() in prompt_injector.py, after
    the .format() call produces the final injection text.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if wasp_path is None:
        wasp_path = Path("wasp")
    if not wasp_path.exists():
        print(f"WASP repo not found at {wasp_path}")
        print("Clone it: git clone https://github.com/facebookresearch/wasp")
        sys.exit(1)

    # Load WASP task configs
    task_dir = wasp_path / "webarena_prompt_injections" / "configs"
    if not task_dir.exists():
        # Try alternative paths
        task_dir = wasp_path / "configs"
    if not task_dir.exists():
        print(f"WASP configs not found. Check the repo structure at {wasp_path}")
        sys.exit(1)

    task_files = sorted(task_dir.glob("*.json"))
    if max_tasks:
        task_files = task_files[:max_tasks]

    print(f"Found {len(task_files)} WASP task configs")
    print(f"Model: {model}")
    print(f"Condition: {condition}")
    print(f"Output format: {output_format}")
    print(f"Output: {output_dir}")

    results = []
    for task_file in task_files:
        task_config = json.loads(task_file.read_text())
        task_id = task_file.stem

        print(f"\n  Running task: {task_id}")

        # Apply injection condition: wrap the injection content
        if condition != "baseline":
            for key in ["instantiated_instruction", "injection_content", "content"]:
                if key in task_config:
                    original = task_config[key]
                    task_config[key] = wrap_injection(original, condition)
                    break
            # Also check nested injection config
            pi_config = task_config.get("prompt_injection_config", {})
            if "instantiated_instruction" in pi_config:
                original = pi_config["instantiated_instruction"]
                pi_config["instantiated_instruction"] = wrap_injection(original, condition)

        # TODO: Integrate with WASP's runner
        # This requires:
        # 1. VisualWebArena Docker environments running (GitLab + Reddit)
        # 2. Running the WASP task scaffolding with the specified model
        # 3. Capturing trajectory output in the specified format
        # 4. Evaluating attack success via WASP's evaluator

        results.append({
            "task_id": task_id,
            "model": model,
            "condition": condition,
            "output_format": output_format,
            "status": "pending_implementation",
        })

    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(results, indent=2))
    print(f"\nRun manifest saved to {manifest_path}")
    print(f"Tasks queued: {len(results)}")


def main():
    parser = argparse.ArgumentParser(description="Run WASP benchmark tasks with eval-awareness injection conditions")
    parser.add_argument("--model", type=str, default="anthropic/claude-opus-4.6",
                        help="Model to use as the agent")
    parser.add_argument("--condition", type=str, default="baseline", choices=ALL_CONDITIONS,
                        help="Injection condition")
    parser.add_argument("--output-dir", type=str,
                        default="eval_awareness_experiments/data/wasp",
                        help="Output directory for trajectories")
    parser.add_argument("--wasp-path", type=str, default=None,
                        help="Path to WASP repo (default: ./wasp)")
    parser.add_argument("--output-format", type=str, default="claude",
                        choices=["webarena", "claude", "gpt_web_tools"],
                        help="WASP output format")
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="Max tasks to run")
    args = parser.parse_args()

    run_wasp_tasks(
        model=args.model,
        condition=args.condition,
        output_dir=Path(args.output_dir),
        wasp_path=Path(args.wasp_path) if args.wasp_path else None,
        output_format=args.output_format,
        max_tasks=args.max_tasks,
    )


if __name__ == "__main__":
    main()
