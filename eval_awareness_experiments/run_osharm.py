"""Run OS-harm web injection tasks with configurable model and injection conditions.

OS-harm prompt injection subset: 50 tasks (10 base tasks x 6 environmental
vectors). We focus on web-only vectors: 'website' and 'thunderbird_mail_received'.

Injection content is in each task's 'content' field. For causal experiments,
we wrap this content with the condition's XML tags.

Usage:
    # Baseline run with Opus 4.6
    python -m eval_awareness_experiments.run_osharm \
        --model anthropic/claude-opus-4.6 \
        --condition baseline

    # Causal experiment: eval_cue condition
    python -m eval_awareness_experiments.run_osharm \
        --model anthropic/claude-opus-4.6 \
        --condition eval_cue

Prerequisites:
    1. Clone OS-harm: git clone https://github.com/tml-epfl/os-harm
    2. Set up the OS-harm environment (see their README)
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

# Web-relevant injection vectors from OS-harm
WEB_VECTORS = ["website", "thunderbird_mail_received", "thunderbird_mail_draft"]


def run_osharm_tasks(
    model: str,
    condition: InjectionCondition,
    output_dir: Path,
    osharm_path: Path | None = None,
    vectors: list[str] | None = None,
    max_tasks: int | None = None,
) -> None:
    """Run OS-harm web injection tasks and save trajectories.

    OS-harm injection content lives in test_injection.json. Each task has
    'injections' array with 'type' and 'content' fields. For causal
    experiments, we wrap injection['content'] with condition tags.

    Modification point: In lib_run_single.py, right before each injection
    vector's content is inserted into the environment.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    vectors = vectors or WEB_VECTORS

    if osharm_path is None:
        osharm_path = Path("os-harm")
    if not osharm_path.exists():
        print(f"OS-harm repo not found at {osharm_path}")
        print("Clone it: git clone https://github.com/tml-epfl/os-harm")
        sys.exit(1)

    # Load test_injection.json
    injection_file = osharm_path / "test_injection.json"
    if not injection_file.exists():
        injection_file = osharm_path / "evaluation_examples" / "test_injection.json"
    if not injection_file.exists():
        print(f"test_injection.json not found in {osharm_path}")
        sys.exit(1)

    tasks = json.loads(injection_file.read_text())
    if not isinstance(tasks, list):
        tasks = tasks.get("tasks", tasks.get("examples", []))

    # Filter to web-relevant vectors
    web_tasks = []
    for task in tasks:
        injections = task.get("injections", [])
        web_injections = [inj for inj in injections if inj.get("type") in vectors]
        if web_injections:
            task_copy = {**task, "injections": web_injections}
            web_tasks.append(task_copy)

    if max_tasks:
        web_tasks = web_tasks[:max_tasks]

    print(f"Found {len(web_tasks)} web-relevant OS-harm tasks (vectors: {vectors})")
    print(f"Model: {model}")
    print(f"Condition: {condition}")
    print(f"Output: {output_dir}")

    results = []
    for task in web_tasks:
        task_id = task.get("id", task.get("task_id", "unknown"))
        print(f"\n  Running task: {task_id}")

        # Apply injection condition: wrap each injection's content
        if condition != "baseline":
            for inj in task.get("injections", []):
                if "content" in inj:
                    inj["content"] = wrap_injection(inj["content"], condition)

        # TODO: Integrate with OS-harm's runner
        # This requires:
        # 1. OS-harm environment setup (OSWorld-based VM or Docker)
        # 2. Running the agent with the specified model
        # 3. Capturing better_log.json trajectory
        # 4. Evaluating attack success via OS-harm's LLM judge

        results.append({
            "task_id": task_id,
            "model": model,
            "condition": condition,
            "vectors": [inj["type"] for inj in task.get("injections", [])],
            "status": "pending_implementation",
        })

    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(results, indent=2))
    print(f"\nRun manifest saved to {manifest_path}")
    print(f"Tasks queued: {len(results)}")


def main():
    parser = argparse.ArgumentParser(description="Run OS-harm web injection tasks with eval-awareness conditions")
    parser.add_argument("--model", type=str, default="anthropic/claude-opus-4.6",
                        help="Model to use as the agent")
    parser.add_argument("--condition", type=str, default="baseline", choices=ALL_CONDITIONS,
                        help="Injection condition")
    parser.add_argument("--output-dir", type=str,
                        default="eval_awareness_experiments/data/os-harm",
                        help="Output directory for trajectories")
    parser.add_argument("--osharm-path", type=str, default=None,
                        help="Path to OS-harm repo (default: ./os-harm)")
    parser.add_argument("--vectors", type=str, nargs="+", default=None,
                        help="Injection vectors to include (default: website, thunderbird_mail_received)")
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="Max tasks to run")
    args = parser.parse_args()

    run_osharm_tasks(
        model=args.model,
        condition=args.condition,
        output_dir=Path(args.output_dir),
        osharm_path=Path(args.osharm_path) if args.osharm_path else None,
        vectors=args.vectors,
        max_tasks=args.max_tasks,
    )


if __name__ == "__main__":
    main()
