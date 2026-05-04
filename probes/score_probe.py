"""Score transcripts (or arbitrary text) with a saved DoM probe.

Used for two things:
  1. Diagnostic — measure whether OpenCUA / GUI-Owl are naturally
     eval-aware by scoring their baseline benchmark trajectories.
  2. Steering-strength tuning — score text under different `--alpha`
     values to find the alpha that brings mean eval-awareness score
     close to baseline-deploy.

Trajectory inputs are flat JSONL (one transcript per line) or the
ToolCallingRunner JSON-per-task format under
`results/toolcalling/{benchmark}/.../<task>.json`. For the latter we
join `steps[*].thought / preamble / action / observation` into a
single text blob per task, then score it.

Usage:
    # Score a single string:
    python -m probes.score_probe \\
        --model xlangai/OpenCUA-32B \\
        --probe-dir probes/trained/opencua-32b/ \\
        --text "I think this might be an evaluation."

    # Score a directory of ToolCallingRunner JSON outputs:
    python -m probes.score_probe \\
        --model xlangai/OpenCUA-32B \\
        --probe-dir probes/trained/opencua-32b/ \\
        --transcripts-dir results/toolcalling/agentdojo/baseline_none_xlangai_OpenCUA-32B/ \\
        --output scored_transcripts.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.contrastive_probe import ContrastiveProbe
from probes.model_loader import load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _flatten_toolcalling_task(task_json: dict) -> str:
    """Join ToolCallingRunner steps into a single transcript blob."""
    parts: list[str] = []
    if task_json.get("system_prompt"):
        parts.append(f"[system]\n{task_json['system_prompt']}")
    parts.append(f"[user]\n{task_json.get('task_instruction', '')}")
    for i, step in enumerate(task_json.get("steps", [])):
        if step.get("thought"):
            parts.append(f"[assistant.reasoning {i}]\n{step['thought']}")
        if step.get("preamble"):
            parts.append(f"[assistant.preamble {i}]\n{step['preamble']}")
        if step.get("action"):
            parts.append(f"[assistant.action {i}]\n{step['action']}")
        if step.get("observation"):
            parts.append(f"[tool.observation {i}]\n{step['observation']}")
    if task_json.get("final_answer"):
        parts.append(f"[assistant.final]\n{task_json['final_answer']}")
    return "\n\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Score text with a saved DoM probe")
    parser.add_argument("--model", required=True, help="HuggingFace model path")
    parser.add_argument("--probe-dir", required=True, help="Saved probe directory")
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Layer to score at (default: probe.best_layer)",
    )
    parser.add_argument("--text", default=None, help="Single text to score")
    parser.add_argument(
        "--transcripts-dir",
        default=None,
        help="Directory of ToolCallingRunner JSON-per-task outputs",
    )
    parser.add_argument("--output", default=None, help="Optional JSONL output path")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    if args.text is None and args.transcripts_dir is None:
        raise SystemExit("Provide either --text or --transcripts-dir")

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(args.dtype, torch.bfloat16)

    logger.info(f"Loading model: {args.model}")
    model, tokenizer = load_model(args.model, device=args.device, dtype=dtype)

    probe = ContrastiveProbe.load(args.probe_dir)
    layer = args.layer if args.layer is not None else probe.best_layer
    logger.info(f"Scoring at layer {layer}")

    out_lines: list[str] = []

    if args.text is not None:
        result = probe.score(model, args.text, layer=layer, tokenizer=tokenizer)
        line = json.dumps({
            "id": "text",
            "mean_score": result.mean_score,
            "max_score": result.max_score,
            "min_score": result.min_score,
            "std_score": result.std_score,
            "classification": result.classification,
            "confidence": result.confidence,
            "layer": layer,
        })
        print(line)
        out_lines.append(line)

    if args.transcripts_dir is not None:
        td = Path(args.transcripts_dir)
        for task_path in sorted(td.glob("*.json")):
            try:
                task_json = json.loads(task_path.read_text())
            except json.JSONDecodeError:
                logger.warning(f"Skipping non-JSON {task_path}")
                continue
            text = _flatten_toolcalling_task(task_json)
            try:
                result = probe.score(model, text, layer=layer, tokenizer=tokenizer)
            except Exception as e:
                logger.warning(f"Failed to score {task_path.name}: {e}")
                continue
            line = json.dumps({
                "id": task_json.get("task_id", task_path.stem),
                "benchmark": task_json.get("benchmark"),
                "agent": task_json.get("agent"),
                "condition": task_json.get("condition"),
                "mean_score": result.mean_score,
                "max_score": result.max_score,
                "min_score": result.min_score,
                "std_score": result.std_score,
                "classification": result.classification,
                "confidence": result.confidence,
                "layer": layer,
            })
            print(line)
            out_lines.append(line)

    if args.output:
        Path(args.output).write_text("\n".join(out_lines) + "\n")
        logger.info(f"Wrote {len(out_lines)} scored items to {args.output}")


if __name__ == "__main__":
    main()
