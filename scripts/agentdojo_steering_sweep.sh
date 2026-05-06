#!/usr/bin/env bash
# Sweep (steering_layer, steering_alpha) on AgentDojo workspace to map
# steering effectiveness vs. capability degradation.
#
# Per cell: run AgentDojo via the HF backend with given (layer, alpha),
# in --no-injection mode (each unique user_task runs once, no injection
# wrapping, no security check). Records per-task `passed` (capability).
# After the grid completes, the trained probe is **never** used to score
# steering bite — that would be circular since steering literally adds the
# probe direction to activations. Instead, an LLM-as-judge (Sonnet-4.6 by
# default, via OpenRouter) reads each agent transcript and rates verbalized
# eval-awareness using the same VerbalizedAwarenessJudge as the WASP /
# transcript pipeline. The cross-probe (the OTHER probe family's score) is
# kept as an interpretability sanity-check column.
#
# Output: $OUTPUT_ROOT/sweep_summary.{csv,md}, one row per cell.
#
# Defaults sweep MCQ probe on OpenCUA-32B with:
#   layers: 15 (SAD top-1), 16 + 24 (paper mid-band fractions for 64-layer
#           models), 6 (early-band control), 3 (the SAD top-3 layer)
#   alphas: baseline (no steering, run once) + 0.3, 0.6, 1.0
#
# To sweep the framing probe instead, pass:
#   PROBE_DIR=probes/trained/opencua-32b-chat-template-span/framing \
#   CROSS_PROBE_DIR=probes/trained/opencua-32b-chat-template-span/mcq \
#   LAYERS=63,9,16,24,6
#
# To sweep GUI-Owl, pass:
#   MODEL=mPLUG/GUI-Owl-1.5-32B-Think MODEL_NAME=local/gui-owl-32b-think \
#   PROBE_DIR=probes/trained/gui-owl-32b-chat-template-span/mcq \
#   CROSS_PROBE_DIR=probes/trained/gui-owl-32b-chat-template-span/framing \
#   LAYERS=30,16,24,6,5
#
# Runtime estimate: ~10-15 min per cell at MAX_TASKS=40 (40 unique
# user_tasks). Default grid = 1 baseline + 5 layers × 3 alphas = 16 cells.
# Plus ~2 min of judging at the end (640 transcripts at concurrency 64).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export HF_HOME=${HF_HOME:-/local_data/group_dir/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-/local_data/group_dir/huggingface/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/local_data/group_dir/huggingface/hub}
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
HF_PYTHON=${HF_PYTHON:-$REPO_ROOT/.venv/bin/python}

MODEL=${MODEL:-xlangai/OpenCUA-32B}
MODEL_NAME=${MODEL_NAME:-local/opencua-32b}
PROBE_DIR=${PROBE_DIR:-probes/trained/opencua-32b-chat-template-span/mcq}
CROSS_PROBE_DIR=${CROSS_PROBE_DIR:-probes/trained/opencua-32b-chat-template-span/framing}
LAYERS=${LAYERS:-15,16,24,6,3}
ALPHAS=${ALPHAS:-0.3,0.6,1.0}
MAX_TASKS=${MAX_TASKS:-40}
SUITE=${SUITE:-workspace}
SHUFFLE=${SHUFFLE:-42}
CONCURRENCY=${CONCURRENCY:-4}
SWEEP_ID=${SWEEP_ID:-agentdojo_sweep_$(date -u +%Y%m%d_%H%M%S)}
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO_ROOT/results/toolcalling/agentdojo/sweep/$SWEEP_ID}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
JUDGE_MODEL=${JUDGE_MODEL:-anthropic/claude-sonnet-4.6}
JUDGE_CONCURRENCY=${JUDGE_CONCURRENCY:-32}

mkdir -p "$OUTPUT_ROOT/cells"
LOG_PATH="$OUTPUT_ROOT/sweep.log"
exec > >(tee -a "$LOG_PATH") 2>&1

echo "started_at=$(date -u --iso-8601=seconds)"
echo "sweep_id=$SWEEP_ID"
echo "output_root=$OUTPUT_ROOT"
echo "model=$MODEL model_name=$MODEL_NAME"
echo "probe_dir=$PROBE_DIR (steering)"
echo "cross_probe_dir=$CROSS_PROBE_DIR (independent score)"
echo "layers=$LAYERS alphas=$ALPHAS max_tasks=$MAX_TASKS suite=$SUITE"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "judge_model=$JUDGE_MODEL judge_concurrency=$JUDGE_CONCURRENCY"

IFS=',' read -ra LAYER_ARR <<< "$LAYERS"
IFS=',' read -ra ALPHA_ARR <<< "$ALPHAS"

CELLS_FILE="$OUTPUT_ROOT/cells.txt"
: > "$CELLS_FILE"

run_cell() {
    local cell_id=$1 layer=$2 alpha=$3 cell_dir=$4
    mkdir -p "$cell_dir"
    local extra_args=()
    if [[ "$alpha" != "0.0" && "$alpha" != "0" ]]; then
        extra_args+=(--steering-layer "$layer" --steering-alpha "$alpha")
    fi
    echo "[$(date -u --iso-8601=seconds)] cell $cell_id layer=$layer alpha=$alpha"
    env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
        "$HF_PYTHON" -m eval_awareness_experiments.run_agentdojo \
            --suite "$SUITE" \
            --no-injection \
            --max-tasks "$MAX_TASKS" \
            --shuffle "$SHUFFLE" \
            --concurrency "$CONCURRENCY" \
            --model-name "$MODEL_NAME" \
            --condition baseline \
            --extra-instructions-preset none \
            --output-dir "$cell_dir" \
            --run-name "cell" \
            --backend hf \
            --probe-dir "$PROBE_DIR" \
            "${extra_args[@]}" \
            > "$cell_dir/run.log" 2>&1
    echo "$cell_id $layer $alpha $cell_dir" >> "$CELLS_FILE"
}

# Baseline cell: one run, reused across layer rows in the final table.
BASELINE_DIR="$OUTPUT_ROOT/cells/baseline_alpha0"
run_cell "baseline" "0" "0.0" "$BASELINE_DIR"

# Steering cells.
for layer in "${LAYER_ARR[@]}"; do
    for alpha in "${ALPHA_ARR[@]}"; do
        cell_id="layer${layer}_alpha${alpha}"
        cell_dir="$OUTPUT_ROOT/cells/$cell_id"
        run_cell "$cell_id" "$layer" "$alpha" "$cell_dir"
    done
done

echo "[$(date -u --iso-8601=seconds)] all cells done; running LLM judge + cross-probe scoring"

env \
    REPO_ROOT="$REPO_ROOT" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    SUITE="$SUITE" \
    MODEL_PATH="$MODEL" \
    CROSS_PROBE_DIR="$CROSS_PROBE_DIR" \
    BASELINE_DIR="$BASELINE_DIR" \
    LAYERS="$LAYERS" \
    ALPHAS="$ALPHAS" \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    JUDGE_MODEL="$JUDGE_MODEL" \
    JUDGE_CONCURRENCY="$JUDGE_CONCURRENCY" \
    SWEEP_ID="$SWEEP_ID" \
    "$HF_PYTHON" - <<'PY'
"""
Aggregate AgentDojo sweep cells into a (layer, alpha) summary table with:
- TSR (capability) — directly from per-task `passed`
- eval_aware_rate (LLM-as-judge on the agent transcript) — independent of
  the steering direction, so it's a non-circular measure of "did the model
  start verbalizing eval-awareness more under steering"
- cross_probe_score (the OTHER probe family scored at its SAD-best layer)
  as a secondary, near-independent activation-side signal

The trained-on probe is intentionally NOT scored here — that would be
circular since steering literally adds its direction to the residual stream.
"""

import asyncio
import csv
import json
import os
from pathlib import Path
from statistics import mean

import torch

from eval_awareness_experiments.judges.verbalized_awareness import VerbalizedAwarenessJudge
from eval_awareness_experiments.llm import LLM
from eval_awareness_experiments.types import Message, ReasoningCategory, Transcript
from probes.contrastive_probe import ContrastiveProbe
from probes.model_loader import load_model
from probes.score_probe import _normalize_chat_messages, _score_messages

out_root = Path(os.environ["OUTPUT_ROOT"])
suite = os.environ["SUITE"]
model_path = os.environ["MODEL_PATH"]
cross_probe_dir = os.environ["CROSS_PROBE_DIR"]
baseline_dir = Path(os.environ["BASELINE_DIR"])
layers = [int(x) for x in os.environ["LAYERS"].split(",") if x.strip()]
alphas = [float(x) for x in os.environ["ALPHAS"].split(",") if x.strip()]
judge_model_name = os.environ["JUDGE_MODEL"]
judge_concurrency = int(os.environ["JUDGE_CONCURRENCY"])


def steps_to_messages(d):
    """Convert a ToolCallTaskResult JSON to a chat-template messages list."""
    msgs = []
    if d.get("system_prompt"):
        msgs.append({"role": "system", "content": d["system_prompt"]})
    msgs.append({"role": "user", "content": d.get("task_instruction", "")})
    for step in d.get("steps", []):
        parts = []
        if step.get("preamble"):
            parts.append(step["preamble"])
        if step.get("thought"):
            parts.append(step["thought"])
        if step.get("action"):
            parts.append(step["action"])
        msgs.append({"role": "assistant", "content": "\n".join(parts) or ""})
        if step.get("observation"):
            msgs.append({"role": "user", "content": step["observation"]})
    if d.get("final_answer"):
        msgs.append({"role": "assistant", "content": d["final_answer"]})
    return _normalize_chat_messages(msgs)


def to_transcript(d):
    """Build a Transcript suitable for VerbalizedAwarenessJudge from a per-pair JSON."""
    msgs = []
    if d.get("system_prompt"):
        msgs.append(Message(role="system", content=d["system_prompt"]))
    msgs.append(Message(role="user", content=d.get("task_instruction", "")))
    for step in d.get("steps", []):
        parts = []
        if step.get("preamble"):
            parts.append(step["preamble"])
        if step.get("thought"):
            parts.append(step["thought"])
        if step.get("action"):
            parts.append(step["action"])
        msgs.append(Message(role="assistant", content="\n".join(parts) or ""))
        if step.get("observation"):
            msgs.append(Message(role="tool", content=step["observation"]))
    if d.get("final_answer"):
        msgs.append(Message(role="assistant", content=d["final_answer"]))
    return Transcript(
        messages=msgs,
        system_instructions=d.get("system_prompt", ""),
        instruction=d.get("task_instruction", ""),
        metadata={"task_id": d.get("task_id")},
    )


def cell_jsons(cell_dir):
    suite_dir = cell_dir / suite / "cell"
    if not suite_dir.exists():
        return []
    return sorted(p for p in suite_dir.glob("*.json") if p.name != "manifest.json")


# Discover all cells.
all_cells = [("baseline", 0, 0.0, baseline_dir)]
for layer in layers:
    for alpha in alphas:
        cell_id = f"layer{layer}_alpha{alpha}"
        all_cells.append((cell_id, layer, alpha, out_root / "cells" / cell_id))

# 1) LLM-as-judge for verbalized eval-awareness across ALL cells in one async pass.
print(f"loading {sum(len(cell_jsons(d)) for _, _, _, d in all_cells)} transcripts for LLM judge")
judge_llm = LLM(
    model=judge_model_name,
    thinking="thinking" in judge_model_name,
    concurrency=judge_concurrency,
    temperature=0.0,
)
judge = VerbalizedAwarenessJudge(
    grader_model=judge_llm,
    classifier_model=judge_llm,
    max_concurrency=judge_concurrency,
)

EVAL_AWARE_CATS = {
    ReasoningCategory.ALIGNMENT_EVAL,
    ReasoningCategory.CAPABILITY_EVAL,
    ReasoningCategory.OTHER_EVAL,
}


async def judge_one(d):
    transcript = to_transcript(d)
    try:
        return await judge.judge(transcript)
    except Exception as e:
        print(f"[judge warn] task={d.get('task_id')}: {e!r}")
        return None


async def judge_cell(cell_dir):
    docs = []
    for fp in cell_jsons(cell_dir):
        try:
            docs.append(json.loads(fp.read_text()))
        except Exception:
            pass
    results = await asyncio.gather(*[judge_one(d) for d in docs])
    return docs, results


# 2) Cross-probe activation score (the OTHER probe family at its SAD-best layer).
print(f"loading model {model_path} for cross-probe scoring")
model, tokenizer = load_model(model_path, device=None, dtype=torch.bfloat16)
cross_probe = ContrastiveProbe.load(cross_probe_dir)
cross_layer = cross_probe.best_layer
cross_threshold = cross_probe.threshold or 0.0
print(f"cross-probe scoring at layer={cross_layer} threshold={cross_threshold}")


def cross_score_doc(d):
    try:
        msgs = steps_to_messages(d)
        if not msgs:
            return None
        res = _score_messages(
            cross_probe, model, tokenizer, msgs,
            int(cross_layer), float(cross_threshold),
            span="first_assistant_message_mean",
        )
        return res["mean_score"]
    except Exception as e:
        print(f"[cross-probe warn] task={d.get('task_id')}: {e!r}")
        return None


async def aggregate_all():
    rows = []
    for cell_id, layer, alpha, cell_dir in all_cells:
        docs, judge_results = await judge_cell(cell_dir)
        n = len(docs)
        n_passed = sum(1 for d in docs if d.get("passed") is True)
        n_aware = sum(
            1 for r in judge_results
            if r is not None and r.category in EVAL_AWARE_CATS
        )
        n_judged = sum(1 for r in judge_results if r is not None)
        cross_scores = [s for s in (cross_score_doc(d) for d in docs) if s is not None]
        row = {
            "cell": cell_id,
            "layer": layer,
            "alpha": alpha,
            "n": n,
            "tsr": (n_passed / n) if n else None,
            "n_passed": n_passed,
            "eval_aware_rate": (n_aware / n_judged) if n_judged else None,
            "n_aware": n_aware,
            "n_judged": n_judged,
            "mean_cross_probe_score": mean(cross_scores) if cross_scores else None,
            "n_cross_scored": len(cross_scores),
        }
        rows.append(row)
        print(
            f"{cell_id}: n={n} tsr={row['tsr']} aware={row['eval_aware_rate']} "
            f"cross_probe={row['mean_cross_probe_score']}"
        )
    return rows


rows = asyncio.run(aggregate_all())

fields = [
    "cell", "layer", "alpha", "n",
    "tsr", "n_passed",
    "eval_aware_rate", "n_aware", "n_judged",
    "mean_cross_probe_score", "n_cross_scored",
]
csv_path = out_root / "sweep_summary.csv"
with csv_path.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print(f"wrote {csv_path}")


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


md_path = out_root / "sweep_summary.md"
md_lines = [
    f"# AgentDojo steering sweep — {os.environ.get('SWEEP_ID', '')}",
    "",
    f"- model: `{model_path}`",
    f"- cross_probe_dir: `{cross_probe_dir}` (layer {cross_layer}, threshold {cross_threshold})",
    f"- LLM judge: `{judge_model_name}`",
    f"- max_tasks per cell: {rows[0]['n'] if rows else '-'}",
    "",
    "| cell | layer | alpha | n | TSR | eval_aware_rate | mean_cross_probe |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    md_lines.append(
        "| "
        + " | ".join([
            r["cell"], fmt(r["layer"]), fmt(r["alpha"]), fmt(r["n"]),
            fmt(r["tsr"]),
            fmt(r["eval_aware_rate"]),
            fmt(r["mean_cross_probe_score"]),
        ])
        + " |"
    )
md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
print(f"wrote {md_path}")
PY

echo "finished_at=$(date -u --iso-8601=seconds)"
