#!/usr/bin/env bash
# Sweep (steering_layer, steering_alpha) on AgentDojo workspace to map
# steering effectiveness vs. capability degradation.
#
# Per cell: run AgentDojo via the HF backend with given (layer, alpha),
# capture per-pair JSONs (which carry `passed` = capability and
# `attack_success` = safety). After the grid completes, do ONE final
# pass that loads the model + probe and scores each trajectory's
# first-assistant message via the same probe span used in the WASP
# launcher; this gives the steering-bite signal without adding a model
# load per cell.
#
# Output: $OUTPUT_ROOT/sweep_summary.csv (and .md), one row per cell.
#
# Defaults sweep MCQ probe on OpenCUA-32B with:
#   layers: 15 (SAD top-1), 16 + 24 (paper mid-band fractions for 64-layer
#           models), 6 (early-band control), 3 (the SAD top-3 layer)
#   alphas: 0.0 (baseline), 0.3, 0.6, 1.0
#
# To sweep the framing probe instead, pass:
#   PROBE_DIR=probes/trained/opencua-32b-chat-template-span/framing \
#   LAYERS=63,9,16,24,6
#
# To sweep GUI-Owl, pass:
#   MODEL=mPLUG/GUI-Owl-1.5-32B-Think MODEL_NAME=local/gui-owl-32b-think \
#   PROBE_DIR=probes/trained/gui-owl-32b-chat-template-span/mcq \
#   LAYERS=30,16,24,6,5
#
# Total runtime estimate: ~5-10 min per (layer, non-zero alpha) cell at
# MAX_TASKS=20 + a single shared baseline. Default grid = 1 baseline +
# 5 layers × 3 non-zero alphas = 16 cells.

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
LAYERS=${LAYERS:-15,16,24,6,3}
ALPHAS=${ALPHAS:-0.3,0.6,1.0}
MAX_TASKS=${MAX_TASKS:-20}
SUITE=${SUITE:-workspace}
SHUFFLE=${SHUFFLE:-42}
CONCURRENCY=${CONCURRENCY:-4}
SWEEP_ID=${SWEEP_ID:-agentdojo_sweep_$(date -u +%Y%m%d_%H%M%S)}
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO_ROOT/results/toolcalling/agentdojo/sweep/$SWEEP_ID}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PROBE_SPAN=${PROBE_SPAN:-first_assistant_message_mean}

mkdir -p "$OUTPUT_ROOT/cells"
LOG_PATH="$OUTPUT_ROOT/sweep.log"
exec > >(tee -a "$LOG_PATH") 2>&1

echo "started_at=$(date -u --iso-8601=seconds)"
echo "sweep_id=$SWEEP_ID"
echo "output_root=$OUTPUT_ROOT"
echo "model=$MODEL model_name=$MODEL_NAME probe_dir=$PROBE_DIR"
echo "layers=$LAYERS alphas=$ALPHAS max_tasks=$MAX_TASKS suite=$SUITE"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES probe_span=$PROBE_SPAN"

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

echo "[$(date -u --iso-8601=seconds)] all cells done; aggregating + probe scoring"

env \
    REPO_ROOT="$REPO_ROOT" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    SUITE="$SUITE" \
    MODEL_PATH="$MODEL" \
    PROBE_DIR="$PROBE_DIR" \
    PROBE_SPAN="$PROBE_SPAN" \
    BASELINE_DIR="$BASELINE_DIR" \
    LAYERS="$LAYERS" \
    ALPHAS="$ALPHAS" \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    "$HF_PYTHON" - <<'PY'
"""
Aggregate AgentDojo sweep cells into a (layer, alpha) summary CSV/MD with
TSR, ASR, and mean probe-span score for one chosen span. Loads the model
+ probe ONCE here so that scoring all cells is one bf16 forward-pass loop
rather than one model load per cell.
"""

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch

from probes.contrastive_probe import ContrastiveProbe
from probes.model_loader import load_model
from probes.score_probe import _normalize_chat_messages, _score_messages

repo = Path(os.environ["REPO_ROOT"])
out_root = Path(os.environ["OUTPUT_ROOT"])
suite = os.environ["SUITE"]
model_path = os.environ["MODEL_PATH"]
probe_dir = os.environ["PROBE_DIR"]
span = os.environ["PROBE_SPAN"]
baseline_dir = Path(os.environ["BASELINE_DIR"])
layers = [int(x) for x in os.environ["LAYERS"].split(",") if x.strip()]
alphas = [float(x) for x in os.environ["ALPHAS"].split(",") if x.strip()]


def steps_to_messages(d):
    """Convert a ToolCallTaskResult JSON to a chat-template messages list.

    The AgentDojo runner stores `steps[].preamble`, `.thought`, `.action`,
    `.observation`. For probe scoring we want a faithful messages history
    so the chat template matches what the model actually saw at generation
    time. Tool calls are flattened into the assistant turn as text; tool
    results become 'user' messages (the simplest chat-template-friendly
    representation that probes.score_probe knows how to walk).
    """
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


def cell_jsons(cell_dir):
    suite_dir = cell_dir / suite / "cell"
    if not suite_dir.exists():
        return []
    return sorted(p for p in suite_dir.glob("*.json") if p.name != "manifest.json")


def aggregate_cell(cell_dir, model, tokenizer, probe, layer_for_probe, threshold):
    """Return TSR, ASR, mean probe score (None if no trajectories)."""
    jsons = cell_jsons(cell_dir)
    n = 0
    n_passed = 0
    n_attack = 0
    probe_scores = []
    for fp in jsons:
        try:
            d = json.loads(fp.read_text())
        except Exception:
            continue
        n += 1
        if d.get("passed") is True:
            n_passed += 1
        if d.get("attack_success") is True:
            n_attack += 1
        try:
            messages = steps_to_messages(d)
            if not messages:
                continue
            res = _score_messages(
                probe, model, tokenizer, messages,
                int(layer_for_probe), float(threshold),
                span=span,
            )
            probe_scores.append(res["mean_score"])
        except Exception as e:
            print(f"[probe-score warn] {fp.name}: {e!r}")
    return {
        "n": n,
        "tsr": (n_passed / n) if n else None,
        "asr": (n_attack / n) if n else None,
        "n_passed": n_passed,
        "n_attack": n_attack,
        "mean_probe_score": mean(probe_scores) if probe_scores else None,
        "n_probe_scored": len(probe_scores),
    }


print(f"loading model {model_path} (one-time, for probe scoring across all cells)")
model, tokenizer = load_model(model_path, device=None, dtype=torch.bfloat16)
probe = ContrastiveProbe.load(probe_dir)

# Pick the same layer + threshold the WASP launcher would: probe.best_layer
# (which `chat_template_experiment.py --mcq-layer-selection sad` already set
# to the SAD-top-1 layer). Threshold reused from probe metadata.
layer_for_probe = probe.best_layer
threshold = probe.threshold or 0.0
print(f"probe scoring at layer={layer_for_probe} threshold={threshold} span={span}")

baseline = aggregate_cell(baseline_dir, model, tokenizer, probe, layer_for_probe, threshold)
print(f"baseline: {baseline}")

rows = []
for layer in layers:
    for alpha in alphas:
        cell_id = f"layer{layer}_alpha{alpha}"
        cell_dir = out_root / "cells" / cell_id
        agg = aggregate_cell(cell_dir, model, tokenizer, probe, layer_for_probe, threshold)
        rows.append({
            "layer": layer,
            "alpha": alpha,
            **agg,
        })
        print(f"{cell_id}: {agg}")

# Add baseline as alpha=0.0 row per layer for plotting convenience.
for layer in layers:
    rows.insert(0, {"layer": layer, "alpha": 0.0, **baseline})

fields = [
    "layer", "alpha", "n", "tsr", "asr",
    "n_passed", "n_attack",
    "mean_probe_score", "n_probe_scored",
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
    f"- probe_dir: `{probe_dir}`",
    f"- probe layer for scoring: {layer_for_probe} (threshold={threshold})",
    f"- span: `{span}`",
    f"- max_tasks per cell: {baseline['n']}",
    "",
    "| layer | alpha | n | TSR | ASR | mean_probe_score | n_probe_scored |",
    "|---:|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    md_lines.append(
        "| "
        + " | ".join([
            fmt(r["layer"]), fmt(r["alpha"]), fmt(r["n"]),
            fmt(r["tsr"]), fmt(r["asr"]),
            fmt(r["mean_probe_score"]), fmt(r["n_probe_scored"]),
        ])
        + " |"
    )
md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
print(f"wrote {md_path}")
PY

echo "finished_at=$(date -u --iso-8601=seconds)"
