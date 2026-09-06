#!/usr/bin/env bash
# Run the 4-cell causal-effect matrix for one (model, benchmark) pair:
#
#                   No primer       Primer (eval_safety_primer)
#   No steering     baseline        primed
#   Steer away      baseline_neg    primed_neg
#
# Cells:
#   1. baseline      — no primer, no steering. Routes via vLLM (fast).
#   2. primed        — primer in system prompt, no steering. vLLM.
#   3. baseline_neg  — no primer, steer α=ALPHA on layer probe.best_layer.
#                      Requires HF generation (vLLM can't take hooks). Slow.
#   4. primed_neg    — primer + steer α=ALPHA. Slow.
#
# Tool-calling benchmarks only (agentdojo, injecagent). The browser
# benchmarks (wasp, doomarena) lack an HF-hook harness in AgentLab, so
# steered cells are not implemented for them. Pass --benchmark wasp /
# doomarena to skip cells 3 + 4 automatically.
#
# Usage:
#   bash eval_awareness_experiments/scripts/probes/run_cua_eval_awareness_matrix.sh \
#       --model local/opencua-32b \
#       --benchmark agentdojo \
#       --probe-dir probes/trained/opencua-32b/ \
#       --steering-alpha -3.0 \
#       --max-tasks 30
#
# Pre-reqs:
#   1. vLLM server up for cells 1 + 2 (e.g. bash eval_awareness_experiments/scripts/serving/serve_opencua_32b.sh).
#   2. Trained probe at --probe-dir for cells 3 + 4
#      (bash eval_awareness_experiments/scripts/probes/train_probe_<model>.sh).
#   3. .env with OPENROUTER_API_KEY for the judges.
#   4. For cells 3 + 4: at least one H100 free for HF model loading.
#      (Can be the same GPU vLLM uses if you stop vLLM between rows; or
#      a separate GPU if you have one.)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

MODEL=""
BENCHMARK=""
MAX_TASKS=20
JUDGE_MODEL="anthropic/claude-opus-4.6:thinking"
PROBE_DIR=""
STEERING_ALPHA="-3.0"
STEERING_LAYER=""           # empty → probe.best_layer
SKIP_BASELINE=0
SKIP_STEERED=0
EXTRA_PIPELINE_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --benchmark) BENCHMARK="$2"; shift 2 ;;
        --max-tasks) MAX_TASKS="$2"; shift 2 ;;
        --judge-model) JUDGE_MODEL="$2"; shift 2 ;;
        --probe-dir) PROBE_DIR="$2"; shift 2 ;;
        --steering-alpha) STEERING_ALPHA="$2"; shift 2 ;;
        --steering-layer) STEERING_LAYER="$2"; shift 2 ;;
        --skip-baseline) SKIP_BASELINE=1; shift ;;
        --skip-steered) SKIP_STEERED=1; shift ;;
        --) shift; EXTRA_PIPELINE_ARGS+=("$@"); break ;;
        *) EXTRA_PIPELINE_ARGS+=("$1"); shift ;;
    esac
done

[[ -z $MODEL ]] && { echo "ERROR: --model required (e.g. local/opencua-32b)"; exit 2; }
[[ -z $BENCHMARK ]] && { echo "ERROR: --benchmark required (agentdojo|injecagent|wasp|doomarena)"; exit 2; }

# Steered cells are only wired for the tool-calling benchmarks. Auto-skip
# them for the browser benchmarks so callers can use the same matrix
# command across all four benchmarks.
case "$BENCHMARK" in
    wasp|doomarena)
        if [[ $SKIP_STEERED -eq 0 ]]; then
            echo "[matrix] benchmark=$BENCHMARK has no HF-hook harness — skipping steered cells."
            SKIP_STEERED=1
        fi
        ;;
esac

echo "=== matrix: model=$MODEL benchmark=$BENCHMARK max_tasks=$MAX_TASKS judge=$JUDGE_MODEL ==="
echo "    probe_dir=${PROBE_DIR:-<none>} steering_alpha=$STEERING_ALPHA steering_layer=${STEERING_LAYER:-best}"

run_cell() {
    local cell_label="$1"
    local preset="$2"
    local alpha="$3"           # 0 for no-steering, e.g. -3 for steer-away
    echo
    echo "--- cell: $cell_label (preset=$preset alpha=$alpha) ---"

    local steer_flags=()
    if [[ "$alpha" != "0" && "$alpha" != "0.0" ]]; then
        if [[ -z $PROBE_DIR ]]; then
            echo "ERROR: --probe-dir required for steered cell $cell_label"
            exit 2
        fi
        steer_flags+=("--probe-dir" "$PROBE_DIR" "--steering-alpha" "$alpha")
        if [[ -n $STEERING_LAYER ]]; then
            steer_flags+=("--steering-layer" "$STEERING_LAYER")
        fi
    fi

    python -m eval_awareness_experiments.run_safety_pipeline \
        --benchmark "$BENCHMARK" \
        --model-name "$MODEL" \
        --condition baseline \
        --extra-instructions-preset "$preset" \
        --judge-model "$JUDGE_MODEL" \
        --tasks-per-split "$MAX_TASKS" \
        "${steer_flags[@]}" \
        "${EXTRA_PIPELINE_ARGS[@]}"
}

if [[ $SKIP_BASELINE -eq 0 ]]; then
    # Cell 1: baseline (no primer, no steering) — reuses existing pipeline.
    run_cell "baseline" "none" "0"
    # Cell 2: primed (eval-awareness primer, no steering).
    run_cell "primed" "eval_safety_primer" "0"
fi

if [[ $SKIP_STEERED -eq 0 ]]; then
    # Cells 3 + 4 use the HF runner; vLLM can stay up but the GPU it's
    # bound to has to have headroom (or stop vLLM first). The HF runner
    # serializes on a per-process asyncio.Lock so concurrency=1.
    run_cell "baseline_neg" "none" "$STEERING_ALPHA"
    run_cell "primed_neg" "eval_safety_primer" "$STEERING_ALPHA"
fi

echo
echo "=== matrix done ==="
echo "Trajectories under: results/toolcalling/$BENCHMARK/.../<run_name>/*.json"
echo "Run names embed condition / preset / alpha so cells don't collide."
