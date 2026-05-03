#!/usr/bin/env bash
# Run the 4-cell causal-effect matrix for one (model, benchmark) pair:
#
#                   No primer       Primer (eval_safety_primer)
#   No steering     baseline        primed
#   Steer away     [held — see notes below]
#
# This runner covers the no-steering row end-to-end via the existing
# `run_safety_pipeline.py`. The steered cells require running
# benchmarks under HF generation with the steering hook active, which
# means *not* going through the vLLM serve path. That's intentionally
# out of scope for this script; once probe-training has produced
# vectors and `score_probe.py` has tuned alpha, run a separate steered
# pass through a HF generation harness (see TODO at the bottom).
#
# Usage:
#   bash scripts/run_cua_eval_awareness_matrix.sh \
#       --model local/opencua-32b \
#       --benchmark agentdojo \
#       --max-tasks 30
#
# Pre-reqs:
#   1. Local vLLM server already running (e.g. bash scripts/serve_opencua_32b.sh).
#   2. (For steered cells) Trained probe at probes/trained/<short-id>/.
#   3. .env with judge-model API keys + OPENROUTER_API_KEY for the judges.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

MODEL=""
BENCHMARK=""
MAX_TASKS=20
JUDGE_MODEL="anthropic/claude-opus-4.6:thinking"
EXTRA_PIPELINE_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --benchmark) BENCHMARK="$2"; shift 2 ;;
        --max-tasks) MAX_TASKS="$2"; shift 2 ;;
        --judge-model) JUDGE_MODEL="$2"; shift 2 ;;
        --) shift; EXTRA_PIPELINE_ARGS+=("$@"); break ;;
        *) EXTRA_PIPELINE_ARGS+=("$1"); shift ;;
    esac
done

[[ -z $MODEL ]] && { echo "ERROR: --model required (e.g. local/opencua-32b)"; exit 2; }
[[ -z $BENCHMARK ]] && { echo "ERROR: --benchmark required (agentdojo|injecagent|wasp|doomarena)"; exit 2; }

echo "=== matrix: model=$MODEL benchmark=$BENCHMARK max_tasks=$MAX_TASKS judge=$JUDGE_MODEL ==="

run_cell() {
    local cell_label="$1"
    local preset="$2"
    echo
    echo "--- cell: $cell_label (preset=$preset) ---"
    python -m eval_awareness_experiments.run_safety_pipeline \
        --benchmark "$BENCHMARK" \
        --model-name "$MODEL" \
        --condition baseline \
        --extra-instructions-preset "$preset" \
        --judge-model "$JUDGE_MODEL" \
        --tasks-per-split "$MAX_TASKS" \
        "${EXTRA_PIPELINE_ARGS[@]}"
}

# Cell 1: baseline (no primer, no steering) — reuses existing pipeline.
run_cell "baseline" "none"

# Cell 2: primed (eval-awareness primer, no steering).
run_cell "primed" "eval_safety_primer"

cat <<NOTE

=== Steered cells ===

The two steered cells (baseline-steered, primed-steered) require running
benchmarks under HF generation with probes.steering.steering_hook
installed on the trained probe's best layer. That is not driven through
run_safety_pipeline; see probes/README.md "Causal-effect experiment
matrix" for the planned harness.

Required artifacts before launching steered cells:
  - probes/trained/${MODEL##local/}/normalized_vectors/layer_*.pt
  - tuned alpha (e.g. via probes.score_probe on the baseline trajectories)

The two no-steering cells produced trajectories under:
  results/{toolcalling,browsergym}/<benchmark>/.../{baseline,eval_safety_primer}_*

NOTE
