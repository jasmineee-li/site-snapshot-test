#!/usr/bin/env bash
# Train the paper-fidelity framing-pair DoM probe on GUI-Owl-1.5-32B-Think.
# See train_probe_framing_opencua_32b.sh for details.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export HF_HOME=${HF_HOME:-/workspace/hf_cache}

MODEL=${MODEL:-xlangai/GUI-Owl-1.5-32B-Think}
DATA=${DATA:-probes/data/framing_pairs.json}
OUTPUT=${OUTPUT:-probes/trained/gui-owl-32b-framing/}
LAYERS=${LAYERS:-all}
TOKEN_POSITION=${TOKEN_POSITION:--2}
TOP_N_LAYERS=${TOP_N_LAYERS:-6}
EVAL_SPLIT=${EVAL_SPLIT:-0.25}

echo "[train_probe_framing] model=$MODEL data=$DATA output=$OUTPUT"

if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

exec python -m probes.train_probe \
    --model "$MODEL" \
    --dataset-format framing \
    --data "$DATA" \
    --output "$OUTPUT" \
    --layers "$LAYERS" \
    --token-position "$TOKEN_POSITION" \
    --top-n-layers "$TOP_N_LAYERS" \
    --eval-split "$EVAL_SPLIT" \
    --dtype bfloat16
