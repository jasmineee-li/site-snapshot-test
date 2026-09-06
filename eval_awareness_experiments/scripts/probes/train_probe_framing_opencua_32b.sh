#!/usr/bin/env bash
# Train the paper-fidelity framing-pair DoM probe on OpenCUA-32B.
#
# Uses the 16 real + 16 fake colon-terminated framing prompts vendored
# from Hua et al. 2025 (steering-eval-awareness-public/data/
# constrastive_prompts/user_and_simple.json), extracts at the colon
# (token position −2 of the chat-template-rendered prompt) at every
# decoder layer, and computes mean(real) − mean(fake) per layer.
#
# Output: probes/trained/opencua-32b-framing/{vectors,normalized_vectors,probe_info.json,per_layer_auroc.json}
#
# Single H100, ~10 minutes (extraction is much cheaper than the MCQ
# probe — only 32 forward passes vs 408).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

export HF_HOME=${HF_HOME:-/workspace/hf_cache}

MODEL=${MODEL:-xlangai/OpenCUA-32B}
DATA=${DATA:-probes/data/framing_pairs.json}
OUTPUT=${OUTPUT:-probes/trained/opencua-32b-framing/}
LAYERS=${LAYERS:-all}
TOKEN_POSITION=${TOKEN_POSITION:--2}
TOP_N_LAYERS=${TOP_N_LAYERS:-6}
EVAL_SPLIT=${EVAL_SPLIT:-0.25}  # 12 train / 4 eval per pool

echo "[train_probe_framing] model=$MODEL data=$DATA output=$OUTPUT"
echo "                       layers=$LAYERS token_position=$TOKEN_POSITION top_n=$TOP_N_LAYERS"

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
