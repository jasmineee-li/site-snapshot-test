#!/usr/bin/env bash
# Train a DoM eval-awareness probe for OpenCUA-32B over all decoder layers.
#
# Loads the model on a single H100 in bf16, extracts contrastive
# activations for the 204 vendored pairs, computes mean(test) -
# mean(deploy) per layer, evaluates per-layer AUROC on a 20% held-out
# split, and saves vectors + per_layer_auroc.json under
# probes/trained/opencua-32b/.
#
# Per the eval-awareness CLAUDE.md guidance:
#   - HF_HOME=/workspace/hf_cache (root partition is tiny)
#   - Pre-download the base before parallel jobs to avoid race
#
# Pre-download (one-shot):
#   HF_HOME=/workspace/hf_cache python -c \
#     "from huggingface_hub import snapshot_download; snapshot_download('xlangai/OpenCUA-32B')"

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export HF_HOME=${HF_HOME:-/workspace/hf_cache}

MODEL=${MODEL:-xlangai/OpenCUA-32B}
DATA=${DATA:-probes/data/contrastive_dataset.json}
OUTPUT=${OUTPUT:-probes/trained/opencua-32b/}
LAYERS=${LAYERS:-all}

echo "[train_probe_opencua_32b] model=$MODEL data=$DATA output=$OUTPUT layers=$LAYERS"

# Activate the project venv (uv-managed). Falls back to whatever python is on PATH.
if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

exec python -m probes.train_probe \
    --model "$MODEL" \
    --data "$DATA" \
    --output "$OUTPUT" \
    --layers "$LAYERS" \
    --dtype bfloat16
