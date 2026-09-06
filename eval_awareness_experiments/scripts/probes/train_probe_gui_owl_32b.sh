#!/usr/bin/env bash
# Train a DoM eval-awareness probe for GUI-Owl-1.5-32B-Think.
# See eval_awareness_experiments/scripts/probes/train_probe_opencua_32b.sh for env knobs.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

export HF_HOME=${HF_HOME:-/workspace/hf_cache}

MODEL=${MODEL:-mPLUG/GUI-Owl-1.5-32B-Think}
DATA=${DATA:-probes/data/contrastive_dataset.json}
OUTPUT=${OUTPUT:-probes/trained/gui-owl-32b-think/}
LAYERS=${LAYERS:-all}

echo "[train_probe_gui_owl_32b] model=$MODEL data=$DATA output=$OUTPUT layers=$LAYERS"

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
