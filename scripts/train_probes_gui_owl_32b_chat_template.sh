#!/usr/bin/env bash
# Train both DoM probes (MCQ + 16-prompt framing) on GUI-Owl-1.5-32B-Think
# with chat-template tokenization, picking each probe's default layer +
# threshold via SAD stages_oversight (paper-fidelity).
#
# Sibling of scripts/train_probes_opencua_32b_chat_template.sh — same wiring,
# different model. Outputs land under
# probes/trained/gui-owl-32b-chat-template-span/ and overwrite the prior
# vectors (the MCQ direction is deterministic; only metadata + the framing
# probe's selected layer move because we now route framing through SAD).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export HF_HOME=${HF_HOME:-/local_data/group_dir/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-/local_data/group_dir/huggingface/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/local_data/group_dir/huggingface/hub}

MODEL=${MODEL:-mPLUG/GUI-Owl-1.5-32B-Think}
OUTPUT_ROOT=${OUTPUT_ROOT:-probes/trained/gui-owl-32b-chat-template-span}
LAYERS=${LAYERS:-all}
DTYPE=${DTYPE:-bfloat16}
TOP_N_LAYERS=${TOP_N_LAYERS:-6}
SAD_STRUCTS_ZIP=${SAD_STRUCTS_ZIP:-/tmp/sad/sad/stages/oversight/structs.zip}
SAD_DEV_FRACTION=${SAD_DEV_FRACTION:-0.5}
MCQ_EVAL_SPLIT=${MCQ_EVAL_SPLIT:-0.0}
FRAMING_EVAL_SPLIT=${FRAMING_EVAL_SPLIT:-0.25}
FRAMING_TOKEN_POSITION=${FRAMING_TOKEN_POSITION:--2}

if [[ ! -f "$SAD_STRUCTS_ZIP" ]]; then
    echo "ERROR: SAD structs.zip not found at $SAD_STRUCTS_ZIP" >&2
    echo "  override with SAD_STRUCTS_ZIP=/path/to/structs.zip" >&2
    exit 1
fi

echo "[train_probes_gui_owl_32b_chat_template]"
echo "  model=$MODEL"
echo "  output_root=$OUTPUT_ROOT"
echo "  layers=$LAYERS dtype=$DTYPE top_n_layers=$TOP_N_LAYERS"
echo "  sad_structs_zip=$SAD_STRUCTS_ZIP sad_dev_fraction=$SAD_DEV_FRACTION"
echo "  mcq_eval_split=$MCQ_EVAL_SPLIT"
echo "  framing_eval_split=$FRAMING_EVAL_SPLIT framing_token_position=$FRAMING_TOKEN_POSITION"

if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

exec python probes/chat_template_experiment.py \
    --model "$MODEL" \
    --output-root "$OUTPUT_ROOT" \
    --layers "$LAYERS" \
    --dtype "$DTYPE" \
    --top-n-layers "$TOP_N_LAYERS" \
    --mcq-layer-selection sad \
    --framing-layer-selection sad \
    --mcq-eval-split "$MCQ_EVAL_SPLIT" \
    --framing-eval-split "$FRAMING_EVAL_SPLIT" \
    --framing-token-position "$FRAMING_TOKEN_POSITION" \
    --sad-structs-zip "$SAD_STRUCTS_ZIP" \
    --sad-dev-fraction "$SAD_DEV_FRACTION"
