#!/usr/bin/env bash
# Train both DoM probes (MCQ + 16-prompt framing) on OpenCUA-32B with
# chat-template tokenization, picking each probe's default layer + threshold
# via SAD stages_oversight (paper-fidelity).
#
# Outputs (under probes/trained/opencua-32b-chat-template-span/):
#   mcq/                                          # MCQ probe vectors
#   framing/                                      # framing probe vectors
#   sad_stages_oversight_eval.json                # MCQ probe scored on SAD
#   sad_stages_oversight_eval_framing.json        # framing probe scored on SAD
#   summary.json                                  # cross-validation + selections
#
# Both probe directories' probe_info.json will record `top_layers` from the
# SAD ranking, so launch_opencua32_wasp_full_attacked_with_probes.sh
# (STEERING_LAYERS=auto) will pick those without manual editing.
#
# Pre-download (one-shot):
#   HF_HOME=${HF_HOME:-/local_data/group_dir/huggingface} \
#       .venv/bin/python -c "from huggingface_hub import snapshot_download; \
#                            snapshot_download('xlangai/OpenCUA-32B')"
#
# Requires the SAD stages/oversight structs.zip — see
# eval_awareness_experiments/HANDOFF_NODE3_TRANSFER_2026-05-05.md.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export HF_HOME=${HF_HOME:-/local_data/group_dir/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-/local_data/group_dir/huggingface/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/local_data/group_dir/huggingface/hub}

MODEL=${MODEL:-xlangai/OpenCUA-32B}
OUTPUT_ROOT=${OUTPUT_ROOT:-probes/trained/opencua-32b-chat-template-span}
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

echo "[train_probes_opencua_32b_chat_template]"
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
