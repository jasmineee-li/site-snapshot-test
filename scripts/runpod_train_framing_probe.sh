#!/usr/bin/env bash
# End-to-end RunPod orchestrator: setup → pre-download → train framing probe.
#
# Idempotent — re-running just no-ops the steps that are already done
# (clone, venv, model download), then re-runs the probe training.
#
# Usage:
#   bash scripts/runpod_train_framing_probe.sh                       # opencua-32b
#   MODEL_SHORT=opencua-32b   bash scripts/runpod_train_framing_probe.sh
#   MODEL_SHORT=gui-owl-32b   bash scripts/runpod_train_framing_probe.sh

set -euo pipefail

WORKSPACE=${WORKSPACE:-/workspace}
REPO_DIR=${REPO_DIR:-$WORKSPACE/warp}
MODEL_SHORT=${MODEL_SHORT:-opencua-32b}

case "$MODEL_SHORT" in
    opencua-32b)
        MODEL_HF=xlangai/OpenCUA-32B
        TRAIN_SCRIPT=scripts/train_probe_framing_opencua_32b.sh
        ;;
    gui-owl-32b|gui-owl-32b-think)
        MODEL_HF=mPLUG/GUI-Owl-1.5-32B-Think
        TRAIN_SCRIPT=scripts/train_probe_framing_gui_owl_32b.sh
        ;;
    *)
        echo "Unknown MODEL_SHORT='$MODEL_SHORT'. Use opencua-32b or gui-owl-32b." >&2
        exit 1
        ;;
esac

# Step 1: setup (clone repo + build venv + install [cua]).
if [ ! -d "$REPO_DIR/.git" ] || [ ! -d "$REPO_DIR/.venv" ]; then
    echo "[orchestrator] running runpod_setup.sh..."
    if [ -f "$(dirname "$0")/runpod_setup.sh" ]; then
        bash "$(dirname "$0")/runpod_setup.sh"
    else
        # If we're being invoked from outside the repo (e.g. via curl),
        # bootstrap by cloning and re-execing the in-repo orchestrator.
        echo "[orchestrator] runpod_setup.sh not found alongside this script;"
        echo "                bootstrapping via clone-and-reexec."
        REPO_URL=${REPO_URL:-https://github.com/jasmineee-li/warp.git}
        BRANCH=${BRANCH:-claude/general-session-rsdA2}
        mkdir -p "$WORKSPACE"
        if [ ! -d "$REPO_DIR/.git" ]; then
            git clone "$REPO_URL" "$REPO_DIR"
        fi
        cd "$REPO_DIR"
        git fetch origin "$BRANCH" && git checkout "$BRANCH" && git pull --ff-only origin "$BRANCH"
        bash scripts/runpod_setup.sh
        exec env MODEL_SHORT="$MODEL_SHORT" bash scripts/runpod_train_framing_probe.sh
    fi
fi

cd "$REPO_DIR"
# shellcheck disable=SC1091
source .venv/bin/activate
export HF_HOME=${HF_HOME:-$WORKSPACE/hf_cache}

# Step 2: pre-download the model. Avoids race if you later run multiple
# things that share the same base.
echo "[orchestrator] pre-downloading $MODEL_HF into $HF_HOME..."
python - <<PY
import os
from huggingface_hub import snapshot_download
snapshot_download("$MODEL_HF", token=os.environ.get("HF_TOKEN") or None)
PY

# Step 3: train.
echo "[orchestrator] launching $TRAIN_SCRIPT..."
exec bash "$TRAIN_SCRIPT"
