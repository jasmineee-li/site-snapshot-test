#!/usr/bin/env bash
# One-shot RunPod setup for CUA × eval-awareness probes.
#
# Assumes a fresh RunPod with a network volume mounted at /workspace and
# nothing else. Clones the repo onto the volume (so it survives pod
# restarts), builds a venv with uv, installs the `[cua]` extras
# (transformers, vllm, qwen-vl-utils, accelerate, scikit-learn), and
# leaves you ready to run probe training or vLLM serving.
#
# Re-running is idempotent — skips clone/venv if already present, and
# re-runs `uv pip install` to pick up new extras.
#
# Usage:
#   bash scripts/runpod_setup.sh                      # default: claude/general-session-rsdA2
#   BRANCH=main bash scripts/runpod_setup.sh
#   REPO_DIR=/workspace/browser-sim bash scripts/runpod_setup.sh

set -euo pipefail

WORKSPACE=${WORKSPACE:-/workspace}
REPO_DIR=${REPO_DIR:-$WORKSPACE/browser-sim}
REPO_URL=${REPO_URL:-https://github.com/jasmineee-li/browser-sim.git}
BRANCH=${BRANCH:-claude/general-session-rsdA2}
HF_HOME_DEFAULT=$WORKSPACE/hf_cache

echo "[runpod_setup] workspace=$WORKSPACE repo_dir=$REPO_DIR branch=$BRANCH"

mkdir -p "$WORKSPACE" "$HF_HOME_DEFAULT"

# 1. Install uv if missing (small, no sudo needed).
if ! command -v uv >/dev/null 2>&1; then
    echo "[runpod_setup] installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # uv installs to ~/.local/bin
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

# 2. Clone or update the repo.
if [ ! -d "$REPO_DIR/.git" ]; then
    echo "[runpod_setup] cloning $REPO_URL → $REPO_DIR"
    git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
git fetch origin "$BRANCH"
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"

# 3. Build venv. Using uv keeps it consistent with the project's
# `[tool.uv.sources]` (AgentLab is an editable local dep).
if [ ! -d .venv ]; then
    echo "[runpod_setup] creating venv with uv..."
    uv venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[runpod_setup] installing project + [cua] extras..."
# `--no-build-isolation` keeps things faster on rebuild; first run will
# still fetch wheels into the uv cache.
uv pip install -e '.[cua]'

# 4. Configure persistent HF cache on the volume.
HF_HOME_LINE='export HF_HOME='"$HF_HOME_DEFAULT"
if ! grep -qF "$HF_HOME_LINE" .venv/bin/activate 2>/dev/null; then
    echo "$HF_HOME_LINE" >> .venv/bin/activate
    echo "[runpod_setup] appended HF_HOME=$HF_HOME_DEFAULT to .venv activate"
fi
export HF_HOME=$HF_HOME_DEFAULT

# 5. Optional .env passthrough — RunPod usually injects secrets via env
# vars; we mirror them into a .env so scripts that read .env see them too.
if [ ! -f .env ]; then
    {
        for var in HF_TOKEN OPENROUTER_API_KEY OPENAI_API_KEY ANTHROPIC_API_KEY; do
            if [ -n "${!var:-}" ]; then
                echo "$var=${!var}"
            fi
        done
    } > .env
fi

# 6. Sanity report.
echo
echo "[runpod_setup] DONE."
echo "  python: $(python --version)"
echo "  torch:  $(python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())')"
echo "  GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "    (nvidia-smi unavailable)"
echo
echo "Next steps:"
echo "  cd $REPO_DIR && source .venv/bin/activate"
echo "  bash scripts/runpod_train_framing_probe.sh                # trains framing probe on OpenCUA-32B"
echo "  MODEL_SHORT=gui-owl-32b bash scripts/runpod_train_framing_probe.sh"
