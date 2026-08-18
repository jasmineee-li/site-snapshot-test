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
# re-runs `uv sync` to bring the venv back to what `uv.lock` pins.
#
# Usage:
#   bash scripts/runpod_setup.sh                      # default: claude/general-session-rsdA2
#   BRANCH=main bash scripts/runpod_setup.sh
#   REPO_DIR=/workspace/warp bash scripts/runpod_setup.sh

set -euo pipefail

WORKSPACE=${WORKSPACE:-/workspace}
REPO_DIR=${REPO_DIR:-$WORKSPACE/warp}
REPO_URL=${REPO_URL:-https://github.com/jasmineee-li/warp.git}
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
# `uv sync --locked` installs exactly what `uv.lock` pins and fails if the lock
# is stale, so this host gets the reviewed resolution. The
# `uv pip install -e '.[cua]'` it replaces could not: uv's pip interface never
# reads `uv.lock`, only the project interface does. What that cost is
# measurable -- on 2026-08-18, against an empty cp312 environment,
#
#   uv pip install -e '.[cua]' --dry-run --python-version 3.12 \
#       --python-platform x86_64-unknown-linux-gnu --python <empty venv>
#
# resolved 358 packages, among them browsergym 0.14.3, transformers 4.57.6 and
# numpy 2.3.5 where the lock pins 0.14.2, 4.57.5 and 2.2.6. That set tracked
# PyPI rather than this repo, and it changed under the host on every re-run.
#
# The drift is a set difference, not only a version one. Comparing that resolve
# against `uv export --locked --extra cua --no-default-groups`, 34 distributions
# it installs are absent from `uv.lock` entirely: browsergym 0.14.3 adds the
# `browsergym-webarena-verified` and `browsergym-webarenalite` subpackages,
# which drag in `webarena-verified`, `geopy`, `usaddress`, `thefuzz` and a
# further tail. Nothing in this tree imports any of them -- canonical WebArena
# Verified scoring installs as a separate Taskgen adapter (`README.md`), and
# this host trains probes and serves vLLM -- so the lock is not missing
# anything the GPU host uses. Recorded because the absence is otherwise
# indistinguishable from an omission.
#
# `--no-default-groups` drops the `dev` group, which uv would otherwise install
# by default. Counted with
# `uv export --locked --extra cua [--no-default-groups] | grep -c '=='`: 326
# packages without the group against 358 with it. This host trains probes and
# serves vLLM; it does not run the lint, type and test gates.
#
# One difference worth knowing at the prompt: the root project has no
# `[build-system]`, so `uv sync` installs its dependencies but not
# `browser-sim` itself, where `uv pip install -e .` did. Every script here
# `cd`s to the repo root and runs `python -m probes.…`, which puts the root on
# `sys.path` regardless.
uv sync --locked --extra cua --no-default-groups

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

# 6. GPU pre-flight. This blocks: a driver below the floor makes the model load
# on CPU without raising, and the run then completes on the wrong device.
echo
echo "[runpod_setup] GPU pre-flight..."
bash "$(dirname "${BASH_SOURCE[0]}")/preflight_gpu.sh"

# 7. Sanity report.
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
