#!/usr/bin/env bash
# Launch one preset of glm-5:thinking on WASP n=100. Used by tmux windows.
# Args: $1 = preset (scratchpad_bare | scratchpad_private | none | scratchpad | ...)
set -euo pipefail

PRESET="${1:?usage: wasp_n100_run.sh <preset>}"
TASK_DIR=/tmp/wasp_full
REPO=/local_data/temp/max/browser-sim
SLUG="${PRESET#scratchpad_}"   # bare / private / etc
STAMP=$(date +%Y%m%d_%H%M)
LOG="$REPO/n100_glm5_${SLUG}_v2_${STAMP}.log"

cd "$REPO"
source .venv/bin/activate
set -a; source .env; set +a
export PYTHONPATH="$REPO/external_benchmarks/wasp:${PYTHONPATH:-}"

# Pin to WASP's existing dockers (the ones with planted state in them).
# `_override_webarena_env_vars` in DoomArena uses os.environ[k]=v on WA_* vars,
# but run_wasp.py's loop does setdefault on the un-prefixed vars, so pre-setting
# these here wins. Keep these in sync with eval_awareness_experiments/DOCKER_PORTS.md.
export GITLAB="http://localhost:9001"
export REDDIT="http://localhost:8080"
export DATASET=webarena_prompt_injections

echo "[$(date)] Launching preset=$PRESET log=$LOG (dockers: gitlab=:9001 reddit=:8080)"

python -m eval_awareness_experiments.run_wasp \
  --task-dir "$TASK_DIR" \
  --online-sites reddit gitlab \
  --max-tasks 100 \
  --shuffle 42 \
  --model-name "z-ai/glm-5:thinking" \
  --condition baseline \
  --extra-instructions-preset "$PRESET" \
  --max-steps 15 \
  --n-jobs 1 \
  2>&1 | tee "$LOG"

echo "[$(date)] DONE preset=$PRESET"
