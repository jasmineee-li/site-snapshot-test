#!/usr/bin/env bash
# Launch one preset of glm-5:thinking on DoomArena n=100. Mirror of
# scripts/wasp_n100_run.sh but pinned to the *duplicate* dockers
# (gitlab_doom :9002, forum_doom :8081) plus the shared shopping containers.
# Used by the parallel DoomArena Claude in tmux session `doom_n100`.
#
# Args:
#   $1 = preset (scratchpad_bare | scratchpad_private | none | scratchpad | ...)
#   $2 = single-site (reddit | gitlab | shopping | shopping_admin)  [default: reddit]
#   $3 = max-tasks                                                  [default: 100]
set -euo pipefail

PRESET="${1:?usage: doomarena_n100_run.sh <preset> [site=reddit] [max-tasks=100]}"
SITE="${2:-reddit}"
MAX_TASKS="${3:-100}"

REPO=/local_data/temp/max/browser-sim
SLUG="${PRESET#scratchpad_}"
STAMP=$(date +%Y%m%d_%H%M)
LOG="$REPO/n${MAX_TASKS}_glm5_doom_${SITE}_${SLUG}_v2_${STAMP}.log"

cd "$REPO"
source .venv/bin/activate
set -a; source .env; set +a

# Pin to DoomArena's *duplicate* dockers (clean, separate from WASP's planted state).
# Keep in sync with eval_awareness_experiments/DOCKER_PORTS.md.
# Same trick as wasp_n100_run.sh: pre-set un-prefixed env vars so the
# loop in run_*.py's main() honors them via os.environ.setdefault.
export GITLAB="http://localhost:9002"          # gitlab_doom
export REDDIT="http://localhost:8081"          # forum_doom
export SHOPPING="http://localhost:8082"        # shared with WASP's environment
export SHOPPING_ADMIN="http://localhost:8083"  # shared
# WA_* mirrors (DoomArena's _override_webarena_env_vars overwrites these,
# so we ALSO patch via DOOMARENA_WEBARENA_BASE_URL — but the un-prefixed
# vars above are what BrowserGym actually reads, so this is belt-and-braces).
export WA_GITLAB="$GITLAB"
export WA_REDDIT="$REDDIT"
export WA_SHOPPING="$SHOPPING"
export WA_SHOPPING_ADMIN="$SHOPPING_ADMIN"

echo "[$(date)] Launching DoomArena preset=$PRESET site=$SITE log=$LOG"
echo "    dockers: gitlab=:9002 reddit=:8081 shopping=:8082 shopping_admin=:8083"

python -m eval_awareness_experiments.run_doomarena_glm5_popup \
  --single-site "$SITE" \
  --online-sites "$SITE" \
  --max-tasks "$MAX_TASKS" \
  --shuffle 42 \
  --model-name "z-ai/glm-5:thinking" \
  --extra-instructions-preset "$PRESET" \
  --max-steps 15 \
  --n-jobs 1 \
  2>&1 | tee "$LOG"

echo "[$(date)] DONE preset=$PRESET site=$SITE"
