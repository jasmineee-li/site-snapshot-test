#!/usr/bin/env bash
# Cleanup all WASP planted state from one GitLab/forum stack.
#
# Usage:
#   ./scripts/wasp_cleanup_full_stack.sh default 9001 8080
#   ./scripts/wasp_cleanup_full_stack.sh glm 9201 8201
#
# Path defaults match wasp_plant_full_stack.sh:
#   default stack: /tmp/wasp_full_plant/ and /tmp/wasp_full/
#   named stacks: /tmp/wasp_full_plant_<stack>/ and /tmp/wasp_full_<stack>/

set -euo pipefail

STACK="${STACK:-${1:-}}"
GITLAB_PORT="${GITLAB_PORT:-${2:-}}"
REDDIT_PORT="${REDDIT_PORT:-${3:-}}"

if [ -z "$STACK" ] || [ -z "$GITLAB_PORT" ] || [ -z "$REDDIT_PORT" ]; then
    echo "usage: STACK=<slug> GITLAB_PORT=<port> REDDIT_PORT=<port> $0"
    echo "   or: $0 <slug> <gitlab_port> <reddit_port>"
    exit 1
fi

REPO="${REPO:-/local_data/temp/max/browser-sim}"
WASP="${WASP:-$REPO/external_benchmarks/wasp/webarena_prompt_injections}"
if [ "$STACK" = "default" ]; then
    PLANT_ROOT="${PLANT_ROOT:-/tmp/wasp_full_plant}"
    FINAL="${FINAL:-/tmp/wasp_full}"
else
    PLANT_ROOT="${PLANT_ROOT:-/tmp/wasp_full_plant_${STACK}}"
    FINAL="${FINAL:-/tmp/wasp_full_${STACK}}"
fi
GITLAB_URL="http://localhost:${GITLAB_PORT}"
REDDIT_URL="http://localhost:${REDDIT_PORT}"

cd "$WASP"

echo "=== cleanup WASP stack=$STACK gitlab=$GITLAB_URL reddit=$REDDIT_URL ==="
echo "    plant_root=$PLANT_ROOT"
echo "    final=$FINAL"

# Cleanup the merged config first. Each entry's cleanup_fn is idempotent and
# this usually covers all planted state in one pass.
INST="$FINAL/instantiated_prompt_injections_config.json"
if [ ! -f "$INST" ]; then
    echo "no merged config at $INST -- falling back to per-subdir configs"
else
    echo "=== cleanup using merged config ($(wc -l < "$INST") lines) ==="
    GITLAB="$GITLAB_URL" REDDIT="$REDDIT_URL" \
    PYTHONPATH=. "$REPO/.venv/bin/python" environment_cleanup.py \
        --prompt-injection-config-path "$INST" \
        --gitlab-domain "$GITLAB_URL" --reddit-domain "$REDDIT_URL" \
        --delete-attacker-account 2>&1 | tail -20
fi

# Belt-and-braces: retry per-subdir configs because the merged file only
# contains entries that made it into the final task pool.
for sub in "$PLANT_ROOT"/*/; do
    inst="$sub/instantiated_prompt_injections_config.json"
    [ -f "$inst" ] || continue
    echo "=== cleanup subdir: $(basename "$sub") ==="
    GITLAB="$GITLAB_URL" REDDIT="$REDDIT_URL" \
    PYTHONPATH=. "$REPO/.venv/bin/python" environment_cleanup.py \
        --prompt-injection-config-path "$inst" \
        --gitlab-domain "$GITLAB_URL" --reddit-domain "$REDDIT_URL" 2>&1 | tail -5
done

echo "=== cleanup done for stack=$STACK ==="
