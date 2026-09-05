#!/usr/bin/env bash
# Plant WASP task pools into every WASP-only per-model docker stack.
#
# By default this runs sequentially to avoid hammering the host and GitLab
# bootstraps. Set PARALLEL=1 to plant all selected stacks concurrently.
#
# Usage:
#   ./eval_awareness_experiments/scripts/wasp/plant_per_model_stacks.sh
#   MODEL_STACK_FILTER="gpt gemini25 kimi25" ./eval_awareness_experiments/scripts/wasp/plant_per_model_stacks.sh
#   PARALLEL=1 ./eval_awareness_experiments/scripts/wasp/plant_per_model_stacks.sh

set -euo pipefail

cd "$(dirname "$0")/../../.."

MODEL_STACK_FILTER="${MODEL_STACK_FILTER:-all}"
PARALLEL="${PARALLEL:-0}"
LOG_DIR="${LOG_DIR:-logs/wasp_plant_per_model}"

STACKS=(
    "glm       9201 8201"
    "sonnet    9211 8211"
    "opus      9221 8221"
    "gpt       9231 8231"
    "gemini25  9241 8241"
    "kimi25    9251 8251"
)

contains_word() {
    local needle=$1 haystack=$2
    if [ "$haystack" = "all" ]; then
        return 0
    fi
    for word in $haystack; do
        if [ "$word" = "$needle" ]; then
            return 0
        fi
    done
    return 1
}

mkdir -p "$LOG_DIR"

PIDS=()
LABELS=()
for stack in "${STACKS[@]}"; do
    read -r slug gitlab_port reddit_port <<< "$stack"
    if ! contains_word "$slug" "$MODEL_STACK_FILTER"; then
        continue
    fi

    log="$LOG_DIR/${slug}.log"
    echo "[$slug] plant gitlab=:$gitlab_port reddit=:$reddit_port -> $log"

    if [ "$PARALLEL" = "1" ]; then
        ./eval_awareness_experiments/scripts/wasp/plant_stack.sh "$slug" "$gitlab_port" "$reddit_port" \
            > "$log" 2>&1 &
        PIDS+=("$!")
        LABELS+=("$slug")
    else
        ./eval_awareness_experiments/scripts/wasp/plant_stack.sh "$slug" "$gitlab_port" "$reddit_port" \
            > "$log" 2>&1
    fi
done

overall_status=0
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    label="${LABELS[$i]}"
    if wait "$pid"; then
        echo "[$label] plant completed"
    else
        status=$?
        echo "[$label] plant FAILED with $status" >&2
        overall_status="$status"
    fi
done

exit "$overall_status"
