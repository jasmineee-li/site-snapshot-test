#!/usr/bin/env bash
# Plant WASP task pools into every WASP-only per-model docker stack.
#
# By default this runs sequentially to avoid hammering the host and GitLab
# bootstraps. Set PARALLEL=1 to plant all selected stacks concurrently.
#
# Usage:
#   ./scripts/wasp_plant_per_model_dockers.sh
#   MODEL_STACK_FILTER="gpt gemini25 kimi25" ./scripts/wasp_plant_per_model_dockers.sh
#   MODEL_STACK_FILTER="w01 w02" ./scripts/wasp_plant_per_model_dockers.sh
#   PARALLEL=1 ./scripts/wasp_plant_per_model_dockers.sh

set -euo pipefail

cd "$(dirname "$0")/.."

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
    "w01       9301 8301"
    "w02       9311 8311"
    "w03       9321 8321"
    "w04       9331 8331"
    "w05       9341 8341"
    "w06       9351 8351"
    "w07       9361 8361"
    "w08       9371 8371"
    "w09       9381 8381"
    "w10       9391 8391"
    "w11       9401 8401"
    "w12       9411 8411"
    "w13       9421 8421"
    "w14       9431 8431"
    "w15       9441 8441"
    "w16       9451 8451"
    "w17       9461 8461"
    "w18       9471 8471"
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
        ./scripts/wasp_plant_full_stack.sh "$slug" "$gitlab_port" "$reddit_port" \
            > "$log" 2>&1 &
        PIDS+=("$!")
        LABELS+=("$slug")
    else
        ./scripts/wasp_plant_full_stack.sh "$slug" "$gitlab_port" "$reddit_port" \
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
