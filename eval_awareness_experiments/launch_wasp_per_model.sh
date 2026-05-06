#!/usr/bin/env bash
# WASP launcher with strict 1-cell-per-stack isolation.
#
# Prereqs:
#   ./scripts/setup_wasp_per_model_dockers.sh up
#   ./scripts/wasp_plant_per_model_dockers.sh
#
# Topology: selected (model x arm) cells go into a shared queue, and selected
# WASP docker stacks act as generic workers. Each worker uses:
#   - gitlab_wasp_<stack> on its own port
#   - forum_wasp_<stack> on its own port
#   - /tmp/wasp_full_<stack> planted task JSONs
#
# Usage:
#   N_TASKS=100 MODEL_STACK_FILTER="gpt gemini25 kimi25" \
#     ARM_FILTER="bare xml_safety xml_scenario" \
#     ./eval_awareness_experiments/launch_wasp_per_model.sh
#   N_TASKS=200 MODEL_STACK_FILTER="glm sonnet opus gpt gemini25 kimi25" \
#     WORKER_STACK_FILTER="gpt gemini25 kimi25" \
#     ARM_FILTER="xml_control" \
#     ./eval_awareness_experiments/launch_wasp_per_model.sh

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON=.venv/bin/python
OUTPUT_BASE="${OUTPUT_BASE:-eval_awareness_experiments/results/causal_pilot}"
LOG_DIR="${LOG_DIR:-logs/causal_pilot_wasp_per_model}"
JUDGE_MODEL="${JUDGE_MODEL:-anthropic/claude-opus-4.7:thinking}"
N_TASKS="${N_TASKS:-100}"
SPLITS="${SPLITS:-reddit gitlab}"
MODEL_STACK_FILTER="${MODEL_STACK_FILTER:-all}"
WORKER_STACK_FILTER="${WORKER_STACK_FILTER:-$MODEL_STACK_FILTER}"
ARM_FILTER="${ARM_FILTER:-all}"
JUDGES="${JUDGES:-}"
MAX_STEPS="${MAX_STEPS:-15}"
AVG_STEP_TIMEOUT="${AVG_STEP_TIMEOUT:-60}"
BROWSER_STAGE1_OVERHEAD="${BROWSER_STAGE1_OVERHEAD:-1800}"
BROWSER_STAGE1_TIMEOUT="${BROWSER_STAGE1_TIMEOUT:-}"
BROWSER_RELAUNCH_INCOMPLETE="${BROWSER_RELAUNCH_INCOMPLETE:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
WASP_TASK_DIR_BASE="${WASP_TASK_DIR_BASE:-/tmp/wasp_full}"

DRY_RUN=""
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN="1"
    echo "DRY-RUN MODE"
    echo
fi

mkdir -p "$LOG_DIR"

MODEL_STACKS=(
    "z-ai/glm-5:thinking|glm|9201|8201"
    "anthropic/claude-sonnet-4.6:thinking|sonnet|9211|8211"
    "anthropic/claude-opus-4.7:thinking|opus|9221|8221"
    "openai/gpt-5.2:thinking|gpt|9231|8231"
    "google/gemini-2.5-pro:thinking|gemini25|9241|8241"
    "moonshotai/kimi-k2.5:thinking|kimi25|9251|8251"
)

ARMS=(
    "bare         none              none                  verbalized_awareness purpose_continue_5q"
    "xml_control  none              interaction_context   verbalized_awareness"
    "xml_safety   none              safety_test           verbalized_awareness"
    "xml_scenario none              hypothetical_scenario verbalized_awareness"
    "native       scratchpad_native none                  verbalized_awareness"
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

echo "Config:"
echo "  output_base=$OUTPUT_BASE"
echo "  log_dir=$LOG_DIR"
echo "  n_tasks_per_split=$N_TASKS"
echo "  splits=$SPLITS"
echo "  model_stack_filter=$MODEL_STACK_FILTER"
echo "  worker_stack_filter=$WORKER_STACK_FILTER"
echo "  arm_filter=$ARM_FILTER"
echo "  max_steps=$MAX_STEPS"
echo "  avg_step_timeout=$AVG_STEP_TIMEOUT"
echo "  browser_stage1_overhead=$BROWSER_STAGE1_OVERHEAD"
if [ -n "$BROWSER_STAGE1_TIMEOUT" ]; then
    echo "  browser_stage1_timeout=$BROWSER_STAGE1_TIMEOUT"
fi
echo "  browser_relaunch_incomplete=$BROWSER_RELAUNCH_INCOMPLETE"
echo "  skip_existing=$SKIP_EXISTING"
echo "  wasp_task_dir_base=$WASP_TASK_DIR_BASE"
if [ -n "$JUDGES" ]; then
    echo "  judges=$JUDGES"
else
    echo "  judges=<per-arm defaults>"
fi
echo

PIDS=()
PID_LABELS=()
N_WORKERS=0
N_SELECTED_ARMS=0
SELECTED_MODELS=()
SELECTED_WORKERS=()

for entry in "${MODEL_STACKS[@]}"; do
    IFS='|' read -r MODEL STACK GITLAB_PORT REDDIT_PORT <<< "$entry"
    if contains_word "$STACK" "$MODEL_STACK_FILTER"; then
        SELECTED_MODELS+=("$MODEL")
    fi
    if contains_word "$STACK" "$WORKER_STACK_FILTER"; then
        SELECTED_WORKERS+=("$entry")
    fi
done

for arm_spec in "${ARMS[@]}"; do
    read -r ARM _ <<< "$arm_spec"
    if contains_word "$ARM" "$ARM_FILTER"; then
        N_SELECTED_ARMS=$((N_SELECTED_ARMS + 1))
    fi
done

CELL_SPECS=()
for MODEL in "${SELECTED_MODELS[@]}"; do
    for arm_spec in "${ARMS[@]}"; do
        read -r ARM PRESET FRAME judges <<< "$arm_spec"
        if ! contains_word "$ARM" "$ARM_FILTER"; then
            continue
        fi
        if [ -n "$JUDGES" ]; then
            judges="$JUDGES"
        fi
        CELL_SPECS+=("$MODEL|$ARM|$PRESET|$FRAME|$judges")
    done
done

if [ "${#CELL_SPECS[@]}" -eq 0 ]; then
    echo "No cells selected."
    exit 0
fi

if [ "${#SELECTED_WORKERS[@]}" -eq 0 ]; then
    echo "No worker stacks selected." >&2
    exit 1
fi

QUEUE_DIR="$LOG_DIR/.queue_$(date +%Y%m%d_%H%M%S)_$$"
QUEUE_FILE="$QUEUE_DIR/cells.tsv"
QUEUE_LOCK="$QUEUE_DIR/lock"
if [ -z "$DRY_RUN" ]; then
    if ! command -v flock >/dev/null 2>&1; then
        echo "flock is required for generic worker queue locking." >&2
        exit 1
    fi
    mkdir -p "$QUEUE_DIR"
    printf '%s\n' "${CELL_SPECS[@]}" > "$QUEUE_FILE"
fi

next_cell() {
    local line
    exec 9>"$QUEUE_LOCK"
    flock 9
    line="$(head -n 1 "$QUEUE_FILE" || true)"
    if [ -n "$line" ]; then
        tail -n +2 "$QUEUE_FILE" > "$QUEUE_FILE.tmp"
        mv "$QUEUE_FILE.tmp" "$QUEUE_FILE"
    fi
    flock -u 9
    exec 9>&-
    printf '%s\n' "$line"
}

for entry in "${SELECTED_WORKERS[@]}"; do
    IFS='|' read -r _ STACK GITLAB_PORT REDDIT_PORT <<< "$entry"
    task_dir="${WASP_TASK_DIR_BASE}_${STACK}"
    log="$LOG_DIR/${STACK}.log"
    echo "  worker=$STACK -> gitlab=:$GITLAB_PORT reddit=:$REDDIT_PORT task_dir=$task_dir -> $log"
    N_WORKERS=$((N_WORKERS + 1))

    if [ -n "$DRY_RUN" ]; then
        continue
    fi

    if [ ! -d "$task_dir/webarena_tasks" ]; then
        echo "Missing planted WASP task dir for worker $STACK: $task_dir/webarena_tasks" >&2
        exit 1
    fi

    (
        worker_status=0
        while true; do
            cell="$(next_cell)"
            if [ -z "$cell" ]; then
                echo "[$STACK] queue empty; worker exiting" >&2
                break
            fi
            IFS='|' read -r MODEL ARM PRESET FRAME judges <<< "$cell"
            echo "[$STACK] === model=$MODEL arm=$ARM preset=$PRESET frame=$FRAME ===" >&2
            set +e
            cmd=(
                "$PYTHON" -m eval_awareness_experiments.run_causal_experiment
                --models "$MODEL"
                --benchmarks wasp
                --conditions baseline
                --extra-instructions-presets "$PRESET"
                --system-prompt-frames "$FRAME"
                --judges $judges
                --judge-model "$JUDGE_MODEL"
                --tasks-per-split "$N_TASKS"
                --benchmark-splits "wasp=${SPLITS// /,}"
                --max-steps "$MAX_STEPS"
                --avg-step-timeout "$AVG_STEP_TIMEOUT"
                --browser-stage1-overhead "$BROWSER_STAGE1_OVERHEAD"
                --wasp-task-dir "$task_dir"
                --output-base "$OUTPUT_BASE"
            )
            if [ -n "$BROWSER_STAGE1_TIMEOUT" ]; then
                cmd+=(--browser-stage1-timeout "$BROWSER_STAGE1_TIMEOUT")
            fi
            if [ "$SKIP_EXISTING" = "1" ]; then
                cmd+=(--skip-existing)
            fi
            if [ "$BROWSER_RELAUNCH_INCOMPLETE" = "1" ] || [ "$BROWSER_RELAUNCH_INCOMPLETE" = "true" ]; then
                cmd+=(--browser-relaunch-incomplete)
            fi

            env \
                GITLAB="http://localhost:$GITLAB_PORT" \
                REDDIT="http://localhost:$REDDIT_PORT" \
                DATASET="webarena_prompt_injections" \
                "${cmd[@]}"
            status=$?
            set -e
            if [ "$status" -ne 0 ]; then
                echo "[$STACK] model=$MODEL arm=$ARM FAILED with $status" >&2
                worker_status="$status"
            fi
        done
        exit "$worker_status"
    ) > "$log" 2>&1 &
    PIDS+=("$!")
    PID_LABELS+=("$STACK")
    sleep 0.5
done

if [ -n "$DRY_RUN" ]; then
    echo
    echo "Would launch $N_WORKERS generic stack workers for ${#CELL_SPECS[@]} selected cells (${#SELECTED_MODELS[@]} models x $N_SELECTED_ARMS arms)."
    exit 0
fi

echo
echo "Launched ${#PIDS[@]} generic stack workers for ${#CELL_SPECS[@]} selected cells. PIDs: ${PIDS[*]}"
echo
echo "Monitor:"
echo "  tail -f $LOG_DIR/*.log"
echo "  $PYTHON -m eval_awareness_experiments.run_manifest --results-dir $OUTPUT_BASE --tasks-per-split $N_TASKS --benchmark-splits wasp=$(echo "$SPLITS" | tr ' ' ',') --print"

overall_status=0
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    label="${PID_LABELS[$i]}"
    if wait "$pid"; then
        echo "[$label] worker completed"
    else
        status=$?
        echo "[$label] worker FAILED with $status" >&2
        overall_status="$status"
    fi
done

if [ "$overall_status" -eq 0 ]; then
    echo "All stack workers completed."
else
    echo "One or more stack workers failed." >&2
fi
exit "$overall_status"
