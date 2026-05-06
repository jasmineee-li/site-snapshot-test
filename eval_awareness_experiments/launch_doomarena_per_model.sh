#!/bin/bash
# DoomArena launcher with strict 1-cell-per-stack isolation.
#
# Topology: selected (model × arm) cells go into a shared queue, and selected
# docker stacks act as generic workers. So at any moment:
#   - one active cell per worker stack
#   - each stack handles 1 active session per configured site split at a time
#   - when a model finishes, its stack can pick up another model's remaining cell
#
# That's the strict-isolation invariant the user asked for: no docker container
# ever has two concurrent agent sessions. Compare to the old `launch_pilot.sh`
# which had 4 arm-streams sharing one stack at the default ports — see
# DOOMARENA_ROOT_CAUSE_HANDOFF.md.
#
# Default splits are scoped to reddit + gitlab. For an all-site run, set:
#   SPLITS="reddit shopping shopping_admin gitlab"
#
# Per-model docker stacks must be up:
#   ./scripts/setup_doomarena_per_model_dockers.sh up
#
# Usage:
#   ./eval_awareness_experiments/launch_doomarena_per_model.sh
#   N_TASKS=50 SPLITS="reddit shopping shopping_admin gitlab" \
#     MODEL_STACK_FILTER="glm sonnet opus gpt gemini25 kimi25" \
#     WORKER_STACK_FILTER="glm sonnet opus gpt gemini25 kimi25" \
#     ARM_FILTER="bare xml_safety xml_scenario" \
#     JUDGES="verbalized_awareness purpose_continue_5q" \
#     ./eval_awareness_experiments/launch_doomarena_per_model.sh
#   ./eval_awareness_experiments/launch_doomarena_per_model.sh --dry-run

set -e

cd "$(dirname "$0")/.."

PYTHON=.venv/bin/python
OUTPUT_BASE="${OUTPUT_BASE:-eval_awareness_experiments/results/causal_pilot}"
LOG_DIR="${LOG_DIR:-logs/causal_pilot_doom_per_model}"
JUDGE_MODEL="${JUDGE_MODEL:-anthropic/claude-opus-4.7:thinking}"
N_TASKS="${N_TASKS:-20}"
SPLITS="${SPLITS:-reddit gitlab}"
MODEL_STACK_FILTER="${MODEL_STACK_FILTER:-all}"
WORKER_STACK_FILTER="${WORKER_STACK_FILTER:-$MODEL_STACK_FILTER}"
ARM_FILTER="${ARM_FILTER:-all}"
JUDGES="${JUDGES:-}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

# Split subprocess timeout is a last-resort guard for process-level hangs.
# AgentLab still has the tighter per-task episode timeout. Default this to
# 15 min per requested task plus 30 min setup/teardown slack, so larger
# per-site runs do not get killed after the old fixed 30 min cap.
BROWSER_TASK_TIMEOUT="${BROWSER_TASK_TIMEOUT:-900}"
BROWSER_STAGE1_OVERHEAD="${BROWSER_STAGE1_OVERHEAD:-1800}"
BROWSER_STAGE1_TIMEOUT="${BROWSER_STAGE1_TIMEOUT:-$((N_TASKS * BROWSER_TASK_TIMEOUT + BROWSER_STAGE1_OVERHEAD))}"
BROWSER_STAGE1_IDLE_TIMEOUT="${BROWSER_STAGE1_IDLE_TIMEOUT:-3600}"
BROWSER_STAGE1_RELAUNCH_ATTEMPTS="${BROWSER_STAGE1_RELAUNCH_ATTEMPTS:-3}"
BROWSER_RELAUNCH_INCOMPLETE="${BROWSER_RELAUNCH_INCOMPLETE:-0}"
SKIP_EXISTING_FLAG=""
if [ "$SKIP_EXISTING" = "1" ] || [ "$SKIP_EXISTING" = "true" ]; then
    SKIP_EXISTING_FLAG="--skip-existing"
fi
BROWSER_RELAUNCH_INCOMPLETE_FLAG=""
if [ "$BROWSER_RELAUNCH_INCOMPLETE" = "1" ] || [ "$BROWSER_RELAUNCH_INCOMPLETE" = "true" ]; then
    BROWSER_RELAUNCH_INCOMPLETE_FLAG="--browser-relaunch-incomplete"
fi

DRY_RUN=""
if [ "$1" = "--dry-run" ]; then
    DRY_RUN="--dry-run"
    echo "DRY-RUN MODE"
    echo
fi

mkdir -p "$LOG_DIR"

echo "Config:"
echo "  output_base=$OUTPUT_BASE"
echo "  log_dir=$LOG_DIR"
echo "  n_tasks_per_split=$N_TASKS"
echo "  splits=$SPLITS"
echo "  model_stack_filter=$MODEL_STACK_FILTER"
echo "  worker_stack_filter=$WORKER_STACK_FILTER"
echo "  arm_filter=$ARM_FILTER"
if [ -n "$JUDGES" ]; then
    echo "  judges=$JUDGES"
else
    echo "  judges=<per-arm defaults>"
fi
echo "  browser_stage1_timeout=${BROWSER_STAGE1_TIMEOUT}s"
echo "  browser_stage1_idle_timeout=${BROWSER_STAGE1_IDLE_TIMEOUT}s"
echo "  browser_stage1_relaunch_attempts=${BROWSER_STAGE1_RELAUNCH_ATTEMPTS}"
echo "  browser_relaunch_incomplete=${BROWSER_RELAUNCH_INCOMPLETE}"
echo "  skip_existing=${SKIP_EXISTING}"
echo

# (model_id | stack_slug | gitlab_port | reddit_port)
# Stack slugs match scripts/setup_doomarena_per_model_dockers.sh.
# Original GLM-5 stack uses ports 9002/8081 (no model slug suffix).
MODEL_STACKS=(
    "z-ai/glm-5:thinking|glm|9002|8081"
    "anthropic/claude-sonnet-4.6:thinking|sonnet|9012|8091"
    "anthropic/claude-opus-4.7:thinking|opus|9022|8101"
    "openai/gpt-5.2:thinking|gpt|9032|8111"
    "google/gemini-3-flash-preview|flash|9042|8121"
    "google/gemini-3.1-pro-preview|pro|9052|8131"
    "google/gemini-2.5-pro:thinking|gemini25|9062|8141"
    "moonshotai/kimi-k2.5:thinking|kimi25|9072|8151"
)

# Each arm = "name preset frame judges"
# Judge list: bare arm gets purpose_continue_5q + verbalized_awareness;
# others just verbalized_awareness (5PQ doesn't add signal under causal frames).
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
    SHOPPING_PORT=$((REDDIT_PORT + 1))
    ADMIN_PORT=$((REDDIT_PORT + 2))

    log="$LOG_DIR/${STACK}.log"
    echo "  worker=$STACK → gitlab=:$GITLAB_PORT reddit=:$REDDIT_PORT  →  $log"
    N_WORKERS=$((N_WORKERS + 1))

    if [ -n "$DRY_RUN" ]; then
        continue
    fi

    # One worker per docker stack. The parent waits for all workers below;
    # otherwise the launcher can exit while orphaned browser subprocesses are
    # still running.
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
            env \
                GITLAB="http://localhost:$GITLAB_PORT" \
                REDDIT="http://localhost:$REDDIT_PORT" \
                SHOPPING="http://localhost:$SHOPPING_PORT" \
                SHOPPING_ADMIN="http://localhost:$ADMIN_PORT/admin" \
                $PYTHON -m eval_awareness_experiments.run_causal_experiment \
                --models "$MODEL" \
                --benchmarks doomarena \
                --conditions baseline \
                --extra-instructions-presets "$PRESET" \
                --system-prompt-frames "$FRAME" \
                --judges $judges \
                --judge-model "$JUDGE_MODEL" \
                --tasks-per-split "$N_TASKS" \
                --browser-stage1-timeout "$BROWSER_STAGE1_TIMEOUT" \
                --browser-stage1-idle-timeout "$BROWSER_STAGE1_IDLE_TIMEOUT" \
                --browser-stage1-relaunch-attempts "$BROWSER_STAGE1_RELAUNCH_ATTEMPTS" \
                --splits $SPLITS \
                --output-base "$OUTPUT_BASE" \
                $BROWSER_RELAUNCH_INCOMPLETE_FLAG \
                $SKIP_EXISTING_FLAG
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
    echo "Would launch $N_WORKERS generic stack workers for ${#CELL_SPECS[@]} selected cells (${#SELECTED_MODELS[@]} models × $N_SELECTED_ARMS arms)."
    exit 0
fi

echo
echo "Launched ${#PIDS[@]} generic stack workers for ${#CELL_SPECS[@]} selected cells. PIDs: ${PIDS[*]}"
echo
echo "Monitor:"
echo "  tail -f $LOG_DIR/*.log"
echo "  $PYTHON -m eval_awareness_experiments.run_manifest --results-dir $OUTPUT_BASE --tasks-per-split $N_TASKS --benchmark-splits doomarena=$(echo "$SPLITS" | tr ' ' ',') --print"
echo
echo "Kill:"
echo "  pkill -f 'run_causal_experiment.*$OUTPUT_BASE'"

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
