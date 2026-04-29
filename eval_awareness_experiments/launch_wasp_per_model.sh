#!/usr/bin/env bash
# Per-model WASP launcher with WASP-only docker isolation.
#
# Prereqs:
#   ./scripts/setup_wasp_per_model_dockers.sh up
#   ./scripts/wasp_plant_per_model_dockers.sh
#
# Each selected model uses:
#   - gitlab_wasp_<stack> on its own port
#   - forum_wasp_<stack> on its own port
#   - /tmp/wasp_full_<stack> planted task JSONs
#
# Usage:
#   N_TASKS=100 MODEL_STACK_FILTER="gpt gemini25 kimi25" \
#     ARM_FILTER="bare xml_safety xml_scenario" \
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
ARM_FILTER="${ARM_FILTER:-all}"
JUDGES="${JUDGES:-}"
MAX_STEPS="${MAX_STEPS:-15}"
AVG_STEP_TIMEOUT="${AVG_STEP_TIMEOUT:-60}"
BROWSER_STAGE1_OVERHEAD="${BROWSER_STAGE1_OVERHEAD:-1800}"
BROWSER_STAGE1_TIMEOUT="${BROWSER_STAGE1_TIMEOUT:-}"
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
echo "  arm_filter=$ARM_FILTER"
echo "  max_steps=$MAX_STEPS"
echo "  avg_step_timeout=$AVG_STEP_TIMEOUT"
echo "  browser_stage1_overhead=$BROWSER_STAGE1_OVERHEAD"
if [ -n "$BROWSER_STAGE1_TIMEOUT" ]; then
    echo "  browser_stage1_timeout=$BROWSER_STAGE1_TIMEOUT"
fi
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
N_STREAMS=0
N_SELECTED_ARMS=0

for entry in "${MODEL_STACKS[@]}"; do
    IFS='|' read -r MODEL STACK GITLAB_PORT REDDIT_PORT <<< "$entry"
    if ! contains_word "$STACK" "$MODEL_STACK_FILTER"; then
        continue
    fi

    task_dir="${WASP_TASK_DIR_BASE}_${STACK}"
    log="$LOG_DIR/${STACK}.log"
    echo "  $STACK ($MODEL) -> gitlab=:$GITLAB_PORT reddit=:$REDDIT_PORT task_dir=$task_dir -> $log"
    N_STREAMS=$((N_STREAMS + 1))

    if [ -n "$DRY_RUN" ]; then
        continue
    fi

    (
        stream_status=0
        for arm_spec in "${ARMS[@]}"; do
            read -r ARM PRESET FRAME judges <<< "$arm_spec"
            if ! contains_word "$ARM" "$ARM_FILTER"; then
                continue
            fi
            if [ -n "$JUDGES" ]; then
                judges="$JUDGES"
            fi
            echo "[$STACK] === arm=$ARM preset=$PRESET frame=$FRAME ===" >&2
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

            env \
                GITLAB="http://localhost:$GITLAB_PORT" \
                REDDIT="http://localhost:$REDDIT_PORT" \
                DATASET="webarena_prompt_injections" \
                "${cmd[@]}"
            status=$?
            set -e
            if [ "$status" -ne 0 ]; then
                echo "[$STACK] arm=$ARM FAILED with $status" >&2
                stream_status="$status"
            fi
        done
        exit "$stream_status"
    ) > "$log" 2>&1 &
    PIDS+=("$!")
    PID_LABELS+=("$STACK")
    sleep 0.5
done

for arm_spec in "${ARMS[@]}"; do
    read -r ARM _ <<< "$arm_spec"
    if contains_word "$ARM" "$ARM_FILTER"; then
        N_SELECTED_ARMS=$((N_SELECTED_ARMS + 1))
    fi
done

if [ -n "$DRY_RUN" ]; then
    echo
    echo "Would launch $N_STREAMS per-model WASP streams x $N_SELECTED_ARMS arms."
    exit 0
fi

echo
echo "Launched ${#PIDS[@]} per-model WASP streams. PIDs: ${PIDS[*]}"
echo
echo "Monitor:"
echo "  tail -f $LOG_DIR/*.log"
echo "  $PYTHON -m eval_awareness_experiments.run_manifest --results-dir $OUTPUT_BASE --print"

overall_status=0
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    label="${PID_LABELS[$i]}"
    if wait "$pid"; then
        echo "[$label] stream completed"
    else
        status=$?
        echo "[$label] stream FAILED with $status" >&2
        overall_status="$status"
    fi
done

if [ "$overall_status" -eq 0 ]; then
    echo "All per-model WASP streams completed."
else
    echo "One or more per-model WASP streams failed." >&2
fi
exit "$overall_status"
