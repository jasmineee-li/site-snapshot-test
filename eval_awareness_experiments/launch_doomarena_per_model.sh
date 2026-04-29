#!/bin/bash
# Per-model DoomArena launcher with strict 1-session-per-container isolation.
#
# Topology: one stream per model (8 streams total), each stream iterates the
# 4 arms SEQUENTIALLY on its own dedicated docker stack. So at any moment:
#   - 8 concurrent streams (one per model)
#   - Each stream on ONE arm at a time
#   - Each stack handles 1 active session per site (reddit + gitlab) at a time
#
# That's the strict-isolation invariant the user asked for: no docker
# container ever has two concurrent agent sessions. Compare to the old
# `launch_pilot.sh` which had 4 arm-streams sharing one stack at the
# default ports — see DOOMARENA_ROOT_CAUSE_HANDOFF.md.
#
# Splits scoped to reddit + gitlab (shopping + shopping_admin had
# capacity issues in the prior pilot). Per-model docker stacks must be up:
#   ./scripts/setup_doomarena_per_model_dockers.sh up
#
# Usage:
#   ./eval_awareness_experiments/launch_doomarena_per_model.sh
#   ./eval_awareness_experiments/launch_doomarena_per_model.sh --dry-run

set -e

cd "$(dirname "$0")/.."

PYTHON=.venv/bin/python
OUTPUT_BASE=eval_awareness_experiments/results/causal_pilot
LOG_DIR=logs/causal_pilot_doom_per_model
JUDGE_MODEL="anthropic/claude-opus-4.7:thinking"
N_TASKS=20
SPLITS="reddit gitlab"
BROWSER_STAGE1_TIMEOUT="${BROWSER_STAGE1_TIMEOUT:-1800}"

DRY_RUN=""
if [ "$1" = "--dry-run" ]; then
    DRY_RUN="--dry-run"
    echo "DRY-RUN MODE"
    echo
fi

mkdir -p "$LOG_DIR"

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
    "xml_safety   none              safety_test           verbalized_awareness"
    "xml_scenario none              hypothetical_scenario verbalized_awareness"
    "native       scratchpad_native none                  verbalized_awareness"
)

PIDS=()
for entry in "${MODEL_STACKS[@]}"; do
    IFS='|' read -r MODEL STACK GITLAB_PORT REDDIT_PORT <<< "$entry"
    SHOPPING_PORT=$((REDDIT_PORT + 1))
    ADMIN_PORT=$((REDDIT_PORT + 2))

    log="$LOG_DIR/${STACK}.log"
    echo "  $STACK ($MODEL) → gitlab=:$GITLAB_PORT reddit=:$REDDIT_PORT  →  $log"

    if [ -n "$DRY_RUN" ]; then
        continue
    fi

    # One stream per MODEL. Inside, run all 4 arms sequentially via
    # run_causal_experiment iterating presets×frames. The product of
    # presets×frames yields 16 combos but only 4 canonical arms — the
    # 12 non-canonical pairs get fallback names. To keep the matrix
    # exactly the 4 canonical arms, we launch 4 separate sub-streams
    # (one per arm) and queue them sequentially via &&.
    (
        for arm_spec in "${ARMS[@]}"; do
            read -r ARM PRESET FRAME judges <<< "$arm_spec"
            echo "[$STACK] === arm=$ARM preset=$PRESET frame=$FRAME ===" >&2
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
                --splits $SPLITS \
                --output-base "$OUTPUT_BASE" \
                || echo "[$STACK] arm=$ARM FAILED with $?" >&2
        done
    ) > "$log" 2>&1 &
    PIDS+=("$!")
    sleep 0.5
done

if [ -n "$DRY_RUN" ]; then
    echo
    echo "Would launch ${#MODEL_STACKS[@]} per-model streams × ${#ARMS[@]} arms (sequential within each)."
    exit 0
fi

echo
echo "Launched ${#PIDS[@]} per-model streams. PIDs: ${PIDS[*]}"
echo
echo "Monitor:"
echo "  tail -f $LOG_DIR/*.log"
echo "  $PYTHON -m eval_awareness_experiments.run_manifest --results-dir $OUTPUT_BASE --print"
echo
echo "Kill:"
echo "  pkill -f 'run_causal_experiment.*causal_pilot'"
