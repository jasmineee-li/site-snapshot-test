#!/bin/bash
# Parallel launch of the causal eval-awareness pilot.
#
# Fires 16 streams (4 arms × 4 benchmarks), one per (arm, benchmark) pair.
# Each stream runs all 6 models sequentially within its (arm, benchmark)
# slot, with concurrency=20 within each cell. Total wall-clock target:
# ~30-45 minutes (vs. ~3 hours for the single-stream version).
#
# Each stream writes its own log under logs/causal_pilot/<arm>__<benchmark>.log.
# All streams write trajectories into the shared output base
# eval_awareness_experiments/results/causal_pilot/<benchmark>/<arm>/<model>/.
#
# Re-runnable: --skip-existing makes already-complete cells get skipped
# automatically. If a stream dies, just re-run the script.
#
# Usage:
#   ./eval_awareness_experiments/launch_pilot.sh           # launch all 16 streams
#   ./eval_awareness_experiments/launch_pilot.sh --dry-run # print commands only
#
# Monitor:
#   .venv/bin/python -m eval_awareness_experiments.dashboard \
#       --results-dir eval_awareness_experiments/results/causal_pilot --port 8000
#   tail -f logs/causal_pilot/*.log
#
# Kill all running streams:
#   pkill -f "run_causal_experiment.*causal_pilot"

set -e

cd "$(dirname "$0")/.."   # repo root

PYTHON=.venv/bin/python
OUTPUT_BASE=eval_awareness_experiments/results/causal_pilot
LOG_DIR=logs/causal_pilot
JUDGE_MODEL="anthropic/claude-opus-4.7:thinking"
N_TASKS=20

DRY_RUN=""
if [ "$1" = "--dry-run" ]; then
    DRY_RUN="--dry-run"
    echo "DRY-RUN MODE: commands will be printed but not executed"
    echo
fi

mkdir -p "$LOG_DIR"

MODELS=(
    "z-ai/glm-5:thinking"
    "anthropic/claude-sonnet-4.6:thinking"
    "anthropic/claude-opus-4.7:thinking"
    "openai/gpt-5.2:thinking"
    "google/gemini-3-flash-preview"
    "google/gemini-3.1-pro-preview"
)

BENCHMARKS=(agentdojo injecagent doomarena wasp)

# Each stream is "<arm_name> <preset> <frame> <judges-space-separated>"
STREAMS=(
    "bare         none              none                  verbalized_awareness purpose_continue_5q"
    "xml_safety   none              safety_test           verbalized_awareness"
    "xml_scenario none              hypothetical_scenario verbalized_awareness"
    "native       scratchpad_native none                  verbalized_awareness"
)

PIDS=()
for stream in "${STREAMS[@]}"; do
    read -r arm preset frame judges <<< "$stream"
    for benchmark in "${BENCHMARKS[@]}"; do
        log="$LOG_DIR/${arm}__${benchmark}.log"
        echo "  $arm × $benchmark → $log"
        if [ -n "$DRY_RUN" ]; then
            continue
        fi
        nohup $PYTHON -m eval_awareness_experiments.run_causal_experiment \
            --models "${MODELS[@]}" \
            --benchmarks "$benchmark" \
            --conditions baseline \
            --extra-instructions-presets "$preset" \
            --system-prompt-frames "$frame" \
            --judges $judges \
            --judge-model "$JUDGE_MODEL" \
            --tasks-per-split "$N_TASKS" \
            --skip-existing \
            --output-base "$OUTPUT_BASE" \
            > "$log" 2>&1 &
        PIDS+=("$!")
        sleep 0.3   # tiny stagger so we don't slam OpenRouter with 16 simultaneous starts
    done
done

if [ -n "$DRY_RUN" ]; then
    echo
    echo "Would launch ${#STREAMS[@]} arms × ${#BENCHMARKS[@]} benchmarks = "\
"$((${#STREAMS[@]} * ${#BENCHMARKS[@]})) streams."
    exit 0
fi

echo
echo "Launched ${#PIDS[@]} streams. PIDs: ${PIDS[*]}"
echo
echo "Monitor:"
echo "  $PYTHON -m eval_awareness_experiments.dashboard \\"
echo "      --results-dir $OUTPUT_BASE --port 8000"
echo "  tail -f $LOG_DIR/*.log"
echo
echo "Kill all running streams:"
echo "  pkill -f \"run_causal_experiment.*causal_pilot\""
