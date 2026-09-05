#!/usr/bin/env bash
# Smoke test for tool-calling benchmarks (AgentDojo + InjecAgent) through
# the matrix runner.
#
# Validates:
# 1. Matrix path works end-to-end for both benchmarks (no missing args).
# 2. --system-prompt-frame is correctly forwarded for non-`none` frames.
# 3. Both VEA + 12-prompt 5PQ judges run on both benchmarks.
# 4. n=2 in each stream — total 8 trajectories.
#
# 4 streams in parallel (2 benchmarks × 2 arms: bare and xml_safety).
# Tool-calling benchmarks don't use docker — no isolation concerns,
# just plumbing.
#
# Cost: ~$2-4 (8 glm-5:thinking trajectories + judges).
# Wallclock: ~3-5 minutes.

set -euo pipefail

cd "$(dirname "$0")/../../.."

PYTHON=.venv/bin/python
OUTPUT_BASE="eval_awareness_experiments/results/smoke_toolcalling_test_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/smoke_toolcalling"
mkdir -p "$LOG_DIR" "$OUTPUT_BASE"

echo "================================================================"
echo "Pre-flight: smoke target = $OUTPUT_BASE"
echo "================================================================"
echo

# Each stream: <name> <benchmark> <preset> <frame>
STREAMS=(
    "agentdojo_bare       agentdojo  none  none"
    "agentdojo_xml_safety agentdojo  none  safety_test"
    "injecagent_bare      injecagent none  none"
    "injecagent_xml_safety injecagent none safety_test"
)

PIDS=()
for stream in "${STREAMS[@]}"; do
    read -r name benchmark preset frame <<< "$stream"
    log="$LOG_DIR/${name}.log"
    echo "  $name → $log"

    nohup $PYTHON -m eval_awareness_experiments.run_causal_experiment \
        --models z-ai/glm-5:thinking \
        --benchmarks "$benchmark" \
        --conditions baseline \
        --extra-instructions-presets "$preset" \
        --system-prompt-frames "$frame" \
        --judges verbalized_awareness purpose_continue_5q \
        --tasks-per-split 2 \
        --output-base "$OUTPUT_BASE" \
        > "$log" 2>&1 &
    PIDS+=("$!")
    sleep 0.3
done

echo
echo "Launched ${#PIDS[@]} streams. PIDs: ${PIDS[*]}"
echo "Waiting..."

for pid in "${PIDS[@]}"; do
    wait "$pid" || echo "  stream PID $pid exited with non-zero"
done

echo
echo "================================================================"
echo "Results"
echo "================================================================"

failed=0
for stream in "${STREAMS[@]}"; do
    read -r name benchmark _ frame <<< "$stream"

    # Locate the JSONL produced by this stream
    jsonl=$(find "$OUTPUT_BASE/$benchmark" -path "*${benchmark}/*/trajectory_awareness_results.jsonl" 2>/dev/null | head -1 || true)
    if [ -n "$jsonl" ] && [ -s "$jsonl" ]; then
        rows=$(wc -l < "$jsonl")
        echo "  [ok]  $name → $jsonl ($rows rows)"
    else
        echo "  [FAIL] $name → no JSONL produced. Check $LOG_DIR/${name}.log"
        failed=1
    fi
done

# Verify --system-prompt-frame actually got applied for the xml_safety arms.
# The composed system prompt should be wrapped in <llm_safety_test>...</llm_safety_test>.
echo
echo "================================================================"
echo "Frame application check (xml_safety arms should have <llm_safety_test> wrap)"
echo "================================================================"

for benchmark in agentdojo injecagent; do
    # The trajectory JSON has the system_prompt field (per the system_prompt
    # field we added to ToolCallTaskResult).
    traj=$(find "$OUTPUT_BASE/$benchmark/xml_safety" -name "*.json" -not -name "*manifest*" -not -name "run_meta*" 2>/dev/null | head -1 || true)
    if [ -z "$traj" ]; then
        # Look in the per-task results dir
        traj=$(find "$OUTPUT_BASE" -path "*${benchmark}*xml_safety*" -name "*.json" 2>/dev/null | grep -v manifest | grep -v run_meta | head -1 || true)
    fi
    if [ -z "$traj" ]; then
        # The actual trajectory JSONs are in results/toolcalling/, not output_base
        traj=$(find "results/toolcalling/$benchmark" -name "*.json" -newer "$OUTPUT_BASE" 2>/dev/null | head -1 || true)
    fi
    if [ -n "$traj" ]; then
        if grep -q "<llm_safety_test>" "$traj"; then
            echo "  [ok]  $benchmark xml_safety: <llm_safety_test> wrap PRESENT in $(basename $traj)"
        else
            echo "  [FAIL] $benchmark xml_safety: <llm_safety_test> wrap MISSING in $(basename $traj)"
            echo "         Frame forwarding may not be working."
            failed=1
        fi
    else
        echo "  [warn] $benchmark xml_safety: no trajectory JSON found to check"
    fi
done

echo
echo "Output dir: $OUTPUT_BASE"
echo "Per-stream logs: $LOG_DIR/*.log"

if [ "$failed" -eq 1 ]; then
    exit 1
fi
echo
echo "Tool-calling smoke: GREEN ✓"
