#!/usr/bin/env bash
# Parallel WASP e2e replay in model lanes.
#
# Each model gets one WASP stack. For PHASE=no_reset, the stack is reset and
# planted once, then that model's arms replay sequentially without cell resets.
# This preserves the cross-arm contamination question while using six stacks in
# parallel. For PHASE=reset_per_cell, the same lanes reset/replant before every
# arm.

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
RESULTS_DIR="${RESULTS_DIR:-eval_awareness_experiments/results/n200_2026-04-29}"
WASP_DIR="${WASP_DIR:-$RESULTS_DIR/wasp}"
OUT_BASE="${OUT_BASE:-$RESULTS_DIR/wasp_e2e_replay_20260509_lanes}"
LOG_DIR="${LOG_DIR:-logs/wasp_e2e_replay_lanes_20260509}"
PHASE="${PHASE:-no_reset}"
ARMS="${ARMS:-bare xml_safety xml_scenario xml_control}"
MODEL_FILTER="${MODEL_FILTER:-all}"
MAX_TASKS="${MAX_TASKS:-}"
SINGLE_SITE="${SINGLE_SITE:-}"
SKIP_STACK_RESET="${SKIP_STACK_RESET:-0}"

MODEL_LANES=(
    "anthropic_claude-opus-4.7_thinking|opus|9221|8221"
    "anthropic_claude-sonnet-4.6_thinking|sonnet|9211|8211"
    "google_gemini-2.5-pro_thinking|gemini25|9241|8241"
    "moonshotai_kimi-k2.5_thinking|kimi25|9251|8251"
    "openai_gpt-5.2_thinking|gpt|9231|8231"
    "z-ai_glm-5_thinking|glm|9201|8201"
)

case "$PHASE" in
    no_reset|reset_per_cell) ;;
    *)
        echo "PHASE must be no_reset or reset_per_cell, got: $PHASE" >&2
        exit 1
        ;;
esac

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

task_dir_for_stack() {
    local stack=$1
    echo "/tmp/wasp_full_${stack}"
}

run_model_lane() {
    local model=$1
    local stack=$2
    local gitlab_port=$3
    local reddit_port=$4
    local out_dir="$OUT_BASE/$PHASE"
    local task_dir
    local extra_args=()

    task_dir="$(task_dir_for_stack "$stack")"
    mkdir -p "$out_dir"

    if [ -n "$MAX_TASKS" ]; then
        extra_args+=(--max-tasks "$MAX_TASKS")
    fi
    if [ -n "$SINGLE_SITE" ]; then
        extra_args+=(--single-site "$SINGLE_SITE")
    fi

    echo "=== lane model=$model stack=$stack phase=$PHASE task_dir=$task_dir ==="

    if [ "$PHASE" = "no_reset" ] && [ "$SKIP_STACK_RESET" != "1" ]; then
        ./scripts/wasp_reset_stack.sh "$stack" "$gitlab_port" "$reddit_port"
    fi

    for arm in $ARMS; do
        local cell_dir="$WASP_DIR/$arm/$model"
        local output_jsonl="$out_dir/${arm}__${model}.jsonl"

        if [ ! -d "$cell_dir" ]; then
            echo "[$stack] skip missing $arm/$model"
            continue
        fi

        if [ "$PHASE" = "reset_per_cell" ] && [ "$SKIP_STACK_RESET" != "1" ]; then
            ./scripts/wasp_reset_stack.sh "$stack" "$gitlab_port" "$reddit_port"
        fi

        echo "[$stack] replay $arm/$model -> $output_jsonl"
        "$PYTHON" -m eval_awareness_experiments.wasp_replay_e2e_evaluator \
            --cell-dir "$cell_dir" \
            --task-dir "$task_dir" \
            --output-jsonl "$output_jsonl" \
            "${extra_args[@]}"
    done
}

mkdir -p "$LOG_DIR" "$OUT_BASE/$PHASE"

echo "results_dir=$RESULTS_DIR"
echo "out_base=$OUT_BASE"
echo "phase=$PHASE"
echo "arms=$ARMS"
echo "model_filter=$MODEL_FILTER"
echo "max_tasks=${MAX_TASKS:-all}"
echo "single_site=${SINGLE_SITE:-all}"
echo "skip_stack_reset=$SKIP_STACK_RESET"

PIDS=()
LABELS=()
for lane in "${MODEL_LANES[@]}"; do
    IFS='|' read -r model stack gitlab_port reddit_port <<<"$lane"
    if ! contains_word "$model" "$MODEL_FILTER" && ! contains_word "$stack" "$MODEL_FILTER"; then
        continue
    fi

    log="$LOG_DIR/${PHASE}_${stack}.log"
    (
        run_model_lane "$model" "$stack" "$gitlab_port" "$reddit_port"
    ) > "$log" 2>&1 &
    pid="$!"
    PIDS+=("$pid")
    LABELS+=("$stack")
    echo "launched lane stack=$stack model=$model pid=$pid log=$log"
done

if [ "${#PIDS[@]}" -eq 0 ]; then
    echo "No lanes selected." >&2
    exit 1
fi

overall_status=0
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    label="${LABELS[$i]}"
    if wait "$pid"; then
        echo "[$label] lane completed"
    else
        status=$?
        echo "[$label] lane FAILED with $status" >&2
        overall_status="$status"
    fi
done

OUT_DIR="$OUT_BASE/$PHASE" "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["OUT_DIR"])
scored = success = errors = 0
for path in sorted(root.glob("*.jsonl")):
    summary = None
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("type") == "summary":
                summary = row
    if not summary:
        continue
    n = summary.get("n_scored", 0) or 0
    s = summary.get("n_success", 0) or 0
    e = summary.get("n_errors", 0) or 0
    scored += n
    success += s
    errors += e
    asr = None if not n else s / n
    asr_str = "NA" if asr is None else f"{100 * asr:.2f}%"
    print(f"{path.name}: n={n} success={s} asr={asr_str} errors={e}")

total = None if not scored else success / scored
total_str = "NA" if total is None else f"{100 * total:.2f}%"
print(f"TOTAL: n={scored} success={success} asr={total_str} errors={errors}")
PY

exit "$overall_status"
