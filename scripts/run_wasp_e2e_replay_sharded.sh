#!/usr/bin/env bash
# Parallel WASP e2e replay across task shards.
#
# Default layout uses 24 WASP stacks: six models x four shards/model. Each
# shard resets/plants one stack, then replays all selected arms sequentially on
# that stack. That keeps the no-reset comparison meaningful while using the
# extra generic workers.

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
RESULTS_DIR="${RESULTS_DIR:-eval_awareness_experiments/results/n200_2026-04-29}"
WASP_DIR="${WASP_DIR:-$RESULTS_DIR/wasp}"
OUT_BASE="${OUT_BASE:-$RESULTS_DIR/wasp_e2e_replay_20260509_sharded}"
LOG_DIR="${LOG_DIR:-logs/wasp_e2e_replay_sharded_20260509}"
PHASE="${PHASE:-no_reset}"
ARMS="${ARMS:-bare xml_safety xml_scenario xml_control}"
MODEL_FILTER="${MODEL_FILTER:-all}"
WORKER_STACK_FILTER="${WORKER_STACK_FILTER:-all}"
SHARDS_PER_MODEL="${SHARDS_PER_MODEL:-4}"
MAX_TASKS="${MAX_TASKS:-}"
SINGLE_SITE="${SINGLE_SITE:-}"
SKIP_STACK_RESET="${SKIP_STACK_RESET:-0}"

MODELS=(
    "anthropic_claude-opus-4.7_thinking"
    "anthropic_claude-sonnet-4.6_thinking"
    "google_gemini-2.5-pro_thinking"
    "moonshotai_kimi-k2.5_thinking"
    "openai_gpt-5.2_thinking"
    "z-ai_glm-5_thinking"
)

# stack | gitlab_port | reddit_port
STACKS=(
    "glm|9201|8201"
    "sonnet|9211|8211"
    "opus|9221|8221"
    "gpt|9231|8231"
    "gemini25|9241|8241"
    "kimi25|9251|8251"
    "w01|9301|8301"
    "w02|9311|8311"
    "w03|9321|8321"
    "w04|9331|8331"
    "w05|9341|8341"
    "w06|9351|8351"
    "w07|9361|8361"
    "w08|9371|8371"
    "w09|9381|8381"
    "w10|9391|8391"
    "w11|9401|8401"
    "w12|9411|8411"
    "w13|9421|8421"
    "w14|9431|8431"
    "w15|9441|8441"
    "w16|9451|8451"
    "w17|9461|8461"
    "w18|9471|8471"
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

task_ids_for_shard() {
    local task_dir=$1
    local shard_idx=$2
    local shard_count=$3
    local max_tasks=${4:-}

    "$PYTHON" - "$task_dir" "$shard_idx" "$shard_count" "$max_tasks" <<'PY'
import sys
from pathlib import Path

task_dir = Path(sys.argv[1])
shard_idx = int(sys.argv[2])
shard_count = int(sys.argv[3])
max_tasks = int(sys.argv[4]) if sys.argv[4] else None

cfg_dir = task_dir / "webarena_tasks" if (task_dir / "webarena_tasks").is_dir() else task_dir
task_nums = sorted(int(path.stem) for path in cfg_dir.glob("*.json") if path.stem.isdigit())
task_ids = [
    f"webarena.{task_num}"
    for idx, task_num in enumerate(task_nums)
    if idx % shard_count == shard_idx
]
if max_tasks is not None:
    task_ids = task_ids[:max_tasks]
print(",".join(task_ids))
PY
}

run_shard() {
    local model=$1
    local shard_idx=$2
    local shard_count=$3
    local stack=$4
    local gitlab_port=$5
    local reddit_port=$6
    local out_dir="$OUT_BASE/$PHASE"
    local task_dir
    local extra_args=()
    local task_ids

    task_dir="$(task_dir_for_stack "$stack")"
    mkdir -p "$out_dir"

    if [ -n "$SINGLE_SITE" ]; then
        extra_args+=(--single-site "$SINGLE_SITE")
    fi

    echo "=== shard model=$model shard=$shard_idx/$shard_count stack=$stack phase=$PHASE task_dir=$task_dir ==="

    if [ "$PHASE" = "no_reset" ]; then
        if [ "$SKIP_STACK_RESET" != "1" ]; then
            ./scripts/wasp_reset_stack.sh "$stack" "$gitlab_port" "$reddit_port"
        fi
        task_ids="$(task_ids_for_shard "$task_dir" "$shard_idx" "$shard_count" "$MAX_TASKS")"
    elif [ "$SKIP_STACK_RESET" = "1" ]; then
        task_ids="$(task_ids_for_shard "$task_dir" "$shard_idx" "$shard_count" "$MAX_TASKS")"
    fi

    for arm in $ARMS; do
        local cell_dir="$WASP_DIR/$arm/$model"
        local output_jsonl="$out_dir/${arm}__${model}__shard${shard_idx}of${shard_count}__${stack}.jsonl"

        if [ ! -d "$cell_dir" ]; then
            echo "[$stack] skip missing $arm/$model"
            continue
        fi

        if [ "$PHASE" = "reset_per_cell" ] && [ "$SKIP_STACK_RESET" != "1" ]; then
            ./scripts/wasp_reset_stack.sh "$stack" "$gitlab_port" "$reddit_port"
            task_ids="$(task_ids_for_shard "$task_dir" "$shard_idx" "$shard_count" "$MAX_TASKS")"
        fi
        if [ -z "$task_ids" ]; then
            echo "[$stack] no task ids selected for shard $shard_idx/$shard_count from $task_dir" >&2
            exit 1
        fi

        echo "[$stack] replay $arm/$model shard=$shard_idx/$shard_count -> $output_jsonl"
        "$PYTHON" -m eval_awareness_experiments.wasp_replay_e2e_evaluator \
            --cell-dir "$cell_dir" \
            --task-dir "$task_dir" \
            --task-id "$task_ids" \
            --output-jsonl "$output_jsonl" \
            "${extra_args[@]}"
    done
}

SELECTED_MODELS=()
for model in "${MODELS[@]}"; do
    if contains_word "$model" "$MODEL_FILTER"; then
        SELECTED_MODELS+=("$model")
    fi
done

SELECTED_STACKS=()
for entry in "${STACKS[@]}"; do
    IFS='|' read -r stack gitlab_port reddit_port <<<"$entry"
    if contains_word "$stack" "$WORKER_STACK_FILTER"; then
        SELECTED_STACKS+=("$entry")
    fi
done

if [ "${#SELECTED_MODELS[@]}" -eq 0 ]; then
    echo "No models selected." >&2
    exit 1
fi

needed_stacks=$((${#SELECTED_MODELS[@]} * SHARDS_PER_MODEL))
if [ "${#SELECTED_STACKS[@]}" -lt "$needed_stacks" ]; then
    echo "Need $needed_stacks stacks for ${#SELECTED_MODELS[@]} models x $SHARDS_PER_MODEL shards, got ${#SELECTED_STACKS[@]}." >&2
    exit 1
fi

mkdir -p "$LOG_DIR" "$OUT_BASE/$PHASE"

echo "results_dir=$RESULTS_DIR"
echo "out_base=$OUT_BASE"
echo "phase=$PHASE"
echo "arms=$ARMS"
echo "model_filter=$MODEL_FILTER"
echo "worker_stack_filter=$WORKER_STACK_FILTER"
echo "shards_per_model=$SHARDS_PER_MODEL"
echo "max_tasks_per_shard=${MAX_TASKS:-all}"
echo "single_site=${SINGLE_SITE:-all}"
echo "skip_stack_reset=$SKIP_STACK_RESET"

PIDS=()
LABELS=()
worker_idx=0
for model in "${SELECTED_MODELS[@]}"; do
    for ((shard_idx = 0; shard_idx < SHARDS_PER_MODEL; shard_idx++)); do
        IFS='|' read -r stack gitlab_port reddit_port <<<"${SELECTED_STACKS[$worker_idx]}"
        log="$LOG_DIR/${PHASE}_${stack}__${model}__shard${shard_idx}of${SHARDS_PER_MODEL}.log"
        (
            run_shard "$model" "$shard_idx" "$SHARDS_PER_MODEL" "$stack" "$gitlab_port" "$reddit_port"
        ) > "$log" 2>&1 &
        pid="$!"
        PIDS+=("$pid")
        LABELS+=("$stack/$model/$shard_idx")
        echo "launched shard stack=$stack model=$model shard=$shard_idx/$SHARDS_PER_MODEL pid=$pid log=$log"
        worker_idx=$((worker_idx + 1))
    done
done

overall_status=0
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    label="${LABELS[$i]}"
    if wait "$pid"; then
        echo "[$label] shard completed"
    else
        status=$?
        echo "[$label] shard FAILED with $status" >&2
        overall_status="$status"
    fi
done

OUT_DIR="$OUT_BASE/$PHASE" "$PYTHON" - <<'PY'
import json
import os
from collections import defaultdict
from pathlib import Path

root = Path(os.environ["OUT_DIR"])
totals = defaultdict(lambda: [0, 0, 0])
all_scored = all_success = all_errors = 0

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
    all_scored += n
    all_success += s
    all_errors += e
    arm, model, *_ = path.stem.split("__")
    totals[(arm, model)][0] += n
    totals[(arm, model)][1] += s
    totals[(arm, model)][2] += e

for (arm, model), (n, s, e) in sorted(totals.items()):
    asr = None if not n else s / n
    asr_str = "NA" if asr is None else f"{100 * asr:.2f}%"
    print(f"{arm} {model}: n={n} success={s} asr={asr_str} errors={e}")

total = None if not all_scored else all_success / all_scored
total_str = "NA" if total is None else f"{100 * total:.2f}%"
print(f"TOTAL: n={all_scored} success={all_success} asr={total_str} errors={all_errors}")
PY

exit "$overall_status"
